# Cambodia ANPR

Automatic number plate recognition for Cambodian license plates. See
[anpr-plan.txt](anpr-plan.txt) for the full design, phased rollout, and rationale.

## Status

The full pipeline runs end to end: camera → detect → warp → OCR → track →
validate → vote → Postgres → outbox → gate decision. Verified against real
PostgreSQL 17. Run `python -m scripts.demo_pipeline` to see it work.

**The three ML models are not trained.** They sit behind protocols in
`apps/inference_worker/interfaces.py`, currently backed by classical-CV
fallbacks that genuinely run but are far less accurate than trained models.
Training needs the labeled dataset from Phase 0.

Implemented and tested:

- PostgreSQL schema with monthly-partitioned `plate_reads`, trigram index, audit log
- FastAPI with JWT auth and role-based access (`viewer` → `list_editor` → `gate_operator` → `admin`)
- CRUD for cameras, plate reads, and vehicles; list edits write to the audit log
- **Camera worker**: RTSP capture with exponential-backoff reconnect and
  read-timeout detection, FPS thinning, motion + ROI gating, drop-oldest queue
- **Inference pipeline**: contour plate detection, perspective correction + zone
  splitting, template OCR, IoU tracking, position-aware correction, track-scoped
  voting
- **Event worker**: transactional outbox relay, RabbitMQ topic publishing, and
  gate decisions that fail closed
- Prometheus counters across camera and inference workers

Still stubbed (`NotImplementedError`): all ML training/eval/export scripts and
dataset prep — every one depends on a labeled dataset.

### The ML fallbacks — read before trusting them

| Interface | Production plan | Current fallback | Gap |
| --- | --- | --- | --- |
| `Detector` | YOLOv8/11 segmentation | Contour + morphology | Misses angled, dirty, night plates |
| `OCR` | PaddleOCR | Hershey-font template match | Font mismatch; weak on real plates |
| `ProvinceClassifier` | CNN on Khmer top zone | Fuzzy-match English bottom zone | **Inverts the plan's design** |

That last row matters most. The plan deliberately classifies the *Khmer* zone
because Khmer OCR is unreliable — no classical-CV substitute can read Khmer, so
the fallback reads the English zone instead and fails when it is worn or cropped.
It keeps the pipeline whole; it is not the real approach.

In practice `PrefixProvinceClassifier` handles most cases anyway: Cambodian plate
numbers start with the province code, so the province comes from the plate text
without an image. The image classifier is the fallback for when that fails.

To swap in a trained model, implement the protocol and change `build_pipeline()`
in `apps/inference_worker/main.py` — no orchestration changes.

### Camera worker notes

`RTSPSource` reconnects forever by default, which is what a live camera wants.
Pass `max_reconnects` to make a finite source (a file, a test) terminate instead
of spinning at end-of-stream.

Motion gating runs only on frames that survive FPS sampling, and the ROI is
fractional (`ROI(x1=0.0, y1=0.0, x2=0.5, y2=1.0)`) so it survives a resolution
change. Tune `motion_min_changed_fraction` per camera — too low wastes GPU on
noise, too high misses distant vehicles.

### Gate safety

`handlers.decide()` fails closed. A gate opens only on an explicit whitelist
match with `is_valid=True` and confidence ≥ 0.75. Unregistered, blacklisted,
low-confidence, and format-invalid reads are all denied, and every decision
writes a `gate_events` row with its reason.

## Python versions

- API and workers that don't need ML: Python 3.13
- ML worker image: pinned to Python 3.12 — `torch`, `paddleocr`, and
  `onnxruntime-gpu` wheels lag newer releases
- `opencv` and `av` live in `requirements/gpu.txt`, not `base.txt`, so the API
  installs without needing `pkg-config` and ffmpeg headers

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements/dev.txt

cp .env.example .env          # then edit JWT_SECRET
docker compose -f docker/docker-compose.yml up -d postgres

.venv/bin/alembic upgrade head
.venv/bin/python -m scripts.create_user admin "<password>" admin
.venv/bin/uvicorn apps.api.main:app --reload
```

API docs at <http://localhost:8000/docs>, health at `/health`, Prometheus metrics
at `/metrics/` (with the trailing slash — `/metrics` 307-redirects to it).

### Local Postgres without Docker

This machine's Homebrew `postgresql@14` is broken (built against `icu4c` 73,
which is no longer installed). PostgreSQL 17 was installed alongside it on port
**5433**, leaving the 14 data directory untouched:

```bash
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
pg_ctl -D /opt/homebrew/var/postgresql@17 -o "-p 5433" -l /tmp/pg17.log start
```

`.env` points at port 5433. The compose file still uses the standard 5432.

## Demo

```bash
.venv/bin/python -m scripts.demo_pipeline
```

Feeds synthetic frames through the real pipeline into Postgres and prints each
stage: voting, persistence, outbox relay, and the gate decision.

## Tests

```bash
.venv/bin/pytest

# Include the Postgres integration tests (skipped without this):
TEST_DATABASE_URL=postgresql+asyncpg://anpr:anpr@localhost:5433/anpr .venv/bin/pytest
```

84 tests, 81% coverage. The integration tests cover partitioning, JSONB, the
trigram index, and the outbox relay — none of which SQLite can verify.

## Partitions

`plate_reads` is partitioned by month on `frame_ts`, with a `DEFAULT` partition
so inserts never fail. Provision monthly partitions ahead of time in production
rather than relying on the default:

```sql
CREATE TABLE plate_reads_2026_10 PARTITION OF plate_reads
  FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
```

Because the table is partitioned, the primary key is `(id, frame_ts)`, so
`session.get(PlateRead, id)` does not work — query by `id` with a `select()`.

## Before production

**Blocking — do not run gates on the current fallbacks.** The classical-CV
detector and OCR are for pipeline development, not recognition accuracy. Train
the real models first (Phase 0/1) and measure full-plate exact-match accuracy
against a held-out test set, as the plan specifies.

Also required:

- Replace `JWT_SECRET`; the default is a placeholder
- Collect real plate samples for government, commercial, NGO, and diplomatic
  types — their regexes in `validator.py` are unwritten and those plates
  currently fail validation
- Keep camera hosts NTP-synced; `frame_ts` drives tracking, partitioning, and
  gate event ordering
- Provision monthly partitions on a schedule instead of relying on `DEFAULT`
- Test against a real RTSP camera — capture is only verified against video files,
  so codec quirks, network stalls, and camera buffering are unexercised
- Review `MIN_GATE_CONFIDENCE` (0.75) against measured accuracy once models exist
