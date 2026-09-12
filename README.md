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

Dataset prep and the detection train/evaluate/export scripts are implemented and
run on Apple-silicon MPS — see [Training](#training). They still need the labeled
dataset from Phase 0 to produce anything useful. Province and OCR training remain
stubbed (`NotImplementedError`).

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
- Training on an Apple-silicon Mac uses `requirements/mac-ml.txt` on Python
  3.13 — see [Training](#training)

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

## Training

The detection model trains on an Apple-silicon Mac via MPS. The server never
needs torch: training produces an ONNX file, and the inference worker loads it
with `onnxruntime` on CPU. Verified on an M1 Pro (16 GB) — YOLO11n at 640 px
exports to ONNX and runs at ~13 FPS on that machine's CPU, so the same file on a
4-vCPU server lands in the 3–6 FPS range, enough for a barrier gate but not for
high-speed multi-camera capture.

```bash
python3.13 -m venv .venv-ml
.venv-ml/bin/pip install -r requirements/mac-ml.txt
.venv-ml/bin/python -c "import torch; print(torch.backends.mps.is_available())"  # True
```

Do not install `requirements/gpu.txt` on a Mac — `onnxruntime-gpu` and the
pinned CUDA `torch` have no macOS wheels. The Mac's torch version does not need
to match the server's, because only the ONNX crosses between them.

### The pipeline

```bash
# 1. Frames out of recorded RTSP video, deduplicated and deblurred
.venv-ml/bin/python -m ml.datasets.prepare --videos recordings/ --out dataset/v1

# 2. Label them in Label Studio, then export back to dataset/v1/labels/
#    (see Labelling below)

# 3. Group-aware split — frames from one video never straddle train and test
.venv-ml/bin/python -m ml.datasets.split --dataset dataset/v1

# 4. Train (device is auto-detected: MPS, else CUDA, else CPU)
.venv-ml/bin/python -m ml.detection.train --data dataset/v1/yolo/data.yaml

# 5. Held-out metrics, then ONNX with a parity check against the torch model
.venv-ml/bin/python -m ml.detection.evaluate --weights runs/detect/plate/weights/best.pt \
    --data dataset/v1/yolo/data.yaml
.venv-ml/bin/python -m ml.detection.export --weights runs/detect/plate/weights/best.pt
```

Step 5 writes `models/plate_detector.onnx`, which `config.detector_model_path`
already points at. `scp` it to the server and implement the `Detector` protocol
against it; no orchestration changes.

`split.py` refuses to run with fewer than three groups, because a dataset that
cannot hold out a separate val and test set cannot produce an honest number.
`export.py` refuses to write a model whose ONNX output drifts more than 1e-3
(relative) from torch — a mismatched opset produces a model that loads, runs,
and quietly returns worse boxes.

Expect 1–3 hours for 100 epochs of YOLO11n on a few thousand images on an M1
Pro. Keep `--batch` at 16 or below: MPS shares the 16 GB with the system.

### Labelling

Label Studio runs locally in its own venv — it pulls Django and would collide
with the ML venv.

```bash
python3.13 -m venv .venv-label
.venv-label/bin/pip install label-studio

.venv-ml/bin/python -m ml.labeling.import_tasks --dataset dataset/v1

LOCAL_FILES_SERVING_ENABLED=true \
LOCAL_FILES_DOCUMENT_ROOT=$PWD/dataset \
  .venv-label/bin/label-studio start --port 8081
```

Then at <http://localhost:8081>: sign up, create a project, paste
`ml/labeling/plate_config.xml` into Labeling Interface → Code, and import
`dataset/v1/label_studio_tasks.json`.

Frames are referenced by path, not uploaded — `LOCAL_FILES_DOCUMENT_ROOT` must
contain them or every task renders a broken image. Serving is authenticated, so
a signed-out request for a frame returns 401; that is the server working, not a
misconfiguration.

One pass per frame produces the labels for all three models: the box trains the
detector, the transcription trains OCR, the province choice trains the
classifier. Every attribute is per-region, because a frame with two plates would
otherwise attach the first plate's text to both.

Export from Label Studio in **JSON** format (not JSON-MIN — it drops the
per-region attributes), then:

```bash
.venv-ml/bin/python -m ml.labeling.export_labels --export export.json --dataset dataset/v1
```

That writes `dataset/v1/labels/*.txt` for the detector and
`dataset/v1/plates.jsonl` (text, province, plate type per box) for OCR and
province training. Label Studio's own YOLO export is not used: it keeps the
boxes and discards the attributes, which is the whole point of labelling once.

A frame whose annotation was cancelled becomes an empty label file — a real
"no plates here" sample. A task nobody opened is skipped entirely, so an
unlabelled backlog never teaches the detector that plates are absent.

Per the plan, route a sample of each annotator's work through a second reviewer.
Province and plate-type are the labels a single annotator gets systematically
wrong, and that error caps end-to-end accuracy no matter how long the detector
trains.

### Still stubbed

`ml/province/` and OCR training remain `NotImplementedError`. They need label
types the detection dataset does not carry — per-crop province codes and
per-character transcriptions — so they are blocked on Phase 0 labelling rather
than on tooling. PaddleOCR has no MPS backend and fine-tunes on CPU only, which
is the one step worth renting a GPU box for.

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
