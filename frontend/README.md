# ANPR Test Console (Angular)

A small UI for exercising the FastAPI backend by hand.

## Run

```bash
# backend must be up first
cd .. && .venv/bin/uvicorn apps.api.main:app --reload

# then, in frontend/
npm install
npm start          # http://localhost:4200
```

**The port matters.** `apps/api/main.py` sets its CORS allowlist to
`http://localhost:4200` only, so serving on any other port makes every
request fail preflight.

## Log in

Use a row from the `users` table. The seeded account is `admin`.
Note the endpoint is `POST /auth/token` (OAuth2 password flow,
form-encoded) — `guideline.txt` calls it `/auth/login`, which is stale.

## Pages

| Route | What you can do | Role needed |
|---|---|---|
| `/live` | Watch an annotated MJPEG preview per camera and see how each plate was read, stage by stage | `viewer` |
| `/plate-reads` | Filter by camera, plate, time range, active-model; verify a read; insert a synthetic read | read `viewer`, write `list_editor` |
| `/cameras` | List, create, activate/deactivate, edit RTSP URL | read `viewer`, write `list_editor` |
| `/vehicles` | List/search, create, change list type, delete | read `viewer`, write `list_editor` |

Write controls are hidden when your role is below `list_editor`.

Cameras whose RTSP host cannot resolve (e.g. the seeded `rtsp://x` and
`rtsp://demo`) are flagged **unreachable** — those are what make the camera
worker log `rtsp_connect_failed` and retry forever.

## Config

The API URL lives in `src/app/core/api-base.ts`.


## Live view

`/live` streams an annotated MJPEG preview from `GET /live/{camera_id}/stream`
and polls `GET /live/{camera_id}/events` once a second for the pipeline trace.

The API opens its **own** RTSP read for the preview — the camera and inference
workers keep frames in an in-process `asyncio.Queue` the API cannot reach. The
preview never writes `plate_reads`; the inference worker stays the only writer,
so what you see is a faithful re-run of the same stages, not the committed path.

A preview starts when the first viewer connects and stops about ten seconds
after the last one leaves, so a closed tab does not hold an RTSP connection open.

An `<img>` tag cannot send an `Authorization` header, so the stream endpoint
accepts the JWT as a `?token=` query parameter and validates it identically.
