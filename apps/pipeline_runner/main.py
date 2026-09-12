"""Run capture and inference together in one process.

    python -m apps.pipeline_runner.main

The camera worker and the inference worker cannot be separate services as
written: each builds its own `BoundedFrameQueue`, and that wraps an
`asyncio.Queue`, which exists only inside one process. Started apart, the camera
worker fills a queue nobody drains and the inference worker waits on a queue
nobody fills -- with no error and no log line, just silence. This runner is the
missing piece: one queue, both halves, one event loop.

That is also the right shape for a gate. One or two cameras on a CPU box do not
need the two halves scaled independently, and keeping frames in memory avoids
serialising every frame through a broker. If inference later moves to its own
machine or a GPU, replace `BoundedFrameQueue` with a Redis-backed queue and split
them again -- nothing else here has to change.
"""

import asyncio
import contextlib

import structlog
from prometheus_client import start_http_server

from apps.camera_worker.main import load_active_cameras, run_camera
from apps.camera_worker.sampler import BoundedFrameQueue
from apps.inference_worker.main import build_pipeline, consume
from config import get_settings

log = structlog.get_logger()


async def run(metrics_port: int | None = None) -> None:
    settings = get_settings()

    cameras = await load_active_cameras()
    if not cameras:
        # Exiting non-zero would make systemd restart-loop against an empty
        # database, which is a configuration state rather than a failure.
        log.warning("no_active_cameras", hint="register a camera via POST /cameras")
        return

    if metrics_port:
        start_http_server(metrics_port)
        log.info("metrics_server_started", port=metrics_port)

    queue = BoundedFrameQueue(settings.frame_queue_maxsize)
    pipeline = build_pipeline(
        settings.model_version,
        settings.votes_required,
        ocr_backend=settings.ocr_backend,
        ocr_use_gpu=settings.ocr_use_gpu,
        detector_backend=settings.detector_backend,
    )

    log.info(
        "pipeline_runner_started",
        cameras=len(cameras),
        model_version=settings.model_version,
        detector_backend=settings.detector_backend,
        target_fps=settings.target_fps,
    )

    inference = asyncio.create_task(consume(queue, pipeline), name="inference")
    capture = [
        asyncio.create_task(
            run_camera(
                str(camera.id),
                camera.rtsp_url,
                queue,
                settings.target_fps,
                min_changed_fraction=settings.motion_min_changed_fraction,
            ),
            name=f"camera:{camera.name}",
        )
        for camera in cameras
    ]

    # If inference dies the capture tasks would keep filling a queue nobody
    # drains, so the whole process stops together and systemd restarts it.
    done, pending = await asyncio.wait(
        [inference, *capture], return_when=asyncio.FIRST_COMPLETED
    )

    for task in done:
        if task.cancelled():
            continue
        if task.exception() is not None:
            log.error("task_failed", task=task.get_name(), error=repr(task.exception()))
        else:
            log.warning("task_finished_unexpectedly", task=task.get_name())

    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            await task


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="serve Prometheus metrics on this port")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.metrics_port))
    except KeyboardInterrupt:
        log.info("pipeline_runner_stopped")


if __name__ == "__main__":
    main()
