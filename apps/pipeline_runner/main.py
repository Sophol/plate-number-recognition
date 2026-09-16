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

The camera list is not read once. The runner re-reads the cameras table every
`camera_reload_seconds` and reconciles: a camera switched on in the admin
console starts capturing within that interval, one switched off stops, and a
changed RTSP URL reconnects. So the Activate / Deactivate buttons on the web
are the on/off switch for 24/7 detection, with no service restart. An empty
table is a waiting state, not an exit: the runner idles until a camera appears.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

import structlog
from prometheus_client import start_http_server

from apps.camera_worker.main import load_active_cameras, run_camera
from apps.camera_worker.sampler import BoundedFrameQueue
from apps.inference_worker.main import build_pipeline, consume
from config import get_settings

log = structlog.get_logger()


class CaptureSet:
    """The capture tasks that are currently running, keyed by camera id.

    Reconciling against the database each tick means the set of running tasks
    always converges on the set of active cameras, whatever the admin did in
    between: toggled, edited the URL, deleted, re-added.
    """

    def __init__(self, queue: BoundedFrameQueue, target_fps: float,
                 min_changed_fraction: float) -> None:
        self._queue = queue
        self._target_fps = target_fps
        self._min_changed_fraction = min_changed_fraction
        self._tasks: dict[str, asyncio.Task] = {}
        self._urls: dict[str, str] = {}

    @property
    def tasks(self) -> list[asyncio.Task]:
        return list(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def running(self, camera_id: str) -> bool:
        task = self._tasks.get(camera_id)
        return task is not None and not task.done()

    async def reconcile(self, cameras: list) -> None:
        wanted = {str(c.id): c for c in cameras}

        # Stop what is no longer wanted, or whose URL changed (a fresh task is
        # the only way to make RTSPSource pick up a new address).
        for camera_id in list(self._tasks):
            camera = wanted.get(camera_id)
            if camera is None:
                await self._stop(camera_id, "camera_disabled")
            elif camera.rtsp_url != self._urls[camera_id]:
                await self._stop(camera_id, "camera_url_changed")

        # A capture task that finished on its own (a finite source, or a crash
        # in the reader) is restarted next tick rather than left dead forever.
        for camera_id, task in list(self._tasks.items()):
            if task.done():
                if not task.cancelled() and task.exception() is not None:
                    log.error("capture_failed", camera_id=camera_id,
                              error=repr(task.exception()))
                del self._tasks[camera_id]
                del self._urls[camera_id]

        for camera_id, camera in wanted.items():
            if camera_id not in self._tasks:
                self._start(camera)

    def _start(self, camera) -> None:
        camera_id = str(camera.id)
        self._tasks[camera_id] = asyncio.create_task(
            run_camera(
                camera_id,
                camera.rtsp_url,
                self._queue,
                self._target_fps,
                min_changed_fraction=self._min_changed_fraction,
            ),
            name=f"camera:{camera.name}",
        )
        self._urls[camera_id] = camera.rtsp_url
        log.info("capture_started", camera_id=camera_id, name=camera.name)

    async def _stop(self, camera_id: str, reason: str) -> None:
        task = self._tasks.pop(camera_id)
        self._urls.pop(camera_id, None)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        log.info("capture_stopped", camera_id=camera_id, reason=reason)

    async def stop_all(self) -> None:
        for camera_id in list(self._tasks):
            await self._stop(camera_id, "runner_stopping")


async def run(
    metrics_port: int | None = None,
    metrics_addr: str = "127.0.0.1",
    load_cameras: Callable[[], Awaitable[list]] | None = None,
    reload_seconds: float | None = None,
) -> None:
    settings = get_settings()
    load_cameras = load_cameras or load_active_cameras
    reload_seconds = settings.camera_reload_seconds if reload_seconds is None else reload_seconds

    if metrics_port:
        # Bind the metrics endpoint to loopback by default. prometheus_client
        # defaults to 0.0.0.0, which on a gate box publishes camera ids, read
        # counts, and gate decisions to anyone who can route to it -- an
        # unauthenticated read of who drove through and when. A Prometheus
        # running elsewhere should reach this over the same tunnel or reverse
        # proxy as the rest of the console, or be given an explicit address.
        start_http_server(metrics_port, metrics_addr)
        log.info("metrics_server_started", port=metrics_port, addr=metrics_addr)

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
        model_version=settings.model_version,
        detector_backend=settings.detector_backend,
        target_fps=settings.target_fps,
        camera_reload_seconds=reload_seconds,
    )

    inference = asyncio.create_task(consume(queue, pipeline), name="inference")
    captures = CaptureSet(queue, settings.target_fps, settings.motion_min_changed_fraction)
    idle_logged = False

    try:
        while not inference.done():
            try:
                cameras = await load_cameras()
            except Exception as exc:  # a database blip must not stop capture
                log.warning("camera_reload_failed", error=str(exc))
            else:
                await captures.reconcile(cameras)
                if not cameras and not idle_logged:
                    log.warning("no_active_cameras",
                                hint="enable a camera on the Cameras page; polling continues")
                    idle_logged = True
                elif cameras:
                    idle_logged = False

            # Wake early if inference dies so capture is not left filling a
            # queue nobody drains; the whole process stops together and
            # systemd restarts it.
            await asyncio.wait([inference], timeout=reload_seconds)
    finally:
        if inference.done() and not inference.cancelled():
            if inference.exception() is not None:
                log.error("task_failed", task="inference", error=repr(inference.exception()))
            else:
                log.warning("task_finished_unexpectedly", task="inference")
        await captures.stop_all()
        if not inference.done():
            inference.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inference


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="serve Prometheus metrics on this port")
    parser.add_argument("--metrics-addr", default="127.0.0.1",
                        help="address the metrics endpoint binds to "
                             "(default: 127.0.0.1; use 0.0.0.0 to expose it)")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.metrics_port, args.metrics_addr))
    except KeyboardInterrupt:
        log.info("pipeline_runner_stopped")


if __name__ == "__main__":
    main()
