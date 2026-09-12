import asyncio
import time

import structlog
from prometheus_client import Counter, Gauge
from sqlalchemy import select

from apps.camera_worker.motion import ROI, MotionGate
from apps.camera_worker.rtsp import RTSPSource
from apps.camera_worker.sampler import BoundedFrameQueue, FPSSampler, Frame
from config import get_settings
from db.models import Camera
from db.session import SessionLocal

log = structlog.get_logger()

frames_captured = Counter("anpr_frames_captured_total", "Frames read from RTSP", ["camera_id"])
frames_emitted = Counter("anpr_frames_emitted_total", "Frames passed downstream", ["camera_id"])
frames_dropped = Counter("anpr_frames_dropped_total", "Frames dropped by backpressure", ["camera_id"])
frames_static = Counter("anpr_frames_static_total", "Frames suppressed by motion gate", ["camera_id"])
queue_depth = Gauge("anpr_frame_queue_depth", "Pending frames awaiting inference")


async def run_camera(
    camera_id: str,
    rtsp_url: str,
    queue: BoundedFrameQueue,
    target_fps: float,
    roi: ROI | None = None,
    min_changed_fraction: float = 0.002,
    max_reconnects: int | None = None,
) -> None:
    source = RTSPSource(rtsp_url, camera_id, max_reconnects=max_reconnects)
    sampler = FPSSampler(target_fps)
    gate = MotionGate(roi, min_changed_fraction)

    async for image, frame_ts in source.frames():
        frames_captured.labels(camera_id).inc()

        if not sampler.should_emit(time.monotonic()):
            continue

        # Motion runs only on sampled frames; it is cheap but not free, and the
        # gate's frame-to-frame delta stays meaningful at the sampled rate.
        if not gate.has_motion(image):
            frames_static.labels(camera_id).inc()
            continue

        before = queue.dropped
        await queue.put(Frame(camera_id=camera_id, image=image, frame_ts=frame_ts))
        if queue.dropped > before:
            frames_dropped.labels(camera_id).inc(queue.dropped - before)

        frames_emitted.labels(camera_id).inc()
        queue_depth.set(queue.qsize())


async def load_active_cameras() -> list[Camera]:
    async with SessionLocal() as session:
        result = await session.execute(select(Camera).where(Camera.is_active.is_(True)))
        return list(result.scalars().all())


async def main() -> None:
    settings = get_settings()
    cameras = await load_active_cameras()
    if not cameras:
        log.warning("no_active_cameras")
        return

    queue = BoundedFrameQueue(settings.frame_queue_maxsize)
    log.info("starting_camera_workers", count=len(cameras))

    await asyncio.gather(
        *(
            run_camera(
                str(c.id),
                c.rtsp_url,
                queue,
                settings.target_fps,
                min_changed_fraction=settings.motion_min_changed_fraction,
            )
            for c in cameras
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
