import asyncio
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(slots=True)
class Frame:
    camera_id: str
    image: np.ndarray
    frame_ts: datetime


class BoundedFrameQueue:
    """Drops the oldest frame when full so RTSP ingest never blocks on GPU backpressure."""

    def __init__(self, maxsize: int) -> None:
        self._queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    async def put(self, frame: Frame) -> None:
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass

    async def get(self) -> Frame:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


class FPSSampler:
    """Thins a camera's native frame rate down to the configured target.

    Cameras usually deliver 25-30 FPS; the plan samples 5-10 FPS, which is enough
    for a vehicle transit while cutting downstream work several-fold.
    """

    # Float error in (now - last) makes exact-interval ticks miss by ~1e-15,
    # which silently thins output below the target rate.
    _EPSILON = 1e-9

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.interval = 1.0 / target_fps
        self._last_emitted: float | None = None

    def should_emit(self, now: float) -> bool:
        if self._last_emitted is None or now - self._last_emitted >= self.interval - self._EPSILON:
            self._last_emitted = now
            return True
        return False

    def reset(self) -> None:
        self._last_emitted = None
