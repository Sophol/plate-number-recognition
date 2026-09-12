"""Plays a local video file on a loop, for previewing without a live camera.

RTSPSource treats end-of-file as a lost stream and reconnects with growing
backoff, so a short clip freezes for up to 30 seconds between passes. A file
has a known end, so seek back to the start instead.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import cv2
import numpy as np


class LoopingFileSource:
    """Yields (frame, timestamp) forever, restarting at end of file."""

    def __init__(self, path: str, fps: float | None = None) -> None:
        self.path = path
        self._fps = fps
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if ok and frame is not None:
            return frame
        # End of file: rewind rather than reconnect.
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._capture.read()
        return frame if ok and frame is not None else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    async def frames(self) -> AsyncIterator[tuple[np.ndarray, datetime]]:
        self._capture = await asyncio.to_thread(self._open)
        if self._capture is None:
            return

        native = self._capture.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / (self._fps or native)

        try:
            while True:
                frame = await asyncio.to_thread(self._read)
                if frame is None:
                    return
                yield frame, datetime.now(UTC)
                # Play at the clip's own rate; without this the loop runs as
                # fast as the CPU allows and the preview looks fast-forwarded.
                await asyncio.sleep(delay)
        finally:
            self.close()
