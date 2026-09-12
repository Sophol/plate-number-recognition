"""Reads a locally attached camera (built-in webcam, USB camera, capture card).

Useful for exercising the pipeline with real optics and lighting when no IP
camera is available. Addressed as "device:<index>" in a camera's rtsp_url.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import cv2
import numpy as np
import structlog

log = structlog.get_logger()

DEVICE_PREFIX = "device:"


def parse_device(url: str) -> int | None:
    """Return the device index for a "device:N" url, else None."""
    if not url.startswith(DEVICE_PREFIX):
        return None
    index = url[len(DEVICE_PREFIX) :].strip()
    return int(index) if index.isdigit() else None


class DeviceSource:
    """Yields (frame, timestamp) from an attached camera until stopped."""

    def __init__(self, index: int, fps: float = 10.0) -> None:
        self.index = index
        self.fps = fps
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture | None:
        # AVFoundation is the working backend on macOS; the default one opens
        # the device but often never delivers a frame.
        capture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        return frame if ok and frame is not None else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    async def frames(self) -> AsyncIterator[tuple[np.ndarray, datetime]]:
        self._capture = await asyncio.to_thread(self._open)
        if self._capture is None:
            log.warning("device_open_failed", index=self.index)
            return

        delay = 1.0 / self.fps
        misses = 0
        try:
            while True:
                frame = await asyncio.to_thread(self._read)
                if frame is None:
                    # A webcam can drop the odd frame while exposure settles;
                    # only give up once it stops delivering entirely.
                    misses += 1
                    if misses > 30:
                        log.warning("device_stream_ended", index=self.index)
                        return
                    await asyncio.sleep(delay)
                    continue
                misses = 0
                yield frame, datetime.now(UTC)
                await asyncio.sleep(delay)
        finally:
            self.close()
