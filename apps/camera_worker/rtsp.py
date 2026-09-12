import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import cv2
import numpy as np
import structlog

log = structlog.get_logger()

INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
# A live RTSP stream that returns no frame for this long is treated as dead even
# if the socket is still open, which is the common silent-failure mode.
READ_TIMEOUT_SECONDS = 10.0


class RTSPSource:
    """Reads frames from an RTSP URL, reconnecting with exponential backoff.

    OpenCV's VideoCapture is blocking, so reads run in a worker thread to keep
    the event loop free for other cameras.
    """

    def __init__(self, url: str, camera_id: str, max_reconnects: int | None = None) -> None:
        self.url = url
        self.camera_id = camera_id
        # None means reconnect forever, which is what a live camera wants. A finite
        # value lets a finite source (a file, a test) terminate instead of spinning.
        self.max_reconnects = max_reconnects
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            return None
        # Keep the internal buffer short so we read live frames, not stale ones.
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
        """Yield (frame, capture_ts) forever, reconnecting on failure."""
        backoff = INITIAL_BACKOFF_SECONDS
        reconnects = 0
        try:
            while True:
                if self._capture is None:
                    if self.max_reconnects is not None and reconnects > self.max_reconnects:
                        log.info("rtsp_reconnect_limit_reached", camera_id=self.camera_id)
                        return
                    reconnects += 1

                    self._capture = await asyncio.to_thread(self._open)
                    if self._capture is None:
                        log.warning(
                            "rtsp_connect_failed", camera_id=self.camera_id, retry_in=backoff
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                        continue
                    log.info("rtsp_connected", camera_id=self.camera_id)
                    backoff = INITIAL_BACKOFF_SECONDS

                try:
                    frame = await asyncio.wait_for(
                        asyncio.to_thread(self._read), timeout=READ_TIMEOUT_SECONDS
                    )
                except TimeoutError:
                    log.warning("rtsp_read_timeout", camera_id=self.camera_id)
                    frame = None

                if frame is None:
                    log.warning("rtsp_stream_lost", camera_id=self.camera_id)
                    self.close()
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue

                # Wall-clock capture time. Depends on NTP sync across camera hosts.
                yield frame, datetime.now(UTC)
        finally:
            self.close()
