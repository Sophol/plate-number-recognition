from dataclasses import dataclass

import cv2
import numpy as np

# Pixel-difference threshold before a pixel counts as "changed".
PIXEL_DELTA_THRESHOLD = 25
BLUR_KERNEL = (21, 21)


@dataclass(frozen=True, slots=True)
class ROI:
    """Fractional lane/region box, so it survives resolution changes."""

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0

    def crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return frame[
            int(self.y1 * h) : int(self.y2 * h),
            int(self.x1 * w) : int(self.x2 * w),
        ]


class MotionGate:
    """Suppresses static frames so the GPU only sees frames that changed.

    Compares each frame against the previous one within the ROI and reports the
    fraction of changed pixels.
    """

    def __init__(self, roi: ROI | None = None, min_changed_fraction: float = 0.002) -> None:
        self.roi = roi or ROI()
        self.min_changed_fraction = min_changed_fraction
        self._previous: np.ndarray | None = None

    @staticmethod
    def _prepare(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    def changed_fraction(self, frame: np.ndarray) -> float:
        current = self._prepare(self.roi.crop(frame))

        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current
            return 0.0

        delta = cv2.absdiff(self._previous, current)
        _, mask = cv2.threshold(delta, PIXEL_DELTA_THRESHOLD, 255, cv2.THRESH_BINARY)
        self._previous = current
        return float(np.count_nonzero(mask)) / mask.size

    def has_motion(self, frame: np.ndarray) -> bool:
        return self.changed_fraction(frame) >= self.min_changed_fraction

    def reset(self) -> None:
        self._previous = None
