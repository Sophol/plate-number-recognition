"""Draws the pipeline's intermediate results onto a frame for the live preview.

This is presentation only -- nothing here feeds the committed-read path.
"""

import cv2
import numpy as np

from apps.inference_worker.interfaces import Detection

_GREEN = (74, 222, 128)
_AMBER = (56, 189, 248)
_RED = (68, 68, 239)


def _box_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.8:
        return _GREEN
    if confidence >= 0.5:
        return _AMBER
    return _RED


def draw_detection(
    image: np.ndarray,
    detection: Detection,
    label: str | None = None,
    confidence: float | None = None,
) -> None:
    """Outline one plate and caption it with the text read from it."""
    color = _box_color(confidence if confidence is not None else detection.confidence)
    cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)

    if detection.corners is not None:
        # The quad actually used for the perspective warp, which can differ
        # from the axis-aligned box the detector reports.
        pts = detection.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=True, color=_AMBER, thickness=1)

    if not label:
        return

    text = label if confidence is None else f"{label} {confidence:.2f}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(detection.y1 - th - base - 4, 0)
    cv2.rectangle(image, (detection.x1, top), (detection.x1 + tw + 8, top + th + base + 4), color, -1)
    cv2.putText(
        image,
        text,
        (detection.x1 + 4, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (15, 23, 32),
        2,
        cv2.LINE_AA,
    )


def draw_status(image: np.ndarray, lines: list[str]) -> None:
    """Top-left overlay: camera name, fps, and gate state."""
    for i, line in enumerate(lines):
        y = 22 + i * 20
        cv2.putText(image, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def placeholder(width: int, height: int, message: str) -> np.ndarray:
    """Shown while the stream is down, so the browser gets a frame instead of a stall."""
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        image,
        message,
        (max((width - tw) // 2, 10), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 170, 185),
        2,
        cv2.LINE_AA,
    )
    return image
