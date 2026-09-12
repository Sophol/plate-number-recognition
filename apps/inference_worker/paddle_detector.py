"""Locates plates using PaddleOCR's own text detector, filtered by plate format.

ContourPlateDetector looks for bright rectangles, which on real photographs
means it picks road signs, stock-photo watermarks and the plate's own "PHNOM
PENH" line ahead of the digits -- and on a phone snapshot it finds nothing at
all. PaddleOCR already detects and reads every text region in the frame, so
this asks it for all of them and keeps only the ones shaped like a plate.

The trade is cost: this runs a detection model over the whole frame rather
than a cheap contour pass, so it belongs on sampled frames (the camera worker
already thins to a few FPS), not on every frame of a 25 FPS stream. It needs
no training, which is what makes it usable today; a trained YOLO detector is
the faster long-term answer.
"""

from __future__ import annotations

import re

import numpy as np
import structlog

from apps.inference_worker.interfaces import Detection
from apps.inference_worker.validator import restore_separator

log = structlog.get_logger()

# Kept deliberately loose: OCR confusions (O for 0, I for 1) are corrected
# downstream by validator.correct, so rejecting them here would throw away
# plates the pipeline can still recover.
PLATE_SHAPE = re.compile(r"^[0-9A-Z]{1,2}[A-Z0-9]{0,2}-?[0-9A-Z]{3,4}$")
MIN_SCORE = 0.4
MAX_SIDE = 1280


class PaddlePlateDetector:
    """Finds plate-shaped text regions and reports them as Detections."""

    def __init__(self, lang: str = "en", min_score: float = MIN_SCORE, **kwargs) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with:\n"
                "  .venv/bin/pip install paddlepaddle paddleocr"
            ) from exc

        self.min_score = min_score
        self._engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **kwargs,
        )
        # Texts read during detect() are cached so the OCR stage can reuse them
        # instead of running recognition over the same crop a second time.
        self.last_reads: dict[tuple[int, int, int, int], tuple[str, float]] = {}

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        scaled, scale = _fit(image)
        try:
            results = self._engine.predict(scaled)
        except Exception:  # pragma: no cover - a backend fault must not kill a frame
            log.exception("paddle_detect_failed")
            return []

        self.last_reads = {}
        detections: list[Detection] = []

        for page in results or []:
            texts = page.get("rec_texts", []) if hasattr(page, "get") else []
            scores = page.get("rec_scores", []) if hasattr(page, "get") else []
            boxes = page.get("rec_boxes", []) if hasattr(page, "get") else []

            for text, score, box in zip(texts, scores, boxes, strict=False):
                if score < self.min_score:
                    continue
                cleaned = re.sub(r"[^A-Z0-9-]", "", str(text).upper())
                if not _plate_shaped(cleaned):
                    continue

                x1, y1, x2, y2 = (int(v / scale) for v in _as_box(box))
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=round(float(score), 3),
                        plate_type=None,
                        # Paddle returns an axis-aligned box, so there is no
                        # quad to rectify; the crop is used as-is.
                        corners=None,
                    )
                )
                self.last_reads[(x1, y1, x2, y2)] = (cleaned, float(score))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def _plate_shaped(text: str) -> bool:
    """Does this text look like a plate number rather than a sign or watermark?"""
    if not 5 <= len(text) <= 9:
        return False
    if not PLATE_SHAPE.match(text):
        return False
    # A plate always carries digits; ALAMY, PHNOMPENH and TOYOTA do not.
    return sum(c.isdigit() for c in text) >= 3


def _as_box(box) -> tuple[float, float, float, float]:
    """Accept either [x1,y1,x2,y2] or a polygon of corner points."""
    arr = np.asarray(box, dtype=float).reshape(-1)
    if arr.size == 4:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    pts = np.asarray(box, dtype=float).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale large frames; returns the image and the scale applied."""
    longest = max(image.shape[:2])
    if longest <= MAX_SIDE:
        return image, 1.0
    import cv2

    scale = MAX_SIDE / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
