"""PaddleOCR backend for the plate's Latin/digit zone.

TemplateOCR matches against rendered Hershey glyphs, so it only reads text in
that one font -- real embossed plate typefaces come back empty. PaddleOCR is
font-agnostic and tolerates the blur and skew that roadside capture produces.

Paddle is a heavy optional dependency (it is not in requirements/base.txt), so
it is imported lazily: constructing this class is what pulls it in, and the
pipeline keeps working without it as long as nothing asks for this backend.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import structlog

from apps.inference_worker.interfaces import OCRResult

log = structlog.get_logger()

# Plates carry only A-Z and 0-9; anything else is a misread of a bolt, stamp,
# or the surrounding frame, so it is stripped before the result is returned.
_ALLOWED = re.compile(r"[^A-Z0-9-]")

# PP-OCRv6 resizes its input to a fixed shape, so characters that fill a very
# large image end up outside the scale range it was trained on: a 4112px-wide
# plate read as nothing until it was downscaled. Cap the long side well below
# Paddle's own 4000px limit.
MAX_SIDE = 1280


class PaddlePlateOCR:
    """Reads plate text with PaddleOCR, exposing the same OCR protocol."""

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs: Any) -> None:
        # use_gpu is accepted for call-site compatibility; 3.x selects the
        # device automatically and takes an explicit `device=` instead.
        _ = use_gpu
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with:\n"
                "  .venv/bin/pip install paddlepaddle paddleocr"
            ) from exc

        # PaddleOCR 3.x renamed most constructor arguments; the 2.x names
        # (show_log, use_angle_cls, use_gpu) now raise ValueError. Document
        # orientation and unwarping are for scanned pages, not plate crops,
        # so they are turned off to save two model loads per frame.
        self._engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **kwargs,
        )

    def read(self, plate_image: np.ndarray) -> OCRResult:
        if plate_image is None or plate_image.size == 0:
            return OCRResult(text="", confidence=0.0)

        plate_image = _fit(plate_image)

        try:
            # .predict() is the 3.x entry point; .ocr() is deprecated and its
            # det/rec/cls keywords were removed.
            raw = self._engine.predict(plate_image)
        except Exception:  # pragma: no cover - backend failure must not kill a frame
            log.exception("paddle_ocr_failed")
            return OCRResult(text="", confidence=0.0)

        texts, scores = _flatten(raw)
        if not texts:
            return OCRResult(text="", confidence=0.0)

        text = _ALLOWED.sub("", "".join(texts).upper())
        if not text:
            return OCRResult(text="", confidence=0.0)

        return OCRResult(text=text, confidence=round(float(np.mean(scores)), 3))


def _fit(image: np.ndarray) -> np.ndarray:
    """Downscale so the long side is at most MAX_SIDE, preserving aspect."""
    longest = max(image.shape[:2])
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    import cv2

    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _flatten(raw: Any) -> tuple[list[str], list[float]]:
    """Pull (text, score) pairs out of PaddleOCR's nested result shape.

    The shape differs between Paddle versions -- 2.x returns
    [[ [box, (text, score)], ... ]] while newer builds return a dict -- so both
    are handled rather than pinning one version.
    """
    texts: list[str] = []
    scores: list[float] = []

    if not raw:
        return texts, scores

    # Newer dict-style result.
    if isinstance(raw, dict):
        for text, score in zip(
            raw.get("rec_texts", []), raw.get("rec_scores", []), strict=False
        ):
            texts.append(str(text))
            scores.append(float(score))
        return texts, scores

    for page in raw:
        if page is None:
            continue
        if isinstance(page, dict):
            sub_texts, sub_scores = _flatten(page)
            texts.extend(sub_texts)
            scores.extend(sub_scores)
            continue
        for line in page:
            # [box, (text, score)]
            if not line or len(line) < 2:
                continue
            payload = line[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                texts.append(str(payload[0]))
                scores.append(float(payload[1]))

    return texts, scores
