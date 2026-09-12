"""Reads a container number from a crop with the trained CRNN, via onnxruntime.

The recogniser is only half of a trustworthy read; the other half is here. A
raw CTC string is normalised and parsed as ISO 6346, its check digit verified,
and -- when the terminal's PAS list is loaded -- a read that fails the checksum
is snapped to the one known container within a single edit, if there is exactly
one. A read that passes the checksum is also marked known or unknown, which the
gate can weigh: known-and-valid is the strongest evidence this pipeline
produces.

Preprocessing mirrors ml/container/train_ocr.py exactly (grayscale, tall crops
rotated to horizontal, 32x160, 0-1). A vertical side number that reads
bottom-to-top decodes to nothing valid when rotated the usual way, so the other
rotation is tried before giving up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

from apps.inference_worker.container import (
    ContainerNumber, KnownContainers, normalise, parse, snap_to_known,
)

log = structlog.get_logger()

IMG_HEIGHT, IMG_WIDTH = 32, 160
DEFAULT_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass(slots=True)
class ContainerReadResult:
    text: str                       # raw decoded string, pre-snap
    confidence: float               # mean max-prob over emitted characters
    number: ContainerNumber | None  # parsed, checksum-valid (possibly snapped)
    is_known: bool | None           # None when no PAS list is loaded
    was_snapped: bool

    @property
    def ok(self) -> bool:
        return self.number is not None and self.number.checksum_ok


def _decode(indices: np.ndarray, charset: str) -> str:
    out, prev = [], 0
    for idx in indices.tolist():
        if idx != prev and idx != 0:
            out.append(charset[idx - 1] if 0 < idx <= len(charset) else "")
        prev = idx
    return "".join(out)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class OnnxContainerOCR:
    def __init__(
        self,
        model_path: str | Path,
        charset_path: str | Path | None = None,
        known: KnownContainers | None = None,
        providers: list[str] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"container OCR model not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed") from exc

        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        charset_file = Path(charset_path) if charset_path else path.with_name("container_ocr_charset.json")
        self.charset = (json.loads(charset_file.read_text())["charset"]
                        if charset_file.exists() else DEFAULT_CHARSET)
        self.known = known
        log.info("container_ocr_loaded", model=str(path),
                 known=len(known) if known is not None else None)

    # --- preprocessing ------------------------------------------------------

    @staticmethod
    def _prep(crop: np.ndarray, rotation: int | None) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if rotation is not None:
            gray = cv2.rotate(gray, rotation)
        gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        return (gray.astype(np.float32) / 255.0)[None, None]

    def _run(self, batch: np.ndarray) -> tuple[str, float]:
        logits = self.session.run(None, {self.input_name: batch})[0][0]   # (T, C)
        probs = _softmax(logits)
        idx = probs.argmax(axis=1)
        text = _decode(idx, self.charset)
        emitted = [probs[t, idx[t]] for t in range(len(idx)) if idx[t] != 0]
        conf = float(np.mean(emitted)) if emitted else 0.0
        return text, conf

    # --- public -------------------------------------------------------------

    def read(self, crop: np.ndarray) -> ContainerReadResult:
        if crop.size == 0:
            return ContainerReadResult("", 0.0, None, None, False)
        h, w = crop.shape[:2]
        tall = h > w * 1.3
        # Horizontal crops are read as-is; tall ones are rotated the usual way
        # first and the other way only if that yields nothing valid.
        rotations = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE] if tall else [None]

        best: ContainerReadResult | None = None
        for rot in rotations:
            text, conf = self._run(self._prep(crop, rot))
            result = self._resolve(text, conf)
            if result.ok:
                return result
            if best is None or conf > best.confidence:
                best = result
        return best if best is not None else ContainerReadResult("", 0.0, None, None, False)

    def _resolve(self, text: str, conf: float) -> ContainerReadResult:
        raw = normalise(text)
        number = parse(raw)
        snapped = False
        if (number is None or not number.checksum_ok) and self.known is not None:
            fixed = snap_to_known(raw, self.known)
            if fixed is not None:
                number, snapped = parse(fixed), True
        if number is None or not number.checksum_ok:
            return ContainerReadResult(raw, conf, None, None, False)
        is_known = (number.canonical in self.known) if self.known is not None else None
        return ContainerReadResult(raw, conf, number, is_known, snapped)
