from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class Detection:
    """One detected plate in a frame, in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    plate_type: str | None = None
    # Four corners for perspective correction; absent when the backend is
    # box-only, in which case the caller falls back to the plain crop.
    corners: np.ndarray | None = field(default=None)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float


@dataclass(slots=True)
class ProvinceResult:
    code: str | None
    confidence: float


@runtime_checkable
class Detector(Protocol):
    """Locates plates in a frame. Backed by YOLO in production."""

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class OCR(Protocol):
    """Reads the Latin/digit middle zone. Backed by PaddleOCR in production."""

    def read(self, plate_image: np.ndarray) -> OCRResult: ...


@runtime_checkable
class ProvinceClassifier(Protocol):
    """Classifies the Khmer top zone into a province code."""

    def classify(self, top_zone: np.ndarray) -> ProvinceResult: ...
