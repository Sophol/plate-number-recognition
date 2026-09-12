"""Proposes container-number regions in a frame, without a trained detector.

There is no container-number detector yet -- training one needs labelled boxes
this gate has not accumulated. So the region is found the classical way: within
the upper part of a detected vehicle, painted text is a run of high-gradient
strokes that a morphological close merges into one wide band (the top-rail
number) or one tall band (the side number). Every band that is roughly the
right shape is proposed.

That produces false candidates freely, and that is fine: each candidate is
OCR'd, and only a read whose ISO 6346 check digit is correct survives. The
checksum is what makes brute-forcing candidates safe. When labelled boxes exist
(the auto-labeller records them), a YOLO locator replaces this behind the same
interface, exactly as the plate detector replaced its contour fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int
    vertical: bool

    def crop(self, frame: np.ndarray, pad: float = 0.12) -> np.ndarray:
        w, h = self.x2 - self.x1, self.y2 - self.y1
        px, py = int(w * pad), int(h * pad)
        H, W = frame.shape[:2]
        return frame[max(0, self.y1 - py):min(H, self.y2 + py),
                     max(0, self.x1 - px):min(W, self.x2 + px)]


class ContainerLocator:
    def __init__(self, max_candidates: int = 8, upper_fraction: float = 0.7) -> None:
        self.max_candidates = max_candidates
        # The container sits on the chassis: its markings are in the upper part
        # of the vehicle box; the lower part is cab, wheels and road.
        self.upper_fraction = upper_fraction

    def propose(self, frame: np.ndarray, box: tuple[int, int, int, int] | None = None) -> list[Region]:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box if box else (0, 0, W, H)
        y2 = y1 + int((y2 - y1) * self.upper_fraction)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        rh, rw = roi.shape[:2]
        regions: list[tuple[int, Region]] = []
        for vertical, kernel, keep in (
            (False, (max(9, rw // 60), 3), self._is_horizontal_band),
            (True, (3, max(9, rh // 40)), self._is_vertical_band),
        ):
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if keep(w, h, rw, rh):
                    regions.append((w * h, Region(x1 + x, y1 + y, x1 + x + w, y1 + y + h, vertical)))

        regions.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in regions[: self.max_candidates]]

    @staticmethod
    def _is_horizontal_band(w, h, rw, rh) -> bool:
        if h == 0:
            return False
        aspect = w / h
        return 3.5 <= aspect <= 16 and w >= rw * 0.12 and rh * 0.012 <= h <= rh * 0.14

    @staticmethod
    def _is_vertical_band(w, h, rw, rh) -> bool:
        if w == 0:
            return False
        aspect = h / w
        return 3.5 <= aspect <= 16 and h >= rh * 0.15 and rw * 0.012 <= w <= rw * 0.12
