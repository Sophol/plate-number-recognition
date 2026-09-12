import cv2
import numpy as np

from apps.inference_worker.interfaces import Detection

# Cambodian plates are roughly 2:1 to 5:1 wide; anything outside is not a plate.
MIN_ASPECT = 1.8
MAX_ASPECT = 5.5
MIN_AREA_FRACTION = 0.0008
MAX_AREA_FRACTION = 0.25


class ContourPlateDetector:
    """Classical-CV plate locator used until a trained YOLO model exists.

    Finds bright, high-contrast rectangles via morphological gradient and
    contour filtering. Accuracy is far below a trained detector -- it exists so
    the pipeline runs end to end on real images, not to be shipped as-is.
    """

    def __init__(self, max_detections: int = 5) -> None:
        self.max_detections = max_detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        h, w = gray.shape[:2]
        frame_area = float(h * w)

        # Emphasise character strokes, then close them into a plate-sized blob.
        gradient = cv2.morphologyEx(
            gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        )

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[Detection] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0:
                continue
            aspect = cw / ch
            area_fraction = (cw * ch) / frame_area

            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            if not (MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION):
                continue

            # Fill ratio separates plate-like blobs from ragged texture.
            fill = cv2.contourArea(contour) / float(cw * ch)
            if fill < 0.35:
                continue

            candidates.append(
                Detection(
                    x1=x,
                    y1=y,
                    x2=x + cw,
                    y2=y + ch,
                    confidence=round(min(fill, 0.99), 3),
                    plate_type=None,
                    corners=self._corners(contour),
                )
            )

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[: self.max_detections]

    @staticmethod
    def _corners(contour: np.ndarray) -> np.ndarray | None:
        """Minimum-area rectangle corners, for perspective correction."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32) if box is not None else None
