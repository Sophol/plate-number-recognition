import cv2
import numpy as np

from apps.inference_worker.interfaces import OCRResult

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLYPH_SIZE = (32, 48)
MIN_GLYPH_HEIGHT_FRACTION = 0.35
MIN_GLYPH_ASPECT = 0.12
MAX_GLYPH_ASPECT = 1.2


def _render_templates() -> dict[str, np.ndarray]:
    """Render reference glyphs with OpenCV's built-in Hershey font."""
    templates: dict[str, np.ndarray] = {}
    for glyph in GLYPHS:
        canvas = np.zeros((64, 48), dtype=np.uint8)
        cv2.putText(canvas, glyph, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 3)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            canvas = canvas[y : y + h, x : x + w]
        templates[glyph] = cv2.resize(canvas, GLYPH_SIZE, interpolation=cv2.INTER_AREA)
    return templates


class TemplateOCR:
    """Segment-then-match OCR used until PaddleOCR is wired in.

    Binarises the middle zone, splits it into character blobs, and matches each
    against rendered templates. Font mismatch makes this materially less
    accurate than PaddleOCR -- it exists to exercise the pipeline, and the
    confidence it reports is a correlation score, not a calibrated probability.
    """

    def __init__(self) -> None:
        self._templates = _render_templates()

    def read(self, plate_image: np.ndarray) -> OCRResult:
        glyphs = self._segment(plate_image)
        if not glyphs:
            return OCRResult(text="", confidence=0.0)

        characters: list[str] = []
        scores: list[float] = []
        for glyph in glyphs:
            character, score = self._match(glyph)
            characters.append(character)
            scores.append(score)

        return OCRResult(
            text="".join(characters),
            confidence=round(float(np.mean(scores)), 3),
        )

    def _segment(self, image: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # OTSU both ways; plates appear dark-on-light and light-on-dark.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if np.count_nonzero(binary) > binary.size * 0.5:
            binary = cv2.bitwise_not(binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height = binary.shape[0]

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < height * MIN_GLYPH_HEIGHT_FRACTION:
                continue
            if not (MIN_GLYPH_ASPECT <= w / max(h, 1) <= MAX_GLYPH_ASPECT):
                continue
            boxes.append((x, y, w, h))

        # Left-to-right is the reading order.
        boxes.sort(key=lambda b: b[0])
        return [
            cv2.resize(binary[y : y + h, x : x + w], GLYPH_SIZE, interpolation=cv2.INTER_AREA)
            for x, y, w, h in boxes
        ]

    def _match(self, glyph: np.ndarray) -> tuple[str, float]:
        best_character = "?"
        best_score = -1.0
        for character, template in self._templates.items():
            score = float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_score = score
                best_character = character
        return best_character, max(best_score, 0.0)
