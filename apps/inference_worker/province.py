import difflib

import numpy as np

from apps.inference_worker.interfaces import OCR, ProvinceResult

# Plate prefix code -> English province name, matching the seeded provinces table.
PROVINCE_NAMES: dict[str, str] = {
    "1": "BANTEAY MEANCHEY",
    "2": "BATTAMBANG",
    "3": "KAMPONG CHAM",
    "4": "KAMPONG CHHNANG",
    "5": "KAMPONG SPEU",
    "6": "KAMPONG THOM",
    "7": "KAMPOT",
    "8": "KANDAL",
    "9": "KOH KONG",
    "10": "KRATIE",
    "11": "MONDULKIRI",
    "12": "PHNOM PENH",
    "13": "PREAH VIHEAR",
    "14": "PREY VENG",
    "15": "PURSAT",
    "16": "RATANAKIRI",
    "17": "SIEM REAP",
    "18": "PREAH SIHANOUK",
    "19": "STUNG TRENG",
    "20": "SVAY RIENG",
    "21": "TAKEO",
    "22": "ODDAR MEANCHEY",
    "23": "KEP",
    "24": "PAILIN",
    "25": "TBOUNG KHMUM",
}

MIN_NAME_SIMILARITY = 0.55


class EnglishZoneProvinceClassifier:
    """Infers province from the bottom (English) zone until a trained model exists.

    The plan's design classifies the *Khmer* top zone with a CNN, because Khmer
    OCR is unreliable. No classical-CV substitute can read Khmer script, so this
    fallback reads the Latin bottom zone instead and fuzzy-matches it against
    known province names. That inverts the plan's intent and will fail on plates
    whose bottom zone is absent, worn, or cropped -- it is a placeholder to keep
    the pipeline whole, not the real approach.
    """

    def __init__(self, ocr: OCR) -> None:
        self.ocr = ocr

    def classify(self, top_zone: np.ndarray) -> ProvinceResult:
        """Accepts the bottom zone despite the parameter name from the protocol."""
        reading = self.ocr.read(top_zone)
        if not reading.text:
            return ProvinceResult(code=None, confidence=0.0)

        best_code: str | None = None
        best_score = 0.0
        for code, name in PROVINCE_NAMES.items():
            score = difflib.SequenceMatcher(None, reading.text.upper(), name).ratio()
            if score > best_score:
                best_score = score
                best_code = code

        if best_score < MIN_NAME_SIMILARITY:
            return ProvinceResult(code=None, confidence=round(best_score, 3))
        return ProvinceResult(code=best_code, confidence=round(best_score, 3))


class PrefixProvinceClassifier:
    """Derives province from the plate's leading digits.

    Cambodian plate numbers start with the province code, so once OCR has read
    the middle zone this needs no image at all. More reliable than reading the
    province text, and a useful cross-check against the image-based classifier.
    """

    def from_plate_text(self, plate_text: str) -> ProvinceResult:
        prefix = ""
        for character in plate_text:
            if character.isdigit():
                prefix += character
            else:
                break

        # Two-digit codes (10-25) take precedence over their first digit.
        for candidate in (prefix[:2], prefix[:1]):
            if candidate in PROVINCE_NAMES:
                return ProvinceResult(code=candidate, confidence=0.9)
        return ProvinceResult(code=None, confidence=0.0)

    def classify(self, top_zone: np.ndarray) -> ProvinceResult:
        raise NotImplementedError("use from_plate_text(); this backend needs no image")
