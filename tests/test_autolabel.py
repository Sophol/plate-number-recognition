"""Tests for the plate auto-labelling helpers.

The video loop needs a camera and EasyOCR, so it is not unit-tested here; what
is tested is the logic that decides whether an OCR string becomes a label. That
is the part that must never store a wrong label -- a misread that slips through
the format check would poison the training set the whole approach exists to
build cheaply.
"""

import pytest

from ml.ocr.autolabel import clean, extract_plate, plate_shaped


class Box:
    def __init__(self, w, h):
        self.x1, self.y1, self.x2, self.y2 = 0, 0, w, h


def test_clean_strips_separators_and_uppercases():
    assert clean("3h-2257") == "3H2257"
    assert clean(" 2bj 4506 ") == "2BJ4506"


@pytest.mark.parametrize("raw,expected", [
    ("3H2257", "3H2257"),
    ("2BJ4506", "2BJ4506"),
    ("3F4040", "3F4040"),
    ("PP3H2257XX", "3H2257"),      # embedded in noise, still extracted
])
def test_extract_valid_plates(raw, expected):
    assert extract_plate(raw) == expected


@pytest.mark.parametrize("raw", [
    "314040",       # missing the letter
    "C314040",      # OCR hallucinated a leading letter, no valid plate token
    "ABCDEF",
    "",
    "12",
])
def test_reject_non_plates(raw):
    assert extract_plate(raw) is None


def test_plate_shaped_accepts_plate_aspect_ratio():
    assert plate_shaped(Box(200, 60))     # ~3.3:1, a plate
    assert not plate_shaped(Box(100, 100))  # square, not a plate
    assert not plate_shaped(Box(600, 60))   # 10:1, a container marking strip


def test_a_leading_zero_is_not_a_plate():
    # Cambodian plates start 1-9, so a leading 0 must not match.
    assert extract_plate("0H2257") is None
