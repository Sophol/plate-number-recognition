"""Tests for vehicle type and colour.

The colour classifier is the part with real logic and no model, so it is tested
on synthetic swatches where the right answer is known exactly -- including the
achromatic cases (black, white, grey) that a naive hue-only classifier gets
wrong. The type detector is exercised against a stub session, checking the two
things that break silently: the class filter (only gate vehicles survive) and
the letterbox-to-frame coordinate mapping.
"""

import numpy as np
import pytest

from apps.inference_worker.vehicle import (
    OnnxVehicleDetector,
    Vehicle,
    classify_colour,
)


def swatch(bgr: tuple[int, int, int], size: int = 100) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bgr
    return img


FULL_BOX = (0, 0, 100, 100)


@pytest.mark.parametrize("bgr,expected", [
    ((255, 255, 255), "white"),
    ((0, 0, 0), "black"),
    ((128, 128, 128), "grey"),
    ((0, 0, 220), "red"),      # BGR: pure red
    ((220, 0, 0), "blue"),     # BGR: pure blue
    ((0, 200, 0), "green"),
    ((0, 220, 220), "yellow"),
])
def test_named_colours(bgr, expected):
    name, conf = classify_colour(swatch(bgr), FULL_BOX)
    assert name == expected
    assert conf > 0.9      # a uniform swatch should be near-unanimous


def test_two_tone_region_reports_low_confidence():
    """A half-red half-blue box names one but is not confident about it.

    Two chromatic colours, not black/white: dark pixels are treated as shadow
    and excluded from the vote, so a genuine two-tone must use colours that both
    survive that filter for the confidence to be meaningfully split.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :50] = (0, 0, 220)      # red half
    img[:, 50:] = (220, 0, 0)      # blue half
    name, conf = classify_colour(img, FULL_BOX)
    assert name in {"red", "blue"}
    assert conf < 0.7


def test_dark_glass_does_not_override_body_colour():
    """A white body with a dark windscreen strip still reads white, not black.

    This is the overhead-camera failure the exclusion rule exists for: the box
    centre catches tinted glass, which must not be counted as the vehicle colour.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (255, 255, 255)       # white body
    img[35:55, :] = (30, 20, 20)   # dark windscreen strip through the centre
    name, _ = classify_colour(img, FULL_BOX)
    assert name == "white"


def test_mostly_dark_region_is_black():
    """But a genuinely dark vehicle is still black."""
    img = np.zeros((100, 100, 3), np.uint8)  # all near-black
    name, conf = classify_colour(img, FULL_BOX)
    assert name == "black"
    assert conf > 0.9


def test_tiny_box_is_unknown():
    assert classify_colour(swatch((0, 0, 220)), (0, 0, 2, 2)) == ("unknown", 0.0)


def test_colour_samples_centre_not_edges():
    """Background at the edges must not sway the colour: the centre wins."""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (0, 220, 0)          # green border
    img[25:75, 25:75] = (0, 0, 220)  # red centre (the vehicle body)
    name, _ = classify_colour(img, FULL_BOX)
    assert name == "red"


# --- type detector ---------------------------------------------------------


class StubInput:
    name = "images"
    shape = [1, 3, 640, 640]


class StubSession:
    """Emits COCO-layout raw output: (1, 4+80, anchors)."""

    def __init__(self, boxes):
        self.boxes = boxes            # (cx, cy, w, h, class_id, score)

    def get_inputs(self):
        return [StubInput()]

    def run(self, _out, _feed):
        raw = np.zeros((1, 84, max(len(self.boxes), 1)), np.float32)
        for i, (cx, cy, w, h, cls, score) in enumerate(self.boxes):
            raw[0, :4, i] = (cx, cy, w, h)
            raw[0, 4 + cls, i] = score
        return [raw]


def make_detector(boxes):
    d = OnnxVehicleDetector.__new__(OnnxVehicleDetector)
    d.session = StubSession(boxes)
    d.input_name = "images"
    d.imgsz = 640
    d.conf = 0.35
    d.iou = 0.5
    return d


def test_only_vehicle_classes_survive():
    # class 0 = person, 2 = car, 7 = truck. Person must be dropped.
    det = make_detector([
        (320, 320, 100, 80, 0, 0.9),   # person
        (200, 200, 120, 90, 2, 0.8),   # car
        (400, 400, 200, 150, 7, 0.85), # truck
    ])
    found = det.detect(np.zeros((640, 640, 3), np.uint8))
    assert {v.vehicle_type for v in found} == {"car", "truck"}


def test_box_maps_to_frame_pixels_on_a_wide_image():
    det = make_detector([(320, 320, 200, 100, 7, 0.9)])
    (v,) = det.detect(np.zeros((640, 1280, 3), np.uint8))
    assert (v.x1 + v.x2) / 2 == pytest.approx(640, abs=2)
    assert v.vehicle_type == "truck"


def test_low_confidence_dropped():
    det = make_detector([(320, 320, 100, 80, 2, 0.1)])
    assert det.detect(np.zeros((640, 640, 3), np.uint8)) == []


def test_missing_model_raises():
    with pytest.raises(RuntimeError, match="not found"):
        OnnxVehicleDetector("models/no_such_vehicle.onnx")
