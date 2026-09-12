"""Tests for the ONNX detector.

The decoding maths is tested against a stub session rather than a real model,
because the part that silently breaks is the letterbox arithmetic -- a box
returned in letterbox pixels instead of frame pixels still looks like a
plausible detection, and a real model would hide the error behind its own
uncertainty. Feeding known raw output makes the expected box exact.
"""

import numpy as np
import pytest

from apps.inference_worker.onnx_detector import OnnxPlateDetector


class StubInput:
    name = "images"
    shape = [1, 3, 640, 640]


class StubSession:
    """Returns fixed raw output in YOLOv8/11 export layout: (1, 4 + nc, anchors)."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]):
        self.boxes = boxes
        self.last_batch: np.ndarray | None = None

    def get_inputs(self):
        return [StubInput()]

    def run(self, _outputs, feed):
        self.last_batch = next(iter(feed.values()))
        anchors = max(len(self.boxes), 1)
        raw = np.zeros((1, 5, anchors), dtype=np.float32)
        for i, (cx, cy, w, h, score) in enumerate(self.boxes):
            raw[0, :, i] = (cx, cy, w, h, score)
        return [raw]


def make_detector(boxes, **kwargs) -> OnnxPlateDetector:
    detector = OnnxPlateDetector.__new__(OnnxPlateDetector)
    detector.session = StubSession(boxes)
    detector.input_name = "images"
    detector.imgsz = 640
    detector.conf = kwargs.get("conf", 0.25)
    detector.iou = kwargs.get("iou", 0.45)
    detector.max_detections = kwargs.get("max_detections", 10)
    return detector


def test_square_image_maps_box_back_to_frame_pixels():
    # 640x640 in, no padding, scale 1: the box comes back unchanged.
    detector = make_detector([(320.0, 320.0, 100.0, 50.0, 0.9)])
    frame = np.zeros((640, 640, 3), np.uint8)

    (detection,) = detector.detect(frame)

    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (270, 295, 370, 345)
    assert detection.confidence == pytest.approx(0.9)


def test_letterbox_padding_is_undone_for_a_wide_frame():
    # 1280x640 scales by 0.5 to 640x320, leaving 160 px of padding top and
    # bottom. A box at the letterbox centre must land at the frame centre.
    detector = make_detector([(320.0, 320.0, 200.0, 100.0, 0.8)])
    frame = np.zeros((640, 1280, 3), np.uint8)

    (detection,) = detector.detect(frame)

    assert (detection.x1 + detection.x2) / 2 == pytest.approx(640, abs=1)
    assert (detection.y1 + detection.y2) / 2 == pytest.approx(320, abs=1)
    # Widths scale back by 1/0.5, so a 200 px letterbox box is 400 px in frame.
    assert detection.x2 - detection.x1 == pytest.approx(400, abs=1)


def test_low_confidence_boxes_are_dropped():
    detector = make_detector([(320.0, 320.0, 100.0, 50.0, 0.1)], conf=0.25)
    assert detector.detect(np.zeros((640, 640, 3), np.uint8)) == []


def test_overlapping_boxes_are_suppressed_to_one():
    detector = make_detector([
        (320.0, 320.0, 100.0, 50.0, 0.9),
        (322.0, 321.0, 102.0, 51.0, 0.7),   # near-identical, should be merged
    ])
    detections = detector.detect(np.zeros((640, 640, 3), np.uint8))
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)


def test_results_are_sorted_by_confidence_and_capped():
    boxes = [(100.0 + i * 120, 320.0, 60.0, 30.0, 0.3 + i * 0.1) for i in range(4)]
    detector = make_detector(boxes, max_detections=2)
    detections = detector.detect(np.zeros((640, 640, 3), np.uint8))

    assert len(detections) == 2
    assert detections[0].confidence > detections[1].confidence


def test_preprocess_produces_normalised_nchw_rgb():
    detector = make_detector([])
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :, 2] = 255                      # pure red in BGR

    detector.detect(frame)
    batch = detector.session.last_batch

    assert batch.shape == (1, 3, 640, 640)
    assert batch.dtype == np.float32
    assert batch.max() <= 1.0
    # Red must land in channel 0 after the BGR->RGB swap.
    assert batch[0, 0].max() == pytest.approx(1.0)
    assert batch[0, 1].max() < 1.0


def test_grayscale_frame_is_accepted():
    detector = make_detector([(320.0, 320.0, 100.0, 50.0, 0.9)])
    assert len(detector.detect(np.zeros((640, 640), np.uint8))) == 1


def test_empty_frame_returns_nothing():
    detector = make_detector([(320.0, 320.0, 100.0, 50.0, 0.9)])
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_detection_has_no_corners():
    """A box-only model cannot supply corners; the caller must fall back to the crop."""
    detector = make_detector([(320.0, 320.0, 100.0, 50.0, 0.9)])
    (detection,) = detector.detect(np.zeros((640, 640, 3), np.uint8))
    assert detection.corners is None


def test_missing_model_file_raises_runtime_error():
    with pytest.raises(RuntimeError, match="not found"):
        OnnxPlateDetector("models/does_not_exist.onnx")
