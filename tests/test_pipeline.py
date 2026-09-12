from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest

from apps.camera_worker.sampler import Frame
from apps.inference_worker.detector import ContourPlateDetector
from apps.inference_worker.interfaces import Detection, OCRResult, ProvinceResult
from apps.inference_worker.ocr import TemplateOCR
from apps.inference_worker.perspective import order_corners, split_zones, warp_plate
from apps.inference_worker.pipeline import InferencePipeline
from apps.inference_worker.province import PrefixProvinceClassifier

T0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)


def synthetic_scene(plate_text: str = "2D-0888") -> np.ndarray:
    """A frame containing one white plate with dark characters."""
    scene = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(scene, (200, 200), (440, 290), (245, 245, 245), -1)
    cv2.putText(
        scene, plate_text, (215, 262), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 15, 15), 4
    )
    return scene


# --- perspective -----------------------------------------------------------


def test_order_corners_normalises_winding():
    shuffled = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    ordered = order_corners(shuffled)
    assert tuple(ordered[0]) == (10, 10)
    assert tuple(ordered[2]) == (90, 90)


def test_warp_without_corners_falls_back_to_resize():
    plate = np.zeros((50, 100, 3), dtype=np.uint8)
    assert warp_plate(plate, None).shape[:2] == (160, 320)


def test_warp_rectifies_a_skewed_quad():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    corners = np.array([[20, 30], [170, 10], [180, 150], [30, 175]], dtype=np.float32)
    assert warp_plate(image, corners).shape[:2] == (160, 320)


def test_warp_with_origin_keeps_crop_content():
    """Detector corners are frame-space; warping a crop needs the crop origin.

    Without it the transform maps outside the crop and returns an all-black
    plate, which makes OCR silently read nothing.
    """
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    x1, y1, x2, y2 = 100, 80, 300, 180
    frame[y1:y2, x1:x2] = 255
    corners = np.array(
        [[x2 - 1, y2 - 1], [x1, y2 - 1], [x1, y1], [x2 - 1, y1]], dtype=np.float32
    )
    crop = frame[y1:y2, x1:x2]

    without_origin = warp_plate(crop, corners)
    with_origin = warp_plate(crop, corners, origin=(x1, y1))

    # The corrected warp recovers the whole white crop; the uncorrected one
    # samples mostly outside it and comes back nearly empty.
    assert with_origin.mean() > 250
    assert without_origin.mean() < with_origin.mean() / 2


def test_split_zones_covers_the_plate():
    plate = np.zeros((160, 320, 3), dtype=np.uint8)
    zones = split_zones(plate)
    assert set(zones) == {"top", "middle", "bottom"}
    assert all(z.size > 0 for z in zones.values())


# --- detector --------------------------------------------------------------


def test_detector_finds_the_plate_region():
    detections = ContourPlateDetector().detect(synthetic_scene())
    assert detections, "expected at least one plate candidate"

    # The real plate sits at (200,200)-(440,290); a candidate should overlap it.
    def overlaps(d: Detection) -> bool:
        return d.x1 < 440 and d.x2 > 200 and d.y1 < 290 and d.y2 > 200

    assert any(overlaps(d) for d in detections)


def test_detector_returns_nothing_on_blank_frame():
    blank = np.full((480, 640, 3), 70, dtype=np.uint8)
    assert ContourPlateDetector().detect(blank) == []


def test_detector_respects_max_detections():
    assert len(ContourPlateDetector(max_detections=1).detect(synthetic_scene())) <= 1


# --- OCR -------------------------------------------------------------------


def test_ocr_returns_empty_on_blank_input():
    result = TemplateOCR().read(np.zeros((60, 200, 3), dtype=np.uint8))
    assert result.text == ""
    assert result.confidence == 0.0


def test_ocr_reads_some_characters_from_rendered_text():
    canvas = np.full((80, 300, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "2D0888", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    result = TemplateOCR().read(canvas)
    # Classical template matching is weak; assert it segments, not that it is right.
    assert len(result.text) >= 4
    assert 0.0 <= result.confidence <= 1.0


# --- province --------------------------------------------------------------


@pytest.mark.parametrize(
    ("plate", "expected"),
    [("2D-0888", "2"), ("12A-3456", "12"), ("25X-0001", "25"), ("8B-1234", "8")],
)
def test_prefix_province_from_plate_text(plate, expected):
    assert PrefixProvinceClassifier().from_plate_text(plate).code == expected


def test_prefix_province_unknown_when_no_leading_digit():
    assert PrefixProvinceClassifier().from_plate_text("AB-1234").code is None


def test_prefix_province_rejects_out_of_range_code():
    assert PrefixProvinceClassifier().from_plate_text("99-1234").code == "9"


# --- pipeline --------------------------------------------------------------


class StubDetector:
    def detect(self, image):
        return [Detection(x1=200, y1=200, x2=440, y2=290, confidence=0.9)]


class StubOCR:
    def __init__(self, text="2D-0888"):
        self.text = text

    def read(self, plate_image):
        return OCRResult(text=self.text, confidence=0.88)


class StubProvince:
    def classify(self, top_zone):
        return ProvinceResult(code="12", confidence=0.8)


def make_pipeline(votes_required=3, ocr_text="2D-0888") -> InferencePipeline:
    return InferencePipeline(
        detector=StubDetector(),
        ocr=StubOCR(ocr_text),
        province_classifier=StubProvince(),
        model_version="test-v1",
        votes_required=votes_required,
    )


def frame_at(offset_seconds: float) -> Frame:
    return Frame(
        camera_id="cam1",
        image=synthetic_scene(),
        frame_ts=T0 + timedelta(seconds=offset_seconds),
    )


def test_pipeline_commits_only_after_quorum():
    pipeline = make_pipeline(votes_required=3)
    assert pipeline.process(frame_at(0)) == []
    assert pipeline.process(frame_at(0.1)) == []

    committed = pipeline.process(frame_at(0.2))
    assert len(committed) == 1
    assert committed[0].plate_text == "2D-0888"
    assert committed[0].model_version == "test-v1"


def test_pipeline_commits_each_vehicle_once():
    pipeline = make_pipeline(votes_required=2)
    results = [pipeline.process(frame_at(i * 0.1)) for i in range(6)]
    assert sum(len(r) for r in results) == 1


def test_pipeline_marks_valid_plate():
    pipeline = make_pipeline(votes_required=1)
    committed = pipeline.process(frame_at(0))
    assert committed[0].is_valid is True
    assert committed[0].plate_type == "private_car"


def test_pipeline_marks_malformed_plate_invalid():
    pipeline = make_pipeline(votes_required=1, ocr_text="XX-YY")
    committed = pipeline.process(frame_at(0))
    assert committed[0].is_valid is False


def test_pipeline_prefers_province_from_plate_prefix():
    pipeline = make_pipeline(votes_required=1)
    # Plate starts with "2", so province 2 wins over the classifier's "12".
    assert pipeline.process(frame_at(0))[0].province_code == "2"


def test_pipeline_records_all_confidence_scores():
    pipeline = make_pipeline(votes_required=1)
    read = pipeline.process(frame_at(0))[0]
    assert 0 < read.detector_confidence <= 1
    assert 0 < read.ocr_confidence <= 1
    assert 0 < read.province_confidence <= 1
    assert read.track_id


def test_pipeline_ignores_frames_with_no_detection():
    class Empty:
        def detect(self, image):
            return []

    pipeline = make_pipeline()
    pipeline.detector = Empty()
    assert pipeline.process(frame_at(0)) == []


# --- vanity plates ---------------------------------------------------------


class ConfidenceDetector:
    """StubDetector with a settable confidence, to drive the vanity threshold."""

    def __init__(self, confidence: float):
        self.confidence = confidence

    def detect(self, image):
        return [Detection(x1=200, y1=200, x2=440, y2=290, confidence=self.confidence)]


def vanity_pipeline(detector_confidence: float, ocr_text: str) -> InferencePipeline:
    return InferencePipeline(
        detector=ConfidenceDetector(detector_confidence),
        ocr=StubOCR(ocr_text),
        province_classifier=StubProvince(),
        model_version="test-v1",
        votes_required=1,
    )


def test_unreadable_text_from_a_confident_detection_is_marked_vanity():
    (read,) = vanity_pipeline(0.95, "KHMERNAME").process(frame_at(0))
    assert read.plate_type == "vanity"
    assert read.is_valid is False


def test_unreadable_text_from_a_weak_detection_is_not_marked_vanity():
    """A low-confidence box is more likely a misdetection than a VIP plate."""
    (read,) = vanity_pipeline(0.4, "KHMERNAME").process(frame_at(0))
    assert read.plate_type == "private_car"
    assert read.is_valid is False


def test_a_valid_plate_is_never_marked_vanity():
    (read,) = vanity_pipeline(0.99, "2D-0888").process(frame_at(0))
    assert read.plate_type == "private_car"
    assert read.is_valid is True


# --- container evidence -----------------------------------------------------


def test_committed_container_carries_evidence_paths(tmp_path):
    """With a capture store the committed read points at a saved crop and frame."""
    from apps.inference_worker.capture import ContainerCapture
    from apps.inference_worker.container import parse
    from apps.inference_worker.container_locator import Region
    from apps.inference_worker.container_ocr import ContainerReadResult
    from apps.inference_worker.container_pipeline import ContainerDetection

    class StubReader:
        def read_frame(self, image, boxes=()):
            n = parse("TCLU5437389")
            return ContainerDetection(Region(100, 40, 400, 80, False),
                                      ContainerReadResult("TCLU5437389", 0.9, n, True, False))

    pipe = InferencePipeline(
        detector=StubDetector(), ocr=StubOCR("2D-0888"), province_classifier=StubProvince(),
        model_version="test-v1", container_reader=StubReader(), container_votes_required=1,
        container_capture=ContainerCapture(tmp_path),
    )
    (read,) = pipe.process_containers(frame_at(0))
    assert read.container_number == "TCLU5437389"
    assert read.crop_path and (tmp_path / read.crop_path).is_file()
    assert read.frame_path and (tmp_path / read.frame_path).is_file()
    assert str(read.id)[:8] in read.crop_path


def test_committed_container_without_capture_has_no_paths():
    from apps.inference_worker.container import parse
    from apps.inference_worker.container_locator import Region
    from apps.inference_worker.container_ocr import ContainerReadResult
    from apps.inference_worker.container_pipeline import ContainerDetection

    class StubReader:
        def read_frame(self, image, boxes=()):
            n = parse("TCLU5437389")
            return ContainerDetection(Region(0, 0, 10, 5, False),
                                      ContainerReadResult("TCLU5437389", 0.9, n, None, False))

    pipe = InferencePipeline(
        detector=StubDetector(), ocr=StubOCR("2D-0888"), province_classifier=StubProvince(),
        model_version="test-v1", container_reader=StubReader(), container_votes_required=1,
    )
    (read,) = pipe.process_containers(frame_at(0))
    assert read.crop_path is None and read.frame_path is None
