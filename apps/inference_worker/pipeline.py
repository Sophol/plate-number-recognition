import uuid
from dataclasses import dataclass
from datetime import datetime

import structlog

from apps.camera_worker.sampler import Frame
from apps.inference_worker import validator
from apps.inference_worker.interfaces import OCR, Detector, ProvinceClassifier
from apps.inference_worker.perspective import split_zones, warp_plate
from apps.inference_worker.province import PrefixProvinceClassifier
from apps.inference_worker.tracker import IoUTracker
from apps.inference_worker.voting import Candidate, TrackVoter, VoteResult

log = structlog.get_logger()


@dataclass(slots=True)
class CommittedRead:
    """A vote that reached quorum and is ready to persist."""

    id: uuid.UUID
    camera_id: str
    track_id: str
    plate_text: str
    province_code: str | None
    plate_type: str | None
    confidence: float
    detector_confidence: float
    ocr_confidence: float
    province_confidence: float
    frame_ts: datetime
    model_version: str
    is_valid: bool


class InferencePipeline:
    """detect -> warp -> zones -> OCR + province -> track -> validate -> vote.

    Backends are injected, so the classical-CV fallbacks can be swapped for
    trained ONNX models without touching this orchestration.
    """

    def __init__(
        self,
        detector: Detector,
        ocr: OCR,
        province_classifier: ProvinceClassifier,
        model_version: str,
        votes_required: int = 3,
        default_plate_type: str = "private_car",
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.province_classifier = province_classifier
        self.model_version = model_version
        self.default_plate_type = default_plate_type
        self.tracker = IoUTracker()
        self.voter = TrackVoter(votes_required=votes_required)
        self.prefix_province = PrefixProvinceClassifier()

    def process(self, frame: Frame) -> list[CommittedRead]:
        detections = self.detector.detect(frame.image)
        if not detections:
            return []

        committed: list[CommittedRead] = []
        for track_id, detection in self.tracker.update(detections, frame.frame_ts):
            crop = frame.image[detection.y1 : detection.y2, detection.x1 : detection.x2]
            if crop.size == 0:
                continue

            plate = warp_plate(crop, detection.corners, origin=(detection.x1, detection.y1))
            zones = split_zones(plate)

            reading = self.ocr.read(zones["middle"])
            if not reading.text:
                continue

            plate_type = detection.plate_type or self.default_plate_type
            corrected = validator.correct(reading.text, plate_type)
            is_valid = validator.validate(corrected, plate_type)

            province = self._resolve_province(corrected, zones["bottom"])

            result = self.voter.add(
                track_id,
                Candidate(
                    plate_text=corrected,
                    province_code=province.code,
                    confidence=min(detection.confidence, reading.confidence),
                ),
            )
            if result is None:
                continue

            committed.append(
                self._build(
                    frame, track_id, result, detection, reading, province, plate_type, is_valid
                )
            )

        return committed

    def _resolve_province(self, plate_text: str, bottom_zone):
        """Prefer the plate's numeric prefix; fall back to reading the text zone."""
        from_prefix = self.prefix_province.from_plate_text(plate_text)
        if from_prefix.code is not None:
            return from_prefix
        return self.province_classifier.classify(bottom_zone)

    def _build(
        self, frame, track_id, result: VoteResult, detection, reading, province, plate_type, is_valid
    ) -> CommittedRead:
        return CommittedRead(
            id=uuid.uuid4(),
            camera_id=frame.camera_id,
            track_id=track_id,
            plate_text=result.plate_text,
            province_code=result.province_code,
            plate_type=plate_type,
            confidence=result.confidence,
            detector_confidence=detection.confidence,
            ocr_confidence=reading.confidence,
            province_confidence=province.confidence,
            frame_ts=frame.frame_ts,
            model_version=self.model_version,
            is_valid=is_valid,
        )
