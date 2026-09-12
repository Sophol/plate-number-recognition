import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog

from apps.camera_worker.sampler import Frame
from apps.inference_worker import validator
from apps.inference_worker.interfaces import OCR, Detector, ProvinceClassifier
from apps.inference_worker.perspective import split_zones, warp_plate
from apps.inference_worker.province import PrefixProvinceClassifier
from apps.inference_worker.tracker import IoUTracker
from apps.inference_worker.container_pipeline import ContainerVoter
from apps.inference_worker.voting import Candidate, TrackVoter, VoteResult
from plate_types import VANITY

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


@dataclass(slots=True)
class CommittedContainer:
    """A container number that won its vote and is ready to persist."""

    id: uuid.UUID
    camera_id: str
    container_number: str
    owner_code: str
    checksum_ok: bool
    is_known: bool | None
    ocr_text: str
    was_snapped: bool
    confidence: float
    frame_ts: datetime
    model_version: str
    crop_path: str | None = None
    frame_path: str | None = None


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
        # Above this detector confidence, an unparseable read is treated as a
        # vanity plate rather than a failed one. Tune against the trained
        # detector: the contour fallback reports a fill ratio, not a real
        # probability, so this threshold means little until YOLO is in place.
        vanity_min_detector_confidence: float = 0.8,
        vehicle_detector=None,
        container_reader=None,
        container_votes_required: int = 2,
        container_every_n_frames: int = 1,
        container_capture=None,
        container_unknown_min_confidence: float = 0.8,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.province_classifier = province_classifier
        self.model_version = model_version
        self.default_plate_type = default_plate_type
        self.vanity_min_detector_confidence = vanity_min_detector_confidence
        self.tracker = IoUTracker()
        self.voter = TrackVoter(votes_required=votes_required)
        self.prefix_province = PrefixProvinceClassifier()
        self.vehicle_detector = vehicle_detector
        self.container_reader = container_reader
        self.container_voter = ContainerVoter(votes_required=container_votes_required)
        self.container_every_n_frames = max(1, container_every_n_frames)
        self._container_frame_counter = 0
        # Saves crop + frame for each committed read; None keeps nothing.
        self.container_capture = container_capture
        self.container_unknown_min_confidence = container_unknown_min_confidence
        # Rejected candidates are captured for training, but a static texture
        # re-reads as the same string for as long as it is lit, so once per
        # number per camera per this interval is plenty.
        self._rejected_saved_at: dict[tuple[str, str], datetime] = {}

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

            if not is_valid and detection.confidence >= self.vanity_min_detector_confidence:
                # The detector is sure this is a plate, yet the text fits no
                # known format. On a Cambodian gate that is usually a VIP vanity
                # plate carrying a Khmer name instead of a number. It still must
                # not open the gate, but it is a different problem from a dirty
                # plate or a bad read, and the operator needs to see which.
                plate_type = VANITY

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

    def process_containers(self, frame: Frame) -> list[CommittedContainer]:
        """Read a container number from the frame and commit it once per transit.

        Separate from process() so the plate path and its tests are untouched:
        a container read is its own event, voted by number rather than by
        plate track, because the checksum has already filtered the noise that
        plate voting exists to absorb.
        """
        if self.container_reader is None:
            return []
        self._container_frame_counter += 1
        if self._container_frame_counter % self.container_every_n_frames:
            return []
        vehicles = self.vehicle_detector.detect(frame.image) if self.vehicle_detector else []
        found = self.container_reader.read_frame(
            frame.image, [(v.x1, v.y1, v.x2, v.y2) for v in vehicles]
        )
        if found is None:
            return []
        reason = self._reject_reason(found.result, vehicles)
        if reason is not None:
            self._capture_rejected(frame, found, vehicles, reason)
            return []
        vote = self.container_voter.add(frame.camera_id, found.result, frame.frame_ts)
        if vote is None:
            return []
        read_id = uuid.uuid4()
        crop_path = frame_path = None
        if self.container_capture is not None:
            # The evidence is worth more than a dropped record is: a failed write
            # is logged and the read still commits without paths.
            try:
                crop_path, frame_path = self.container_capture.save(
                    frame.image, found.region, vote.number, read_id, frame.frame_ts,
                    meta={"camera_id": frame.camera_id, "ocr_text": vote.ocr_text,
                          "confidence": round(vote.confidence, 4), "is_known": vote.is_known,
                          "was_snapped": vote.was_snapped,
                          "vehicles": [[v.x1, v.y1, v.x2, v.y2, v.vehicle_type, v.colour] for v in vehicles]},
                )
            except Exception:
                log.exception("container_capture_failed", container_number=vote.number)
        return [
            CommittedContainer(
                id=read_id,
                camera_id=frame.camera_id,
                container_number=vote.number,
                owner_code=vote.number[:4],
                checksum_ok=True,
                is_known=vote.is_known,
                ocr_text=vote.ocr_text,
                was_snapped=vote.was_snapped,
                confidence=vote.confidence,
                frame_ts=frame.frame_ts,
                model_version=self.model_version,
                crop_path=crop_path,
                frame_path=frame_path,
            )
        ]

    def _reject_reason(self, result, vehicles) -> str | None:
        """Why a checksum-valid read must not commit, or None if it may.

        The checksum alone is weak: a random string passes it one time in ten,
        and static texture (a kerb, road chevrons, a window frame) re-reads as
        the same string frame after frame, so the voter agrees with itself.
        Every one of the first live commits was exactly that, in an empty lane.
        So: a vehicle must be in the frame, always. A number PAS knows and the
        OCR read outright is then trusted; a snapped read is by construction
        an OCR error corrected by lookup (with 171k known numbers, garbage is
        often one edit from a real one), so it and unknown numbers must also be
        confident.
        """
        if not vehicles:
            return "no_vehicle"
        trusted = result.is_known is True and not result.was_snapped
        if not trusted and result.confidence < self.container_unknown_min_confidence:
            return "snapped_low_confidence" if result.was_snapped else "unknown_low_confidence"
        return None

    def _capture_rejected(self, frame: Frame, found, vehicles, reason: str) -> None:
        number = found.result.number.canonical
        log.info("container_read_rejected", camera_id=frame.camera_id, container_number=number,
                 reason=reason, confidence=round(found.result.confidence, 3),
                 was_snapped=found.result.was_snapped, ocr_text=found.result.text)
        if self.container_capture is None:
            return
        key = (frame.camera_id, number)
        last = self._rejected_saved_at.get(key)
        if last is not None and frame.frame_ts - last < timedelta(minutes=10):
            return
        self._rejected_saved_at[key] = frame.frame_ts
        try:
            self.container_capture.save(
                frame.image, found.region, number, uuid.uuid4(), frame.frame_ts, rejected=True,
                meta={"camera_id": frame.camera_id, "rejected": reason, "ocr_text": found.result.text,
                      "confidence": round(found.result.confidence, 4), "is_known": found.result.is_known,
                      "was_snapped": found.result.was_snapped,
                      "vehicles": [[v.x1, v.y1, v.x2, v.y2, v.vehicle_type, v.colour] for v in vehicles]},
            )
        except Exception:
            log.exception("container_capture_failed", container_number=number)

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
