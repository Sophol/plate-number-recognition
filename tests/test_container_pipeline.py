"""Tests for the frame-level container reader and the per-transit voter.

Stubs stand in for the locator and OCR, so what is tested is the orchestration:
that the whole frame is searched as well as each vehicle box, that the best
checksum-valid candidate wins, that the live PAS answer overrides the cached
flag, and that the voter commits once per container per transit.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from apps.inference_worker.container import parse
from apps.inference_worker.container_locator import Region
from apps.inference_worker.container_ocr import ContainerReadResult
from apps.inference_worker.container_pipeline import ContainerReader, ContainerVoter

T0 = datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC)
FRAME = np.zeros((400, 800, 3), np.uint8)


def result(text, conf=0.9, known=None, snapped=False):
    n = parse(text)
    return ContainerReadResult(text, conf, n if (n and n.checksum_ok) else None, known, snapped)


class StubLocator:
    """Returns different regions for a vehicle box and for the full frame."""

    def propose(self, frame, box=None):
        if box is None:
            return [Region(600, 20, 780, 60, False)]        # the container, outside the box
        return [Region(box[0] + 10, box[1] + 10, box[0] + 110, box[1] + 30, False)]


class StubOCR:
    def __init__(self, by_x1):
        self.by_x1 = by_x1        # region.x1 -> result

    def read(self, crop):
        return self.calls.pop(0)

    def read_region(self, region):
        return self.by_x1[region.x1]


class RegionOCR:
    """OCR keyed by where the crop came from, via a monkeypatched Region.crop."""

    def __init__(self, by_x1):
        self.by_x1 = by_x1
        self.last = None

    def read(self, crop):
        return self.by_x1.get(int(crop[0, 0, 0]), result("GARBAGE"))


def crop_tag(self, frame, pad=0.12):
    # Encode the region's x1 into the crop so the stub OCR can tell regions apart.
    tagged = np.zeros((4, 4, 3), np.uint8)
    tagged[0, 0, 0] = self.x1 % 256
    return tagged


def test_full_frame_is_searched_even_when_a_vehicle_box_misses(monkeypatch):
    monkeypatch.setattr(Region, "crop", crop_tag)
    ocr = RegionOCR({610 % 256: result("GARBAGE"), 600 % 256: result("TCLU5437389")})
    reader = ContainerReader(ocr, locator=StubLocator())

    det = reader.read_frame(FRAME, vehicle_boxes=[(600, 200, 790, 390)])
    assert det is not None
    assert det.result.number.canonical == "TCLU5437389"
    assert det.region.x1 == 600


def test_best_confidence_valid_candidate_wins(monkeypatch):
    monkeypatch.setattr(Region, "crop", crop_tag)

    class TwoRegions:
        def propose(self, frame, box=None):
            return [] if box else [Region(10, 0, 100, 20, False), Region(200, 0, 300, 20, False)]

    ocr = RegionOCR({10: result("TCLU5437389", 0.6), 200: result("CMAU7224270", 0.95)})
    det = ContainerReader(ocr, locator=TwoRegions()).read_frame(FRAME)
    assert det.result.number.canonical == "CMAU7224270"


def test_below_min_confidence_is_ignored(monkeypatch):
    monkeypatch.setattr(Region, "crop", crop_tag)

    class One:
        def propose(self, frame, box=None):
            return [] if box else [Region(10, 0, 100, 20, False)]

    ocr = RegionOCR({10: result("TCLU5437389", 0.3)})
    assert ContainerReader(ocr, locator=One(), min_confidence=0.5).read_frame(FRAME) is None


def test_live_pas_answer_overrides_cached_flag(monkeypatch):
    monkeypatch.setattr(Region, "crop", crop_tag)

    class One:
        def propose(self, frame, box=None):
            return [] if box else [Region(10, 0, 100, 20, False)]

    class Pas:
        def is_known(self, n):
            return False                 # PAS says: never seen it

    ocr = RegionOCR({10: result("TCLU5437389", known=True)})   # cache said known
    det = ContainerReader(ocr, locator=One(), pas=Pas()).read_frame(FRAME)
    assert det.result.is_known is False


def test_pas_silence_keeps_cached_flag(monkeypatch):
    monkeypatch.setattr(Region, "crop", crop_tag)

    class One:
        def propose(self, frame, box=None):
            return [] if box else [Region(10, 0, 100, 20, False)]

    class Pas:
        def is_known(self, n):
            return None

    ocr = RegionOCR({10: result("TCLU5437389", known=True)})
    assert ContainerReader(ocr, locator=One(), pas=Pas()).read_frame(FRAME).result.is_known is True


# --- voter ---------------------------------------------------------------


def test_voter_commits_once_at_quorum():
    voter = ContainerVoter(votes_required=2)
    assert voter.add("cam", result("TCLU5437389", 0.8, known=True), T0) is None
    vote = voter.add("cam", result("TCLU5437389", 0.9, known=True), T0 + timedelta(seconds=1))
    assert vote is not None
    assert vote.number == "TCLU5437389" and vote.votes == 2
    assert vote.confidence == pytest.approx(0.85) and vote.is_known is True
    # The box is still at the barrier: no second record within the cooldown.
    assert voter.add("cam", result("TCLU5437389"), T0 + timedelta(seconds=5)) is None


def test_voter_invalid_reads_never_count():
    voter = ContainerVoter(votes_required=1)
    assert voter.add("cam", result("TCLU5437380"), T0) is None      # bad check digit


def test_voter_window_expires_stale_votes():
    voter = ContainerVoter(votes_required=2, window=timedelta(seconds=10))
    voter.add("cam", result("TCLU5437389"), T0)
    assert voter.add("cam", result("TCLU5437389"), T0 + timedelta(seconds=30)) is None


def test_voter_recommits_after_cooldown():
    voter = ContainerVoter(votes_required=1, cooldown=timedelta(seconds=120))
    assert voter.add("cam", result("TCLU5437389"), T0) is not None
    assert voter.add("cam", result("TCLU5437389"), T0 + timedelta(seconds=60)) is None
    assert voter.add("cam", result("TCLU5437389"), T0 + timedelta(seconds=200)) is not None


def test_voter_keys_by_camera():
    voter = ContainerVoter(votes_required=1)
    assert voter.add("in", result("TCLU5437389"), T0) is not None
    assert voter.add("out", result("TCLU5437389"), T0) is not None


def test_voter_known_is_majority_of_answered_votes():
    voter = ContainerVoter(votes_required=3)
    voter.add("cam", result("TCLU5437389", known=None), T0)
    voter.add("cam", result("TCLU5437389", known=True), T0)
    vote = voter.add("cam", result("TCLU5437389", known=True), T0)
    assert vote.is_known is True
