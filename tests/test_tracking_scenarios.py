"""End-to-end tracking scored against generated ground truth.

Unit tests cover the tracker's matching logic in isolation; these run whole
clips through detector -> tracker -> OCR -> voting and assert on the outcome
that matters: one stable track per vehicle transit.
"""

import subprocess
import sys
from datetime import UTC, datetime, timedelta

import cv2
import pytest

from apps.camera_worker.sampler import Frame
from apps.inference_worker.detector import ContourPlateDetector
from apps.inference_worker.main import build_pipeline


def render(tmp_path, scenario: str):
    out = tmp_path / f"traffic_{scenario}"
    subprocess.run(
        [sys.executable, "-m", "scripts.make_test_video",
         "--scenario", scenario, "--out", str(out)],
        check=True, capture_output=True,
    )
    return out.with_suffix(".mp4")


def run(video) -> dict:
    capture = cv2.VideoCapture(str(video))
    pipeline = build_pipeline("test", votes_required=3)
    detector = ContourPlateDetector()
    base = datetime.now(UTC)

    tracks: set[str] = set()
    reads: list[tuple[str, str]] = []
    n = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        ts = base + timedelta(seconds=n / 10.0)
        for track_id, _ in pipeline.tracker.update(detector.detect(image), ts):
            tracks.add(track_id)
        for read in pipeline.process(Frame(camera_id="c", image=image, frame_ts=ts)):
            reads.append((read.track_id, read.plate_text))
        n += 1
    capture.release()
    return {"tracks": tracks, "reads": reads, "texts": {t for _, t in reads}}


@pytest.mark.slow
def test_separated_vehicles_get_one_track_each(tmp_path):
    result = run(render(tmp_path, "simple"))
    assert len(result["tracks"]) == 3
    assert result["texts"] == {"2D-0888", "1A-4412", "3C-9001"}


@pytest.mark.slow
def test_fast_vehicles_still_hold_one_track_each(tmp_path):
    """Speed alone does not break tracking while the plate stays detectable."""
    result = run(render(tmp_path, "fast"))
    assert len(result["tracks"]) == 3
    assert result["texts"] == {"2D-0888", "1A-4412", "3C-9001"}


@pytest.mark.slow
def test_crossing_vehicles_lose_their_tracks(tmp_path):
    """Documents a known limitation rather than asserting correct behaviour.

    When two plates overlap the contour detector merges them into one blob and
    then loses it entirely for ~7 frames -- past max_misses -- so both tracks
    are dropped and re-created with new ids. The plate text still survives via
    voting. Swapping IoUTracker for ByteTrack is the documented fix; if that
    lands, this test should start failing and be tightened to == 2.
    """
    result = run(render(tmp_path, "crossing"))
    assert len(result["tracks"]) > 2, "crossing now tracks cleanly - tighten this test"
    assert result["texts"] == {"2D-0888", "1A-4412"}
