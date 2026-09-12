from datetime import UTC, datetime, timedelta

from apps.inference_worker.interfaces import Detection
from apps.inference_worker.tracker import IoUTracker, iou

T0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)


def det(x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    assert 0.1 < iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.5


def test_new_detection_creates_track():
    tracker = IoUTracker()
    assignments = tracker.update([det(0, 0, 10, 10)], T0)
    assert len(assignments) == 1
    assert len(tracker.active_tracks) == 1


def test_same_vehicle_keeps_track_id():
    tracker = IoUTracker()
    first = tracker.update([det(0, 0, 100, 50)], T0)[0][0]
    # Vehicle moved slightly; boxes still overlap heavily.
    second = tracker.update([det(5, 2, 105, 52)], T0 + timedelta(seconds=0.1))[0][0]
    assert first == second


def test_distant_vehicle_gets_new_track_id():
    tracker = IoUTracker()
    first = tracker.update([det(0, 0, 100, 50)], T0)[0][0]
    second = tracker.update([det(500, 300, 600, 350)], T0 + timedelta(seconds=0.1))[0][0]
    assert first != second


def test_two_vehicles_tracked_independently():
    tracker = IoUTracker()
    ids = {t for t, _ in tracker.update([det(0, 0, 100, 50), det(400, 0, 500, 50)], T0)}
    assert len(ids) == 2

    later = {
        d.x1: t
        for t, d in tracker.update(
            [det(5, 0, 105, 50), det(405, 0, 505, 50)], T0 + timedelta(seconds=0.1)
        )
    }
    assert set(later.values()) == ids


def test_track_expires_after_max_misses():
    tracker = IoUTracker(max_misses=2)
    tracker.update([det(0, 0, 100, 50)], T0)
    for i in range(3):
        tracker.update([], T0 + timedelta(seconds=i + 1))
    assert tracker.active_tracks == {}


def test_expired_since_reports_stale_tracks():
    tracker = IoUTracker()
    tracker.update([det(0, 0, 100, 50)], T0)
    assert tracker.expired_since(T0 + timedelta(seconds=1), gap_seconds=5) == []
    assert tracker.expired_since(T0 + timedelta(seconds=10), gap_seconds=5) != []


def test_greedy_matching_prefers_best_overlap():
    tracker = IoUTracker()
    tracker.update([det(0, 0, 100, 50)], T0)
    # Near-perfect match plus a weak one; the strong pair must win the track.
    assignments = tracker.update(
        [det(60, 0, 160, 50), det(2, 1, 102, 51)], T0 + timedelta(seconds=0.1)
    )
    by_x = {d.x1: t for t, d in assignments}
    assert by_x[2] != by_x[60]
    assert len(tracker.active_tracks) == 2
