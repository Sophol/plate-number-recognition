from dataclasses import dataclass, field
from datetime import datetime

from apps.inference_worker.interfaces import Detection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class Track:
    track_id: str
    box: tuple[int, int, int, int]
    first_seen: datetime
    last_seen: datetime
    hits: int = 1
    misses: int = 0
    history: list[tuple[int, int, int, int]] = field(default_factory=list)


class IoUTracker:
    """Greedy IoU tracker giving each vehicle transit a stable track_id.

    Frame voting and dedup are scoped to a track, so a vehicle is committed once
    regardless of how many frames it appears in. Sufficient at the plan's 5-10
    FPS with well-separated vehicles; swap for ByteTrack if plates overlap
    heavily or vehicles move fast enough to break frame-to-frame overlap.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: dict[str, Track] = {}
        self._next_id = 0

    @property
    def active_tracks(self) -> dict[str, Track]:
        return self._tracks

    def update(
        self, detections: list[Detection], frame_ts: datetime
    ) -> list[tuple[str, Detection]]:
        """Assign detections to tracks, returning (track_id, detection) pairs."""
        assignments: list[tuple[str, Detection]] = []
        unmatched = list(detections)
        matched_ids: set[str] = set()

        # Greedy: best-overlapping pair first, so a detection cannot be claimed
        # by a worse-matching track that happened to be considered earlier.
        while unmatched:
            best_pair: tuple[float, str, Detection] | None = None
            for track_id, track in self._tracks.items():
                if track_id in matched_ids:
                    continue
                for detection in unmatched:
                    score = iou(track.box, detection.box)
                    if score >= self.iou_threshold and (
                        best_pair is None or score > best_pair[0]
                    ):
                        best_pair = (score, track_id, detection)

            if best_pair is None:
                break

            _, track_id, detection = best_pair
            track = self._tracks[track_id]
            track.history.append(track.box)
            track.box = detection.box
            track.last_seen = frame_ts
            track.hits += 1
            track.misses = 0

            matched_ids.add(track_id)
            unmatched.remove(detection)
            assignments.append((track_id, detection))

        for detection in unmatched:
            track_id = f"t{self._next_id}"
            self._next_id += 1
            self._tracks[track_id] = Track(
                track_id=track_id,
                box=detection.box,
                first_seen=frame_ts,
                last_seen=frame_ts,
            )
            matched_ids.add(track_id)
            assignments.append((track_id, detection))

        for track_id in list(self._tracks):
            if track_id in matched_ids:
                continue
            self._tracks[track_id].misses += 1
            if self._tracks[track_id].misses > self.max_misses:
                del self._tracks[track_id]

        return assignments

    def expired_since(self, frame_ts: datetime, gap_seconds: float) -> list[str]:
        """Track ids untouched for longer than the gap; their transit has ended."""
        return [
            track_id
            for track_id, track in self._tracks.items()
            if (frame_ts - track.last_seen).total_seconds() > gap_seconds
        ]
