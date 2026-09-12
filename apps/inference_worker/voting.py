from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class Candidate:
    plate_text: str
    province_code: str | None
    confidence: float


@dataclass(slots=True)
class VoteResult:
    plate_text: str
    province_code: str | None
    confidence: float
    votes: int
    total: int


class TrackVoter:
    """Accumulates per-track candidates and commits once a plate reaches quorum.

    Voting is scoped to a track_id so votes never mix frames from different
    vehicles, and each track commits at most once.
    """

    def __init__(self, votes_required: int = 3) -> None:
        self.votes_required = votes_required
        self._candidates: dict[str, list[Candidate]] = defaultdict(list)
        self._committed: set[str] = set()

    def add(self, track_id: str, candidate: Candidate) -> VoteResult | None:
        if track_id in self._committed:
            return None

        self._candidates[track_id].append(candidate)
        entries = self._candidates[track_id]
        counts = Counter(c.plate_text for c in entries)
        plate_text, votes = counts.most_common(1)[0]

        if votes < self.votes_required:
            return None

        agreeing = [c for c in entries if c.plate_text == plate_text]
        province_counts = Counter(c.province_code for c in agreeing)
        self._committed.add(track_id)

        return VoteResult(
            plate_text=plate_text,
            province_code=province_counts.most_common(1)[0][0],
            confidence=sum(c.confidence for c in agreeing) / len(agreeing),
            votes=votes,
            total=len(entries),
        )

    def close(self, track_id: str) -> None:
        self._candidates.pop(track_id, None)
        self._committed.discard(track_id)
