"""Container reading for a frame, and per-transit voting.

`ContainerReader` runs the chain the live test proved: propose candidate text
bands (inside each detected vehicle AND across the whole frame -- a vehicle box
that misses the container must not hide it), OCR every candidate, keep the best
read whose ISO 6346 check digit is correct, then ask PAS whether the terminal
knows that number. One PAS query per frame at most, on the winner only, and the
client caches, so a container sitting at the barrier costs one lookup.

`ContainerVoter` turns per-frame reads into one record per transit. Unlike
plates, the read itself is already noise-filtered by the checksum, so votes are
keyed by the number: a valid number seen `votes_required` times within a short
window commits once, then is suppressed for a cooldown so the same box does not
produce a record every frame while it waits at the gate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from apps.inference_worker.container_locator import ContainerLocator, Region
from apps.inference_worker.container_ocr import ContainerReadResult, OnnxContainerOCR
from apps.inference_worker.pas import PasClient


@dataclass(slots=True)
class ContainerDetection:
    region: Region
    result: ContainerReadResult


def _iou(a: Region, b: Region) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union else 0.0


class ContainerReader:
    def __init__(self, ocr: OnnxContainerOCR, locator: ContainerLocator | None = None,
                 pas: PasClient | None = None, min_confidence: float = 0.5) -> None:
        self.ocr = ocr
        self.locator = locator or ContainerLocator()
        self.pas = pas
        self.min_confidence = min_confidence

    def read_frame(self, frame, vehicle_boxes=()) -> ContainerDetection | None:
        regions: list[Region] = []
        for box in [*vehicle_boxes, None]:
            for candidate in self.locator.propose(frame, box):
                if all(_iou(candidate, r) < 0.6 for r in regions):
                    regions.append(candidate)

        best: ContainerDetection | None = None
        for region in regions:
            result = self.ocr.read(region.crop(frame))
            if not result.ok or result.confidence < self.min_confidence:
                continue
            if best is None or result.confidence > best.result.confidence:
                best = ContainerDetection(region, result)

        if best is not None and self.pas is not None:
            live = self.pas.is_known(best.result.number.canonical)
            if live is not None:            # PAS answered (or its cache did)
                best.result.is_known = live
        return best


@dataclass(slots=True)
class ContainerVote:
    number: str
    confidence: float
    votes: int
    is_known: bool | None
    was_snapped: bool
    ocr_text: str


class ContainerVoter:
    def __init__(self, votes_required: int = 2, window: timedelta = timedelta(seconds=15),
                 cooldown: timedelta = timedelta(seconds=120)) -> None:
        self.votes_required = votes_required
        self.window = window
        self.cooldown = cooldown
        self._seen: dict[tuple[str, str], list[tuple[datetime, ContainerReadResult]]] = defaultdict(list)
        self._committed_at: dict[tuple[str, str], datetime] = {}

    def add(self, camera_id: str, result: ContainerReadResult, at: datetime) -> ContainerVote | None:
        if not result.ok:
            return None
        key = (camera_id, result.number.canonical)

        last = self._committed_at.get(key)
        if last is not None and at - last < self.cooldown:
            return None

        entries = self._seen[key]
        entries.append((at, result))
        entries[:] = [(t, r) for t, r in entries if at - t <= self.window]
        if len(entries) < self.votes_required:
            return None

        self._committed_at[key] = at
        self._seen.pop(key, None)
        results = [r for _, r in entries]
        known_votes = [r.is_known for r in results if r.is_known is not None]
        return ContainerVote(
            number=result.number.canonical,
            confidence=sum(r.confidence for r in results) / len(results),
            votes=len(results),
            is_known=(sum(known_votes) > len(known_votes) / 2) if known_votes else None,
            was_snapped=any(r.was_snapped for r in results),
            ocr_text=result.text,
        )
