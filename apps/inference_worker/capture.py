"""Keeps the evidence for every committed container read: the crop and the frame.

A container_committed event is a claim the pipeline cannot prove after the
fact -- the frame is gone. Saving the OCR'd crop and the full frame at commit
time makes each record auditable against the gate's paper log, and turns every
transit into a labelled training example: the sidecar JSON carries the box and
the read, which is exactly what fine-tuning the OCR on real gate crops needs.

Files are laid out by day under the capture root, named so that a directory
listing already tells the story:

    2026-09-12/CRCU4352365_201156_a891417f.jpg          # the crop that was read
    2026-09-12/CRCU4352365_201156_a891417f_frame.jpg    # the whole frame, undrawn
    2026-09-12/CRCU4352365_201156_a891417f.json         # box, read, flags

Paths are returned relative to the root, so the root can move (or be served
by nginx) without touching the database rows that point at the files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import structlog

from apps.inference_worker.container_locator import Region

log = structlog.get_logger()


class ContainerCapture:
    def __init__(self, root: str | Path, jpeg_quality: int = 92) -> None:
        self.root = Path(root)
        self.jpeg_quality = jpeg_quality

    def save(self, frame: np.ndarray, region: Region, number: str, read_id, frame_ts: datetime,
             meta: dict | None = None, rejected: bool = False) -> tuple[str, str]:
        """Write crop, frame and sidecar; return (crop_path, frame_path) relative to root.

        `rejected` files a candidate the gate refused to commit under rejected/:
        no database row points at it, but it is exactly the hard negative (or
        the missed truck) the next model version should be trained on.
        """
        # Local wall-clock time in the name: the file should match the camera's
        # overlay and the gate log, which are what a person compares it against.
        local = frame_ts.astimezone() if frame_ts.tzinfo else frame_ts
        day = (self.root / "rejected" if rejected else self.root) / local.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = f"{number}_{local.strftime('%H%M%S')}_{str(read_id)[:8]}"
        prefix = f"rejected/{day.name}" if rejected else day.name
        crop_rel, frame_rel = f"{prefix}/{stem}.jpg", f"{prefix}/{stem}_frame.jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        crop = region.crop(frame)
        if crop.size:
            cv2.imwrite(str(self.root / crop_rel), crop, params)
        cv2.imwrite(str(self.root / frame_rel), frame, params)
        (day / f"{stem}.json").write_text(json.dumps({
            "container_number": number,
            "read_id": str(read_id),
            "frame_ts": frame_ts.isoformat(),
            "box": [region.x1, region.y1, region.x2, region.y2],
            "vertical": region.vertical,
            "frame_size": [frame.shape[1], frame.shape[0]],
            **(meta or {}),
        }, indent=1))
        return crop_rel, frame_rel

    def resolve(self, rel: str) -> Path | None:
        """Absolute path for a stored relative path, or None if it escapes the root."""
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None
