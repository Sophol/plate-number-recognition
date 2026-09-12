"""The evidence store: files land where the row says, and paths never escape it."""

import json
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np

from apps.inference_worker.capture import ContainerCapture
from apps.inference_worker.container_locator import Region

T0 = datetime(2026, 9, 12, 20, 11, 56, tzinfo=UTC)


def test_save_writes_crop_frame_and_sidecar(tmp_path):
    frame = np.zeros((300, 600, 3), np.uint8)
    frame[40:80, 100:400] = 255
    store = ContainerCapture(tmp_path)
    read_id = uuid.uuid4()

    crop_rel, frame_rel = store.save(frame, Region(100, 40, 400, 80, False), "CRCU4352365", read_id, T0,
                                     meta={"ocr_text": "CRCU4352365"})

    local = T0.astimezone()
    assert crop_rel == f"{local:%Y-%m-%d}/CRCU4352365_{local:%H%M%S}_{str(read_id)[:8]}.jpg"
    assert frame_rel.endswith("_frame.jpg")
    crop = cv2.imread(str(tmp_path / crop_rel))
    full = cv2.imread(str(tmp_path / frame_rel))
    assert full.shape == frame.shape
    assert crop.shape[0] < 80 and crop.shape[1] > 300     # padded band, not the whole frame
    side = json.loads((tmp_path / crop_rel).with_suffix(".json").read_text())
    assert side["box"] == [100, 40, 400, 80] and side["ocr_text"] == "CRCU4352365"
    assert side["frame_size"] == [600, 300]


def test_resolve_stays_inside_root(tmp_path):
    store = ContainerCapture(tmp_path)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.jpg").write_bytes(b"x")
    assert store.resolve("d/a.jpg") == (tmp_path / "d" / "a.jpg").resolve()
    assert store.resolve("d/missing.jpg") is None
    assert store.resolve("../../etc/passwd") is None
    assert store.resolve("/etc/passwd") is None
