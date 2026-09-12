import cv2
import numpy as np
import pytest

from apps.camera_worker import rtsp
from apps.camera_worker.main import run_camera
from apps.camera_worker.motion import ROI
from apps.camera_worker.sampler import BoundedFrameQueue


@pytest.fixture
def video_file(tmp_path):
    """A 60-frame clip with a block moving across the left half, then a static tail."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    for i in range(40):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = i * 4
        frame[80:160, x : x + 40] = 255
        writer.write(frame)
    for _ in range(20):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    assert path.exists()
    return path


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(rtsp.asyncio, "sleep", instant)


class FileSource(rtsp.RTSPSource):
    """Reads a local file through the real RTSPSource read/reconnect path."""

    def _open(self):
        capture = cv2.VideoCapture(str(self.url))
        return capture if capture.isOpened() else None


@pytest.mark.asyncio
async def test_pipeline_samples_and_gates_real_video(video_file, monkeypatch):
    from apps.camera_worker import main as worker_main

    monkeypatch.setattr(worker_main, "RTSPSource", FileSource)

    queue = BoundedFrameQueue(maxsize=64)
    # File playback runs far faster than wall clock, so a high target_fps keeps
    # the sampler from thinning everything away; motion gating is what we assert.
    await run_camera(
        camera_id="cam-test",
        rtsp_url=str(video_file),
        queue=queue,
        target_fps=1000.0,
        roi=ROI(),
        min_changed_fraction=0.002,
        max_reconnects=0,
    )

    emitted = []
    while queue.qsize():
        emitted.append(await queue.get())

    assert emitted, "expected the moving section to produce frames"
    # Static tail must be suppressed: fewer emitted than the 60 source frames.
    assert len(emitted) < 60
    assert all(f.camera_id == "cam-test" for f in emitted)
    assert all(f.frame_ts.tzinfo is not None for f in emitted)
    assert all(f.image.shape == (240, 320, 3) for f in emitted)


@pytest.mark.asyncio
async def test_pipeline_respects_roi(video_file, monkeypatch):
    from apps.camera_worker import main as worker_main

    monkeypatch.setattr(worker_main, "RTSPSource", FileSource)

    queue = BoundedFrameQueue(maxsize=64)
    # All motion is in the left half; watching only the far right sees nothing.
    await run_camera(
        camera_id="cam-roi",
        rtsp_url=str(video_file),
        queue=queue,
        target_fps=1000.0,
        roi=ROI(x1=0.85, y1=0.0, x2=1.0, y2=1.0),
        min_changed_fraction=0.02,
        max_reconnects=0,
    )

    assert queue.qsize() == 0
