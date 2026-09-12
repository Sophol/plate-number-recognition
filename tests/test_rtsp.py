import numpy as np
import pytest

from apps.camera_worker import rtsp
from apps.camera_worker.rtsp import RTSPSource


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Reconnect backoff would otherwise make these tests take ~30s."""

    async def instant(_seconds):
        return None

    monkeypatch.setattr(rtsp.asyncio, "sleep", instant)


class FakeCapture:
    """Stands in for cv2.VideoCapture; only release() is called on it."""

    def release(self) -> None:
        pass


def frame(value: int = 1) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


async def take(source: RTSPSource, n: int) -> list:
    out = []
    async for image, ts in source.frames():
        out.append((image, ts))
        if len(out) == n:
            break
    return out


@pytest.mark.asyncio
async def test_yields_frames_with_timestamps(monkeypatch):
    source = RTSPSource("rtsp://fake", "cam1")
    monkeypatch.setattr(source, "_open", lambda: FakeCapture())
    monkeypatch.setattr(source, "_read", lambda: frame())

    results = await take(source, 3)
    assert len(results) == 3
    assert all(img.shape == (4, 4, 3) for img, _ in results)
    # Timestamps must be timezone-aware; frame_ts is compared across cameras.
    assert all(ts.tzinfo is not None for _, ts in results)


@pytest.mark.asyncio
async def test_reconnects_when_open_fails(monkeypatch):
    source = RTSPSource("rtsp://fake", "cam1")
    attempts = {"n": 0}

    def flaky_open():
        attempts["n"] += 1
        return FakeCapture() if attempts["n"] >= 3 else None

    monkeypatch.setattr(source, "_open", flaky_open)
    monkeypatch.setattr(source, "_read", lambda: frame())

    results = await take(source, 1)
    assert len(results) == 1
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_reopens_after_stream_drops(monkeypatch):
    source = RTSPSource("rtsp://fake", "cam1")
    opens = {"n": 0}
    reads = {"n": 0}

    def counting_open():
        opens["n"] += 1
        return FakeCapture()

    def dropping_read():
        reads["n"] += 1
        # Second read fails, forcing a reconnect before the third succeeds.
        return None if reads["n"] == 2 else frame()

    monkeypatch.setattr(source, "_open", counting_open)
    monkeypatch.setattr(source, "_read", dropping_read)

    results = await take(source, 2)
    assert len(results) == 2
    assert opens["n"] == 2


@pytest.mark.asyncio
async def test_read_timeout_triggers_reconnect(monkeypatch):
    import asyncio

    source = RTSPSource("rtsp://fake", "cam1")
    opens = {"n": 0}
    calls = {"n": 0}

    def counting_open():
        opens["n"] += 1
        return FakeCapture()

    monkeypatch.setattr(source, "_open", counting_open)
    monkeypatch.setattr(source, "_read", lambda: frame())

    real_wait_for = asyncio.wait_for

    async def flaky_wait_for(coro, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            # Close the coroutine we are discarding to avoid a pending-task warning.
            coro.close()
            raise TimeoutError
        return await real_wait_for(coro, timeout)

    monkeypatch.setattr(rtsp.asyncio, "wait_for", flaky_wait_for)

    results = await take(source, 1)
    assert len(results) == 1
    assert opens["n"] == 2
