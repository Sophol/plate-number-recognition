"""Tests for the combined capture + inference runner.

The bug this runner exists to prevent is silent: two processes, two queues, no
frames, no error. So the test that matters is not "does it start" but "does a
frame put in by capture come out at inference" — anything less would have passed
against the broken two-service arrangement too.
"""

import asyncio
from datetime import UTC, datetime

import numpy as np
import pytest

from apps.camera_worker.sampler import BoundedFrameQueue, Frame
from apps.inference_worker.main import consume

T0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)


class RecordingPipeline:
    """Stands in for InferencePipeline; records the frames it was asked to process."""

    def __init__(self, fail_on: str | None = None, delay: float = 0.0):
        self.seen: list[str] = []
        self.fail_on = fail_on
        self.delay = delay

    def process(self, frame: Frame):
        if self.delay:
            # A blocking sleep, deliberately: this is what a real CPU-bound
            # detection does to the thread it runs on.
            import time

            time.sleep(self.delay)
        if frame.camera_id == self.fail_on:
            raise RuntimeError("inference exploded")
        self.seen.append(frame.camera_id)
        return []


def frame(camera_id: str = "cam-1") -> Frame:
    return Frame(camera_id=camera_id, image=np.zeros((64, 64, 3), np.uint8), frame_ts=T0)


@pytest.mark.asyncio
async def test_a_frame_put_on_the_queue_reaches_inference():
    """The whole point of the runner: one queue, both halves."""
    queue = BoundedFrameQueue(8)
    pipeline = RecordingPipeline()
    task = asyncio.create_task(consume(queue, pipeline))

    await queue.put(frame("cam-1"))
    await queue.put(frame("cam-2"))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(pipeline.seen) == 2:
            break

    task.cancel()
    assert pipeline.seen == ["cam-1", "cam-2"]


@pytest.mark.asyncio
async def test_inference_does_not_block_the_event_loop():
    """Capture must keep running while a slow detection is in flight.

    If process() ran inline, the loop would stall for its whole duration and the
    counter below could not advance past 1.
    """
    queue = BoundedFrameQueue(8)
    pipeline = RecordingPipeline(delay=0.15)
    inference = asyncio.create_task(consume(queue, pipeline))
    await queue.put(frame())

    ticks = 0
    for _ in range(30):
        await asyncio.sleep(0.01)
        ticks += 1
        if pipeline.seen:
            break

    inference.cancel()
    assert pipeline.seen == ["cam-1"]
    assert ticks > 5, "event loop was blocked during inference"


@pytest.mark.asyncio
async def test_one_bad_frame_does_not_kill_the_consumer():
    """A single failed detection must not end the run; the next frame still lands."""
    queue = BoundedFrameQueue(8)
    pipeline = RecordingPipeline(fail_on="bad")
    task = asyncio.create_task(consume(queue, pipeline))

    await queue.put(frame("bad"))
    await queue.put(frame("good"))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if pipeline.seen:
            break

    still_running = not task.done()
    task.cancel()
    assert pipeline.seen == ["good"]
    assert still_running


class FakeCamera:
    def __init__(self, id: str, name: str = "gate", rtsp_url: str = "rtsp://example/stream"):
        self.id, self.name, self.rtsp_url = id, name, rtsp_url


class FakeCaptures:
    """Records which camera tasks the runner starts and stops, in place of run_camera."""

    def __init__(self):
        self.started: list[tuple[str, str]] = []
        self.cancelled: list[str] = []

    async def run_camera(self, camera_id, rtsp_url, *args, **kwargs):
        self.started.append((camera_id, rtsp_url))
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.append(camera_id)
            raise


async def never_ending_consume(queue, pipeline):
    await asyncio.sleep(3600)


def install(monkeypatch, runner, cameras_by_tick: list[list], captures: FakeCaptures,
            consume=never_ending_consume):
    """Each call to load_cameras returns the next entry; the last one repeats."""
    ticks = {"n": 0}

    async def load():
        i = min(ticks["n"], len(cameras_by_tick) - 1)
        ticks["n"] += 1
        return cameras_by_tick[i]

    monkeypatch.setattr(runner, "run_camera", captures.run_camera)
    monkeypatch.setattr(runner, "consume", consume)
    monkeypatch.setattr(runner, "build_pipeline", lambda *a, **k: object())
    return load


async def run_for(runner, load, ticks: int, reload_seconds: float = 0.02):
    task = asyncio.create_task(runner.run(load_cameras=load, reload_seconds=reload_seconds))
    await asyncio.sleep(reload_seconds * ticks + 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_runner_waits_when_no_cameras_are_registered_and_starts_one_when_it_appears(monkeypatch):
    """An empty camera table is a waiting state: enabling a camera on the web
    must start capture without a service restart."""
    from apps.pipeline_runner import main as runner

    captures = FakeCaptures()
    load = install(monkeypatch, runner, [[], [], [FakeCamera("cam-1")]], captures)

    await run_for(runner, load, ticks=4)

    assert captures.started == [("cam-1", "rtsp://example/stream")]


@pytest.mark.asyncio
async def test_disabling_a_camera_on_the_web_stops_its_capture(monkeypatch):
    from apps.pipeline_runner import main as runner

    captures = FakeCaptures()
    load = install(monkeypatch, runner, [[FakeCamera("cam-1"), FakeCamera("cam-2")],
                                         [FakeCamera("cam-2")]], captures)

    await run_for(runner, load, ticks=3)

    assert [c for c, _ in captures.started] == ["cam-1", "cam-2"]
    assert captures.cancelled[0] == "cam-1"
    # cam-2 only stops because the test cancels the runner at the end.
    assert captures.cancelled.count("cam-2") == 1


@pytest.mark.asyncio
async def test_changing_a_camera_url_reconnects_it(monkeypatch):
    from apps.pipeline_runner import main as runner

    captures = FakeCaptures()
    load = install(monkeypatch, runner, [[FakeCamera("cam-1", rtsp_url="rtsp://old/s")],
                                         [FakeCamera("cam-1", rtsp_url="rtsp://new/s")]], captures)

    await run_for(runner, load, ticks=3)

    assert captures.started == [("cam-1", "rtsp://old/s"), ("cam-1", "rtsp://new/s")]


@pytest.mark.asyncio
async def test_a_database_blip_does_not_stop_running_captures(monkeypatch):
    from apps.pipeline_runner import main as runner

    captures = FakeCaptures()
    calls = {"n": 0}

    async def flaky_load():
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("db went away")
        return [FakeCamera("cam-1")]

    install(monkeypatch, runner, [[]], captures)
    await run_for(runner, flaky_load, ticks=4)

    assert calls["n"] >= 3
    assert captures.started == [("cam-1", "rtsp://example/stream")]
    assert captures.cancelled == ["cam-1"]  # only the final shutdown


@pytest.mark.asyncio
async def test_capture_is_cancelled_when_inference_dies(monkeypatch):
    """Capture filling a queue nobody drains is the exact failure to avoid."""
    from apps.pipeline_runner import main as runner

    captures = FakeCaptures()

    async def dying_consume(queue, pipeline):
        await asyncio.sleep(0.01)
        raise RuntimeError("inference died")

    load = install(monkeypatch, runner, [[FakeCamera("cam-1")]], captures, consume=dying_consume)

    await asyncio.wait_for(runner.run(load_cameras=load, reload_seconds=5.0), timeout=2.0)

    assert captures.cancelled == ["cam-1"]
