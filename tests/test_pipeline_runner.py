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


@pytest.mark.asyncio
async def test_runner_exits_quietly_when_no_cameras_are_registered(monkeypatch):
    """An empty camera table is configuration, not failure - it must not crash-loop."""
    from apps.pipeline_runner import main as runner

    async def no_cameras():
        return []

    monkeypatch.setattr(runner, "load_active_cameras", no_cameras)
    await runner.run()  # returns rather than raising


@pytest.mark.asyncio
async def test_capture_is_cancelled_when_inference_dies(monkeypatch):
    """Capture filling a queue nobody drains is the exact failure to avoid."""
    from apps.pipeline_runner import main as runner

    class FakeCamera:
        id = "cam-1"
        name = "gate"
        rtsp_url = "rtsp://example/stream"

    async def one_camera():
        return [FakeCamera()]

    capture_cancelled = asyncio.Event()

    async def fake_run_camera(*args, **kwargs):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            capture_cancelled.set()
            raise

    async def dying_consume(queue, pipeline):
        raise RuntimeError("inference died")

    monkeypatch.setattr(runner, "load_active_cameras", one_camera)
    monkeypatch.setattr(runner, "run_camera", fake_run_camera)
    monkeypatch.setattr(runner, "consume", dying_consume)
    monkeypatch.setattr(runner, "build_pipeline", lambda *a, **k: object())

    await runner.run()

    assert capture_cancelled.is_set()
