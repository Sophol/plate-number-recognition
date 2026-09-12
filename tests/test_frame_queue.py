from datetime import UTC, datetime

import numpy as np
import pytest

from apps.camera_worker.sampler import BoundedFrameQueue, Frame


def frame(n: int) -> Frame:
    return Frame(camera_id="cam1", image=np.zeros((2, 2)), frame_ts=datetime.now(UTC))


@pytest.mark.asyncio
async def test_drops_oldest_when_full():
    q = BoundedFrameQueue(maxsize=2)
    a, b, c = frame(1), frame(2), frame(3)
    await q.put(a)
    await q.put(b)
    await q.put(c)

    assert q.dropped == 1
    assert q.qsize() == 2
    assert await q.get() is b
    assert await q.get() is c


@pytest.mark.asyncio
async def test_put_never_blocks_under_pressure():
    q = BoundedFrameQueue(maxsize=1)
    for _ in range(50):
        await q.put(frame(0))
    assert q.qsize() == 1
    assert q.dropped == 49
