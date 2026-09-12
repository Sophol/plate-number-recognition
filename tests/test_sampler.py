import pytest

from apps.camera_worker.sampler import FPSSampler


def test_first_frame_always_emits():
    assert FPSSampler(target_fps=10).should_emit(now=100.0)


def test_thins_to_target_rate():
    sampler = FPSSampler(target_fps=10)  # one frame per 0.1s
    assert sampler.should_emit(100.00)
    assert not sampler.should_emit(100.05)
    assert sampler.should_emit(100.10)
    assert not sampler.should_emit(100.15)
    assert sampler.should_emit(100.20)


def test_thirty_fps_source_down_to_ten():
    sampler = FPSSampler(target_fps=10)
    emitted = sum(sampler.should_emit(i / 30.0) for i in range(30))
    assert emitted == 10


def test_reset_restarts_the_interval():
    sampler = FPSSampler(target_fps=10)
    sampler.should_emit(100.0)
    assert not sampler.should_emit(100.05)
    sampler.reset()
    assert sampler.should_emit(100.05)


def test_rejects_nonpositive_fps():
    with pytest.raises(ValueError):
        FPSSampler(target_fps=0)
