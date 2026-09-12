import numpy as np

from apps.camera_worker.motion import ROI, MotionGate


def blank(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_first_frame_reports_no_motion():
    gate = MotionGate()
    assert not gate.has_motion(blank())


def test_identical_frames_report_no_motion():
    gate = MotionGate()
    frame = blank()
    gate.has_motion(frame)
    assert not gate.has_motion(frame.copy())


def test_large_change_reports_motion():
    gate = MotionGate()
    gate.has_motion(blank())

    moved = blank()
    moved[60:180, 80:240] = 255
    assert gate.has_motion(moved)


def test_tiny_change_is_below_threshold():
    gate = MotionGate(min_changed_fraction=0.05)
    gate.has_motion(blank())

    speck = blank()
    speck[0:2, 0:2] = 255
    assert not gate.has_motion(speck)


def test_roi_ignores_motion_outside_it():
    # Watch the left half only; motion on the right must not trigger.
    gate = MotionGate(ROI(x1=0.0, y1=0.0, x2=0.5, y2=1.0))
    gate.has_motion(blank())

    right_side = blank()
    right_side[:, 240:] = 255
    assert not gate.has_motion(right_side)


def test_roi_detects_motion_inside_it():
    gate = MotionGate(ROI(x1=0.0, y1=0.0, x2=0.5, y2=1.0))
    gate.has_motion(blank())

    left_side = blank()
    left_side[:, :160] = 255
    assert gate.has_motion(left_side)


def test_reset_clears_reference_frame():
    gate = MotionGate()
    moving = blank()
    moving[60:180, 80:240] = 255
    gate.has_motion(blank())
    gate.reset()
    # With no reference frame, the next frame is baseline again.
    assert not gate.has_motion(moving)
