"""Tests for the ONNX container reader's decode, confidence and snap logic.

A stub session emits logits for a chosen string, so the CTC decode, the
confidence estimate and the checksum/snap resolution are tested exactly, without
a model. The rotation of tall crops is exercised for shape only -- a stub that
ignores its input cannot tell orientations apart.
"""

import numpy as np
import pytest

from apps.inference_worker.container import KnownContainers
from apps.inference_worker.container_ocr import DEFAULT_CHARSET, OnnxContainerOCR

T, C = 40, 37


def logits_for(text: str, peak: float = 8.0) -> np.ndarray:
    """(1, T, C) logits that greedy-decode to `text`, blanks between characters."""
    out = np.zeros((1, T, C), np.float32)
    out[0, :, 0] = peak                       # blank everywhere by default
    t = 1
    for ch in text:
        idx = DEFAULT_CHARSET.index(ch) + 1
        out[0, t, :] = 0.0
        out[0, t, idx] = peak
        t += 2                                # leave a blank between characters
    return out


class StubSession:
    def __init__(self, text):
        self.text = text

    def get_inputs(self):
        class I:
            name = "crop"
        return [I()]

    def run(self, _out, _feed):
        return [logits_for(self.text)]


def make(text, known=None) -> OnnxContainerOCR:
    r = OnnxContainerOCR.__new__(OnnxContainerOCR)
    r.session = StubSession(text)
    r.input_name = "crop"
    r.charset = DEFAULT_CHARSET
    r.known = known
    return r


def crop(h=99, w=606):
    return np.full((h, w, 3), 120, np.uint8)


def test_valid_read_decodes_with_high_confidence():
    result = make("TCLU5437389").read(crop())
    assert result.ok
    assert result.number.canonical == "TCLU5437389"
    assert result.text == "TCLU5437389"
    assert result.confidence > 0.95
    assert result.was_snapped is False
    assert result.is_known is None            # no list loaded


def test_known_flag_reflects_the_list():
    known = KnownContainers(["TCLU5437389"])
    assert make("TCLU5437389", known).read(crop()).is_known is True
    assert make("CMAU7224270", known).read(crop()).is_known is False


def test_misread_is_snapped_to_the_known_number():
    known = KnownContainers(["TCLU5437389"])
    result = make("TCLU5437380", known).read(crop())      # last digit wrong
    assert result.ok and result.was_snapped
    assert result.number.canonical == "TCLU5437389"
    assert result.text == "TCLU5437380"                     # raw read preserved


def test_misread_without_a_list_is_not_ok():
    result = make("TCLU5437380").read(crop())
    assert not result.ok and result.number is None


def test_garbage_is_not_ok():
    assert not make("DESU87369").read(crop()).ok


def test_tall_crop_is_read_via_rotation():
    assert make("TCLU5437389").read(crop(h=606, w=99)).ok


def test_empty_crop_is_safe():
    result = make("TCLU5437389").read(np.zeros((0, 0, 3), np.uint8))
    assert not result.ok and result.confidence == 0.0


def test_missing_model_raises():
    with pytest.raises(RuntimeError, match="not found"):
        OnnxContainerOCR("models/no_such_ocr.onnx")
