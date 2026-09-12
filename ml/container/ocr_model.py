"""CRNN + CTC recognition model for container numbers, shared by train and serve.

A container number is a short fixed-alphabet string, so the standard scene-text
recogniser fits: a small CNN reduces the crop to a horizontal feature sequence,
a BiLSTM reads it left to right, and CTC aligns the output to the label without
needing per-character boxes. It is deliberately small -- the alphabet is 36
glyphs and the strings are 11 long -- so it trains in minutes on synthetic data
and exports to a light ONNX the server runs on CPU.

Charset and decode live here so training, evaluation, export and inference all
agree on what index 0 means. Index 0 is the CTC blank.
"""

from __future__ import annotations

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK = 0
# 1-based indices; 0 reserved for CTC blank.
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

IMG_HEIGHT = 32
IMG_WIDTH = 160


def encode(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def greedy_decode(sequence: list[int]) -> str:
    """Collapse a CTC index sequence: drop repeats, then drop blanks."""
    out, prev = [], BLANK
    for idx in sequence:
        if idx != prev and idx != BLANK:
            out.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(out)


def build_model():
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES):
            super().__init__()
            def block(i, o, pool):
                layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)]
                if pool:
                    layers.append(nn.MaxPool2d(*pool))
                return layers

            self.cnn = nn.Sequential(
                *block(1, 64, ((2, 2), (2, 2))),     # 32x160 -> 16x80
                *block(64, 128, ((2, 2), (2, 2))),   # -> 8x40
                *block(128, 256, None),
                *block(256, 256, ((2, 1), (2, 1))),  # -> 4x40 (keep width)
                *block(256, 512, None),
                *block(512, 512, ((2, 1), (2, 1))),  # -> 2x40
                *block(512, 512, ((2, 1), (2, 1))),  # -> 1x40
            )
            self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            f = self.cnn(x)               # (B, 512, 1, W')
            f = f.squeeze(2).permute(0, 2, 1)   # (B, W', 512)
            r, _ = self.rnn(f)
            return self.fc(r)             # (B, W', num_classes)

    return CRNN()
