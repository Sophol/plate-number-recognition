"""Shared definition of the province classifier: input size, model, class order.

Training, evaluation, export and inference all import from here. The class order
is the reason this module exists: the ONNX model emits a bare index, and the
only thing that turns index 7 back into a province code is this list. Derive it
independently in two places and they drift the moment a province is added,
silently relabelling every read.

`classes.json` is written next to the exported model for the same reason - the
server must not depend on importing this file to know what index 7 means.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# The top zone of a 320x160 rectified plate is 320x48. Training on that native
# shape avoids a resize that training would do and inference would not.
CROP_WIDTH = 192
CROP_HEIGHT = 48


def class_names() -> list[str]:
    """Province codes in a stable, sorted order. Numeric, so "2" precedes "10"."""
    from apps.inference_worker.province import PROVINCE_NAMES

    return sorted(PROVINCE_NAMES, key=int)


def preprocess(crop: np.ndarray) -> np.ndarray:
    """Top-zone image -> CHW float32 in 0-1. The single source of truth.

    Khmer province text is high-contrast dark-on-light; colour carries nothing,
    so the crop is greyscaled and repeated to three channels to fit a standard
    ImageNet backbone.
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return np.repeat(normalised[None], 3, axis=0)


def build_model(num_classes: int, pretrained: bool = True):
    """ResNet-18 with a resized head.

    ImageNet weights are worth having even for Khmer script: the early layers
    learn strokes and edges, which is most of what separates one province word
    from another. Training those from scratch needs far more plates than a
    Phase 0 dataset will hold.
    """
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_classes(path: Path, classes: list[str]) -> None:
    path.write_text(json.dumps(classes, indent=1))


def load_classes(path: Path) -> list[str]:
    return json.loads(Path(path).read_text())
