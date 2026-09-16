"""Shared definition of the vehicle-colour classifier: input size, model, and
class order.

Training, export and inference all import from here, and the class order is the
reason the module exists: the ONNX model emits a bare index, and only this list
turns index 5 back into a colour name. Derive it independently in two places and
they drift the moment a colour is added, silently relabelling every read.

`vehicle_colour_classes.json` is written next to the exported model for the same
reason -- the server must not depend on importing this file to know what an
index means.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# A body crop needs little resolution to name its colour; 64x64 keeps CPU
# inference on the server cheap while leaving enough pixels for shadow and
# highlight to average out.
CROP_SIZE = 64


def class_names() -> list[str]:
    """The colour classes, in the one order shared with inference."""
    from apps.inference_worker.vehicle import COLOUR_CLASSES

    return list(COLOUR_CLASSES)


def preprocess(crop: np.ndarray) -> np.ndarray:
    """Body crop (BGR) -> CHW float32 RGB in 0-1. The single source of truth.

    Unlike the province classifier, colour is the whole signal, so the crop
    keeps all three channels rather than being greyscaled.
    """
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


def build_model(num_classes: int, pretrained: bool = True):
    """ResNet-18 with a resized head, matching the province classifier.

    ImageNet weights help even here: the early layers learn the smooth-region
    and highlight cues that separate a shadowed red panel from a shadowed blue
    one, which is most of what a body-colour call turns on.
    """
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_classes(path: Path, classes: list[str]) -> None:
    Path(path).write_text(json.dumps(classes, indent=1))


def load_classes(path: Path) -> list[str]:
    return json.loads(Path(path).read_text())
