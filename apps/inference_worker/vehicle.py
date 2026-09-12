"""Vehicle type and colour, for the container gate.

Two attributes, two very different methods:

- **Type** (car / truck / bus / motorcycle) comes from a general object detector
  pretrained on COCO, exported to ONNX and run through onnxruntime like the
  plate detector. COCO already contains these classes, so no training is needed
  -- the model is used as-is.

- **Colour** needs no model at all. The painted body of a vehicle is a large,
  roughly uniform region, and naming its dominant colour is a classical-CV
  problem: sample the centre of the box, work in HSV, and split on saturation
  and value before hue. A learned classifier would be more robust to shadow and
  reflection, but this is honest, fast, and has nothing to train or download.

Both are deliberately separate from plate and container recognition: a gate log
that records "white truck, plate 2BJ-4506, container CMAU7224270" is built from
independent parts, and each can be wrong without corrupting the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# COCO class indices for the things that roll through a gate. bicycle is kept
# out: at this camera it is background clutter, not a gate vehicle.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(slots=True)
class Vehicle:
    x1: int
    y1: int
    x2: int
    y2: int
    vehicle_type: str
    type_confidence: float
    colour: str
    colour_confidence: float


# --- colour ----------------------------------------------------------------
#
# KNOWN LIMITATION: on this gate's overhead angle the centre of a tall truck box
# is the windscreen, and blue-tinted glass reads as "blue" on a white cab. The
# dark-pixel exclusion below handles black glass and shadow but not tinted glass,
# which is mid-brightness. Type detection is reliable; colour from this heuristic
# is not, at this camera. The real fix is a small trained colour classifier on
# the vehicle body crop (a public dataset exists) -- this heuristic is the
# no-dependency, no-training baseline, not the finished answer.

# Named colours by HSV. Order matters: achromatic tests (value/saturation) run
# before hue, because a dark or washed-out region has no meaningful hue.
_HUE_NAMES = [
    (0, 10, "red"), (10, 20, "orange"), (20, 33, "yellow"),
    (33, 85, "green"), (85, 100, "cyan"), (100, 130, "blue"),
    (130, 160, "purple"), (160, 180, "red"),
]


def classify_colour(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, float]:
    """Name the dominant colour of a vehicle box. Returns (name, confidence).

    Samples the central half of the box only. Vehicle edges catch background,
    windscreen and shadow; the middle of a bonnet or a container wall is the
    body colour. Confidence is the fraction of sampled pixels that agree with
    the winning colour, so a two-tone or reflective surface reports low.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return "unknown", 0.0

    # Central half of the box.
    cx1, cy1 = x1 + w // 4, y1 + h // 4
    cx2, cy2 = x2 - w // 4, y2 - h // 4
    patch = image[cy1:cy2, cx1:cx2]
    if patch.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    # Dark pixels are glass, tyres and deep shadow -- on this overhead angle the
    # centre of a tall truck box is mostly windscreen. They carry no body-colour
    # information, so they are excluded from the vote UNLESS the whole region is
    # dark, which is a genuinely black vehicle.
    dark = val < 60
    if dark.mean() > 0.6:
        return "black", round(float(dark.mean()), 3)
    keep = ~dark
    hue, sat, val = hue[keep], sat[keep], val[keep]
    if len(hue) == 0:
        return "black", 1.0

    names = np.empty(hue.shape, dtype=object)
    white = (val > 180) & (sat < 45)
    grey = (sat < 45) & ~white
    names[white] = "white"
    names[grey] = "grey"

    chromatic = ~(white | grey)
    for lo, hi, name in _HUE_NAMES:
        mask = chromatic & (hue >= lo) & (hue < hi)
        names[mask] = name

    values, counts = np.unique(names.astype(str), return_counts=True)
    winner = values[counts.argmax()]
    confidence = float(counts.max() / len(names))
    return winner, round(confidence, 3)


# --- type ------------------------------------------------------------------


class OnnxVehicleDetector:
    """Multi-class YOLO (COCO) via onnxruntime, filtered to gate vehicles.

    Mirrors OnnxPlateDetector's letterbox preprocessing exactly -- the two share
    the invariant that a box comes back in original-frame pixels -- but keeps the
    winning class index instead of collapsing to one class.
    """

    def __init__(self, model_path: str | Path, conf: float = 0.35,
                 iou: float = 0.5, providers: list[str] | None = None) -> None:
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"vehicle model not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed") from exc

        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.conf = conf
        self.iou = iou

    def detect(self, image: np.ndarray) -> list[Vehicle]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h0, w0 = image.shape[:2]
        if h0 == 0 or w0 == 0:
            return []

        scale = min(self.imgsz / w0, self.imgsz / h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        pad_x, pad_y = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        preds = np.squeeze(raw, 0).T                      # (anchors, 4+classes)
        scores = preds[:, 4:]
        class_ids = scores.argmax(1)
        confs = scores.max(1)

        keep = confs >= self.conf
        # Only the COCO vehicle classes; everything else on the road is noise.
        keep &= np.isin(class_ids, list(VEHICLE_CLASSES))
        if not keep.any():
            return []

        boxes, confs, class_ids = preds[keep, :4], confs[keep], class_ids[keep]
        cx, cy, bw, bh = boxes.T
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, w0 - 1)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, h0 - 1)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, w0 - 1)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, h0 - 1)

        nms = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist(), confs.tolist(), self.conf, self.iou
        )
        if len(nms) == 0:
            return []

        out: list[Vehicle] = []
        for i in np.array(nms).flatten():
            box = (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
            colour, colour_conf = classify_colour(image, box)
            out.append(Vehicle(
                *box,
                vehicle_type=VEHICLE_CLASSES[int(class_ids[i])],
                type_confidence=round(float(confs[i]), 4),
                colour=colour,
                colour_confidence=colour_conf,
            ))
        out.sort(key=lambda v: (v.x2 - v.x1) * (v.y2 - v.y1), reverse=True)
        return out
