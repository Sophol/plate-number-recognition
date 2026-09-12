"""Locates plates with a trained YOLO detector exported to ONNX.

This is the production locator the plan calls for. It runs through onnxruntime
rather than torch, which is what lets the inference worker stay on a CPU box
with no CUDA, no torch and no ultralytics install -- see ml/detection/export.py
for the other half of that split.

Preprocessing must match training exactly or accuracy drops without any error:
letterbox to a square at the model's own input size, BGR to RGB, scale to 0-1.
The letterbox padding is undone on the way out so boxes come back in original
frame pixels.

`corners` is always None: a detection model emits axis-aligned boxes, so the
caller falls back to the plain crop. Perspective correction needs the
segmentation variant the plan mentions, which returns a mask to fit corners to.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

from apps.inference_worker.interfaces import Detection

log = structlog.get_logger()

DEFAULT_IMGSZ = 640
# 0.25 is YOLO's conventional default and is too low for a gate. A busy
# barrier scene is full of plate-shaped rectangles - hazard stripes, container
# corrugation, booth signage - and on the first live run a 0.29-confidence box on
# nothing in particular was voted through as a read. Raising the floor costs a
# little recall on hard plates; the tracker sees each vehicle across many frames,
# so one missed frame is cheap and a phantom read that reaches the gate is not.
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45
MAX_DETECTIONS = 10


class OnnxPlateDetector:
    """Runs a YOLO .onnx export and reports plates as Detections."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        max_detections: int = MAX_DETECTIONS,
        providers: list[str] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(f"detector model not found: {path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError("onnxruntime is not installed") from exc

        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name

        # A static export fixes the input size; take it from the model so a
        # 960 px export is not silently fed 640 px letterboxes.
        shape = model_input.shape
        height, width = shape[2], shape[3]
        self.imgsz = height if isinstance(height, int) else DEFAULT_IMGSZ
        if isinstance(width, int) and isinstance(height, int) and width != height:
            raise RuntimeError(f"non-square model input {shape} is not supported")

        self.conf = conf
        self.iou = iou
        self.max_detections = max_detections
        log.info("onnx_detector_loaded", model=str(path), imgsz=self.imgsz)

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        original_h, original_w = image.shape[:2]
        if original_h == 0 or original_w == 0:
            return []

        batch, scale, pad_x, pad_y = self._preprocess(image)
        raw = self.session.run(None, {self.input_name: batch})[0]
        return self._postprocess(raw, scale, pad_x, pad_y, original_w, original_h)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Letterbox into a square, preserving aspect ratio, padded with grey."""
        h, w = image.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        pad_x = (self.imgsz - new_w) / 2
        pad_y = (self.imgsz - new_h) / 2
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        canvas[top : top + new_h, left : left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        batch = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(batch), scale, float(left), float(top)

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        original_w: int,
        original_h: int,
    ) -> list[Detection]:
        # YOLOv8/11 export shape is (1, 4 + classes, anchors); transpose so each
        # row is one candidate box.
        predictions = np.squeeze(raw, axis=0).T
        if predictions.shape[1] < 5:
            return []

        class_scores = predictions[:, 4:]
        confidences = class_scores.max(axis=1)
        keep = confidences >= self.conf
        if not keep.any():
            return []

        boxes_xywh = predictions[keep, :4]
        confidences = confidences[keep]

        # Centre/size in letterbox pixels -> corners in original frame pixels.
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, original_w - 1)
        y1 = np.clip(y1, 0, original_h - 1)
        x2 = np.clip(x2, 0, original_w - 1)
        y2 = np.clip(y2, 0, original_h - 1)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self.conf, self.iou)
        if len(indices) == 0:
            return []

        detections = [
            Detection(
                x1=int(round(x1[i])),
                y1=int(round(y1[i])),
                x2=int(round(x2[i])),
                y2=int(round(y2[i])),
                confidence=round(float(confidences[i]), 4),
                plate_type=None,
                corners=None,
            )
            for i in np.array(indices).flatten()
        ]
        # Zero-width boxes survive clipping when a detection sits on the frame
        # edge; the downstream crop would be empty.
        detections = [d for d in detections if d.x2 > d.x1 and d.y2 > d.y1]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]
