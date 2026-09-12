"""Evaluate the detection model against the held-out test set.

    python -m ml.detection.evaluate --weights runs/detect/plate/weights/best.pt \
        --data dataset/v1/yolo/data.yaml

Reports the detection metrics the plan asks for: precision, recall, mAP50 and
mAP50-95. Detection mAP is a component metric, not the production KPI - the
plan's primary KPI is full-plate exact-match accuracy, which only the end-to-end
pipeline can measure. A detector that scores well here can still feed OCR a crop
it cannot read.

Defaults to the test split, not val: val steered training and early stopping, so
its numbers are optimistic.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.device import describe, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found - train first")

    device = pick_device(args.device)
    print(f"evaluating {args.weights} on the {args.split} split using {describe(device)}\n")

    model = YOLO(str(args.weights))
    results = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
    )

    box = results.box
    print("\nDetection metrics (plan: IMPORTANT METRICS)")
    print(f"  precision   {box.mp:.4f}")
    print(f"  recall      {box.mr:.4f}")
    print(f"  mAP50       {box.map50:.4f}")
    print(f"  mAP50-95    {box.map:.4f}")

    names = results.names
    if len(names) > 1:
        print("\nPer class")
        for i, class_index in enumerate(results.ap_class_index):
            print(f"  {names[class_index]:<14} mAP50 {box.ap50[i]:.4f}  mAP50-95 {box.ap[i]:.4f}")

    print("\nRecall is the number to watch: a plate the detector misses is a read")
    print("the gate never gets, and no amount of OCR accuracy recovers it.")


if __name__ == "__main__":
    main()
