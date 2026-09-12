"""Train the detection model. Phase 1 deliverable.

    python -m ml.detection.train --data dataset/v1/yolo/data.yaml

Defaults target a gate camera, not a benchmark: yolo11n at 640 px, which is
what the CPU inference worker can actually keep up with. Train a larger model
only after measuring that the small one is the accuracy bottleneck.

Device is picked automatically - MPS on Apple silicon, CUDA on a GPU box, CPU
otherwise. The trained .pt is not what production loads; run ml.detection.export
afterwards to produce the ONNX the inference worker reads.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.device import describe, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml from ml.datasets.split")
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="lower this on a 16 GB Mac - MPS shares system memory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="plate")
    parser.add_argument("--patience", type=int, default=25, help="early-stop after N epochs without gain")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found - run ml.datasets.split first")

    device = pick_device(args.device)
    print(f"training on {describe(device)}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        # Plates are rigid, always upright, and never mirrored: a flipped plate
        # is not a plate. Vertical flip and mosaic hurt more than they help.
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        # Gate cameras see the same scene in daylight, dusk and IR floodlight,
        # so exposure variation is the augmentation that actually pays off.
        hsv_v=0.5,
        hsv_s=0.5,
    )

    weights = args.project / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {weights}")
    print(f"next: python -m ml.detection.evaluate --weights {weights} --data {args.data}")


if __name__ == "__main__":
    main()
