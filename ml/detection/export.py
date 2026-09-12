"""Export the trained detection model to ONNX.

    python -m ml.detection.export --weights runs/detect/plate/weights/best.pt

Writes models/plate_detector.onnx - the path config.detector_model_path already
points at - so the inference worker picks it up without a config change.

ONNX is what lets training and serving diverge cleanly: train on whatever GPU is
available, serve on the CPU box with onnxruntime and no torch install. The
parity check exists because that divergence is exactly where accuracy silently
disappears - a mismatched opset or preprocessing step produces a model that
loads, runs, and quietly returns worse boxes.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def parity_check(weights: Path, onnx_path: Path, imgsz: int, seed: int = 0) -> tuple[float, float]:
    """Run both backends on the same input; return (absolute, relative) max difference.

    The comparison is relative because the head emits box coordinates in pixels:
    an absolute gap of 1e-3 is negligible against a 640 px edge but would be
    alarming against a 0-1 confidence. The torch side is fused first, since
    export fuses Conv+BN and the small numeric shift from that is expected
    rather than a sign the graph is wrong.
    """
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(seed)
    batch = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    torch_model = YOLO(str(weights)).model.float().fuse().eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(batch))
    torch_out = (torch_out[0] if isinstance(torch_out, (list, tuple)) else torch_out).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: batch})[0]

    if torch_out.shape != onnx_out.shape:
        raise SystemExit(f"shape mismatch: torch {torch_out.shape} vs onnx {onnx_out.shape}")

    absolute = float(np.abs(torch_out - onnx_out).max())
    scale = float(np.abs(torch_out).max()) or 1.0
    return absolute, absolute / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/plate_detector.onnx"))
    parser.add_argument("--imgsz", type=int, default=640,
                        help="must match the imgsz the inference worker preprocesses to")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="max allowed torch/onnx difference, relative to the output scale")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found - train first")

    model = YOLO(str(args.weights))
    # Export on CPU: MPS export has produced subtly wrong graphs in the past,
    # and export is fast enough that the device makes no practical difference.
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, device="cpu"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), args.out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.skip_parity:
        print("parity check skipped")
    else:
        absolute, relative = parity_check(args.weights, args.out, args.imgsz)
        status = "OK" if relative <= args.tolerance else "FAILED"
        print(f"parity check {status}: max |torch - onnx| = {absolute:.2e} "
              f"({relative:.2e} relative, tolerance {args.tolerance:.0e})")
        if relative > args.tolerance:
            raise SystemExit("export rejected - do not deploy this model")

    print("\nDeploy:")
    print(f"  scp {args.out} root@<server>:/opt/anpr/models/{args.out.name}")
    print("  the server needs onnxruntime only - no torch, no ultralytics")


if __name__ == "__main__":
    main()
