"""Export the trained province model to ONNX.

    python -m ml.province.export --weights runs/province/best.pt

Writes models/province_classifier.onnx - the path config.province_model_path
already points at - and models/province_classes.json beside it.

The classes file is not optional. The ONNX graph emits a bare index; without the
list that maps index to province code the server has to guess an ordering, and a
wrong guess relabels every read with a plausible-looking neighbour rather than
failing.

The parity check compares torch and ONNX on the same input, because an export
that silently disagrees still returns a confident province.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from ml.province.model import CROP_HEIGHT, CROP_WIDTH, build_model, load_classes, save_classes


def parity_check(model, onnx_path: Path, num_classes: int, seed: int = 0) -> tuple[float, bool]:
    """Return (max absolute logit difference, whether both pick the same class)."""
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(seed)
    batch = rng.random((2, 3, CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: batch})[0]

    if onnx_out.shape != (2, num_classes):
        raise SystemExit(f"unexpected ONNX output shape {onnx_out.shape}")
    same_class = bool((torch_out.argmax(axis=1) == onnx_out.argmax(axis=1)).all())
    return float(np.abs(torch_out - onnx_out).max()), same_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=None,
                        help="default: classes.json beside the weights")
    parser.add_argument("--out", type=Path, default=Path("models/province_classifier.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    import torch

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found - train first")

    classes_path = args.classes or args.weights.parent / "classes.json"
    if not classes_path.exists():
        raise SystemExit(f"{classes_path} not found - the export needs the class order")
    classes = load_classes(classes_path)

    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["crop"],
        output_names=["logits"],
        opset_version=args.opset,
        # A worker batching several plates from one frame should not need a
        # second export.
        dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}},
        # Keep the weights inside the .onnx. The dynamo exporter otherwise
        # writes them to a sidecar .data file, and copying only the .onnx to
        # the server yields a model that loads and predicts from nothing.
        external_data=False,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    classes_out = args.out.with_name("province_classes.json")
    save_classes(classes_out, classes)
    print(f"wrote {classes_out} ({len(classes)} classes)")

    difference, same_class = parity_check(model, args.out, len(classes))
    scale = 1.0
    status = "OK" if difference <= args.tolerance * max(scale, 1.0) and same_class else "FAILED"
    print(f"parity check {status}: max |torch - onnx| = {difference:.2e}, "
          f"same argmax = {same_class}")
    if not same_class:
        raise SystemExit("export rejected - torch and ONNX disagree on the predicted class")

    print("\nDeploy:")
    print(f"  scp {args.out} {classes_out} root@<server>:/opt/anpr/models/")


if __name__ == "__main__":
    main()
