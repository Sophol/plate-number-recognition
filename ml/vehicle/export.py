"""Export the trained vehicle-colour model to ONNX.

    python -m ml.vehicle.export --weights runs/vehicle_colour/best.pt

Writes models/vehicle_colour.onnx and models/vehicle_colour_classes.json beside
it. The classes file is not optional: the ONNX graph emits a bare index, and
without the list that maps index to colour name the server guesses an order and
relabels every read with a plausible neighbour instead of failing.

The parity check compares torch and ONNX on the same input, because an export
that silently disagrees still returns a confident colour.
"""

import argparse
from pathlib import Path

import numpy as np

from ml.vehicle.model import CROP_SIZE, build_model, load_classes, save_classes


def parity_check(model, onnx_path: Path, num_classes: int, seed: int = 0) -> tuple[float, bool]:
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(seed)
    batch = rng.random((2, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
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
    parser.add_argument("--out", type=Path, default=Path("models/vehicle_colour.onnx"))
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

    state = torch.load(args.weights, map_location="cpu")
    # A model trained with dropout has a Sequential head (fc.1.weight); one
    # without has a plain Linear (fc.weight). Rebuild whichever the checkpoint
    # used so load_state_dict matches -- the dropout rate itself is irrelevant at
    # export time (eval mode makes Dropout an identity), only the structure is.
    had_dropout = "fc.1.weight" in state
    model = build_model(len(classes), pretrained=False, dropout=0.5 if had_dropout else 0.0)
    model.load_state_dict(state)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["crop"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}},
        # Keep weights inside the .onnx; a sidecar .data file that does not get
        # copied yields a model that loads and predicts from nothing.
        external_data=False,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    classes_out = args.out.with_name("vehicle_colour_classes.json")
    save_classes(classes_out, classes)
    print(f"wrote {classes_out} ({len(classes)} classes)")

    difference, same_class = parity_check(model, args.out, len(classes))
    status = "OK" if difference <= args.tolerance and same_class else "FAILED"
    print(f"parity check {status}: max |torch - onnx| = {difference:.2e}, same argmax = {same_class}")
    if not same_class:
        raise SystemExit("export rejected - torch and ONNX disagree on the predicted class")

    print("\nDeploy:")
    print(f"  scp {args.out} {classes_out} root@<server>:/opt/anpr/models/")


if __name__ == "__main__":
    main()
