"""Export the trained container OCR to ONNX.

    python -m ml.container.export_ocr --weights runs/container_ocr/best.pt

Writes models/container_ocr.onnx and models/container_ocr_charset.json beside
it. The charset file is not optional: the model emits per-timestep class
indices, and index 7 means "G" only because of that list. A server that guesses
the order reads every container wrong in a way that still looks like a number.

The parity check runs torch and ONNX on the same random input and requires the
decoded strings to agree, not just the logits to be close -- a small numeric
drift that flips one argmax changes a character, and that is the failure that
matters.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.container.ocr_model import (
    CHARSET, IMG_HEIGHT, IMG_WIDTH, build_model, greedy_decode,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/container_ocr.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import onnxruntime as ort
    import torch

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found - train first")

    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
    # The TorchScript exporter: it handles LSTM reliably and writes one file.
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["crop"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    charset_out = args.out.with_name("container_ocr_charset.json")
    charset_out.write_text(json.dumps({"blank": 0, "charset": CHARSET}, indent=1))
    print(f"wrote {charset_out}")

    rng = np.random.default_rng(0)
    batch = rng.random((4, 1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(batch)).numpy()
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"crop": batch})[0]

    diff = float(np.abs(torch_logits - onnx_logits).max())
    torch_text = [greedy_decode(r.tolist()) for r in torch_logits.argmax(2)]
    onnx_text = [greedy_decode(r.tolist()) for r in onnx_logits.argmax(2)]
    same = torch_text == onnx_text
    print(f"parity check {'OK' if same else 'FAILED'}: max |torch - onnx| = {diff:.2e}, "
          f"decoded strings agree = {same}")
    if not same:
        raise SystemExit("export rejected - torch and ONNX decode differently")

    print("\nDeploy:")
    print(f"  scp {args.out} {charset_out} root@<server>:/opt/anpr/models/")


if __name__ == "__main__":
    main()
