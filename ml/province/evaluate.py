"""Evaluate the province model against the held-out test set.

    python -m ml.province.evaluate --weights runs/province/best.pt --crops dataset/v1/province

Reports the two metrics the plan names for this model: overall accuracy and a
confusion matrix. The matrix is the one that matters. Province classes are
badly imbalanced in any real plate population, so a single accuracy figure hides
the failure that actually costs a gate - two provinces whose Khmer words look
alike being swapped for each other.

Uses the same frame-based split and seed as training, so "held out" means the
same crops it meant during training.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ml.device import describe, pick_device
from ml.province.model import build_model, load_classes, preprocess
from ml.province.train import load_crops, split_by_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=None,
                        help="default: classes.json beside the weights")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    classes = load_classes(args.classes or args.weights.parent / "classes.json")
    samples = load_crops(args.crops, classes)
    _, val_samples = split_by_frame(samples, args.val_fraction, args.seed)
    if not val_samples:
        raise SystemExit("no held-out crops - was the model trained on this dataset?")

    device_name = pick_device(args.device)
    device = torch.device(device_name if device_name != "0" else "cuda:0")
    print(f"evaluating on {describe(device_name)}: {len(val_samples)} held-out crops\n")

    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()

    truth, predicted = [], []
    with torch.no_grad():
        for start in range(0, len(val_samples), args.batch):
            batch = val_samples[start : start + args.batch]
            images = torch.from_numpy(np.stack([preprocess(s[0]) for s in batch]))
            outputs = model(images.to(device)).argmax(dim=1).cpu().numpy()
            predicted += outputs.tolist()
            truth += [s[1] for s in batch]

    truth_array, predicted_array = np.array(truth), np.array(predicted)
    accuracy = float((truth_array == predicted_array).mean())
    print(f"Province classification accuracy: {accuracy:.4f}\n")

    present = sorted(set(truth) | set(predicted))
    matrix = np.zeros((len(present), len(present)), dtype=int)
    position = {label: i for i, label in enumerate(present)}
    for t, p in zip(truth, predicted):
        matrix[position[t], position[p]] += 1

    header = "".join(f"{classes[label]:>5}" for label in present)
    print("Confusion matrix (rows = true, columns = predicted)")
    print(f"{'':>6}{header}")
    for label in present:
        row = matrix[position[label]]
        print(f"{classes[label]:>5} " + "".join(f"{v:>5}" for v in row))

    print("\nPer class")
    support = Counter(truth)
    for label in present:
        i = position[label]
        tp = matrix[i, i]
        recall = tp / support[label] if support[label] else 0.0
        column = matrix[:, i].sum()
        precision = tp / column if column else 0.0
        print(f"  {classes[label]:>3}: precision {precision:.3f}  recall {recall:.3f}  "
              f"support {support[label]}")

    worst = [(matrix[position[t], position[p]], classes[t], classes[p])
             for t in present for p in present if t != p and matrix[position[t], position[p]]]
    if worst:
        count, true_code, wrong_code = max(worst)
        print(f"\nMost common confusion: {true_code} read as {wrong_code} ({count}x). "
              "Check whether their Khmer words share a shape,\nand whether the prefix "
              "cross-check in PrefixProvinceClassifier disagrees on those plates.")


if __name__ == "__main__":
    main()
