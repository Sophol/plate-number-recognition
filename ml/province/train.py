"""Train the province model. Phase 1 deliverable.

    python -m ml.province.train --crops dataset/v1/province

Classifies the Khmer top zone, which is the plan's design and the reason this
model exists: Khmer OCR is unreliable, so the province word is recognised as a
shape rather than read as text.

It is a cross-check, not the primary path. `PrefixProvinceClassifier` already
derives the province from the plate number's leading digits with no image at
all, and is more reliable whenever OCR read the number correctly. This model
earns its place on plates where the number is misread or the prefix is absent.

The split is by frame, not by crop, for the same reason ml/datasets/split.py
splits by video: two plates cropped from one frame are not independent samples.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ml.device import describe, pick_device
from ml.province.model import build_model, class_names, preprocess, save_classes


def load_crops(crops_dir: Path, classes: list[str]) -> list[tuple[np.ndarray, int, str]]:
    """Return (image, label index, frame stem) for every crop on disk."""
    index_of = {code: i for i, code in enumerate(classes)}
    samples = []
    for class_dir in sorted(crops_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in index_of:
            print(f"skipping unknown province directory {class_dir.name}")
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            # "<frame>_<n>.jpg" - strip the crop index to recover the frame.
            frame = image_path.stem.rsplit("_", 1)[0]
            samples.append((image, index_of[class_dir.name], frame))
    return samples


def split_by_frame(samples, val_fraction: float, seed: int):
    frames = sorted({frame for _, _, frame in samples})
    random.Random(seed).shuffle(frames)
    n_val = max(1, round(len(frames) * val_fraction)) if len(frames) > 1 else 0
    val_frames = set(frames[:n_val])
    train = [s for s in samples if s[2] not in val_frames]
    val = [s for s in samples if s[2] in val_frames]
    return train, val


def to_batch(samples, indices, torch):
    images = np.stack([preprocess(samples[i][0]) for i in indices])
    labels = np.array([samples[i][1] for i in indices], dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True, help="output of ml.province.crops")
    parser.add_argument("--out", type=Path, default=Path("runs/province"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-per-class", type=int, default=5,
                        help="refuse to train when a present class has fewer crops")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    if not args.crops.exists():
        raise SystemExit(f"{args.crops} not found - run ml.province.crops first")

    classes = class_names()
    samples = load_crops(args.crops, classes)
    if not samples:
        raise SystemExit(f"no crops found under {args.crops}")

    present = Counter(classes[label] for _, label, _ in samples)
    thin = {code: n for code, n in present.items() if n < args.min_per_class}
    if thin:
        raise SystemExit(
            f"classes with fewer than {args.min_per_class} crops: {thin}. "
            "Collect more plates for them, or drop those provinces from the dataset - "
            "training on one or two examples reports an accuracy that means nothing."
        )

    train_samples, val_samples = split_by_frame(samples, args.val_fraction, args.seed)
    if not val_samples:
        raise SystemExit("every crop came from one frame - cannot hold out a validation set")

    device_name = pick_device(args.device)
    device = torch.device(device_name if device_name != "0" else "cuda:0")
    print(f"training on {describe(device_name)}")
    print(f"{len(samples)} crops, {len(present)} of {len(classes)} provinces present")
    print(f"train {len(train_samples)} / val {len(val_samples)} (split by frame)\n")

    model = build_model(len(classes)).to(device)

    # Plate populations are dominated by one or two provinces. Without the
    # weighting the model scores well by answering "Phnom Penh" every time.
    counts = np.array([max(present.get(code, 0), 1) for code in classes], dtype=np.float32)
    weights = torch.from_numpy((counts.sum() / counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    rng = random.Random(args.seed)
    order = list(range(len(train_samples)))
    best_accuracy, best_epoch = 0.0, 0
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), args.batch):
            batch_indices = order[start : start + args.batch]
            images, labels = to_batch(train_samples, batch_indices, torch)
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch_indices)
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(val_samples), args.batch):
                indices = range(start, min(start + args.batch, len(val_samples)))
                images, labels = to_batch(val_samples, list(indices), torch)
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
        accuracy = correct / len(val_samples)

        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), args.out / "best.pt")
            save_classes(args.out / "classes.json", classes)

        print(f"epoch {epoch:>3}/{args.epochs}  loss {total_loss/len(train_samples):.4f}  "
              f"val acc {accuracy:.4f}" + ("  *" if epoch == best_epoch else ""))

    (args.out / "summary.json").write_text(json.dumps(
        {"best_val_accuracy": best_accuracy, "best_epoch": best_epoch,
         "crops": len(samples), "provinces_present": sorted(present)}, indent=1))

    print(f"\nbest val accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"weights: {args.out / 'best.pt'}")
    print(f"next: python -m ml.province.evaluate --weights {args.out / 'best.pt'} "
          f"--crops {args.crops}")


if __name__ == "__main__":
    main()
