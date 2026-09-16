"""Train the vehicle-colour classifier.

    python -m ml.vehicle.train --crops dataset/vehicle_colour/v1 --epochs 20

Reads crops laid out as <crops>/<colour>/<idx>.jpg (the output of
ml.vehicle.synth, and the same layout real crops should use) and trains the
ResNet-18 in ml.vehicle.model. Saves the best-by-validation weights and the
class order beside them.

The split is a plain random hold-out: synthetic crops are independent by
construction, so there is no frame-grouping to respect as there is for plates.
When real crops are mixed in later, keep one camera/session out of training the
same way ml.datasets.split does, rather than trusting a random split across
correlated frames.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ml.device import describe, pick_device
from ml.vehicle.model import build_model, class_names, preprocess, save_classes


def load_crops(crops_dir: Path, classes: list[str]) -> list[tuple[np.ndarray, int]]:
    index_of = {c: i for i, c in enumerate(classes)}
    samples = []
    for class_dir in sorted(crops_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in index_of:
            print(f"skipping unknown colour directory {class_dir.name}")
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is not None:
                samples.append((image, index_of[class_dir.name]))
    return samples


def to_batch(samples, indices, torch):
    images = np.stack([preprocess(samples[i][0]) for i in indices])
    labels = np.array([samples[i][1] for i in indices], dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/vehicle_colour"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-per-class", type=int, default=20)
    args = parser.parse_args()

    import torch
    from torch import nn

    if not args.crops.exists():
        raise SystemExit(f"{args.crops} not found - run ml.vehicle.synth first")

    classes = class_names()
    samples = load_crops(args.crops, classes)
    if not samples:
        raise SystemExit(f"no crops found under {args.crops}")

    present = Counter(classes[label] for _, label in samples)
    thin = {c: n for c, n in present.items() if n < args.min_per_class}
    if thin:
        raise SystemExit(
            f"colours with fewer than {args.min_per_class} crops: {thin}. "
            "Generate more, or drop those colours - training on a handful reports "
            "an accuracy that means nothing."
        )

    rng = random.Random(args.seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_val = max(1, round(len(order) * args.val_fraction))
    val_idx, train_idx = set(order[:n_val]), order[n_val:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in sorted(val_idx)]

    device_name = pick_device(args.device)
    device = torch.device(device_name if device_name != "0" else "cuda:0")
    print(f"training on {describe(device_name)}")
    print(f"{len(samples)} crops, {len(present)} colours: {dict(present)}")
    print(f"train {len(train_samples)} / val {len(val_samples)}\n")

    model = build_model(len(classes)).to(device)

    counts = np.array([max(present.get(c, 0), 1) for c in classes], dtype=np.float32)
    weights = torch.from_numpy(counts.sum() / counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    train_order = list(range(len(train_samples)))
    best_accuracy, best_epoch = 0.0, 0
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_order)
        total_loss = 0.0
        for start in range(0, len(train_order), args.batch):
            batch_indices = train_order[start : start + args.batch]
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
                indices = list(range(start, min(start + args.batch, len(val_samples))))
                images, labels = to_batch(val_samples, indices, torch)
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
         "crops": len(samples), "colours": dict(present)}, indent=1))

    print(f"\nbest val accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"weights: {args.out / 'best.pt'}")
    print(f"next: python -m ml.vehicle.export --weights {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
