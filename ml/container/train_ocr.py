"""Train the container-number OCR (CRNN + CTC) on synthetic crops.

    python -m ml.container.train_ocr --data dataset/container_synth/v1 --epochs 25

Trains on synthetic crops -- unlimited, correctly labelled by construction --
and reports two metrics: full-string exact match, and the fraction of reads that
satisfy the ISO 6346 checksum. The checksum number is the honest one for a gate:
a read that passes the check digit can be trusted, and that is what production
keeps.

Real auto-labelled crops from the gate camera can be mixed in with --real to
teach this camera's degradation; synthetic alone learns the glyphs and format.
Exports the best model to models/container_ocr.onnx.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from apps.inference_worker.container import parse
from ml.container.ocr_model import (
    BLANK, CHARSET, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    build_model, encode, greedy_decode,
)
from ml.device import describe, pick_device


def load_samples(data_dir: Path) -> list[tuple[Path, str]]:
    labels = data_dir / "labels.jsonl"
    out = []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append((data_dir / "crops" / r["crop"], r["container"]))
    return out


def prep(path: Path) -> np.ndarray:
    """Grayscale, rotate tall crops to horizontal, resize to the model input."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH), np.float32)
    h, w = img.shape[:2]
    if h > w * 1.3:                       # a vertical side-number crop
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32) / 255.0


def batches(samples, size, torch):
    for start in range(0, len(samples), size):
        chunk = samples[start:start + size]
        imgs = np.stack([prep(p) for p, _ in chunk])[:, None]
        targets, lengths = [], []
        for _, text in chunk:
            enc = encode(text)
            targets += enc
            lengths.append(len(enc))
        yield (torch.from_numpy(imgs),
               torch.tensor(targets, dtype=torch.long),
               torch.tensor(lengths, dtype=torch.long),
               [t for _, t in chunk])


def evaluate(model, samples, torch, device, batch):
    model.eval()
    exact = valid = total = 0
    with torch.no_grad():
        for imgs, _, _, texts in batches(samples, batch, torch):
            logits = model(imgs.to(device))
            preds = logits.argmax(2).cpu().numpy()
            for row, truth in zip(preds, texts):
                read = greedy_decode(row.tolist())
                total += 1
                exact += read == truth
                p = parse(read)
                valid += bool(p and p.checksum_ok)
    return exact / total, valid / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="synthetic dataset dir")
    ap.add_argument("--real", type=Path, default=None, help="optional real crop dataset to mix in")
    ap.add_argument("--out", type=Path, default=Path("runs/container_ocr"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init", type=Path, default=None,
                    help="warm-start from these weights (a previous best.pt) instead of scratch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    samples = load_samples(args.data)
    if args.real and args.real.exists():
        samples += load_samples(args.real)
    if not samples:
        raise SystemExit("no samples found")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_fraction))
    val, train = samples[:n_val], samples[n_val:]

    device_name = pick_device(args.device)
    device = torch.device(device_name if device_name != "0" else "cuda:0")
    print(f"training on {describe(device_name)}")
    print(f"{len(train)} train / {len(val)} val crops, {NUM_CLASSES} classes\n")

    model = build_model()
    if args.init:
        # Warm start: a model that has already learned the CTC alignment and the
        # glyph shapes adapts to a harder domain (thinner fonts, real camera
        # crops) far faster than one that must rediscover them from noise.
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"warm-started from {args.init}")
    model = model.to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_valid = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        total_loss = 0.0
        for imgs, targets, lengths, _ in batches(train, args.batch, torch):
            imgs = imgs.to(device)
            logits = model(imgs).log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)
            loss = criterion(logits, targets.to(device), input_lengths, lengths)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * imgs.size(0)
        scheduler.step()

        exact, valid = evaluate(model, val, torch, device, args.batch)
        marker = ""
        if valid >= best_valid:
            best_valid = valid
            torch.save(model.state_dict(), args.out / "best.pt")
            marker = " *"
        print(f"epoch {epoch:>3}/{args.epochs}  loss {total_loss/len(train):.3f}  "
              f"exact {exact:.3f}  checksum-valid {valid:.3f}{marker}")

    print(f"\nbest checksum-valid {best_valid:.3f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
