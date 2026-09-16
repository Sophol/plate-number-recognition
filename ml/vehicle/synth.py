"""Generate synthetic vehicle body-colour crops for classifier training.

    python -m ml.vehicle.synth --out dataset/vehicle_colour/v2 --count 15000

Colour is the one vehicle attribute a heuristic already handles adequately in
good light; a learned model only earns its place by being robust to the things
that break the heuristic -- shadow, tinted glass, specular highlight, and the
camera's own white balance. So the generator does not paint flat swatches. Each
crop starts from a per-class body colour and is degraded with exactly those
confounders, widely and at random, so the network learns "red under a hard
shadow" and "white with a warm cast", not "the mean hue is X".

v2 widens the variety over the first pass, which saturated a ResNet-18 by epoch
four: broader per-class colour spread, several lighting models (linear, radial,
vignette) instead of one ramp, stronger and per-channel white balance, up to two
windows and two highlights, optional rotation, motion blur, and JPEG
recompression artifacts. More variety per crop is what stops the model
memorising a fixed set; the companion change is the runtime augmentation in
ml.vehicle.train, which re-varies every crop each epoch.

This is a bootstrap, in the spirit of ml.container.synth: it lets the pipeline
train and export with no camera and no download. Real gate crops mixed in with
--real once the camera is up are what make it trustworthy; a model trained on
synthetic alone is a baseline to beat, and a higher synthetic score is not the
goal -- robustness to real conditions is.

Output layout matches what ml.vehicle.train expects:

    dataset/vehicle_colour/v2/<colour>/<idx>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Per-class HSV centres and spreads (OpenCV HSV: H 0-179). Wider than v1 so the
# model meets the edges of each colour, not just its centre.
_CHROMATIC = {
    "red":    (0,   10),
    "orange": (16,  7),
    "yellow": (28,  7),
    "green":  (60, 22),
    "blue":   (112, 16),
}


def _base_bgr(rng: np.random.Generator, colour: str) -> np.ndarray:
    if colour == "black":
        h, s, v = rng.integers(0, 180), rng.integers(0, 70), rng.integers(8, 60)
    elif colour == "white":
        h, s, v = rng.integers(0, 180), rng.integers(0, 40), rng.integers(185, 256)
    elif colour == "grey":
        h, s, v = rng.integers(0, 180), rng.integers(0, 40), rng.integers(70, 190)
    else:
        centre, spread = _CHROMATIC[colour]
        h = int(centre + rng.normal(0, spread)) % 180
        s = rng.integers(100, 256)
        v = rng.integers(70, 245)
    hsv = np.array([[[h, np.clip(s, 0, 255), np.clip(v, 0, 255)]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.int16)


def _lighting(rng: np.random.Generator, size: int) -> np.ndarray:
    """A per-pixel additive brightness field, one of several shapes at random."""
    kind = rng.integers(0, 4)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if kind == 0:                                   # linear ramp, random axis
        ramp = np.linspace(rng.uniform(-55, 0), rng.uniform(0, 55), size).astype(np.float32)
        field = ramp[:, None] if rng.random() < 0.5 else ramp[None, :]
        field = np.broadcast_to(field, (size, size)).copy()
    elif kind == 1:                                 # diagonal gradient
        field = ((xx + yy) - 1.0) * rng.uniform(20, 70)
    elif kind == 2:                                 # radial spotlight
        cy, cx = rng.uniform(0.2, 0.8, 2)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        field = (rng.uniform(0.3, 0.7) - d) * rng.uniform(60, 140)
    else:                                           # vignette (dark corners)
        d = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        field = -(d ** 2) * rng.uniform(150, 320)
    return field


def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([5, 7, 9]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(0, 180)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0), (k, k))
    s = kernel.sum()
    return cv2.filter2D(img, -1, kernel / s if s else kernel)


def _render(rng: np.random.Generator, colour: str, size: int = 96) -> np.ndarray:
    base = _base_bgr(rng, colour)
    img = np.empty((size, size, 3), np.int16)
    img[:] = base

    # A same-colour second shade over part of the panel (a reflection or a
    # differently lit body section). Same colour family, so the label holds.
    if rng.random() < 0.4:
        shade = np.clip(base * rng.uniform(0.6, 1.4), 0, 255).astype(np.int16)
        y0, x0 = rng.integers(0, size // 2, 2)
        img[y0:, x0:] = shade

    img += _lighting(rng, size)[..., None].astype(np.int16)

    # Global brightness + contrast (overcast vs glare) and a per-channel white
    # balance cast -- the daily colour shift that a fixed HSV threshold cannot
    # follow.
    mean = img.mean()
    img = ((img - mean) * rng.uniform(0.7, 1.35) + mean * rng.uniform(0.75, 1.2)).astype(np.int16)
    img += rng.normal(0, 14, 3).astype(np.int16)

    # Up to two dark windows/shadows (the truck windscreen the heuristic trips on).
    for _ in range(int(rng.integers(0, 3))):
        wh, ww = rng.integers(size // 5, size // 2), rng.integers(size // 5, size)
        y0, x0 = rng.integers(0, size - wh + 1), rng.integers(0, size - ww + 1)
        img[y0:y0 + wh, x0:x0 + ww] = (img[y0:y0 + wh, x0:x0 + ww] * rng.uniform(0.1, 0.45)).astype(np.int16)

    # Up to two specular highlights (sun glint off paint).
    for _ in range(int(rng.integers(0, 3))):
        cy, cx = rng.integers(0, size, 2)
        rad = rng.integers(size // 10, size // 3)
        yy, xx = np.ogrid[:size, :size]
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < rad ** 2
        img[blob] = np.clip(img[blob] + rng.integers(50, 170), 0, 255)

    img += rng.normal(0, rng.uniform(2, 16), img.shape).astype(np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-12, 12), 1.0)
        img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3:
        img = _motion_blur(img, rng)
    elif rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (int(rng.choice([3, 5])),) * 2, 0)
    if rng.random() < 0.5:                           # JPEG recompression artifacts
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 92))])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=15000, help="total crops across all colours")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from apps.inference_worker.vehicle import COLOUR_CLASSES

    rng = np.random.default_rng(args.seed)
    per_class = args.count // len(COLOUR_CLASSES)
    for colour in COLOUR_CLASSES:
        d = args.out / colour
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            cv2.imwrite(str(d / f"{i:05d}.jpg"), _render(rng, colour))

    total = per_class * len(COLOUR_CLASSES)
    print(f"generated {total} synthetic crops -> {args.out} "
          f"({per_class} x {len(COLOUR_CLASSES)} colours)")
    print("colours:", ", ".join(COLOUR_CLASSES))


if __name__ == "__main__":
    main()
