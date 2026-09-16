"""Generate synthetic vehicle body-colour crops for classifier training.

    python -m ml.vehicle.synth --out dataset/vehicle_colour/v1 --count 8000

Colour is the one vehicle attribute a heuristic already handles adequately in
good light; a learned model only earns its place by being robust to the things
that break the heuristic -- shadow, tinted glass, specular highlight, and the
camera's own white balance. So the generator does not paint flat swatches. Each
crop starts from a per-class body colour and is then degraded with exactly those
confounders, so the network learns "red under a hard shadow" and "white with a
warm cast", not "the mean hue is X".

This is a bootstrap, deliberately in the spirit of ml.container.synth: it lets
the pipeline train and export today with no camera and no download. Real gate
crops, mixed in with --real once the camera is up, are what make it trustworthy;
a model trained on synthetic alone should be treated as a baseline to beat.

Output layout matches what ml.vehicle.train expects:

    dataset/vehicle_colour/v1/<colour>/<idx>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Per-class HSV centres and spreads (OpenCV HSV: H 0-179, S/V 0-255). Achromatic
# classes (black/white/grey) are defined by value/saturation, not hue.
_CHROMATIC = {
    "red":    (0,   8),
    "orange": (15,  6),
    "yellow": (27,  6),
    "green":  (60, 18),
    "blue":   (110, 12),
}


def _base_bgr(rng: np.random.Generator, colour: str) -> np.ndarray:
    """A representative BGR triple for one colour, with in-class variation."""
    if colour == "black":
        v = rng.integers(10, 55)
        s = rng.integers(0, 60)
        h = rng.integers(0, 180)
    elif colour == "white":
        v = rng.integers(190, 255)
        s = rng.integers(0, 35)
        h = rng.integers(0, 180)
    elif colour == "grey":
        v = rng.integers(80, 180)
        s = rng.integers(0, 35)
        h = rng.integers(0, 180)
    else:
        centre, spread = _CHROMATIC[colour]
        h = int(centre + rng.normal(0, spread)) % 180
        s = rng.integers(120, 256)
        v = rng.integers(90, 240)
    hsv = np.array([[[h, np.clip(s, 0, 255), np.clip(v, 0, 255)]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.int16)


def _render(rng: np.random.Generator, colour: str, size: int = 96) -> np.ndarray:
    base = _base_bgr(rng, colour)
    img = np.empty((size, size, 3), np.int16)
    img[:] = base

    # Smooth lighting gradient across the panel (sun on one side of the body).
    axis = rng.integers(0, 2)
    ramp = np.linspace(rng.uniform(-45, 0), rng.uniform(0, 45), size)
    grad = ramp[:, None] if axis == 0 else ramp[None, :]
    img += grad[..., None].astype(np.int16)

    # Global brightness (overcast vs. glare) and a warm/cool white-balance cast,
    # which is exactly what makes a fixed HSV threshold mislabel across the day.
    img = (img.astype(np.float32) * rng.uniform(0.7, 1.25)).astype(np.int16)
    cast = rng.normal(0, 10, 3).astype(np.int16)
    img += cast

    # A dark window/shadow rectangle over part of the crop (the truck windscreen
    # that fools the centre-sampling heuristic).
    if rng.random() < 0.6:
        wh = rng.integers(size // 4, size // 2)
        ww = rng.integers(size // 4, size)
        y0 = rng.integers(0, size - wh)
        x0 = rng.integers(0, size - ww)
        img[y0:y0 + wh, x0:x0 + ww] = (img[y0:y0 + wh, x0:x0 + ww] * rng.uniform(0.15, 0.4)).astype(np.int16)

    # A specular highlight blob (sun glint off paint).
    if rng.random() < 0.5:
        cy, cx = rng.integers(0, size, 2)
        rad = rng.integers(size // 8, size // 3)
        yy, xx = np.ogrid[:size, :size]
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < rad ** 2
        img[blob] = np.clip(img[blob] + rng.integers(60, 160), 0, 255)

    # Sensor noise and a touch of blur.
    img += rng.normal(0, rng.uniform(2, 12), img.shape).astype(np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=8000, help="total crops across all colours")
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
