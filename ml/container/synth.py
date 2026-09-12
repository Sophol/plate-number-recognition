"""Generate synthetic ISO 6346 container-number crops for OCR training.

Container numbers are the ideal synthetic-data target: a rigid format, a fixed
family of marine fonts, and a check digit that lets every generated number be
valid by construction. So a model can learn the glyphs and layout from
unlimited correctly-labelled examples, then be fine-tuned on the handful of real
crops the gate camera produces -- the real crops teach it this camera's blur,
glare and rust, which is the part synthesis cannot know.

The first version of this generator rendered Arial Black on saturated panels and
the resulting model read real crops at 0%. A side-by-side showed why, and each
difference is now modelled deliberately:

- real numbers are a thin, wide DIN-style face, not a heavy grotesque -> DIN
  Alternate / DIN Condensed / Arial Narrow lead the font mix, and strokes are
  randomly eroded thinner;
- characters are widely tracked with a large gap between the owner code, the
  serial and the check digit -> rendered per character with random spacing;
- the check digit sits inside a small box on almost every real container -> a
  box is drawn around it most of the time;
- real paint is faint tan on rust, not white on blue -> text colour is blended
  toward the wall colour to lower contrast, and rust tones join the walls.

Owner-code prefixes are weighted by the terminal's own PAS list when it has been
fetched, so the priors match what actually rolls through this gate.

    python -m ml.container.synth --out dataset/container_synth/v2 --count 8000
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from apps.inference_worker.container import compute_check_digit

SUP = "/System/Library/Fonts/Supplemental/"
# (path, face index for .ttc, sampling weight). DIN faces lead: they are what
# container stencils are based on. Arial Black kept at low weight for variety.
FONT_CANDIDATES = [
    (SUP + "DIN Alternate Bold.ttf", 0, 5),
    (SUP + "DIN Condensed Bold.ttf", 0, 3),
    (SUP + "Arial Narrow.ttf", 0, 3),
    (SUP + "Arial Narrow Bold.ttf", 0, 2),
    (SUP + "Avenir Next Condensed.ttc", 0, 2),
    ("/System/Library/Fonts/Helvetica.ttc", 0, 2),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, 2),
    (SUP + "Courier New.ttf", 0, 1),
    ("/System/Library/Fonts/Menlo.ttc", 0, 1),
    (SUP + "Arial Black.ttf", 0, 1),
]

OWNERS = ["MSC", "MAE", "CMA", "TCL", "TGH", "HLC", "OOL", "GES", "SEG", "TEM",
          "CSG", "MRS", "TII", "CAI", "FCI", "DFS", "BMO", "WHL", "ONE", "APL"]

# Container-wall colours (BGR). Rust and faded tones matter more than the
# saturated ones: that is what a working box at a gate looks like.
WALLS = [
    (110, 78, 45), (95, 70, 50), (60, 55, 140), (70, 60, 120), (45, 55, 95),
    (55, 70, 110), (40, 50, 85), (120, 120, 120), (150, 150, 145),
    (235, 235, 235), (70, 110, 70), (60, 90, 65), (150, 160, 170), (90, 100, 120),
]
# Paint colours (BGR): tans, off-whites, blacks, dark greys.
LIGHT_PAINT = [(140, 190, 215), (180, 215, 235), (225, 230, 235), (190, 225, 240), (200, 210, 220)]
DARK_PAINT = [(20, 20, 20), (50, 50, 55), (40, 40, 70), (35, 45, 60)]


def load_prefixes(pas_json: Path | None) -> tuple[list[str], list[int]]:
    """Owner+category prefixes with weights, from the terminal's own PAS data."""
    if pas_json and pas_json.exists():
        top = json.loads(pas_json.read_text()).get("top_owner_codes", [])
        ok = [(c, n) for c, n in top if len(c) == 4 and c[:3].isalpha() and c[3] in "UJZ"]
        if ok:
            return [c for c, _ in ok], [n for _, n in ok]
    return [o + "U" for o in OWNERS], [1] * len(OWNERS)


def load_numbers(pas_json: Path | None) -> list[str]:
    """Every checksum-valid container number the terminal has handled."""
    if pas_json and pas_json.exists():
        data = json.loads(pas_json.read_text())
        return [c["number"] for c in data["containers"] if c.get("checksum_ok")]
    return []


def random_number(rng: random.Random, prefixes) -> str:
    codes, weights = prefixes
    prefix = rng.choices(codes, weights=weights, k=1)[0]
    body = f"{prefix}{rng.randint(0, 999999):06d}"
    return f"{body}{compute_check_digit(body)}"


def sample_number(rng: random.Random, prefixes, real: list[str], real_fraction: float) -> str:
    """A real PAS number most of the time, a random valid one otherwise.

    Real numbers give the model the terminal's actual owner codes and serial
    ranges. The random share is deliberate: a container this gate has never seen
    must still read correctly, so the model must not learn the finite list.
    """
    if real and rng.random() < real_fraction:
        return rng.choice(real)
    return random_number(rng, prefixes)


def load_fonts():
    fonts = [(p, i, w) for p, i, w in FONT_CANDIDATES if Path(p).exists()]
    if not fonts:
        raise SystemExit("no usable fonts found")
    return fonts


def corrugated_panel(w: int, h: int, colour, rng: random.Random) -> np.ndarray:
    panel = np.full((h, w, 3), colour, np.float32)
    period = rng.randint(14, 30)
    x = np.arange(w)
    shade = (np.sin(2 * np.pi * x / period) * rng.uniform(4, 18)).astype(np.float32)
    panel += shade[None, :, None]
    for _ in range(rng.randint(0, 4)):
        cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(3, 14)
        cv2.circle(panel, (cx, cy), r, tuple(float(c) * rng.uniform(0.4, 0.8) for c in colour), -1)
    return np.clip(panel, 0, 255).astype(np.uint8)


def luminance(bgr) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def render(number: str, fonts, rng: random.Random, vertical: bool) -> np.ndarray:
    path, index, _ = rng.choices(fonts, weights=[w for _, _, w in fonts], k=1)[0]
    size = rng.randint(30, 50)
    font = ImageFont.truetype(path, size, index=index)

    # Tracking: wide gaps between glyphs, wider between the three groups.
    gap = rng.uniform(0.12, 0.55) * size
    group_gap = rng.uniform(0.6, 1.6) * size
    advances = [font.getlength(c) for c in number]
    gaps = []
    for i in range(len(number)):
        gaps.append(group_gap if i in (3, 9) else gap)
    text_w = int(sum(advances) + sum(gaps[:-1]))
    ascent, descent = font.getmetrics()
    text_h = ascent + descent

    pad_x = int(rng.uniform(0.4, 1.2) * size)
    pad_y = int(rng.uniform(0.25, 0.6) * size)
    W, H = text_w + 2 * pad_x, text_h + 2 * pad_y

    # Draw the glyph mask, then optionally thin the strokes.
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    x = float(pad_x)
    last_box = None
    for i, ch in enumerate(number):
        draw.text((x, pad_y), ch, font=font, fill=255)
        l, t, r, b = font.getbbox(ch)
        last_box = (x + l, pad_y + t, x + r, pad_y + b)
        x += advances[i] + gaps[i]

    # The boxed check digit, most of the time.
    if last_box and rng.random() < 0.75:
        l, t, r, b = last_box
        m = rng.uniform(0.08, 0.22) * size
        draw.rectangle([l - m, t - m, r + m, b + m], outline=255, width=rng.randint(1, 3))

    mask_np = np.array(mask)
    if rng.random() < 0.5:
        k = rng.choice([2, 3])
        mask_np = cv2.erode(mask_np, np.ones((k, k), np.uint8))

    wall = rng.choice(WALLS)
    paint = rng.choice(LIGHT_PAINT if luminance(wall) < 140 else DARK_PAINT)
    # Lower the contrast: real paint is faint, weathered.
    f = rng.uniform(0.0, 0.5)
    paint = tuple(int(p * (1 - f) + c * f) for p, c in zip(paint, wall))

    panel = corrugated_panel(W, H, wall, rng).astype(np.float32)
    alpha = (mask_np.astype(np.float32) / 255.0)[..., None]
    img = (panel * (1 - alpha) + np.array(paint, np.float32) * alpha).astype(np.uint8)

    if vertical:
        img = cv2.rotate(img, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
    return augment(img, rng)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(w, h) * rng.uniform(0.0, 0.14)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, m), rng.uniform(0, m)],
                      [w - rng.uniform(0, m), rng.uniform(0, m)],
                      [w - rng.uniform(0, m), h - rng.uniform(0, m)],
                      [rng.uniform(0, m), h - rng.uniform(0, m)]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    ang = rng.uniform(-4, 4)
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), (w, h),
                         borderMode=cv2.BORDER_REPLICATE)
    img = np.clip(img.astype(np.float32) * rng.uniform(0.6, 1.25) + rng.uniform(-25, 25),
                  0, 255).astype(np.uint8)
    k = rng.choice([1, 1, 3, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.7:
        img = np.clip(img.astype(np.int16) + rng.randint(3, 12) * np.random.randn(h, w, 3),
                      0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 85)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vertical-fraction", type=float, default=0.3)
    ap.add_argument("--owners", type=Path, default=Path("dataset/pas/containers.json"))
    ap.add_argument("--numbers", type=Path, default=Path("dataset/pas/containers.json"),
                    help="PAS list; its real numbers are rendered directly")
    ap.add_argument("--real-fraction", type=float, default=0.7,
                    help="share of crops that render a real PAS number rather than a random one")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    fonts = load_fonts()
    prefixes = load_prefixes(args.owners)
    real_numbers = load_numbers(args.numbers)
    crops = args.out / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    labels = []
    for i in range(args.count):
        number = sample_number(rng, prefixes, real_numbers, args.real_fraction)
        vertical = rng.random() < args.vertical_fraction
        img = render(number, fonts, rng, vertical)
        name = f"{i:06d}_{number}.jpg"
        cv2.imwrite(str(crops / name), img)
        labels.append({"crop": name, "container": number, "vertical": vertical})

    (args.out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")
    print(f"generated {len(labels)} synthetic crops -> {crops}")
    print(f"fonts: {len(fonts)} faces, DIN-led; owner prefixes: {len(prefixes[0])} "
          f"({'PAS-weighted' if args.owners.exists() else 'static fallback'})")
    print(f"real PAS numbers available: {len(real_numbers)} "
          f"(rendered for ~{int(args.real_fraction * 100)}% of crops)")


if __name__ == "__main__":
    main()
