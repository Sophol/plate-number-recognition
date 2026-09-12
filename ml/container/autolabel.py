"""Auto-label container-number crops from gate video, validated by ISO 6346.

The container check digit makes auto-labelling stronger than it is for plates: a
read is kept only if its checksum is correct, so a misread is arithmetically
rejected, not merely format-filtered. What survives is trustworthy without a
human checking it -- on real footage the same container read as ...5437381 (a
misread of the last digit) is dropped while the ...5437389 reads are kept.

OCR runs on the FULL FRAME, not on detector boxes. The plate detector finds
horizontal plate-shaped regions and misses container numbers, which sit on the
top rail and down the side; EasyOCR's own text detector finds them wherever they
are. This was the fix that turned a 0-result run into a working one.

The crop saved is the union of only the text boxes that formed the valid number.
Bounding every text box in the frame instead produced near-full-frame "crops",
because the timestamp and camera-name overlays are text too -- and a whole frame
squashed to the recogniser's input size is noise, not training data.

    python -m ml.container.autolabel --videos recordings/ --out dataset/container/v1
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from apps.inference_worker.container import normalise, parse

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
CONTAINER = re.compile(r"[A-Z]{4}\d{7}")


def scan_frame(reader, frame: np.ndarray):
    """Full-frame OCR. Return (ContainerNumber, tight crop) for a checksum-valid
    number, else (None, None).

    The owner code and serial often come back as separate boxes, so runs of up to
    three adjacent boxes in reading order are tried, and only the boxes that
    formed the match are cropped.
    """
    results = reader.readtext(frame, allowlist=ALLOW)
    kept = [(b, normalise(t)) for b, t, c in results if c > 0.2 and t.strip()]
    # Reading order: rough row, then left to right.
    kept.sort(key=lambda bt: (min(p[1] for p in bt[0]) // 40, min(p[0] for p in bt[0])))

    n = len(kept)
    for span in (1, 2, 3):
        for start in range(0, n - span + 1):
            group = kept[start:start + span]
            match = CONTAINER.search("".join(t for _, t in group))
            if not match:
                continue
            cn = parse(match.group())
            if not (cn and cn.checksum_ok):
                continue
            xs = [int(p[0]) for b, _ in group for p in b]
            ys = [int(p[1]) for b, _ in group for p in b]
            pad_x = max(4, (max(xs) - min(xs)) // 10)
            pad_y = max(4, (max(ys) - min(ys)) // 4)
            crop = frame[max(0, min(ys) - pad_y):max(ys) + pad_y,
                         max(0, min(xs) - pad_x):max(xs) + pad_x]
            if crop.size:
                return cn, crop
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    videos = []
    for it in args.videos:
        videos += sorted(it.rglob("*.mp4")) if it.is_dir() else [it]

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    labels, stats, seen = [], Counter(), Counter()
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % args.sample_every:
                continue
            stats["frames"] += 1
            cn, crop = scan_frame(reader, frame)
            if cn is None:
                continue
            stats["valid"] += 1
            seen[cn.canonical] += 1
            name = f"{video.stem}_{i}_{cn.canonical}.jpg"
            cv2.imwrite(str(crops_dir / name), crop)
            labels.append({"crop": name, "container": cn.canonical})
        cap.release()

    (args.out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")
    print(f"\nframes scanned         {stats['frames']}")
    print(f"checksum-valid reads   {stats['valid']}")
    print(f"distinct containers    {len(seen)}: {', '.join(sorted(seen)[:10])}")
    print(f"crops saved            {len(labels)} -> {args.out / 'labels.jsonl'}")


if __name__ == "__main__":
    main()
