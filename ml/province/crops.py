"""Cut labelled plates into top-zone crops for the province classifier.

    python -m ml.province.crops --dataset dataset/v1

Reads dataset/v1/plates.jsonl (written by ml.labeling.export_labels) and writes

    dataset/v1/province/<province_code>/<frame>_<n>.jpg

Crops go through the same warp_plate + split_zones the inference pipeline uses,
so the classifier trains on exactly what it will be shown in production. Cutting
the top third by hand here instead would train it on a slightly different band
and lose accuracy that never shows up as an error.

Boxes without a province label are skipped: an unlabelled plate is not a
negative example, and folding it into a class would poison that class.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from apps.inference_worker.perspective import split_zones, warp_plate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <dataset>/province")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="fraction of box size to expand by before warping, so a "
                             "slightly tight detection box does not clip the top zone")
    args = parser.parse_args()

    plates = args.dataset / "plates.jsonl"
    if not plates.exists():
        raise SystemExit(f"{plates} not found - export labels first")

    out = args.out or args.dataset / "province"
    counts: Counter[str] = Counter()
    skipped = 0

    per_frame: dict[str, int] = {}
    for line in plates.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        province = record.get("province")
        if not province:
            skipped += 1
            continue

        frame_path = next(
            (p for p in (args.dataset / "frames").glob(f"{record['frame']}.*")), None
        )
        if frame_path is None:
            print(f"missing frame for {record['frame']}, skipped")
            skipped += 1
            continue

        image = cv2.imread(str(frame_path))
        h, w = image.shape[:2]
        x, y, bw, bh = record["box"]
        mx, my = bw * args.margin, bh * args.margin
        x1 = max(0, int((x - mx) * w))
        y1 = max(0, int((y - my) * h))
        x2 = min(w, int((x + bw + mx) * w))
        y2 = min(h, int((y + bh + my) * h))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        plate = warp_plate(image[y1:y2, x1:x2], corners=None)
        top = split_zones(plate)["top"]

        index = per_frame.get(record["frame"], 0)
        per_frame[record["frame"]] = index + 1
        target = out / province
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / f"{record['frame']}_{index}.jpg"), top)
        counts[province] += 1

    if not counts:
        raise SystemExit("no province-labelled plates found")

    print(f"wrote {sum(counts.values())} crops to {out} ({skipped} skipped)\n")
    print("per class:")
    for code, n in sorted(counts.items(), key=lambda kv: int(kv[0])):
        flag = "  <- too few to learn" if n < 20 else ""
        print(f"  {code:>3}: {n}{flag}")

    thin = [c for c, n in counts.items() if n < 20]
    if thin:
        print(f"\n{len(thin)} of {len(counts)} classes have under 20 crops. The classifier "
              "will predict the common provinces and effectively ignore these -\n"
              "collect more plates for them before trusting per-class accuracy.")


if __name__ == "__main__":
    main()
