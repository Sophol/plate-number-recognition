"""Convert a Label Studio JSON export into YOLO labels and a plate attribute file.

    python -m ml.labeling.export_labels --export export.json --dataset dataset/v1

Writes two things, because one labelling pass feeds three models:

    dataset/v1/labels/<frame>.txt   YOLO boxes -> ml.detection.train
    dataset/v1/plates.jsonl         text/province/type per box -> OCR + province

Label Studio's own YOLO export is not used: it drops the per-region text and
province attributes, which are the whole reason for labelling in one pass.

Boxes arrive as percentages of the image, which is what makes this conversion
safe against resizing - YOLO wants normalised centre coordinates, so the two
formats differ only by the centre shift and a factor of 100.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_CLASSES = ["plate"]


def region_attributes(results: list[dict]) -> dict[str, dict]:
    """Collect the per-region text/choice answers, keyed by the region id they annotate."""
    attributes: dict[str, dict] = {}
    for result in results:
        region_id = result.get("id")
        if region_id is None or result["type"] == "rectanglelabels":
            continue
        value = result.get("value", {})
        entry = attributes.setdefault(region_id, {})
        if result["type"] == "textarea":
            texts = value.get("text") or []
            entry["plate_text"] = texts[0].strip() if texts else ""
        elif result["type"] == "choices":
            choices = value.get("choices") or []
            if choices:
                entry[result["from_name"]] = choices[0]
    return attributes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True,
                        help="Label Studio export in JSON format (not JSON-MIN)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--skip-unreadable", action="store_true",
                        help="drop boxes marked plate_type=unreadable instead of keeping "
                             "them as detector-only samples")
    args = parser.parse_args()

    tasks = json.loads(args.export.read_text())
    labels_dir = args.dataset / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_index = {name: i for i, name in enumerate(args.classes)}
    plate_records: list[dict] = []
    counts: Counter[str] = Counter()
    empty_frames = 0

    for task in tasks:
        image = task.get("data", {}).get("image", "")
        stem = Path(image.split("?d=")[-1]).stem
        if not stem:
            counts["skipped_no_image"] += 1
            continue

        annotations = task.get("annotations") or []
        # An annotation the labeller cancelled means "nothing here", which is a
        # real answer - but a task nobody opened is not, and must not become an
        # empty label file that teaches the detector this frame has no plates.
        done = [a for a in annotations if not a.get("was_cancelled")]
        if not annotations:
            counts["unannotated"] += 1
            continue

        results = done[-1]["result"] if done else []
        attributes = region_attributes(results)

        lines: list[str] = []
        for result in results:
            if result["type"] != "rectanglelabels":
                continue
            value = result["value"]
            names = value.get("rectanglelabels") or ["plate"]
            name = names[0]
            if name not in class_index:
                counts[f"unknown_class:{name}"] += 1
                continue

            attrs = attributes.get(result.get("id"), {})
            if args.skip_unreadable and attrs.get("plate_type") == "unreadable":
                counts["skipped_unreadable"] += 1
                continue

            x, y = value["x"] / 100, value["y"] / 100
            width, height = value["width"] / 100, value["height"] / 100
            lines.append(
                f"{class_index[name]} {x + width / 2:.6f} {y + height / 2:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            counts["boxes"] += 1

            text = attrs.get("plate_text", "")
            province = attrs.get("province")
            if text or province:
                plate_records.append({
                    "frame": stem,
                    "box": [round(x, 6), round(y, 6), round(width, 6), round(height, 6)],
                    "plate_text": text,
                    "province": None if province in (None, "unknown") else province,
                    "plate_type": attrs.get("plate_type"),
                })

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts["frames"] += 1
        if not lines:
            empty_frames += 1

    plates = args.dataset / "plates.jsonl"
    with plates.open("w") as handle:
        for record in plate_records:
            handle.write(json.dumps(record) + "\n")

    print(f"frames written      {counts['frames']} ({empty_frames} with no plate)")
    print(f"boxes               {counts['boxes']}")
    print(f"plate attributes    {len(plate_records)} -> {plates}")
    for key, value in sorted(counts.items()):
        if key not in {"frames", "boxes"}:
            print(f"  {key}: {value}")

    transcribed = sum(1 for r in plate_records if r["plate_text"])
    with_province = sum(1 for r in plate_records if r["province"])
    print(f"\ntranscribed {transcribed}/{counts['boxes']} boxes, "
          f"province on {with_province}/{counts['boxes']}")
    print(f"next: python -m ml.datasets.split --dataset {args.dataset}")


if __name__ == "__main__":
    main()
