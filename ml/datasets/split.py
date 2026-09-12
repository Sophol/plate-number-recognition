"""Split into train/val/test, keeping near-identical frames from one video together.

The split is by group, never by frame. Two frames of the same car half a second
apart are effectively the same sample; separating them across train and test
turns the test set into a memorisation check and reports an accuracy the gate
will never reach in the field.

    python -m ml.datasets.split --dataset dataset/v1

Materialises the layout Ultralytics expects, using symlinks so a dataset version
is not duplicated on disk:

    dataset/v1/yolo/{train,val,test}/{images,labels}
    dataset/v1/yolo/data.yaml
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# One class by default: finding the plate and classifying its type are separate
# problems, and plate_type is far easier to get from the validated text than
# from a handful of pixels. Pass --classes to train a multi-class detector.
DEFAULT_CLASSES = ["plate"]


def load_manifest(dataset: Path) -> list[dict]:
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found - run ml.datasets.prepare first")
    with manifest.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assign_groups(groups: list[str], ratios: tuple[float, float, float],
                  seed: int) -> dict[str, str]:
    """Assign whole groups to splits, giving val and test a group each before train.

    Rounding the ratios directly starves val and test when there are only a
    handful of recordings - and an empty val split means no early stopping and
    no honest test number, which is worse than a slightly small train split.
    """
    total = len(groups)
    if total < 3:
        raise SystemExit(
            f"only {total} group(s) in the manifest - need at least 3 to hold out "
            "separate val and test sets. Record more videos before training."
        )

    random.Random(seed).shuffle(groups)
    _, val_ratio, test_ratio = ratios
    n_val = max(1, round(total * val_ratio))
    n_test = max(1, round(total * test_ratio))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise SystemExit(f"ratios {ratios} leave no groups for training out of {total}")

    assignment = {}
    for i, group in enumerate(groups):
        if i < n_train:
            assignment[group] = "train"
        elif i < n_train + n_val:
            assignment[group] = "val"
        else:
            assignment[group] = "test"
    return assignment


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset version dir, e.g. dataset/v1")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-unlabelled", action="store_true",
                        help="keep frames with no .txt label (treated as background)")
    args = parser.parse_args()

    records = load_manifest(args.dataset)
    frames_dir = args.dataset / "frames"
    labels_dir = args.dataset / "labels"

    labelled, unlabelled = [], []
    for record in records:
        label = labels_dir / f"{Path(record['file']).stem}.txt"
        (labelled if label.exists() else unlabelled).append(record)

    if unlabelled and not args.allow_unlabelled:
        print(f"{len(unlabelled)} of {len(records)} frames have no label in {labels_dir}.")
        print("Label them, or pass --allow-unlabelled to treat them as background.")
        if not labelled:
            raise SystemExit("nothing to split")
    usable = records if args.allow_unlabelled else labelled

    by_group: dict[str, list[dict]] = defaultdict(list)
    for record in usable:
        by_group[record["group"]].append(record)

    assignment = assign_groups(list(by_group), tuple(args.ratios), args.seed)

    counts: dict[str, int] = defaultdict(int)
    out = args.dataset / "yolo"
    for group, group_records in by_group.items():
        split = assignment[group]
        for record in group_records:
            stem = Path(record["file"]).stem
            link(frames_dir / record["file"], out / split / "images" / record["file"])
            label = labels_dir / f"{stem}.txt"
            if label.exists():
                link(label, out / split / "labels" / f"{stem}.txt")
            counts[split] += 1

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(args.classes)}\n"
        f"names: {list(args.classes)}\n"
    )

    print(f"\ngroups: {len(by_group)}  frames: {sum(counts.values())}")
    for split in ("train", "val", "test"):
        group_count = sum(1 for g, s in assignment.items() if s == split)
        print(f"  {split:5} {counts[split]:5} frames from {group_count} groups")
    print(f"\nwrote {data_yaml}")


if __name__ == "__main__":
    main()
