"""Extract representative frames from recorded RTSP video and build a dataset manifest.

A gate camera at 25 FPS produces thousands of near-identical frames per vehicle.
Labelling all of them wastes annotator time and inflates the apparent dataset
size while adding almost no information, so frames are kept only when they
differ enough from the last kept one.

Every frame records the video it came from as its `group`. `split.py` keeps a
group whole, which is what stops near-identical frames of the same car landing
in both train and test and quietly inflating the reported accuracy.

    python -m ml.datasets.prepare --videos recordings/ --out dataset/v1

Writes dataset/v1/frames/*.jpg and dataset/v1/manifest.jsonl. Label the frames
externally (Label Studio or Roboflow, per the plan) and drop the YOLO .txt
labels into dataset/v1/labels/ before running split.py.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".ts"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that changed between two frames, on a coarse grayscale."""
    small_a = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
    small_b = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
    changed = cv2.absdiff(small_a, small_b) > 25
    return float(changed.mean())


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian - low means motion-blurred, which OCR cannot read."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video(path: Path, out_dir: Path, min_difference: float,
                  min_sharpness: float, every: int) -> list[dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")

    records: list[dict] = []
    previous: np.ndarray | None = None
    index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index % every:
            continue
        if previous is not None and frame_difference(previous, frame) < min_difference:
            continue
        if sharpness(frame) < min_sharpness:
            continue

        name = f"{path.stem}_{index:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame)
        records.append({
            "file": name,
            "group": path.stem,
            "source": str(path),
            "frame_index": index,
            "sharpness": round(sharpness(frame), 1),
        })
        previous = frame

    capture.release()
    return records


def copy_image(path: Path, out_dir: Path) -> dict:
    """Still photos are their own group - one image cannot leak into two splits."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    name = f"{path.stem}.jpg"
    cv2.imwrite(str(out_dir / name), image)
    return {
        "file": name,
        "group": path.stem,
        "source": str(path),
        "frame_index": 0,
        "sharpness": round(sharpness(image), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="*", default=[],
                        help="video files or directories of recordings")
    parser.add_argument("--images", type=Path, nargs="*", default=[],
                        help="still images or directories to fold into the dataset")
    parser.add_argument("--out", type=Path, required=True, help="dataset version dir, e.g. dataset/v1")
    parser.add_argument("--every", type=int, default=5,
                        help="consider only every Nth frame before the difference check")
    parser.add_argument("--min-difference", type=float, default=0.02,
                        help="fraction of pixels that must change to keep a frame")
    parser.add_argument("--min-sharpness", type=float, default=40.0,
                        help="drop frames blurrier than this (variance of Laplacian)")
    args = parser.parse_args()

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def expand(paths: list[Path], suffixes: set[str]) -> list[Path]:
        found: list[Path] = []
        for path in paths:
            if path.is_dir():
                found += sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)
            elif path.suffix.lower() in suffixes:
                found.append(path)
        return found

    records: list[dict] = []
    for video in expand(args.videos, VIDEO_SUFFIXES):
        kept = extract_video(video, frames_dir, args.min_difference, args.min_sharpness, args.every)
        print(f"{video.name}: kept {len(kept)} frames")
        records += kept
    for image in expand(args.images, IMAGE_SUFFIXES):
        records.append(copy_image(image, frames_dir))
        print(f"{image.name}: copied")

    if not records:
        raise SystemExit("no frames extracted - check --videos/--images paths")

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    groups = {record["group"] for record in records}
    print(f"\n{len(records)} frames from {len(groups)} groups -> {manifest}")
    print(f"Next: label {frames_dir} and write YOLO .txt files to {args.out / 'labels'}")


if __name__ == "__main__":
    main()
