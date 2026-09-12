"""Auto-label plate crops from gate video, using format + track voting.

Manual transcription is the bottleneck for plate OCR. This removes most of it by
exploiting two things the gate gives for free:

1. **Format.** A Cambodian plate number has a rigid shape, so an OCR read either
   matches the pattern or it does not. A misread is rejected, not stored as a
   wrong label -- the same guarantee the container check digit gives.

2. **Repetition.** The tracker sees each vehicle across many frames. Only one of
   them has to read correctly: the majority format-valid read becomes the label
   for the whole track, so the blurry frames the OCR could not read are labelled
   from the clear ones of the same plate. Those hard crops are exactly what a
   trained model most needs to see.

The output is a labelled crop dataset ready for a recognition model, and the same
read+vote logic can run live (no training) as a first plate reader. EasyOCR does
the reading here because it is only needed offline on the training machine; the
trained model that replaces it exports to ONNX and needs no torch on the server.

    python -m ml.ocr.autolabel --videos recordings/session_x --out dataset/plate_ocr/v1
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from apps.inference_worker.onnx_detector import OnnxPlateDetector
from apps.inference_worker.tracker import IoUTracker

# digit, one or two letters, four digits -- separator-free, as OCR returns it.
PLATE_RE = re.compile(r"[1-9][A-Z]{1,2}\d{4}")
MIN_ASPECT, MAX_ASPECT = 1.8, 5.5


def clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def extract_plate(text: str) -> str | None:
    """Pull a plate-shaped token out of an OCR string, or None."""
    match = PLATE_RE.search(clean(text))
    return match.group() if match else None


def read_crop(reader, crop: np.ndarray) -> str:
    """OCR one plate crop, upscaled. Returns the raw cleaned string."""
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(up, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
    return clean("".join(t for _, t, c in results if c > 0.3))


def plate_shaped(det) -> bool:
    h = max(det.y2 - det.y1, 1)
    return MIN_ASPECT < (det.x2 - det.x1) / h < MAX_ASPECT


def process_video(path: Path, detector, reader, sample_every: int):
    """Return {track_id: [(crop, raw_read), ...]} for one video."""
    tracker = IoUTracker()
    per_track: dict[int, list[tuple[np.ndarray, str]]] = defaultdict(list)
    capture = cv2.VideoCapture(str(path))
    index = -1
    from datetime import UTC, datetime, timedelta

    base = datetime.now(UTC)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index % sample_every:
            continue
        dets = [d for d in detector.detect(frame) if plate_shaped(d)]
        if not dets:
            continue
        # frame_ts must advance for the tracker's motion model.
        ts = base + timedelta(seconds=index / 25.0)
        for track_id, det in tracker.update(dets, ts):
            crop = frame[max(0, det.y1 - 5):det.y2 + 5, max(0, det.x1 - 5):det.x2 + 5]
            if crop.size == 0:
                continue
            per_track[track_id].append((crop.copy(), read_crop(reader, crop)))
    capture.release()
    return per_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True,
                        help="video files or directories of gate footage")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/plate_detector.onnx"))
    parser.add_argument("--sample-every", type=int, default=3,
                        help="run detection on every Nth frame")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="a track needs this many agreeing format-valid reads to be labelled")
    parser.add_argument("--gpu", action="store_true", help="run EasyOCR on GPU/MPS")
    args = parser.parse_args()

    import easyocr

    detector = OnnxPlateDetector(str(args.model), conf=0.5)
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    videos: list[Path] = []
    for item in args.videos:
        if item.is_dir():
            videos += sorted(p for p in item.rglob("*") if p.suffix.lower() in {".mp4", ".mkv", ".avi"})
        else:
            videos.append(item)
    if not videos:
        raise SystemExit("no videos found")

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    stats = Counter()
    saved = 0
    for video in videos:
        per_track = process_video(video, detector, reader, args.sample_every)
        for track_id, samples in per_track.items():
            stats["tracks"] += 1
            valid = [extract_plate(raw) for _, raw in samples]
            valid = [v for v in valid if v]
            if not valid:
                stats["tracks_unreadable"] += 1
                continue
            plate, count = Counter(valid).most_common(1)[0]
            if count < args.min_votes:
                stats["tracks_below_min_votes"] += 1
                continue

            # Label every crop from this track with the consensus, including the
            # ones OCR could not read -- those are the samples worth training on.
            stats["tracks_labelled"] += 1
            for i, (crop, raw) in enumerate(samples):
                name = f"{video.stem}_{track_id}_{i}.jpg"
                cv2.imwrite(str(crops_dir / name), crop)
                labels.append({
                    "crop": name, "plate": plate, "track": f"{video.stem}_{track_id}",
                    "ocr_raw": raw, "ocr_agreed": extract_plate(raw) == plate,
                })
                saved += 1

    (args.out / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")

    print(f"\ntracks seen            {stats['tracks']}")
    print(f"  labelled             {stats['tracks_labelled']}")
    print(f"  no valid read        {stats['tracks_unreadable']}")
    print(f"  below {args.min_votes} votes         {stats['tracks_below_min_votes']}")
    print(f"crops saved            {saved} ({args.out / 'labels.jsonl'})")
    plates = {r['plate'] for r in labels}
    print(f"distinct plates        {len(plates)}: {', '.join(sorted(plates)[:12])}")
    agreed = sum(1 for r in labels if r['ocr_agreed'])
    print(f"\ncrops the OCR read right: {agreed}/{saved} - the rest are the hard ones,")
    print("labelled by track consensus, which is the point.")


if __name__ == "__main__":
    main()
