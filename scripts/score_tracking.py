"""Run a video through the real inference pipeline and score the tracking.

Answers the questions eyeballing a preview cannot: did each vehicle get one
stable track id, or did the tracker fragment one transit into several (an ID
switch) or merge two vehicles into one?
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2

from apps.camera_worker.sampler import Frame
from apps.inference_worker.detector import ContourPlateDetector
from apps.inference_worker.main import build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--truth", help="ground-truth json (defaults to <video>.json)")
    ap.add_argument("--votes", type=int, default=3)
    args = ap.parse_args()

    truth_path = Path(args.truth) if args.truth else Path(args.video).with_suffix(".json")
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else None

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    pipeline = build_pipeline("tracking-test", votes_required=args.votes)
    detector_only = ContourPlateDetector()

    base = datetime.now(UTC)
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0

    frames = 0
    frames_with_detection = 0
    detections_total = 0
    track_texts: dict[str, Counter] = defaultdict(Counter)
    track_frames: dict[str, list[int]] = defaultdict(list)
    committed: list[tuple[str, str]] = []

    while True:
        ok, image = capture.read()
        if not ok:
            break
        ts = base + timedelta(seconds=frames / fps)

        found = detector_only.detect(image)
        detections_total += len(found)
        frames_with_detection += 1 if found else 0

        # Re-run the tracker through the pipeline so assignments match production.
        for track_id, _det in pipeline.tracker.update(found, ts):
            track_frames[track_id].append(frames)

        for read in pipeline.process(Frame(camera_id="test", image=image, frame_ts=ts)):
            committed.append((read.track_id, read.plate_text))
            track_texts[read.track_id][read.plate_text] += 1

        frames += 1

    capture.release()

    print(f"frames                 {frames}")
    print(f"frames with detection  {frames_with_detection}")
    print(f"detections total       {detections_total}")
    print(f"distinct track ids     {len(track_frames)}")
    print(f"committed reads        {len(committed)}")

    if track_frames:
        print("\ntrack id    frames  span")
        for track_id, seen in sorted(track_frames.items(), key=lambda kv: kv[1][0]):
            print(f"  {track_id:<9} {len(seen):>5}  {seen[0]}-{seen[-1]}")

    if committed:
        print("\ncommitted reads")
        for track_id, text in committed:
            print(f"  {track_id:<9} {text}")

    if truth:
        expected = truth["expected_tracks"]
        actual = len(track_frames)
        print(f"\nexpected vehicles      {expected}")
        print(f"observed track ids     {actual}")
        if actual > expected:
            print(f"  -> {actual - expected} extra id(s): the tracker split a transit "
                  f"(IoU fell below its threshold between frames)")
        elif actual < expected:
            print(f"  -> {expected - actual} missing: vehicles merged or were never detected")
        else:
            print("  -> one track per vehicle")

        plates = {v["plate"] for v in truth["vehicles"]}
        read_texts = {t for _, t in committed}
        if read_texts:
            print(f"\nplates in video        {sorted(plates)}")
            print(f"plates read            {sorted(read_texts)}")
            print(f"exact matches          {sorted(plates & read_texts)}")


if __name__ == "__main__":
    main()
