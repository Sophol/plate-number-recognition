"""Generate a synthetic traffic clip with known ground truth.

Real footage tells you a read looked wrong; a generated clip tells you exactly
which vehicle it belonged to, so tracking can be scored instead of eyeballed.
Writes <out>.mp4 plus <out>.json describing every vehicle and its frames.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 960, 540, 10
PLATE_W, PLATE_H = 300, 150


def draw_plate(frame, x, y, text, province, scale=1.0, blur=0.0):
    w, h = int(PLATE_W * scale), int(PLATE_H * scale)
    # Skip partially-visible plates entirely. The contour detector splits a
    # clipped plate into its three text zones, spawning phantom tracks that
    # measure edge behaviour rather than tracking.
    if x < 0 or x + w > WIDTH:
        return None

    plate = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(plate, (0, 0), (w - 1, h - 1), (25, 25, 25), max(2, int(3 * scale)))
    cv2.putText(plate, province, (int(52 * scale), int(34 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (30, 30, 30), max(1, int(2 * scale)))
    cv2.putText(plate, text, (int(22 * scale), int(104 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7 * scale, (15, 15, 15), max(2, int(4 * scale)))
    cv2.putText(plate, "CAMBODIA", (int(70 * scale), int(140 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (30, 30, 30), 1)

    if blur > 0:
        k = int(blur) * 2 + 1
        plate = cv2.GaussianBlur(plate, (k, k), 0)

    frame[y : y + h, x : x + w] = plate
    return (x, y, x + w, y + h)


def build(scenario: str):
    """Return a list of vehicles: (plate, province, start_frame, x0, speed, y, scale, blur)."""
    if scenario == "simple":
        # Well separated, one at a time - the baseline case.
        return [
            ("2D-0888", "KAMPONG", 0, -300, 9, 200, 1.0, 0),
            ("1A-4412", "PHNOMPENH", 70, -300, 9, 210, 1.0, 0),
            ("3C-9001", "SIEMREAP", 140, -300, 9, 195, 1.0, 0),
        ]
    if scenario == "fast":
        # Speed raised until frame-to-frame IoU falls under the 0.3 threshold.
        return [
            ("2D-0888", "KAMPONG", 0, -300, 20, 200, 1.0, 0),
            ("1A-4412", "PHNOMPENH", 40, -300, 45, 210, 1.0, 0),
            ("3C-9001", "SIEMREAP", 80, -300, 90, 195, 1.0, 0),
        ]
    if scenario == "crossing":
        # Two plates overlapping - greedy matching's weak spot.
        return [
            ("2D-0888", "KAMPONG", 0, -300, 11, 180, 1.0, 0),
            ("1A-4412", "PHNOMPENH", 0, WIDTH, -11, 240, 1.0, 0),
        ]
    if scenario == "hard":
        # Small, blurred and low contrast, like a distant night plate.
        return [
            ("2D-0888", "KAMPONG", 0, -300, 9, 200, 0.55, 2),
            ("1A-4412", "PHNOMPENH", 60, -300, 14, 215, 0.75, 1),
            ("3C-9001", "SIEMREAP", 120, -300, 9, 190, 1.0, 4),
        ]
    raise SystemExit(f"unknown scenario {scenario!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="simple",
                    choices=["simple", "fast", "crossing", "hard"])
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--out", default="test_traffic")
    args = ap.parse_args()

    vehicles = build(args.scenario)
    out = Path(args.out).with_suffix(".mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    truth: dict[str, dict] = {
        plate: {"plate": plate, "province": province, "frames": []}
        for plate, province, *_ in vehicles
    }

    for n in range(args.frames):
        frame = np.full((HEIGHT, WIDTH, 3), 70, np.uint8)
        cv2.rectangle(frame, (0, 380), (WIDTH, HEIGHT), (55, 55, 58), -1)
        # A dashed lane line would be detected as plates: bright, plate-shaped,
        # and moving too fast to hold a track. A thin continuous line gives the
        # motion gate texture without generating phantom detections.
        cv2.rectangle(frame, (0, 456 + (n * 3) % 4), (WIDTH, 460 + (n * 3) % 4),
                      (120, 120, 118), -1)

        for plate, province, start, x0, speed, y, scale, blur in vehicles:
            if n < start:
                continue
            x = int(x0 + (n - start) * speed)
            box = draw_plate(frame, x, y, plate, province, scale, blur)
            if box:
                truth[plate]["frames"].append({"frame": n, "box": box})

        writer.write(frame)

    writer.release()

    meta = {
        "scenario": args.scenario,
        "fps": FPS,
        "size": [WIDTH, HEIGHT],
        "frames": args.frames,
        "vehicles": [
            {
                "plate": v["plate"],
                "province": v["province"],
                "visible_frames": len(v["frames"]),
                "first_frame": v["frames"][0]["frame"] if v["frames"] else None,
                "last_frame": v["frames"][-1]["frame"] if v["frames"] else None,
            }
            for v in truth.values()
        ],
        "expected_tracks": sum(1 for v in truth.values() if v["frames"]),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out} ({args.frames} frames @ {FPS}fps)")
    for v in meta["vehicles"]:
        print(f"  {v['plate']:8} visible in {v['visible_frames']:3} frames "
              f"({v['first_frame']}-{v['last_frame']})")
    print(f"expected distinct tracks: {meta['expected_tracks']}")


if __name__ == "__main__":
    main()
