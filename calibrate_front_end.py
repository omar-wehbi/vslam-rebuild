#!/usr/bin/env python3
"""Stage 04j (F2): reproduce the frozen SIFT front-end protocol of front_end.PROTOCOL on the
primary dataset. Reads the four calibration pairs of front_end.CALIBRATION_PAIRS from
<root>/rgb.txt, runs front_end.calibrate on them, and prints one JSON object on stdout."""

import argparse
import json
import sys
from pathlib import Path

import cv2

import front_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    rgb_txt = root / "rgb.txt"
    if not rgb_txt.exists():
        print(f"dataset not available: {rgb_txt} does not exist", file=sys.stderr)
        sys.exit(2)

    names = [
        line.split()[1]
        for line in rgb_txt.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    pairs = []
    for i, j in front_end.CALIBRATION_PAIRS:
        img_a = cv2.imread(str(root / names[i]), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(root / names[j]), cv2.IMREAD_GRAYSCALE)
        pairs.append((img_a, img_b))

    out = front_end.calibrate(pairs)
    print(json.dumps(out, sort_keys=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
