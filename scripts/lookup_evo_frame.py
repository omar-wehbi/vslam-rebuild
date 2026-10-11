#!/usr/bin/env python3
"""
Look up between evo_ape plot x-axis (relative time) and image frame index.

Usage:
  # x on plot -> frame index (e.g. spike at x=50, which frame?)
  python scripts/lookup_evo_frame.py <evo_frame_index.txt> <x_value>

  # frame index -> x on plot (e.g. frame 423 appears at which x?)
  python scripts/lookup_evo_frame.py <evo_frame_index.txt> --frame <frame_index>

Examples:
  python scripts/lookup_evo_frame.py e_evo_frame_index.txt 50
  python scripts/lookup_evo_frame.py e_evo_frame_index.txt -f 423

The evo_frame_index file is written when you run the pipeline with
--save_estimated, --ground_truth, and timestamps. It is named
<your_estimate_stem>_evo_frame_index.txt next to your trajectory file.
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Look up between evo_ape plot x-axis and frame index."
    )
    parser.add_argument(
        "evo_frame_index_file",
        type=str,
        help="Path to *_evo_frame_index.txt",
    )
    parser.add_argument(
        "value",
        type=float,
        help="x-axis value (relative time in sec) for x->frame; or frame index for --frame",
    )
    parser.add_argument(
        "--frame", "-f",
        action="store_true",
        help="Reverse lookup: value is frame index, output is x (relative time) on the plot",
    )
    args = parser.parse_args()

    data = []
    with open(args.evo_frame_index_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    frame_idx = int(parts[1])
                    data.append((t, frame_idx))
                except ValueError:
                    continue

    if not data:
        print("No data rows in file.", file=sys.stderr)
        sys.exit(1)

    if args.frame:
        # Reverse: frame index -> x (relative time) on plot
        frame_idx = int(args.value)
        matches = [(t, fi) for t, fi in data if fi == frame_idx]
        if not matches:
            # Nearest frame that exists
            frames_seen = sorted(set(fi for _, fi in data))
            nearest = min(frames_seen, key=lambda fi: abs(fi - frame_idx))
            matches = [(t, fi) for t, fi in data if fi == nearest]
            print(f"Frame {frame_idx} not in file; nearest frame index is {nearest}:", file=sys.stderr)
        rel_times = [t for t, _ in matches]
        x_min, x_max = min(rel_times), max(rel_times)
        if x_min == x_max:
            print(f"Frame {frame_idx} appears at x = {x_min:.3f} on the plot")
            print(x_min)
        else:
            print(f"Frame {frame_idx} appears at x in [{x_min:.3f}, {x_max:.3f}] on the plot")
            print(f"{x_min} {x_max}")
    else:
        # x on plot -> frame index
        best = min(data, key=lambda row: abs(row[0] - args.value))
        rel_t, frame_idx = best
        print(f"At x = {args.value}: nearest row has relative_time = {rel_t:.3f} -> frame index = {frame_idx}")
        print(frame_idx)


if __name__ == "__main__":
    main()
