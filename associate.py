#!/usr/bin/env python3
"""
Associate RGB frame timestamps with ground-truth poses (TUM-style).

Reads rgb.txt (timestamp + filename) and groundtruth.txt (timestamp + tx ty tz qx qy qz qw),
finds the nearest ground-truth pose for each RGB timestamp within max_difference seconds,
and writes associations.txt: one line per frame with
  rgb_timestamp rgb_filename gt_timestamp tx ty tz qx qy qz qw

Usage:
  python associate.py rgb.txt groundtruth.txt -o associations.txt
  python associate.py rgb.txt groundtruth.txt --max_difference 0.02

Based on TUM RGB-D benchmark associate.py (Juergen Sturm, TUM).
"""

import argparse
import bisect
import sys


def read_file_list(filename):
    """
    Read a TUM-style file: lines "stamp d1 d2 d3 ..." (comment lines # skipped).
    Returns dict of (stamp -> list of remaining string values).
    """
    result = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) < 2:
                continue
            try:
                stamp = float(parts[0])
            except ValueError:
                continue
            result[stamp] = parts[1:]
    return result


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dicts of (stamp -> data) by nearest timestamp.
    For each stamp in first_list, find the stamp in second_list that is closest
    (within max_difference) to (stamp - offset).
    Returns list of (rgb_ts, rgb_data, gt_ts, gt_data) sorted by rgb_ts.
    """
    first_keys = sorted(first_list.keys())
    second_keys = sorted(second_list.keys())
    if not second_keys:
        return []

    matches = []
    for a in first_keys:
        target = a - offset
        idx = bisect.bisect_left(second_keys, target)
        candidates = []
        if idx > 0:
            candidates.append(second_keys[idx - 1])
        if idx < len(second_keys):
            candidates.append(second_keys[idx])
        best_b = None
        best_diff = float("inf")
        for b in candidates:
            diff = abs(b - target)
            if diff < best_diff:
                best_diff = diff
                best_b = b
        if best_b is not None and best_diff <= max_difference:
            matches.append((a, first_list[a], best_b, second_list[best_b]))
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Associate rgb.txt with groundtruth.txt by timestamp (TUM-style)."
    )
    parser.add_argument(
        "first_file",
        help="First file (e.g. rgb.txt): lines 'timestamp path_or_data...'",
    )
    parser.add_argument(
        "second_file",
        help="Second file (e.g. groundtruth.txt): lines 'timestamp tx ty tz qx qy qz qw'",
    )
    parser.add_argument(
        "-o", "--output",
        default="associations.txt",
        help="Output association file (default: associations.txt)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Time offset added to timestamps of the second file (default: 0.0)",
    )
    parser.add_argument(
        "--max_difference",
        type=float,
        default=0.02,
        help="Max allowed time difference for matching in seconds (default: 0.02)",
    )
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)
    if not first_list:
        print("Error: no entries in first file", file=sys.stderr)
        sys.exit(1)
    if not second_list:
        print("Error: no entries in second file", file=sys.stderr)
        sys.exit(1)

    matches = associate(first_list, second_list, args.offset, args.max_difference)
    if not matches:
        print("Error: no associations found (try --max_difference)", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w") as f:
        f.write("# rgb_timestamp rgb_filename gt_timestamp tx ty tz qx qy qz qw\n")
        for rgb_ts, rgb_data, gt_ts, gt_data in matches:
            rgb_path = " ".join(rgb_data)
            gt_vals = " ".join(gt_data)
            f.write(f"{rgb_ts} {rgb_path} {gt_ts} {gt_vals}\n")

    print(f"Wrote {len(matches)} associations to {args.output}")


if __name__ == "__main__":
    main()
