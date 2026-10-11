#!/usr/bin/env python3
"""
Sanity check for Bag-of-Words loop closure: load ORBvoc.txt and run one add + query.

Usage:
  python scripts/test_orbvoc_load.py [path/to/ORBvoc.txt]

If no path is given, uses ORBvoc.txt in the project root or in Vocabulary/
(obtain from ORB-SLAM2: https://github.com/raulmur/ORB_SLAM2, possibly via git lfs pull).

Requires: pip install pyDBoW3, and OpenCV (for ORB descriptors).
"""

import sys
from pathlib import Path

def main():
    try:
        import pyDBoW3 as bow
    except ImportError:
        print("pyDBoW3 is not installed. Install with: pip install pyDBoW3")
        sys.exit(1)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV (cv2) is required for generating ORB descriptors.")
        sys.exit(1)

    base = Path(__file__).resolve().parent.parent
    if len(sys.argv) >= 2:
        vocab_path = Path(sys.argv[1])
    else:
        for candidate in [base / "ORBvoc.txt", base / "Vocabulary" / "ORBvoc.txt"]:
            if candidate.is_file():
                vocab_path = candidate
                break
        else:
            print("Usage: python scripts/test_orbvoc_load.py [path/to/ORBvoc.txt]")
            print("No ORBvoc.txt found in project root or Vocabulary/. Download from ORB-SLAM2 repo.")
            sys.exit(1)

    if not vocab_path.is_file():
        print(f"File not found: {vocab_path}")
        sys.exit(1)

    print(f"Loading vocabulary from {vocab_path} ...")
    try:
        voc = bow.Vocabulary()
        voc.load(str(vocab_path))
    except Exception as e:
        print(f"Failed to load vocabulary: {e}")
        sys.exit(1)
    print("Vocabulary loaded.")

    db = bow.Database()
    db.setVocabulary(voc)

    # Synthetic ORB-like descriptors (256-dim uint8) for add + query test.
    np.random.seed(42)
    dummy_descs = np.random.randint(0, 256, (100, 256), dtype=np.uint8)
    db.add(dummy_descs)
    print("Added 100 dummy ORB-like descriptors to database.")

    try:
        results = db.query(dummy_descs, 1)
    except TypeError:
        results = db.query(dummy_descs)
        if hasattr(results, "__len__"):
            results = results[:1] if results else []
        else:
            results = [results] if results is not None else []
    print(f"Query returned: {results!r}")

    print("Sanity check passed: ORBvoc load + add + query OK.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
