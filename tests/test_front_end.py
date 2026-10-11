"""
Stage 04j (spec T): the SIFT front end -- expand_argv/preset_argv wiring into main.py's
parser, the frozen PROTOCOL table, the pure calibration-protocol functions on hand-built
arrays, keypoints_match.match_frames's default ratio, and the no-leak rule. No dataset,
no image file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import front_end as fe  # noqa: E402
import keypoints_match as km  # noqa: E402
import main as m  # noqa: E402


# ---------------------------------------------------------------------------
# T1: expand_argv and the parser
# ---------------------------------------------------------------------------

def test_expand_argv_no_front_end_unchanged():
    argv = ["--dataset", "rgb", "--undistort"]
    assert fe.expand_argv(argv) == argv


def test_expand_argv_sift_at_start():
    argv = ["--front_end", "sift", "--dataset", "rgb"]
    expanded = fe.expand_argv(argv)
    assert expanded == fe.preset_argv("sift") + argv


def test_expand_argv_sift_in_middle():
    argv = ["--dataset", "rgb", "--front_end", "sift", "--undistort"]
    expanded = fe.expand_argv(argv)
    assert expanded == fe.preset_argv("sift") + argv


def test_expand_argv_sift_at_end():
    argv = ["--dataset", "rgb", "--front_end", "sift"]
    expanded = fe.expand_argv(argv)
    assert expanded == fe.preset_argv("sift") + argv


def test_expand_argv_sift_equals_form():
    argv = ["--dataset", "rgb", "--front_end=sift"]
    expanded = fe.expand_argv(argv)
    assert expanded == fe.preset_argv("sift") + argv


def test_expand_argv_orb_preset_argv_unchanged_content():
    argv = ["--dataset", "rgb", "--front_end", "orb"]
    expanded = fe.expand_argv(argv)
    # orb expands to [] so content (order and values) is unchanged.
    assert expanded == argv


def test_expand_argv_two_occurrences_last_wins():
    argv = ["--front_end", "orb", "--dataset", "rgb", "--front_end", "sift"]
    expanded = fe.expand_argv(argv)
    assert expanded == fe.preset_argv("sift") + argv


def test_parser_sift_preset_resolves():
    args = m.get_parser().parse_args(
        fe.expand_argv(["--dataset", "rgb", "--front_end", "sift"])
    )
    S = fe.PROTOCOL["SIFT"]
    assert args.method == "SIFT"
    assert args.front_end == "sift"
    assert args.lm_max_desc_dist == S["desc_bound"]
    assert args.df_max_desc_dist == S["desc_bound"]
    assert args.lm_max_desc_dist == args.df_max_desc_dist
    assert args.relocalization_ratio_threshold == S["reloc_ratio"]


def test_parser_sift_preset_explicit_flag_wins():
    args = m.get_parser().parse_args(
        fe.expand_argv(
            ["--dataset", "rgb", "--front_end", "sift", "--lm_max_desc_dist", "150"]
        )
    )
    assert args.lm_max_desc_dist == 150.0


def test_parser_default_is_orb():
    args = m.get_parser().parse_args(["--dataset", "rgb"])
    assert args.front_end == "orb"
    assert args.method == "ORB"
    assert args.lm_max_desc_dist == 50.0
    assert args.df_max_desc_dist == 50.0
    assert args.relocalization_ratio_threshold == 0.85


def test_front_end_other_raises_systemexit():
    with pytest.raises(SystemExit):
        m.get_parser().parse_args(["--dataset", "rgb", "--front_end", "other"])


# ---------------------------------------------------------------------------
# T2: PROTOCOL table
# ---------------------------------------------------------------------------

def test_protocol_orb_row():
    assert fe.PROTOCOL["ORB"] == {"desc_bound": 50.0, "track_ratio": 0.75, "reloc_ratio": 0.85}


def test_protocol_sift_row_bounds():
    S = fe.PROTOCOL["SIFT"]
    assert S["desc_bound"] == int(S["desc_bound"])
    assert 247 <= S["desc_bound"] <= 249
    assert 0.5 < S["track_ratio"] < 1.0
    assert 0.5 < S["reloc_ratio"] < 1.0


def test_track_ratio_function():
    assert fe.track_ratio("ORB") == 0.75
    assert fe.track_ratio("anything") == 0.75
    assert fe.track_ratio("SIFT") == fe.PROTOCOL["SIFT"]["track_ratio"]


# ---------------------------------------------------------------------------
# T3: pure protocol functions on hand-built arrays
# ---------------------------------------------------------------------------

def test_acceptance_at_bound_and_bound_at_rate_round_trip():
    rng = np.random.default_rng(0)
    d1 = rng.uniform(0, 100, size=500)
    q = fe.acceptance_at_bound(d1, 50.0)
    expected_q = float(np.mean(d1 <= 50.0))
    assert q == pytest.approx(expected_q)
    bound = fe.bound_at_rate(d1, q)
    expected_bound = float(np.quantile(d1, q))
    assert bound == pytest.approx(expected_bound)


def test_acceptance_at_ratio_counts_d2_zero_as_not_accepted():
    d1 = np.array([1.0, 2.0, 0.0, 3.0])
    d2 = np.array([2.0, 0.0, 0.0, 6.0])
    # entry 0: 1 < 0.75*2=1.5 -> accepted
    # entry 1: d2 == 0 -> not accepted regardless of d1
    # entry 2: d2 == 0 -> not accepted
    # entry 3: 3 < 0.75*6=4.5 -> accepted
    r = fe.acceptance_at_ratio(d1, d2, 0.75)
    assert r == pytest.approx(0.5)


def test_ratio_at_rate_ignores_d2_zero():
    d1 = np.array([1.0, 2.0, 3.0, 100.0])
    d2 = np.array([2.0, 4.0, 6.0, 0.0])
    # ratios: 0.5, 0.5, 0.5 (the d2=0 entry excluded)
    r = fe.ratio_at_rate(d1, d2, 0.5)
    assert r == pytest.approx(0.5)


def _textured_pair(rng, w=240, h=180, n_blobs=300, shift=(4, 0)):
    img_a = np.zeros((h, w), dtype=np.uint8)
    centers = rng.integers(10, min(w, h) - 10, size=(n_blobs, 2))
    for cx, cy in centers:
        cv2_pt = (int(cx), int(cy))
        img_a[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = rng.integers(
            120, 255, size=img_a[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].shape
        )
    dx, dy = shift
    img_b = np.zeros_like(img_a)
    ys, xs = np.nonzero(img_a)
    for y, x in zip(ys, xs):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            img_b[ny, nx] = img_a[y, x]
    return img_a, img_b


def test_calibrate_on_synthetic_pairs():
    rng = np.random.default_rng(42)
    pair1 = _textured_pair(rng, shift=(4, 0))
    pair2 = _textured_pair(rng, shift=(3, 2))
    out = fe.calibrate([pair1, pair2])
    for key in (
        "orb_rate_desc", "sift_desc_bound", "orb_rate_track", "sift_track_ratio",
        "orb_rate_reloc", "sift_reloc_ratio", "n_orb", "n_sift", "n_sift_d2_zero",
    ):
        assert key in out
    assert out["n_orb"] > 0
    assert out["n_sift"] > 0


# ---------------------------------------------------------------------------
# T4: match_frames default ratio
# ---------------------------------------------------------------------------

class _StubFrame:
    def __init__(self, descriptors, method, match_type="KNN"):
        self.Descriptors = descriptors
        self.method = method
        self.match_type = match_type


def test_match_frames_default_orb_matches_explicit_075():
    rng = np.random.default_rng(1)
    d1 = rng.integers(0, 256, size=(40, 32), dtype=np.uint8)
    d2 = rng.integers(0, 256, size=(40, 32), dtype=np.uint8)
    f1 = _StubFrame(d1, "ORB")
    f2 = _StubFrame(d2, "ORB")
    default_matches = km.match_frames(f1, f2, "ORB")
    explicit_matches = km.match_frames(f1, f2, "ORB", ratio_threshold=0.75)
    assert len(default_matches) == len(explicit_matches)
    for a, b in zip(default_matches, explicit_matches):
        assert a.queryIdx == b.queryIdx
        assert a.trainIdx == b.trainIdx
        assert a.distance == b.distance


def test_match_frames_default_sift_matches_explicit_track_ratio():
    rng = np.random.default_rng(2)
    d1 = rng.uniform(0, 1, size=(40, 128)).astype(np.float32)
    d2 = rng.uniform(0, 1, size=(40, 128)).astype(np.float32)
    f1 = _StubFrame(d1, "SIFT")
    f2 = _StubFrame(d2, "SIFT")
    default_matches = km.match_frames(f1, f2, "SIFT")
    explicit_matches = km.match_frames(f1, f2, "SIFT", ratio_threshold=fe.track_ratio("SIFT"))
    assert len(default_matches) == len(explicit_matches)
    for a, b in zip(default_matches, explicit_matches):
        assert a.queryIdx == b.queryIdx
        assert a.trainIdx == b.trainIdx
        assert a.distance == b.distance


# ---------------------------------------------------------------------------
# T5: no-leak
# ---------------------------------------------------------------------------

def test_front_end_no_leak_tokens():
    src = Path(ROOT / "front_end.py").read_text()
    for tok in (
        "groundtruth", "depth.txt", "import dataset", "from dataset",
        "import pipeline", "from pipeline", "world_map",
    ):
        assert tok not in src, f"front_end.py must not contain {tok!r}"


def test_keypoints_match_imports_front_end_not_pipeline():
    src = Path(ROOT / "keypoints_match.py").read_text()
    assert "front_end" in src
    assert "import pipeline" not in src
    assert "from pipeline" not in src
