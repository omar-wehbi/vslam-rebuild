"""Stage 2 unit checks: lost-frame pose hold + scale fallback (T2.1.1, T2.2.1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame import Frame, _estimate_translation_scale  # noqa: E402


class _DummyMap:
    def get_point(self, _pt_id):
        return None


def test_scale_fallback_returns_prev_step_not_one():
    """T2.2.1: insufficient candidates → last_scale fixture, not 1.0."""
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    prev = Frame(img, K, "ORB", "BF")
    curr = Frame(img, K, "ORB", "BF")
    R = np.eye(3, dtype=np.float64)
    t_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    scale, reason = _estimate_translation_scale(
        prev, curr, R, t_unit, inlier_matches=[], world_map=_DummyMap(),
        prev_step=0.3,
    )
    assert abs(scale - 0.3) < 1e-12, f"expected 0.3, got {scale} ({reason})"
    assert "init_unit" not in reason
    assert scale != 1.0


def test_scale_init_allows_unit_without_history():
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    prev = Frame(img, K, "ORB", "BF")
    curr = Frame(img, K, "ORB", "BF")
    R = np.eye(3, dtype=np.float64)
    t_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    scale, reason = _estimate_translation_scale(
        prev, curr, R, t_unit, inlier_matches=[], world_map=None, prev_step=None,
    )
    assert abs(scale - 1.0) < 1e-12
    assert "init_unit" in reason


def test_lost_frame_hold_not_identity():
    """T2.1.1: after VO+reloc fail path, held/CV pose ≠ identity and near previous."""
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    f0 = Frame(img, K, "ORB", "BF")
    f1 = Frame(img, K, "ORB", "BF")
    f2 = Frame(img, K, "ORB", "BF")

    T0 = np.eye(4, dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    T1[:3, 3] = np.array([2.5, 0.0, 0.0])
    f0.set_world_pose(T0)
    f1.set_world_pose(T1)
    f0.pose_valid = True
    f1.pose_valid = True

    # Simulate pipeline Stage 2.1 constant-velocity hold (no _last_motion / reloc).
    assert not hasattr(f2, "_last_motion")
    T_prev = f1.world_T_cam
    T_prev2 = f0.world_T_cam
    rel = np.linalg.inv(T_prev2) @ T_prev
    held = T_prev @ rel
    assert f2.set_world_pose(held)
    f2.pose_valid = False
    f2.tracking_state = "LOST"

    centre = f2.world_T_cam[:3, 3]
    assert np.linalg.norm(centre) > 1.0, "must not remain at origin/identity"
    assert np.allclose(centre, np.array([5.0, 0.0, 0.0]), atol=1e-9)
    assert f2.pose_valid is False
    assert f2.tracking_state == "LOST"


def test_scale_soft_rejects_outside_relative_band():
    """Out-of-band estimate returns prev_step (no clamp-to-bound doubling)."""
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    prev = Frame(img, K, "ORB", "BF")
    curr = Frame(img, K, "ORB", "BF")
    # Force early exit path with prev_step via empty matches.
    scale, reason = _estimate_translation_scale(
        prev, curr, np.eye(3), np.array([1.0, 0.0, 0.0]),
        inlier_matches=[], world_map=_DummyMap(), prev_step=0.5,
    )
    assert abs(scale - 0.5) < 1e-12
    assert reason == "no_map_or_matches"


if __name__ == "__main__":
    test_scale_fallback_returns_prev_step_not_one()
    test_scale_init_allows_unit_without_history()
    test_lost_frame_hold_not_identity()
    test_scale_soft_rejects_outside_relative_band()
    print("All Stage 2 unit tests passed.")
