"""Stage 02 change A: exactly one [SCALE-LOG] line per Frame.update_pose call,
on every exit path (frame.py, Frame.update_pose)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import frame as frame_mod  # noqa: E402
from frame import Frame  # noqa: E402
from tracking import MotionEstimate  # noqa: E402

SCALE_LOG_RE = re.compile(
    r"^\s*>> \[SCALE-LOG\] frame=(\d+) model=(NONE|HOMOGRAPHY|ESSENTIAL_MATRIX) "
    r"norm_t=(\S+) scale=(\S+) step=(\S+) reason=(\S+)$"
)


def _frames():
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    prev = Frame(img, K, "ORB", "BF")
    curr = Frame(img, K, "ORB", "BF")
    T_prev = np.eye(4, dtype=np.float64)
    prev.set_world_pose(T_prev)
    return prev, curr


def _scale_log_lines(capsys):
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "[SCALE-LOG]" in l]
    return lines


def test_vo_fail_logs_none_nan(monkeypatch, capsys):
    prev, curr = _frames()
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: None)
    curr.update_pose(prev, world_map=None)
    lines = _scale_log_lines(capsys)
    assert len(lines) == 1
    m = SCALE_LOG_RE.match(lines[0])
    assert m is not None, lines[0]
    frame_id, model, norm_t, scale, step, reason = m.groups()
    assert model == "NONE"
    assert norm_t == "nan"
    assert scale == "nan"
    assert step == "nan"
    assert reason == "vo_fail"


def test_homography_logs_unit_scale(monkeypatch, capsys):
    prev, curr = _frames()
    t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    motion = MotionEstimate(
        R=np.eye(3, dtype=np.float64), t=t, inlier_matches=[], all_matches=[],
        model="HOMOGRAPHY", inliers_E=0, inliers_H=10,
        epipolar_err_before=0.0, epipolar_err_after=0.0,
    )
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: motion)
    curr.update_pose(prev, world_map=None)
    lines = _scale_log_lines(capsys)
    assert len(lines) == 1
    m = SCALE_LOG_RE.match(lines[0])
    assert m is not None, lines[0]
    _, model, norm_t, scale, step, reason = m.groups()
    assert model == "HOMOGRAPHY"
    assert abs(float(norm_t) - float(np.linalg.norm(t))) < 1e-6
    assert scale == "1.000000"
    assert abs(float(step) - float(np.linalg.norm(t))) < 1e-6
    assert reason == "homography"


def test_essential_switch_off_logs_unit_disabled(monkeypatch, capsys):
    prev, curr = _frames()
    t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    motion = MotionEstimate(
        R=np.eye(3, dtype=np.float64), t=t, inlier_matches=[], all_matches=[],
        model="ESSENTIAL_MATRIX", inliers_E=10, inliers_H=0,
        epipolar_err_before=0.0, epipolar_err_after=0.0,
    )
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: motion)
    curr.update_pose(prev, world_map=None, use_relative_scale=False)
    lines = _scale_log_lines(capsys)
    assert len(lines) == 1
    m = SCALE_LOG_RE.match(lines[0])
    assert m is not None, lines[0]
    _, model, norm_t, scale, step, reason = m.groups()
    assert model == "ESSENTIAL_MATRIX"
    assert scale == "1.000000"
    assert reason == "unit_disabled"


def test_essential_switch_on_no_map_logs_fallback_token(monkeypatch, capsys):
    """Switch on, but world_map is None: the estimator's own fallback path
    fires, so the reported reason must be a genuine fallback token, not the
    unit_disabled default (which only applies when the switch is off)."""
    prev, curr = _frames()
    t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    motion = MotionEstimate(
        R=np.eye(3, dtype=np.float64), t=t, inlier_matches=[], all_matches=[],
        model="ESSENTIAL_MATRIX", inliers_E=10, inliers_H=0,
        epipolar_err_before=0.0, epipolar_err_after=0.0,
    )
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: motion)
    curr.update_pose(prev, world_map=None, use_relative_scale=True)
    lines = _scale_log_lines(capsys)
    assert len(lines) == 1
    m = SCALE_LOG_RE.match(lines[0])
    assert m is not None, lines[0]
    _, model, norm_t, scale, step, reason = m.groups()
    assert model == "ESSENTIAL_MATRIX"
    assert reason != "unit_disabled"
    assert reason.startswith("no_map_or_matches")
    # Pose composition must not have used the log-only fallback scale value:
    # with no map, the actual chained step still equals the unit translation.
    assert abs(float(step) - float(np.linalg.norm(t))) < 1e-6


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
