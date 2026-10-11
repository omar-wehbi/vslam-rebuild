"""Stage 03a (spec E1): the online-stream instrumentation helpers in pipeline.py, pinned
without running the pipeline: pose_delta_stats, the [POSE-ONLINE] / [CORRECTION] line
formats, correction_stats, and the pose_of keyword of _write_estimated_trajectory_tum."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402
from pipeline import (  # noqa: E402
    _write_estimated_trajectory_tum,
    format_correction_line,
    format_pose_online_line,
    pose_delta_stats,
)
from pose_log import correction_stats, print_correction  # noqa: E402

POSE_ONLINE_RE = re.compile(
    r"^\s*>> \[POSE-ONLINE\] frame=(\d+) source=(init|vo_essential|vo_homography|pnp|reloc|"
    r"hold_prev|constant_velocity|hold_prev_fallback) state=(\S+) rel_rot_deg=(\S+) step=(\S+) "
    r"pnp=([01]) ba=(none|skipped|accepted|rejected|error) lc=([01])$"
)
CORRECTION_RE = re.compile(
    r"^\s*>> \[CORRECTION\] frame=(\d+) kind=(pnp_distributed|ba|loop_closure) "
    r"frames_touched=(\d+) first=(-?\d+) last=(-?\d+) max_rot_deg=(\S+) max_trans=(\S+)$"
)


def _rot_x(deg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", deg, degrees=True).as_matrix()
    return T


def _pose(rvec_deg, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.deg2rad(np.asarray(rvec_deg, float))).as_matrix()
    T[:3, 3] = t
    return T


# --- pose_delta_stats -----------------------------------------------------------------

def test_pose_delta_stats_identity():
    T = _pose([10.0, -20.0, 5.0], [1.0, 2.0, 3.0])
    rot, trans = pose_delta_stats(T, T)
    assert rot == 0.0 and trans == 0.0


def test_pose_delta_stats_rotx90_offset2():
    T_a = _pose([10.0, -20.0, 5.0], [1.0, 2.0, 3.0])
    T_b = T_a @ _rot_x(90.0)
    T_b[:3, 3] = T_a[:3, 3] + np.array([0.0, 0.0, 2.0])
    rot, trans = pose_delta_stats(T_a, T_b)
    assert abs(rot - 90.0) <= 1e-6
    assert abs(trans - 2.0) <= 1e-9


def test_pose_delta_stats_clips_trace_roundoff():
    # A rotation whose trace overshoots 3 by round-off must give 0, not nan.
    T = np.eye(4)
    T_b = T.copy()
    T_b[:3, :3] = np.eye(3) * (1.0 + 1e-15)
    rot, _ = pose_delta_stats(T, T_b)
    assert np.isfinite(rot) and rot == 0.0


# --- line formats ---------------------------------------------------------------------

def test_pose_online_line_format():
    line = format_pose_online_line(17, "vo_essential", "OK", 0.5, 1.25, 0, "none", 0)
    m = POSE_ONLINE_RE.match(line)
    assert m is not None, line
    assert m.groups() == ("17", "vo_essential", "OK", "0.500000", "1.250000", "0", "none", "0")
    line = format_pose_online_line(0, "init", "OK", 0.0, 0.0, False, "none", False)
    assert POSE_ONLINE_RE.match(line) is not None, line
    line = format_pose_online_line(230, "pnp", "RELOCALIZED", 12.3456789, 0.1, True, "accepted", True)
    m = POSE_ONLINE_RE.match(line)
    assert m is not None, line
    assert m.group(4) == "12.345679" and m.group(6) == "1" and m.group(8) == "1"
    # Every source / ba token the pipeline can emit matches the regex.
    for src in ("init", "vo_essential", "vo_homography", "pnp", "reloc", "hold_prev",
                "constant_velocity", "hold_prev_fallback"):
        for ba in ("none", "skipped", "accepted", "rejected", "error"):
            assert POSE_ONLINE_RE.match(format_pose_online_line(3, src, "LOST", 1.0, 2.0, 1, ba, 0))


def test_correction_line_format():
    # frame= is the trajectory index k (Frame.id - 1); first=/last= are Frame.ids, so a
    # correction at k=40 (Frame.id 41) over the previous nine frames reads 32..40.
    line = format_correction_line(40, "pnp_distributed", 9, 32, 40, 1.303084, 1.007068)
    m = CORRECTION_RE.match(line)
    assert m is not None, line
    assert m.groups() == ("40", "pnp_distributed", "9", "32", "40", "1.303084", "1.007068")
    line = format_correction_line(60, "ba", 0, -1, -1, 0.0, 0.0)
    m = CORRECTION_RE.match(line)
    assert m is not None, line
    assert m.group(4) == "-1" and m.group(5) == "-1"
    assert CORRECTION_RE.match(format_correction_line(7, "loop_closure", 3, 2, 6, 0.1, 0.2))


# --- correction_stats -------------------------------------------------------------------

class _F:
    def __init__(self, fid, T):
        self.id = fid
        self.world_T_cam = T


def test_correction_stats_counts_changed_frames_excluding_current():
    frames = [_F(i, _pose([0, 0, 0], [i, 0, 0])) for i in range(5)]
    before = {f.id: f.world_T_cam.copy() for f in frames}
    # Frames 1 and 3 change (3 by a 30 deg turn and a 0.5 shift); frame 4 is "current" and
    # changes too but must be excluded; frames 0 and 2 stay put.
    frames[1].world_T_cam = _pose([0, 0, 0], [1.0, 0.25, 0])
    frames[3].world_T_cam = _pose([0, 30.0, 0], [3.5, 0, 0])
    frames[4].world_T_cam = _pose([0, 90.0, 0], [9, 9, 9])
    n, first, last, max_rot, max_trans = correction_stats(
        before, [(f, f.world_T_cam) for f in frames], exclude_id=4
    )
    assert (n, first, last) == (2, 1, 3)
    assert abs(max_rot - 30.0) < 1e-9
    assert abs(max_trans - 0.5) < 1e-12


def test_correction_stats_nothing_changed():
    frames = [_F(i, _pose([0, 0, 0], [i, 0, 0])) for i in range(3)]
    before = {f.id: f.world_T_cam.copy() for f in frames}
    frames[1].world_T_cam = before[1] + 1e-13  # below the 1e-12 tolerance
    assert correction_stats(before, [(f, f.world_T_cam) for f in frames], exclude_id=2) == (0, -1, -1, 0.0, 0.0)


def test_print_correction_prints_matching_line(capsys):
    frames = [_F(i, _pose([0, 0, 0], [i, 0, 0])) for i in range(3)]
    before = {f.id: f.world_T_cam.copy() for f in frames}
    frames[1].world_T_cam = _pose([0, 0, 0], [1.0, 1.0, 0])
    print_correction(2, "ba", before, [(f, f.world_T_cam) for f in frames], exclude_id=2)
    out = [l for l in capsys.readouterr().out.splitlines() if "[CORRECTION]" in l]
    assert len(out) == 1
    m = CORRECTION_RE.match(out[0])
    assert m is not None and m.group(3) == "1" and m.group(4) == "1" and m.group(5) == "1"


# --- _write_estimated_trajectory_tum with pose_of -----------------------------------------

class _StandIn:
    def __init__(self, fid, pose, online):
        self.id = fid
        self.pose = pose
        self.online_world_T_cam = online


def _rows(path):
    return [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def test_writer_pose_of_default_and_online(tmp_path):
    poses = [_pose([0, 0, 0], [0, 0, 0]), _pose([0, 5, 0], [1, 0, 0]), _pose([0, 10, 0], [2, 0, 0])]
    onlines = [_pose([0, 0, 0], [0, 0, 0]), _pose([0, 4, 0], [1.1, 0, 0]), _pose([0, 12, 1], [2, 0.3, 0])]
    frames = [_StandIn(i + 1, poses[i], onlines[i]) for i in range(3)]
    ts = [100.0, 100.1, 100.2]

    final_p = tmp_path / "estimated.txt"
    online_p = tmp_path / "estimated_online.txt"
    idx_final = _write_estimated_trajectory_tum(frames, ts, final_p)
    idx_online = _write_estimated_trajectory_tum(
        frames, ts, online_p,
        pose_of=lambda f: getattr(f, "online_world_T_cam", None) if getattr(f, "online_world_T_cam", None) is not None else f.pose,
    )
    rf, ro = _rows(final_p), _rows(online_p)
    assert len(rf) == 3 and len(ro) == 3
    for i in range(3):
        assert float(rf[i][0]) == ts[i] and float(ro[i][0]) == ts[i]
        exp_f = pipeline._pose_to_tum_line(poses[i]).split()
        exp_o = pipeline._pose_to_tum_line(onlines[i]).split()
        assert rf[i][1:] == exp_f
        assert ro[i][1:] == exp_o
    # The two streams genuinely differ where the snapshot differs from the final pose.
    assert rf[1][1:] != ro[1][1:] and rf[2][1:] != ro[2][1:]
    assert rf[0][1:] == ro[0][1:]
    # Companion index files exist and have one row per frame.
    assert Path(idx_final) == tmp_path / "estimated_frame_index.txt"
    assert Path(idx_online) == tmp_path / "estimated_online_frame_index.txt"
    assert len(_rows(idx_final)) == 3 and len(_rows(idx_online)) == 3


def test_writer_pose_of_falls_back_to_pose_when_no_snapshot(tmp_path):
    class _NoSnap:
        def __init__(self, fid, pose):
            self.id = fid
            self.pose = pose
    frames = [_NoSnap(1, _pose([0, 0, 0], [0, 0, 0])), _NoSnap(2, _pose([0, 5, 0], [1, 0, 0]))]
    p = tmp_path / "estimated_online.txt"
    _write_estimated_trajectory_tum(
        frames, None, p,
        pose_of=lambda f: getattr(f, "online_world_T_cam", None) if getattr(f, "online_world_T_cam", None) is not None else f.pose,
    )
    ro = _rows(p)
    assert len(ro) == 2
    assert ro[1][1:] == pipeline._pose_to_tum_line(frames[1].pose).split()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
