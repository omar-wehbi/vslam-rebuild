"""Stage 04v: the rotation-only state inside a map (map_rotation.py), its flag (main.py) and the
pipeline's label table and summary hook (pipeline.py).

M1 the --map_rot_state flag and its refusals; M2 run_pipeline's tracker class; M3 the label table;
M4 entry; M5 the relocalization fallback and its exit; M6 the exit by normal tracking; M7 the labels
and the unsupported fit inside a stint; M8 the refused entry; M9 the constant twist and the summary
counts; M10 the no-leak tokens and imports. A last test runs the real fit and the real tracking on a
synthetic perfect-map scene.
"""
from __future__ import annotations

import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import map_rotation as mr  # noqa: E402
import map_tracker as mt  # noqa: E402
import pipeline  # noqa: E402
import rotation_only as ro  # noqa: E402
from Map import Map  # noqa: E402
from frame import DEFAULT_K, Frame  # noqa: E402
from optimization import so3_exp  # noqa: E402

K = np.asarray(DEFAULT_K, dtype=np.float64)
W, H = 640, 480

M1_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "map", "--m1_map", "/tmp/m1.txt",
           "--m1_mode", "m1a", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba",
           "--no_point_fusion"]
PNP = ["--map_reloc", "pnp"]
ON = ["--map_rot_state", "on"]

P = r"^\s*>>\s+"
RX_ROT = re.compile(P + r"\[MAP-ROT\] frame=(\d+) event=(enter|stay|exit|refused) label=(bridged|rotation|lost|tracked) "
                    r"anchor=(-?\d+) age=(-?\d+) kf=(-?\d+) rot=(measured|unsupported) exit=(none|track|reloc) "
                    r"track_inliers=(-?\d+) twist=(\S+),(\S+),(\S+)\s*$")
RX_TRACK = re.compile(P + r"\[MAP-TRACK\] frame=(\d+) state=(tracked|lost) mode=(m1a|m1b) pred_rot=(twoview|cv) "
                      r"step_a=(-?\d+) step_a_inliers=(-?\d+) step_a_wide=([01]) step_b=(-?\d+) obs=(\d+) "
                      r"inliers=(\d+) rmse_px=(\S+)\s*$")
RX_KF = re.compile(P + r"\[ROT-KF\] frame=(\d+) prev=(-?\d+) n=(\d+)\s*$")
RX_SUM = re.compile(P + r"\[MAP-ROT-SUMMARY\] stints=(\d+) refused=(\d+) exits_track=(\d+) exits_reloc=(\d+) "
                    r"bridged=(\d+) rotation=(\d+) lost_in_state=(\d+) tracked=(\d+) lost=(\d+) map_points=(-?\d+)\s*$")


def _refused(argv):
    args = main.get_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.validate_tracking_args(args)
        return None
    except SystemExit as e:
        return str(e)


# --------------------------------------------------------------------------- M1
def test_m1_flag_default_choices_and_refusals():
    pr = main.get_parser()
    assert pr.parse_args(["--dataset", "rgb"]).map_rot_state == "off"
    assert pr.parse_args(M1_ARGV + PNP + ON).map_rot_state == "on"
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        pr.parse_args(["--dataset", "rgb", "--map_rot_state", "bridge"])
    assert _refused(M1_ARGV + PNP + ON) is None
    assert _refused(M1_ARGV + PNP + ["--map_rot_state", "off"]) is None
    only = "--map_rot_state on is only valid with --tracking map and --map_reloc pnp"
    assert _refused(M1_ARGV + ON) == only
    assert _refused(M1_ARGV + ["--map_reloc", "none"] + ON) == only
    assert _refused(["--dataset", "rgb"] + ON) == only
    rot = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "rotation", "--pnp_interval", "0",
           "--no_loop_closure", "--no_relocalization", "--no_ba", "--no_point_fusion"]
    assert _refused(rot + ON) == only
    assert _refused(M1_ARGV + PNP + ["--reloc_shadow"] + ON) == "--map_rot_state on refuses --reloc_shadow"
    for s in ("sqpnp", "newton"):
        assert _refused(M1_ARGV + PNP + ["--reloc_solver", s] + ON) == \
            "--map_rot_state on refuses --reloc_solver other than epnp"
        assert _refused(M1_ARGV + PNP + ["--reloc_solver", s]) is None      # switch off: 04q as it was
    assert _refused(M1_ARGV + PNP + ["--reloc_shadow"]) is None
    # B2 (iv): the existing rules stay (a missing 04l flag is still refused with the switch on)
    assert "--no_ba" in _refused([x for x in M1_ARGV if x != "--no_ba"] + PNP + ON)


def test_m1_rules_read_the_flag_with_a_default():
    """validate_tracking_args reads the flag with getattr: namespaces without it are the switch off."""
    args = main.get_parser().parse_args(M1_ARGV + PNP)
    del args.map_rot_state
    with contextlib.redirect_stdout(io.StringIO()):
        assert main.validate_tracking_args(args) is None


# --------------------------------------------------------------------------- M2
def test_m2_run_pipeline_builds_the_class_of_the_switch(monkeypatch):
    seen = {}

    class _SpyOff(mt.MapTracker):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")

    class _SpyOn(mr.MapRotationTracker):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(main, "MapTracker", _SpyOff)
    monkeypatch.setattr(mr, "MapRotationTracker", _SpyOn)
    monkeypatch.setattr(main, "load_m1_map", lambda path: mt.M1Map(header={}, points=[], corr={}))
    for val, cls in (("on", _SpyOn), ("off", _SpyOff)):
        args = main.get_parser().parse_args(M1_ARGV + PNP + ["--map_rot_state", val])
        with pytest.raises(SystemExit):
            main.run_pipeline(args)
        t = seen.pop("tracker")
        assert type(t) is cls
        assert t.reloc == "pnp" and t.reloc_solver == "epnp" and t.reloc_shadow is False
        assert t.reloc_params == mt.MapRelocParams()
    # a namespace without the attribute builds the 04m tracker
    args = main.get_parser().parse_args(M1_ARGV + PNP)
    del args.map_rot_state
    with pytest.raises(SystemExit):
        main.run_pipeline(args)
    assert type(seen.pop("tracker")) is _SpyOff


# --------------------------------------------------------------------------- M3
def test_m3_label_table():
    assert pipeline.M1_ONLINE_LABELS == {"tracked": ("map", "OK"), "lost": ("predicted", "LOST"),
                                         "bridged": ("bridged", "BRIDGED"), "rotation": ("rotation", "ROT")}


def test_m3_setup_and_fresh_summary_lines():
    empty = mt.M1Map(header={}, points=[], corr={})
    tr = mr.MapRotationTracker(empty, "m1a")
    assert tr.setup_line().split("\n") == [mt.MapTracker(empty, "m1a").setup_line(),
                                           "  >> [MAP-ROT-SETUP] horizon=30"]
    assert tr.summary_line() == ("  >> [MAP-ROT-SUMMARY] stints=0 refused=0 exits_track=0 exits_reloc=0 bridged=0 "
                                 "rotation=0 lost_in_state=0 tracked=0 lost=0 map_points=-1")
    assert mr.RotStateParams().horizon == 30 and tr.rot_params == mr.RotStateParams()
    assert tr.reloc == "none" and tr.reloc_solver == "epnp" and tr.reloc_shadow is False


# --------------------------------------------------------------------------- stub scene for M4-M9
def _sanitized(T):
    """The pose a frame stores after set_camera_pose (frame.Frame's SO(3) projection)."""
    return Frame._sanitize_pose(T)


class _F:
    """A frame stub whose set_camera_pose projects the rotation onto SO(3) as frame.Frame does (without
    it the rounding of R_step R_(k-1) would compound over a long stint)."""

    def __init__(self, fid, T, state):
        self.id = fid
        self.K = K
        self.keypoint_to_point = {}
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.tracking_state = state
        self.pose_valid = state == "OK"

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(_sanitized(T), dtype=np.float64, copy=True)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _stub_frames(n, n_tracked=3):
    """Frames 0..n_tracked-1 tracked with a constant body twist (rotation and translation), the rest
    unposed (identity)."""
    V = np.eye(4)
    V[:3, :3] = so3_exp([0.002, 0.03, -0.001])
    V[:3, 3] = [0.011, -0.002, 0.004]
    frames = []
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, -0.3]
    for k in range(n):
        if k < n_tracked:
            frames.append(_F(k + 1, T, "OK"))
            T = V @ T
        else:
            frames.append(_F(k + 1, np.eye(4), "LOST"))
    return frames


class _FakeMap:
    def __init__(self):
        self.points = [object()] * 7
        self.links = []

    def link(self, frame, kp, point):
        self.links.append((frame.id, kp, point))
        frame.keypoint_to_point[int(kp)] = id(point)
        return True


class _Script:
    """Stubs of the parent's on_frame, the segment's fit and normal tracking, scripted per frame."""

    def __init__(self, monkeypatch, tr, parent=None, fit=None, track=None):
        self.parent = parent or {}       # k -> "tracked" | "lost" (default lost)
        self.fit = fit or {}             # k -> rotation (3x3) or None for unsupported (default: a yaw step)
        self.track = track or {}         # k -> (T, n_inliers) for an accepted normal tracking
        self.parent_calls = []           # (k, frames[k-1].tracking_state at the call)
        self.fit_calls = []
        self.fit_R = {}                  # k -> the rotation block the segment wrote
        self.track_calls = []            # (k, T_pred)
        script = self

        def parent_on_frame(self_, k, frames, world_map):
            script.parent_calls.append((k, frames[k - 1].tracking_state))
            fr = frames[k]
            if script.parent.get(k, "lost") == "tracked":
                T = np.eye(4)
                T[:3, 3] = [k * 1.0, 0.0, 0.0]
                fr.set_camera_pose(T)
                fr.pose_valid, fr.tracking_state = True, "OK"
                print(f"  >> [MAP-TRACK] frame={k} state=tracked mode=m1a pred_rot=cv step_a=-1 step_a_inliers=-1 "
                      f"step_a_wide=0 step_b=0 obs=50 inliers=50 rmse_px=0.1000")
                return "tracked"
            T = np.eye(4)
            T[:3, :3] = so3_exp([0.0, 0.3, 0.0])
            T[:3, 3] = [-5.0, 1.0, 2.0]
            fr.set_camera_pose(T)
            fr.pose_valid, fr.tracking_state = False, "LOST"
            print(f"  >> [MAP-TRACK] frame={k} state=lost mode=m1a pred_rot=cv step_a=0 step_a_inliers=0 "
                  f"step_a_wide=1 step_b=0 obs=0 inliers=0 rmse_px=nan")
            return "lost"

        def seg_on_frame(self_, k, frames):
            script.fit_calls.append(k)
            fr = frames[k]
            R = script.fit.get(k, "default")
            if isinstance(R, str):
                R = so3_exp([0.001 * k, 0.04, 0.0]) @ np.asarray(frames[k - 1].cam_T_world)[:3, :3]
            T = np.eye(4)
            if R is None:
                T[:3, 3] = [9.0, 9.0, 9.0]           # whatever the segment writes is overwritten
                fr.set_camera_pose(T)
                fr.pose_valid, fr.tracking_state = False, "LOST"
                print(f"  >> [ROT-ONLY] frame={k} state=unsupported kf={self_.kf}")
                return "unsupported"
            T[:3, :3] = R
            T[:3, 3] = [7.0, 7.0, 7.0]
            fr.set_camera_pose(T)
            script.fit_R[k] = fr.cam_T_world[:3, :3].copy()
            fr.pose_valid, fr.tracking_state = True, "ROT"
            print(f"  >> [ROT-ONLY] frame={k} state=measured kf={self_.kf}")
            return "measured"

        def track_m1a(k, frames, T_pred):
            script.track_calls.append((k, np.array(T_pred, copy=True)))
            counts = dict(step_a=0, step_a_inliers=0, step_a_wide=1, step_b=3)
            if k in script.track:
                T, n = script.track[k]
                return T, list(range(n)), list(range(n)), np.ones(n, dtype=bool), counts
            return None, [0, 1, 2], [0, 1, 2], np.zeros(3, dtype=bool), counts

        monkeypatch.setattr(mt.MapTracker, "on_frame", parent_on_frame)
        monkeypatch.setattr(ro.RotationOnlyTracker, "on_frame", seg_on_frame)
        monkeypatch.setattr(tr, "_track_m1a", track_m1a)
        monkeypatch.setattr(tr, "_rmse", lambda fr, T, rows, kps, mask: 0.25)


def _tracker():
    tr = mr.MapRotationTracker(mt.M1Map(header={}, points=[], corr={}), "m1a", None)
    tr.reloc = "pnp"
    tr._rows_pt = [object() for _ in range(64)]
    return tr


def _run(tr, frames, wmap, ks):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        states = [tr.on_frame(k, frames, wmap) for k in ks]
    return states, buf.getvalue().splitlines()


def _rot_lines(out):
    return {int(m.group(1)): m for m in (RX_ROT.match(l) for l in out) if m}


# --------------------------------------------------------------------------- M4
def test_m4_entry_bridges_from_the_anchor(monkeypatch):
    frames = _stub_frames(6)
    tr = _tracker()
    R_known = so3_exp([0.01, 0.05, 0.02]) @ frames[2].cam_T_world[:3, :3]
    sc = _Script(monkeypatch, tr, fit={3: R_known})
    T2 = frames[2].cam_T_world.copy()
    T1 = frames[1].cam_T_world.copy()
    states, out = _run(tr, frames, _FakeMap(), [3])
    assert states == ["bridged"]
    want = mt.predict_pose(T2, T1, sc.fit_R[3] @ T2[:3, :3].T)
    assert np.array_equal(frames[3].cam_T_world, _sanitized(want))
    assert frames[3].pose_valid is True and frames[3].tracking_state == "ROT"
    kf = [RX_KF.match(l) for l in out if RX_KF.match(l)]
    assert [m.groups() for m in kf] == [("2", "-1", "1")]
    m = _rot_lines(out)[3]
    assert m.group(2, 3, 4, 5, 6, 7, 8, 9) == ("enter", "bridged", "2", "1", "2", "measured", "none", "-1")
    assert out[-1] == m.group(0)                          # the last line of the frame
    assert sc.parent_calls == [(3, "OK")] and sc.fit_calls == [3] and sc.track_calls == []
    # the twist is the entry step's: T_2 inv(T_1)
    tw = np.array([float(x) for x in m.group(10, 11, 12)])
    assert np.allclose(tw, (T2 @ np.linalg.inv(T1))[:3, 3], atol=1e-9)


def test_m4_entry_twist_unknown_when_e_minus_1_untracked(monkeypatch):
    frames = _stub_frames(6)
    frames[1].tracking_state = "LOST"
    tr = _tracker()
    R_known = so3_exp([0.0, 0.05, 0.0]) @ frames[2].cam_T_world[:3, :3]
    sc = _Script(monkeypatch, tr, fit={3: R_known})
    T2 = frames[2].cam_T_world.copy()
    states, out = _run(tr, frames, _FakeMap(), [3, 4])
    assert states == ["bridged", "bridged"]
    assert np.array_equal(frames[3].cam_T_world, _sanitized(mt.predict_pose(T2, None, sc.fit_R[3] @ T2[:3, :3].T)))
    # zero twist, and it stays zero on the next frame (T_prev2 = T_(k-2) there)
    lines = _rot_lines(out)
    for k in (3, 4):
        assert np.allclose([float(x) for x in lines[k].group(10, 11, 12)], 0.0, atol=1e-12)


def test_m4_entry_at_frame_1_uses_no_frame_minus_1(monkeypatch):
    frames = _stub_frames(4, n_tracked=1)
    tr = _tracker()
    R_known = so3_exp([0.0, 0.05, 0.0])
    sc = _Script(monkeypatch, tr, fit={1: R_known})
    T0 = frames[0].cam_T_world.copy()
    frames[3].cam_T_world = np.full((4, 4), 1e6)            # frames[-1] must never be read
    states, out = _run(tr, frames, _FakeMap(), [1])
    assert states == ["bridged"]
    assert np.array_equal(frames[1].cam_T_world, _sanitized(mt.predict_pose(T0, None, sc.fit_R[1] @ T0[:3, :3].T)))
    assert _rot_lines(out)[1].group(4) == "0"


def test_m4_no_entry_after_an_untracked_frame(monkeypatch):
    frames = _stub_frames(6)
    frames[2].tracking_state = "LOST"
    tr = _tracker()
    sc = _Script(monkeypatch, tr)
    states, out = _run(tr, frames, _FakeMap(), [3])
    assert states == ["lost"] and sc.fit_calls == [] and not _rot_lines(out)


# --------------------------------------------------------------------------- M5
def test_m5_relocalization_fallback_and_reloc_exit(monkeypatch):
    frames = _stub_frames(8)
    tr = _tracker()
    sc = _Script(monkeypatch, tr, parent={5: "tracked", 6: "tracked"})
    states, out = _run(tr, frames, _FakeMap(), [3, 4, 5, 6])
    assert states == ["bridged", "bridged", "tracked", "tracked"]
    # frame 4: normal tracking from the bridged pose failed, the parent ran with frame 3 seen LOST,
    # and frame 3's state was restored
    assert sc.parent_calls == [(3, "OK"), (4, "LOST"), (5, "LOST"), (6, "OK")]
    assert frames[4].tracking_state == "ROT"              # frame 4 stayed (parent lost), restored for frame 5
    assert [k for k, _T in sc.track_calls] == [4, 5]
    lines = _rot_lines(out)
    assert lines[4].group(2, 3, 8, 9) == ("stay", "bridged", "none", "0")
    assert lines[5].group(2, 3, 4, 5, 8, 9) == ("exit", "tracked", "2", "3", "reloc", "0")
    # the next frame goes through the parent with no fit
    assert sc.fit_calls == [3, 4, 5] and 6 not in lines
    # exactly one [MAP-TRACK] line per frame: on a failed (6) only the parent's line is printed
    for k in (3, 4, 5, 6):
        assert sum(1 for l in out if f"[MAP-TRACK] frame={k} " in l) == 1
    assert tr._anchor is None


def test_m5_state_restored_when_the_parent_raises(monkeypatch):
    frames = _stub_frames(6)
    tr = _tracker()
    _Script(monkeypatch, tr)
    _run(tr, frames, _FakeMap(), [3])

    def boom(self_, k, frames_, world_map):
        raise RuntimeError("parent boom")
    monkeypatch.setattr(mt.MapTracker, "on_frame", boom)
    states, out = _run(tr, frames, _FakeMap(), [4])
    assert frames[3].tracking_state == "ROT"
    assert states == ["lost"] and any("[MAP-ROT-ERROR] frame=4 error=RuntimeError: parent boom" in l for l in out)
    assert tr._anchor == 2                                 # the stint continues


# --------------------------------------------------------------------------- M6
def test_m6_exit_by_normal_tracking(monkeypatch):
    frames = _stub_frames(8)
    tr = _tracker()
    T_fin = np.eye(4)
    T_fin[:3, 3] = [0.5, 0.0, 0.0]
    sc = _Script(monkeypatch, tr, track={4: (T_fin, 40)})
    wmap = _FakeMap()
    states, out = _run(tr, frames, wmap, [3, 4])
    assert states == ["bridged", "tracked"]
    assert [k for k, _T in sc.track_calls] == [4]
    assert np.array_equal(frames[4].cam_T_world, T_fin)
    assert frames[4].pose_valid is True and frames[4].tracking_state == "OK"
    assert len(wmap.links) == 40 and all(fid == frames[4].id for fid, _kp, _pt in wmap.links)
    tl = [RX_TRACK.match(l) for l in out if "[MAP-TRACK] frame=4 " in l]
    assert len(tl) == 1 and tl[0].group(2, 4, 10) == ("tracked", "cv", "40")
    assert tl[0].group(5, 6, 7, 8, 9) == ("0", "0", "1", "3", "40")
    m = _rot_lines(out)[4]
    assert m.group(2, 3, 8, 9) == ("exit", "tracked", "track", "40")
    assert sc.parent_calls == [(3, "OK")]                  # the parent never ran on frame 4
    assert tr._anchor is None and tr.n_exits_track == 1


def test_m6_bridged_pose_seeds_normal_tracking(monkeypatch):
    frames = _stub_frames(8)
    tr = _tracker()
    sc = _Script(monkeypatch, tr)
    _run(tr, frames, _FakeMap(), [3])
    T3 = frames[3].cam_T_world.copy()
    T2 = frames[2].cam_T_world.copy()
    _run(tr, frames, _FakeMap(), [4])
    want = mt.predict_pose(T3, T2, sc.fit_R[4] @ T3[:3, :3].T)
    assert np.array_equal(sc.track_calls[0][1], want)
    assert np.array_equal(frames[4].cam_T_world, _sanitized(want))     # the stay frame keeps T_b


# --------------------------------------------------------------------------- M7
def test_m7_labels_at_the_horizon_and_unsupported_fits(monkeypatch):
    frames = _stub_frames(40)
    tr = _tracker()
    _Script(monkeypatch, tr, fit={10: None})
    ks = list(range(3, 36))
    snap = {}
    real_inside = tr._inside

    def spy(k, frames_, world_map):
        snap[k] = (frames_[k - 1].cam_T_world.copy(), frames_[k - 2].cam_T_world.copy())
        return real_inside(k, frames_, world_map)
    monkeypatch.setattr(tr, "_inside", spy)
    states, out = _run(tr, frames, _FakeMap(), ks)
    lines = _rot_lines(out)
    e = 2
    assert lines[e + 30].group(3) == "bridged" and states[ks.index(e + 30)] == "bridged"
    assert lines[e + 31].group(3) == "rotation" and states[ks.index(e + 31)] == "rotation"
    assert int(lines[e + 31].group(5)) == 31 and lines[e + 31].group(4) == "2"
    # the unsupported fit: lost, pose predict_pose(T_(k-1), T_(k-2), None), the stint continues
    assert states[ks.index(10)] == "lost" and lines[10].group(2, 3, 7) == ("stay", "lost", "unsupported")
    T1, T2 = snap[10]
    assert np.array_equal(frames[10].cam_T_world, _sanitized(mt.predict_pose(T1, T2, None)))
    assert frames[10].pose_valid is False and frames[10].tracking_state == "LOST"
    assert lines[11].group(2, 3, 4) == ("stay", "bridged", "2")
    assert frames[11].tracking_state == "ROT" and frames[11].pose_valid is True
    # every stint frame has exactly one [MAP-ROT] line, the last line printed for it
    assert sorted(lines) == ks


def test_m7_error_inside_a_stint_is_lost_and_continues(monkeypatch):
    frames = _stub_frames(8)
    tr = _tracker()
    sc = _Script(monkeypatch, tr)
    _run(tr, frames, _FakeMap(), [3])

    def boom(k, frames_, T_pred):
        raise RuntimeError("track boom")
    monkeypatch.setattr(tr, "_track_m1a", boom)
    T3, T2 = frames[3].cam_T_world.copy(), frames[2].cam_T_world.copy()
    states, out = _run(tr, frames, _FakeMap(), [4])
    assert states == ["lost"]
    assert np.array_equal(frames[4].cam_T_world, _sanitized(mt.predict_pose(T3, T2, sc.fit_R[4] @ T3[:3, :3].T)))
    assert frames[4].pose_valid is False and frames[4].tracking_state == "LOST"
    assert "  >> [MAP-ROT-ERROR] frame=4 error=RuntimeError: track boom" in out
    assert sum(1 for l in out if "[MAP-TRACK] frame=4 " in l) == 1
    assert _rot_lines(out)[4].group(2, 3, 7, 9) == ("stay", "lost", "unsupported", "-1")
    assert tr._anchor == 2


# --------------------------------------------------------------------------- M8
def test_m8_refused_entry(monkeypatch):
    frames = _stub_frames(8)
    tr = _tracker()
    sc = _Script(monkeypatch, tr, fit={3: None})
    captured = {}
    real_parent = mt.MapTracker.on_frame

    def parent(self_, k, frames_, world_map):
        s = real_parent(self_, k, frames_, world_map)
        captured[k] = (frames_[k].cam_T_world.copy(), frames_[k].world_T_cam.copy())
        return s
    monkeypatch.setattr(mt.MapTracker, "on_frame", parent)
    states, out = _run(tr, frames, _FakeMap(), [3, 4])
    assert states == ["lost", "lost"]
    assert np.array_equal(frames[3].cam_T_world, captured[3][0])
    assert np.array_equal(frames[3].world_T_cam, captured[3][1])
    assert frames[3].pose_valid is False and frames[3].tracking_state == "LOST"
    m = _rot_lines(out)[3]
    assert m.group(2, 3, 4, 5, 6, 7, 8, 9) == ("refused", "lost", "2", "1", "2", "unsupported", "none", "-1")
    # no stint: frame 4 goes through the parent alone (its predecessor is lost, no fit is run)
    assert sc.fit_calls == [3] and 4 not in _rot_lines(out) and tr._anchor is None
    assert tr.n_refused == 1 and tr.n_stints == 0


# --------------------------------------------------------------------------- M9
def test_m9_twist_constant_and_summary_counts(monkeypatch):
    frames = _stub_frames(50)
    tr = _tracker()
    T_fin = np.eye(4)
    sc = _Script(monkeypatch, tr, fit={8: None, 20: None, 41: None}, parent={44: "tracked"},
                 track={46: (T_fin, 35)})
    ks = list(range(3, 47))
    states, out = _run(tr, frames, _FakeMap(), ks)
    lines = _rot_lines(out)
    # stint 1: frames 3..44 (anchor 2, exit by relocalization at 44); stint 2: 45..46 (anchor 44,
    # whose predecessor 43 is not tracked: zero twist; exit by normal tracking at 46)
    stint1 = np.array([float(x) for x in lines[3].group(10, 11, 12)])
    assert not np.allclose(stint1, 0.0)
    assert np.allclose([float(x) for x in lines[45].group(10, 11, 12)], 0.0, atol=1e-12)
    first = None
    for k in sorted(lines):
        m = lines[k]
        if m.group(2) in ("enter", "stay"):
            tw = np.array([float(x) for x in m.group(10, 11, 12)])
            if m.group(2) == "enter":
                first = tw
            assert np.all(np.abs(tw - first) <= 1e-9), (k, tw, first)
    assert lines[3].group(2) == "enter" and lines[44].group(2, 8) == ("exit", "reloc")
    assert lines[45].group(2) == "enter" and lines[46].group(2, 8) == ("exit", "track")
    g = RX_SUM.match(tr.summary_line())
    assert g is not None
    ev = [m.group(2) for m in lines.values()]
    stint_lab = [m.group(3) for m in lines.values() if m.group(2) in ("enter", "stay")]
    want = (ev.count("enter"), ev.count("refused"),
            sum(1 for m in lines.values() if m.group(8) == "track"),
            sum(1 for m in lines.values() if m.group(8) == "reloc"),
            stint_lab.count("bridged"), stint_lab.count("rotation"), stint_lab.count("lost"),
            states.count("tracked"), states.count("lost"), 7)
    assert tuple(int(x) for x in g.groups()) == want
    assert want[:4] == (2, 0, 1, 1) and want[6] == 3
    assert sc.fit_calls == list(range(3, 47))


# --------------------------------------------------------------------------- M10
def test_m10_no_leak_tokens_and_imports():
    src = (ROOT / "map_rotation.py").read_text(encoding="utf-8")
    for tok in ("groundtruth", "gt_path", "ground_truth", "depth.txt", "depth/", "cv2.imread"):
        assert tok not in src, tok
    assert not re.search(r"['\"]depth['\"]", src)
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {x.name.split(".")[0] for x in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    assert {x for x in mods if x not in sys.stdlib_module_names} <= {"numpy", "map_tracker", "rotation_only"}
    for bad in ("oracle", "dataset", "pipeline", "npnp", "initializer", "opening"):
        assert bad not in mods
    own = set()
    for n in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            own |= {x.name.split(".")[0] for x in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            own.add(n.module.split(".")[0])
    assert "npnp" not in own and "initializer" not in own


# --------------------------------------------------------------------------- the real paths
class _Frame:
    """Frame-like stub (copied from tests/test_map_tracker.py, with match_type for match_frames):
    SIFT keypoints at the exact projections of the visible scene points."""

    def __init__(self, fid, cam_T_world, X, desc, pids):
        self.id = fid
        self.K = K
        self.method = "SIFT"
        self.match_type = "KNN"
        self.img_shape = (H, W)
        self.keypoint_to_point = {}
        self.cam_T_world = np.array(cam_T_world, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.pose_valid = True
        self.tracking_state = "OK"
        uv, z = mt._project(K, self.cam_T_world, X)
        vis = mt._in_image(uv, z, W, H)
        self.pid_of_kp = [int(p) for p in np.asarray(pids)[vis]]
        self.KeyPoints = []
        for u, v in uv[vis]:
            kp = cv2.KeyPoint(float(u), float(v), 3.0)
            kp.octave = 0
            self.KeyPoints.append(kp)
        self.Descriptors = np.asarray(desc, dtype=np.float32)[vis]

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _world(n_frames=6, step=0.01, seed=11):
    """Points 0..99 belong to keyframe 0, points 100..199 to keyframe 2. Camera moves +x by `step`."""
    rng = np.random.default_rng(seed)
    T_true = []
    for k in range(n_frames):
        T = np.eye(4)
        T[0, 3] = -step * k
        T_true.append(T)
    X = []
    for T in (T_true[0], T_true[2]):
        Tinv = np.linalg.inv(T)
        for _ in range(100):
            z = rng.uniform(1.5, 6.0)
            u, v = rng.uniform(40, 600), rng.uniform(40, 440)
            X.append(Tinv[:3, :3] @ (z * np.linalg.solve(K, [u, v, 1.0])) + Tinv[:3, 3])
    X = np.array(X)
    desc = rng.uniform(0, 100, size=(len(X), 128))
    pids = np.arange(len(X))
    frames = [_Frame(k + 1, T_true[k], X, desc, pids) for k in range(n_frames)]
    for f in frames[1:]:
        f.set_camera_pose(np.eye(4))
    return frames, X, T_true


def _m1map(frames, X):
    pts = []
    for kf, lo in ((0, 0), (2, 100)):
        fr = frames[kf]
        for kp, pid in enumerate(fr.pid_of_kp):
            if lo <= pid < lo + 100:
                u, v = fr.KeyPoints[kp].pt
                pts.append((pid, kf, kp, u, v, *X[pid]))
    pts.sort()
    header = dict(front_end="sift", method="SIFT", undistort="1", keyframe_interval="2", scale_s="1.0",
                  anchor_k="0", n_keyframes="2", n_points=str(len(pts)), n_corr="0")
    return mt.M1Map(header=header, points=pts, corr={})


@pytest.fixture
def no_twoview(monkeypatch):
    monkeypatch.setattr(mt.tracking, "estimate_motion", lambda a, b, use_homography=False: None)


def test_real_fit_enters_and_tracking_exits(no_twoview, monkeypatch):
    """Frame 2's map pose is forced to fail: the real rotation fit enters the state, the real
    tracking from the bridged pose at frame 3 exits it with links, and the map holds only the
    released points."""
    frames, X, T_true = _world()
    wmap = Map(points=[], frames=frames)
    tr = mr.MapRotationTracker(_m1map(frames, X), "m1a", None)
    tr.reloc = "pnp"
    real = tr._track_m1a
    seen = []

    def forced(k, frames_, T_pred):
        seen.append(k)
        if k == 2:
            return None, [], [], np.zeros(0, dtype=bool), dict(step_a=0, step_a_inliers=0, step_a_wide=1, step_b=0)
        return real(k, frames_, T_pred)
    monkeypatch.setattr(tr, "_track_m1a", forced)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.release(0, frames, wmap)
        states = []
        for k in range(1, 5):
            states.append(tr.on_frame(k, frames, wmap))
            tr.release(k, frames, wmap)
        summary = tr.summary_line()
    out = buf.getvalue().splitlines()
    assert states == ["tracked", "bridged", "tracked", "tracked"]
    lines = _rot_lines(out)
    assert lines[2].group(2, 3, 4, 7) == ("enter", "bridged", "1", "measured")
    assert lines[3].group(2, 3, 8) == ("exit", "tracked", "track")
    # the bridged frame 2 continues the 1 cm step of frames 0 and 1; its orientation is the fit's,
    # which absorbs the small parallax of the step as a sub-degree rotation
    assert np.allclose(frames[2].world_T_cam[:3, 3], np.linalg.inv(T_true[2])[:3, 3], atol=1e-3)
    assert ro.rotation_angle_deg(frames[2].cam_T_world[:3, :3] @ T_true[2][:3, :3].T) < 0.5
    assert np.allclose(frames[3].cam_T_world, T_true[3], atol=1e-6)
    # the bridged frame made no tracking link: its only links are keyframe 2's own released points
    assert frames[2].keypoint_to_point and all(frames[2].pid_of_kp[kp] >= 100 for kp in frames[2].keypoint_to_point)
    assert len(frames[3].keypoint_to_point) >= 30
    assert seen == [1, 2, 3, 4]
    for k in range(1, 5):
        assert sum(1 for l in out if f"[MAP-TRACK] frame={k} " in l) == 1
    assert "[MAP-ROT-ERROR]" not in buf.getvalue() and "[MAP-TRACK-ERROR]" not in buf.getvalue()
    g = RX_SUM.match(summary)
    released = sum(int(m.group(1)) for m in (re.search(r"\[M1-RELEASE\] frame=\d+ points=(\d+)", l) for l in out) if m)
    assert int(g.group(10)) == released == len(wmap.points)
    assert wmap.check_invariants(frames) == {k: 0 for k in wmap.check_invariants(frames)}


# =========================================================================== stage 04x: --map_rot_state r52
# X1 the flag; X2 run_pipeline's variant and the fresh tracker; X3 fov_h_deg, history_frames,
# detect_rotation; X4 bridge_window, fit_body_velocity; X5 the entry attempt; X6 inside a stint; X7 the
# lost prediction; X8 PanoramaSegment; X9 the anchor's depth f, the setup and summary lines.
import math  # noqa: E402
import types  # noqa: E402

R52 = ["--map_rot_state", "r52"]
GAMMA = math.degrees(2.0 * math.atan(640.0 / (2.0 * float(K[0, 0]))))
RX_ENTRY = re.compile(P + r"\[MAP-ROT-ENTRY\] frame=(\d+) anchor=(-?\d+) hist=(\d+) f=(\S+) gamma_deg=(\S+) kept=(\d+) "
                      r"r_max=(\S+) alpha_deg=(\S+) detected=([01]) fit=(measured|unsupported|none) "
                      r"result=(enter|refused_detect|refused_fit|error)\s*$")
RX_LOSTVEL = re.compile(P + r"\[MAP-LOST-VEL\] frame=(\d+) anchor=(-?\d+) window_first=(-?\d+) window_n=(\d+) "
                        r"fitted=([01]) vel=(\S+?),(\S+?),(\S+)\s*$")
RX_HOLD = re.compile(P + r"\[MAP-ROT-KF-HOLD\] frame=(\d+) kf=(-?\d+) r=(\S+)\s*$")
RX_R52SUM = re.compile(P + r"\[MAP-ROT-R52-SUMMARY\] attempts=(\d+) detected=(\d+) refused_detect=(\d+) "
                       r"refused_fit=(\d+) entered=(\d+) kf_holds=(\d+) lost_runs=(\d+) lost_runs_fitted=(\d+)\s*$")


def _Ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _wtc(R, c):
    """world_T_cam from the rotation R_wc and the centre c."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c
    return T


class _F52(_F):
    """The 04v frame stub with the image shape (the field of view reads it)."""

    def __init__(self, fid, T, state):
        super().__init__(fid, T, state)
        self.img_shape = (H, W)


C0 = np.array([0.3, -0.1, 1.2])


def _turn_frames(n, n_tracked, step_deg, drift=None):
    """Frames 0..n_tracked-1 tracked, frame j turned by step_deg * j about y around C0 (plus drift * j
    when given), the rest unposed."""
    frames = []
    for k in range(n):
        if k < n_tracked:
            c = C0 + (np.zeros(3) if drift is None else np.asarray(drift, dtype=float) * k)
            frames.append(_F52(k + 1, np.linalg.inv(_wtc(_Ry(step_deg * k), c)), "OK"))
        else:
            frames.append(_F52(k + 1, np.eye(4), "LOST"))
    return frames


class _Pt:
    def __init__(self, pid, X):
        self.id = pid
        self.x, self.y, self.z = (float(v) for v in X)


class _PtMap:
    """A map stub with point_by_id (C4 reads it) and link."""

    def __init__(self):
        self.points = []
        self.point_by_id = {}
        self.links = []

    def add(self, X):
        p = _Pt(1000 + len(self.points), X)
        self.points.append(p)
        self.point_by_id[p.id] = p
        return p

    def link(self, frame, kp, point, **kw):
        frame.keypoint_to_point[int(kp)] = getattr(point, "id", id(point))
        self.links.append((frame.id, int(kp), point))
        return True


def _link_points(wmap, frame, depths):
    """Points at the given depths in front of the frame's camera, linked to its keypoints 0, 1, ..."""
    Wf = np.asarray(frame.world_T_cam)
    for i, z in enumerate(depths):
        Xc = np.array([0.1 * i - 0.2, 0.05 * i, z])
        wmap.link(frame, i, wmap.add(Wf[:3, :3] @ Xc + Wf[:3, 3]))


def _tracker52(n, times=None):
    tr = _tracker()
    tr.variant = "r52"
    tr.timestamps = [k / 30.0 for k in range(n)] if times is None else list(times)
    return tr


def _held_fit(monkeypatch, sc):
    """The segment's fit as 04r writes it: rotation about the panorama keyframe's centre (measured: the
    scripted rotation; unsupported: frame k-1's rotation, the constant-angular-velocity stand-in)."""
    sc.fit_written = {}

    def seg_on_frame(self_, k, frames, **kw):
        sc.fit_calls.append(k)
        fr = frames[k]
        c_P = np.asarray(frames[self_.kf].world_T_cam)[:3, 3]
        R = sc.fit.get(k, "default")
        if isinstance(R, str):
            R = so3_exp([0.0, 0.01, 0.0]) @ np.asarray(frames[k - 1].cam_T_world)[:3, :3]
        measured = R is not None
        if not measured:
            R = np.asarray(frames[k - 1].cam_T_world)[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ c_P
        fr.set_camera_pose(T)
        sc.fit_written[k] = (fr.cam_T_world.copy(), fr.world_T_cam.copy())
        fr.pose_valid, fr.tracking_state = (True, "ROT") if measured else (False, "LOST")
        print(f"  >> [ROT-ONLY] frame={k} state={'measured' if measured else 'unsupported'} kf={self_.kf}")
        return "measured" if measured else "unsupported"
    monkeypatch.setattr(ro.RotationOnlyTracker, "on_frame", seg_on_frame)


def _enter_scene(monkeypatch, n=60, e=20, step=1.0, drift=None, **script):
    """Frames 0..e tracked (turning step deg per frame about C0), the anchor e accepted with three linked
    points at depths 2, 3, 4 (f = 3), the parent and the fit scripted."""
    frames = _turn_frames(n, e + 1, step, drift=drift)
    tr = _tracker52(n)
    wmap = _PtMap()
    _link_points(wmap, frames[e], [2.0, 3.0, 4.0])
    tr._record_inliers(e, frames, wmap)
    tr._rows_pt = [wmap.add([0.1 * i, 0.0, 3.0]) for i in range(64)]
    sc = _Script(monkeypatch, tr, **script)
    _held_fit(monkeypatch, sc)
    return frames, tr, wmap, sc


def _idx(out, pred):
    hits = [i for i, l in enumerate(out) if pred(l)]
    assert len(hits) == 1, hits
    return hits[0]


def _centre(fr):
    return np.asarray(fr.world_T_cam)[:3, 3]


# --------------------------------------------------------------------------- X1
def test_x1_flag_r52_choices_and_refusals():
    pr = main.get_parser()
    assert pr.parse_args(["--dataset", "rgb"]).map_rot_state == "off"
    assert pr.parse_args(M1_ARGV + PNP + R52).map_rot_state == "r52"
    for bad in ("bridge", "window", "held"):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            pr.parse_args(["--dataset", "rgb", "--map_rot_state", bad])
    assert _refused(M1_ARGV + PNP + R52) is None
    for val in ("on", "r52"):
        S = ["--map_rot_state", val]
        only = f"--map_rot_state {val} is only valid with --tracking map and --map_reloc pnp"
        assert _refused(M1_ARGV + S) == only
        assert _refused(M1_ARGV + ["--map_reloc", "none"] + S) == only
        assert _refused(["--dataset", "rgb"] + S) == only
        assert _refused(M1_ARGV + PNP + ["--reloc_shadow"] + S) == f"--map_rot_state {val} refuses --reloc_shadow"
        for s in ("sqpnp", "newton"):
            assert _refused(M1_ARGV + PNP + ["--reloc_solver", s] + S) == \
                f"--map_rot_state {val} refuses --reloc_solver other than epnp"
        assert "--no_ba" in _refused([x for x in M1_ARGV if x != "--no_ba"] + PNP + S)
    # the on messages are the 04v text character for character
    assert _refused(M1_ARGV + ON) == "--map_rot_state on is only valid with --tracking map and --map_reloc pnp"
    assert _refused(M1_ARGV + PNP + ["--reloc_shadow"] + ON) == "--map_rot_state on refuses --reloc_shadow"


# --------------------------------------------------------------------------- X2
def test_x2_run_pipeline_sets_variant_r52_and_fresh_tracker_is_on(monkeypatch):
    seen = {}

    class _Spy(mr.MapRotationTracker):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(mr, "MapRotationTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda path, **kw: mt.M1Map(header={}, points=[], corr={}))
    for val, want in (("r52", "r52"), ("on", "on")):
        with pytest.raises(SystemExit):
            main.run_pipeline(main.get_parser().parse_args(M1_ARGV + PNP + ["--map_rot_state", val]))
        t = seen.pop("tracker")
        assert type(t) is _Spy and t.variant == want
        assert t.reloc == "pnp" and t.reloc_solver == "epnp" and t.reloc_shadow is False
    fresh = mr.MapRotationTracker(mt.M1Map(header={}, points=[], corr={}), "m1a")
    assert fresh.variant == "on" and fresh.timestamps is None and fresh.fb_params == mr.FallbackParams()
    fp = mr.FallbackParams()
    assert (fp.hist_s, fp.alpha_max_deg, fp.r_min, fp.window_s, fp.min_frames, fp.gap_s) == (2.0, 5.0, 0.2, 1.0, 10, 0.05)
    assert type(fp.min_frames) is int and list(vars(fp)) == ["hist_s", "alpha_max_deg", "r_min", "window_s",
                                                              "min_frames", "gap_s"]


def test_x2_r52_requires_per_frame_timestamps(monkeypatch):
    seen = {}

    class _Spy(mr.MapRotationTracker):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            return True
    monkeypatch.setattr(mr, "MapRotationTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda path, **kw: mt.M1Map(header={}, points=[], corr={}))
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (K, 3, iter([])))

    def reached(*a, **k):
        raise SystemExit("reached show_trajectory")
    monkeypatch.setattr(main, "show_trajectory", reached)
    # r52 without --save_estimated: no timestamps, refused before any frame
    with pytest.raises(SystemExit) as ex, contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(main.get_parser().parse_args(M1_ARGV + PNP + R52))
    assert str(ex.value) == ("--map_rot_state r52 requires per-frame timestamps: give --save_estimated and a usable "
                             "--timestamp_file (rgb.txt)")
    assert seen.pop("tracker").timestamps is None
    # on: nothing in run_pipeline changes (no refusal, no timestamps handed over)
    with pytest.raises(SystemExit) as ex, contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(main.get_parser().parse_args(M1_ARGV + PNP + ON))
    assert str(ex.value) == "reached show_trajectory"
    t = seen.pop("tracker")
    assert t.variant == "on" and t.timestamps is None


# --------------------------------------------------------------------------- X3
def test_x3_fov_history_and_detection():
    assert abs(mr.fov_h_deg(DEFAULT_K, 640) - math.degrees(2 * math.atan(640 / 1041.8))) <= 1e-9
    assert abs(mr.fov_h_deg(DEFAULT_K, 640) - 63.1266) < 1e-4
    ts = [j / 30.0 for j in range(100)]
    assert mr.history_frames(80, lambda j: True, ts, 2.0) == list(range(20, 80))
    assert mr.history_frames(80, lambda j: j != 50, ts, 2.0) == [j for j in range(20, 80) if j != 50]
    # a frame 2.0 s + 1 ms back is out, one exactly 2.0 s back is in
    t2 = [0.0, 0.999, 1.0, 3.0]
    assert mr.history_frames(3, lambda j: True, t2, 2.0) == [2]
    t3 = [0.0, 0.999, 1.0, 3.001]
    assert mr.history_frames(3, lambda j: True, t3, 2.0) == []
    # the anchor itself is never in the list; an untracked frame does not stop the scan
    assert mr.history_frames(3, lambda j: j != 2, t2, 2.0) == []
    assert mr.history_frames(3, lambda j: j != 2, [0.0, 1.5, 1.6, 3.0], 2.0) == [1]
    FP = mr.FallbackParams()
    We = _wtc(np.eye(3), np.zeros(3))
    det, kept, rmax, alpha = mr.detect_rotation(We, [_wtc(_Ry(15.0), np.zeros(3))], 2.0, GAMMA, FP)
    assert det is True and kept == 1 and abs(rmax - 15.0 / GAMMA) <= 1e-9 and abs(alpha) <= 1e-12
    assert mr.detect_rotation(We, [_wtc(_Ry(10.0), np.zeros(3))], 2.0, GAMMA, FP)[:2] == (False, 1)
    d6 = 2 * 2.0 * math.tan(math.radians(3.0))                     # alpha = 6 deg at f = 2
    assert mr.detect_rotation(We, [_wtc(_Ry(15.0), np.array([d6, 0.0, 0.0]))], 2.0, GAMMA, FP) == (False, 0, -1.0, -1.0)
    d4 = 2 * 2.0 * math.tan(math.radians(2.0))                     # alpha = 4 deg: kept
    det, kept, rmax, alpha = mr.detect_rotation(We, [_wtc(_Ry(15.0), np.array([d4, 0.0, 0.0]))], 2.0, GAMMA, FP)
    assert det is True and kept == 1 and abs(alpha - 4.0) <= 1e-9
    for f in (float("nan"), 0.0, -1.0, float("inf")):
        assert mr.detect_rotation(We, [_wtc(_Ry(15.0), np.zeros(3))], f, GAMMA, FP) == (False, 0, -1.0, -1.0)
    assert mr.detect_rotation(We, [], 2.0, GAMMA, FP) == (False, 0, -1.0, -1.0)
    # r_max over the kept poses only, alpha that pose's; a far pose with a larger turn is not kept
    hist = [_wtc(_Ry(30.0), np.array([1.0, 0.0, 0.0])), _wtc(_Ry(5.0), np.zeros(3)),
            _wtc(_Ry(14.0), np.array([d4, 0.0, 0.0]))]
    det, kept, rmax, alpha = mr.detect_rotation(We, hist, 2.0, GAMMA, FP)
    assert det is True and kept == 2 and abs(rmax - 14.0 / GAMMA) <= 1e-9 and abs(alpha - 4.0) <= 1e-9


# --------------------------------------------------------------------------- X4
def test_x4_bridge_window():
    P0 = mr.FallbackParams()
    ts = [j / 30.0 for j in range(60)]
    assert mr.bridge_window(50, lambda j: j != 50, ts, P0) == []
    w = mr.bridge_window(50, lambda j: True, ts, P0)
    assert w == list(range(20, 51)) and len(w) == 31
    assert mr.bridge_window(50, lambda j: j != 40, ts, P0) == list(range(41, 51))
    tg = [t if j < 45 else t + 0.06 - 1 / 30.0 for j, t in enumerate(ts)]
    assert mr.bridge_window(50, lambda j: True, tg, P0) == list(range(45, 51))
    assert mr.bridge_window(0, lambda j: True, ts, P0) == [0]


def _steady_turn(n, s_true, times, c0=(1.0, 2.0, 0.5)):
    """world_T_cam of a steady turn about the world z axis moving with the constant body velocity s_true:
    c_j = c_(j-1) + R_j s_true (t_j - t_(j-1))."""
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    out = []
    c = np.array(c0, dtype=float)
    for j in range(n):
        a = 0.4 * times[j]
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R0
        if j:
            c = c + R @ s_true * (times[j] - times[j - 1])
        out.append(_wtc(R, c))
    return out


def test_x4_fit_body_velocity():
    s_true = np.array([0.03, -0.01, 0.25])
    times = [j / 30.0 + (0.004 if j % 3 == 0 else 0.0) for j in range(31)]
    Wl = _steady_turn(31, s_true, times)
    s_hat, fitted = mr.fit_body_velocity(Wl, times, 10)
    assert fitted is True and s_hat.shape == (3,) and s_hat.dtype == np.float64
    assert np.max(np.abs(s_hat - s_true)) <= 1e-9
    s9, f9 = mr.fit_body_velocity(Wl[:9], times[:9], 10)
    assert f9 is False and np.array_equal(s9, np.zeros(3))
    s10, f10 = mr.fit_body_velocity(Wl[:10], times[:10], 10)
    assert f10 is True and np.max(np.abs(s10 - s_true)) <= 1e-9
    # the velocity is in the camera frame: the world displacement of a pure body-z motion follows the turn
    s_z = np.array([0.0, 0.0, 1.0])
    Wz = _steady_turn(12, s_z, times[:12])
    assert np.max(np.abs(mr.fit_body_velocity(Wz, times[:12], 10)[0] - s_z)) <= 1e-9


# --------------------------------------------------------------------------- X5
def test_x5_entry_detected_and_measured(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch)
    states, out = _run(tr, frames, wmap, [21])
    assert states == ["rotation"]
    assert np.allclose(_centre(frames[21]), _centre(frames[20]), atol=1e-12, rtol=0.0)
    assert frames[21].pose_valid is True and frames[21].tracking_state == "ROT"
    i_kf = _idx(out, lambda l: RX_KF.match(l) is not None)
    assert RX_KF.match(out[i_kf]).groups() == ("20", "-1", "1")
    i_en = _idx(out, lambda l: "[MAP-ROT-ENTRY]" in l)
    i_rot = _idx(out, lambda l: "[MAP-ROT] " in l)
    i_tr = _idx(out, lambda l: "[MAP-TRACK] frame=21 " in l)
    assert i_tr < i_kf < i_en < i_rot == len(out) - 1
    g = RX_ENTRY.match(out[i_en]).groups()
    assert g[:3] == ("21", "20", "20") and float(g[3]) == pytest.approx(3.0, abs=1e-6)
    assert float(g[4]) == pytest.approx(GAMMA, abs=1e-6) and g[5] == "20"
    assert float(g[6]) == pytest.approx(20.0 / GAMMA, abs=1e-6) and float(g[7]) == pytest.approx(0.0, abs=1e-6)
    assert g[8:] == ("1", "measured", "enter")
    m = RX_ROT.match(out[i_rot])
    assert m.group(2, 3, 4, 5, 6, 7, 8, 9) == ("enter", "rotation", "20", "1", "20", "measured", "none", "-1")
    assert np.all(np.abs([float(x) for x in m.group(10, 11, 12)]) <= 1e-9)
    assert not any("[MAP-LOST-VEL]" in l for l in out)
    assert tr._anchor == 20 and tr.n_stints == 1 and tr.n_entered == 1 and tr.n_refused == 0
    assert sc.fit_calls == [21] and isinstance(tr._seg, mr.PanoramaSegment) and tr._seg.kf == 20


def test_x5_entry_not_detected_runs_no_fit(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch, step=0.3)          # 6 deg over the history: r 0.095
    states, out = _run(tr, frames, wmap, [21])
    assert states == ["lost"] and sc.fit_calls == [] and tr._segments == []
    assert frames[21].pose_valid is False and frames[21].tracking_state == "LOST"
    assert not any(RX_KF.match(l) for l in out) and not any("[MAP-ROT] " in l for l in out)
    i_en = _idx(out, lambda l: "[MAP-ROT-ENTRY]" in l)
    i_lv = _idx(out, lambda l: "[MAP-LOST-VEL]" in l)
    assert i_en < i_lv
    g = RX_ENTRY.match(out[i_en]).groups()
    assert g[5] == "20" and float(g[6]) == pytest.approx(6.0 / GAMMA, abs=1e-6)
    assert g[8:] == ("0", "none", "refused_detect")
    assert tr._anchor is None and tr.n_refused_detect == 1 and tr.n_refused == 0


def test_x5_entry_refused_by_parallax(monkeypatch):
    """The same turn while the camera drives 0.1 per frame: only the two newest history poses pass the
    5 deg parallax gate at f = 3, and they turned too little."""
    frames, tr, wmap, sc = _enter_scene(monkeypatch, drift=(0.1, 0.0, 0.0))
    states, out = _run(tr, frames, wmap, [21])
    g = RX_ENTRY.match(next(l for l in out if "[MAP-ROT-ENTRY]" in l)).groups()
    assert g[5] == "2" and float(g[6]) == pytest.approx(2.0 / GAMMA, abs=1e-6)
    assert g[8:] == ("0", "none", "refused_detect") and sc.fit_calls == [] and states == ["lost"]


def test_x5_entry_detected_and_unsupported(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch, fit={21: None})
    T_lost = {}
    real_parent = mt.MapTracker.on_frame

    def parent(self_, k, frames_, world_map, **kw):
        s = real_parent(self_, k, frames_, world_map)
        T_lost[k] = frames_[k].cam_T_world.copy()
        return s
    monkeypatch.setattr(mt.MapTracker, "on_frame", parent)
    states, out = _run(tr, frames, wmap, [21])
    assert states == ["lost"] and sc.fit_calls == [21] and tr._anchor is None
    i_en = _idx(out, lambda l: "[MAP-ROT-ENTRY]" in l)
    i_lv = _idx(out, lambda l: "[MAP-LOST-VEL]" in l)
    i_rot = _idx(out, lambda l: "[MAP-ROT] " in l)
    assert i_en < i_lv < i_rot == len(out) - 1
    assert RX_ENTRY.match(out[i_en]).groups()[8:] == ("1", "unsupported", "refused_fit")
    m = RX_ROT.match(out[i_rot])
    assert m.group(2, 3, 4, 5, 6, 7, 8, 9) == ("refused", "lost", "20", "1", "20", "unsupported", "none", "-1")
    # the restored parent rotation, the centre from the lost prediction (C6) on top of it
    assert np.allclose(frames[21].cam_T_world[:3, :3], T_lost[21][:3, :3], atol=1e-12)
    R = frames[21].cam_T_world[:3, :3]
    want = _centre(frames[20]) + R.T @ tr._lost_vel * (tr.timestamps[21] - tr.timestamps[20])
    assert np.allclose(_centre(frames[21]), want, atol=1e-12)
    assert frames[21].pose_valid is False and frames[21].tracking_state == "LOST"
    assert tr.n_refused == 1 and tr.n_refused_fit == 1 and tr.n_stints == 0


def test_x5_entry_error_restores_and_is_lost(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch)

    def boom(self_, k, frames_, **kw):
        raise RuntimeError("fit boom")
    monkeypatch.setattr(ro.RotationOnlyTracker, "on_frame", boom)
    states, out = _run(tr, frames, wmap, [21])
    assert states == ["lost"] and tr._anchor is None
    assert "  >> [MAP-ROT-ERROR] frame=21 error=RuntimeError: fit boom" in out
    assert RX_ENTRY.match(next(l for l in out if "[MAP-ROT-ENTRY]" in l)).groups()[8:] == ("1", "none", "error")
    assert not any("[MAP-ROT] " in l for l in out) and any("[MAP-LOST-VEL]" in l for l in out)
    assert frames[21].tracking_state == "LOST"


# --------------------------------------------------------------------------- X6
def test_x6_stint_holds_the_anchor_centre(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch, fit={30: None})
    ks = list(range(21, 56))
    states, out = _run(tr, frames, wmap, ks)
    lines = _rot_lines(out)
    c_e = _centre(frames[20])
    for k in ks:
        assert np.allclose(_centre(frames[k]), c_e, atol=1e-12, rtol=0.0), k
        assert np.all(np.abs([float(x) for x in lines[k].group(10, 11, 12)]) <= 1e-9), k
        if k > 21:
            # the stay pose is T_h bitwise, although the failed relocalization moved the frame
            assert np.array_equal(frames[k].cam_T_world, sc.fit_written[k][0])
            assert np.array_equal(frames[k].world_T_cam, sc.fit_written[k][1])
    assert states[ks.index(30)] == "lost" and lines[30].group(2, 3, 7) == ("stay", "lost", "unsupported")
    assert frames[30].pose_valid is False and frames[30].tracking_state == "LOST"
    assert states[ks.index(31)] == "rotation" and frames[31].tracking_state == "ROT"
    assert lines[51].group(3, 5) == ("rotation", "31") and states[ks.index(51)] == "rotation"
    assert "bridged" not in states and all(m.group(3) != "bridged" for m in lines.values())
    # every stay ran the relocalization with frame k-1 seen as LOST, and frame k-1's state came back
    assert [k for k, st in sc.parent_calls] == ks and all(st == "LOST" for k, st in sc.parent_calls[1:])
    assert frames[54].tracking_state == "ROT" and tr._anchor == 20 and tr.n_bridged == 0


def test_x6_exit_by_normal_tracking(monkeypatch):
    T_fin = np.eye(4)
    T_fin[:3, 3] = [0.5, 0.0, 0.0]
    frames, tr, wmap, sc = _enter_scene(monkeypatch, track={23: (T_fin, 40)})
    states, out = _run(tr, frames, wmap, [21, 22, 23])
    assert states == ["rotation", "rotation", "tracked"]
    assert np.array_equal(frames[23].cam_T_world, T_fin) and frames[23].tracking_state == "OK"
    lines = _rot_lines(out)
    assert lines[23].group(2, 3, 8, 9) == ("exit", "tracked", "track", "40")
    assert sum(1 for l in out if "[MAP-TRACK] frame=23 " in l) == 1
    assert tr._anchor is None and tr.n_exits_track == 1
    # C4 on the exit: the 40 points the accepted pose linked
    assert tr._inl_k == 23 and len(tr._inl_X) == 40
    # the seed of normal tracking is T_h (the held pose), not a bridged one
    assert np.array_equal(sc.track_calls[-1][1], sc.fit_written[23][0])


def test_x6_exit_by_relocalization(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch, parent={23: "tracked"})
    states, out = _run(tr, frames, wmap, [21, 22, 23])
    assert states == ["rotation", "rotation", "tracked"]
    assert sc.parent_calls == [(21, "OK"), (22, "LOST"), (23, "LOST")]
    assert frames[22].tracking_state == "ROT"
    lines = _rot_lines(out)
    assert lines[23].group(2, 3, 4, 5, 8) == ("exit", "tracked", "20", "3", "reloc")
    assert tr._anchor is None and tr.n_exits_reloc == 1 and tr._inl_k == 23
    for k in (21, 22, 23):
        assert sum(1 for l in out if f"[MAP-TRACK] frame={k} " in l) == 1


def test_x6_errors_inside_a_stint(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch)
    _run(tr, frames, wmap, [21])

    def boom(k, frames_, T_pred, **kw):
        raise RuntimeError("track boom")
    monkeypatch.setattr(tr, "_track_m1a", boom)
    states, out = _run(tr, frames, wmap, [22])
    assert states == ["lost"] and tr._anchor == 20
    assert np.array_equal(frames[22].cam_T_world, sc.fit_written[22][0])     # T_h, bitwise
    assert frames[22].pose_valid is False and frames[22].tracking_state == "LOST"
    assert sum(1 for l in out if "[MAP-TRACK] frame=22 " in l) == 1
    assert _rot_lines(out)[22].group(2, 3, 7) == ("stay", "lost", "unsupported")

    # an error before the fit wrote anything: [R_(k-1) | -R_(k-1) c_e]
    def seg_boom(self_, k, frames_, **kw):
        raise RuntimeError("seg boom")
    monkeypatch.setattr(ro.RotationOnlyTracker, "on_frame", seg_boom)
    R1 = frames[22].cam_T_world[:3, :3].copy()
    states, out = _run(tr, frames, wmap, [23])
    assert states == ["lost"] and "  >> [MAP-ROT-ERROR] frame=23 error=RuntimeError: seg boom" in out
    want = np.eye(4)
    want[:3, :3] = R1
    want[:3, 3] = -R1 @ _centre(frames[20])
    assert np.array_equal(frames[23].cam_T_world, _sanitized(want))
    assert np.allclose(_centre(frames[23]), _centre(frames[20]), atol=1e-12)


# --------------------------------------------------------------------------- X7
def _lost_scene(monkeypatch, n_tracked, n=30, gap_at=None):
    dts = [0.030 if j % 2 else 0.041 for j in range(n)]
    times = [0.0]
    for j in range(1, n):
        times.append(times[-1] + dts[j] + (0.06 if j == gap_at else 0.0))
    s_true = np.array([0.02, -0.005, 0.3])
    Wl = _steady_turn(n_tracked, s_true, times)
    frames = [_F52(k + 1, np.linalg.inv(Wl[k]), "OK") if k < n_tracked else _F52(k + 1, np.eye(4), "LOST")
              for k in range(n)]
    tr = _tracker52(n, times)                     # no anchor recorded: f nan, every attempt refused_detect
    sc = _Script(monkeypatch, tr)
    _held_fit(monkeypatch, sc)
    return frames, tr, sc, s_true


def test_x7_lost_prediction_windowed_velocity(monkeypatch):
    frames, tr, sc, s_true = _lost_scene(monkeypatch, 15)
    ks = list(range(15, 21))
    states, out = _run(tr, frames, _PtMap(), ks)
    assert states == ["lost"] * len(ks) and sc.fit_calls == []
    lv = [RX_LOSTVEL.match(l) for l in out if "[MAP-LOST-VEL]" in l]
    assert len(lv) == 1 and lv[0].group(1, 2, 3, 4, 5) == ("15", "14", "0", "15", "1")
    assert np.allclose([float(x) for x in lv[0].group(6, 7, 8)], s_true, atol=2e-9)
    assert np.max(np.abs(tr._lost_vel - s_true)) <= 1e-9
    assert sum(1 for l in out if "[MAP-ROT-ENTRY]" in l) == 1
    R_parent = _sanitized(np.block([[so3_exp([0.0, 0.3, 0.0]), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    c = _centre(frames[14]).copy()
    dts = set()
    for k in ks:
        R_k = frames[k].cam_T_world[:3, :3]
        assert np.allclose(R_k, R_parent[:3, :3], atol=1e-12)         # the tracker's rotation, as before
        dt = tr.timestamps[k] - tr.timestamps[k - 1]
        dts.add(round(dt, 6))
        c = c + R_k.T @ tr._lost_vel * dt
        assert np.allclose(_centre(frames[k]), c, atol=1e-12, rtol=0.0), k
        assert frames[k].pose_valid is False and frames[k].tracking_state == "LOST"
    assert len(dts) == 2                                                # unequal intervals
    assert not np.allclose(_centre(frames[20]), _centre(frames[14]), atol=1e-3)
    assert tr.n_lost_runs == 1 and tr.n_lost_runs_fitted == 1


def test_x7_short_window_holds_the_centre(monkeypatch):
    frames, tr, sc, s_true = _lost_scene(monkeypatch, 6)
    states, out = _run(tr, frames, _PtMap(), [6, 7, 8, 9])
    lv = [RX_LOSTVEL.match(l) for l in out if "[MAP-LOST-VEL]" in l]
    assert len(lv) == 1 and lv[0].group(1, 2, 3, 4, 5) == ("6", "5", "0", "6", "0")
    assert lv[0].group(6, 7, 8) == ("0.000000000", "0.000000000", "0.000000000")
    for k in (6, 7, 8, 9):
        assert np.allclose(_centre(frames[k]), _centre(frames[5]), atol=1e-12, rtol=0.0)
    assert tr.n_lost_runs == 1 and tr.n_lost_runs_fitted == 0


def test_x7_gap_ends_the_window_and_a_new_run_refits(monkeypatch):
    frames, tr, sc, s_true = _lost_scene(monkeypatch, 15, gap_at=8)
    states, out = _run(tr, frames, _PtMap(), [15, 16])
    lv = [RX_LOSTVEL.match(l) for l in out if "[MAP-LOST-VEL]" in l]
    assert lv[0].group(3, 4, 5) == ("8", "7", "0")                      # 8..14: under 10 frames
    # a tracked frame ends the run (the velocity is cleared); the next loss fits again
    monkeypatch.setattr(mt.MapTracker, "on_frame", lambda self_, k, frames_, world_map, **kw: (
        frames_[k].__setattr__("tracking_state", "OK") or "tracked"))
    _run(tr, frames, _PtMap(), [17])
    assert tr._lost_vel is None
    sc2 = _Script(monkeypatch, tr)
    _held_fit(monkeypatch, sc2)
    states, out = _run(tr, frames, _PtMap(), [18])
    lv = [RX_LOSTVEL.match(l) for l in out if "[MAP-LOST-VEL]" in l]
    assert len(lv) == 1 and lv[0].group(1, 2, 3, 4, 5) == ("18", "17", "17", "1", "0")
    assert tr.n_lost_runs == 2


# --------------------------------------------------------------------------- X8
def _seg_frames():
    fr = []
    for k in range(9):
        f = _F52(k + 1, np.linalg.inv(_wtc(np.eye(3), C0)), "OK")
        f.method = "SIFT"
        f.KeyPoints = [cv2.KeyPoint(10.0 * i, 5.0 * i, 3.0) for i in range(6)]
        fr.append(f)
    return fr


def test_x8_panorama_segment_switch_rule(monkeypatch):
    frames = _seg_frames()
    rel = {}
    monkeypatch.setattr(ro, "match_frames", lambda a, b, *x, **kw: [cv2.DMatch(i, i, 0.0) for i in range(6)])
    monkeypatch.setattr(ro, "_keypoint_sigmas", lambda f, *x, **kw: np.ones(len(f.KeyPoints)))
    cov = {}
    monkeypatch.setattr(ro, "solve_rotation", lambda a, b, sa, sb, f, p, **kw: (rel["R"], np.ones(len(a), bool), "none"))
    monkeypatch.setattr(ro, "view_coverage", lambda K_, R, w, h, p, **kw: cov["c"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seg = mr.PanoramaSegment(0.2)
        seg._frames = frames
        seg._new_keyframe(5)                              # the anchor keyframe is always taken
        assert seg.kf == 5
        rel["R"], cov["c"] = _Ry(10.0), 0.5               # coverage-triggered, r = 10 / 63.13 <= 0.2: hold
        assert seg.on_frame(6, frames) == "measured"
        assert seg.kf == 5 and seg.n_holds == 1
        rel["R"], cov["c"] = _Ry(15.0), 0.5               # r = 15 / 63.13 > 0.2: switch
        assert seg.on_frame(7, frames) == "measured"
        assert seg.kf == 7 and seg.n_holds == 1
        rel["R"], cov["c"] = _Ry(20.0), 0.9               # coverage high: no switch test at all
        assert seg.on_frame(8, frames) == "measured"
        assert seg.kf == 7
    out = buf.getvalue().splitlines()
    kf = [RX_KF.match(l).groups() for l in out if RX_KF.match(l)]
    assert kf == [("5", "-1", "1"), ("7", "5", "2")]
    holds = [RX_HOLD.match(l) for l in out if RX_HOLD.match(l)]
    assert len(holds) == 1 and holds[0].group(1, 2) == ("6", "5")
    assert float(holds[0].group(3)) == pytest.approx(10.0 / GAMMA, abs=1e-6)
    i_hold = out.index(holds[0].group(0))
    assert "[ROT-ONLY] frame=6 state=measured" in out[i_hold - 1]
    # a fresh segment takes the anchor even with _frames set and a tiny angle
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = mr.PanoramaSegment(0.2)
        s2._frames = frames
        s2._new_keyframe(3)
    assert s2.kf == 3 and s2.n_holds == 0


# --------------------------------------------------------------------------- X9
def test_x9_anchor_depth_from_the_linked_points(monkeypatch):
    frames = _turn_frames(8, 3, 1.0)
    tr = _tracker52(8)
    wmap = _PtMap()
    ghost = 99999                                         # a link to a point no longer in the map

    def parent(self_, k, frames_, world_map, **kw):
        fr = frames_[k]
        fr.set_camera_pose(np.linalg.inv(_wtc(_Ry(4.0), C0 + [0.0, 0.0, 0.1])))
        fr.keypoint_to_point[7] = ghost
        _link_points(world_map, fr, [1.5, 2.5, 3.5, 6.5])
        fr.pose_valid, fr.tracking_state = True, "OK"
        return "tracked"
    monkeypatch.setattr(mt.MapTracker, "on_frame", parent)
    states, _out = _run(tr, frames, wmap, [3])
    assert states == ["tracked"] and tr._inl_k == 3
    want_X = np.array([[p.x, p.y, p.z] for p in wmap.points])
    assert np.array_equal(tr._inl_X, want_X)
    T = frames[3].cam_T_world
    assert tr._anchor_depth(3, frames) == pytest.approx(float(np.mean(want_X @ T[2, :3] + T[2, 3])), abs=1e-12)
    assert tr._anchor_depth(3, frames) == pytest.approx(3.5, abs=1e-12)
    assert math.isnan(tr._anchor_depth(2, frames))
    tr._inl_k = None
    assert math.isnan(tr._anchor_depth(3, frames))
    tr._inl_k, tr._inl_X = 3, np.zeros((0, 3))
    assert math.isnan(tr._anchor_depth(3, frames))


def test_x9_unrecorded_anchor_gives_nan_and_no_detection(monkeypatch):
    frames, tr, wmap, sc = _enter_scene(monkeypatch)
    tr._inl_k = 19                                        # not the anchor
    states, out = _run(tr, frames, wmap, [21])
    g = RX_ENTRY.match(next(l for l in out if "[MAP-ROT-ENTRY]" in l)).groups()
    assert g[3] == "nan" and g[5:] == ("0", "-1.000000", "-1.000000", "0", "none", "refused_detect")
    assert sc.fit_calls == [] and states == ["lost"]


def test_x9_setup_and_summary_lines(monkeypatch):
    empty = mt.M1Map(header={}, points=[], corr={})
    on = mr.MapRotationTracker(empty, "m1a")
    base_setup, base_sum = on.setup_line(), on.summary_line()
    assert base_setup.split("\n") == [mt.MapTracker(empty, "m1a").setup_line(), "  >> [MAP-ROT-SETUP] horizon=30"]
    assert "\n" not in base_sum and "R52" not in base_setup
    t52 = mr.MapRotationTracker(empty, "m1a")
    t52.variant = "r52"
    assert t52.setup_line().split("\n") == base_setup.split("\n") + [
        "  >> [MAP-ROT-R52-SETUP] hist_s=2.000000 alpha_max_deg=5.000000 r_min=0.200000 window_s=1.000000 "
        "min_frames=10 gap_s=0.050000"]
    assert t52.summary_line().split("\n") == [base_sum, "  >> [MAP-ROT-R52-SUMMARY] attempts=0 detected=0 "
                                              "refused_detect=0 refused_fit=0 entered=0 kf_holds=0 lost_runs=0 "
                                              "lost_runs_fitted=0"]
    # counts on a run: entry, stays, an exit by tracking, then a refused_detect attempt and a lost run
    T_fin = np.eye(4)
    frames, tr, wmap, sc = _enter_scene(monkeypatch, track={24: (T_fin, 40)}, fit={23: None})
    states, out = _run(tr, frames, wmap, [21, 22, 23, 24, 25, 26])
    assert states == ["rotation", "rotation", "lost", "tracked", "lost", "lost"]
    tr._segments[0].n_holds = 2                           # kf_holds sums the segments' holds
    sm = tr.summary_line().split("\n")
    g4 = RX_SUM.match(sm[0])
    lines = _rot_lines(out)
    assert tuple(int(x) for x in g4.groups()[:9]) == (
        1, 0, 1, 0, 0, sum(1 for m in lines.values() if m.group(3) == "rotation"),
        sum(1 for m in lines.values() if m.group(3) == "lost"), states.count("tracked"), states.count("lost"))
    entries = [RX_ENTRY.match(l).groups() for l in out if RX_ENTRY.match(l)]
    lv = [RX_LOSTVEL.match(l).groups() for l in out if RX_LOSTVEL.match(l)]
    g = tuple(int(x) for x in RX_R52SUM.match(sm[1]).groups())
    assert g == (len(entries), sum(e[8] == "1" for e in entries), sum(e[10] == "refused_detect" for e in entries),
                 sum(e[10] == "refused_fit" for e in entries), sum(e[10] == "enter" for e in entries), 2,
                 len(lv), sum(v[4] == "1" for v in lv))
    assert g == (2, 1, 1, 0, 1, 2, 1, 0)
    # under on, the same tracker class prints the 04v lines only
    on2 = mr.MapRotationTracker(empty, "m1a")
    on2.n_attempts = 5
    assert on2.summary_line() == base_sum and on2.setup_line() == base_setup
