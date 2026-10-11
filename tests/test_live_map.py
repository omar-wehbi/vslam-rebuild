"""Stage 04-m0: the integrated system M0 behind --live_map (tests L1-L8).

L1 the switch, its refusals and the hand-off in main.run_pipeline; L2 MapTracker.sync_live on a real
Map; L3 bundle_adjust's camera_supported predicate (the M0 form of R57-2); L4 the depth filter's seeds
follow their reference keyframe; L5 OpeningTracker.tracker_factory; L6 the pipeline's M0 helpers and
line formats; L7 m0_clear_map; L8 the integrated frame of show_trajectory with a scripted opening driver
(the no-write rule of the fallback states, the hand-off keyframe, the false start, the line order).
The stand-ins are copied from tests/test_ba_participation.py and tests/test_opening.py (not imported).
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import depth_filter as df  # noqa: E402
import depth_filter_arm as dfa  # noqa: E402
import main  # noqa: E402
import map_rotation  # noqa: E402
import map_tracker  # noqa: E402
import opening  # noqa: E402
import optimization as opt  # noqa: E402
import pipeline  # noqa: E402
from frame import DEFAULT_K  # noqa: E402
from Map import Map  # noqa: E402
from optimization import build_ba_problem, bundle_adjust, so3_exp, solve_ba  # noqa: E402
from Point import Point  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
M = opt.BA_MIN_FREE_CAMERA_POINTS

# the spec H flag set (the harness adds --default_k through the config)
M0_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
           "--init", "twoview", "--live_map", "on", "--map_rot_state", "r52", "--pnp_interval", "0",
           "--no_loop_closure", "--no_relocalization", "--point_source", "depth_filter", "--df_promote_factor", "1.0",
           "--ba_outlier_reject", "on", "--ba_jump_gate", "off", "--ba_free_cameras", "supported"]
ROT_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
            "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba", "--no_point_fusion",
            "--init", "twoview"]


def _parse(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        return main.get_parser().parse_args(main.front_end.expand_argv(argv))


def _refusal(argv):
    args = _parse(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.validate_tracking_args(args)
        return None
    except SystemExit as e:
        return str(e)


def _without(argv, *flags):
    out = list(argv)
    for f in flags:
        i = out.index(f)
        n = 2 if (i + 1 < len(out) and not out[i + 1].startswith("--")) else 1
        del out[i:i + n]
    return out


def _replaced(argv, flag, value):
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


# =========================================================================== L1
def test_l1_parser_default_and_values(tmp_path):
    pr = main.get_parser()
    assert pr.parse_args(["--dataset", "rgb"]).live_map == "off"
    assert pr.parse_args(["--dataset", "rgb", "--live_map", "off"]).live_map == "off"
    assert pr.parse_args(["--dataset", "rgb", "--live_map", "on"]).live_map == "on"
    for bad in ("maybe", "ON", "1", ""):
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            pr.parse_args(["--dataset", "rgb", "--live_map", bad])
    helps = {a.dest: a.help for a in pr._actions}
    assert helps["live_map"] == ("the integrated system M0 (design.md 04-4 row): the map tracker tracks the map the "
                                 "mapper builds")
    out = tmp_path / "resolved_settings.json"
    main.write_resolved_settings(_parse(M0_ARGV), out)
    assert json.load(open(out))["live_map"] == "on"
    main.write_resolved_settings(pr.parse_args(["--dataset", "rgb"]), out)
    assert json.load(open(out))["live_map"] == "off"


REFUSALS = [
    (_replaced(M0_ARGV, "--tracking", "chain"), "--live_map on requires --tracking rotation"),
    (_replaced(M0_ARGV, "--tracking", "map"), "--live_map on requires --tracking rotation"),
    (_without(M0_ARGV, "--init"), "--live_map on requires --init twoview"),
    (_without(M0_ARGV, "--default_k"), "--live_map on requires --default_k"),
    (_without(M0_ARGV, "--undistort"), "--live_map on requires --undistort"),
    (_replaced(M0_ARGV, "--pnp_interval", "10"), "--live_map on requires --pnp_interval 0"),
    (_without(M0_ARGV, "--no_loop_closure"), "--live_map on requires --no_loop_closure"),
    (_without(M0_ARGV, "--no_relocalization"), "--live_map on requires --no_relocalization"),
    (M0_ARGV + ["--m1_map", "/tmp/m1.txt"], "--live_map on refuses --m1_map"),
    (M0_ARGV + ["--map_reloc", "pnp"], "--live_map on refuses --map_reloc pnp"),
    (M0_ARGV + ["--reloc_solver", "sqpnp"], "--live_map on refuses --reloc_solver other than epnp"),
    (M0_ARGV + ["--reloc_solver", "newton"], "--live_map on refuses --reloc_solver other than epnp"),
    (M0_ARGV + ["--reloc_shadow"], "--live_map on refuses --reloc_shadow"),
    (M0_ARGV + ["--oracle", "o3"], "--live_map on refuses --oracle other than none"),
    (M0_ARGV + ["--lm_confirm"], "--live_map on refuses --lm_confirm"),
    (M0_ARGV + ["--df_lifetime", "r2"], "--live_map on refuses --df_lifetime r2"),
    (_replaced(M0_ARGV, "--map_rot_state", "on"), "--live_map on refuses --map_rot_state on"),
]


@pytest.mark.parametrize("argv,msg", REFUSALS)
def test_l1_refusals_by_message(argv, msg):
    assert _refusal(argv) == msg


def test_l1_refusal_order():
    # every requirement fails at once: the first rule of A2 is reported, then the next one once it holds
    argv = ["--dataset", "rgb", "--live_map", "on", "--m1_map", "/tmp/m1.txt", "--lm_confirm"]
    assert _refusal(argv) == "--live_map on requires --tracking rotation"
    argv += ["--tracking", "rotation"]
    assert _refusal(argv) == "--live_map on requires --init twoview"
    argv += ["--init", "twoview"]
    assert _refusal(argv) == "--live_map on requires --default_k"
    argv += ["--default_k", "--undistort", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization"]
    assert _refusal(argv) == "--live_map on refuses --m1_map"
    assert _refusal(_without(argv, "--m1_map")) == "--live_map on refuses --lm_confirm"
    # refusals in order: --reloc_shadow before --oracle before --lm_confirm before --df_lifetime r2
    base = M0_ARGV + ["--lm_confirm", "--df_lifetime", "r2"]
    assert _refusal(base + ["--reloc_shadow", "--oracle", "o3"]) == "--live_map on refuses --reloc_shadow"
    assert _refusal(base + ["--oracle", "o3"]) == "--live_map on refuses --oracle other than none"
    assert _refusal(base) == "--live_map on refuses --lm_confirm"


@pytest.mark.parametrize("argv", [
    M0_ARGV,
    _replaced(M0_ARGV, "--map_rot_state", "off"),
    _without(M0_ARGV, "--map_rot_state"),
    _without(M0_ARGV, "--point_source", "--df_promote_factor"),
    M0_ARGV + ["--no_ba", "--no_point_fusion"],
    M0_ARGV + ["--epi_refine", "--init_refit", "live"],
    M0_ARGV + ["--init_refit", "shadow", "--ba_mode", "keyframe_local_covisibility"],
    _replaced(_replaced(M0_ARGV, "--ba_free_cameras", "all"), "--ba_jump_gate", "on"),
])
def test_l1_accepted_flag_sets(argv):
    assert _refusal(argv) is None


def test_l1_the_04u_refusals_hold_without_the_switch():
    assert _refusal(ROT_ARGV) is None
    assert _refusal(ROT_ARGV + ["--point_source", "depth_filter"]) == \
        "--tracking rotation refuses --point_source depth_filter"
    assert _refusal(_without(ROT_ARGV, "--no_ba")) == "--tracking rotation requires --no_ba"
    assert _refusal(_without(ROT_ARGV, "--no_point_fusion")) == "--tracking rotation requires --no_point_fusion"
    assert _refusal(ROT_ARGV + ["--live_map", "off", "--point_source", "depth_filter"]) == \
        "--tracking rotation refuses --point_source depth_filter"
    # the 04x rule is unchanged without the switch
    assert _refusal(ROT_ARGV + ["--map_rot_state", "r52"]) == \
        "--map_rot_state r52 is only valid with --tracking map and --map_reloc pnp"
    assert _refusal(["--dataset", "rgb"]) is None


def test_l1_namespace_without_the_flag():
    args = _parse(ROT_ARGV)
    del args.live_map
    with contextlib.redirect_stdout(io.StringIO()):
        assert main.validate_tracking_args(args) is None


def test_l1_show_trajectory_keyword_and_default():
    sp = inspect.signature(pipeline.show_trajectory).parameters
    assert "live_map" in sp and sp["live_map"].default == "off"
    assert 'live_map=getattr(args, "live_map", "off")' in inspect.getsource(main.run_pipeline)


def _capture_run_pipeline(monkeypatch, argv, timestamps_ok=True):
    seen = {}

    def fake_show(frame_iter, **kw):
        seen.update(kw)

    monkeypatch.setattr(main, "show_trajectory", fake_show)
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (K, 0, iter(())))
    monkeypatch.setattr(main, "load_timestamps_for_images", lambda *a, **k: [] if timestamps_ok else None)
    args = _parse(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(args)
    return seen


def _m1map():
    return map_tracker.M1Map(header={}, points=[], corr={})


def test_l1_run_pipeline_hand_off_r52(monkeypatch, tmp_path):
    kw = _capture_run_pipeline(monkeypatch, M0_ARGV + ["--save_estimated", str(tmp_path / "estimated.txt")])
    assert kw["live_map"] == "on"
    op = kw["open_tracker"]
    assert isinstance(op, opening.OpeningTracker) and op.refit == "off"
    assert kw["rot_tracker"] is None and kw["m1_tracker"] is None
    tr = op.tracker_factory(_m1map())
    assert type(tr) is map_rotation.MapRotationTracker
    assert tr.mode == "m1a" and tr.variant == "r52" and tr.timestamps == [] and tr.live_map is True
    assert op.tracker_factory.map_rot_state == "r52"


def test_l1_run_pipeline_hand_off_off(monkeypatch):
    kw = _capture_run_pipeline(monkeypatch, _replaced(M0_ARGV, "--map_rot_state", "off"))
    tr = kw["open_tracker"].tracker_factory(_m1map())
    assert type(tr) is map_tracker.MapTracker and tr.mode == "m1a" and tr.live_map is True
    assert kw["open_tracker"].tracker_factory.map_rot_state == "off"
    # a second call builds a new tracker (one per hand-off)
    assert kw["open_tracker"].tracker_factory(_m1map()) is not tr


def test_l1_run_pipeline_r52_requires_timestamps(monkeypatch):
    with pytest.raises(SystemExit) as e:
        _capture_run_pipeline(monkeypatch, M0_ARGV, timestamps_ok=False)
    assert str(e.value).startswith("--map_rot_state r52 requires per-frame timestamps")


def test_l1_run_pipeline_switch_off_has_no_factory(monkeypatch):
    kw = _capture_run_pipeline(monkeypatch, ROT_ARGV)
    assert kw["live_map"] == "off" and kw["open_tracker"].tracker_factory is None


# =========================================================================== L2
class _SFrame:
    """A frame with an id, a pose, keypoints and descriptors (enough for Map, release and sync_live)."""

    def __init__(self, fid, centre, n_kp=6):
        self.id = fid
        self.K = K
        self.method = "SIFT"
        self.img_shape = (480, 640)
        self.keypoint_to_point = {}
        self.KeyPoints = [cv2.KeyPoint(50.0 + 30.0 * i, 60.0 + 20.0 * i, 3.0) for i in range(n_kp)]
        self.Descriptors = np.arange(n_kp * 4, dtype=np.float32).reshape(n_kp, 4) + fid
        self.world_T_cam = np.eye(4)
        self.world_T_cam[:3, 3] = centre
        self.cam_T_world = np.linalg.inv(self.world_T_cam)
        self.pose_valid = True
        self.tracking_state = "OK"

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _move(frame, centre):
    W = np.eye(4)
    W[:3, 3] = centre
    frame.set_camera_pose(np.linalg.inv(W))


def _live_world():
    frames = [_SFrame(1, [0.0, 0.0, 0.0]), _SFrame(2, [1.0, 0.0, 0.0]), _SFrame(3, [2.0, 0.5, 0.0])]
    wm = Map(frames=frames)
    good_a = Point(np.array([0.0, 0.0, 5.0]), descriptor=np.ones(4, dtype=np.float32))
    good_b = Point(np.array([1.0, 1.0, 6.0]), descriptor=np.full(4, 2.0, dtype=np.float32))
    no_desc = Point(np.array([2.0, 0.0, 4.0]), descriptor=None)
    no_fin = Point(np.array([np.nan, 0.0, 4.0]), descriptor=np.ones(4, dtype=np.float32))
    no_obs = Point(np.array([0.5, 0.5, 3.0]), descriptor=np.ones(4, dtype=np.float32))
    for p in (good_a, no_desc, no_fin, no_obs, good_b):
        wm.add_point(p, frames)
    assert wm.link(frames[1], 0, good_a) and wm.link(frames[2], 0, good_a)
    assert wm.link(frames[2], 1, good_b)
    assert wm.link(frames[0], 2, no_desc) and wm.link(frames[0], 3, no_fin)
    return frames, wm, dict(a=good_a, b=good_b, no_desc=no_desc, no_fin=no_fin, no_obs=no_obs)


def _live_tracker(cls=map_tracker.MapTracker):
    tr = cls(_m1map(), "m1a")
    assert tr.live_map is False
    tr.live_map = True
    return tr


def test_l2_rows_only_for_usable_points():
    frames, wm, P = _live_world()
    tr = _live_tracker()
    n = tr.sync_live(frames, wm)
    assert n == 2
    assert tr._pid_of_row == [P["a"].id, P["b"].id]           # world_map.points order, the point ids
    assert tr._rows_pt == [P["a"], P["b"]]
    assert tr._row_of_point_id == {P["a"].id: 0, P["b"].id: 1}
    assert np.array_equal(tr._X_list[0], [0.0, 0.0, 5.0]) and np.array_equal(tr._X_list[1], [1.0, 1.0, 6.0])
    assert np.array_equal(tr._D_list[0], P["a"].descriptor) and tr._D_list[0] is not P["a"].descriptor
    assert tr._X_list[0] is not P["a"].point
    # reference frames: the first observation's frame (2 for a, 3 for b)
    assert np.array_equal(tr._ckf_list[0], [1.0, 0.0, 0.0]) and np.array_equal(tr._ckf_list[1], [2.0, 0.5, 0.0])
    assert tr._dirty is True
    X, D, C = tr._arrays()
    assert X.shape == (2, 3) and D.shape == (2, 4) and C.shape == (2, 3)
    assert tr.points_by_pid == {}


def test_l2_positions_and_reference_centres_read_live():
    frames, wm, P = _live_world()
    tr = _live_tracker()
    tr.sync_live(frames, wm)
    P["a"].point = np.array([0.1, 0.2, 5.5])
    _move(frames[1], [1.5, -0.5, 0.25])
    assert tr.sync_live(frames, wm) == 2
    assert np.array_equal(tr._X_list[0], [0.1, 0.2, 5.5])
    np.testing.assert_allclose(tr._ckf_list[0], [1.5, -0.5, 0.25], rtol=0, atol=1e-12)
    # the reference stays frame 2 after its observation is unlinked (the point is still seen by 3)
    assert wm.unlink(frames[1], 0)
    assert next(iter(P["a"].observations)) == 3
    _move(frames[1], [3.0, 3.0, 3.0])
    assert tr.sync_live(frames, wm) == 2
    np.testing.assert_allclose(tr._ckf_list[0], [3.0, 3.0, 3.0], rtol=0, atol=1e-12)
    # a point whose reference frame is not among the frames passed is skipped
    assert tr.sync_live([frames[0], frames[2]], wm) == 1
    assert tr._pid_of_row == [P["b"].id]


def test_l2_a_removed_point_leaves_the_table():
    frames, wm, P = _live_world()
    tr = _live_tracker()
    tr.sync_live(frames, wm)
    assert P["a"].id in tr._live_ref
    wm.remove_point(P["a"].id, frames)
    assert tr.sync_live(frames, wm) == 1
    assert tr._pid_of_row == [P["b"].id] and P["a"].id not in tr._row_of_point_id
    assert P["a"].id not in tr._live_ref
    # a point that later gains a descriptor joins with the reference fixed when it was first seen
    P["no_desc"].descriptor = np.zeros(4, dtype=np.float32)
    assert tr.sync_live(frames, wm) == 2
    assert tr._pid_of_row == [P["no_desc"].id, P["b"].id]
    np.testing.assert_allclose(tr._ckf_list[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_l2_switch_off_changes_nothing():
    for cls in (map_tracker.MapTracker, map_rotation.MapRotationTracker):
        fs = [_SFrame(1, [0.0, 0.0, 0.0]), _SFrame(2, [1.0, 0.0, 0.0])]
        fr = fs[1]
        recs = [(7, 1, 2, fr.KeyPoints[2].pt[0], fr.KeyPoints[2].pt[1], 0.5, 0.5, 4.0),
                (8, 1, 4, fr.KeyPoints[4].pt[0], fr.KeyPoints[4].pt[1], -0.5, 0.2, 6.0)]
        wm = Map(frames=fs)
        tr = cls(map_tracker.M1Map(header={}, points=recs, corr={}), "m1a")
        assert tr.live_map is False
        with contextlib.redirect_stdout(io.StringIO()):
            assert tr.release(1, fs, wm) == 2
        before = dict(pid=list(tr._pid_of_row), row=dict(tr._row_of_point_id), pt=list(tr._rows_pt),
                      X=[x.tobytes() for x in tr._X_list], D=[d.tobytes() for d in tr._D_list],
                      C=[c.tobytes() for c in tr._ckf_list], dirty=tr._dirty, by=dict(tr.points_by_pid))
        extra = Point(np.array([0.0, 0.0, 9.0]), observations={2: 0}, descriptor=np.ones(4, dtype=np.float32))
        wm.add_point(extra, fs)
        assert tr.sync_live(fs, wm) == 0
        after = dict(pid=list(tr._pid_of_row), row=dict(tr._row_of_point_id), pt=list(tr._rows_pt),
                     X=[x.tobytes() for x in tr._X_list], D=[d.tobytes() for d in tr._D_list],
                     C=[c.tobytes() for c in tr._ckf_list], dirty=tr._dirty, by=dict(tr.points_by_pid))
        assert after == before and before["pid"] == [7, 8]
        assert tr._live_ref == {}
        # with the switch on, the table is the map's (file pids replaced by point ids)
        tr.live_map = True
        assert tr.sync_live(fs, wm) == 3
        assert tr._pid_of_row == [p.id for p in wm.points]
        assert tr.points_by_pid == before["by"]


# =========================================================================== L3
class _KP:
    def __init__(self, u, v):
        self.pt = (float(u), float(v))


class _Frame:
    def __init__(self, fid, cam_T_world, pose_valid=True):
        self.id = fid
        self.cam_T_world = np.array(cam_T_world, dtype=float)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.keypoint_to_point = {}
        self.KeyPoints = []
        self.sigmas = []
        self.pose_valid = pose_valid
        self.set_calls = 0

    def observe(self, pid, u, v):
        idx = len(self.KeyPoints)
        self.KeyPoints.append(_KP(u, v))
        self.sigmas.append(1.0)
        self.keypoint_to_point[idx] = pid
        return idx

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=float)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.set_calls += 1
        return True


class _Point:
    def __init__(self, pid, xyz):
        self.id = pid
        self.point = np.array(xyz, dtype=float)
        self.x, self.y, self.z = map(float, self.point)
        self.frames = []
        self.observations = {}


class _Map:
    def __init__(self, points):
        self.points = list(points)
        self.by_id = {p.id: p for p in points}

    def get_point(self, pid):
        return self.by_id.get(pid)


def _sigma_fn(frame):
    return np.asarray(frame.sigmas, dtype=float)


def _project(cam_T_world, X):
    x = cam_T_world[:3, :3] @ X + cam_T_world[:3, 3]
    uv = K @ x
    return uv[0] / uv[2], uv[1] / uv[2]


def _make_scene(rng, n_cams, n_pts):
    pts = rng.uniform([-1.5, -1.0, 3.0], [1.5, 1.0, 7.0], size=(n_pts, 3))
    poses = []
    for c in range(n_cams):
        w = rng.normal(0, 0.3, 3) if c else np.zeros(3)
        t = rng.normal(0, 0.4, 3) if c else np.zeros(3)
        T = np.eye(4); T[:3, :3] = so3_exp(w); T[:3, 3] = t
        poses.append(T)
    return poses, pts


def _build_world(cam_T_world_list, pts, observers):
    frames = [_Frame(100 + c, T) for c, T in enumerate(cam_T_world_list)]
    points = [_Point(500 + p, pts[p]) for p in range(len(pts))]
    for c, f in enumerate(frames):
        for p in observers[c]:
            u, v = _project(f.cam_T_world, pts[p])
            idx = f.observe(points[p].id, u, v)
            points[p].frames.append(f.id)
            points[p].observations[f.id] = idx
    return frames, _Map(points)


def _kp_of(frame, pid):
    return next(kp for kp, q in frame.keypoint_to_point.items() if q == pid)


def _displace(frame, kp, du, dv):
    u, v = frame.KeyPoints[kp].pt
    frame.KeyPoints[kp] = _KP(u + du, v + dv)


def _perturb(frames, idx, scale=1.0):
    for c in idx:
        T = frames[c].cam_T_world.copy()
        T[:3, :3] = so3_exp(scale * np.array([0.01, -0.02, 0.015])) @ T[:3, :3]
        T[:3, 3] += scale * np.array([0.03, -0.01, 0.02])
        frames[c].set_camera_pose(T)
        frames[c].set_calls = 0


def _run(frames, wm, n_window, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames[:n_window], wm, K, all_frames=frames, **kw)
    return info, buf.getvalue()


def _thin_world(seed=3):
    """Window of 5 cameras (0..4) plus two fixed outer observers (5, 6), 60 points.  Window
    camera 3 observes 12 points (0..11); every other camera observes all 60.  Cameras 1..4
    perturbed from the truth."""
    rng = np.random.default_rng(seed)
    cams, pts = _make_scene(rng, 7, 60)
    obs = [list(range(60))] * 7
    obs[3] = list(range(12))
    frames, wm = _build_world(cams, pts, obs)
    truth = [f.cam_T_world.copy() for f in frames]
    _perturb(frames, (1, 2, 3, 4))
    return frames, wm, truth


def _p2_world(seed=11, n_window=4, n_thin_rows=35, n_bad=10):
    rng = np.random.default_rng(seed)
    n = n_window + 2
    cams, pts = _make_scene(rng, n, 60)
    obs = [list(range(60))] * n
    thin = n_window - 1
    obs[thin] = list(range(n_thin_rows))
    frames, wm = _build_world(cams, pts, obs)
    _perturb(frames, range(1, n_window), scale=0.2)
    for i in range(n_bad):
        pid = 500 + 2 * i
        kp = _kp_of(frames[thin], pid)
        a = 2.0 * np.pi * i / n_bad
        _displace(frames[thin], kp, 30.0 * np.cos(a), 30.0 * np.sin(a))
    return frames, wm


def test_l3_signatures():
    for fn in (bundle_adjust, build_ba_problem):
        p = inspect.signature(fn).parameters["camera_supported"]
        assert p.default is None and p.kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("m", [0, M])
def test_l3_rejected_camera_is_held(m):
    frames, wm, _ = _thin_world()
    before = [f.cam_T_world.copy() for f in frames]
    calls = []

    def pred(f):
        calls.append(f.id)
        return f.id != 102

    info, out = _run(frames, wm, 5, min_camera_points=m, camera_supported=pred)
    assert info["status"] == "completed" and out.count("BA completed") == 1
    assert calls == [101, 102, 103, 104]                 # the window cameras after index 0, in order
    assert frames[2].cam_T_world.tobytes() == before[2].tobytes() and frames[2].set_calls == 0
    assert info["held_frame_ids"] == [102] and info["n_held_build"] == 1
    # the predicate replaces the row test: camera 3 (12 rows) is free although 12 < 30
    assert 103 in info["free_frame_ids"] and frames[3].set_calls == 1
    assert info["free_min_rows"] == 12
    assert 102 in info["fixed_frame_ids"] and 102 not in info["free_frame_ids"]
    assert sorted(info["free_frame_ids"]) == [101, 103, 104]


def test_l3_all_rejected_is_skipped_and_writes_nothing():
    frames, wm, _ = _thin_world()
    poses0 = [f.cam_T_world.copy() for f in frames]
    pts0 = {p.id: p.point.copy() for p in wm.points}
    for kw in ({}, dict(min_camera_points=M), dict(min_camera_points=M, reject_outliers=True, sigma_fn=_sigma_fn)):
        info, out = _run(frames, wm, 5, camera_supported=lambda f: False, **kw)
        assert info["status"] == "skipped" and info["reason"] == "no_supported_camera", kw
        assert out.count("  >> BA skipped: no_supported_camera") == 1 and "BA completed" not in out
        assert info["n_held_build"] == 4
        for f, T in zip(frames, poses0):
            assert f.cam_T_world.tobytes() == T.tobytes() and f.set_calls == 0
        for p in wm.points:
            assert p.point.tobytes() == pts0[p.id].tobytes()
    # control: the same window without the predicate completes
    info, _ = _run(copy.deepcopy(frames), wm, 5)
    assert info["status"] == "completed"


def test_l3_gauge_fixing_the_last_supported_camera_is_skipped():
    # window 0..2 without outer observers; the predicate holds camera 2; the fixed cameras {0, 2} share 5
    # points (< 10 anchor points): two_fixed fixes camera 1, the last free camera
    rng = np.random.default_rng(31)
    cams, pts = _make_scene(rng, 3, 40)
    frames, wm = _build_world(cams, pts, [list(range(40)), list(range(40)), list(range(5))])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames, wm, K, min_point_observations=2, camera_supported=lambda f: f.id != 102)
    assert info["status"] == "skipped" and info["reason"] == "no_supported_camera"
    assert info["n_held_build"] == 1 and all(f.set_calls == 0 for f in frames)


@pytest.mark.parametrize("kw", [{}, dict(min_camera_points=M),
                                dict(min_camera_points=M, reject_outliers=True, sigma_fn=_sigma_fn)])
def test_l3_none_is_the_call_without_the_keyword(kw):
    fa, wa, _ = _thin_world()
    fb, wb = copy.deepcopy((fa, wa))
    info_a, out_a = _run(fa, wa, 5, **kw)
    info_b, out_b = _run(fb, wb, 5, camera_supported=None, **kw)
    assert info_a == info_b and out_a == out_b
    assert "camera_supported" not in info_a
    for f, g in zip(fa, fb):
        assert f.cam_T_world.tobytes() == g.cam_T_world.tobytes()
    for p in wa.points:
        assert p.point.tobytes() == wb.get_point(p.id).point.tobytes()


def test_l3_phase2_hold_with_a_predicate():
    frames, wm = _p2_world()
    info, out = _run(frames, wm, 4, reject_outliers=True, sigma_fn=_sigma_fn, min_camera_points=M,
                     camera_supported=lambda f: True)
    assert info["status"] == "completed" and out.count("BA completed") == 1
    assert info["n_held_build"] == 0                     # every camera supported at build time
    assert info["n_held_p2"] == 1 and info["p2_min_rows"] == 25 and info["p2_ran"] == 1
    # the phase-2 test still reads min_camera_points: without it nothing is held in phase 2
    f2, w2 = _p2_world()
    info2, _ = _run(f2, w2, 4, reject_outliers=True, sigma_fn=_sigma_fn, camera_supported=lambda f: True)
    assert info2["n_held_p2"] == 0 and info2["p2_ran"] == 1


# =========================================================================== L4
class _DFrame:
    """A keyframe for make_seeds: SIFT keypoints at octave 0, K, a pose, descriptors."""

    def __init__(self, fid, n_kp=40, seed=5):
        rng = np.random.default_rng(seed)
        self.id = fid
        self.K = np.asarray(DEFAULT_K, dtype=np.float64)
        self.method = "SIFT"
        self.img_shape = (480, 640)
        self.keypoint_to_point = {}
        self.KeyPoints = []
        for u, v in zip(rng.uniform(10, 630, n_kp), rng.uniform(10, 470, n_kp)):
            kp = cv2.KeyPoint(float(u), float(v), 3.0)
            kp.octave = 0
            self.KeyPoints.append(kp)
        self.Descriptors = rng.uniform(0, 255, (n_kp, 128)).astype(np.float32)
        self.cam_T_world = np.eye(4)
        self.world_T_cam = np.eye(4)
        self.pose_valid = True

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _pose(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, dtype=np.float64))
    T[:3, 3] = t
    return T


def _seed_state(s):
    return (s.mu, s.sigma2, s.a, s.b, s.rho_range, s.n_updates, list(s.measurements))


def _arm_with_seeds():
    ref_a, ref_b = _DFrame(11, seed=5), _DFrame(12, seed=6)
    ref_a.set_camera_pose(_pose([0.02, -0.01, 0.03], [0.1, 0.0, -0.2]))
    ref_b.set_camera_pose(_pose([-0.04, 0.05, 0.0], [0.6, -0.1, 0.1]))
    ref_a.keypoint_to_point[3] = 999                      # a linked keypoint gets no seed
    P = df.DepthFilterParams()
    arm = dfa.DepthFilterArm(P)
    arm.live = df.make_seeds(ref_a, 0, 3.0, P) + df.make_seeds(ref_b, 1, 4.0, P)
    for i, s in enumerate(arm.live):                      # a seed state that is not the prior
        s.mu, s.sigma2, s.a, s.b, s.n_updates = 0.3 + 0.001 * i, 1e-3, 12.0 + i, 9.0, 4
        s.measurements = [df.Measurement(1, None, i, 0.3, 1e-4, 0.9, 10.0)]
    return arm, ref_a, ref_b, P


def _expected(s):
    W = s.ref.world_T_cam
    r_cam = df._unit_rays_cam(s.ref.K, df._kp_xy(s.ref))[s.kp]
    return W[:3, 3], r_cam @ W[:3, :3].T


def _capture_update(monkeypatch, store):
    def fake_update(seeds, frame, traj_idx, params, refine=None):
        store.extend((s.centre.copy(), s.ray.copy(), _seed_state(s)) for s in seeds)
        return {key: 0 for key in df.COUNTER_KEYS}
    monkeypatch.setattr(dfa.df, "update_seeds", fake_update)


def test_l4_default_off():
    assert dfa.DepthFilterArm().follow_ref_pose is False
    assert "self.follow_ref_pose = False" in inspect.getsource(dfa.DepthFilterArm.__init__)


@pytest.mark.parametrize("follow", [True, False])
def test_l4_on_frame_refresh(monkeypatch, follow):
    arm, ref_a, ref_b, P = _arm_with_seeds()
    assert len(arm.live) == 39 + 40
    # make_seeds' own formula holds at seeding
    for s in arm.live:
        c, r = _expected(s)
        np.testing.assert_allclose(s.centre, c, rtol=0, atol=1e-12)
        np.testing.assert_allclose(s.ray, r, rtol=0, atol=1e-12)
    before = [(s.centre.tobytes(), s.ray.tobytes(), _seed_state(s)) for s in arm.live]
    # bundle adjustment moves both reference keyframes
    ref_a.set_camera_pose(_pose([0.05, 0.02, -0.04], [0.3, 0.2, -0.1]) @ ref_a.cam_T_world)
    ref_b.set_camera_pose(_pose([0.0, -0.07, 0.02], [-0.2, 0.1, 0.4]) @ ref_b.cam_T_world)
    arm.follow_ref_pose = follow
    store = []
    _capture_update(monkeypatch, store)
    frame = _DFrame(20, seed=9)
    with contextlib.redirect_stdout(io.StringIO()):
        arm.on_frame(5, frame, [(0, ref_a), (1, ref_b)], Map(frames=[ref_a, ref_b, frame]), [ref_a, ref_b, frame])
    assert len(store) == len(before)
    moved = 0
    for (centre, ray, state), (c0, r0, st0), s in zip(store, before, arm.live):
        assert state == st0                               # the filter's state is never touched
        if follow:
            c, r = _expected(s)
            np.testing.assert_allclose(centre, c, rtol=0, atol=1e-12)
            np.testing.assert_allclose(ray, r, rtol=0, atol=1e-12)
            assert abs(np.linalg.norm(ray) - 1.0) < 1e-12
            moved += int(centre.tobytes() != c0)
        else:
            assert centre.tobytes() == c0 and ray.tobytes() == r0
    if follow:
        assert moved == len(before)


def test_l4_on_keyframe_refreshes_first():
    arm, ref_a, ref_b, P = _arm_with_seeds()
    ref_b.set_camera_pose(_pose([0.03, 0.03, 0.03], [0.5, 0.5, 0.5]) @ ref_b.cam_T_world)
    arm.follow_ref_pose = True
    seen = []

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        seen.extend((s, s.centre.copy(), s.ray.copy(), _seed_state(s)) for s in arm.live)
        raise _Stop()

    arm._keyframe_step_d1 = stop
    with pytest.raises(_Stop):
        arm.on_keyframe(6, _DFrame(30), [(0, ref_a), (1, ref_b)], Map(), [ref_a, ref_b])
    assert len(seen) == 79
    for s, centre, ray, _st in seen:
        c, r = _expected(s)
        np.testing.assert_allclose(centre, c, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ray, r, rtol=0, atol=1e-12)


def test_l4_promotion_reads_the_followed_ray():
    # a converged seed of a keyframe that BA moved is promoted at mu along the moved ray
    arm, ref_a, ref_b, P = _arm_with_seeds()
    W_new = _pose([0.0, 0.1, 0.0], [1.0, 0.0, 0.0]) @ ref_a.cam_T_world
    ref_a.set_camera_pose(W_new)
    arm.follow_ref_pose = True
    arm._refresh_ref_poses()
    s = arm.live[0]
    c, r = _expected(s)
    np.testing.assert_allclose(df.seed_point(s), c + r / s.mu, rtol=0, atol=1e-12)


# =========================================================================== L5
class _OFrame:
    """Frame-like stub as in tests/test_opening.py."""

    def __init__(self, fid, R_cw, dirs, desc):
        Kd = np.asarray(DEFAULT_K, dtype=np.float64)
        self.id = fid
        self.K = Kd
        self.method = "SIFT"
        self.match_type = "knn"
        self.img_shape = (480, 640)
        self.pose_valid = True
        self.tracking_state = "OK"
        self.cam_T_world = np.eye(4)
        self.world_T_cam = np.eye(4)
        self.keypoint_to_point = {}
        self.KeyPoints = []
        q = dirs @ R_cw.T
        z = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (q @ Kd.T)[:, :2] / z[:, None]
        vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        for u, v in uv[vis]:
            kp = cv2.KeyPoint(float(u), float(v), 3.0)
            kp.octave = 0
            self.KeyPoints.append(kp)
        self.Descriptors = np.asarray(desc, dtype=np.float32)[vis]

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _yaw_scene(n_frames, seed=0, n_dirs=3000):
    rng = np.random.default_rng(seed)
    az = np.radians(rng.uniform(-50.0, 100.0, n_dirs))
    el = np.radians(rng.uniform(-55.0, 55.0, n_dirs))
    dirs = np.column_stack((np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)))
    desc = rng.uniform(0.0, 255.0, (n_dirs, 128)).astype(np.float32)
    return [_OFrame(k, _rot_y(float(k)).T, dirs, desc) for k in range(n_frames)]


def _accepted(n, seed=1):
    """Stand-in for an accepted two-view result: the InitResult fields the opening reads (only
    opening.py may import the initializer, check initializer_importers_04u)."""
    rng = np.random.default_rng(seed)
    R = _rot_y(-2.0)
    t = np.array([0.1, -0.2, 0.05])
    pts = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.5, 2.0, n)))
    idx = np.arange(n, dtype=np.int64)
    return SimpleNamespace(
        status="accepted", reason="none", model="F", rh=0.2, score_h=1.0, score_f=2.0, n_matches=n + 10,
        n_inliers=n + 5, good=(n, 1, 0, 0), chosen=0, parallax51_deg=2.5, n_triangulated=n, R=R, t=t,
        points=pts, match_idx=idx.copy(), kp_a=idx.copy(), kp_b=idx[::-1].copy(),
        parallax_deg=rng.uniform(1.2, 5.0, n), rows=3, cols=4, cells=12, ba_cost_initial=1.0,
        ba_cost_final=0.5, ba_iterations=3)


def _refusal_result(reason):
    """Stand-in for a refused result (no pose, no point), the fields as a fresh refusal has them."""
    return SimpleNamespace(
        status="refused", reason=reason, model="none", rh=math.nan, score_h=math.nan, score_f=math.nan,
        n_matches=0, n_inliers=0, good=(), chosen=-1, parallax51_deg=math.nan, n_triangulated=0, R=None,
        t=None, points=np.zeros((0, 3)), match_idx=np.zeros(0, dtype=np.int64),
        kp_a=np.zeros(0, dtype=np.int64), kp_b=np.zeros(0, dtype=np.int64), parallax_deg=np.zeros(0),
        rows=0, cols=0, cells=0, ba_cost_initial=math.nan, ba_cost_final=math.nan, ba_iterations=-1)


def _hand_off_at_1(monkeypatch, factory=None):
    frames = _yaw_scene(3)
    res = _accepted(40)
    # patched at the opening's level: its one initializer call returns the stand-ins
    monkeypatch.setattr(opening.OpeningTracker, "_attempt",
                        lambda self, a, k, frames_, refit: ((res if (a, k) == (0, 1)
                                                             else _refusal_result("parallax")), None))
    tr = opening.OpeningTracker(None)
    assert tr.tracker_factory is None
    tr.tracker_factory = factory
    tr.world_map = Map(frames=frames)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.start(frames)
        state = tr.on_frame(1, frames)
    return tr, frames, state, buf.getvalue().splitlines()


def test_l5_without_factory_the_04u_tracker(monkeypatch):
    calls = []
    real = map_tracker.MapTracker

    class _Rec(real):
        def __init__(self, *a, **k):
            calls.append((a, k))
            super().__init__(*a, **k)

    monkeypatch.setattr(map_tracker, "MapTracker", _Rec)
    tr, frames, state, lines = _hand_off_at_1(monkeypatch)
    assert state == "init" and len(calls) == 1
    a, k = calls[0]
    assert len(a) == 3 and isinstance(a[0], map_tracker.M1Map) and a[1:] == ("m1a", None) and k == {}
    assert len(a[0].points) == 40
    t = tr.tracker
    assert type(t) is _Rec and t.live_map is False
    assert (t.reloc, t.reloc_solver, t.reloc_shadow) == ("pnp", "epnp", False)
    assert len(tr.world_map.points) == 40


def test_l5_factory_called_once_with_the_m1map(monkeypatch):
    got = []

    def factory(m1map):
        got.append(m1map)
        t = map_rotation.MapRotationTracker(m1map, "m1a", None)
        t.variant = "r52"
        t.live_map = True
        return t

    released = []
    real_release = map_tracker.MapTracker.release

    def spy_release(self, k, frames, world_map):
        released.append(k)
        return real_release(self, k, frames, world_map)

    monkeypatch.setattr(map_tracker.MapTracker, "release", spy_release)
    tr, frames, state, lines = _hand_off_at_1(monkeypatch, factory)
    assert state == "init" and len(got) == 1
    assert isinstance(got[0], map_tracker.M1Map) and len(got[0].points) == 40
    assert got[0].points[0][1] == 1 and got[0].header == {} and got[0].corr == {}
    t = tr.tracker
    assert type(t) is map_rotation.MapRotationTracker and t.m1map is got[0]
    assert (t.reloc, t.reloc_solver, t.reloc_shadow) == ("pnp", "epnp", False)
    assert isinstance(t.reloc_params, map_tracker.MapRelocParams)
    assert released == [1] and len(t.points_by_pid) == 40 and len(tr.world_map.points) == 40
    assert t.live_map is True and t.variant == "r52"
    assert any("[MAP-ROT-R52-SETUP]" in l for l in lines)       # the factory's tracker printed its setup
    assert tr.state == "SURVIVAL"


# =========================================================================== L6
def test_l6_labels_and_writes():
    assert pipeline.M0_ONLINE_LABELS == {
        "measured": ("rotation", "ROT"), "unsupported": ("predicted", "LOST"), "lost": ("predicted", "LOST"),
        "init": ("init", "OK"), "tracked": ("map", "OK"), "rotation": ("rotation", "ROT")}
    assert pipeline.M1_ONLINE_LABELS == {"tracked": ("map", "OK"), "lost": ("predicted", "LOST"),
                                         "bridged": ("bridged", "BRIDGED"), "rotation": ("rotation", "ROT")}
    want = {"init": True, "tracked": True, "measured": False, "unsupported": False, "lost": False,
            "rotation": False, "bridged": False}
    for s, w in want.items():
        assert pipeline.m0_writes_map(s) is w, s


def test_l6_frame_line():
    assert pipeline.format_m0_frame_line(605, "init", True, True, 55, 55, 55, 0) == (
        "  >> [M0-FRAME] frame=605 state=init mapper=1 kf=1 points=55 init_points=55 new_points=0 inl_init=55 inl_new=0")
    assert pipeline.format_m0_frame_line(12, "rotation", False, False, 80, 50, 0, 0) == (
        "  >> [M0-FRAME] frame=12 state=rotation mapper=0 kf=0 points=80 init_points=50 new_points=30 inl_init=0 "
        "inl_new=0")


# =========================================================================== L7
def test_l7_clear_map():
    frames = [_SFrame(i, [float(i), 0.0, 0.0]) for i in range(1, 5)]
    wm = Map(frames=frames)
    pts = []
    for j in range(5):
        p = Point(np.array([0.1 * j, 0.0, 5.0]), descriptor=np.ones(4, dtype=np.float32))
        wm.add_point(p, frames)
        pts.append(p)
    for i, f in enumerate(frames):
        for j, p in enumerate(pts):
            if (i + j) % 2 == 0:
                assert wm.link(f, j, p)
    assert sum(len(f.keypoint_to_point) for f in frames) == 10
    assert sum(wm.check_invariants(frames).values()) == 0
    assert pipeline.m0_clear_map(wm, frames) == 5
    assert wm.points == [] and wm.point_by_id == {}
    assert all(f.keypoint_to_point == {} for f in frames)
    assert sum(wm.check_invariants(frames).values()) == 0
    assert all(p.observations == {} and p.frames == [] for p in pts)
    assert pipeline.m0_clear_map(wm, frames) == 0


# =========================================================================== L8
class _PFrame:
    """A pipeline frame: id, K, pose, keypoints, a link table."""

    def __init__(self, fid, n_kp=12):
        self.id = fid
        self.K = K
        self.method = "SIFT"
        self.img_shape = (480, 640)
        self.keypoint_to_point = {}
        self.KeyPoints = [cv2.KeyPoint(20.0 + 40.0 * i, 30.0 + 30.0 * i, 3.0) for i in range(n_kp)]
        self.Descriptors = np.zeros((n_kp, 4), dtype=np.float32)
        self.cam_T_world = np.eye(4)
        self.world_T_cam = np.eye(4)
        self.pose_valid = True
        self.tracking_state = "OK"

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


class _ScriptTracker:
    """The opening's hand-off tracker: records sync_live calls."""

    def __init__(self):
        self.synced = []

    def sync_live(self, frames, world_map):
        self.synced.append(len(world_map.points))
        return len(world_map.points)


class _ScriptOpening:
    """A scripted opening driver: per frame a state; an "init" frame hands off a map of n_init points
    linked to frame b; a "tracked" frame links one initial point and one other point; a "reset" frame
    removes the initial points as opening._on_survival does and reports "measured"."""

    def __init__(self, script, n_init=4):
        self.script = script
        self.n_init = n_init
        self.world_map = None
        self.use_homography = False
        self.tracker = None
        self.tracker_factory = None
        self.n_resets = 0
        self.n_maps = 0
        self.init_ids = []
        self.calls = []

    def setup_line(self):
        return "  >> [SCRIPT-SETUP]"

    def start(self, frames):
        pass

    def summary_line(self):
        return "  >> [SCRIPT-SUMMARY]"

    def on_frame(self, k, frames):
        self.calls.append((k, None if self.tracker is None else len(self.tracker.synced)))
        st = self.script[k]
        fr = frames[k]
        T = np.eye(4)
        T[:3, 3] = [0.01 * k, 0.0, 0.0]
        fr.set_camera_pose(T)
        if st == "init":
            self.n_maps += 1
            self.init_ids = []
            for i in range(self.n_init):
                p = Point(np.array([0.2 * i, 0.1, 4.0]), observations={fr.id: i},
                          descriptor=np.ones(4, dtype=np.float32))
                self.world_map.add_point(p, frames)
                self.init_ids.append(p.id)
            self.tracker = _ScriptTracker()
            fr.tracking_state = "OK"
            return "init"
        if st == "tracked":
            pts = [p for p in self.world_map.points]
            init = [p for p in pts if p.id in self.init_ids]
            other = [p for p in pts if p.id not in self.init_ids]
            if init:
                self.world_map.link(fr, 0, init[0])
            if other:
                self.world_map.link(fr, 1, other[-1])
            fr.tracking_state = "OK"
            return "tracked"
        if st == "reset":
            for pid in self.init_ids:
                self.world_map.remove_point(pid, frames)
            print(f"  >> [INIT-RESET] map={self.n_maps} frame={k}")
            self.n_resets += 1
            self.tracker = None
            fr.tracking_state = "ROT"
            return "measured"
        fr.tracking_state = {"measured": "ROT", "rotation": "ROT"}.get(st, "LOST")
        return st


def _fake_keyframe_step(created):
    def step(k, kf, keyframes, world_map, frames, lm_params, stats, refine=None):
        p = Point(np.array([1.0, 1.0, 5.0 + k]), descriptor=np.ones(4, dtype=np.float32))
        world_map.add_point(p, frames)
        world_map.link(kf, 5, p)
        created.append((k, [i for i, _ in keyframes]))
        print(f"  >> [LOCAL-MAP] kf={k} (stand-in)")
    return step


RX_FRAME = re.compile(r"^  >> \[M0-FRAME\] frame=(\d+) state=(\S+) mapper=([01]) kf=([01]) points=(\d+) "
                      r"init_points=(\d+) new_points=(-?\d+) inl_init=(\d+) inl_new=(\d+)$")
RX_ONLINE = re.compile(r"^  >> \[POSE-ONLINE\] frame=(\d+) source=(\S+) state=(\S+) ")


def _run_m0(monkeypatch, script, live_map="on", **kw):
    created = []
    monkeypatch.setattr(pipeline, "process_new_keyframe", _fake_keyframe_step(created))
    frames = [_PFrame(i + 1) for i in range(len(script))]
    op = _ScriptOpening(script)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "summary.txt"
        with contextlib.redirect_stdout(io.StringIO()):      # the progress line goes to the terminal
            pipeline.show_trajectory(frames, show_realtime=False, open_tracker=op, live_map=live_map,
                                     pnp_interval=0, use_loop_closure=False, use_relocalization=False,
                                     use_bundle_adjustment=False, use_point_fusion=False, keyframe_interval=5,
                                     log_file_path=log, **kw)
        lines = log.read_text().splitlines()
    return frames, op, created, lines


SCRIPT = (["-", "measured", "unsupported", "init", "tracked", "rotation", "tracked", "lost", "tracked", "tracked",
           "tracked", "tracked", "reset", "measured", "init", "tracked", "lost", "tracked"])


def test_l8_integrated_frame(monkeypatch):
    frames, op, created, lines = _run_m0(monkeypatch, SCRIPT)
    n = len(SCRIPT)
    i_setup = [i for i, l in enumerate(lines) if "[M0-SETUP]" in l]
    assert len(i_setup) == 1
    assert lines[i_setup[0]] == ("  >> [M0-SETUP] point_source=keyframe_pair map_rot_state=off "
                                 "ba_free_cameras=all ba_jump_gate=on ba_outlier_reject=off follow_ref_pose=0")
    fr_lines = [(i, RX_FRAME.match(l)) for i, l in enumerate(lines) if "[M0-FRAME]" in l]
    assert all(m for _, m in fr_lines) and [int(m.group(1)) for _, m in fr_lines] == list(range(1, n))
    assert fr_lines[0][0] > i_setup[0]
    rec = {int(m.group(1)): m.groups() for _, m in fr_lines}
    online = {int(m.group(1)): (m.group(2), m.group(3)) for m in (RX_ONLINE.match(l) for l in lines) if m}
    owner = {}
    start = i_setup[0] + 1
    for i, m in fr_lines:
        for j in range(start, i + 1):
            owner[j] = int(m.group(1))
        start = i + 1
    states = {1: "measured", 2: "unsupported", 3: "init", 4: "tracked", 5: "rotation", 6: "tracked", 7: "lost",
              8: "tracked", 9: "tracked", 10: "tracked", 11: "tracked", 12: "measured", 13: "measured",
              14: "init", 15: "tracked", 16: "lost", 17: "tracked"}
    for k, st in states.items():
        g = rec[k]
        assert g[1] == st, k
        assert online[k] == pipeline.M0_ONLINE_LABELS[st], k
        assert g[2] == ("1" if st in ("init", "tracked") else "0"), k
        assert int(g[6]) == int(g[4]) - int(g[5])
        # the [POSE-ONLINE] line of frame k sits in frame k's segment, before its [M0-FRAME] line
        i_on = next(i for i, l in enumerate(lines) if (RX_ONLINE.match(l) and int(RX_ONLINE.match(l).group(1)) == k))
        assert owner[i_on] == k
    # the keyframes: frame 0 is not one; the hand-off frames are; then the interval rule on mapper frames
    kf_frames = [k for k, g in rec.items() if g[3] == "1"]
    assert kf_frames == [3, 10, 14, 15]
    assert [k for k, _ in created] == [3, 10, 14, 15]
    assert created[0][1] == [3] and created[1][1] == [3, 10] and created[2][1] == [14] and created[3][1] == [14, 15]
    # the no-write rule: no mapper line, no link, no point change on frames the map did not pose
    write_tags = ("[LOCAL-MAP", "[M0-HANDOFF]", "BA completed", "BA skipped", "[FUSION]", "[DF-")
    for i, l in enumerate(lines):
        k = owner.get(i)
        if k is not None and rec[k][2] == "0":
            assert not any(t in l for t in write_tags), (k, l)
    prev = 0
    for k in range(1, n):
        g = rec[k]
        if g[2] == "0":
            assert int(g[7]) + int(g[8]) == 0, k
            if k != 12:
                assert int(g[4]) == prev, k
        prev = int(g[4])
    # the hand-offs
    ho = [l for l in lines if "[M0-HANDOFF]" in l]
    assert ho == ["  >> [M0-HANDOFF] frame=3 map=1 init_points=4", "  >> [M0-HANDOFF] frame=14 map=2 init_points=4"]
    assert rec[3][4:] == ("5", "4", "1", "4", "0")       # 4 initial points + the keyframe step's point
    assert rec[4][7:] == ("1", "1")
    assert rec[11][4:6] == ("6", "4")
    # the false start: the whole map goes, the keyframes go
    rs = [l for l in lines if "[M0-RESET]" in l]
    assert rs == ["  >> [M0-RESET] frame=12 map=1 removed=2 seeds_cleared=0 keyframes_cleared=2"]
    i_rs = lines.index(rs[0])
    assert owner[i_rs] == 12 and lines.index("  >> [INIT-RESET] map=1 frame=12") < i_rs
    assert rec[12][4:6] == ("0", "0") and rec[13][4:6] == ("0", "0")
    assert all(not f.keypoint_to_point for f in frames[:12])
    assert rec[17][4:] == ("6", "4", "2", "1", "1")
    # the supported flag: the map's own acceptance only
    assert [k for k in range(1, n) if getattr(frames[k], "m0_supported", False)] == \
        [k for k, st in states.items() if st == "tracked"]
    # sync_live before on_frame, on every frame the hand-off tracker exists for (none on the init frame)
    assert op.calls[2] == (3, None)
    assert op.calls[3] == (4, 1) and op.calls[4] == (5, 2) and op.calls[10] == (11, 8)
    assert op.calls[11] == (12, 9)                       # the reset frame: synced, then the tracker is dropped
    assert op.calls[12] == (13, None) and op.calls[13] == (14, None) and op.calls[14] == (15, 1)
    assert op.tracker.synced[0] == 5                     # the second map: 4 initial points + the hand-off's
    # the summary, after the opening's
    s = [l for l in lines if "[M0-SUMMARY]" in l]
    assert s == ["  >> [M0-SUMMARY] frames=18 mapper_frames=10 keyframes_added=4 handoffs=2 resets=1 points=6 "
                 "init_points=4"]
    assert lines.index(s[0]) == lines.index("  >> [SCRIPT-SUMMARY]") + 1
    assert not any("[M0-ERROR]" in l for l in lines)
    # the online snapshot of every frame is its pose at the end of the frame
    for k in range(1, n):
        assert frames[k].online_world_T_cam.tobytes() == frames[k].world_T_cam.tobytes()


def test_l8_switch_off_keeps_the_04u_path(monkeypatch):
    frames, op, created, lines = _run_m0(monkeypatch, ["-", "measured", "unsupported", "measured"], live_map="off")
    assert not any("[M0-" in l for l in lines)
    assert created == []
    assert online_states(lines) == {0: ("init", "OK"), 1: ("rotation", "ROT"), 2: ("predicted", "LOST"),
                                    3: ("rotation", "ROT")}


def online_states(lines):
    return {int(m.group(1)): (m.group(2), m.group(3)) for m in (RX_ONLINE.match(l) for l in lines) if m}


def test_l8_ba_predicate_reaches_bundle_adjust(monkeypatch):
    seen = []

    def fake_ba(frames_window, world_map, K_, **kw):
        seen.append(kw.get("camera_supported"))
        print("  >> BA skipped: no_supported_camera")
        return {"status": "skipped", "reason": "no_supported_camera"}

    monkeypatch.setattr(pipeline, "bundle_adjust", fake_ba)
    script = ["-", "measured", "init"] + ["tracked"] * 20
    common = dict(use_bundle_adjustment=True, ba_interval=10, ba_jump_gate="off")
    created = []
    monkeypatch.setattr(pipeline, "process_new_keyframe", _fake_keyframe_step(created))
    for cams, want in (("supported", True), ("all", False)):
        seen.clear()
        frames = [_PFrame(i + 1) for i in range(len(script))]
        op = _ScriptOpening(script)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.show_trajectory(frames, show_realtime=False, open_tracker=op, live_map="on", pnp_interval=0,
                                     use_loop_closure=False, use_relocalization=False, use_point_fusion=False,
                                     keyframe_interval=5, ba_free_cameras=cams, **common)
        assert len(seen) == 2                               # frames 10 and 20
        if want:
            pred = seen[0]
            assert callable(pred)
            assert pred(frames[10]) is True and pred(frames[2]) is False and pred(frames[0]) is False
        else:
            assert seen == [None, None]
    # the chain path never passes a predicate
    seen.clear()
    assert inspect.signature(pipeline.show_trajectory).parameters["live_map"].default == "off"
