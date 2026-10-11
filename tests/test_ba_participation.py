"""Stage 04z: local BA lets in only supported cameras (ruling R57-2, tests P1-P10).

conventions.md#local-ba-participation: a window camera observing fewer than 30 points in the BA
problem is held fixed like an outer observer (and counts as fixed for the gauge's anchor count);
the test is applied again after the outlier step's mid classification (a camera left with fewer
than 30 inlier rows keeps its phase-1 pose in phase 2); a solve with no free camera is skipped.
The jump gate in absolute map units becomes a switch; the R57-3 phase-2 instrument is logged on a
[BA-SUPPORT] line; an exception in the 04y outlier epilogue no longer revisits the accounting.
The synthetic stand-ins are copied from tests/test_ba_outlier_rejection.py (not imported from it).
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimization as opt  # noqa: E402
from optimization import build_ba_problem, bundle_adjust, so3_exp, solve_ba  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
M = opt.BA_MIN_FREE_CAMERA_POINTS


# --------------------------------------------------------------------------- #
# stand-ins (copied from tests/test_ba_outlier_rejection.py)
# --------------------------------------------------------------------------- #

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


def _pose_err1(frame, truth):
    D = frame.cam_T_world @ np.linalg.inv(truth)
    return float(np.linalg.norm(opt.so3_log(D[:3, :3]))) + float(np.linalg.norm(D[:3, 3]))


def _run(frames, wm, n_window, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames[:n_window], wm, K, all_frames=frames, **kw)
    return info, buf.getvalue()


def _spy_solve_ba(monkeypatch):
    calls = []
    real = opt.solve_ba

    def spy(problem, **kw):
        calls.append({"n_obs": problem.n_obs, "fixed_mask": problem.fixed_mask.copy(),
                      "max_iterations": kw["max_iterations"]})
        return real(problem, **kw)

    monkeypatch.setattr(opt, "solve_ba", spy)
    return calls


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
    """Window of n_window cameras plus two fixed outer observers, 60 points.  The last window
    camera observes n_thin_rows points (0 .. n_thin_rows - 1), n_bad of them displaced by 30 px in
    alternating directions (sigma 1 px); every other camera observes all 60 points without
    displacement.  Free window cameras perturbed slightly."""
    rng = np.random.default_rng(seed)
    n = n_window + 2
    cams, pts = _make_scene(rng, n, 60)
    obs = [list(range(60))] * n
    thin = n_window - 1
    obs[thin] = list(range(n_thin_rows))
    frames, wm = _build_world(cams, pts, obs)
    _perturb(frames, range(1, n_window), scale=0.2)
    bad = []
    for i in range(n_bad):
        pid = 500 + 2 * i
        kp = _kp_of(frames[thin], pid)
        a = 2.0 * np.pi * i / n_bad
        _displace(frames[thin], kp, 30.0 * np.cos(a), 30.0 * np.sin(a))
        bad.append((frames[thin].id, kp, pid))
    return frames, wm, bad


def _mid_classification(frames, wm, n_window, m):
    """The phase-1 state and the mid classification computed independently of bundle_adjust."""
    with contextlib.redirect_stdout(io.StringIO()):
        prob, meta = build_ba_problem(frames[:n_window], wm, K, all_frames=frames, min_camera_points=m)
    p1, _ = solve_ba(prob, method="lm", max_iterations=5)
    sig = opt._row_sigmas(meta["_cams"], meta["_obs_keys"], _sigma_fn)
    out_mid = opt.classify_ba_outliers(p1, sig)
    r2 = {c: int(np.count_nonzero((prob.cam_idx == c) & ~out_mid)) for c in prob.free_cam_indices}
    return prob, meta, p1, out_mid, r2


# --------------------------------------------------------------------------- #
# P1 parser, refusal, resolved settings, show_trajectory keywords
# --------------------------------------------------------------------------- #

def test_p1_parser_defaults_values_and_refusals(tmp_path):
    import main as m
    pr = m.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.ba_free_cameras == "all" and a.ba_jump_gate == "on"
    b = pr.parse_args(["--dataset", "rgb", "--ba_free_cameras", "supported", "--ba_jump_gate", "off"])
    assert b.ba_free_cameras == "supported" and b.ba_jump_gate == "off"
    assert pr.parse_args(["--dataset", "rgb", "--ba_free_cameras", "all"]).ba_free_cameras == "all"
    assert pr.parse_args(["--dataset", "rgb", "--ba_jump_gate", "on"]).ba_jump_gate == "on"
    for flag, bad in (("--ba_free_cameras", "thin"), ("--ba_free_cameras", "SUPPORTED"),
                      ("--ba_free_cameras", ""), ("--ba_jump_gate", "maybe"), ("--ba_jump_gate", "1"),
                      ("--ba_jump_gate", "OFF")):
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            pr.parse_args(["--dataset", "rgb", flag, bad])
    helps = {a.dest: a.help for a in pr._actions}
    help_text = helps["ba_free_cameras"] + " " + helps["ba_jump_gate"]
    assert "R57-2" in help_text and "conventions.md#local-ba-participation" in help_text
    assert "absolute map units" in help_text and "byte identity" in help_text
    out = tmp_path / "resolved_settings.json"
    m.write_resolved_settings(b, out)
    s = json.load(open(out))
    assert s["ba_free_cameras"] == "supported" and s["ba_jump_gate"] == "off"
    m.write_resolved_settings(a, out)
    s = json.load(open(out))
    assert s["ba_free_cameras"] == "all" and s["ba_jump_gate"] == "on"


def test_p1_show_trajectory_keywords_and_hand_off():
    import main as m
    import pipeline as pl
    sp = inspect.signature(pl.show_trajectory).parameters
    assert sp["ba_free_cameras"].default == "all"
    assert sp["ba_jump_gate"].default == "on"
    src = inspect.getsource(m.run_pipeline)
    assert "ba_free_cameras=getattr(args, 'ba_free_cameras', 'all')" in src
    assert "ba_jump_gate=getattr(args, 'ba_jump_gate', 'on')" in src
    assert src.index("validate_ba_args(args)") < src.index("write_resolved_settings(")


def test_p1_validate_ba_args():
    import main as m
    NS = types.SimpleNamespace
    with pytest.raises(SystemExit) as e:
        m.validate_ba_args(NS(ba_free_cameras="supported", ba_jump_gate="on"))
    assert str(e.value) == "--ba_free_cameras supported refuses --ba_jump_gate on"
    with pytest.raises(SystemExit):
        m.validate_ba_args(NS(ba_free_cameras="supported"))     # gate read with default on
    for cams, gate in (("all", "on"), ("all", "off"), ("supported", "off")):
        assert m.validate_ba_args(NS(ba_free_cameras=cams, ba_jump_gate=gate)) is None
    assert m.validate_ba_args(NS()) is None
    assert m.validate_ba_args(NS(ba_jump_gate="off")) is None


def test_p1_run_pipeline_refuses_before_anything_runs(monkeypatch):
    import main as m
    pr = m.get_parser()
    a = pr.parse_args(["--dataset", "rgb", "--ba_free_cameras", "supported"])

    def boom(*x, **k):
        raise AssertionError("pipeline work started before the refusal")

    monkeypatch.setattr(m, "iter_frames", boom, raising=False)
    monkeypatch.setattr(m, "write_resolved_settings", boom)
    with pytest.raises(SystemExit) as e:
        m.run_pipeline(a)
    assert "refuses --ba_jump_gate on" in str(e.value)


# --------------------------------------------------------------------------- #
# P2 m = 0 is today's code
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reject", [False, True])
def test_p2_zero_equals_call_without_keyword(reject):
    frames_a, wm_a, _ = _thin_world()
    frames_b, wm_b = copy.deepcopy((frames_a, wm_a))
    kw = dict(reject_outliers=True, sigma_fn=_sigma_fn) if reject else {}
    info_a, out_a = _run(frames_a, wm_a, 5, **kw)
    info_b, out_b = _run(frames_b, wm_b, 5, min_camera_points=0, **kw)
    assert info_a["status"] == "completed"
    assert info_a == info_b
    assert out_a == out_b and out_a.count("BA completed") == 1
    assert info_a["held_frame_ids"] == [] and info_a["n_held_build"] == 0
    assert info_a["min_camera_points"] == 0
    assert info_a["free_min_rows"] == 12            # computed for every m: the thin camera is free
    assert "_cams" not in info_a and "_obs_keys" not in info_a
    assert not any(k.startswith("_") for k in info_a)
    for fa, fb in zip(frames_a, frames_b):
        assert fa.cam_T_world.tobytes() == fb.cam_T_world.tobytes()
        assert fa.set_calls == fb.set_calls
    for pa, pb in zip(wm_a.points, wm_b.points):
        assert pa.point.tobytes() == pb.point.tobytes()
    # the thin camera is free and moved when the rule is off
    assert frames_a[3].set_calls == 1


def test_p2_signatures():
    for fn in (opt.build_ba_problem, opt.bundle_adjust):
        p = inspect.signature(fn).parameters["min_camera_points"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY and p.default == 0
    assert type(opt.BA_MIN_FREE_CAMERA_POINTS) is int and opt.BA_MIN_FREE_CAMERA_POINTS == 30
    assert list(inspect.signature(opt.solve_ba).parameters)[0] == "problem"
    assert "min_camera_points" not in inspect.signature(opt.solve_ba).parameters
    assert "min_camera_points" not in inspect.signature(opt.BAProblem.__init__).parameters
    src = inspect.getsource(opt)
    for tok in ("import local_mapping", "from local_mapping", ".unlink(", "keypoint_to_point.pop(",
                "groundtruth", "depth.txt"):
        assert tok not in src


# --------------------------------------------------------------------------- #
# P3 a thin camera is held
# --------------------------------------------------------------------------- #

def test_p3_thin_camera_is_held():
    frames, wm, truth = _thin_world()
    before = [f.cam_T_world.copy() for f in frames]
    err0 = {c: _pose_err1(frames[c], truth[c]) for c in (1, 2, 4)}
    info, out = _run(frames, wm, 5, min_camera_points=M)
    assert info["status"] == "completed" and out.count("BA completed") == 1
    assert frames[3].cam_T_world.tobytes() == before[3].tobytes()
    assert frames[3].set_calls == 0
    assert info["held_frame_ids"] == [frames[3].id]
    assert info["n_held_build"] == 1
    assert info["free_min_rows"] >= M
    assert info["min_camera_points"] == M
    assert frames[3].id in info["fixed_frame_ids"] and frames[3].id not in info["free_frame_ids"]
    assert info["n_free"] == 3
    assert info["gauge"] == "fixed_observers"
    for c in (1, 2, 4):
        assert frames[c].set_calls == 1
        assert _pose_err1(frames[c], truth[c]) < err0[c], c
    # the held camera keeps contributing rows (as an outer observer would)
    assert info["n_obs"] == 6 * 60 + 12


def test_p3_threshold_is_strict_less_than():
    # a camera with exactly m rows is free; with m + 1 required it is held
    frames, wm, _ = _thin_world()
    with contextlib.redirect_stdout(io.StringIO()):
        _, meta12 = build_ba_problem(frames[:5], wm, K, all_frames=frames, min_camera_points=12)
        _, meta13 = build_ba_problem(frames[:5], wm, K, all_frames=frames, min_camera_points=13)
    assert meta12["held_frame_ids"] == [] and meta12["free_min_rows"] == 12
    assert meta13["held_frame_ids"] == [frames[3].id] and meta13["free_min_rows"] == 60


# --------------------------------------------------------------------------- #
# P4 a held camera anchors the gauge
# --------------------------------------------------------------------------- #

def test_p4_held_camera_anchors_the_gauge():
    # window 0..3; camera 3 thin (12 points, all also seen by camera 0); one outer observer (4)
    # sees five points 20..24.  Without the rule the fixed cameras {0, 4} share 5 points
    # (< min_anchor_points 10): two_fixed.  With the rule camera 3 is held and counts as fixed:
    # 5 + 12 = 17 anchor points: fixed_observers.
    rng = np.random.default_rng(17)
    cams, pts = _make_scene(rng, 5, 50)
    obs = [list(range(50))] * 3 + [list(range(12)), list(range(20, 25))]
    frames, wm = _build_world(cams, pts, obs)
    with contextlib.redirect_stdout(io.StringIO()):
        p_off, meta_off = build_ba_problem(frames[:4], wm, K, all_frames=frames)
        p_on, meta_on = build_ba_problem(frames[:4], wm, K, all_frames=frames, min_camera_points=M)
    assert meta_off["gauge"] == "two_fixed" and meta_off["n_anchor_points"] == 5
    assert meta_on["gauge"] == "fixed_observers" and meta_on["n_anchor_points"] == 17
    assert meta_off["fixed_frame_ids"] == [100, 101, 104]
    assert meta_on["fixed_frame_ids"] == [100, 103, 104]
    assert meta_on["free_frame_ids"] == [101, 102]


# --------------------------------------------------------------------------- #
# P5 no free camera
# --------------------------------------------------------------------------- #

def _assert_skipped_untouched(frames, wm, info, out, poses0, pts0):
    assert info["status"] == "skipped" and info["reason"] == "no_supported_camera"
    assert out.count("  >> BA skipped: no_supported_camera") == 1
    assert "BA completed" not in out
    for f, T in zip(frames, poses0):
        assert f.cam_T_world.tobytes() == T.tobytes() and f.set_calls == 0
    for p in wm.points:
        assert p.point.tobytes() == pts0[p.id].tobytes()


def test_p5_every_window_camera_thin_is_skipped():
    # window 0..2; cameras 1 and 2 see 20 points each; two outer observers see all 40
    rng = np.random.default_rng(23)
    cams, pts = _make_scene(rng, 5, 40)
    obs = [list(range(40)), list(range(20)), list(range(20, 40)), list(range(40)), list(range(40))]
    frames, wm = _build_world(cams, pts, obs)
    _perturb(frames, (1, 2))
    poses0 = [f.cam_T_world.copy() for f in frames]
    pts0 = {p.id: p.point.copy() for p in wm.points}
    for kw in ({}, dict(reject_outliers=True, sigma_fn=_sigma_fn)):
        info, out = _run(frames, wm, 3, min_camera_points=M, **kw)
        _assert_skipped_untouched(frames, wm, info, out, poses0, pts0)
    # control: the same scene without the rule completes
    info, out = _run(copy.deepcopy(frames), wm, 3)
    assert info["status"] == "completed"


def test_p5_gauge_fixing_the_last_free_camera_is_skipped():
    # window 0..2 without outer observers; camera 2 sees 5 points (held), so the fixed cameras
    # {0, 2} share 5 anchor points (< 10): two_fixed fixes camera 1 (40 rows, free at the hold),
    # the last free camera.  min_point_observations=2 so that camera 1's 40 points are selected.
    rng = np.random.default_rng(31)
    cams, pts = _make_scene(rng, 3, 40)
    obs = [list(range(40)), list(range(40)), list(range(5))]
    frames, wm = _build_world(cams, pts, obs)
    poses0 = [f.cam_T_world.copy() for f in frames]
    pts0 = {p.id: p.point.copy() for p in wm.points}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames, wm, K, min_point_observations=2, all_frames=None, min_camera_points=M)
    _assert_skipped_untouched(frames, wm, info, buf.getvalue(), poses0, pts0)
    assert info["n_held_build"] == 1                 # only camera 2 was held; the gauge fixed camera 1
    # control: without the rule the same window builds with two_fixed and camera 2 free
    with contextlib.redirect_stdout(io.StringIO()):
        _, meta = build_ba_problem(frames, wm, K, min_point_observations=2, all_frames=None)
    assert meta["gauge"] == "two_fixed" and meta["free_frame_ids"] == [102]


def test_p5_all_held_is_skipped_before_the_gauge():
    # window of two cameras without outer observers, camera 1 thin (5 points): the gauge cannot
    # be anchored even with camera 1 counted as fixed (5 anchor points < 10, gauge_unanchored);
    # with the rule the build stops before the gauge step, no_supported_camera
    rng = np.random.default_rng(37)
    cams, pts = _make_scene(rng, 2, 20)
    frames, wm = _build_world(cams, pts, [list(range(20)), list(range(5))])
    poses0 = [f.cam_T_world.copy() for f in frames]
    pts0 = {p.id: p.point.copy() for p in wm.points}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames, wm, K, min_point_observations=1, min_camera_points=M)
    _assert_skipped_untouched(frames, wm, info, buf.getvalue(), poses0, pts0)
    with contextlib.redirect_stdout(io.StringIO()):
        _, meta = build_ba_problem(frames, wm, K, min_point_observations=1)
    assert meta["status"] == "skipped" and meta["reason"] == "gauge_unanchored"


# --------------------------------------------------------------------------- #
# P6 phase-2 hold
# --------------------------------------------------------------------------- #

def test_p6_phase2_hold(monkeypatch):
    frames, wm, bad = _p2_world()
    _, meta, p1, out_mid, r2 = _mid_classification(copy.deepcopy(frames), copy.deepcopy(wm), 4, M)
    thin = 3
    # preconditions of the scene: exactly the 10 displaced rows of the thin camera are mid
    # outliers there, and it is the thinnest free camera
    assert r2[thin] == 25, r2
    assert all(v >= M for c, v in r2.items() if c != thin), r2
    calls = _spy_solve_ba(monkeypatch)
    info, out = _run(frames, wm, 4, reject_outliers=True, sigma_fn=_sigma_fn, min_camera_points=M)
    assert info["status"] == "completed" and out.count("BA completed") == 1
    assert info["n_held_build"] == 0                 # 35 rows at build: free
    assert info["n_held_p2"] == 1
    assert info["p2_min_rows"] == 25
    assert info["p2_under10"] == 0
    assert info["p2_ran"] == 1
    assert len(calls) == 2
    fm1, fm2 = calls[0]["fixed_mask"], calls[1]["fixed_mask"]
    assert not fm1[thin] and fm2[thin]               # held for phase 2 only
    for c in (1, 2):
        assert not fm1[c] and not fm2[c]             # the other free cameras stay free
    assert fm2[0] and np.all(fm2[4:])
    # a phase-2-held camera is written with its phase-1 pose
    assert frames[thin].set_calls == 1
    np.testing.assert_allclose(frames[thin].cam_T_world, p1.poses()[thin], rtol=0, atol=1e-12)
    assert set(bad) <= set(info["outlier_obs"])


def test_p6_without_rule_the_camera_stays_free_in_phase2(monkeypatch):
    frames, wm, _ = _p2_world()
    calls = _spy_solve_ba(monkeypatch)
    info, _ = _run(frames, wm, 4, reject_outliers=True, sigma_fn=_sigma_fn)
    assert info["n_held_p2"] == 0 and info["p2_min_rows"] == 25 and info["p2_ran"] == 1
    assert len(calls) == 2 and not calls[1]["fixed_mask"][3]


# --------------------------------------------------------------------------- #
# P7 phase 2 with no free camera left is skipped
# --------------------------------------------------------------------------- #

def test_p7_phase2_without_free_camera_is_skipped(monkeypatch):
    # window of two cameras: camera 1 (the only free one) has 35 rows, 10 displaced
    frames, wm, _ = _p2_world(seed=13, n_window=2)
    ref_frames, ref_wm = copy.deepcopy((frames, wm))
    prob, meta, p1, out_mid, r2 = _mid_classification(ref_frames, ref_wm, 2, M)
    assert r2 == {1: 25}, r2
    calls = _spy_solve_ba(monkeypatch)
    info, out = _run(frames, wm, 2, reject_outliers=True, sigma_fn=_sigma_fn, min_camera_points=M)
    assert info["status"] == "completed" and out.count("BA completed") == 1
    assert len(calls) == 1 and calls[0]["max_iterations"] == 5
    assert info["p2_ran"] == 0 and info["n_held_p2"] == 1 and info["p2_min_rows"] == 25
    # the written state is the phase-1 state
    assert frames[1].cam_T_world.tobytes() == p1.poses()[1].tobytes()
    pts1 = p1.points()
    for j, pid in enumerate(meta["point_ids"]):
        assert wm.get_point(pid).point.tobytes() == pts1[j].tobytes()
    # control: without the rule phase 2 runs on the same scene
    calls.clear()
    frames2, wm2, _ = _p2_world(seed=13, n_window=2)
    info2, _ = _run(frames2, wm2, 2, reject_outliers=True, sigma_fn=_sigma_fn)
    assert info2["p2_ran"] == 1 and len(calls) == 2


# --------------------------------------------------------------------------- #
# P8 the R57-3 instrument
# --------------------------------------------------------------------------- #

def test_p8_instrument_with_rule_off():
    # the last window camera: 12 rows, 4 displaced -> 8 inlier rows (< 10) after the mid step
    frames, wm, _ = _p2_world(seed=19, n_thin_rows=12, n_bad=4)
    _, _, _, out_mid, r2 = _mid_classification(copy.deepcopy(frames), copy.deepcopy(wm), 4, 0)
    want_min = min(r2.values())
    want_u10 = sum(1 for v in r2.values() if v < 10)
    assert want_u10 >= 1                              # the probe is not vacuous
    info, _ = _run(frames, wm, 4, reject_outliers=True, sigma_fn=_sigma_fn, min_camera_points=0)
    assert info["p2_min_rows"] == want_min
    assert info["p2_under10"] == want_u10
    assert info["n_outliers_mid"] == int(np.count_nonzero(out_mid))
    assert info["n_held_p2"] == 0 and info["p2_ran"] == 1


def test_p8_instrument_without_outlier_step():
    frames, wm, _ = _p2_world(seed=19, n_thin_rows=12, n_bad=4)
    for m in (0, M):
        f2, w2 = copy.deepcopy((frames, wm))
        info, _ = _run(f2, w2, 4, min_camera_points=m)
        assert info["status"] == "completed"
        assert (info["p2_min_rows"], info["p2_under10"], info["n_held_p2"], info["p2_ran"]) == (-1, -1, 0, 0)


# --------------------------------------------------------------------------- #
# P9 the [BA-SUPPORT] line
# --------------------------------------------------------------------------- #

def test_p9_format_ba_support_line():
    import pipeline as pl
    assert pl.format_ba_support_line(120, 4, 7, 31, 1, 2, 12, 1) == (
        "  >> [BA-SUPPORT] frame=120 free=4 held=7 min_free_rows=31 p2_ran=1 held_p2=2 "
        "p2_min_rows=12 p2_under10=1")
    assert pl.format_ba_support_line(5, 9, 0, 40, False, 0, -1, -1) == (
        "  >> [BA-SUPPORT] frame=5 free=9 held=0 min_free_rows=40 p2_ran=0 held_p2=0 "
        "p2_min_rows=-1 p2_under10=-1")
    assert pl.format_ba_support_line(5, 9, 0, 40, True, 0, -1, -1).count("p2_ran=1") == 1


def test_p9_pipeline_wiring_tokens():
    import pipeline as pl
    src = inspect.getsource(pl.show_trajectory)
    assert "optimization.BA_MIN_FREE_CAMERA_POINTS" in src and "min_camera_points=min_cam_pts" in src
    assert 'bad_jump = (ba_jump_gate == "on") and post_max_step > max_allowed_step' in src
    i_out = src.index("print(ba_outlier_epilogue(")
    i_sup = src.index("print(format_ba_support_line(")
    i_after = src.index("Reprojection error AFTER")
    assert i_out < i_sup < i_after


# --------------------------------------------------------------------------- #
# P10 the never-raising outlier epilogue
# --------------------------------------------------------------------------- #

class _TableFrame:
    def __init__(self, fid):
        self.id = fid
        self.keypoint_to_point = {}


def _real_map():
    from Map import Map
    from Point import Point
    frames = [_TableFrame(10), _TableFrame(11), _TableFrame(12)]
    world = Map(frames=frames)
    pts = [Point(point=np.array([0.0, 0.0, 3.0 + i])) for i in range(3)]
    for p in pts:
        world.add_point(p)
    for f in frames:
        for i, p in enumerate(pts):
            assert world.link(f, i, p)
    return world, frames, pts


def test_p10_epilogue_accepted_and_rolled_back():
    import pipeline as pl
    world, frames, pts = _real_map()
    info = {"n_outliers_mid": 3, "n_outliers_end": 2,
            "outlier_obs": [(10, 0, pts[0].id), (11, 1, pts[1].id)]}
    # rolled back: nothing removed
    line = pl.ba_outlier_epilogue(world, frames, 7, 9, info, False)
    assert line == "  >> [BA-OUTLIERS] frame=7 obs=9 mid=3 end=2 accepted=0 removed=0 inv_total=0"
    assert frames[0].keypoint_to_point[0] == pts[0].id
    # accepted: the listed links are removed through Map.unlink
    line = pl.ba_outlier_epilogue(world, frames, 7, 9, info, True)
    assert line == "  >> [BA-OUTLIERS] frame=7 obs=9 mid=3 end=2 accepted=1 removed=2 inv_total=0"
    assert 0 not in frames[0].keypoint_to_point and 10 not in pts[0].observations
    assert 1 not in frames[1].keypoint_to_point and 11 not in pts[1].observations
    assert sum(world.check_invariants(frames).values()) == 0


def test_p10_epilogue_never_raises(monkeypatch):
    import pipeline as pl
    world, frames, pts = _real_map()
    info = {"n_outliers_mid": 1, "n_outliers_end": 1, "outlier_obs": [(10, 0, pts[0].id)]}

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "apply_ba_outlier_removal", boom)
    line = pl.ba_outlier_epilogue(world, frames, 7, 9, info, True)
    assert line.startswith("  >> [BA-OUTLIER-ERROR] frame=7: ") and "boom" in line
    monkeypatch.undo()
    # an exception after the removal: links already removed stay removed, tables consistent
    monkeypatch.setattr(world, "check_invariants", boom)
    line = pl.ba_outlier_epilogue(world, frames, 7, 9, info, True)
    assert line.startswith("  >> [BA-OUTLIER-ERROR] frame=7: ") and "boom" in line
    monkeypatch.undo()
    assert 0 not in frames[0].keypoint_to_point and 10 not in pts[0].observations
    assert sum(world.check_invariants(frames).values()) == 0


def test_p10_pipeline_handler_does_not_revisit_recorded_outcome():
    import pipeline as pl
    src = inspect.getsource(pl.show_trajectory)
    i_rec_rej = src.index('record_ba_outcome(stats, "completed", accepted=False)')
    i_rec_acc = src.index('record_ba_outcome(stats, "completed", accepted=True)')
    i_except = src.index("except Exception as e:\n                if outcome_recorded:")
    assert src.index("outcome_recorded = True", i_rec_rej) < i_rec_acc
    assert src.index("outcome_recorded = True", i_rec_acc) < src.index("print_correction(", i_rec_acc)
    assert i_except > i_rec_acc
