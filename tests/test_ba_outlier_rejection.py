"""Stage 04y: local-BA outlier rejection behind --ba_outlier_reject (spec A, tests D1-D9).

ORB-SLAM section VI.D: observations whose sigma-normalized squared residual exceeds 5.991
(or whose point is behind its camera) are dropped at the middle of the solve and classified
again at the end; the final outliers are unlinked from the map on an accepted solve.  The
synthetic stand-ins are copied from tests/test_ba_chart.py (not imported from it).
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimization as opt  # noqa: E402
from optimization import BAProblem, build_ba_problem, bundle_adjust, so3_exp, solve_ba  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# stand-ins (copied from tests/test_ba_chart.py, plus a per-keypoint sigma array)
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


def _d3_world(seed=29, sigma_at_outlier=1.0):
    """5 cameras (window of 4 plus one fixed outer observer), 12 points, sigma 1 px; frames 2
    and 3 (free) perturbed; frame 2's observation of point 500 displaced by 30 px."""
    rng = np.random.default_rng(seed)
    cams, pts = _make_scene(rng, 5, 12)
    frames, wm = _build_world(cams, pts, [list(range(12))] * 5)
    truth = [f.cam_T_world.copy() for f in frames]
    for c in (2, 3):
        T = frames[c].cam_T_world.copy()
        T[:3, :3] = so3_exp([0.01, -0.02, 0.015]) @ T[:3, :3]
        T[:3, 3] += [0.03, -0.01, 0.02]
        frames[c].set_camera_pose(T)
        frames[c].set_calls = 0
    kp = _kp_of(frames[2], 500)
    _displace(frames[2], kp, 30.0, 0.0)
    frames[2].sigmas[kp] = float(sigma_at_outlier)
    return frames, wm, truth, (frames[2].id, kp, 500)


def _pose_err(frames, truth, idx=(1, 2, 3)):
    tot = 0.0
    for c in idx:
        D = frames[c].cam_T_world @ np.linalg.inv(truth[c])
        tot += float(np.linalg.norm(opt.so3_log(D[:3, :3]))) + float(np.linalg.norm(D[:3, 3]))
    return tot


def _run(frames, wm, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        info = bundle_adjust(frames[:4], wm, K, all_frames=frames, **kw)
    return info, buf.getvalue()


# --------------------------------------------------------------------------- #
# D1 parser, resolved settings, show_trajectory keyword
# --------------------------------------------------------------------------- #

def test_d1_parser_and_settings(tmp_path):
    import main as m
    import pipeline as pl
    pr = m.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.ba_outlier_reject == "off"
    b = pr.parse_args(["--dataset", "rgb", "--ba_outlier_reject", "on"])
    assert b.ba_outlier_reject == "on"
    assert pr.parse_args(["--dataset", "rgb", "--ba_outlier_reject", "off"]).ba_outlier_reject == "off"
    for bad in ("paper", "ON", "1", ""):
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            pr.parse_args(["--dataset", "rgb", "--ba_outlier_reject", bad])
    help_text = pr.format_help()
    assert "VI.D" in help_text
    out = tmp_path / "resolved_settings.json"
    m.write_resolved_settings(b, out)
    assert json.load(open(out))["ba_outlier_reject"] == "on"
    m.write_resolved_settings(a, out)
    assert json.load(open(out))["ba_outlier_reject"] == "off"
    p = inspect.signature(pl.show_trajectory).parameters["ba_outlier_reject"]
    assert p.default == "off"
    src = inspect.getsource(m.run_pipeline)
    assert "ba_outlier_reject=getattr(args, 'ba_outlier_reject', 'off')" in src


# --------------------------------------------------------------------------- #
# D2 off is today's code
# --------------------------------------------------------------------------- #

def test_d2_switch_off_equals_call_without_keywords():
    frames_a, wm_a, _, _ = _d3_world()
    frames_b, wm_b = copy.deepcopy((frames_a, wm_a))
    info_a, out_a = _run(frames_a, wm_a)
    info_b, out_b = _run(frames_b, wm_b, reject_outliers=False, sigma_fn=None)
    assert info_a == info_b
    assert out_a == out_b and out_a.count("BA completed") == 1
    assert "outlier_obs" not in info_a and "_obs_keys" not in info_a and "_cams" not in info_a
    for fa, fb in zip(frames_a, frames_b):
        assert fa.cam_T_world.tobytes() == fb.cam_T_world.tobytes()
        assert fa.set_calls == fb.set_calls
    for pa, pb in zip(wm_a.points, wm_b.points):
        assert pa.point.tobytes() == pb.point.tobytes()


def test_d2_obs_keys_follow_residual_rows():
    frames, wm, _, _ = _d3_world()
    prob, meta = build_ba_problem(frames[:4], wm, K, all_frames=frames)
    keys = meta["_obs_keys"]
    assert len(keys) == prob.n_obs
    cams = meta["_cams"]
    for k, (c, fid, kp, pid) in enumerate(keys):
        assert prob.cam_idx[k] == c and cams[c].id == fid
        assert meta["point_ids"][prob.pt_idx[k]] == pid
        assert cams[c].keypoint_to_point[kp] == pid
        assert tuple(prob.uv[k]) == cams[c].KeyPoints[kp].pt


# --------------------------------------------------------------------------- #
# D3 a gross outlier is found and the free poses end closer to the truth
# --------------------------------------------------------------------------- #

def test_d3_gross_outlier_rejected():
    frames_on, wm_on, truth, bad = _d3_world()
    frames_off, wm_off = copy.deepcopy((frames_on, wm_on))
    info_off, _ = _run(frames_off, wm_off)
    info_on, out = _run(frames_on, wm_on, reject_outliers=True, sigma_fn=_sigma_fn)
    assert info_on["status"] == "completed"
    assert bad in info_on["outlier_obs"]
    assert info_on["n_outliers_mid"] >= 1 and info_on["n_outliers_end"] >= 1
    assert info_on["n_outliers_end"] == len(info_on["outlier_obs"])
    err_on, err_off = _pose_err(frames_on, truth), _pose_err(frames_off, truth)
    assert err_on < err_off, (err_on, err_off)
    # bundle_adjust never edits a link table
    assert frames_on[2].keypoint_to_point[bad[1]] == 500


# --------------------------------------------------------------------------- #
# D4 sigma enters the test
# --------------------------------------------------------------------------- #

def test_d4_sigma_scales_the_test():
    frames, wm, _, bad = _d3_world(sigma_at_outlier=16.0)
    assert (30.0 / 16.0) ** 2 < opt.BA_OUTLIER_CHI2
    info, _ = _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn)
    assert bad not in info["outlier_obs"]
    # and the same row at 1 px is one (control for the probe)
    frames1, wm1, _, bad1 = _d3_world(sigma_at_outlier=1.0)
    info1, _ = _run(frames1, wm1, reject_outliers=True, sigma_fn=_sigma_fn)
    assert bad1 in info1["outlier_obs"]


def test_d4_classify_threshold_and_sigma():
    # one camera at the origin, points straight ahead; residuals set by the observation
    cam = np.eye(4)[None]
    pts = np.array([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0], [0.0, 0.1, 4.0]])
    proj = [_project(cam[0], p) for p in pts]
    e = np.sqrt(opt.BA_OUTLIER_CHI2)
    obs = [(0, 0, proj[0][0] + 0.99 * e, proj[0][1]),        # just inside at sigma 1
           (0, 1, proj[1][0] + 1.01 * e, proj[1][1]),        # just outside at sigma 1
           (0, 2, proj[2][0] + 3.0 * 1.01 * e, proj[2][1])]  # outside at sigma 1, inside at sigma 4
    prob = BAProblem.from_arrays(K, cam, [True], pts, obs)
    assert opt.classify_ba_outliers(prob, [1.0, 1.0, 1.0]).tolist() == [False, True, True]
    assert opt.classify_ba_outliers(prob, [1.0, 1.0, 4.0]).tolist() == [False, True, False]


# --------------------------------------------------------------------------- #
# D5 a row whose point is behind its camera is an outlier
# --------------------------------------------------------------------------- #

def test_d5_behind_row_is_an_outlier_classify():
    cam = np.eye(4)[None]
    pts = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, -2.0]])
    u0, v0 = _project(cam[0], pts[0])
    # the behind row is given a perfectly "fitting" pixel so only the behind clause can mark it
    obs = [(0, 0, u0, v0), (0, 1, 320.0, 240.0)]
    prob = BAProblem.from_arrays(K, cam, [True], pts, obs)
    assert prob.behind_mask().tolist() == [False, True]
    assert opt.classify_ba_outliers(prob, [1e9, 1e9]).tolist() == [False, True]


def test_d5_behind_row_is_an_outlier_bundle_adjust():
    # window cameras near the origin looking +z; the fixed outer observer sits at z = 3.0, so
    # points at z >= 3.5 are in front of it and the extra point at z = 2.0 is behind it.
    rng = np.random.default_rng(5)
    cams = []
    for c, cx in enumerate((0.0, 0.25, 0.5, 0.75)):
        T = np.eye(4)
        T[:3, :3] = so3_exp(rng.normal(0, 0.02, 3)) if c else np.eye(3)
        T[:3, 3] = -T[:3, :3] @ np.array([cx, 0.05 * c, 0.0])
        cams.append(T)
    T4 = np.eye(4); T4[:3, 3] = -np.array([0.3, 0.0, 3.0])
    cams.append(T4)
    pts = rng.uniform([-1.0, -0.7, 3.5], [1.0, 0.7, 6.0], size=(12, 3))
    pts = np.vstack([pts, [[0.3, 0.1, 2.0]]])
    obs_lists = [list(range(13))] * 4 + [list(range(12))]
    frames, wm = _build_world(cams, pts, obs_lists)
    # the outer observer's fabricated observation of the point behind it
    kp = frames[4].observe(512, 320.0, 240.0)
    wm.get_point(512).frames.append(frames[4].id)
    wm.get_point(512).observations[frames[4].id] = kp
    prob, meta = build_ba_problem(frames[:4], wm, K, all_frames=frames)
    assert int(np.count_nonzero(prob.behind_mask())) == 1
    info, _ = _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn)
    assert (frames[4].id, kp, 512) in info["outlier_obs"]
    assert info["n_outliers_mid"] >= 1


# --------------------------------------------------------------------------- #
# D6 budget and output
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_iter", [1, 2, 7, 10])
def test_d6_budget_and_output(n_iter):
    frames, wm, _, _ = _d3_world()
    info, out = _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn, max_iterations=n_iter)
    assert info["iterations"] <= n_iter
    assert out.count("BA completed") == 1
    assert "BA completed (lm): 4 frames, 12 points, 60 observations, cost" in out
    assert "_obs_keys" not in info and "_cams" not in info
    assert info["n_obs"] == 60 and info["n_points"] == 12 and info["n_obs_fixed"] == 24


def test_d6_split_of_the_budget(monkeypatch):
    calls = []
    real = opt.solve_ba

    def spy(problem, **kw):
        calls.append((problem.n_obs, kw["max_iterations"]))
        return real(problem, **kw)

    monkeypatch.setattr(opt, "solve_ba", spy)
    frames, wm, _, _ = _d3_world()
    _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn, max_iterations=10)
    assert [c[1] for c in calls] == [5, 5]
    assert calls[0][0] == 60 and calls[1][0] < 60          # phase 2 drops the mid outliers
    calls.clear()
    frames, wm, _, _ = _d3_world()
    _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn, max_iterations=1)
    assert [c[1] for c in calls] == [1]                     # n2 == 0: phase 2 skipped


def test_d6_sigma_fn_required():
    frames, wm, _, _ = _d3_world()
    with pytest.raises(ValueError):
        bundle_adjust(frames[:4], wm, K, all_frames=frames, reject_outliers=True, sigma_fn=None)
    for f in frames:
        assert f.set_calls == 0


def test_d6_signatures_kept():
    sig = inspect.signature(bundle_adjust).parameters
    for name, default in (("reject_outliers", False), ("sigma_fn", None)):
        assert sig[name].kind == inspect.Parameter.KEYWORD_ONLY and sig[name].default is default
    assert opt.BA_OUTLIER_CHI2 == 5.991
    src = inspect.getsource(opt)
    assert "import local_mapping" not in src and ".unlink(" not in src


# --------------------------------------------------------------------------- #
# D7 the two reprojection figures are over the final inliers
# --------------------------------------------------------------------------- #

def _direct_mean_error(frames, wm, poses, pts, excluded):
    errs = []
    for f in frames:
        for kp, pid in f.keypoint_to_point.items():
            if (f.id, kp, pid) in excluded:
                continue
            X = pts[pid]
            x = poses[f.id][:3, :3] @ X + poses[f.id][:3, 3]
            if x[2] <= opt.BEHIND_CAMERA_Z:
                continue
            u, v = _project(poses[f.id], X)
            errs.append(np.hypot(u - f.KeyPoints[kp].pt[0], v - f.KeyPoints[kp].pt[1]))
    return float(np.mean(errs))


def test_d7_reprojection_figures_over_final_inliers():
    frames, wm, _, bad = _d3_world()
    poses0 = {f.id: f.cam_T_world.copy() for f in frames}
    pts0 = {p.id: p.point.copy() for p in wm.points}
    info, _ = _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn)
    excluded = set(info["outlier_obs"])
    assert bad in excluded
    poses1 = {f.id: f.cam_T_world.copy() for f in frames}
    pts1 = {p.id: p.point.copy() for p in wm.points}
    before = _direct_mean_error(frames, wm, poses0, pts0, excluded)
    after = _direct_mean_error(frames, wm, poses1, pts1, excluded)
    assert info["reproj_before_px"] == pytest.approx(before, rel=1e-9, abs=1e-12)
    assert info["reproj_after_px"] == pytest.approx(after, rel=1e-9, abs=1e-12)
    # over all rows the before figure would include the 30 px row and be larger
    assert _direct_mean_error(frames, wm, poses0, pts0, set()) > before


# --------------------------------------------------------------------------- #
# D8 a point all of whose rows were mid outliers keeps its phase-1 coordinates
# --------------------------------------------------------------------------- #

def test_d8_point_without_rows_keeps_phase1_coordinates():
    frames, wm, _, _ = _d3_world()
    # point 511: every one of its five observations displaced by 40 px, in five directions
    for c, f in enumerate(frames):
        kp = _kp_of(f, 511)
        a = 2.0 * np.pi * c / 5.0
        _displace(f, kp, 40.0 * np.cos(a), 40.0 * np.sin(a))
    frames_ref, wm_ref = copy.deepcopy((frames, wm))
    # the phase-1 state, independently: the same problem, solve_ba for n1 = 5 iterations
    with contextlib.redirect_stdout(io.StringIO()):
        prob, meta = build_ba_problem(frames_ref[:4], wm_ref, K, all_frames=frames_ref)
    p1, _ = solve_ba(prob, method="lm", max_iterations=5)
    sig = opt._row_sigmas(meta["_cams"], meta["_obs_keys"], _sigma_fn)
    out_mid = opt.classify_ba_outliers(p1, sig)
    j = meta["point_ids"].index(511)
    rows_511 = prob.pt_idx == j
    assert rows_511.sum() == 5 and np.all(out_mid[rows_511]), "precondition: all rows of 511 are mid outliers"
    assert np.count_nonzero(~out_mid) >= 10
    info, _ = _run(frames, wm, reject_outliers=True, sigma_fn=_sigma_fn, max_iterations=10)
    assert info["n_outliers_mid"] == int(np.count_nonzero(out_mid))
    assert wm.get_point(511).point.tobytes() == p1.points()[j].tobytes()
    # the other points did move on in phase 2
    k = meta["point_ids"].index(500)
    assert wm.get_point(500).point.tobytes() != p1.points()[k].tobytes()


# --------------------------------------------------------------------------- #
# D9 apply_ba_outlier_removal on a real Map and the [BA-OUTLIERS] line
# --------------------------------------------------------------------------- #

class _TableFrame:
    def __init__(self, fid):
        self.id = fid
        self.keypoint_to_point = {}


def test_d9_apply_removal_on_real_map():
    from Map import Map
    from Point import Point
    import pipeline as pl
    frames = [_TableFrame(10), _TableFrame(11), _TableFrame(12)]
    world = Map(frames=frames)
    pts = [Point(point=np.array([0.0, 0.0, 3.0 + i])) for i in range(3)]
    for p in pts:
        world.add_point(p)
    for f in frames:
        for i, p in enumerate(pts):
            assert world.link(f, i, p)
    zero = {k: 0 for k in world.check_invariants(frames)}
    assert world.check_invariants(frames) == zero
    # a stale entry: keypoint 2 of frame 12 is relinked to another point before the removal
    world.unlink(frames[2], 2)
    p_new = Point(point=np.array([1.0, 0.0, 4.0]))
    world.add_point(p_new)
    assert world.link(frames[2], 2, p_new)
    outliers = [(10, 0, pts[0].id), (11, 1, pts[1].id), (12, 2, pts[2].id), (99, 0, pts[0].id)]
    removed = pl.apply_ba_outlier_removal(world, frames, outliers)
    assert removed == 2
    assert 0 not in frames[0].keypoint_to_point and 10 not in pts[0].observations
    assert 1 not in frames[1].keypoint_to_point and 11 not in pts[1].observations
    assert 10 not in pts[0].frames and 11 not in pts[1].frames
    assert frames[2].keypoint_to_point[2] == p_new.id            # skipped: no longer that point
    assert p_new.observations[12] == 2
    assert world.get_point(pts[0].id) is pts[0]                   # no point deleted
    assert world.check_invariants(frames) == zero
    # removing again is a no-op
    assert pl.apply_ba_outlier_removal(world, frames, outliers) == 0
    assert world.check_invariants(frames) == zero


def test_d9_line_format():
    import pipeline as pl
    assert pl.format_ba_outliers_line(120, 900, 12, 9, 1, 9, 0) == \
        "  >> [BA-OUTLIERS] frame=120 obs=900 mid=12 end=9 accepted=1 removed=9 inv_total=0"
    assert pl.format_ba_outliers_line(140, 50, 3, 2, False, 0, 0) == \
        "  >> [BA-OUTLIERS] frame=140 obs=50 mid=3 end=2 accepted=0 removed=0 inv_total=0"
    assert pl.format_ba_outliers_line(7, 1, 0, 0, True, 0, 3).endswith("accepted=1 removed=0 inv_total=3")
