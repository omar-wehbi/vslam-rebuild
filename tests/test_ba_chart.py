"""Stage 01a: bundle adjustment on the left-perturbation chart (spec C2).

Covers the Lie helpers (exp/log round trips, FD checks of the analytical SO(3)/SE(3)
Jacobians), BAProblem (Jacobian vs central finite differences through retract, residual
rows for fixed observers, immutability of retract), solve_ba recovery from a known
perturbation, every skip reason of build_ba_problem, and bundle_adjust write-back.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimization as opt  # noqa: E402
from optimization import (  # noqa: E402
    BAProblem,
    build_ba_problem,
    bundle_adjust,
    finite_difference_jacobian,
    se3_exp,
    se3_left_jacobian,
    se3_log,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inv,
    so3_log,
    solve_ba,
)


# --------------------------------------------------------------------------- #
# Lie helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("omega", [
    [0.0, 0.0, 0.0], [1e-9, -2e-9, 3e-9], [0.3, -0.2, 0.5], [1.5, 0.7, -1.1],
    [0.0, 0.0, 3.0], [2.2, 2.0, 0.3],  # last two: angle > pi/2, one close to pi
])
def test_so3_exp_log_round_trip(omega):
    omega = np.array(omega, dtype=float)
    R = so3_exp(omega)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    back = so3_log(R)
    assert np.allclose(back, omega, atol=1e-9), (back, omega)
    assert np.allclose(so3_exp(back), R, atol=1e-12)


def test_so3_log_near_pi():
    axis = np.array([1.0, 2.0, -0.5]); axis /= np.linalg.norm(axis)
    for theta in (math.pi - 1e-5, math.pi - 1e-7):
        R = so3_exp(theta * axis)
        w = so3_log(R)
        assert np.allclose(so3_exp(w), R, atol=1e-8)


@pytest.mark.parametrize("xi", [
    [0.0] * 6, [0.3, -0.2, 0.5, 1.0, -2.0, 0.5], [1.4, 0.2, -0.9, 0.1, 0.3, -0.7],
    [1e-7, 2e-7, -1e-7, 0.4, 0.2, 0.1],
])
def test_se3_exp_log_round_trip(xi):
    xi = np.array(xi, dtype=float)
    T = se3_exp(xi)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(se3_log(T), xi, atol=1e-9)


def test_so3_left_jacobian_matches_finite_differences():
    """exp(omega + d) ~= exp(J_l(omega) d) exp(omega): FD check of the analytical J_l."""
    rng = np.random.default_rng(1)
    eps = 1e-6
    for _ in range(5):
        omega = rng.normal(0, 0.8, 3)
        Jl = so3_left_jacobian(omega)
        assert np.allclose(Jl @ so3_left_jacobian_inv(omega), np.eye(3), atol=1e-10)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            lhs = so3_log(so3_exp(omega + d) @ so3_exp(omega).T) / eps
            assert np.allclose(lhs, Jl[:, k], atol=1e-5), (lhs, Jl[:, k])


def test_se3_left_jacobian_matches_finite_differences():
    """se3_exp(xi + d) ~= se3_exp(J_l(xi) d) se3_exp(xi): FD check of the SE(3) left Jacobian."""
    rng = np.random.default_rng(2)
    eps = 1e-6
    for _ in range(5):
        xi = np.concatenate([rng.normal(0, 0.7, 3), rng.normal(0, 1.0, 3)])
        Jl = se3_left_jacobian(xi)
        for k in range(6):
            d = np.zeros(6); d[k] = eps
            lhs = se3_log(se3_exp(xi + d) @ np.linalg.inv(se3_exp(xi))) / eps
            assert np.allclose(lhs, Jl[:, k], atol=1e-5), (k, lhs, Jl[:, k])


# --------------------------------------------------------------------------- #
# synthetic scene: 4 cameras, 8 points; camera 0 fixed, camera 3 a fixed outer observer
# --------------------------------------------------------------------------- #

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _make_scene(rng, n_cams=4, n_pts=8, noise_px=0.0):
    pts = rng.uniform([-1.5, -1.0, 3.0], [1.5, 1.0, 7.0], size=(n_pts, 3))
    poses = []
    for c in range(n_cams):
        w = rng.normal(0, 0.3, 3) if c else np.zeros(3)
        t = rng.normal(0, 0.4, 3) if c else np.zeros(3)
        T = np.eye(4); T[:3, :3] = so3_exp(w); T[:3, 3] = t
        poses.append(T)
    poses = np.array(poses)
    obs = []
    for c in range(n_cams):
        R, t = poses[c, :3, :3], poses[c, :3, 3]
        for p in range(n_pts):
            x = R @ pts[p] + t
            assert x[2] > 0.5
            uv = K @ x; uv = uv[:2] / uv[2]
            obs.append((c, p, uv[0] + rng.normal(0, noise_px), uv[1] + rng.normal(0, noise_px)))
    return poses, pts, obs


FIXED = np.array([True, False, False, True])  # camera 0 fixed, camera 3 = fixed outer observer


def _fd_rel_err_per_block(prob, eps=1e-6):
    J = prob.jacobian()
    Jfd = finite_difference_jacobian(prob, eps=eps)
    n_free = prob.n_free
    cols = {
        "rot": [6 * c + k for c in range(n_free) for k in range(3)],
        "t": [6 * c + 3 + k for c in range(n_free) for k in range(3)],
        "pt": list(range(6 * n_free, prob.x_dim)),
    }
    return {name: float(np.linalg.norm(Jfd[:, c] - J[:, c]) / max(np.linalg.norm(Jfd[:, c]), 1e-12))
            for name, c in cols.items()}


def test_jacobian_matches_central_finite_differences_per_block():
    rng = np.random.default_rng(3)
    poses, pts, obs = _make_scene(rng, noise_px=1.0)
    prob = BAProblem.from_arrays(K, poses, FIXED, pts, obs)
    assert prob.n_cams == 4 and prob.n_free == 2 and prob.n_fixed == 2
    assert prob.free_cam_indices == [1, 2]
    assert prob.x_dim == 6 * 2 + 3 * 8
    errs = _fd_rel_err_per_block(prob)
    for name, e in errs.items():
        assert e < 1e-5, (name, e, errs)


def test_jacobian_is_not_the_right_perturbation_formula():
    """Mutation guard: the right-perturbation block -P R [p]x differs from ours at nonzero R."""
    rng = np.random.default_rng(4)
    poses, pts, obs = _make_scene(rng)
    prob = BAProblem.from_arrays(K, poses, FIXED, pts, obs)
    J = prob.jacobian()
    Jfd = finite_difference_jacobian(prob)
    # build the wrong (right-side) rotation block for the first free camera's first observation
    k = next(i for i, (c, p, _, _) in enumerate(obs) if c == 1)
    c, p = obs[k][0], obs[k][1]
    R, t = poses[c, :3, :3], poses[c, :3, 3]
    x = R @ pts[p] + t
    P = np.array([[K[0, 0] / x[2], 0, -K[0, 0] * x[0] / x[2] ** 2], [0, K[1, 1] / x[2], -K[1, 1] * x[1] / x[2] ** 2]])
    wrong = -P @ R @ opt.skew(pts[p])
    assert np.allclose(J[2 * k:2 * k + 2, 0:3], Jfd[2 * k:2 * k + 2, 0:3], atol=1e-5)
    assert not np.allclose(wrong, Jfd[2 * k:2 * k + 2, 0:3], atol=1e-3)


def test_residual_rows_cover_fixed_observer_observations():
    rng = np.random.default_rng(5)
    poses, pts, obs = _make_scene(rng, noise_px=0.5)
    prob = BAProblem.from_arrays(K, poses, FIXED, pts, obs)
    r = prob.residuals()
    assert r.shape == (2 * len(obs),)
    assert prob.n_obs == len(obs) == 32
    J = prob.jacobian()
    assert J.shape == (2 * len(obs), prob.x_dim)
    n_fixed_obs = 0
    for k, (c, p, _, _) in enumerate(obs):
        if FIXED[c]:
            n_fixed_obs += 1
            # no camera columns for a fixed camera, but a live point block
            assert np.all(J[2 * k:2 * k + 2, :6 * prob.n_free] == 0.0)
            assert np.linalg.norm(J[2 * k:2 * k + 2, 6 * prob.n_free + 3 * p:6 * prob.n_free + 3 * p + 3]) > 1e-3
    assert n_fixed_obs == 16
    # residuals at the noisy observations are the noise (nonzero), also for fixed rows
    assert np.linalg.norm(r[:2]) > 1e-3


def test_retract_leaves_original_untouched_and_moves_state():
    rng = np.random.default_rng(6)
    poses, pts, obs = _make_scene(rng)
    prob = BAProblem.from_arrays(K, poses, FIXED, pts, obs)
    P0, Q0, r0 = prob.poses(), prob.points(), prob.residuals()
    delta = rng.normal(0, 0.05, prob.x_dim)
    moved = prob.retract(delta)
    assert np.array_equal(prob.poses(), P0)
    assert np.array_equal(prob.points(), Q0)
    assert np.array_equal(prob.residuals(), r0)
    P1 = moved.poses()
    # fixed cameras unchanged, free cameras moved on the left with additive t
    for c in range(4):
        if FIXED[c]:
            assert np.array_equal(P1[c], P0[c])
    for i, c in enumerate(prob.free_cam_indices):
        d = delta[6 * i:6 * i + 6]
        assert np.allclose(P1[c, :3, :3], so3_exp(d[:3]) @ P0[c, :3, :3])
        assert np.allclose(P1[c, :3, 3], P0[c, :3, 3] + d[3:])
    assert np.allclose(moved.points(), Q0 + delta[6 * prob.n_free:].reshape(-1, 3))
    # inputs to from_arrays are copied, not referenced
    poses[1, 0, 3] += 100.0
    assert prob.poses()[1, 0, 3] != poses[1, 0, 3]


def test_behind_camera_observation_gets_penalty_rows_and_zero_jacobian():
    rng = np.random.default_rng(7)
    poses, pts, obs = _make_scene(rng)
    pts2 = pts.copy(); pts2[0] = np.array([0.0, 0.0, -2.0])  # behind camera 0 (identity pose)
    prob = BAProblem.from_arrays(K, poses, FIXED, pts2, obs)
    k = next(i for i, (c, p, _, _) in enumerate(obs) if c == 0 and p == 0)
    r = prob.residuals()
    assert r[2 * k] == 100.0 and r[2 * k + 1] == 100.0
    assert np.all(prob.jacobian()[2 * k:2 * k + 2] == 0.0)
    assert prob.behind_mask().sum() >= 1
    _, info = solve_ba(prob, max_iterations=1)
    assert info["n_obs_behind"] >= 1


def test_huber_weights_and_cost():
    prob = BAProblem.from_arrays(K, np.eye(4)[None], [True], np.zeros((1, 3)) + [0, 0, 5.0], [(0, 0, 320.0, 240.0)], huber_delta_px=2.0)
    r = np.array([1.0, 0.0, 3.0, 4.0])  # e = 1 (inside), e = 5 (outside)
    w = prob.robust_weights(r)
    assert np.allclose(w, [1.0, 2.0 / 5.0])
    assert np.isclose(prob.cost(r), 0.5 * 1.0 + 2.0 * (5.0 - 1.0))


def test_solve_ba_recovers_from_5deg_0p3_perturbation_noise_free():
    rng = np.random.default_rng(8)
    poses, pts, obs = _make_scene(rng, n_pts=12, noise_px=0.0)
    fixed = FIXED.copy()
    P0 = poses.copy()
    for c in range(4):
        if fixed[c]:
            continue
        axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis)
        P0[c, :3, :3] = so3_exp(math.radians(5.0) * axis) @ P0[c, :3, :3]
        step = rng.normal(0, 1, 3); step = step / np.linalg.norm(step) * 0.3
        P0[c, :3, 3] = P0[c, :3, 3] + step
    Q0 = pts + rng.normal(0, 0.02, pts.shape)
    prob = BAProblem.from_arrays(K, P0, fixed, Q0, obs)
    before = prob.mean_reprojection_error()
    assert before > 5.0
    solved, info = solve_ba(prob, max_iterations=100)
    assert info["status"] == "completed"
    assert info["cost_final"] < info["cost_initial"]
    after = solved.mean_reprojection_error()
    assert after < 1e-3, (before, after, info)
    # gauge fully fixed (two fixed cameras): the free poses come back to the truth
    for c in (1, 2):
        assert np.linalg.norm(solved.poses()[c, :3, 3] - poses[c, :3, 3]) < 1e-4
    assert np.array_equal(prob.poses(), P0)  # the input problem was not mutated


@pytest.mark.parametrize("method", ["newton", "gradient_descent"])
def test_other_methods_decrease_cost_on_the_chart(method):
    rng = np.random.default_rng(9)
    poses, pts, obs = _make_scene(rng, n_pts=10)
    P0 = poses.copy()
    P0[1, :3, :3] = so3_exp([0.02, -0.01, 0.015]) @ P0[1, :3, :3]
    P0[2, :3, 3] += [0.03, -0.02, 0.01]
    prob = BAProblem.from_arrays(K, P0, FIXED, pts, obs)
    solved, info = solve_ba(prob, method=method, max_iterations=20)
    assert info["method"] == method
    assert info["cost_final"] < info["cost_initial"]
    if method == "newton":
        assert solved.mean_reprojection_error() < 1e-6


def test_unknown_method_raises():
    rng = np.random.default_rng(10)
    poses, pts, obs = _make_scene(rng)
    with pytest.raises(ValueError):
        solve_ba(BAProblem.from_arrays(K, poses, FIXED, pts, obs), method="adam")


def test_numerical_jacobian_path_matches_analytical_solve():
    rng = np.random.default_rng(11)
    poses, pts, obs = _make_scene(rng, n_pts=8)
    P0 = poses.copy(); P0[1, :3, 3] += [0.02, 0.0, -0.01]
    prob = BAProblem.from_arrays(K, P0, FIXED, pts, obs)
    a, _ = solve_ba(prob, max_iterations=15, use_analytical_jacobian=True)
    b, _ = solve_ba(prob, max_iterations=15, use_analytical_jacobian=False)
    assert np.allclose(a.poses(), b.poses(), atol=1e-5)


# --------------------------------------------------------------------------- #
# build_ba_problem / bundle_adjust with minimal Frame / Point / Map stand-ins
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
        self.pose_valid = pose_valid
        self.set_calls = 0

    def observe(self, pid, u, v):
        idx = len(self.KeyPoints)
        self.KeyPoints.append(_KP(u, v))
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


def _project(cam_T_world, X):
    x = cam_T_world[:3, :3] @ X + cam_T_world[:3, 3]
    uv = K @ x
    return uv[0] / uv[2], uv[1] / uv[2]


def _build_world(rng, cam_T_world_list, pts, observers, extra_obs_frames=()):
    """
    Frames observing points: observers[c] is the list of point indices camera c observes.
    extra_obs_frames pads Point.frames with ids of frames that do not exist (so that a point
    can pass the min_point_observations gate without adding residual rows).
    """
    frames = [_Frame(100 + c, T) for c, T in enumerate(cam_T_world_list)]
    points = [_Point(500 + p, pts[p]) for p in range(len(pts))]
    for c, f in enumerate(frames):
        for p in observers[c]:
            u, v = _project(f.cam_T_world, pts[p])
            idx = f.observe(points[p].id, u, v)
            points[p].frames.append(f.id)
            points[p].observations[f.id] = idx
    for p in points:
        for fid in extra_obs_frames:
            p.frames.append(fid)
    return frames, _Map(points)


def _scene(rng, n_cams, n_pts):
    poses, pts, _ = _make_scene(rng, n_cams=n_cams, n_pts=n_pts)
    return [poses[c] for c in range(n_cams)], pts


def test_skip_window_too_small(capsys):
    rng = np.random.default_rng(20)
    cams, pts = _scene(rng, 1, 8)
    frames, wm = _build_world(rng, cams, pts, [list(range(8))])
    prob, meta = build_ba_problem(frames[:1], wm, K)
    assert prob is None and meta["status"] == "skipped" and meta["reason"] == "window_too_small"
    assert capsys.readouterr().out.count("BA skipped: window_too_small") == 1


def test_skip_too_few_points(capsys):
    rng = np.random.default_rng(21)
    cams, pts = _scene(rng, 3, 4)
    frames, wm = _build_world(rng, cams, pts, [list(range(4))] * 3)
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is None and meta["reason"] == "too_few_points"
    assert capsys.readouterr().out.count("BA skipped: too_few_points") == 1


def test_skip_too_few_points_by_observation_gate():
    rng = np.random.default_rng(22)
    cams, pts = _scene(rng, 3, 8)
    frames, wm = _build_world(rng, cams, pts, [list(range(8))] * 3)
    prob, meta = build_ba_problem(frames, wm, K, min_point_observations=4)
    assert prob is None and meta["reason"] == "too_few_points"


def test_skip_too_few_observations():
    rng = np.random.default_rng(23)
    cams, pts = _scene(rng, 3, 6)
    # each point observed by exactly one window frame (6 rows), padded to pass the obs gate
    observers = [[0, 1], [2, 3], [4, 5]]
    frames, wm = _build_world(rng, cams, pts, observers, extra_obs_frames=(900, 901))
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is None and meta["reason"] == "too_few_observations"


def test_skip_gauge_unanchored_two_frame_window():
    rng = np.random.default_rng(24)
    cams, pts = _scene(rng, 2, 8)
    # points padded past the observation gate; no outer observers -> two fixed frames needed,
    # which leaves nothing free in a 2-frame window
    frames, wm = _build_world(rng, cams, pts, [list(range(8))] * 2, extra_obs_frames=(900,))
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is None and meta["reason"] == "gauge_unanchored"


def test_skip_gauge_unanchored_zero_baseline():
    rng = np.random.default_rng(25)
    cams, pts = _scene(rng, 3, 8)
    cams[1] = cams[0].copy()  # frames 0 and 1 coincide -> no baseline to anchor scale
    frames, wm = _build_world(rng, cams, pts, [list(range(8))] * 3)
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is None and meta["reason"] == "gauge_unanchored"


def test_skip_underdetermined():
    rng = np.random.default_rng(26)
    cams, pts = _scene(rng, 3, 12)
    # 12 points, each seen by one window frame: 24 rows vs 6 + 36 columns (two_fixed gauge)
    observers = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    frames, wm = _build_world(rng, cams, pts, observers, extra_obs_frames=(900, 901))
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is None and meta["reason"] == "underdetermined"


def test_build_two_fixed_gauge_without_outer_observers():
    rng = np.random.default_rng(27)
    cams, pts = _scene(rng, 4, 8)
    frames, wm = _build_world(rng, cams, pts, [list(range(8))] * 4)
    prob, meta = build_ba_problem(frames, wm, K)
    assert prob is not None and meta["gauge"] == "two_fixed"
    assert meta["fixed_frame_ids"] == [frames[0].id, frames[1].id]
    assert meta["free_frame_ids"] == [frames[2].id, frames[3].id]
    assert prob.n_free == 2 and prob.n_obs == 32
    # ties by point id ascending, deterministic
    assert meta["point_ids"] == sorted(meta["point_ids"])


def test_build_fixed_observers_gauge_and_rows():
    rng = np.random.default_rng(28)
    cams, pts = _scene(rng, 6, 12)
    observers = [list(range(12))] * 6
    frames, wm = _build_world(rng, cams, pts, observers)
    window = frames[:4]
    outer = frames[4:]
    outer[0].pose_valid = False  # an invalid-pose frame is never an observer
    prob, meta = build_ba_problem(window, wm, K, all_frames=frames)
    assert prob is not None and meta["gauge"] == "fixed_observers"
    assert meta["n_fixed_observers"] == 1
    assert meta["fixed_frame_ids"] == [window[0].id, outer[1].id]
    assert prob.n_cams == 5 and prob.n_fixed == 2 and prob.n_free == 3
    assert prob.n_obs == 5 * 12  # the observer's observations are residual rows
    assert prob.residuals().shape == (2 * 60,)
    # max_fixed_observers = 1 keeps only the most recent observer when several qualify
    outer[0].pose_valid = True
    prob2, meta2 = build_ba_problem(window, wm, K, all_frames=frames, max_fixed_observers=1)
    assert meta2["n_fixed_observers"] == 1 and meta2["fixed_frame_ids"][-1] == outer[1].id
    prob3, meta3 = build_ba_problem(window, wm, K, all_frames=frames)
    assert meta3["n_fixed_observers"] == 2 and prob3.n_fixed == 3
    # too few anchored points -> falls back to two_fixed even with an observer present
    prob4, meta4 = build_ba_problem(window, wm, K, all_frames=frames, min_anchor_points=13)
    assert meta4["gauge"] == "two_fixed" and prob4.n_fixed == 4


def test_bundle_adjust_writes_back_and_prints_one_completed_line(capsys):
    rng = np.random.default_rng(29)
    cams, pts = _scene(rng, 5, 12)
    frames, wm = _build_world(rng, cams, pts, [list(range(12))] * 5)
    truth = [f.cam_T_world.copy() for f in frames]
    # Window = frames 0..3, frame 4 is a fixed outer observer.  Frame 0 and the observer see
    # all 12 points, so scale is anchored ("fixed_observers") and frames 1, 2, 3 are free.
    # Perturb two of the free frames; frame 1 must be written back but stay at the truth.
    for c in (2, 3):
        T = frames[c].cam_T_world.copy()
        T[:3, :3] = so3_exp([0.01, -0.02, 0.015]) @ T[:3, :3]
        T[:3, 3] += [0.03, -0.01, 0.02]
        frames[c].set_camera_pose(T)
        frames[c].set_calls = 0
    info = bundle_adjust(frames[:4], wm, K, all_frames=frames, max_iterations=30)
    out = capsys.readouterr().out
    assert info["status"] == "completed"
    assert out.count("BA completed") == 1
    assert "BA completed (lm): 4 frames, 12 points, 60 observations, cost" in out
    assert info["gauge"] == "fixed_observers"
    assert info["reproj_after_px"] < info["reproj_before_px"]
    assert info["reproj_after_px"] < 1e-4
    assert info["n_obs_fixed"] == 24 and info["n_obs"] == 60
    assert info["fixed_frame_ids"] == [frames[0].id, frames[4].id]
    assert info["free_frame_ids"] == [frames[1].id, frames[2].id, frames[3].id]
    for c in (1, 2, 3):
        assert frames[c].set_calls == 1
        assert np.allclose(frames[c].cam_T_world, truth[c], atol=1e-5)
    for c in (0, 4):
        assert frames[c].set_calls == 0
        assert np.array_equal(frames[c].cam_T_world, truth[c])
    for p in wm.points:
        assert np.allclose(p.point, [p.x, p.y, p.z])


def test_bundle_adjust_skip_returns_info_and_writes_nothing(capsys):
    rng = np.random.default_rng(30)
    cams, pts = _scene(rng, 2, 8)
    frames, wm = _build_world(rng, cams, pts, [list(range(8))] * 2, extra_obs_frames=(900,))
    before = [f.cam_T_world.copy() for f in frames]
    info = bundle_adjust(frames, wm, K)
    out = capsys.readouterr().out
    assert info["status"] == "skipped" and info["reason"] == "gauge_unanchored"
    assert "BA completed" not in out and out.count("BA skipped") == 1
    for f, T in zip(frames, before):
        assert np.array_equal(f.cam_T_world, T) and f.set_calls == 0


def test_flat_additive_state_is_gone():
    for name in ("_build_residuals_jac", "_build_residuals", "_pose_to_params", "_params_to_pose"):
        assert not hasattr(opt, name)
