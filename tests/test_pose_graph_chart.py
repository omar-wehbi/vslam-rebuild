"""Stage 01a: pose graph on the right-perturbation SE(3) chart (spec C3).

se3 exp/log round trip, FD check of the inverse right Jacobian, PoseGraphProblem.jacobian
vs central finite differences through retract on a drifted chain with a nonzero loop
residual, solve_pose_graph recovery of a perturbed exact chain, and the
correct_trajectory_pose_graph write-back path with stand-in frames and map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loop_closure as lc  # noqa: E402
from loop_closure import (  # noqa: E402
    PoseGraphProblem,
    build_pose_graph_problem,
    correct_trajectory_pose_graph,
    solve_pose_graph,
)
from optimization import (  # noqa: E402
    se3_adjoint,
    se3_exp,
    se3_log,
    se3_right_jacobian,
    se3_right_jacobian_inv,
    so3_log,
)


class _Stand:
    """Minimal frame stand-in: world_T_cam, cam_T_world, id, set_world_pose."""

    def __init__(self, fid, world_T_cam):
        self.id = fid
        self.world_T_cam = np.array(world_T_cam, dtype=float)
        self.set_calls = 0

    @property
    def cam_T_world(self):
        return np.linalg.inv(self.world_T_cam)

    def set_world_pose(self, T):
        self.world_T_cam = np.array(T, dtype=float)
        self.set_calls += 1
        return True


def _chain(rng, n):
    T = np.eye(4); poses = [T.copy()]
    for _ in range(n):
        T = T @ se3_exp(np.concatenate([rng.normal(0, 0.25, 3), rng.normal(0, 0.5, 3)]))
        poses.append(T.copy())
    return poses


def _pose_diff(Ta, Tb):
    d = np.linalg.inv(Ta) @ Tb
    return float(np.linalg.norm(d[:3, 3]) + np.linalg.norm(so3_log(d[:3, :3])))


def test_se3_exp_log_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        xi = np.concatenate([rng.normal(0, 0.8, 3), rng.normal(0, 1.5, 3)])
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)
    assert np.allclose(se3_log(np.eye(4)), np.zeros(6))


def test_se3_right_jacobian_matches_exponential_map_by_fd():
    """exp(xi)^-1 exp(xi + d) ~= exp(Jr(xi) d), and Jr^-1 is its inverse."""
    rng = np.random.default_rng(1)
    eps = 1e-6
    for _ in range(6):
        xi = np.concatenate([rng.normal(0, 0.6, 3), rng.normal(0, 1.0, 3)])
        Jr = se3_right_jacobian(xi)
        Jr_inv = se3_right_jacobian_inv(xi)
        assert np.allclose(Jr @ Jr_inv, np.eye(6), atol=1e-9)
        for k in range(6):
            d = np.zeros(6); d[k] = eps
            lhs = se3_log(np.linalg.inv(se3_exp(xi)) @ se3_exp(xi + d)) / eps
            assert np.allclose(lhs, Jr[:, k], atol=1e-5), (k, lhs, Jr[:, k])
    # small-angle branch
    xi = np.array([1e-5, -2e-5, 1e-5, 0.3, -0.1, 0.2])
    assert np.allclose(se3_right_jacobian(xi) @ se3_right_jacobian_inv(xi), np.eye(6), atol=1e-9)


def _jr_inv_by_fd(xi, eps=1e-6):
    """Central FD of d/dd log(exp(xi) exp(d)) at d = 0, which is Jr^-1(xi) by definition."""
    T = se3_exp(xi)
    J = np.empty((6, 6))
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        J[:, k] = (se3_log(T @ se3_exp(d)) - se3_log(T @ se3_exp(-d))) / (2 * eps)
    return J


def test_se3_inverse_right_jacobian_small_angle_branch_matches_fd_tightly():
    """
    Residual rotation ~5e-3 rad (inside the series branch of _se3_Q, theta < 1e-2) with a
    translation of order 1.  The bar of 1e-7 relative error is an order of magnitude below
    what the wrong sign of the theta^4 coefficient produces here (about 2e-6), and two
    orders above the FD noise of the correct closed form (about 1e-10).
    """
    rng = np.random.default_rng(21)
    for _ in range(5):
        w = rng.normal(size=3)
        w *= 5e-3 / np.linalg.norm(w)
        xi = np.concatenate([w, rng.normal(0, 1.0, 3)])
        assert 1e-3 < np.linalg.norm(xi[:3]) < 1e-2 and np.linalg.norm(xi[3:]) > 0.3
        Ja = se3_right_jacobian_inv(xi)
        Jfd = _jr_inv_by_fd(xi)
        err = np.linalg.norm(Ja - Jfd) / np.linalg.norm(Jfd)
        assert err < 1e-7, err


def test_se3_Q_is_continuous_across_the_series_branch_boundary():
    """_se3_Q switches from series to closed form at theta = 1e-2; both sides must agree."""
    rho = np.array([0.7, -1.1, 0.4])
    axis = np.array([0.3, -0.5, 0.8]); axis /= np.linalg.norm(axis)
    from optimization import _se3_Q
    Q_minus = _se3_Q(rho, (1e-2 - 1e-9) * axis)
    Q_plus = _se3_Q(rho, (1e-2 + 1e-9) * axis)
    assert np.abs(Q_minus - Q_plus).max() < 1e-9


def test_pose_graph_jacobian_small_residual_rotation_matches_fd_tightly():
    """
    PoseGraphProblem.jacobian at residuals with rotation ~5e-3 rad and translation of order
    1 (exact chain retracted by small rotations and unit-scale translations), against
    central FD through retract, at 1e-7 relative error (old sign: about 8e-7).
    """
    rng = np.random.default_rng(11)
    n = 5
    truth = _chain(rng, n)
    frames = [_Stand(i, T) for i, T in enumerate(truth)]
    exact = build_pose_graph_problem(frames, 0, n, np.linalg.inv(truth[n]))
    delta = np.concatenate([np.concatenate([rng.normal(0, 2e-3, 3), rng.normal(0, 0.6, 3)])
                            for _ in range(n)])
    prob = exact.retract(delta)
    r = prob.residuals()
    rot_norms = [np.linalg.norm(r[6 * i:6 * i + 3]) for i in range(n + 1)]
    tr_norms = [np.linalg.norm(r[6 * i + 3:6 * i + 6]) for i in range(n + 1)]
    assert max(rot_norms) < 1e-2 and max(rot_norms) > 2e-3, rot_norms   # series branch
    assert max(tr_norms) > 0.5, tr_norms                                  # translation O(1)
    J, Jfd = _fd_jacobian(prob)
    err = np.linalg.norm(J - Jfd) / np.linalg.norm(Jfd)
    assert err < 1e-7, err


def test_se3_adjoint_identity():
    rng = np.random.default_rng(2)
    T = se3_exp(np.concatenate([rng.normal(0, 0.7, 3), rng.normal(0, 1.0, 3)]))
    xi = rng.normal(0, 0.3, 6)
    lhs = T @ se3_exp(xi) @ np.linalg.inv(T)
    assert np.allclose(lhs, se3_exp(se3_adjoint(T) @ xi), atol=1e-9)


def _fd_jacobian(problem, eps=1e-6):
    J = problem.jacobian()
    J = np.asarray(J.toarray() if hasattr(J, "toarray") else J, dtype=float)
    Jfd = np.empty_like(J)
    for j in range(problem.x_dim):
        e = np.zeros(problem.x_dim); e[j] = eps
        Jfd[:, j] = (problem.retract(e).residuals() - problem.retract(-e).residuals()) / (2 * eps)
    return J, Jfd


def test_jacobian_matches_central_finite_differences_on_drifted_chain():
    rng = np.random.default_rng(3)
    n = 5
    truth = _chain(rng, n)
    drifted = [truth[0]] + [truth[i] @ se3_exp(np.concatenate([rng.normal(0, 0.05, 3), rng.normal(0, 0.1, 3)]))
                            for i in range(1, n + 1)]
    frames = [_Stand(i, T) for i, T in enumerate(drifted)]
    prob = build_pose_graph_problem(frames, 0, n, np.linalg.inv(truth[n]))
    assert isinstance(prob, PoseGraphProblem)
    assert prob.n_nodes == n and prob.dof == 6 and prob.x_dim == 6 * n
    assert np.isclose(prob.w_loop, np.sqrt(n))
    r = prob.residuals()
    assert r.shape == (6 * (n + 1),)
    assert np.linalg.norm(r[-6:]) > 1e-2, "loop residual must be nonzero at the base"
    J, Jfd = _fd_jacobian(prob)
    def rel(cols):
        return float(np.linalg.norm(Jfd[:, cols] - J[:, cols]) / max(np.linalg.norm(Jfd[:, cols]), 1e-12))
    rot = [6 * j + k for j in range(n) for k in range(3)]
    tr = [6 * j + 3 + k for j in range(n) for k in range(3)]
    assert rel(list(range(6 * n))) < 1e-5
    assert rel(rot) < 1e-5 and rel(tr) < 1e-5
    # structure: block bidiagonal + loop block; node 0 fixed -> row block 0 has one column block
    assert np.all(J[0:6, 6:] == 0.0)
    assert np.all(J[-6:, :6 * (n - 1)] == 0.0)


def test_jacobian_sign_flip_is_detected():
    """Mutation guard: negating the d r_i / d delta_i block breaks the FD agreement."""
    rng = np.random.default_rng(4)
    n = 4
    truth = _chain(rng, n)
    frames = [_Stand(i, truth[i] @ se3_exp(rng.normal(0, 0.05, 6))) for i in range(n + 1)]
    prob = build_pose_graph_problem(frames, 0, n, np.linalg.inv(truth[n]))
    J, Jfd = _fd_jacobian(prob)
    Jbad = J.copy()
    Jbad[6:12, 0:6] *= -1.0
    assert np.linalg.norm(Jbad - Jfd) / np.linalg.norm(Jfd) > 0.05


def test_build_copies_and_uses_only_world_T_cam_and_id():
    rng = np.random.default_rng(5)
    truth = _chain(rng, 3)
    frames = [_Stand(10 + i, T) for i, T in enumerate(truth)]
    prob = build_pose_graph_problem(frames, 0, 3, np.linalg.inv(truth[3]), w_loop=2.5)
    assert prob.w_loop == 2.5 and prob.node_ids == [10, 11, 12, 13]
    frames[1].world_T_cam[0, 3] += 5.0  # mutate after build: problem unaffected
    assert np.allclose(prob.poses()[0], truth[1])
    with pytest.raises(ValueError):
        build_pose_graph_problem(frames, 2, 2, np.eye(4))


def test_retract_leaves_original_untouched():
    rng = np.random.default_rng(6)
    truth = _chain(rng, 3)
    frames = [_Stand(i, T) for i, T in enumerate(truth)]
    prob = build_pose_graph_problem(frames, 0, 3, np.linalg.inv(truth[3]))
    r0 = prob.residuals().copy()
    delta = rng.normal(0, 0.1, prob.x_dim)
    moved = prob.retract(delta)
    assert np.array_equal(prob.residuals(), r0)
    for j in range(3):
        assert np.allclose(moved.poses()[j], truth[j + 1] @ se3_exp(delta[6 * j:6 * j + 6]))
    assert np.allclose(moved.T[0], truth[0])


def test_solve_recovers_perturbed_exact_chain():
    rng = np.random.default_rng(7)
    n = 6
    truth = _chain(rng, n)
    frames = [_Stand(i, T) for i, T in enumerate(truth)]
    exact = build_pose_graph_problem(frames, 0, n, np.linalg.inv(truth[n]))
    assert np.linalg.norm(exact.residuals()) < 1e-12
    delta = np.concatenate([np.concatenate([rng.normal(0, 0.03, 3), rng.normal(0, 0.05, 3)]) for _ in range(n)])
    solved, info = solve_pose_graph(exact.retract(delta), max_iterations=50)
    assert info["status"] == "completed" and info["converged"]
    assert info["cost_final"] < 1e-18
    err = max(_pose_diff(solved.poses()[j], truth[j + 1]) for j in range(n))
    assert err < 1e-6, err


def test_solve_drifted_chain_reduces_loop_error():
    rng = np.random.default_rng(8)
    n = 8
    truth = _chain(rng, n)
    drift = se3_exp(np.array([0.02, -0.01, 0.03, 0.2, -0.1, 0.15]))
    # accumulate drift along the chain, loop constraint is the true end pose
    drifted = [truth[0]]
    D = np.eye(4)
    for i in range(1, n + 1):
        D = D @ drift
        drifted.append(truth[i] @ D)
    frames = [_Stand(i, T) for i, T in enumerate(drifted)]
    prob = build_pose_graph_problem(frames, 0, n, np.linalg.inv(truth[n]))
    solved, info = solve_pose_graph(prob)
    assert info["cost_final"] < info["cost_initial"]
    before = _pose_diff(drifted[n], truth[n])
    after = _pose_diff(solved.poses()[n - 1], truth[n])
    assert after < 0.2 * before


class _Pt:
    def __init__(self, pid, xyz, frames):
        self.id = pid
        self.point = np.array(xyz, dtype=float)
        self.x, self.y, self.z = map(float, self.point)
        self.frames = list(frames)


class _Map:
    def __init__(self, points):
        self.points = list(points)


def test_correct_trajectory_pose_graph_writes_back_and_corrects_points():
    """
    Drifted chain with the true end pose as loop measurement.  Odometry is taken from the
    drifted poses at build time, so the optimum balances odometry against the loop factor:
    the end-pose error must shrink, every free node is written back exactly once through
    set_world_pose, the fixed node is untouched, and the map point follows the correction of
    the frame at its mean observation fraction (delta = T_new @ inv(T_old)).
    """
    rng = np.random.default_rng(9)
    n = 5
    truth = _chain(rng, n)
    drift = se3_exp(np.array([0.02, -0.01, 0.015, 0.05, -0.03, 0.04]))
    drifted, D = [truth[0]], np.eye(4)
    for i in range(1, n + 1):
        D = D @ drift
        drifted.append(truth[i] @ D)
    frames = [_Stand(i, T) for i, T in enumerate(drifted)]
    old = [f.world_T_cam.copy() for f in frames]
    p_in = _Pt(1, [0.5, -0.2, 3.0], frames=[3])         # observed inside the span
    p_out = _Pt(2, [1.0, 1.0, 1.0], frames=[0])          # only at the fixed keyframe
    p_none = _Pt(3, [2.0, 2.0, 2.0], frames=[999])       # unknown frame id
    wm = _Map([p_in, p_out, p_none])
    before = _pose_diff(drifted[n], truth[n])
    n_corr = correct_trajectory_pose_graph(frames, 0, n, np.linalg.inv(truth[n]), world_map=wm)
    assert n_corr == 1
    assert frames[0].set_calls == 0 and np.array_equal(frames[0].world_T_cam, old[0])
    for i in range(1, n + 1):
        assert frames[i].set_calls == 1
        assert _pose_diff(frames[i].world_T_cam, old[i]) > 1e-6   # every free node moved
    after = _pose_diff(frames[n].world_T_cam, truth[n])
    assert after < 0.5 * before, (before, after)
    # the point followed frame 3's correction delta = T_new @ inv(T_old)
    d3 = frames[3].world_T_cam @ np.linalg.inv(old[3])
    assert np.linalg.norm(d3 - np.eye(4)) > 1e-6
    assert np.allclose(p_in.point, d3[:3, :3] @ np.array([0.5, -0.2, 3.0]) + d3[:3, 3])
    assert np.allclose([p_in.x, p_in.y, p_in.z], p_in.point)
    assert np.array_equal(p_out.point, [1.0, 1.0, 1.0])
    assert np.array_equal(p_none.point, [2.0, 2.0, 2.0])
    assert correct_trajectory_pose_graph(frames, 3, 3, np.eye(4), world_map=wm) == 0


def test_correct_trajectory_pose_graph_exact_loop_is_a_fixed_point():
    """When the loop measurement already agrees with the chain, nothing moves."""
    rng = np.random.default_rng(10)
    n = 4
    truth = _chain(rng, n)
    frames = [_Stand(i, T) for i, T in enumerate(truth)]
    p = _Pt(1, [0.3, 0.1, 2.0], frames=[2])
    assert correct_trajectory_pose_graph(frames, 0, n, np.linalg.inv(truth[n]), world_map=_Map([p])) == 1
    for i in range(n + 1):
        assert _pose_diff(frames[i].world_T_cam, truth[i]) < 1e-9
    assert np.allclose(p.point, [0.3, 0.1, 2.0], atol=1e-9)


def test_flat_pose_graph_state_is_gone():
    for name in ("_world_pose_to_params", "_params_to_world_pose", "_se3_log"):
        assert not hasattr(lc, name)
    assert not hasattr(lc, "least_squares")
