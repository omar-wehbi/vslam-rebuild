#!/usr/bin/env python3
"""
Pose-graph chart probe (task H, rewritten 2026-09-17 for stage 01a).

Grades the on-manifold pose-graph interface declared in plan/stages/01a-chart-and-backend.yaml
(spec B1/B2): loop_closure.build_pose_graph_problem -> PoseGraphProblem with residuals /
jacobian / retract / poses, and loop_closure.solve_pose_graph. The chart is the RIGHT
(body-frame) perturbation T <- T * Exp(delta), delta = (omega, nu), on the SE(3) sub-chart of the
ruled Sim(3) chart (theory/conventions.md#perturbation-model-for-pose-graphs; scale arrives in
stage 05 and this probe will then gain a seventh coordinate).

INPUTS   loop_closure.py from the checkout on PYTHONPATH; a synthetic chain built here from
         minimal frame stand-ins (world_T_cam, cam_T_world, id, set_world_pose), nonzero
         rotations, an exact loop constraint. Flags: --tol, --eps, --seed, --n-nodes, --self-test.
METRICS  fd_pg_rel_err_total   relative error, central FD through retract() vs jacobian(), all columns
         fd_pg_rel_err_rot     same, omega columns only
         fd_pg_rel_err_t       same, nu columns only
         pg_recovery_err_max   exact chain perturbed through retract(), solved with solve_pose_graph:
                               max over free nodes of (translation error + rotation angle) vs truth
         pg_coord_invariance_err  the same solve in coordinates moved by a global rigid transform,
                               mapped back: max pose difference
         fd_pg_interface_missing  1 if the declared interface is absent (all others nan)
FAILURE  exit 1 after printing every METRIC line. Missing interface (current main before 01a):
         nan metrics, interface_missing=1, exit 1, which is the intended pre-dispatch state.
KNOWN PASS  `--self-test`: the internal reference implementation passes the probe, and the
            reference's right Jacobian is itself verified against the exponential map by FD;
KNOWN FAIL  `--self-test` flips the sign of the reference's odometry block and asserts the
            probe reports it. Both must hold or the self-test exits 1.
Only the FD bars are decided here; recovery/invariance bars live in the contract.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

METRICS = ["fd_pg_rel_err_total", "fd_pg_rel_err_rot", "fd_pg_rel_err_t", "pg_recovery_err_max", "pg_coord_invariance_err"]


def _emit(vals: dict, interface_missing: int = 0):
    for k in METRICS:
        v = vals.get(k, float("nan"))
        print(f"METRIC {k}={v:.6e}" if isinstance(v, float) and not math.isnan(v) else f"METRIC {k}={v}")
    print(f"METRIC fd_pg_interface_missing={interface_missing}")


# --------------------------------------------------------------------------- #
# probe-owned SE(3) helpers, (omega, nu) ordering, independent of loop_closure.py
# --------------------------------------------------------------------------- #
def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=float)


def _so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + _hat(w)
    K = _hat(w / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _so3_log(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / 2 if th < 1e-12 else th / (2 * math.sin(th)) * v


def _so3_Jl(w):
    th = np.linalg.norm(w)
    W = _hat(w)
    if th < 1e-6:
        return np.eye(3) + 0.5 * W
    return np.eye(3) + (1 - math.cos(th)) / th ** 2 * W + (th - math.sin(th)) / th ** 3 * W @ W


def _se3_exp(xi):
    w, nu = xi[:3], xi[3:]
    T = np.eye(4); T[:3, :3] = _so3_exp(w); T[:3, 3] = _so3_Jl(w) @ nu
    return T


def _se3_log(T):
    w = _so3_log(T[:3, :3])
    return np.concatenate([w, np.linalg.solve(_so3_Jl(w), T[:3, 3])])


def _adjoint(T):
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6)); A[:3, :3] = R; A[3:, :3] = _hat(t) @ R; A[3:, 3:] = R
    return A


def _Q(nu, w):
    """Barfoot's Q(rho, phi) block with rho = nu (translation part), phi = w (rotation part)."""
    th = np.linalg.norm(w)
    Wx, Nx = _hat(w), _hat(nu)
    if th < 1e-6:
        return 0.5 * Nx
    s, c = math.sin(th), math.cos(th)
    a = (th - s) / th ** 3
    b = (1 - th ** 2 / 2 - c) / th ** 4
    d = 0.5 * (b - 3 * (th - s - th ** 3 / 6) / th ** 5)
    return (0.5 * Nx + a * (Wx @ Nx + Nx @ Wx + Wx @ Nx @ Wx)
            - b * (Wx @ Wx @ Nx + Nx @ Wx @ Wx - 3 * Wx @ Nx @ Wx)
            - d * (Wx @ Nx @ Wx @ Wx + Wx @ Wx @ Nx @ Wx))


def _se3_Jl(xi):
    w, nu = xi[:3], xi[3:]
    J = np.zeros((6, 6)); Jw = _so3_Jl(w)
    J[:3, :3] = Jw; J[3:, 3:] = Jw; J[3:, :3] = _Q(nu, w)
    return J


def _se3_Jr(xi):
    return _se3_Jl(-xi)


def _se3_Jr_inv(xi):
    return np.linalg.inv(_se3_Jr(xi))


class _Stand:
    """Minimal frame stand-in: what the spec says build_pose_graph_problem may rely on."""
    def __init__(self, fid, world_T_cam):
        self.id = fid; self.world_T_cam = np.array(world_T_cam, float)

    @property
    def cam_T_world(self):
        return np.linalg.inv(self.world_T_cam)

    def set_world_pose(self, T):
        self.world_T_cam = np.array(T, float)


def make_chain(rng, n_nodes):
    """Exact chain of n_nodes+1 world_T_cam poses with nonzero rotations and a consistent loop constraint."""
    T = np.eye(4); poses = [T.copy()]
    for _ in range(n_nodes):
        step = np.concatenate([rng.normal(0, 0.25, 3), rng.normal(0, 0.5, 3)])
        T = T @ _se3_exp(step); poses.append(T.copy())
    return poses


# --------------------------------------------------------------------------- #
# the probe proper
# --------------------------------------------------------------------------- #
def fd_compare(problem, eps):
    J = problem.jacobian()
    J = np.asarray(J.toarray() if hasattr(J, "toarray") else J, dtype=float)
    x_dim = J.shape[1]
    Jfd = np.empty_like(J)
    for j in range(x_dim):
        e = np.zeros(x_dim); e[j] = eps
        Jfd[:, j] = (np.asarray(problem.retract(e).residuals(), float) - np.asarray(problem.retract(-e).residuals(), float)) / (2 * eps)
    dof = 6
    rot = [dof * j + k for j in range(x_dim // dof) for k in range(3)]
    tr = [dof * j + 3 + k for j in range(x_dim // dof) for k in range(3)]
    def rel(cols):
        a, b = Jfd[:, cols], J[:, cols]
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    return rel(list(range(x_dim))), rel(rot), rel(tr)


def pose_diff(Ta, Tb):
    d = np.linalg.inv(Ta) @ Tb
    return float(np.linalg.norm(d[:3, 3]) + np.linalg.norm(_so3_log(d[:3, :3])))


def run_probe(build, solve, args, rng):
    out = {}
    n = args.n_nodes
    truth = make_chain(rng, n)
    # 1. FD vs analytic at a drifted base (nonzero residuals): build from drifted poses, exact loop pose
    drift = [np.concatenate([rng.normal(0, 0.05, 3), rng.normal(0, 0.1, 3)]) for _ in range(n + 1)]
    drifted = [truth[0]] + [truth[i] @ _se3_exp(drift[i]) for i in range(1, n + 1)]
    frames = [_Stand(i, T) for i, T in enumerate(drifted)]
    prob = build(frames, 0, n, np.linalg.inv(truth[n]))
    tot, rot, tr = fd_compare(prob, args.eps)
    out.update(fd_pg_rel_err_total=tot, fd_pg_rel_err_rot=rot, fd_pg_rel_err_t=tr)

    # 2. recovery: exact chain (zero residual at truth), perturbed through retract(), solved
    frames = [_Stand(i, T) for i, T in enumerate(truth)]
    exact = build(frames, 0, n, np.linalg.inv(truth[n]))
    delta = np.concatenate([np.concatenate([rng.normal(0, 0.03, 3), rng.normal(0, 0.05, 3)]) for _ in range(n)])
    solved, info = solve(exact.retract(delta), max_iterations=args.max_iter)
    sp = [np.asarray(T, float) for T in solved.poses()]
    out["pg_recovery_err_max"] = max(pose_diff(sp[j], truth[j + 1]) for j in range(n))

    # 3. coordinate invariance: world' = G world, same perturbation, map back
    G = _se3_exp(np.array([0.3, -0.2, 0.5, 1.0, -2.0, 0.5]))
    frames_g = [_Stand(i, G @ T) for i, T in enumerate(truth)]
    exact_g = build(frames_g, 0, n, np.linalg.inv(G @ truth[n]))
    solved_g, _ = solve(exact_g.retract(delta), max_iterations=args.max_iter)
    back = [np.linalg.inv(G) @ np.asarray(T, float) for T in solved_g.poses()]
    out["pg_coord_invariance_err"] = max(pose_diff(back[j], sp[j]) for j in range(n))
    return out


# --------------------------------------------------------------------------- #
# reference implementation for the self-test (independent of loop_closure.py)
# --------------------------------------------------------------------------- #
class _RefPG:
    def __init__(self, T, Z, T_loop, w_loop, flip=False):
        self.T = [np.array(x, float) for x in T]      # T[0] fixed, T[1..n] free
        self.Z = [np.array(x, float) for x in Z]; self.T_loop = np.array(T_loop, float)
        self.w_loop = float(w_loop); self.flip = flip
        self.n_nodes = len(self.T) - 1; self.dof = 6; self.x_dim = 6 * self.n_nodes

    def residuals(self):
        n = self.n_nodes; r = np.zeros(6 * (n + 1))
        for i in range(n):
            r[6 * i:6 * i + 6] = _se3_log(np.linalg.inv(self.Z[i]) @ np.linalg.inv(self.T[i]) @ self.T[i + 1])
        r[6 * n:] = self.w_loop * _se3_log(np.linalg.inv(self.T_loop) @ self.T[n])
        return r

    def jacobian(self):
        n = self.n_nodes; J = np.zeros((6 * (n + 1), self.x_dim)); r = self.residuals()
        for i in range(n):
            ri = r[6 * i:6 * i + 6]
            J[6 * i:6 * i + 6, 6 * i:6 * i + 6] = _se3_Jr_inv(ri)                       # node i+1 = free i
            if i >= 1:
                blk = -_se3_Jr_inv(ri) @ _adjoint(np.linalg.inv(self.T[i + 1]) @ self.T[i])
                J[6 * i:6 * i + 6, 6 * (i - 1):6 * i] = -blk if self.flip else blk        # node i = free i-1
        rl = r[6 * n:] / self.w_loop
        J[6 * n:, 6 * (n - 1):6 * n] = self.w_loop * _se3_Jr_inv(rl)
        return J

    def retract(self, delta):
        T = [self.T[0]] + [self.T[j + 1] @ _se3_exp(delta[6 * j:6 * j + 6]) for j in range(self.n_nodes)]
        return _RefPG(T, self.Z, self.T_loop, self.w_loop, self.flip)

    def poses(self):
        return [t.copy() for t in self.T[1:]]


def _ref_build(frames, loop_kf_idx, curr_idx, corrected_cam_T_world, w_loop=None, flip=False):
    T = [np.array(frames[i].world_T_cam, float) for i in range(loop_kf_idx, curr_idx + 1)]
    Z = [np.linalg.inv(T[i]) @ T[i + 1] for i in range(len(T) - 1)]
    n = len(T) - 1
    return _RefPG(T, Z, np.linalg.inv(corrected_cam_T_world), math.sqrt(n) if w_loop is None else w_loop, flip)


def _ref_solve(problem, *, max_iterations=20, **_):
    lam = 1e-3
    for _ in range(max_iterations):
        r = problem.residuals(); J = problem.jacobian(); A = J.T @ J; g = J.T @ r
        for _retry in range(6):
            d = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(len(A)), -g)
            cand = problem.retract(d)
            if np.sum(cand.residuals() ** 2) < np.sum(r ** 2):
                problem = cand; lam *= 0.1; break
            lam *= 10
        if np.linalg.norm(d) < 1e-12:
            break
    return problem, {"status": "completed"}


def _check_reference_jr(rng) -> float:
    """exp(xi) * exp(Jr(xi) dxi) ~= exp(xi + dxi): verifies the reference right Jacobian by FD."""
    worst = 0.0
    for _ in range(5):
        xi = np.concatenate([rng.normal(0, 0.6, 3), rng.normal(0, 1.0, 3)])
        for k in range(6):
            d = np.zeros(6); d[k] = 1e-6
            lhs = _se3_log(np.linalg.inv(_se3_exp(xi)) @ _se3_exp(xi + d)) / 1e-6
            worst = max(worst, float(np.linalg.norm(lhs - _se3_Jr(xi) @ d / 1e-6)))
    return worst


def self_test(args) -> int:
    rng = np.random.default_rng(args.seed)
    jr_err = _check_reference_jr(rng)
    good = run_probe(_ref_build, _ref_solve, args, np.random.default_rng(args.seed))
    bad = run_probe(lambda *a, **k: _ref_build(*a, flip=True, **k), _ref_solve, args, np.random.default_rng(args.seed))
    print(f"self-test, reference right Jacobian vs exponential map: {jr_err:.2e}")
    print("self-test, reference chart:", {k: f"{v:.2e}" for k, v in good.items()})
    print("self-test, sign-flipped odometry block:", {k: f"{v:.2e}" for k, v in bad.items()})
    ok = (jr_err < 1e-4 and good["fd_pg_rel_err_total"] <= args.tol and good["pg_recovery_err_max"] <= 1e-6
          and good["pg_coord_invariance_err"] <= 1e-6 and bad["fd_pg_rel_err_total"] > 0.1)
    print("self-test:", "OK (passes on the reference, catches the sign flip)" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-nodes", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args)
    try:
        from loop_closure import build_pose_graph_problem, solve_pose_graph  # noqa: E402
    except Exception as e:
        print(f"declared pose-graph chart interface not found in loop_closure.py: {type(e).__name__}: {e}")
        _emit({}, interface_missing=1)
        return 1
    try:
        out = run_probe(build_pose_graph_problem, solve_pose_graph, args, np.random.default_rng(args.seed))
    except Exception as e:
        print(f"probe could not run the interface to completion: {type(e).__name__}: {e}")
        _emit({}, interface_missing=1)
        return 1
    _emit(out)
    ok = out["fd_pg_rel_err_total"] <= args.tol
    print("FD:", "PASS" if ok else f"FAIL (tol {args.tol})", "| recovery/invariance bars are judged by the contract")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
