#!/usr/bin/env python3
"""
Bundle-adjustment chart probe (task H, rewritten 2026-09-17 for stage 01a).

Grades the on-manifold BA interface declared in plan/stages/01a-chart-and-backend.yaml
(spec A2/A3): optimization.BAProblem.from_arrays / residuals / jacobian / retract / poses /
points and optimization.solve_ba. The left perturbation R <- exp([dtheta]x) R with additive
t and additive points is the ruled chart (theory/conventions.md#rotation-parameterization).

INPUTS   optimization.py from the checkout on PYTHONPATH; a synthetic scene built here
         (nonzero rotations, points in front of every camera, two fixed cameras).
         Flags: --tol (pass bar for the FD blocks), --eps, --seed, --n-cams, --n-pts,
         --self-test (run this probe against an internal reference implementation).
METRICS  fd_ba_rot_rel_err_max   worst relative error, FD vs jacobian(), rotation columns
         fd_ba_t_rel_err_max     same, translation columns
         fd_ba_pt_rel_err_max    same, point columns
         ba_fixed_observer_rows_ok  1 if observations of FIXED cameras contribute residual
                                 rows that respond to point motion and no columns
         ba_recovery_reproj_rms_px  RMS reprojection residual after solve_ba from a perturbed
                                 start on a noise-free scene (two fixed cameras)
         ba_recovery_pose_err_max   max translation error over free cameras after that solve
         ba_coord_invariance_err    max pose/point difference between solving the scene in
                                 world coordinates and in coordinates moved by a global
                                 rigid transform, mapped back
         fd_ba_interface_missing    1 if the declared interface is absent (all others nan)
FAILURE  exit 1 after printing every METRIC line. An interface that does not exist yet
         (current main before stage 01a) prints nan metrics plus interface_missing=1 and
         exits 1: that is the intended "fails for the right reason" state before dispatch.
         A crash without METRIC lines is a harness failure (system/verify/run.py treats it so).
KNOWN PASS  `--self-test`: the probe machinery against the internal reference chart passes;
KNOWN FAIL  `--self-test` also flips the sign of the reference rotation block and asserts the
            probe reports it (fd_ba_rot_rel_err_max ~ 2). Both must hold or the self-test exits 1.
The pass bar for the recovery/invariance metrics lives in the contract's acceptance list,
not here; this probe only measures.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

METRICS = ["fd_ba_rot_rel_err_max", "fd_ba_t_rel_err_max", "fd_ba_pt_rel_err_max",
           "ba_fixed_observer_rows_ok", "ba_recovery_reproj_rms_px", "ba_recovery_pose_err_max",
           "ba_coord_invariance_err"]


def _emit(vals: dict, interface_missing: int = 0):
    for k in METRICS:
        v = vals.get(k, float("nan"))
        print(f"METRIC {k}={v:.6e}" if isinstance(v, float) and not math.isnan(v) else f"METRIC {k}={v}")
    print(f"METRIC fd_ba_interface_missing={interface_missing}")


# --------------------------------------------------------------------------- #
# minimal Lie helpers (probe-owned; independent of the implementation under test)
# --------------------------------------------------------------------------- #
def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=float)


def _so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + _hat(w)
    k = w / th
    K = _hat(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _so3_log(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-12:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# --------------------------------------------------------------------------- #
# synthetic scene
# --------------------------------------------------------------------------- #
def make_scene(rng, n_cams, n_pts, noise_px):
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1.0]])
    pts = rng.uniform([-1.5, -1.0, 3.0], [1.5, 1.0, 7.0], size=(n_pts, 3))
    poses = []
    for c in range(n_cams):
        w = rng.normal(0, 0.35, 3) if c else np.zeros(3)  # nonzero rotations except the first
        t = rng.normal(0, 0.4, 3) if c else np.zeros(3)
        T = np.eye(4); T[:3, :3] = _so3_exp(w); T[:3, 3] = t
        poses.append(T)
    poses = np.array(poses)
    obs = []
    for c in range(n_cams):
        R, t = poses[c, :3, :3], poses[c, :3, 3]
        for p in range(n_pts):
            xc = R @ pts[p] + t
            if xc[2] <= 0.2:
                continue
            uv = K @ xc; uv = uv[:2] / uv[2]
            obs.append((c, p, uv[0] + rng.normal(0, noise_px), uv[1] + rng.normal(0, noise_px)))
    return K, poses, pts, obs


# --------------------------------------------------------------------------- #
# the probe proper (works on any object exposing the A2/A3 interface)
# --------------------------------------------------------------------------- #
def fd_blocks(problem, eps):
    """Central FD of residuals() through retract() vs jacobian(); relative error per block."""
    J = np.asarray(problem.jacobian(), dtype=float)
    x_dim = J.shape[1]
    Jfd = np.empty_like(J)
    for j in range(x_dim):
        e = np.zeros(x_dim); e[j] = eps
        rp = np.asarray(problem.retract(e).residuals(), dtype=float)
        rm = np.asarray(problem.retract(-e).residuals(), dtype=float)
        Jfd[:, j] = (rp - rm) / (2 * eps)
    n_free = problem.n_free
    def rel(cols):
        if not len(cols):
            return 0.0
        a, b = Jfd[:, cols], J[:, cols]
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    rot_cols = [6 * c + k for c in range(n_free) for k in range(3)]
    t_cols = [6 * c + 3 + k for c in range(n_free) for k in range(3)]
    pt_cols = list(range(6 * n_free, x_dim))
    return rel(rot_cols), rel(t_cols), rel(pt_cols), J, Jfd


def fixed_observer_rows_ok(problem, obs, fixed_mask, Jfd):
    """Fixed cameras: their observations own residual rows (that move with their point) and no columns."""
    if problem.x_dim != 6 * problem.n_free + 3 * problem.n_points:
        return 0
    if len(np.asarray(problem.residuals())) != 2 * len(obs):
        return 0
    n_free = problem.n_free
    for k, (c, p, _, _) in enumerate(obs):
        if fixed_mask[c]:
            block = Jfd[2 * k:2 * k + 2, 6 * n_free + 3 * p: 6 * n_free + 3 * p + 3]
            if np.linalg.norm(block) < 1e-6:
                return 0
    return 1


def perturb_state(rng, poses, pts, fixed_mask, rot_deg=2.0, t_frac=0.02, pt_frac=0.01):
    P = poses.copy(); Q = pts.copy()
    for c in range(len(P)):
        if fixed_mask[c]:
            continue
        w = rng.normal(0, 1, 3); w = w / np.linalg.norm(w) * math.radians(rot_deg)
        P[c, :3, :3] = _so3_exp(w) @ P[c, :3, :3]
        P[c, :3, 3] = P[c, :3, 3] + rng.normal(0, t_frac, 3)
    Q = Q + rng.normal(0, pt_frac, Q.shape)
    return P, Q


def pose_err(Pa, Pb, free):
    return max(float(np.linalg.norm(Pa[c, :3, 3] - Pb[c, :3, 3])) for c in free) if free else 0.0


def run_probe(BAProblem, solve_ba, args, rng):
    out = {}
    # 1. FD vs analytic on a noisy scene with two fixed cameras
    K, poses, pts, obs = make_scene(rng, args.n_cams, args.n_pts, noise_px=1.0)
    fixed = np.zeros(args.n_cams, dtype=bool); fixed[:2] = True
    prob = BAProblem.from_arrays(K, poses, fixed, pts, obs)
    r_rot, r_t, r_pt, J, Jfd = fd_blocks(prob, args.eps)
    out.update(fd_ba_rot_rel_err_max=r_rot, fd_ba_t_rel_err_max=r_t, fd_ba_pt_rel_err_max=r_pt)
    out["ba_fixed_observer_rows_ok"] = float(fixed_observer_rows_ok(prob, obs, fixed, Jfd))

    # 2. recovery on a noise-free scene from a perturbed start (two fixed cameras: gauge fully fixed)
    K, poses, pts, obs = make_scene(rng, args.n_cams, args.n_pts, noise_px=0.0)
    P0, Q0 = perturb_state(rng, poses, pts, fixed)
    solved, info = solve_ba(BAProblem.from_arrays(K, P0, fixed, Q0, obs), max_iterations=args.max_iter)
    r = np.asarray(solved.residuals(), dtype=float)
    out["ba_recovery_reproj_rms_px"] = float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))  # RMS of the 2-vector norm
    free = [c for c in range(args.n_cams) if not fixed[c]]
    out["ba_recovery_pose_err_max"] = pose_err(np.asarray(solved.poses()), poses, free)

    # 3. coordinate invariance: same perturbed start expressed in world' = G world
    Rg = _so3_exp(np.array([0.3, -0.2, 0.5])); tg = np.array([1.0, -2.0, 0.5])
    G = np.eye(4); G[:3, :3] = Rg; G[:3, 3] = tg
    Ginv = np.linalg.inv(G)
    P0g = np.array([T @ Ginv for T in P0])                       # cam_T_world' = cam_T_world G^-1
    Q0g = (Rg @ Q0.T).T + tg                                       # world' points
    solved_g, _ = solve_ba(BAProblem.from_arrays(K, P0g, fixed, Q0g, obs), max_iterations=args.max_iter)
    back_P = np.array([T @ G for T in np.asarray(solved_g.poses())])   # cam_T_world = cam_T_world' G
    back_Q = (Rg.T @ (np.asarray(solved_g.points()) - tg).T).T
    dP = pose_err(back_P, np.asarray(solved.poses()), free)
    dR = max(float(np.linalg.norm(_so3_log(back_P[c, :3, :3] @ np.asarray(solved.poses())[c, :3, :3].T))) for c in free) if free else 0.0
    dQ = float(np.max(np.linalg.norm(back_Q - np.asarray(solved.points()), axis=1)))
    out["ba_coord_invariance_err"] = max(dP, dR, dQ)
    return out


# --------------------------------------------------------------------------- #
# reference implementation for the self-test (independent of optimization.py)
# --------------------------------------------------------------------------- #
class _RefBA:
    def __init__(self, K, poses, fixed, pts, obs, flip_rot=False):
        self.K = np.array(K, float); self.P = np.array(poses, float); self.fixed = np.array(fixed, bool)
        self.Q = np.array(pts, float); self.obs = list(obs); self.flip = flip_rot
        self.n_cams = len(self.P); self.free_cam_indices = [c for c in range(self.n_cams) if not self.fixed[c]]
        self.n_free = len(self.free_cam_indices); self.n_fixed = self.n_cams - self.n_free
        self.n_points = len(self.Q); self.n_obs = len(self.obs); self.x_dim = 6 * self.n_free + 3 * self.n_points
        self.col = {c: i for i, c in enumerate(self.free_cam_indices)}

    @classmethod
    def from_arrays(cls, K, cam_T_world, fixed_mask, points, observations, huber_delta_px=2.4477):
        return cls(K, cam_T_world, fixed_mask, points, observations)

    def _proj(self, c, p):
        R, t = self.P[c, :3, :3], self.P[c, :3, 3]
        x = R @ self.Q[p] + t
        return x, self.K @ x

    def residuals(self):
        r = np.empty(2 * self.n_obs)
        for k, (c, p, u, v) in enumerate(self.obs):
            x, h = self._proj(c, p)
            if x[2] <= 1e-6:
                r[2 * k:2 * k + 2] = 100.0; continue
            r[2 * k] = h[0] / h[2] - u; r[2 * k + 1] = h[1] / h[2] - v
        return r

    def jacobian(self):
        J = np.zeros((2 * self.n_obs, self.x_dim))
        fx, fy = self.K[0, 0], self.K[1, 1]
        for k, (c, p, u, v) in enumerate(self.obs):
            x, _ = self._proj(c, p)
            if x[2] <= 1e-6:
                continue
            X, Y, Z = x
            Pm = np.array([[fx / Z, 0, -fx * X / Z ** 2], [0, fy / Z, -fy * Y / Z ** 2]])
            R = self.P[c, :3, :3]
            if c in self.col:
                i = self.col[c]
                drot = -_hat(R @ self.Q[p])
                if self.flip:
                    drot = -drot
                J[2 * k:2 * k + 2, 6 * i:6 * i + 3] = Pm @ drot
                J[2 * k:2 * k + 2, 6 * i + 3:6 * i + 6] = Pm
            J[2 * k:2 * k + 2, 6 * self.n_free + 3 * p:6 * self.n_free + 3 * p + 3] = Pm @ R
        return J

    def retract(self, delta):
        P = self.P.copy(); Q = self.Q.copy()
        for i, c in enumerate(self.free_cam_indices):
            d = delta[6 * i:6 * i + 6]
            P[c, :3, :3] = _so3_exp(d[:3]) @ P[c, :3, :3]; P[c, :3, 3] = P[c, :3, 3] + d[3:]
        Q = Q + delta[6 * self.n_free:].reshape(-1, 3)
        return _RefBA(self.K, P, self.fixed, Q, self.obs, self.flip)

    def poses(self):
        return self.P.copy()

    def points(self):
        return self.Q.copy()


def _ref_solve_ba(problem, *, max_iterations=10, **_):
    lam = 1e-3
    for _ in range(max_iterations):
        r = problem.residuals(); J = problem.jacobian()
        A = J.T @ J; g = J.T @ r
        for _retry in range(6):
            d = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(len(A)), -g)
            cand = problem.retract(d)
            if 0.5 * np.sum(cand.residuals() ** 2) < 0.5 * np.sum(r ** 2):
                problem = cand; lam *= 0.1; break
            lam *= 10
        if np.linalg.norm(d) < 1e-10:
            break
    return problem, {"status": "completed"}


def self_test(args) -> int:
    rng = np.random.default_rng(args.seed)
    good = run_probe(_RefBA, _ref_solve_ba, args, rng)
    rng = np.random.default_rng(args.seed)

    class _Flipped(_RefBA):
        @classmethod
        def from_arrays(cls, K, cam_T_world, fixed_mask, points, observations, huber_delta_px=2.4477):
            return cls(K, cam_T_world, fixed_mask, points, observations, flip_rot=True)
    bad = run_probe(_Flipped, _ref_solve_ba, args, rng)
    print("self-test, reference chart:", {k: f"{v:.2e}" for k, v in good.items()})
    print("self-test, sign-flipped rotation block:", {k: f"{v:.2e}" for k, v in bad.items()})
    ok = (good["fd_ba_rot_rel_err_max"] <= args.tol and good["fd_ba_t_rel_err_max"] <= args.tol
          and good["fd_ba_pt_rel_err_max"] <= args.tol and good["ba_fixed_observer_rows_ok"] == 1
          and good["ba_recovery_pose_err_max"] <= 1e-3 and good["ba_recovery_reproj_rms_px"] <= 1e-2
          and good["ba_coord_invariance_err"] <= 1e-3
          and bad["fd_ba_rot_rel_err_max"] > 0.5)
    print("self-test:", "OK (passes on the reference, catches the sign flip)" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cams", type=int, default=5)
    ap.add_argument("--n-pts", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args)
    try:
        from optimization import BAProblem, solve_ba  # noqa: E402
        for name in ("from_arrays",):
            getattr(BAProblem, name)
    except Exception as e:  # interface absent (expected on main before stage 01a)
        print(f"declared BA chart interface not found in optimization.py: {type(e).__name__}: {e}")
        _emit({}, interface_missing=1)
        return 1
    rng = np.random.default_rng(args.seed)
    try:
        out = run_probe(BAProblem, solve_ba, args, rng)
    except Exception as e:
        print(f"probe could not run the interface to completion: {type(e).__name__}: {e}")
        _emit({}, interface_missing=1)
        return 1
    _emit(out)
    fd_ok = all(out[k] <= args.tol for k in ("fd_ba_rot_rel_err_max", "fd_ba_t_rel_err_max", "fd_ba_pt_rel_err_max"))
    print("FD blocks:", "PASS" if fd_ok else f"FAIL (tol {args.tol})", "| recovery/invariance bars are judged by the contract")
    return 0 if fd_ok and out["ba_fixed_observer_rows_ok"] == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
