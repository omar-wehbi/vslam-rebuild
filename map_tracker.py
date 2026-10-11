"""
Stage 04l (milestone 04-2, rulings G1-G4, workshop/22): the default arm of the slot "Pose from the
map", run against a PERFECT map (rungs M1a and M1b, design.md#oracle-ladder-rung-definitions).

Per frame k >= 1 (papers/orb-slam.md#tracking-map-based-not-frame-to-frame;
conventions.md#tracking-model): the pose is predicted from the two-view ROTATION and a constant
velocity translation (predict_pose); map points are searched around their predicted projections,
first the points linked in frame k-1 (step A), then the local map (step B); the pose is solved by a
motion-only bundle adjustment with a Huber cost on the 01a chart (motion_only_ba); the frame is
accepted on the inlier count after optimization (no distance-from-prediction gate, the O4
finding). A frame that fails is LOST and carries its prediction. Under M1b the 2D-3D
correspondences are read from the map file instead of searched.

The map file (spec F) is written by the harness builder; this module only reads it. Points of a
keyframe are released into the Map only after that keyframe's pose is final (release), so a
keyframe's own points never help track it. The released map is never edited afterwards.

This module reads no dataset file other than the --m1_map file.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import Point as point_mod
import residual_test
import tracking
from front_end import PROTOCOL
from local_mapping import _desc_dist_matrix, _keypoint_sigmas, _norm_type
from optimization import so3_exp

PIXEL_TOL = 1e-3          # spec C3 / D: a file pixel must equal the product keypoint's to 1e-3 px
Z_MIN = 1e-6              # the BAProblem convention: camera z at or below this is "behind"
BEHIND_RESIDUAL = 100.0


@dataclass
class MapTrackerParams:
    """Constants of the map-based tracker (spec B1).

    Provenance: ORB-SLAM2 implementation constants of the paper's tracking (the paper names the
    steps without numbers, as with the 0.01 and 1.1 degree of 04a): radius 15 for the monocular
    motion-model search, retried at twice the radius under 20 matches, success at 10 map inliers;
    local-map radius 4 (the wider value of its viewing-angle rule); 30 inliers to accept the frame;
    ratio 0.8 in the local-map search; the 60 degree viewing-angle test; chi-square 5.991 (2 dof,
    95 percent); four optimizations of ten iterations with the robust kernel removed after the
    third. The distance-ratio window [0.5, 2] is this contract's reading of the paper's
    [d_min, d_max] scale-invariance region for SIFT (one octave each way from the distance at which
    the point was observed); the paper derives it from the ORB pyramid. Every radius is in units of
    the candidate keypoint's sigma (local_mapping._keypoint_sigmas: SIFT 2^max(octave, 0), ORB
    1.2^level).
    """
    radius_a: float = 15.0
    radius_b: float = 4.0
    min_matches_a: int = 20
    min_inliers_a: int = 10
    min_inliers: int = 30
    ratio_b: float = 0.8
    max_view_angle_deg: float = 60.0
    dist_ratio_min: float = 0.5
    dist_ratio_max: float = 2.0
    chi2: float = 5.991
    rounds: int = 4
    iterations: int = 10
    huber_rounds: int = 3


@dataclass
class MapRelocParams:
    """Constants of the relocalizing state (stage 04m spec C; papers/orb-slam.md section V.C).

    Provenance: ORB-SLAM2 implementation constants of the paper's relocalization (the paper names
    the steps without numbers): the PnP solver's 300 iterations at 0.99 probability with at least 10
    inliers; its per-point threshold chi-square 5.991 (2 dof, 95 percent) at sigma 1 px, here one
    pixel threshold sqrt(5.991) = 2.4477 because cv2.solvePnPRansac takes a single threshold (SIFT
    keypoints at octave <= 0 have sigma 1, the model 04j stated); a candidate needs 15 descriptor
    matches; ratio 0.75 in the relocalization matcher; the guided search after the first pose
    optimization at radius 10, in units of the candidate keypoint's sigma as every radius of 04l.
    A relocalized frame is accepted at MapTrackerParams.min_inliers (30), the count of a tracked
    frame; ORB-SLAM2's 50 for a relocalized frame is not carried.
    """
    ransac_iterations: int = 300
    ransac_reproj_px: float = 2.4477
    ransac_confidence: float = 0.99
    ransac_min_inliers: int = 10
    min_matches_m1a: int = 15
    ratio_m1a: float = 0.75
    search_radius: float = 10.0


# --------------------------------------------------------------------------- the map file (B2, F)
@dataclass
class M1Map:
    header: dict = field(default_factory=dict)
    points: list = field(default_factory=list)   # (pid, kf, kp, u, v, X, Y, Z)
    corr: dict = field(default_factory=dict)     # frame -> [(kp, pid, u, v)]


def load_m1_map(path):
    """Parse the perfect-map file of spec F. A malformed line raises ValueError naming its number."""
    header = None
    points = []
    corr = {}
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        try:
            if f[0] == "H":
                h = {}
                for tok in f[1:]:
                    if "=" not in tok:
                        raise ValueError(f"header token {tok!r} is not key=value")
                    key, val = tok.split("=", 1)
                    h[key] = val
                if header is not None:
                    raise ValueError("second H line")
                header = h
            elif f[0] == "P" and len(f) == 9:
                points.append((int(f[1]), int(f[2]), int(f[3]), float(f[4]), float(f[5]),
                               float(f[6]), float(f[7]), float(f[8])))
            elif f[0] == "C" and len(f) == 6:
                corr.setdefault(int(f[1]), []).append((int(f[2]), int(f[3]), float(f[4]), float(f[5])))
            else:
                raise ValueError(f"unknown record or wrong field count ({len(f)} fields)")
        except (ValueError, IndexError) as e:
            raise ValueError(f"{path}: line {i}: malformed: {e}") from None
    if header is None:
        raise ValueError(f"{path}: no H line")
    return M1Map(header=header, points=points, corr=corr)


# --------------------------------------------------------------------------- prediction (B3)
def predict_pose(T_prev, T_prev2, R_twoview):
    """cam_T_world prediction of frame k from frames k-1, k-2 (spec B3, C1).

    V = T_prev inv(T_prev2) (identity when T_prev2 is None) is the last relative motion in the
    convention x_k = R x_(k-1) + t; its rotation block is replaced by the two-view rotation when
    there is one (translation block unchanged); the prediction is V' T_prev.
    """
    T_prev = np.asarray(T_prev, dtype=np.float64)
    if T_prev2 is None:
        V = np.eye(4)
    else:
        V = T_prev @ np.linalg.inv(np.asarray(T_prev2, dtype=np.float64))
    V = np.array(V, dtype=np.float64, copy=True)
    if R_twoview is not None:
        V[:3, :3] = np.asarray(R_twoview, dtype=np.float64)
    return V @ T_prev


# --------------------------------------------------------------------------- motion-only BA (B4)
def _as_obs(points, uv, sigma):
    X = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    UV = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    S = np.asarray(sigma, dtype=np.float64).reshape(-1)
    return X, UV, S


def motion_only_residuals(K, cam_T_world, points, uv, sigma):
    """(2n,) whitened residuals (pi(K (R X_i + t)) - uv_i) / sigma_i; 100 in both rows when behind."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(cam_T_world, dtype=np.float64)
    X, UV, S = _as_obs(points, uv, sigma)
    Xc = X @ T[:3, :3].T + T[:3, 3]
    r = np.full((len(X), 2), BEHIND_RESIDUAL)
    ok = Xc[:, 2] > Z_MIN
    if np.any(ok):
        p = Xc[ok] @ K.T
        r[ok] = (p[:, :2] / p[:, 2:3] - UV[ok]) / S[ok, None]
    return r.reshape(-1)


def _proj_jac(K, Xc):
    """(n, 2, 3) d(u, v)/d(x_cam) for the pinhole projection through K (x_cam in front)."""
    p = Xc @ K.T
    z = p[:, 2]
    u = p[:, 0] / z
    v = p[:, 1] / z
    P = np.empty((len(Xc), 2, 3))
    P[:, 0, :] = (K[0][None, :] - u[:, None] * K[2][None, :]) / z[:, None]
    P[:, 1, :] = (K[1][None, :] - v[:, None] * K[2][None, :]) / z[:, None]
    return P


def motion_only_jacobian(K, cam_T_world, points, uv, sigma):
    """(2n, 6) derivative of motion_only_residuals w.r.t. delta = (dtheta, dt) at delta = 0 on the
    01a chart R <- exp([dtheta]x) R, t <- t + dt: rows -P [R X]x / sigma and P / sigma,
    P = d(u,v)/d(x_cam); zero rows for an observation behind the camera."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(cam_T_world, dtype=np.float64)
    X, _UV, S = _as_obs(points, uv, sigma)
    n = len(X)
    J = np.zeros((n, 2, 6))
    RX = X @ T[:3, :3].T
    Xc = RX + T[:3, 3]
    ok = Xc[:, 2] > Z_MIN
    if np.any(ok):
        P = _proj_jac(K, Xc[ok])
        a = RX[ok]
        # -[a]x = [[0, a2, -a1], [-a2, 0, a0], [a1, -a0, 0]]
        negskew = np.zeros((len(a), 3, 3))
        negskew[:, 0, 1] = a[:, 2]
        negskew[:, 0, 2] = -a[:, 1]
        negskew[:, 1, 0] = -a[:, 2]
        negskew[:, 1, 2] = a[:, 0]
        negskew[:, 2, 0] = a[:, 1]
        negskew[:, 2, 1] = -a[:, 0]
        s = S[ok][:, None, None]
        J[ok, :, :3] = np.einsum("nij,njk->nik", P, negskew) / s
        J[ok, :, 3:] = P / s
    return J.reshape(2 * n, 6)


def retract_pose(cam_T_world, delta):
    """The 01a chart: R <- exp([dtheta]x) R, t <- t + dt, delta = (dtheta, dt)."""
    T = np.array(cam_T_world, dtype=np.float64, copy=True)
    d = np.asarray(delta, dtype=np.float64).reshape(6)
    T[:3, :3] = so3_exp(d[:3]) @ T[:3, :3]
    T[:3, 3] = T[:3, 3] + d[3:]
    return T


def _cam_z(T, X):
    return X @ T[2, :3] + T[2, 3]


def motion_only_ba(K, cam_T_world, points, uv, sigma, params):
    """Motion-only BA (spec B4): points fixed, pose optimized with Levenberg-Marquardt on the 01a
    chart, Huber cost (delta sqrt(chi2)) in the first params.huber_rounds rounds and L2 after;
    every observation re-classified after each round (outlier iff ||e||^2 > chi2 or behind).
    Returns (cam_T_world, inlier_mask, info). Deterministic."""
    K = np.asarray(K, dtype=np.float64)
    X, UV, S = _as_obs(points, uv, sigma)
    n = len(X)
    T = np.array(cam_T_world, dtype=np.float64, copy=True)
    if n < 3:
        return T, np.zeros(n, dtype=bool), dict(status="too_few", n_obs=n, n_inliers=0,
                                                cost_initial=float("nan"), cost_final=float("nan"),
                                                iterations=0)
    chi2 = float(params.chi2)
    hub = math.sqrt(chi2)

    def err_norms(Tc, sel):
        e = motion_only_residuals(K, Tc, X[sel], UV[sel], S[sel]).reshape(-1, 2)
        return e, np.sqrt((e * e).sum(1))

    def cost(Tc, sel, huber):
        _e, en = err_norms(Tc, sel)
        if huber:
            return float(np.sum(np.where(en <= hub, 0.5 * en * en, hub * (en - 0.5 * hub))))
        return float(np.sum(0.5 * en * en))

    inl = np.ones(n, dtype=bool)
    cost_initial = None
    iters = 0
    for rnd in range(int(params.rounds)):
        huber = rnd < int(params.huber_rounds)
        sel = np.flatnonzero(inl)
        if len(sel) >= 3:
            lam = 1e-3
            c = cost(T, sel, huber)
            if cost_initial is None:
                cost_initial = c
            for _it in range(int(params.iterations)):
                e, en = err_norms(T, sel)
                J = motion_only_jacobian(K, T, X[sel], UV[sel], S[sel]).reshape(-1, 2, 6)
                if huber:
                    w = np.where(en <= hub, 1.0, hub / np.maximum(en, 1e-12))
                else:
                    w = np.ones(len(en))
                H = np.einsum("n,nai,naj->ij", w, J, J)
                g = np.einsum("n,nai,na->i", w, J, e)
                iters += 1
                accepted = False
                for _try in range(6):          # the first solve plus at most 5 retries
                    dx = np.linalg.solve(H + lam * np.eye(6), -g)
                    Tn = retract_pose(T, dx)
                    cn = cost(Tn, sel, huber)
                    if cn < c:
                        accepted = True
                        break
                    lam *= 10.0
                if not accepted:
                    break
                rel = (c - cn) / max(c, 1e-300)
                T, c = Tn, cn
                lam *= 0.1
                if rel < 1e-6 or float(np.linalg.norm(dx)) < 1e-8:
                    break
        e_all = motion_only_residuals(K, T, X, UV, S).reshape(-1, 2)
        inl = ((e_all * e_all).sum(1) <= chi2) & (_cam_z(T, X) > Z_MIN)
    sel = np.flatnonzero(inl)
    cost_final = cost(T, sel, False) if len(sel) else float("nan")
    return T, inl, dict(status="ok", n_obs=n, n_inliers=int(inl.sum()),
                        cost_initial=(float("nan") if cost_initial is None else cost_initial),
                        cost_final=cost_final, iterations=iters)


# --------------------------------------------------------------------------- relocalization (04m D5)
def relocalize_from_correspondences(K, points, uv, sigma, params, reloc_params):
    """Pose from 2D-3D correspondences without a prior (stage 04m spec D5): EPnP inside RANSAC,
    unseeded as the 01c loop-closure PnP (conventions.md#loop-closure-relocalization-seed), then
    motion_only_ba from the RANSAC pose on ALL n observations (its rounds re-classify every one).

    points (n, 3), uv (n, 2) undistorted pixels, sigma (n,). Returns (cam_T_world, inlier_mask,
    info): (None, all-False, status "too_few") for n < 4; (None, all-False, status "pnp") when RANSAC
    fails or keeps fewer than reloc_params.ransac_min_inliers; otherwise the BA pose and mask with
    info status "ok". Deterministic (the OpenCV RNG is seeded with 0 before each solve)."""
    import cv2
    X, UV, S = _as_obs(points, uv, sigma)
    n = len(X)
    if n < 4:
        return None, np.zeros(n, dtype=bool), dict(status="too_few", n_obs=n, ransac_inliers=-1, n_inliers=0)
    cv2.setRNGSeed(0)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(np.ascontiguousarray(X), np.ascontiguousarray(UV),
                                             np.asarray(K, dtype=np.float64), None, useExtrinsicGuess=False,
                                             iterationsCount=int(reloc_params.ransac_iterations),
                                             reprojectionError=float(reloc_params.ransac_reproj_px),
                                             confidence=float(reloc_params.ransac_confidence),
                                             flags=cv2.SOLVEPNP_EPNP)
    n_r = 0 if inl is None else int(len(inl))
    if not ok or n_r < int(reloc_params.ransac_min_inliers):
        return None, np.zeros(n, dtype=bool), dict(status="pnp", n_obs=n, ransac_inliers=n_r, n_inliers=0)
    T_r = np.eye(4)
    T_r[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T_r[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    T, mask, _info = motion_only_ba(K, T_r, X, UV, S, params)
    mask = np.asarray(mask, dtype=bool)
    return T, mask, dict(status="ok", n_obs=n, ransac_inliers=n_r, n_inliers=int(mask.sum()))


# --------------------------------------------------------------------------- the solver arm (04q C)
RELOC_SOLVERS = ("epnp", "sqpnp", "newton")


def _npnp_record():
    """The not-reached NPnP solve record of spec C2."""
    return dict(npnp_converged=-1, npnp_gap_rel=float("nan"), npnp_gamma=float("nan"), npnp_cost=float("nan"),
                npnp_outer=-1, npnp_newton=-1, npnp_ls_evals=-1)


def _finite_pose(T):
    return T is not None and bool(np.all(np.isfinite(T)))


def behind_count(cam_T_world, points):
    """Stage 04q spec C1: the number of points whose camera depth z = (R X + t)_z is <= 0 (D1c-5 (1))."""
    T = np.asarray(cam_T_world, dtype=np.float64)
    X = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    z = (X @ T[:3, :3].T + T[:3, 3])[:, 2]
    return int(np.count_nonzero(z <= 0.0))


def refit_consensus(K, points, uv, T_ransac, solver, scale=1.0):
    """Stage 04q spec C2: refit the RANSAC consensus set (points (n, 3) map units, uv (n, 2) undistorted
    pixels) with the arm's solver. Returns (cam_T_world or None, info); None is a refit failure.

    epnp: a copy of T_ransac (cv2.solvePnPRansac with SOLVEPNP_EPNP already re-solves its inliers by
    EPnP, so this is the 04m pose). sqpnp: cv2.solvePnP with SOLVEPNP_SQPNP. newton: npnp.solve on
    P = points / scale and the bearings inv(K) (u, v, 1); the pose is [R | scale t]; not converged is a
    failure; the certificate npnp_gap_rel is recorded, never used. An exception in the refit is a
    failure. info: refit_ms (wall time of this call) and the NPnP solve record (nan / -1 unless newton
    and npnp.solve returned)."""
    if solver not in RELOC_SOLVERS:
        raise ValueError(f"unknown relocalization solver {solver!r}")
    t0 = time.perf_counter()
    info = _npnp_record()
    T = None
    try:
        Kf = np.array(K, dtype=np.float64)
        X = np.array(points, dtype=np.float64).reshape(-1, 3)
        UV = np.array(uv, dtype=np.float64).reshape(-1, 2)
        if solver == "epnp":
            T = np.array(T_ransac, dtype=np.float64, copy=True)
        elif solver == "sqpnp":
            import cv2
            ok, rvec, tvec = cv2.solvePnP(X, UV, Kf, None, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
                T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        else:
            import npnp                                  # never at module level (spec C2)
            P = X / float(scale)
            V = np.column_stack([UV, np.ones(len(UV))]) @ np.linalg.inv(Kf).T
            r = npnp.solve(P, V)
            info.update(npnp_converged=int(bool(r.converged)), npnp_gamma=float(r.gamma), npnp_cost=float(r.cost),
                        npnp_outer=int(r.outer_iters), npnp_newton=int(r.newton_iters),
                        npnp_ls_evals=int(r.line_search_evals))
            info["npnp_gap_rel"] = float(r.gap) / float(np.linalg.norm(npnp.rotation_cost_matrix(P, V)))
            if r.converged:
                T = np.eye(4)
                T[:3, :3] = np.asarray(r.R, dtype=np.float64)
                T[:3, 3] = float(scale) * np.asarray(r.t, dtype=np.float64).reshape(3)
        if not _finite_pose(T):
            T = None
    except Exception:
        T = None
    info["refit_ms"] = (time.perf_counter() - t0) * 1e3
    return T, info


def _peak_refit_bytes(K, points, uv, T_ransac, solver, scale=1.0):
    """Stage 04q spec E1a: tracemalloc's peak over a second, discarded refit_consensus call on copies of
    the inputs; -1 when tracemalloc is already tracing (never disturbed)."""
    import tracemalloc
    if tracemalloc.is_tracing():
        return -1
    args = (np.array(K, copy=True), np.array(points, copy=True), np.array(uv, copy=True),
            np.array(T_ransac, copy=True))
    tracemalloc.start()
    try:
        refit_consensus(*args, solver, scale)
        peak = int(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()
    return peak


def _attempt_info(n):
    info = dict(status="too_few", n_obs=int(n), ransac_inliers=-1, n_inliers=0, consensus=np.zeros(n, dtype=bool),
                behind=-1, refit_pose=None, refit_ms=-1.0)
    info.update(_npnp_record())
    return info


def _ransac_consensus(K, X, UV, reloc_params):
    """04m's RANSAC step (spec C3 (2)): (ransac_inliers, consensus mask, T_ransac or None). T_ransac is
    None when RANSAC failed or kept fewer than reloc_params.ransac_min_inliers (consensus all False)."""
    import cv2
    n = len(X)
    cv2.setRNGSeed(0)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(np.ascontiguousarray(X), np.ascontiguousarray(UV),
                                             np.asarray(K, dtype=np.float64), None, useExtrinsicGuess=False,
                                             iterationsCount=int(reloc_params.ransac_iterations),
                                             reprojectionError=float(reloc_params.ransac_reproj_px),
                                             confidence=float(reloc_params.ransac_confidence),
                                             flags=cv2.SOLVEPNP_EPNP)
    n_r = 0 if inl is None else int(len(inl))
    cons = np.zeros(n, dtype=bool)
    if not ok or n_r < int(reloc_params.ransac_min_inliers):
        return n_r, cons, None
    cons[np.asarray(inl, dtype=np.int64).reshape(-1)] = True
    T_r = np.eye(4)
    T_r[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T_r[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return n_r, cons, T_r


def _solve_from_consensus(K, X, UV, S, params, T_ransac, cons, solver, info, scale=1.0, peak=False):
    """Spec C3 (3)-(5) from a consensus set: the arm's refit, the cheirality rule of D1c-5 (1), then
    motion_only_ba on ALL observations. Fills info; returns (T or None, mask). With peak set (the shadow
    frames of E1a) info["refit_peak_b"] gets the second, traced refit's peak."""
    n = len(X)
    Xc, UVc = X[cons], UV[cons]
    t0 = time.perf_counter()
    try:
        T_ref, rinfo = refit_consensus(K, Xc, UVc, T_ransac, solver, scale)
    except Exception:                                    # a refit failure (spec C2), whatever raised
        T_ref, rinfo = None, dict(refit_ms=(time.perf_counter() - t0) * 1e3)
    for key, val in _npnp_record().items():
        info[key] = rinfo.get(key, val)
    info["refit_ms"] = float(rinfo.get("refit_ms", -1.0))
    if peak:
        info["refit_peak_b"] = _peak_refit_bytes(K, Xc, UVc, T_ransac, solver, scale)
    if T_ref is None:
        info["status"] = "refit"
        return None, np.zeros(n, dtype=bool)
    T_ref = np.asarray(T_ref, dtype=np.float64)
    info["refit_pose"] = T_ref
    info["behind"] = behind_count(T_ref, Xc)
    if 2 * info["behind"] > int(cons.sum()):
        info["status"] = "cheirality"
        return None, np.zeros(n, dtype=bool)
    T, mask, _i = motion_only_ba(K, T_ref, X, UV, S, params)
    mask = np.asarray(mask, dtype=bool)
    info.update(status="ok", n_inliers=int(mask.sum()))
    return T, mask


def relocalize_with_solver(K, points, uv, sigma, params, reloc_params, solver, scale=1.0):
    """Stage 04q spec C3: the 04m relocalization with the solver arm. (1) n < 4: status too_few;
    (2) cv2.solvePnPRansac exactly as relocalize_from_correspondences (the same hypotheses and consensus
    set in every arm), status ransac on failure or fewer than ransac_min_inliers; (3) refit_consensus
    on the consensus rows, status refit on None; (4) status cheirality when more than half the
    consensus set is at non-positive depth under the refit; (5) motion_only_ba from the refit on all
    n observations, status ok. Returns (cam_T_world or None, inlier_mask, info); info carries status,
    n_obs, ransac_inliers, n_inliers, consensus, behind, refit_pose, refit_ms and the NPnP solve record."""
    if solver not in RELOC_SOLVERS:
        raise ValueError(f"unknown relocalization solver {solver!r}")
    X, UV, S = _as_obs(points, uv, sigma)
    n = len(X)
    info = _attempt_info(n)
    if n < 4:
        return None, np.zeros(n, dtype=bool), info
    n_r, cons, T_r = _ransac_consensus(K, X, UV, reloc_params)
    info["ransac_inliers"] = n_r
    if T_r is None:
        info["status"] = "ransac"
        return None, np.zeros(n, dtype=bool), info
    info["consensus"] = cons
    T, mask = _solve_from_consensus(K, X, UV, S, params, T_r, cons, solver, info, scale)
    return T, mask, info


def _hull_share(uv, w, h):
    """Spec D: convex-hull area of uv over the image area w * h; 0 with fewer than 3 points."""
    import cv2
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) < 3:
        return 0.0
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull)) / float(w * h)


def _fmt_pose(T):
    if T is None:
        return "none"
    return ",".join(repr(float(x)) for x in np.asarray(T, dtype=np.float64)[:3, :4].reshape(-1))


def _fmt_floats(A):
    return ",".join(repr(float(x)) for x in np.asarray(A, dtype=np.float64).reshape(-1))


def global_match(kp_desc, map_desc, norm, bound, ratio, chunk=256):
    """Stage 04m spec D4: for each keypoint j in index order, the map row with the smallest
    descriptor distance (ties to the lower row) is kept when it is <= bound and, with a second row
    available, best < ratio * second. One row per keypoint and one keypoint per row: the smaller
    distance wins, then the lower keypoint index. Distances are computed in chunks of keypoints.
    Returns [(j, row, dist)] sorted by keypoint index."""
    n = 0 if kp_desc is None else len(kp_desc)
    m = 0 if map_desc is None else len(map_desc)
    if n == 0 or m == 0:
        return []
    props = []
    for s in range(0, n, chunk):
        D = _desc_dist_matrix(kp_desc[s:s + chunk], map_desc, norm)
        rb = np.argmin(D, axis=1)                        # first minimum: the lower row
        best = D[np.arange(len(D)), rb]
        second = np.partition(D, 1, axis=1)[:, 1] if m >= 2 else np.full(len(D), np.inf)
        for q in range(len(D)):
            b = float(best[q])
            if not (b <= bound):
                continue
            if np.isfinite(second[q]) and not (b < ratio * float(second[q])):
                continue
            props.append((b, s + q, int(rb[q])))
    taken = {}
    for b, j, row in sorted(props):
        if row not in taken:
            taken[row] = (j, row, b)
    return sorted(taken.values())


# --------------------------------------------------------------------------- helpers
def _project(K, T, X):
    """(uv (n, 2), z (n,)) of world points X under cam_T_world T; uv is -1 where z <= 0."""
    Xc = X @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    uv = np.full((len(X), 2), -1.0)
    front = z > 0
    if np.any(front):
        p = Xc[front] @ np.asarray(K, dtype=np.float64).T
        uv[front] = p[:, :2] / p[:, 2:3]
    return uv, z


def _in_image(uv, z, w, h):
    return (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def _kp_uv(frame):
    kps = getattr(frame, "KeyPoints", None) or []
    if len(kps) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in kps], dtype=np.float64)


def _frame_desc(frame):
    d = getattr(frame, "Descriptors", None)
    if d is None:
        d = getattr(frame, "descriptors", None)
    return d


def _image_wh(frame):
    shp = getattr(frame, "img_shape", None)
    if shp is None:
        return 640, 480
    return int(shp[1]), int(shp[0])


def _search(proj, cand_desc, cand_ids, kp_uv, kp_desc, kp_sigma, radius, free, bound, norm, ratio=None,
            chunk=512):
    """Guided search: for each projected candidate (row), the keypoints j with free[j] and
    ||uv_j - proj|| <= radius * sigma_j are compared by descriptor; the best (lowest distance, ties
    to the lower keypoint index) is kept when it is <= bound and, with ratio set and a second
    keypoint in the window, best < ratio * second. One keypoint takes one point: the smaller
    distance wins, then the lower candidate id. Returns [(row, j, dist)] sorted by row."""
    m = len(proj)
    n = len(kp_uv)
    if m == 0 or n == 0:
        return []
    r2 = (float(radius) * kp_sigma) ** 2
    props = []
    for s in range(0, m, chunk):
        P = proj[s:s + chunk]
        d2 = ((P[:, None, :] - kp_uv[None, :, :]) ** 2).sum(-1)
        win = (d2 <= r2[None, :]) & free[None, :]
        cols = np.flatnonzero(win.any(0))
        if len(cols) == 0:
            continue
        rows = np.flatnonzero(win[:, cols].any(1))
        W = win[np.ix_(rows, cols)]
        D = _desc_dist_matrix(cand_desc[s + rows], kp_desc[cols], norm)
        D = np.where(W, D, np.inf)
        jb = np.argmin(D, axis=1)                       # first minimum: the lower keypoint index
        best = D[np.arange(len(rows)), jb]
        if ratio is not None:
            if D.shape[1] >= 2:
                second = np.partition(D, 1, axis=1)[:, 1]
            else:
                second = np.full(len(rows), np.inf)
        for q in range(len(rows)):
            b = float(best[q])
            if not (b <= bound):
                continue
            if ratio is not None and np.isfinite(second[q]) and not (b < ratio * float(second[q])):
                continue
            props.append((b, int(cand_ids[s + rows[q]]), s + int(rows[q]), int(cols[jb[q]])))
    taken = {}
    for b, cid, row, j in sorted(props):
        if j not in taken:
            taken[j] = (row, j, b)
    return sorted(taken.values())


# --------------------------------------------------------------------------- the tracker (B5)
class MapTracker:
    """The map-based tracker against a perfect map (spec B5, C, D, E)."""

    def __init__(self, m1map, mode, params=None):
        if mode not in ("m1a", "m1b"):
            raise ValueError(f"unknown M1 mode {mode!r}")
        self.m1map = m1map
        self.mode = mode
        self.params = params if params is not None else MapTrackerParams()
        self.use_homography = False     # set by the pipeline from the run's flag (spec C1)
        self.reloc = "none"             # stage 04m: set by main.run_pipeline from --map_reloc
        self.reloc_params = MapRelocParams()
        self.reloc_solver = "epnp"      # stage 04q: set by main.run_pipeline from --reloc_solver
        self.reloc_shadow = False       # stage 04q: set by main.run_pipeline from --reloc_shadow
        self.header = dict(m1map.header)
        self._p_by_kf = {}
        for rec in sorted(m1map.points, key=lambda r: r[0]):
            self._p_by_kf.setdefault(int(rec[1]), []).append(rec)
        self.n_corr = sum(len(v) for v in m1map.corr.values())
        # the released map: file pid -> Point, and flat arrays for the search
        self.points_by_pid = {}
        self._pid_of_row = []
        self._row_of_point_id = {}
        self._rows_pt = []
        self._X_list = []
        self._D_list = []
        self._ckf_list = []
        self._X = np.zeros((0, 3))
        self._D = None
        self._ckf = np.zeros((0, 3))
        self._dirty = False
        # stage 04-m0 (spec B1, B2): under --live_map on the pipeline sets live_map True and the
        # table is rebuilt from the pipeline's Map before every frame (sync_live); point id ->
        # the id of its reference frame, fixed the first time sync_live sees the point
        self.live_map = False
        self._live_ref = {}
        # stage 04-m1 (spec A3, C1): set by main.run_pipeline from --residual_test (off, log, on)
        self.residual_test = "off"

    # ---------------------------------------------------------------- header / setup
    def check_header(self, front_end, method, undistort, keyframe_interval):
        """Spec E: the file's front_end, undistort and keyframe_interval must equal the run's; a
        mismatch prints one [M1-ERROR] kind=header line per field. Returns True when they agree."""
        want = {"front_end": str(front_end), "method": str(method), "undistort": str(int(bool(undistort))),
                "keyframe_interval": str(int(keyframe_interval))}
        ok = True
        for key, val in want.items():
            got = self.header.get(key)
            if key == "method" and got is None:
                continue
            if got != val:
                ok = False
                print(f"  >> [M1-ERROR] frame=-1 kind=header detail={key}:file={got},run={val}")
        return ok

    def setup_line(self):
        kfs = self.header.get("n_keyframes")
        try:
            n_kf = int(kfs)
        except (TypeError, ValueError):
            n_kf = len(self._p_by_kf)
        try:
            s = float(self.header.get("scale_s", "nan"))
        except ValueError:
            s = float("nan")
        try:
            interval = int(self.header.get("keyframe_interval", -1))
        except ValueError:
            interval = -1
        return (f"  >> [M1-SETUP] mode={self.mode} keyframes={n_kf} points={len(self.m1map.points)} "
                f"corr={self.n_corr} front_end={self.header.get('front_end', 'unknown')} interval={interval} "
                f"scale_s={s:.12g}")

    # ---------------------------------------------------------------- the released map
    def _arrays(self):
        if self._dirty:
            self._X = np.array(self._X_list, dtype=np.float64).reshape(-1, 3)
            self._D = np.array(self._D_list) if self._D_list else None
            self._ckf = np.array(self._ckf_list, dtype=np.float64).reshape(-1, 3)
            self._dirty = False
        return self._X, self._D, self._ckf

    def release(self, k, frames, world_map):
        """Spec D: insert keyframe k's P lines as landmarks once its pose is final. Returns the
        number inserted (shadowed points included)."""
        recs = self._p_by_kf.get(int(k))
        if not recs:
            return 0
        fr = frames[k]
        kp_uv = _kp_uv(fr)
        desc = _frame_desc(fr)
        c_kf = np.asarray(fr.world_T_cam, dtype=np.float64)[:3, 3].copy()
        inserted = shadowed = refused = 0
        for pid, kf, kp, u, v, X, Y, Z in recs:
            if not (0 <= kp < len(kp_uv)) or max(abs(kp_uv[kp, 0] - u), abs(kp_uv[kp, 1] - v)) > PIXEL_TOL:
                have = f"{kp_uv[kp, 0]:.4f},{kp_uv[kp, 1]:.4f}" if 0 <= kp < len(kp_uv) else "none"
                print(f"  >> [M1-ERROR] frame={k} kind=pixel detail=P,pid={pid},kp={kp},file={u:.4f},{v:.4f},frame={have}")
                refused += 1
                continue
            d = np.asarray(desc[kp])
            pt = point_mod.Point(np.array([X, Y, Z], dtype=np.float64), observations={fr.id: int(kp)}, descriptor=d)
            world_map.add_point(pt, frames)
            if pt.observations.get(fr.id) != int(kp):
                shadowed += 1
            inserted += 1
            self.points_by_pid[int(pid)] = pt
            self._row_of_point_id[pt.id] = len(self._pid_of_row)
            self._pid_of_row.append(int(pid))
            self._rows_pt.append(pt)
            self._X_list.append(np.array([X, Y, Z], dtype=np.float64))
            self._D_list.append(np.array(d, copy=True))
            self._ckf_list.append(c_kf)
            self._dirty = True
        print(f"  >> [M1-RELEASE] frame={k} points={inserted} shadowed={shadowed} refused={refused}")
        return inserted

    def sync_live(self, frames, world_map):
        """Stage 04-m0 (spec B2): with live_map False, return 0 and change nothing. With True, rebuild
        the point table from world_map, in world_map.points order: every point with a descriptor, a
        finite position and at least one observation is one row (the Point, its id, a copy of its
        position and descriptor read now, and the camera centre, read now, of its reference frame).
        A point's reference frame is the first key of its observations the first time sync_live sees
        it, kept until the point leaves the map; a point whose reference frame is not among
        ``frames`` is skipped. points_by_pid is not touched. Returns the number of rows."""
        if not self.live_map:
            return 0
        by_id = {getattr(f, "id", None): f for f in frames}
        in_map = set()
        rows_pt, pid_of_row, row_of_point_id = [], [], {}
        X_list, D_list, ckf_list = [], [], []
        for pt in world_map.points:
            in_map.add(pt.id)
            obs = getattr(pt, "observations", None) or {}
            if pt.id not in self._live_ref and len(obs) > 0:
                self._live_ref[pt.id] = next(iter(obs))
            if getattr(pt, "descriptor", None) is None or len(obs) == 0:
                continue
            X = np.array(pt.point, dtype=np.float64).reshape(-1)
            if X.shape[0] != 3 or not np.all(np.isfinite(X)):
                continue
            ref = by_id.get(self._live_ref.get(pt.id))
            if ref is None:
                continue
            row_of_point_id[pt.id] = len(pid_of_row)
            pid_of_row.append(pt.id)
            rows_pt.append(pt)
            X_list.append(X.copy())
            D_list.append(np.array(pt.descriptor, copy=True))
            ckf_list.append(np.array(np.asarray(ref.world_T_cam, dtype=np.float64)[:3, 3], dtype=np.float64))
        for pid in [p for p in self._live_ref if p not in in_map]:
            del self._live_ref[pid]
        self._rows_pt = rows_pt
        self._pid_of_row = pid_of_row
        self._row_of_point_id = row_of_point_id
        self._X_list = X_list
        self._D_list = D_list
        self._ckf_list = ckf_list
        self._dirty = True
        return len(rows_pt)

    # ---------------------------------------------------------------- per frame
    def on_frame(self, k, frames, world_map):
        """Spec C: track frame k >= 1; returns "tracked" or "lost" and prints one [MAP-TRACK] line."""
        fr = frames[k]
        T_prev = np.asarray(frames[k - 1].cam_T_world, dtype=np.float64)
        T_pred = None
        pred_rot = "cv"
        # stage 04m spec D1: under --map_reloc pnp a frame whose predecessor is LOST is relocalized
        reloc_on = getattr(self, "reloc", "none") == "pnp"
        relocalizing = reloc_on and k >= 2 and getattr(frames[k - 1], "tracking_state", None) == "LOST"
        rinfo = dict(matches=0, ransac_inliers=-1, reason="error", attempt=None)
        try:
            if reloc_on:
                # spec D2: the constant velocity restarts from two tracked frames after a loss
                use2 = k >= 2 and (k - 2 == 0 or getattr(frames[k - 2], "tracking_state", None) == "OK")
                T_prev2 = np.asarray(frames[k - 2].cam_T_world, dtype=np.float64) if use2 else None
            else:
                T_prev2 = np.asarray(frames[k - 2].cam_T_world, dtype=np.float64) if k >= 2 else None
            motion = tracking.estimate_motion(frames[k - 1], fr, use_homography=bool(self.use_homography))
            R_tw = None if motion is None else np.asarray(motion.R, dtype=np.float64)
            pred_rot = "twoview" if R_tw is not None else "cv"
            T_pred = predict_pose(T_prev, T_prev2, R_tw)
            if relocalizing:
                res = self._relocalize(k, frames, rinfo)
            elif self.mode == "m1a":
                res = self._track_m1a(k, frames, T_pred)
            else:
                res = self._track_m1b(k, frames, T_pred)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
            print(f"  >> [MAP-TRACK-ERROR] frame={k} error={msg}")
            T_fall = T_pred if T_pred is not None else T_prev
            self._set_lost(fr, T_fall)
            if relocalizing:
                # stage 04q spec D: the attempt with what is known, before the [MAP-RELOC] line
                att = rinfo["attempt"] if rinfo["attempt"] is not None else self._new_attempt(k, 0, self.reloc_solver)
                att.update(outcome="error", matches=rinfo["matches"], ransac_inliers=rinfo["ransac_inliers"],
                           inliers=0, pose=None)
                self._print_attempt(att, with_set=True)
                print(self._reloc_line(k, "fail", "error", rinfo["matches"], rinfo["ransac_inliers"], 0))
                print(self._track_line(k, "lost", pred_rot, -1, -1, 0, -1 if self.mode == "m1b" else 0, 0, 0,
                                       float("nan")))
                return "lost"
            print(self._track_line(k, "lost", pred_rot, -1 if self.mode == "m1b" else 0, -1 if self.mode == "m1b" else 0,
                                   0, -1 if self.mode == "m1b" else 0, 0, 0, float("nan")))
            return "lost"

        T_fin, obs_rows, obs_kps, mask, counts = res
        n_inl = int(np.sum(mask)) if len(mask) else 0
        rmse = self._rmse(fr, T_fin, obs_rows, obs_kps, mask) if T_fin is not None else float("nan")
        # stage 04-m1 (spec C3 (a)): the count's acceptance, then the residual test on it (off: always True)
        count_ok = T_fin is not None and n_inl >= int(self.params.min_inliers)
        accept = count_ok and self._resid_accept(k, "reloc" if relocalizing else "track", fr, T_fin, obs_rows,
                                                 obs_kps, mask)
        if relocalizing:
            ok = accept
            if ok:
                reason = "none"
            elif count_ok:
                reason = "residual"
            else:
                reason = rinfo["reason"] if rinfo["reason"] != "error" else "inliers"
            att = rinfo["attempt"]
            if count_ok and not ok:
                att["outcome"] = "residual"
            att.update(matches=rinfo["matches"], ransac_inliers=rinfo["ransac_inliers"])
            self._print_attempt(att, with_set=True)
            print(self._reloc_line(k, "ok" if ok else "fail", reason, rinfo["matches"], rinfo["ransac_inliers"],
                                   n_inl))
        if accept:
            fr.set_camera_pose(T_fin)
            fr.pose_valid = True
            fr.tracking_state = "OK"
            for row, kp, keep in zip(obs_rows, obs_kps, mask):
                if keep:
                    world_map.link(fr, int(kp), self._rows_pt[int(row)])
            state = "tracked"
        else:
            self._set_lost(fr, T_pred)
            state = "lost"
        print(self._track_line(k, state, pred_rot, counts["step_a"], counts["step_a_inliers"], counts["step_a_wide"],
                               counts["step_b"], len(obs_rows), n_inl, rmse))
        if state == "tracked" and reloc_on and getattr(self, "reloc_shadow", False):
            try:
                self._shadow(k, frames)            # stage 04q spec E: logged, never used
            except Exception:
                pass
        return state

    @staticmethod
    def _set_lost(fr, T):
        fr.set_camera_pose(T)
        fr.pose_valid = False
        fr.tracking_state = "LOST"

    def _track_line(self, k, state, pred_rot, step_a, step_a_inl, wide, step_b, obs, inl, rmse):
        return (f"  >> [MAP-TRACK] frame={k} state={state} mode={self.mode} pred_rot={pred_rot} "
                f"step_a={int(step_a)} step_a_inliers={int(step_a_inl)} step_a_wide={int(wide)} "
                f"step_b={int(step_b)} obs={int(obs)} inliers={int(inl)} rmse_px={float(rmse):.4f}")

    def _rmse(self, fr, T, rows, kps, mask):
        if len(rows) == 0 or not np.any(mask):
            return float("nan")
        X, _D, _c = self._arrays()
        rows = np.asarray(rows, dtype=np.int64)[mask]
        kps = np.asarray(kps, dtype=np.int64)[mask]
        uv, _z = _project(fr.K, T, X[rows])
        d = uv - _kp_uv(fr)[kps]
        return float(np.sqrt(np.mean((d * d).sum(1))))

    def _ba(self, fr, T0, rows, kps):
        X, _D, _c = self._arrays()
        rows = np.asarray(rows, dtype=np.int64)
        kps = np.asarray(kps, dtype=np.int64)
        sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
        uv = _kp_uv(fr)
        if len(rows) == 0:
            return motion_only_ba(fr.K, T0, np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0), self.params)
        return motion_only_ba(fr.K, T0, X[rows], uv[kps], sig[kps], self.params)

    def _local_map_search(self, fr, T0, excl_rows, free, radius):
        """The step-B guided search of spec C2 at pose T0: every released point outside excl_rows
        in front of the camera and inside the image, within max_view_angle_deg of its creating
        keyframe's viewing direction and within [dist_ratio_min, dist_ratio_max] of its distance,
        searched in a window of radius * sigma_j over the keypoints with free[j], ratio ratio_b.
        Returns [(row, j, dist)] (one keypoint per point)."""
        P = self.params
        X, D, CKF = self._arrays()
        kp_uv = _kp_uv(fr)
        kp_desc = _frame_desc(fr)
        kp_sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
        n_kp = len(kp_uv)
        w, h = _image_wh(fr)
        norm = _norm_type(fr)
        bound = float(PROTOCOL[getattr(fr, "method", "ORB")]["desc_bound"])
        b_matches = []
        if len(X) and n_kp:
            cand = np.ones(len(X), dtype=bool)
            if len(excl_rows):
                cand[np.asarray(excl_rows, dtype=np.int64)] = False
            uv_b, z_b = _project(fr.K, T0, X)
            cand &= _in_image(uv_b, z_b, w, h)
            c_k = np.linalg.inv(T0)[:3, 3]
            v1 = X - c_k
            v2 = X - CKF
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            good = (n1 > 0) & (n2 > 0)
            cosang = np.full(len(X), -1.0)
            cosang[good] = (v1[good] * v2[good]).sum(1) / (n1[good] * n2[good])
            cand &= good & (cosang >= math.cos(math.radians(float(P.max_view_angle_deg))))
            ratio_d = np.zeros(len(X))
            ratio_d[good] = n1[good] / n2[good]
            cand &= (ratio_d >= float(P.dist_ratio_min)) & (ratio_d <= float(P.dist_ratio_max))
            rb = np.flatnonzero(cand)
            if len(rb):
                ids = np.array([self._pid_of_row[r] for r in rb], dtype=np.int64)
                found = _search(uv_b[rb], D[rb], ids, kp_uv, kp_desc, kp_sig, radius, free, bound, norm,
                                ratio=float(P.ratio_b))
                b_matches = [(int(rb[row]), j, d) for row, j, d in found]
        return b_matches

    # ---------------------------------------------------------------- relocalization (04m spec D)
    def _reloc_line(self, k, status, reason, matches, ransac_inliers, inliers):
        return (f"  >> [MAP-RELOC] frame={k} mode={self.mode} status={status} reason={reason} matches={int(matches)} "
                f"ransac_inliers={int(ransac_inliers)} inliers={int(inliers)}")

    def _relocalize(self, k, frames, rinfo):
        """Spec D3 / D4 (04m), with the solver arm of 04q spec C4: relocalize frame k without a prior.
        Fills rinfo (matches, ransac_inliers, reason for a failure, the attempt record of spec D) and
        returns (T or None, rows, kps, mask, counts) of the last motion_only_ba; T is None when no pose
        was found."""
        fr = frames[k]
        RP = self.reloc_params
        X, D, _CKF = self._arrays()
        kp_uv = _kp_uv(fr)
        sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
        counts = dict(step_a=-1, step_a_inliers=-1, step_a_wide=0, step_b=(-1 if self.mode == "m1b" else 0))
        empty = (None, [], [], np.zeros(0, dtype=bool), counts)
        att = self._new_attempt(k, 0, self.reloc_solver)
        rinfo["attempt"] = att
        rows, kps, min_matches = self._reloc_matches(k, frames, X, D)
        rinfo["matches"] = len(rows)
        att["matches"] = len(rows)
        if len(rows) < min_matches:
            rinfo["reason"] = "too_few"
            att["outcome"] = "too_few"
            return empty
        r_arr = np.asarray(rows, dtype=np.int64)
        k_arr = np.asarray(kps, dtype=np.int64)
        T1, mask1, info = relocalize_with_solver(fr.K, X[r_arr], kp_uv[k_arr], sig[k_arr], self.params, RP,
                                                 self.reloc_solver)
        rinfo["ransac_inliers"] = int(info.get("ransac_inliers", -1))
        self._attempt_fill(att, info, X[r_arr], kp_uv[k_arr], fr)
        if info["status"] == "too_few":
            rinfo["reason"] = "too_few"
            att["outcome"] = "too_few"
            return empty
        if info["status"] != "ok":
            rinfo["reason"] = "pnp"
            att["outcome"] = info["status"] if info["status"] in ("ransac", "refit", "cheirality") else "ransac"
            return empty
        rinfo["reason"] = "inliers"
        T, rows2, kps2, mask = self._reloc_continue(fr, rows, kps, T1, np.asarray(mask1, dtype=bool), att, counts)
        return T, rows2, kps2, mask, counts

    def _reloc_matches(self, k, frames, X, D, quiet=False):
        """The correspondences fed to RANSAC: M1b the usable C lines, M1a the global descriptor match
        over the released map (04m spec D3 / D4). Returns (rows, kps, min_matches)."""
        fr = frames[k]
        RP = self.reloc_params
        if self.mode == "m1b":
            rows, kps = self._m1b_obs(k, frames, quiet=quiet)
            return rows, kps, 4
        norm = _norm_type(fr)
        bound = float(PROTOCOL[getattr(fr, "method", "ORB")]["desc_bound"])
        gm = global_match(_frame_desc(fr), D, norm, bound, float(RP.ratio_m1a))
        return [m[1] for m in gm], [m[0] for m in gm], max(4, int(RP.min_matches_m1a))

    def _reloc_continue(self, fr, rows, kps, T1, mask1, att, counts):
        """After the first motion-only BA of a relocalization: M1b ends there; M1a stops under
        ransac_min_inliers first-BA inliers, else runs the guided search at T_1 over the points other
        than its inliers and the final BA (04m spec D4). Completes the attempt record. Returns
        (T, rows, kps, mask) of the last BA."""
        RP = self.reloc_params
        n1 = int(mask1.sum())
        att["ba1_inliers"] = n1
        if self.mode == "m1b":
            att.update(inliers=n1, pose=T1, outcome="ok" if n1 >= int(self.params.min_inliers) else "inliers")
            return T1, rows, kps, mask1
        if n1 < int(RP.ransac_min_inliers):
            att.update(inliers=n1, outcome="ba_inliers")
            return T1, rows, kps, mask1
        # spec D4: the guided search at T_1 over the points other than its inliers, then the final BA
        kp_uv = _kp_uv(fr)
        inl_rows = [r for r, keep in zip(rows, mask1) if keep]
        inl_kps = [j for j, keep in zip(kps, mask1) if keep]
        free = np.ones(len(kp_uv), dtype=bool)
        for j in inl_kps:
            free[j] = False
        g_matches = self._local_map_search(fr, T1, inl_rows, free, float(RP.search_radius))
        counts["step_b"] = len(g_matches)
        att["guided"] = len(g_matches)
        rows2 = inl_rows + [m[0] for m in g_matches]
        kps2 = inl_kps + [m[1] for m in g_matches]
        T, mask, _info = self._ba(fr, T1, rows2, kps2)
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        att.update(inliers=n, pose=T, outcome="ok" if n >= int(self.params.min_inliers) else "inliers")
        return T, rows2, kps2, mask

    # ---------------------------------------------------------------- the attempt record (04q spec D, D2)
    def _new_attempt(self, k, shadow, solver, matches=0):
        a = dict(frame=int(k), shadow=int(shadow), solver=str(solver), outcome="error", matches=int(matches),
                 ransac_inliers=-1, behind=-1, ba1_inliers=-1, guided=-1, inliers=0, hull=-1.0, refit_ms=-1.0,
                 refit_pose=None, pose=None, set=None, refit_peak_b=-1)
        a.update(_npnp_record())
        return a

    @staticmethod
    def _copy_solve(att, info):
        att["behind"] = int(info.get("behind", -1))
        att["refit_pose"] = info.get("refit_pose")
        att["refit_ms"] = float(info.get("refit_ms", -1.0))
        for key, val in _npnp_record().items():
            att[key] = info.get(key, val)
        if "refit_peak_b" in info:
            att["refit_peak_b"] = int(info["refit_peak_b"])

    def _attempt_fill(self, att, info, Xin, UVin, fr):
        att["ransac_inliers"] = int(info.get("ransac_inliers", -1))
        self._copy_solve(att, info)
        cons = info.get("consensus")
        if cons is not None and np.any(cons):
            cons = np.asarray(cons, dtype=bool)
            att["set"] = (np.asarray(Xin, dtype=np.float64)[cons], np.asarray(UVin, dtype=np.float64)[cons])
            w, h = _image_wh(fr)
            att["hull"] = _hull_share(att["set"][1], w, h)

    def _set_line(self, att):
        P, UV = att["set"]
        return (f"  >> [RELOC-SET] frame={att['frame']} mode={self.mode} shadow={att['shadow']} n={len(P)} "
                f"points={_fmt_floats(P)} uv={_fmt_floats(UV)}")

    @staticmethod
    def _solve_line(att):
        return (f"  >> [RELOC-SOLVE] frame={att['frame']} shadow={att['shadow']} solver={att['solver']} "
                f"npnp_gamma={float(att['npnp_gamma'])!r} npnp_cost={float(att['npnp_cost'])!r} "
                f"npnp_outer={int(att['npnp_outer'])} npnp_newton={int(att['npnp_newton'])} "
                f"npnp_ls_evals={int(att['npnp_ls_evals'])} refit_peak_b={int(att['refit_peak_b'])}")

    def _attempt_line(self, att):
        return (f"  >> [MAP-RELOC-ATTEMPT] frame={att['frame']} mode={self.mode} shadow={att['shadow']} "
                f"solver={att['solver']} outcome={att['outcome']} matches={int(att['matches'])} "
                f"ransac_inliers={int(att['ransac_inliers'])} behind={int(att['behind'])} "
                f"ba1_inliers={int(att['ba1_inliers'])} guided={int(att['guided'])} inliers={int(att['inliers'])} "
                f"hull={float(att['hull'])!r} refit_ms={float(att['refit_ms'])!r} "
                f"npnp_converged={int(att['npnp_converged'])} npnp_gap_rel={float(att['npnp_gap_rel'])!r} "
                f"refit_pose={_fmt_pose(att['refit_pose'])} pose={_fmt_pose(att['pose'])}")

    def _print_attempt(self, att, with_set):
        """Spec D2 order: [RELOC-SET] (with a consensus set), [RELOC-SOLVE], [MAP-RELOC-ATTEMPT]."""
        if with_set and att["set"] is not None:
            print(self._set_line(att))
        print(self._solve_line(att))
        print(self._attempt_line(att))

    # ---------------------------------------------------------------- the shadow mode (04q spec E)
    def _shadow(self, k, frames):
        """Spec E1: the relocalization of the tracked frame k as if its predecessor were lost, once per
        solver (epnp, sqpnp, newton) from ONE matching and ONE RANSAC consensus set, printed with
        shadow=1 and never used: no pose, link, point or tracker state is written (E2)."""
        fr = frames[k]
        RP = self.reloc_params
        atts = [self._new_attempt(k, 1, s) for s in RELOC_SOLVERS]
        finals = [None] * len(atts)
        shared = None
        try:
            X, D, _CKF = self._arrays()
            kp_uv = _kp_uv(fr)
            sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
            rows, kps, min_matches = self._reloc_matches(k, frames, X, D, quiet=True)
            for a in atts:
                a["matches"] = len(rows)
            if len(rows) < min_matches:
                for a in atts:
                    a["outcome"] = "too_few"
            else:
                r_arr = np.asarray(rows, dtype=np.int64)
                k_arr = np.asarray(kps, dtype=np.int64)
                Xo, UVo, So = _as_obs(X[r_arr], kp_uv[k_arr], sig[k_arr])
                n_r, cons, T_r = _ransac_consensus(fr.K, Xo, UVo, RP)
                for a in atts:
                    a["ransac_inliers"] = n_r
                if T_r is None:
                    for a in atts:
                        a["outcome"] = "ransac"
                else:
                    shared = (Xo[cons], UVo[cons])
                    w, h = _image_wh(fr)
                    hull = _hull_share(shared[1], w, h)
                    for a in atts:
                        a.update(set=shared, hull=hull)
                    for i, a in enumerate(atts):
                        finals[i] = self._shadow_solve(fr, a, Xo, UVo, So, T_r, cons, rows, kps, k % 10 == 0)
        except Exception:
            for a in atts:
                if a["outcome"] not in ("too_few", "ransac"):
                    a.update(outcome="error", inliers=0, pose=None)
        if shared is not None:
            print(self._set_line(atts[0]))
        for a in atts:
            self._print_attempt(a, with_set=False)
        if getattr(self, "residual_test", "off") != "off":
            # stage 04-m1 (spec C3 (c)): the test on every ok shadow attempt, logged, never used
            for a, fin in zip(atts, finals):
                if a["outcome"] == "ok" and fin is not None:
                    self._resid_shadow(k, a, fr, *fin)

    def _shadow_solve(self, fr, a, Xo, UVo, So, T_r, cons, rows, kps, peak):
        """One solver of a shadow frame: C3 (3)-(5) from the shared consensus set, then (M1a) the
        guided search and the final BA; an exception gives outcome error (E3). Returns the final
        (T, rows, kps, mask) of the attempt's _reloc_continue, None when it did not get there."""
        try:
            info = _attempt_info(len(Xo))
            info.update(ransac_inliers=int(cons.sum()), consensus=cons)
            T1, mask1 = _solve_from_consensus(fr.K, Xo, UVo, So, self.params, T_r, cons, a["solver"], info,
                                              peak=peak)
            self._copy_solve(a, info)
            if info["status"] != "ok":
                a["outcome"] = info["status"]
                return
            scratch = dict(step_a=-1, step_a_inliers=-1, step_a_wide=0, step_b=0)
            return self._reloc_continue(fr, rows, kps, T1, np.asarray(mask1, dtype=bool), a, scratch)
        except Exception:
            a.update(outcome="error", inliers=0, pose=None)
        return None

    # ---------------------------------------------------------------- the residual test (04-m1 spec C)
    def _resid_eval(self, k, kind, fr, T, rows, kps, mask, shadow, solver, applied):
        """Spec C2 / C4: the test on the final inliers (the rows / kps whose mask is True, in the order of
        rows) with the arrays motion_only_ba used; prints the [RESID-TEST] and [RESID-SET] lines and
        returns evaluate's dict. Raises on any failure, before anything is printed."""
        X, _D, _c = self._arrays()
        m = np.asarray(mask, dtype=bool).reshape(-1)
        r = np.asarray(rows, dtype=np.int64).reshape(-1)
        j = np.asarray(kps, dtype=np.int64).reshape(-1)
        if not (len(m) == len(r) == len(j)):
            raise ValueError(f"rows ({len(r)}), kps ({len(j)}) and mask ({len(m)}) differ in length")
        uv = _kp_uv(fr)
        sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
        X_inl = X[r[m]].reshape(-1, 3)
        uv_inl = uv[j[m]].reshape(-1, 2)
        sig_inl = sig[j[m]].reshape(-1)
        e = motion_only_residuals(fr.K, T, X_inl, uv_inl, sig_inl).reshape(-1, 2)
        res = residual_test.evaluate((e * e).sum(1), self.params.chi2)
        w, h = _image_wh(fr)
        hull = _hull_share(uv_inl, w, h)
        n = int(res["n"])
        test = (f"  >> [RESID-TEST] frame={int(k)} mode={self.mode} kind={kind} shadow={int(shadow)} "
                f"solver={solver} n={n} stat={float(res['stat'])!r} crit={float(res['crit'])!r} "
                f"max_e2={float(res['max_e2'])!r} pass={int(bool(res['passed']))} applied={int(bool(applied))} "
                f"hull={float(hull)!r} pose={_fmt_pose(T)}")
        rset = (f"  >> [RESID-SET] frame={int(k)} kind={kind} shadow={int(shadow)} solver={solver} n={n} "
                f"points={_fmt_floats(X_inl)} uv={_fmt_floats(uv_inl)} sigma={_fmt_floats(sig_inl)}")
        print(test)
        print(rset)
        return res

    @staticmethod
    def _resid_error(k, kind, e):
        msg = f"{type(e).__name__}: {e}".replace("\r", " ").replace("\n", " ")
        print(f"  >> [RESID-ERROR] frame={int(k)} kind={kind} error={msg}")

    def _resid_accept(self, k, kind, fr, T, rows, kps, mask):
        """Stage 04-m1 spec C2, called only for a pose the count has accepted (kind track, reloc or exit).
        With the attribute residual_test off: True, nothing computed or printed. Otherwise the test is
        computed and logged (spec C4); returns True under log and the test's decision under on. A failure
        of the computation prints [RESID-ERROR] and returns True under log, False under on."""
        mode = getattr(self, "residual_test", "off")
        if mode == "off":
            return True
        solver = getattr(self, "reloc_solver", "epnp") if kind == "reloc" else "none"
        try:
            res = self._resid_eval(k, kind, fr, T, rows, kps, mask, 0, solver, mode == "on")
        except Exception as e:
            self._resid_error(k, kind, e)
            return mode != "on"
        if mode == "on":
            return bool(res["passed"])
        return True

    def _resid_shadow(self, k, att, fr, T, rows, kps, mask):
        """Stage 04-m1 spec C3 (c): the test on an ok 04q shadow attempt's final inliers, logged with kind
        shadow and applied 0; it never changes the attempt. A failure prints [RESID-ERROR] and nothing more."""
        try:
            self._resid_eval(k, "shadow", fr, T, rows, kps, mask, 1, str(att["solver"]), False)
        except Exception as e:
            self._resid_error(k, "shadow", e)

    def _m1b_obs(self, k, frames, quiet=False):
        """Spec C3: (rows, kps) of the file's C lines of frame k whose point is released and whose
        pixel matches the product keypoint; [M1-ERROR] lines for the others (none when quiet: the
        04q shadow re-reads a frame the tracker has read already)."""
        fr = frames[k]
        kp_uv = _kp_uv(fr)
        rows, kps = [], []
        for kp, pid, u, v in sorted(self.m1map.corr.get(int(k), [])):
            pt = self.points_by_pid.get(int(pid))
            if pt is None:
                if not quiet:
                    print(f"  >> [M1-ERROR] frame={k} kind=unreleased detail=C,kp={kp},pid={pid}")
                continue
            if not (0 <= kp < len(kp_uv)) or max(abs(kp_uv[kp, 0] - u), abs(kp_uv[kp, 1] - v)) > PIXEL_TOL:
                if not quiet:
                    have = f"{kp_uv[kp, 0]:.4f},{kp_uv[kp, 1]:.4f}" if 0 <= kp < len(kp_uv) else "none"
                    print(f"  >> [M1-ERROR] frame={k} kind=pixel detail=C,kp={kp},pid={pid},file={u:.4f},{v:.4f},"
                          f"frame={have}")
                continue
            rows.append(self._row_of_point_id[pt.id])
            kps.append(int(kp))
        return rows, kps

    def _track_m1b(self, k, frames, T_pred):
        """Spec C3: the observations are the file's C lines of frame k whose point is released."""
        fr = frames[k]
        rows, kps = self._m1b_obs(k, frames)
        T, mask, _info = self._ba(fr, T_pred, rows, kps)
        counts = dict(step_a=-1, step_a_inliers=-1, step_a_wide=0, step_b=-1)
        return T, rows, kps, np.asarray(mask, dtype=bool), counts

    def _track_m1a(self, k, frames, T_pred):
        """Spec C2: step A (the points of the last frame), then step B (the local map)."""
        P = self.params
        fr = frames[k]
        prev = frames[k - 1]
        X, D, CKF = self._arrays()
        kp_uv = _kp_uv(fr)
        kp_desc = _frame_desc(fr)
        kp_sig = np.asarray(_keypoint_sigmas(fr), dtype=np.float64)
        n_kp = len(kp_uv)
        w, h = _image_wh(fr)
        norm = _norm_type(fr)
        bound = float(PROTOCOL[getattr(fr, "method", "ORB")]["desc_bound"])
        free = np.ones(n_kp, dtype=bool)

        # ---- step A: points linked in frame k-1, projected with T_pred
        rows_a = []
        for i in sorted(prev.keypoint_to_point):
            row = self._row_of_point_id.get(prev.keypoint_to_point[i])
            if row is not None:
                rows_a.append(row)
        rows_a = np.array(sorted(set(rows_a)), dtype=np.int64)
        if len(rows_a):
            uv_a, z_a = _project(fr.K, T_pred, X[rows_a])
            vis = _in_image(uv_a, z_a, w, h)
        else:
            uv_a, vis = np.zeros((0, 2)), np.zeros(0, dtype=bool)
        ra = rows_a[vis]
        ids = np.array([self._pid_of_row[r] for r in ra], dtype=np.int64)
        desc_a = D[ra] if len(ra) else None
        matches = _search(uv_a[vis], desc_a, ids, kp_uv, kp_desc, kp_sig, P.radius_a, free, bound, norm)
        wide = 0
        if len(matches) < int(P.min_matches_a):
            wide = 1
            matches = _search(uv_a[vis], desc_a, ids, kp_uv, kp_desc, kp_sig, 2.0 * P.radius_a, free, bound, norm)
        matches = [(int(ra[row]), j, d) for row, j, d in matches]
        a_rows = [m[0] for m in matches]
        a_kps = [m[1] for m in matches]
        T_a, mask_a, _info_a = self._ba(fr, T_pred, a_rows, a_kps)
        mask_a = np.asarray(mask_a, dtype=bool)
        n_inl_a = int(mask_a.sum())
        success_a = n_inl_a >= int(P.min_inliers_a)

        # ---- step B: the local map from T_0
        if success_a:
            T0 = T_a
            radius = float(P.radius_b)
            inl_rows = [r for r, keep in zip(a_rows, mask_a) if keep]
            inl_kps = [j for j, keep in zip(a_kps, mask_a) if keep]
        else:
            # the widened search from the prediction stands in for the paper's reference-keyframe
            # search, which starts afresh: a failed step A carries no associations forward
            T0 = T_pred
            radius = float(P.radius_a)
            inl_rows, inl_kps = [], []
        for j in inl_kps:
            free[j] = False
        b_matches = self._local_map_search(fr, T0, inl_rows, free, radius)
        rows = list(inl_rows) + [m[0] for m in b_matches]
        kps = list(inl_kps) + [m[1] for m in b_matches]
        T, mask, _info = self._ba(fr, T0, rows, kps)
        counts = dict(step_a=len(matches), step_a_inliers=n_inl_a, step_a_wide=wide, step_b=len(b_matches))
        return T, rows, kps, np.asarray(mask, dtype=bool), counts
