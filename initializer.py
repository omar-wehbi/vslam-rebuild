"""
Two-view map initializer (stage 04s, milestone 04-3a): ORB-SLAM's H/F model selection with refusal
under ambiguity, followed by the design's publication rules. Not wired into the pipeline in this
stage; nothing in the product imports it.

Source and credit. The selection and the refusal are ORB-SLAM's (Mur-Artal, Montiel, Tardos 2015,
section IV; theory/papers/orb-slam.md#initialization-and-model-selection). The paper gives no
numbers, so the constants of InitParams are the ORB-SLAM2/3 implementation's, read from ORB-SLAM3
(Campos, Elvira, Gomez Rodriguez, Montiel, Tardos; GPLv3), src/TwoViewReconstruction.cc
(Reconstruct, FindHomography, FindFundamental, ComputeH21, ComputeF21, CheckHomography,
CheckFundamental, ReconstructF, ReconstructH, CheckRT, Normalize, DecomposeE),
src/GeometricTools.cc (Triangulate) and src/Tracking.cc (MonocularInitialization,
CreateInitialMapMonocular). This file is a numpy reimplementation of those routines, not a copy.
Four deliberate differences from that code:
  (1) R_H > 0.45, the paper's eq. 3 (ORB-SLAM2's code uses 0.40, ORB-SLAM3's 0.50);
  (2) per-keypoint sigma: every chi-square and reprojection test divides by the sigma of the
      keypoint in the image where the error is measured (local_mapping._keypoint_sigmas); with every
      sigma equal to 1 the computation is ORB-SLAM's;
  (3) the publication rules of initialize() steps E1-E8 (one match per feature, the project's
      per-point theta_min on the BA'd geometry, at least 50 points, spatial support over the view
      grid), which the design adds (P1);
  (4) an optional refit of the selected model on its inliers (refit_model, ruling R37-2), off by
      default: ORB-SLAM3 keeps the best-scoring 8-point model exactly as fitted
      (src/TwoViewReconstruction.cc l.206-228), and the paper describes no refit.

Convention (theory/conventions.md#pose-convention): camera a is the earlier frame and the world of
the reconstruction; x_b = R x_a + t, so cam_T_world of b is [R | t]. Pixels are undistorted
keypoint pixels; one K for both images.

Gates read at the estimate (conventions.md#map-ladder-conventions): every input of check_rt, the
selection of spec D8 and the re-check E4 is read at the estimate (the pixels are measured; R, t,
the points and the parallax come from the same two noisy views). The selection bias this admits is
registered and read by the harness; the gates stay as written.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import keypoints_match
import local_mapping
import optimization


@dataclass
class InitParams:
    iterations: int = 200            # ORB-SLAM TwoViewReconstruction mMaxIterations
    seed: int = 0                    # ORB-SLAM seeds its sampler with 0
    min_matches: int = 100           # ORB-SLAM3 MonocularInitialization: under 100 matches, no attempt
    chi2_h: float = 5.991            # chi-square, 2 dof, 95 %: the homography transfer test
    chi2_f: float = 3.841            # chi-square, 1 dof, 95 %: the epipolar distance test
    score_gamma: float = 5.991       # Gamma of the paper's eq. 2 (ORB-SLAM thScore)
    rh_select: float = 0.45          # the paper's eq. 3
    reproj_sq_sigmas: float = 4.0    # CheckRT th2 = 4 sigma^2, a 2-sigma radius
    infinite_cos: float = 0.99998    # CheckRT: parallax below arccos(0.99998) = 0.362 deg is "at infinity"
    min_triangulated: int = 50       # ORB-SLAM minTriangulated
    good_frac: float = 0.9           # the winner's good count must exceed 0.9 N
    f_similar_frac: float = 0.7      # ReconstructF: n_good > 0.7 maxGood counts as similar
    h_second_frac: float = 0.75      # ReconstructH: the second best must stay below 0.75 of the best
    min_parallax_deg: float = 1.0    # ORB-SLAM minParallax, on the 51st-largest parallax
    h_degenerate_ratio: float = 1.00001  # ReconstructH: d1/d2 or d2/d3 below this is degenerate
    theta_min_deg: float = 1.1       # the project's per-point minimum parallax (conventions.md#map-ladder-conventions)
    ba_iterations: int = 20          # ORB-SLAM GlobalBundleAdjustemnt(map, 20) on the initial map
    min_points: int = 50             # published points after the re-check (ORB-SLAM3 resets under 50 tracked points)
    grid_cols: int = 4               # the view grid of 04r's RotOnlyParams (Pirchheim's 4 x 3)
    grid_rows: int = 3
    min_rows: int = 2                # spatial support, E7
    min_cols: int = 2
    dup_tol_px: float = 0.001        # same-location duplicates (map_tracker.PIXEL_TOL)


@dataclass
class InitResult:
    status: str
    reason: str
    model: str
    rh: float
    score_h: float
    score_f: float
    n_matches: int
    n_inliers: int
    good: tuple
    chosen: int
    parallax51_deg: float
    n_triangulated: int
    R: object
    t: object
    points: np.ndarray
    match_idx: np.ndarray
    kp_a: np.ndarray
    kp_b: np.ndarray
    parallax_deg: np.ndarray
    rows: int
    cols: int
    cells: int
    ba_cost_initial: float
    ba_cost_final: float
    ba_iterations: int


REASONS = ("none", "matches", "scores", "degenerate", "few_good", "ambiguous", "parallax", "ba",
           "points", "support")


def _new_state():
    """The fields of an InitResult before any step has run (spec E, 'before D5' etc.)."""
    return dict(model="none", rh=math.nan, score_h=math.nan, score_f=math.nan, n_matches=0,
                n_inliers=0, good=(), chosen=-1, parallax51_deg=math.nan, n_triangulated=0,
                rows=0, cols=0, cells=0, ba_cost_initial=math.nan, ba_cost_final=math.nan,
                ba_iterations=-1)


def _refused(reason, state):
    """A refused result: R = t = None and empty arrays (spec E8)."""
    return InitResult(status="refused", reason=reason, R=None, t=None,
                      points=np.zeros((0, 3)), match_idx=np.zeros(0, dtype=np.int64),
                      kp_a=np.zeros(0, dtype=np.int64), kp_b=np.zeros(0, dtype=np.int64),
                      parallax_deg=np.zeros(0), **state)


# --------------------------------------------------------------------------- #
# D1-D5: normalization, sampling, models, scores, model selection
# --------------------------------------------------------------------------- #

def _normalize(uv):
    """ORB-SLAM Normalize: subtract the mean, divide each axis by its mean absolute deviation."""
    mean = uv.mean(axis=0)
    d = uv - mean
    s = 1.0 / np.mean(np.abs(d), axis=0)
    T = np.array([[s[0], 0.0, -mean[0] * s[0]],
                  [0.0, s[1], -mean[1] * s[1]],
                  [0.0, 0.0, 1.0]])
    return d * s, T


def score_homography(H21, uv_a, uv_b, sigma_a, sigma_b, params):
    """ORB-SLAM CheckHomography with per-keypoint sigma (spec D4). Returns (score, inlier mask)."""
    uv_a = np.asarray(uv_a, dtype=np.float64)
    uv_b = np.asarray(uv_b, dtype=np.float64)
    sigma_a = np.asarray(sigma_a, dtype=np.float64)
    sigma_b = np.asarray(sigma_b, dtype=np.float64)
    H21 = np.asarray(H21, dtype=np.float64)
    H12 = np.linalg.inv(H21)
    th = params.chi2_h
    score = 0.0
    errs = []
    # e_a: uv_b transferred into image a by H12; e_b: uv_a transferred into image b by H21.
    for Hm, src, dst, sg in ((H12, uv_b, uv_a, sigma_a), (H21, uv_a, uv_b, sigma_b)):
        ph = np.column_stack([src, np.ones(len(src))]) @ Hm.T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d2 = (dst[:, 0] - ph[:, 0] / ph[:, 2]) ** 2 + (dst[:, 1] - ph[:, 1] / ph[:, 2]) ** 2
        d2[(ph[:, 2] == 0) | ~np.isfinite(d2)] = np.inf
        e = d2 / sg ** 2
        score += float(np.sum(th - e[e <= th]))
        errs.append(e)
    return score, (errs[0] <= th) & (errs[1] <= th)


def score_fundamental(F21, uv_a, uv_b, sigma_a, sigma_b, params):
    """ORB-SLAM CheckFundamental with per-keypoint sigma (spec D4). Returns (score, inlier mask)."""
    uv_a = np.asarray(uv_a, dtype=np.float64)
    uv_b = np.asarray(uv_b, dtype=np.float64)
    sigma_a = np.asarray(sigma_a, dtype=np.float64)
    sigma_b = np.asarray(sigma_b, dtype=np.float64)
    F21 = np.asarray(F21, dtype=np.float64)
    th = params.chi2_f
    score = 0.0
    errs = []
    # e_b: uv_b against the line F21 [uv_a, 1]; e_a: uv_a against the line F21^T [uv_b, 1].
    for Fm, p, q, sg in ((F21, uv_a, uv_b, sigma_b), (F21.T, uv_b, uv_a, sigma_a)):
        ln = np.column_stack([p, np.ones(len(p))]) @ Fm.T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d2 = (ln[:, 0] * q[:, 0] + ln[:, 1] * q[:, 1] + ln[:, 2]) ** 2 / (ln[:, 0] ** 2 + ln[:, 1] ** 2)
        d2[~np.isfinite(d2)] = np.inf
        e = d2 / sg ** 2
        score += float(np.sum(params.score_gamma - e[e <= th]))
        errs.append(e)
    return score, (errs[0] <= th) & (errs[1] <= th)


def _find_models(uv_a, uv_b, sigma_a, sigma_b, params):
    """Spec D1-D5 up to the scores: the best H and F over the shared 8-point sets."""
    n = len(uv_a)
    pa, Ta = _normalize(uv_a)
    pb, Tb = _normalize(uv_b)
    rng = np.random.default_rng(params.seed)
    sets = np.array([rng.choice(n, 8, replace=False) for _ in range(params.iterations)])
    a, b = pa[sets], pb[sets]                               # (iterations, 8, 2)
    u1, v1, u2, v2 = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    z, o = np.zeros_like(u1), np.ones_like(u1)
    # ComputeH21: rows 2i and 2i+1 per correspondence; ComputeF21: one row per correspondence.
    r0 = np.stack([z, z, z, -u1, -v1, -o, v2 * u1, v2 * v1, v2], axis=-1)
    r1 = np.stack([u1, v1, o, z, z, z, -u2 * u1, -u2 * v1, -u2], axis=-1)
    AH = np.stack([r0, r1], axis=2).reshape(len(sets), 16, 9)
    AF = np.stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, o], axis=-1)
    Tb_inv = np.linalg.inv(Tb)
    best_h, best_f = 0.0, 0.0
    H_best = F_best = None
    mask_h = np.zeros(n, dtype=bool)
    mask_f = np.zeros(n, dtype=bool)
    for i in range(len(sets)):
        Hn = np.linalg.svd(AH[i], full_matrices=True)[2][-1].reshape(3, 3)
        H21 = Tb_inv @ Hn @ Ta
        if np.all(np.isfinite(H21)):
            try:
                sh, mh = score_homography(H21, uv_a, uv_b, sigma_a, sigma_b, params)
            except np.linalg.LinAlgError:
                sh = None                                   # singular H21: the set is skipped
            if sh is not None and sh > best_h:
                best_h, H_best, mask_h = sh, H21, mh
        Fn = np.linalg.svd(AF[i], full_matrices=True)[2][-1].reshape(3, 3)
        U, w, Vt = np.linalg.svd(Fn)
        F21 = Tb.T @ (U @ np.diag([w[0], w[1], 0.0]) @ Vt) @ Ta
        if np.all(np.isfinite(F21)):
            sf, mf = score_fundamental(F21, uv_a, uv_b, sigma_a, sigma_b, params)
            if sf > best_f:
                best_f, F_best, mask_f = sf, F21, mf
    return best_h, H_best, mask_h, best_f, F_best, mask_f


def refit_model(model, uv_a, uv_b, sigma_a, sigma_b, mask, params):
    """Spec C1 (ruling R37-2): a single refit of the selected model on its inliers, on the same
    normalization the 8-point sets used. Returns (M21, score, mask_refit); (None, nan, a copy of
    mask) when there are fewer than 8 inliers, the fitted model is not finite, or scoring it raises
    (a singular M21)."""
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    sigma_a = np.asarray(sigma_a, dtype=np.float64).reshape(-1)
    sigma_b = np.asarray(sigma_b, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    N0 = int(np.count_nonzero(mask))
    if N0 < 8:
        return None, math.nan, mask.copy()
    pa, Ta = _normalize(uv_a)
    pb, Tb = _normalize(uv_b)
    idx = np.flatnonzero(mask)                     # increasing index order
    a, b = pa[idx], pb[idx]
    u1, v1, u2, v2 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    z, o = np.zeros_like(u1), np.ones_like(u1)
    if model == "H":
        r0 = np.stack([z, z, z, -u1, -v1, -o, v2 * u1, v2 * v1, v2], axis=-1)
        r1 = np.stack([u1, v1, o, z, z, z, -u2 * u1, -u2 * v1, -u2], axis=-1)
        A = np.stack([r0, r1], axis=1).reshape(2 * N0, 9)
        Hn = np.linalg.svd(A, full_matrices=True)[2][-1].reshape(3, 3)
        M21 = np.linalg.inv(Tb) @ Hn @ Ta
    else:
        A = np.stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, o], axis=-1)
        Fn = np.linalg.svd(A, full_matrices=True)[2][-1].reshape(3, 3)
        U, w, Vt = np.linalg.svd(Fn)
        M21 = Tb.T @ (U @ np.diag([w[0], w[1], 0.0]) @ Vt) @ Ta
    if not np.all(np.isfinite(M21)):
        return None, math.nan, mask.copy()
    try:
        if model == "H":
            score, m = score_homography(M21, uv_a, uv_b, sigma_a, sigma_b, params)
        else:
            score, m = score_fundamental(M21, uv_a, uv_b, sigma_a, sigma_b, params)
    except np.linalg.LinAlgError:
        return None, math.nan, mask.copy()
    return M21, score, m


# --------------------------------------------------------------------------- #
# D6: motion hypotheses
# --------------------------------------------------------------------------- #

def essential_hypotheses(E):
    """ORB-SLAM DecomposeE: [(R1, t), (R2, t), (R1, -t), (R2, -t)]."""
    U, _, Vt = np.linalg.svd(np.asarray(E, dtype=np.float64))
    t = U[:, 2] / np.linalg.norm(U[:, 2])
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1
    R2 = U @ W.T @ Vt
    if np.linalg.det(R2) < 0:
        R2 = -R2
    return [(R1, t.copy()), (R2, t.copy()), (R1, -t), (R2, -t)]


def homography_hypotheses(H21, K):
    """Faugeras-Lustman decomposition as ORB-SLAM3 ReconstructH: eight (R, t), or [] when degenerate
    (d1/d2 or d2/d3 below InitParams.h_degenerate_ratio)."""
    return _decompose_homography(H21, K, InitParams().h_degenerate_ratio)


def _decompose_homography(H21, K, ratio):
    K = np.asarray(K, dtype=np.float64)
    A = np.linalg.inv(K) @ np.asarray(H21, dtype=np.float64) @ K
    if not np.all(np.isfinite(A)):
        return []
    U, w, Vt = np.linalg.svd(A)
    s = np.linalg.det(U) * np.linalg.det(Vt)
    d1, d2, d3 = float(w[0]), float(w[1]), float(w[2])
    if d2 <= 0.0:
        return []                                   # rank <= 1: no decomposition exists
    r23 = d2 / d3 if d3 > 0.0 else math.inf         # ORB-SLAM's float division gives inf here
    if d1 / d2 < ratio or r23 < ratio:
        return []
    aux1 = math.sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
    aux3 = math.sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3))
    x1 = (aux1, aux1, -aux1, -aux1)
    x3 = (aux3, -aux3, aux3, -aux3)
    out = []
    # case d' = +d2
    aux_st = math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2)
    ct = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2)
    st = (aux_st, -aux_st, -aux_st, aux_st)
    for i in range(4):
        Rp = np.array([[ct, 0.0, -st[i]], [0.0, 1.0, 0.0], [st[i], 0.0, ct]])
        R = s * U @ Rp @ Vt
        t = U @ (np.array([x1[i], 0.0, -x3[i]]) * (d1 - d3))
        out.append((R, t / np.linalg.norm(t)))
    # case d' = -d2
    aux_sp = math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2)
    cp = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2)
    sp = (aux_sp, -aux_sp, -aux_sp, aux_sp)
    for i in range(4):
        Rp = np.array([[cp, 0.0, sp[i]], [0.0, -1.0, 0.0], [sp[i], 0.0, -cp]])
        R = s * U @ Rp @ Vt
        t = U @ (np.array([x1[i], 0.0, x3[i]]) * (d1 + d3))
        out.append((R, t / np.linalg.norm(t)))
    return out


# --------------------------------------------------------------------------- #
# D7: CheckRT
# --------------------------------------------------------------------------- #

def _triangulate(uv_a, uv_b, P1, P2):
    """GeometricTools::Triangulate, batched: the last right singular vector of the 4x4 DLT."""
    A = np.empty((len(uv_a), 4, 4))
    A[:, 0] = uv_a[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = uv_a[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = uv_b[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = uv_b[:, 1:2] * P2[2] - P2[1]
    if len(A) == 0:
        return np.zeros((0, 3))
    h = np.linalg.svd(A)[2][:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :3] / h[:, 3:4]


def check_rt(R, t, uv_a, uv_b, sigma_a, sigma_b, K, mask, params):
    """ORB-SLAM CheckRT with per-keypoint sigma (spec D7).

    Returns (n_good, X, triangulated, cos_parallax, parallax51_deg): X (n, 3) in camera a with nan
    rows for matches that are not good; triangulated = good and cos_parallax < infinite_cos;
    cos_parallax (n,), nan where not computed."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    uv_a = np.asarray(uv_a, dtype=np.float64)
    uv_b = np.asarray(uv_b, dtype=np.float64)
    sigma_a = np.asarray(sigma_a, dtype=np.float64)
    sigma_b = np.asarray(sigma_b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = len(uv_a)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    O2 = -R.T @ t
    idx = np.flatnonzero(mask)
    X = np.full((n, 3), np.nan)
    cos_par = np.full(n, np.nan)
    good = np.zeros(n, dtype=bool)
    if len(idx):
        Xi = _triangulate(uv_a[idx], uv_b[idx], P1, P2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            finite = np.all(np.isfinite(Xi), axis=1)
            n1 = Xi
            n2 = Xi - O2
            cp = np.sum(n1 * n2, axis=1) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1))
            at_inf = cp >= params.infinite_cos
            Xb = Xi @ R.T + t
            front_a = (Xi[:, 2] > 0) | at_inf
            front_b = (Xb[:, 2] > 0) | at_inf
            e1 = ((fx * Xi[:, 0] / Xi[:, 2] + cx - uv_a[idx, 0]) ** 2
                  + (fy * Xi[:, 1] / Xi[:, 2] + cy - uv_a[idx, 1]) ** 2)
            e2 = ((fx * Xb[:, 0] / Xb[:, 2] + cx - uv_b[idx, 0]) ** 2
                  + (fy * Xb[:, 1] / Xb[:, 2] + cy - uv_b[idx, 1]) ** 2)
            ok_a = e1 <= params.reproj_sq_sigmas * sigma_a[idx] ** 2
            ok_b = e2 <= params.reproj_sq_sigmas * sigma_b[idx] ** 2
        g = finite & front_a & front_b & ok_a & ok_b
        good[idx] = g
        cos_par[idx] = np.where(finite, cp, np.nan)
        X[idx[g]] = Xi[g]
    n_good = int(np.count_nonzero(good))
    triangulated = good & (cos_par < params.infinite_cos)
    if n_good:
        cs = np.sort(cos_par[good])
        c = cs[min(50, n_good - 1)]
        parallax51 = math.degrees(math.acos(min(1.0, max(-1.0, float(c)))))
    else:
        parallax51 = 0.0
    return n_good, X, triangulated, cos_par, parallax51


# --------------------------------------------------------------------------- #
# E1 and E7 helpers
# --------------------------------------------------------------------------- #

def distinct_matches(uv_a, uv_b, desc_dist, kp_a, kp_b, params):
    """Spec E1: the indices of the kept matches, increasing. (i) one match per keypoint id on each
    side (smallest desc_dist, ties to the lower index; kp_a first, then kp_b); (ii) same-location
    duplicates, visited in order of (desc_dist, index), dropped when uv_a or uv_b lies within
    dup_tol_px (max-abs) of an already kept one."""
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    desc = np.asarray(desc_dist, dtype=np.float64).reshape(-1)
    n = len(uv_a)
    keep = np.ones(n, dtype=bool)
    for ids in (np.asarray(kp_a).reshape(-1), np.asarray(kp_b).reshape(-1)):
        vals, counts = np.unique(ids[keep], return_counts=True)
        for k in vals[counts > 1]:
            rows = np.flatnonzero(keep & (ids == k))
            win = rows[np.lexsort((rows, desc[rows]))[0]]
            keep[rows[rows != win]] = False
    tol = params.dup_tol_px
    kept_a = np.empty((n, 2))
    kept_b = np.empty((n, 2))
    nk = 0
    for i in np.lexsort((np.arange(n), desc)):
        if not keep[i]:
            continue
        if nk and (np.any(np.max(np.abs(kept_a[:nk] - uv_a[i]), axis=1) <= tol)
                   or np.any(np.max(np.abs(kept_b[:nk] - uv_b[i]), axis=1) <= tol)):
            keep[i] = False
            continue
        kept_a[nk] = uv_a[i]
        kept_b[nk] = uv_b[i]
        nk += 1
    return np.flatnonzero(keep)


def grid_support(uv, image_wh, params):
    """Spec E7: (rows, cols, cells) of the grid_cols x grid_rows view grid occupied by uv; a pixel
    outside the image counts in the edge cell."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) == 0:
        return 0, 0, 0
    w, h = float(image_wh[0]), float(image_wh[1])
    c = np.clip(np.floor(uv[:, 0] / (w / params.grid_cols)), 0, params.grid_cols - 1).astype(np.int64)
    r = np.clip(np.floor(uv[:, 1] / (h / params.grid_rows)), 0, params.grid_rows - 1).astype(np.int64)
    cells = len(np.unique(r * params.grid_cols + c))
    return int(len(np.unique(r))), int(len(np.unique(c))), int(cells)


def _ray_angle_deg(X, R, t):
    """Angle at X (camera a) between the rays from the centres 0 and -R^T t, in degrees."""
    c_b = -R.T @ t
    a, b = X, X - c_b
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))


# --------------------------------------------------------------------------- #
# initialize: D then E
# --------------------------------------------------------------------------- #

def initialize(uv_a, uv_b, sigma_a, sigma_b, desc_dist, K, image_wh, params=None, kp_a=None, kp_b=None,
               refit=False):
    """Two-view initialization of the matches (uv_a[i], uv_b[i]) (spec D and E). Returns InitResult.

    refit (ruling R37-2, off by default): after model selection, refit_model() runs once on the
    selected model's inliers; when it returns a finite model, the model, mask and n_inliers become
    its result (score_h, score_f and rh, which record the selection, are unchanged)."""
    params = InitParams() if params is None else params
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    n_in = len(uv_a)
    sigma_a = np.asarray(sigma_a, dtype=np.float64).reshape(-1)
    sigma_b = np.asarray(sigma_b, dtype=np.float64).reshape(-1)
    desc_dist = np.asarray(desc_dist, dtype=np.float64).reshape(-1)
    kp_a = np.arange(n_in) if kp_a is None else np.asarray(kp_a).reshape(-1)
    kp_b = np.arange(n_in) if kp_b is None else np.asarray(kp_b).reshape(-1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    st = _new_state()

    # E1 distinctness, E2 the match count
    keep = distinct_matches(uv_a, uv_b, desc_dist, kp_a, kp_b, params)
    ua, ub, sa, sb = uv_a[keep], uv_b[keep], sigma_a[keep], sigma_b[keep]
    n = len(keep)
    st["n_matches"] = int(n)
    if n < params.min_matches:
        return _refused("matches", st)

    # D1-D5
    s_h, H21, mask_h, s_f, F21, mask_f = _find_models(ua, ub, sa, sb, params)
    st["score_h"], st["score_f"] = float(s_h), float(s_f)
    if s_h + s_f == 0:
        return _refused("scores", st)
    rh = s_h / (s_h + s_f)
    st["rh"] = float(rh)
    model = "H" if rh > params.rh_select else "F"
    st["model"] = model
    mask = mask_h if model == "H" else mask_f
    N = int(np.count_nonzero(mask))
    st["n_inliers"] = N

    # R: the refit switch (spec C2). score_h, score_f and rh record the selection and are untouched.
    if refit:
        M, _, m = refit_model(model, ua, ub, sa, sb, mask, params)
        if M is not None:
            if model == "H":
                H21 = M
            else:
                F21 = M
            mask = m
            N = int(np.count_nonzero(mask))
            st["n_inliers"] = N

    # D6-D8
    if model == "F":
        hyps = essential_hypotheses(K.T @ F21 @ K)
    else:
        hyps = _decompose_homography(H21, K, params.h_degenerate_ratio)
        if not hyps:
            return _refused("degenerate", st)
    checks = [check_rt(R, t, ua, ub, sa, sb, K, mask, params) for R, t in hyps]
    goods = [c[0] for c in checks]
    st["good"] = tuple(int(g) for g in goods)
    if model == "F":
        max_good = max(goods)
        chosen = goods.index(max_good)
        st["chosen"] = chosen
        st["parallax51_deg"] = float(checks[chosen][4])
        st["n_triangulated"] = int(np.count_nonzero(checks[chosen][2]))
        if max_good < max(int(params.good_frac * N), params.min_triangulated):
            return _refused("few_good", st)
        if sum(g > params.f_similar_frac * max_good for g in goods) > 1:
            return _refused("ambiguous", st)
        if not (checks[chosen][4] > params.min_parallax_deg):
            return _refused("parallax", st)
    else:
        best, second, chosen = 0, 0, -1
        for i, g in enumerate(goods):
            if g > best:
                second, best, chosen = best, g, i
            elif g > second:
                second = g
        st["chosen"] = chosen
        if chosen >= 0:
            st["parallax51_deg"] = float(checks[chosen][4])
            st["n_triangulated"] = int(np.count_nonzero(checks[chosen][2]))
        if not (best > params.min_triangulated and best > params.good_frac * N):
            return _refused("few_good", st)
        if not (second < params.h_second_frac * best):
            return _refused("ambiguous", st)
        if not (checks[chosen][4] >= params.min_parallax_deg):
            return _refused("parallax", st)
    R0, t0 = hyps[chosen]
    X0, tri = checks[chosen][1], checks[chosen][2]
    sel = np.flatnonzero(tri)                         # indices into the n kept matches

    # E3 full BA of the initial map, camera a fixed
    T_b = np.eye(4)
    T_b[:3, :3] = R0
    T_b[:3, 3] = t0
    obs = []
    for j, i in enumerate(sel):
        obs.append((0, j, ua[i, 0], ua[i, 1]))
        obs.append((1, j, ub[i, 0], ub[i, 1]))
    try:
        problem = optimization.BAProblem.from_arrays(K, np.stack([np.eye(4), T_b]), [True, False],
                                                     X0[sel].reshape(-1, 3), obs)
        solved, info = optimization.solve_ba(problem, method="lm", max_iterations=params.ba_iterations)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return _refused("ba", st)
    st["ba_cost_initial"] = float(info.get("cost_initial", math.nan))
    st["ba_cost_final"] = float(info.get("cost_final", math.nan))
    st["ba_iterations"] = int(info.get("iterations", -1))
    poses = solved.poses()
    P = solved.points()
    if (info.get("status") != "completed" or not math.isfinite(st["ba_cost_final"])
            or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(P))):
        return _refused("ba", st)
    R, t = poses[1, :3, :3].copy(), poses[1, :3, 3].copy()

    # E4 re-check on the BA'd geometry
    ua_s, ub_s, sa_s, sb_s = ua[sel], ub[sel], sa[sel], sb[sel]
    Pb = P @ R.T + t
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pa = P @ K.T
        pb = Pb @ K.T
        ea = np.sum((pa[:, :2] / pa[:, 2:3] - ua_s) ** 2, axis=1)
        eb = np.sum((pb[:, :2] / pb[:, 2:3] - ub_s) ** 2, axis=1)
        par = _ray_angle_deg(P, R, t)
        ok = ((P[:, 2] > 0) & (Pb[:, 2] > 0)
              & (ea <= params.reproj_sq_sigmas * sa_s ** 2)
              & (eb <= params.reproj_sq_sigmas * sb_s ** 2)
              & (par >= params.theta_min_deg))
    surv = np.flatnonzero(ok)

    # E5 scale: the survivors' median z_a is 1
    if len(surv) == 0:
        return _refused("points", st)
    m = float(np.median(P[surv, 2]))
    t = t / m
    pts = P[surv] / m
    # E6 the point count
    if len(surv) < params.min_points:
        return _refused("points", st)
    # E7 spatial support on the newer frame's pixels
    rows, cols, cells = grid_support(ub_s[surv], image_wh, params)
    st["rows"], st["cols"], st["cells"] = rows, cols, cells
    if rows < params.min_rows or cols < params.min_cols:
        return _refused("support", st)

    # E8 accepted
    midx = keep[sel[surv]].astype(np.int64)
    return InitResult(status="accepted", reason="none", R=R, t=t, points=pts, match_idx=midx,
                      kp_a=kp_a[midx].copy(), kp_b=kp_b[midx].copy(),
                      parallax_deg=par[surv].copy(), **st)


def initialize_frames(frame_a, frame_b, params=None, refit=False):
    """initialize() on the matches of two product frames: frame_a is the query of
    keypoints_match.match_frames; pixels from KeyPoints (undistorted at extraction under
    --undistort), sigmas from local_mapping._keypoint_sigmas, K from frame_b."""
    params = InitParams() if params is None else params
    matches = keypoints_match.match_frames(frame_a, frame_b, frame_b.method)
    if len(matches) < 8:
        st = _new_state()
        st["n_matches"] = int(len(matches))
        return _refused("matches", st)
    qi = np.array([mt.queryIdx for mt in matches], dtype=np.int64)
    ti = np.array([mt.trainIdx for mt in matches], dtype=np.int64)
    dist = np.array([mt.distance for mt in matches], dtype=np.float64)
    uv_a = np.array([frame_a.KeyPoints[i].pt for i in qi], dtype=np.float64).reshape(-1, 2)
    uv_b = np.array([frame_b.KeyPoints[i].pt for i in ti], dtype=np.float64).reshape(-1, 2)
    sig_a = local_mapping._keypoint_sigmas(frame_a)[qi]
    sig_b = local_mapping._keypoint_sigmas(frame_b)[ti]
    h, w = frame_b.img_shape[0], frame_b.img_shape[1]
    return initialize(uv_a, uv_b, sig_a, sig_b, dist, frame_b.K, (w, h), params=params, kp_a=qi, kp_b=ti,
                      refit=refit)


def _f6(x):
    return "%.6f" % float(x)


def format_init_line(a, b, result):
    """The one [INIT-PAIR] line of a pair (spec E); nothing prints it in this stage."""
    r = result
    good = "/".join(str(int(g)) for g in r.good) if len(r.good) else "-"
    m = len(r.points)
    med = float(np.median(r.parallax_deg)) if m else math.nan
    return (f"  >> [INIT-PAIR] a={a} b={b} status={r.status} reason={r.reason} model={r.model} "
            f"rh={_f6(r.rh)} matches={int(r.n_matches)} inliers={int(r.n_inliers)} good={good} "
            f"chosen={int(r.chosen)} parallax51_deg={_f6(r.parallax51_deg)} "
            f"triangulated={int(r.n_triangulated)} published={m} rows={int(r.rows)} cols={int(r.cols)} "
            f"median_parallax_deg={_f6(med)} ba_cost0={_f6(r.ba_cost_initial)} ba_cost={_f6(r.ba_cost_final)}")
