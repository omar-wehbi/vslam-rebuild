"""
Stage 04i: the R2 seed lifetime of the depth-filter arm (ruling R2, human,
2026-09-22, theory/workshop/19; design.md#04-1-after-04c-the-mechanism-and-
the-route item 3) as PURE FUNCTIONS of seeds, frames and parameters.  No Map,
no dataset, no image file, no truth file: this module classifies, tests and
orders seeds; the driver (depth_filter_arm.py, --df_lifetime r2) owns the live
list, the Map side and the log lines.  Imports: numpy, math, scipy.special
(betainc) and depth_filter (the verified 04f module and its private
vectorised helpers _band_segments, _project; the module is not edited).

THE STATE MACHINE IN WORDS.  A seed is a feature of one reference keyframe
with an inverse-range posterior N(mu, sigma2) x Beta(a, b) (depth_filter).
In every posed frame it is one of

  DORMANT   its +-2 sigma credible interval, projected into the frame, has no
            point inside the image (or lies behind the camera): the frame
            cannot say anything about it.  Nothing is counted against it.
  SKIPPED   the interval is in the image but the frame's predicted disparity
            is below epsilon_match, the matcher's own along-line error scale:
            a measurement here would be noise of the matcher, not of the
            geometry.  Nothing is counted against it.
  ELIGIBLE  the frame is informative: the seed attempts one match
            (depth_filter.update_seeds).  A match updates the posterior; a
            no-match is an OUTLIER OBSERVATION (b <- b + 1, Vogiatzis and
            Hernandez section 4.2: the mixture's uniform component absorbs the
            frame) and counts one eligible failure.

Retirement is decided only on eligible attempts, by two tests: HOPELESS (the
paper's failure criterion, P(pi < eta) >= conf under Beta(a, b): the inlier
ratio is almost surely tiny; catches seeds that were bad from the start) and
DEAD (descriptor death: the camera has travelled death_travel_ratio x z_init
since the seed last matched; catches a feature that stopped matching after
being good).  A seed is CONVERGED when the 04f promotion test holds
(depth_filter.promotable: (i) sigma_rel <= eps1, (ii) an inlier view at
theta_min read at the posterior point) AND its inlier measurements come from
at least N baselines the matcher can tell apart (predicted disparities of the
posterior point pairwise >= epsilon_match).  A converged seed is TENTATIVE: a
public handle (X, variance) the driver exposes, still filtered, not a map
landmark; the status follows the posterior.  A tentative seed is CONFIRMED
(a landmark is made) early, when theta_acc >= k theta_min (k = promote_factor:
1 is the R2 reference timing, confirm at convergence; 9.5 is the deferred
timing), or at the end of its observability: descriptor death, eviction by the
live-seed cap, the end of the recording.  A seed whose reference keypoint gets
linked meanwhile is retired (linked).  Nothing retires a seed by age.

CONSTANTS (DepthFilterArmParams of depth_filter_arm.py; every one a --df_
flag), with their derivation:

  match_error_px = 9.5      epsilon_match: 04d und_attr_o3plus_match_err_px_p50
      = 9.51 px (runs/04d attempt 1), the matcher's along-line error on this
      hall.  Skip threshold (B4) and distinctness gap (B6).
  death_travel_ratio = 0.3  04b measured 14 of 500 ORB keypoints still matching
      within Hamming 50 at 0.84 m of baseline; the hall's scene range is
      2.5-3 m (04g); 0.84 / 2.75 = 0.305, a ratio of two map-unit lengths
      (never a metric distance or a frame count).
  outlier_eta = 0.05, outlier_conf = 0.99   Vogiatzis and Hernandez section
      4.3, criterion 1: P(pi < 0.05) >= 0.99.
  distinct_baselines = 3    N: two measurements with along-line errors can
      always agree on some depth; a third at a baseline the matcher can tell
      apart is the first that can contradict them (the route's third view in
      filter form).  With epsilon_match = 9.5 px = 1.04 deg at f = 521 the
      disparity skip makes a seed's very first measurement theta_min-worth, so
      without N the relative test would pass on one or two measurements (the
      n_meas < 10 population measured at -0.38 in 04h).
  seed_cap = 25000          cost, not method: SVO's accurate setting retains
      50 keyframes; 04h seeded 59,355 / 127 = 467 seeds per keyframe;
      50 x 467 = 23,368, rounded to 25,000.
  bootstrap_window = 5      the bootstrap depth window of scene_range, decoupled
      from the lifetime (the value max_seed_keyframes gave it in 04h).

WHAT EACH TEST READS (conventions.md#map-ladder-conventions: "gates read at
the estimate select on the error"; every reading is stated so the selection
bias can be measured):

  credible_segments / dormancy (B1, B2)   pose-only: centres, rays, poses, K,
      z_init.  Read at the estimate: mu, sigma2 (the interval's ends).  A
      too-close posterior narrows the interval; dormancy never counts a
      failure, so this reading cannot retire a seed.
  disparity skip (B3, B4)   read at the FAR end of the credible interval, so
      that a too-close posterior cannot make a frame look informative.  For a
      fresh seed the far end is at infinity (disparity exactly zero), so
      while the far end is unbounded the skip reads the disparity at the
      seed's INITIAL mean 1 / z_init, the keyframe's scene range: a quantity
      that is NOT the seed's own estimate (estimate-independent).  The
      population measured under that reading is reported apart
      (first_at_mean).
  hopeless (B5)   estimate-independent of depth: a and b only.
  distinct baselines (B6)   read at the estimate (X(mu)), but a COUNT: the
      along-line error moves each disparity by ~epsilon, which cannot merge
      two baselines that differ by >= epsilon in expectation.
  converged (B7)   promotable's (i) and (ii) are read at the estimate (the 04h
      finding: a relative test passes too-close seeds), plus the count of B6.
  dead (B8)   pose-only: the camera's travel since the last match against
      z_init (the keyframe's scene range).
  eviction (B9)   dormancy time and time since the last match: pose-only.

REGISTERED PREDICTION of the remaining selection bias (design.md item 3, the
04i predictions): the signed median relative depth error at confirmation is
better than -0.10 (04h reference arm: -0.277 on the exported map).

Conventions (conventions.md#pose-convention): cam_T_world maps world to
camera, world_T_cam is its inverse, the camera centre is world_T_cam[:3, 3].
Pixels are in the frame's own convention (raw or undistorted, 04d).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc

import depth_filter as df
from depth_filter import _MIN_DEPTH, _band_segments, _project

STATE_DORMANT, STATE_SKIPPED, STATE_ELIGIBLE = 0, 1, 2


# --------------------------------------------------------------------------- #
# B1. the credible segments and the reference points, vectorised
# --------------------------------------------------------------------------- #
def _stack(seeds):
    n = len(seeds)
    centres = np.array([s.centre for s in seeds], dtype=np.float64).reshape(n, 3)
    rays = np.array([s.ray for s in seeds], dtype=np.float64).reshape(n, 3)
    mu = np.fromiter((s.mu for s in seeds), dtype=np.float64, count=n)
    sigma2 = np.fromiter((s.sigma2 for s in seeds), dtype=np.float64, count=n)
    z_init = np.fromiter((float(getattr(s, "z_init", 1.0 / s.mu)) for s in seeds), dtype=np.float64, count=n)
    return centres, rays, mu, sigma2, z_init


def credible_segments(seeds, frame, s_band, arrays=None):
    """
    For every seed of the list, in ``frame``: p_far, p_near, ok (exactly
    depth_filter._band_segments, the +-s_band sigma interval), p_inf = the projection
    of the seed's ray at infinity (rotation only), inf_ok (its camera-F depth > 1e-12),
    far_unbounded (mu - s_band sigma <= 0: the interval reaches infinity), p_mean0 =
    the projection of X(1 / z_init) (the seed's INITIAL mean; seed.z_init set by the
    driver at seeding) and mean0_ok.  Pose-only inputs: centres, rays, poses, K,
    z_init; read at the estimate: mu, sigma2.  ``arrays`` = (centres, rays, mu, sigma2,
    z_init) already stacked by the caller (the driver keeps them aligned with its live
    list) skips the per-seed gather.  Vectorised: no per-seed Python loop.
    """
    centres, rays, mu, sigma2, z_init = arrays if arrays is not None else _stack(seeds)
    n = len(mu)
    s = float(s_band)
    R_F = np.asarray(frame.cam_T_world[:3, :3], dtype=np.float64)
    t_F = np.asarray(frame.cam_T_world[:3, 3], dtype=np.float64)
    K = np.asarray(frame.K, dtype=np.float64)
    if n == 0:
        e2 = np.zeros((0, 2)); e1 = np.zeros(0, dtype=bool)
        return dict(p_far=e2, p_near=e2.copy(), ok=e1, p_inf=e2.copy(), inf_ok=e1.copy(), far_unbounded=e1.copy(),
                    p_mean0=e2.copy(), mean0_ok=e1.copy(), centres=centres, rays=rays, mu=mu, sigma2=sigma2, z_init=z_init)
    p_far, p_near, ok = _band_segments(centres, rays, mu, sigma2, frame, s)
    d_inf = rays @ R_F.T
    p_inf = _project(K, d_inf)
    inf_ok = (d_inf[:, 2] > _MIN_DEPTH) & np.all(np.isfinite(p_inf), axis=1)
    sig = np.sqrt(np.maximum(sigma2, 0.0))
    far_unbounded = (mu - s * sig) <= 0.0
    xc0 = (centres + rays * z_init[:, None]) @ R_F.T + t_F
    p_mean0 = _project(K, xc0)
    mean0_ok = (xc0[:, 2] > _MIN_DEPTH) & np.all(np.isfinite(p_mean0), axis=1)
    return dict(p_far=p_far, p_near=p_near, ok=ok, p_inf=p_inf, inf_ok=inf_ok, far_unbounded=far_unbounded,
                p_mean0=p_mean0, mean0_ok=mean0_ok, centres=centres, rays=rays, mu=mu, sigma2=sigma2, z_init=z_init)


# --------------------------------------------------------------------------- #
# B2. the dormancy test: does the segment touch the image?
# --------------------------------------------------------------------------- #
def segment_in_image(p_a, p_b, ok, w, h):
    """
    True where the segment from p_a to p_b (n, 2) has a point inside [0, w) x [0, h)
    (Liang-Barsky clipping in float); False where ``ok`` is False (a seed behind the
    camera is dormant too).  Pose-only inputs plus the interval's ends.
    """
    p_a = np.asarray(p_a, dtype=np.float64).reshape(-1, 2)
    p_b = np.asarray(p_b, dtype=np.float64).reshape(-1, 2)
    ok = np.asarray(ok, dtype=bool).reshape(-1)
    n = len(ok)
    if n == 0:
        return np.zeros(0, dtype=bool)
    good = ok & np.all(np.isfinite(p_a), axis=1) & np.all(np.isfinite(p_b), axis=1)
    xa = np.where(good, p_a[:, 0], 0.0); ya = np.where(good, p_a[:, 1], 0.0)
    xb = np.where(good, p_b[:, 0], 0.0); yb = np.where(good, p_b[:, 1], 0.0)
    dx = xb - xa
    dy = yb - ya
    x_max = np.nextafter(float(w), 0.0)          # the largest float below w: [0, w) in float
    y_max = np.nextafter(float(h), 0.0)
    t0 = np.zeros(n)
    t1 = np.ones(n)
    inside = good.copy()
    for p, q in ((-dx, xa - 0.0), (dx, x_max - xa), (-dy, ya - 0.0), (dy, y_max - ya)):
        parallel = p == 0.0
        inside &= ~(parallel & (q < 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(parallel, 0.0, q / np.where(parallel, 1.0, p))
        neg = p < 0.0
        pos = p > 0.0
        t0 = np.where(neg, np.maximum(t0, r), t0)
        t1 = np.where(pos, np.minimum(t1, r), t1)
    inside &= t0 <= t1
    return inside


# --------------------------------------------------------------------------- #
# B3. the predicted disparity
# --------------------------------------------------------------------------- #
def predicted_disparity(p_x, p_inf, inf_ok):
    """
    || p_x - p_inf || in pixels per row; +inf where ``inf_ok`` is False (the infinity
    point is behind the camera: the disparity is unbounded and the skip never fires).
    Rows whose p_x is not finite give nan.
    """
    p_x = np.asarray(p_x, dtype=np.float64).reshape(-1, 2)
    p_inf = np.asarray(p_inf, dtype=np.float64).reshape(-1, 2)
    inf_ok = np.asarray(inf_ok, dtype=bool).reshape(-1)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(p_x - np.where(inf_ok[:, None], p_inf, 0.0), axis=1)
    return np.where(inf_ok, d, np.inf)


# --------------------------------------------------------------------------- #
# B4. classification
# --------------------------------------------------------------------------- #
def classify(seeds, frame, params_arm, params_df, segments=None):
    """
    (state, d_skip, d_mean, first_at_mean) over the seed list: state 0 dormant, 1
    skipped, 2 eligible.  dormant when the credible segment (p_far, p_near) has no point
    in the image (B2); else d_skip = the predicted disparity at the far end when it is
    finite, else at the seed's INITIAL mean X(1 / z_init) with first_at_mean True (the
    contract's reading while the far end is unbounded, module docstring); skipped when
    d_skip < params_arm.match_error_px; eligible otherwise.  d_mean = the predicted
    disparity at the CURRENT posterior mean X(mu) (written to the trace as d_pred; no
    decision reads it).  ``segments`` = credible_segments(...) already computed.
    """
    seg = segments if segments is not None else credible_segments(seeds, frame, float(params_df.search_band_sigmas))
    n = len(seg["mu"])
    if n == 0:
        z = np.zeros(0)
        return np.zeros(0, dtype=np.int8), z, z.copy(), np.zeros(0, dtype=bool)
    h, w = frame.img_shape[:2]
    in_img = segment_in_image(seg["p_far"], seg["p_near"], seg["ok"], w, h)
    d_far = predicted_disparity(seg["p_far"], seg["p_inf"], seg["inf_ok"])
    d_m0 = predicted_disparity(seg["p_mean0"], seg["p_inf"], seg["inf_ok"])
    first_at_mean = seg["far_unbounded"].copy()
    d_skip = np.where(first_at_mean, d_m0, d_far)
    R_F = np.asarray(frame.cam_T_world[:3, :3], dtype=np.float64)
    t_F = np.asarray(frame.cam_T_world[:3, 3], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        X_mu = seg["centres"] + seg["rays"] / seg["mu"][:, None]
    xc = X_mu @ R_F.T + t_F
    p_mu = _project(np.asarray(frame.K, dtype=np.float64), np.where((xc[:, 2] > _MIN_DEPTH)[:, None], xc, np.nan))
    d_mean = predicted_disparity(p_mu, seg["p_inf"], seg["inf_ok"])
    eps = float(params_arm.match_error_px)
    with np.errstate(invalid="ignore"):
        skipped = d_skip < eps                 # nan (no reference point) is never skipped
    state = np.where(~in_img, STATE_DORMANT, np.where(skipped, STATE_SKIPPED, STATE_ELIGIBLE)).astype(np.int8)
    return state, d_skip, d_mean, first_at_mean


# --------------------------------------------------------------------------- #
# B5. the failure test (Vogiatzis and Hernandez section 4.3, criterion 1)
# --------------------------------------------------------------------------- #
def hopeless(a, b, eta, conf):
    """P(pi < eta) >= conf under Beta(a, b): betainc(a, b, eta) is the regularised incomplete
    beta I_eta(a, b) = P(Beta(a, b) <= eta).  Estimate-independent of depth (a, b only).
    With a = b = 10 at seeding it needs b in the hundreds (a = 10: b = 320 gives 0.969,
    the 0.99 crossing lies between b = 360 and 365, b = 400 gives 0.997)."""
    return bool(float(betainc(float(a), float(b), float(eta))) >= float(conf))


# --------------------------------------------------------------------------- #
# B6. distinct baselines
# --------------------------------------------------------------------------- #
def point_disparities(X, rays, R, t, K):
    """
    Per row p: the predicted disparity of the world point X[p] in the camera (R[p],
    t[p], K[p]) (cam_T_world rows): || pi(X) - pi(ray at infinity) || with rays[p] the
    seed's unit world ray; +inf where the infinity point is behind that camera (z <=
    1e-12); nan where X itself is behind it.  Read at the estimate (X = X(mu)).
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(-1, 3, 3)
    P = len(X)
    if P == 0:
        return np.zeros(0)
    xc = np.einsum("pij,pj->pi", R, X) + t
    d_inf = np.einsum("pij,pj->pi", R, rays)
    uvw = np.einsum("pij,pj->pi", K, xc)
    uvw_inf = np.einsum("pij,pj->pi", K, d_inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_x = uvw[:, :2] / uvw[:, 2:3]
        p_inf = uvw_inf[:, :2] / uvw_inf[:, 2:3]
        d = np.linalg.norm(p_x - p_inf, axis=1)
    inf_ok = d_inf[:, 2] > _MIN_DEPTH
    x_ok = xc[:, 2] > _MIN_DEPTH
    return np.where(~inf_ok, np.inf, np.where(x_ok, d, np.nan))


def count_distinct(values, gap):
    """The size of the largest subset of ``values`` whose members differ pairwise by
    >= gap: sort ascending, count the first, then every value >= gap above the last
    counted (greedy, optimal on a line); +inf counts once; nan never counts."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.sort(v[np.isfinite(v)])
    n = 0
    last = -np.inf
    gap = float(gap)
    for x in finite.tolist():
        if x >= last + gap:
            n += 1
            last = x
    if np.any(np.isposinf(v)):
        n += 1
    return int(n)


def distinct_counts(disp, group, n_groups, gap):
    """count_distinct per group: ``disp`` (P,) disparities, ``group`` (P,) ints in
    [0, n_groups); returns an int array (n_groups,).  One sort, one Python pass."""
    disp = np.asarray(disp, dtype=np.float64).reshape(-1)
    group = np.asarray(group, dtype=np.int64).reshape(-1)
    out = np.zeros(int(n_groups), dtype=np.int64)
    if len(disp) == 0:
        return out
    finite = np.isfinite(disp)
    inf_groups = np.unique(group[np.isposinf(disp)])
    order = np.lexsort((disp[finite], group[finite]))
    g_s = group[finite][order].tolist()
    d_s = disp[finite][order].tolist()
    gap = float(gap)
    last_g = -1
    last = -np.inf
    for g, x in zip(g_s, d_s):
        if g != last_g:
            last_g = g
            last = -np.inf
        if x >= last + gap:
            out[g] += 1
            last = x
    out[inf_groups] += 1
    return out


def distinct_baselines(seed, params_df, params_arm):
    """
    (n_distinct, disparities): over the seed's INLIER measurements (resp >=
    responsibility_inlier, frame not None) the predicted disparity of the posterior
    point X(mu) in each measurement's frame (B3 with that frame's pose and K; +inf when
    its infinity point is behind that camera); n_distinct = count_distinct(disparities,
    match_error_px).  O(n_inlier).
    """
    inl = [m for m in seed.measurements if m.resp >= float(params_df.responsibility_inlier) and m.frame is not None]
    if not inl:
        return 0, np.zeros(0)
    X = df.seed_point(seed)
    R = np.array([m.frame.cam_T_world[:3, :3] for m in inl], dtype=np.float64)
    t = np.array([m.frame.cam_T_world[:3, 3] for m in inl], dtype=np.float64)
    K = np.array([m.frame.K for m in inl], dtype=np.float64)
    n = len(inl)
    disp = point_disparities(np.repeat(X[None, :], n, axis=0), np.repeat(seed.ray[None, :], n, axis=0), R, t, K)
    return count_distinct(disp, float(params_arm.match_error_px)), disp


# --------------------------------------------------------------------------- #
# B7. convergence
# --------------------------------------------------------------------------- #
def converged(seed, params_df, params_arm, n_distinct=None):
    """
    (ok, theta_acc_deg, theta_max_deg, n_distinct): ok = promotable(seed)[0] (the 04f
    test: (i) sigma_rel <= eps1, (ii) an inlier view at parallax >= theta_min sigma_L /
    sigma_0 read at the posterior point; both read at the estimate) AND n_distinct >=
    params_arm.distinct_baselines (B6).  ``n_distinct`` already computed (the driver's
    batched count) skips B6.
    """
    ok_p, theta_acc, theta_max = df.promotable(seed, params_df)
    if n_distinct is None:
        n_distinct, _ = distinct_baselines(seed, params_df, params_arm)
    n_distinct = int(n_distinct)
    return bool(ok_p and n_distinct >= int(params_arm.distinct_baselines)), float(theta_acc), float(theta_max), n_distinct


# --------------------------------------------------------------------------- #
# B8. descriptor death
# --------------------------------------------------------------------------- #
def travel_since_match(seed, C_F):
    """|| C_F - seed.last_match_centre || (map units)."""
    return float(np.linalg.norm(np.asarray(C_F, dtype=np.float64) - np.asarray(seed.last_match_centre, dtype=np.float64)))


def dead(seed, C_F, params_arm):
    """The camera has travelled >= death_travel_ratio * z_init (the seed's own scene range
    at seeding, map units) since the seed's last match.  Pose-only inputs.  Evaluated by
    the driver only on an eligible attempt that found no match."""
    return travel_since_match(seed, C_F) >= float(params_arm.death_travel_ratio) * float(seed.z_init)


# --------------------------------------------------------------------------- #
# B9. eviction order
# --------------------------------------------------------------------------- #
def eviction_order(seeds, k):
    """
    The seeds in eviction priority (D5): first dormant and not tentative, longest
    dormant first (smallest dormant_since); then active and not tentative, longest
    since their last match first (smallest last_match_traj, the seeding frame when
    none); then tentative seeds, dormant before active, the same tie rules.  Ties by
    seeding order (the list order).  ``k`` stands in for a missing time stamp.
    """
    k = int(k)

    def key(item):
        i, s = item
        tent = 1 if getattr(s, "tentative", False) else 0
        dormant = getattr(s, "state", "active") == "dormant"
        if dormant:
            since = getattr(s, "dormant_since", None)
            t = int(since) if since is not None else k
            return (tent, 0, t, i)
        last = getattr(s, "last_match_traj", None)
        t = int(last) if last is not None else k
        return (tent, 1, t, i)

    return [s for _, s in sorted(enumerate(seeds), key=key)]
