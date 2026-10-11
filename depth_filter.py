"""
Stage 04f: the SVO depth filter as a feature-based MODULE (the second arm of
the multi-baseline route of design.md#04-1-after-04c-the-mechanism-and-the-
route, papers/svo.md#depth-filter and #what-this-project-takes-from-it).
Not wired into the product: local_mapping.py, pipeline.py and main.py are
untouched; the wiring as the --point_source depth_filter arm is stage 04g.

A SEED lives on the ray of one keypoint i of a reference keyframe R: unit
world direction d = world_T_cam_R[:3, :3] @ normalize(inv(K_R) @ [u_i, v_i,
1]), centre C = world_T_cam_R[:3, 3].  The estimated quantity is the INVERSE
RANGE rho = 1 / r along that ray, X(rho) = C + d / rho (r is the distance
along the ray, not the camera-axis depth).  The state is the parametric
posterior of Vogiatzis and Hernandez (2011): a Gaussian N(mu, sigma2) on rho
times a Beta(a, b) on the inlier ratio; the measurement model is the mixture
pi * N(x; rho, tau2) + (1 - pi) * U(0, rho_range].  In every later posed
frame a seed takes at most ONE measurement: a descriptor match along the
epipolar segment inside the +-search_band_sigmas band of the current
estimate, triangulated against the reference ray, with the geometric
variance tau2 of SVO's computeTau, folded into the state by SVO's updateSeed
(update_one).  No per-measurement gate truncates the error distribution: a
wrong match is weighed by the mixture, not refused, so the 04c mechanism (a
gate admitting the along-line error on one sign only) cannot arise.

The module imports numpy, cv2, math, dataclasses and four helpers of
local_mapping (the keypoint pixel / level-sigma / norm-type / descriptor-
distance readers, so a frame is read exactly as the default arm reads it).
It never reads a dataset, an image file or any truth file (the 03c no-leak
rule), and it has NO Map argument anywhere: it creates, links and removes
nothing; the caller (04g) turns promoted seeds into landmarks and links
inlier_measurements as observations.

Constants (DepthFilterParams; every one becomes a --df_ flag in 04g;
provenance or regime derivation for each):

  min_parallax_deg = 1.1
      theta_min of conventions.md#map-ladder-conventions: no arm creates a
      finite landmark from views whose parallax is below it (the same
      constant as LocalMappingParams.min_parallax_deg: a 10 percent single-
      pixel range error at f = 521 px needs theta >= 1 / (521 * 0.10) rad).
      It enters the promotion rule (below) through BOTH conditions.
  init_depth_ratio_min = 0.1
      The prior on rho is uniform on (0, rho_range] with rho_range = 1 /
      (init_depth_ratio_min * z_init): the seed admits ranges from a tenth of
      its initial range (the keyframe's mean scene range, supplied by the
      caller) to infinity.  SVO seeds with z_range = 1 / depth_min and
      depth_min a fraction of the mean depth; a tenth is this contract's
      value (a room-sized scene at 2 to 4 m never puts a feature closer than
      0.2 m to the lens).  sigma2 = (rho_range / 6) ** 2 at seeding, SVO's
      Seed constructor (a 6-sigma prior spanning the range), a = b =
      beta_init.
  search_band_sigmas = 2.0
      The paper's +-2 sigma epipolar search band (the segment from X(mu -
      2 sigma) to X(mu + 2 sigma); the far end is the ray's point at infinity
      when mu - 2 sigma <= 0, as for every fresh seed).
  epipolar_chi2 = 3.84
      Band half-width test: squared point-to-segment distance <= 3.84 *
      sigma_ki ** 2 with sigma_ki the candidate keypoint's level sigma, the
      chi-square 1-dof 95 percent quantile local_mapping.match_pair applies
      to its epipolar band (ORB-SLAM's CheckDistEpipolarLine).
  max_desc_dist = 50.0
      ORB Hamming bound of a measurement (ORB-SLAM's TH_LOW = 50, the bound
      match_pair uses).
  beta_init = 10.0
      a = b = 10 at seeding (SVO's Seed constructor): the inlier ratio starts
      at 0.5 with the weight of 20 pseudo-observations, so a single wrong
      match moves it little.
  max_seed_keyframes = 5
      A seed whose reference keyframe is this many keyframes old is retired:
      the window size of LocalMappingParams.max_neighbours; beyond it the
      reference keyframe leaves the mapper's neighbourhood and the frames no
      longer overlap it usefully.
  responsibility_inlier = 0.5
      A measurement whose mixture responsibility C1 (the posterior
      probability that it came from the Gaussian) is >= 0.5 is an inlier
      measurement: it counts for the promotion test (ii) and 04g links it as
      an observation.  The natural threshold of a two-component mixture.

PROMOTION: THE theta_min STATEMENT (promotable).  The design asks that a
seed become a map point "when the variance implies >= theta_min of
accumulated parallax".  The reference point of the depth (a) bar
(design.md#depth-a-bar-derivation) is a two-view triangulation at theta_min
with sigma_0 = 1 px (level 0): its relative range error per image is
eps1 = sigma_0 / (f * theta_min) (= 0.0999 at f = 521 px), the two images
together giving the bar's m_fo(theta_min, sigma_0) = 0.095 median.  A
promoted seed's error has the same two parts, and the rule bounds EACH by
the single-image share eps1 of that reference point:

  (i)  the part that AVERAGES over measurements: each frame's own keypoint
       noise, which tau2 models and the posterior measures.  Condition:
       sigma_rel = sqrt(sigma2) / mu <= eps1.  Equivalently theta_acc :=
       sigma_0 / (fx_R * sigma_rel) >= theta_min: the parallax a single
       level-0 measurement would need to reach this posterior on its own
       ("the variance implies theta_min of accumulated parallax").
  (ii) the part that DOES NOT average: the reference keypoint's own
       along-line error, shared by every measurement of the seed, which no
       number of measurements reduces and whose relative size is sigma_R /
       (f * theta_max) with theta_max the widest parallax any INLIER
       measurement reached.  Condition: theta_max >= theta_min * sigma_R /
       sigma_0, so that sigma_R / (f * theta_max) <= eps1.  theta_max is
       read at the POSTERIOR point X(mu) between the reference centre C and
       the measurement's centre C_F (the angle between X - C and X - C_F, as
       local_mapping.triangulate_pair computes it), never from the matched
       pixel: a parallax read from the matched pixel is inflated by the very
       along-line error 04c found, and a test on it would select on the
       error exactly as the pair gate did; the posterior point and the two
       centres are pose quantities.

  Why (ii) scales with the seed's level: a level-3 seed (sigma_R = 1.2 ** 3 =
  1.73 px) needs 1.9 deg of true parallax for its shared error to be as
  small as a level-0 point's at 1.1 deg; promoting it at 1.1 deg would put
  its noise floor (m_fo(theta_min, sigma_3) = 0.165) above the 0.122 bar
  before any mismatch is counted.  Both conditions are regime quantities
  (sigma_0, f, theta_min, the level sigmas); nothing is tuned.  Promotion
  TIMING (every frame, at keyframes, at retirement) is the caller's policy,
  decided in 04g.

Conventions (conventions.md#pose-convention): cam_T_world maps world to
camera, world_T_cam is its inverse, the camera centre is world_T_cam[:3, 3].
Keypoint pixels are in the frame's own convention (raw or undistorted, 04d);
the geometry is pinhole with the frame's K either way
(conventions.md#camera-model-and-distortion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np

from epipolar_refine import REFINE_COUNTER_KEYS, STATUS_KEY, refine_along_epipolar
from local_mapping import _desc_dist_matrix, _keypoint_levels, _keypoint_sigmas, _kp_xy, _norm_type

SIGMA_0 = 1.0                 # px: the level-0 keypoint sigma the depth (a) bar was derived for
_ZERO_TRANSLATION = 1e-9      # ||t|| at or below this is "numerically zero" (spec C1(a), as local_mapping)
_MIN_DEPTH = 1e-12            # camera-axis depth at or below this is "behind" (spec C1(b))
_SINGULAR_DET = 1e-12         # |det| of the 2 x 2 normal equations at or below this: parallel rays
_SEED_CHUNK = 256             # seeds per chunk of the seeds x keypoints band test (memory bound)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class DepthFilterParams:
    """Constants of the depth filter; see the module docstring for each."""
    min_parallax_deg: float = 1.1
    init_depth_ratio_min: float = 0.1
    search_band_sigmas: float = 2.0
    epipolar_chi2: float = 3.84
    max_desc_dist: float = 50.0
    beta_init: float = 10.0
    max_seed_keyframes: int = 5
    responsibility_inlier: float = 0.5

    @classmethod
    def from_args(cls, args, prefix="df_"):
        """Build from an argparse namespace whose flags carry ``prefix`` (04g's main.py)."""
        kw = {}
        for f in fields(cls):
            v = getattr(args, prefix + f.name, None)
            if v is not None:
                kw[f.name] = v      # argparse already applied the field's type
        return cls(**kw)


class Measurement:
    """One update of a seed (spec B2): provenance, the measured inverse range, its variance,
    the mixture responsibility C1 of that update and the descriptor distance of the match."""

    def __init__(self, traj_idx, frame, kp, rho, tau2, resp, desc_dist):
        self.traj_idx = int(traj_idx)
        self.frame = frame
        self.kp = int(kp)
        self.rho = float(rho)
        self.tau2 = float(tau2)
        self.resp = float(resp)
        self.desc_dist = float(desc_dist)


class Seed:
    """The state of spec B2 (plain attributes)."""

    def __init__(self, ref, ref_kf_pos, kp, level, sigma_px, desc, centre, ray,
                 mu, sigma2, a, b, rho_range):
        self.ref = ref
        self.ref_kf_pos = int(ref_kf_pos)
        self.kp = int(kp)
        self.level = int(level)
        self.sigma_px = float(sigma_px)
        self.desc = desc
        self.centre = np.asarray(centre, dtype=np.float64)
        self.ray = np.asarray(ray, dtype=np.float64)
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.a = float(a)
        self.b = float(b)
        self.rho_range = float(rho_range)
        self.n_updates = 0
        self.measurements = []


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _unit_rays_cam(K, xy):
    """Unit rays in camera coordinates through the pixels ``xy`` (n, 2) of a camera with ``K``."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    h = np.column_stack([xy, np.ones(len(xy))]) @ Kinv.T
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def _translation_in_ref(ref, frame):
    """The position of ``frame``'s centre in ``ref``'s camera coordinates (spec C1(a))."""
    C_F = frame.world_T_cam[:3, 3]
    return ref.cam_T_world[:3, :3] @ C_F + ref.cam_T_world[:3, 3]


def _project(K, xc):
    """K applied to camera coordinates (n, 3), dehomogenised -> (n, 2); rows with z <= 0 are nan."""
    uvw = xc @ np.asarray(K, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        p = uvw[:, :2] / uvw[:, 2:3]
    return p


def seed_point(seed):
    """The posterior point X(mu) = centre + ray / mu (3,) in the world."""
    return seed.centre + seed.ray / float(seed.mu)


def inlier_ratio(seed):
    """The Beta mean a / (a + b): the seed's estimated inlier ratio."""
    return float(seed.a / (seed.a + seed.b))


def inlier_measurements(seed, params):
    """The measurements with resp >= responsibility_inlier, in order (04g links these)."""
    thr = float(params.responsibility_inlier)
    return [m for m in seed.measurements if m.resp >= thr]


# --------------------------------------------------------------------------- #
# B3. seeding
# --------------------------------------------------------------------------- #
def make_seeds(kf, kf_pos, z_init, params):
    """
    One seed per UNLINKED keypoint of ``kf`` (kp not in kf.keypoint_to_point), in
    keypoint order, at the caller's mean scene range ``z_init`` (map units):
    mu = 1 / z_init, rho_range = 1 / (init_depth_ratio_min * z_init), sigma2 =
    (rho_range / 6) ** 2, a = b = beta_init.  No measurements yet.
    """
    kps = getattr(kf, "KeyPoints", None) or []
    if len(kps) == 0:
        return []
    z_init = float(z_init)
    mu = 1.0 / z_init
    rho_range = 1.0 / (float(params.init_depth_ratio_min) * z_init)
    sigma2 = (rho_range / 6.0) ** 2
    beta = float(params.beta_init)
    xy = _kp_xy(kf)
    levels = _keypoint_levels(kf)
    sigmas = _keypoint_sigmas(kf)
    rays_cam = _unit_rays_cam(kf.K, xy)
    R_wc = kf.world_T_cam[:3, :3]
    rays = rays_cam @ R_wc.T
    centre = np.asarray(kf.world_T_cam[:3, 3], dtype=np.float64)
    linked = kf.keypoint_to_point
    descs = getattr(kf, "Descriptors", None)
    out = []
    for i in range(len(kps)):
        if i in linked:
            continue
        desc = np.array(descs[i], copy=True) if descs is not None else None
        out.append(Seed(kf, kf_pos, i, int(levels[i]), float(sigmas[i]), desc, centre.copy(),
                        rays[i].copy(), mu, sigma2, beta, beta, rho_range))
    return out


# --------------------------------------------------------------------------- #
# C1(b). the band segment
# --------------------------------------------------------------------------- #
def _band_segments(centres, rays, mu, sigma2, frame, s):
    """
    Vectorised spec C1(b) for seeds given as arrays: (p_far (n, 2), p_near (n, 2),
    ok (n,)).  rho_near = mu + s sigma, rho_far = mu - s sigma; p_near = pi_F(X(rho_near));
    p_far = pi_F(X(rho_far)) when rho_far > 0, else pi_F of the ray's direction in F
    (rotation only: the point at infinity).  ok is False when the camera-F depth of
    either endpoint (or the z of the infinity direction) is <= 1e-12.
    """
    R_F = np.asarray(frame.cam_T_world[:3, :3], dtype=np.float64)
    t_F = np.asarray(frame.cam_T_world[:3, 3], dtype=np.float64)
    sig = np.sqrt(np.maximum(sigma2, 0.0))
    rho_near = mu + s * sig
    rho_far = mu - s * sig
    far_finite = rho_far > 0.0
    near_ok = rho_near > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        X_near = centres + rays / np.where(near_ok, rho_near, 1.0)[:, None]
        X_far = centres + rays / np.where(far_finite, rho_far, 1.0)[:, None]
    xc_near = X_near @ R_F.T + t_F
    xc_far = np.where(far_finite[:, None], X_far @ R_F.T + t_F, rays @ R_F.T)
    z_near = xc_near[:, 2]
    z_far = xc_far[:, 2]
    ok = near_ok & np.isfinite(z_near) & np.isfinite(z_far) & (z_near > _MIN_DEPTH) & (z_far > _MIN_DEPTH)
    p_near = _project(frame.K, xc_near)
    p_far = _project(frame.K, xc_far)
    ok &= np.all(np.isfinite(p_near), axis=1) & np.all(np.isfinite(p_far), axis=1)
    return p_far, p_near, ok


def band_segment(seed, frame, params):
    """
    Spec C1(b) for one seed: (p_far (2,), p_near (2,), ok).  The segment in ``frame``
    between the projections of X(mu - s sigma) (the ray's point at infinity when that
    inverse range is not positive) and X(mu + s sigma), s = search_band_sigmas; ok is
    False when either endpoint has camera-F depth <= 1e-12.
    """
    p_far, p_near, ok = _band_segments(seed.centre[None, :], seed.ray[None, :],
                                       np.array([seed.mu]), np.array([seed.sigma2]),
                                       frame, float(params.search_band_sigmas))
    return p_far[0], p_near[0], bool(ok[0])


# --------------------------------------------------------------------------- #
# C1(e). the geometric variance (SVO's computeTau in the reference camera frame)
# --------------------------------------------------------------------------- #
def measurement_tau2(seed, frame, z_ref, px_noise):
    """
    tau2 of a measurement of ``seed`` taken in ``frame`` at the measured range
    ``z_ref`` along the seed's unit ray, for a pixel noise ``px_noise`` (the MATCHED
    keypoint's level sigma).  SVO's computeTau: the range at which the measurement
    ray perturbed by one pixel-noise angle meets the reference ray, z_plus, gives
    tau_z = z_plus - z_ref, converted to inverse range by the symmetric difference
    tau_rho = 0.5 * (1 / (z_ref - tau_z) - 1 / (z_ref + tau_z)).  Returns inf when the
    perturbed ray does not meet the reference ray in front of the camera (gamma_plus
    <= 0): the measurement then carries no range information.
    """
    t = _translation_in_ref(seed.ref, frame)
    f = np.asarray(seed.ref.cam_T_world[:3, :3], dtype=np.float64) @ seed.ray   # unit ray in R's camera
    nt = float(np.linalg.norm(t))
    if not (nt > 0.0):
        return float("inf")
    fx_F = float(np.asarray(frame.K, dtype=np.float64)[0, 0])
    px_error_angle = 2.0 * math.atan(float(px_noise) / (2.0 * fx_F))
    a_vec = f * float(z_ref) - t
    na = float(np.linalg.norm(a_vec))
    if not (na > 0.0):
        return float("inf")
    ca = float(np.dot(f, t) / nt)
    cb = float(-np.dot(a_vec, t) / (na * nt))
    if not (np.isfinite(ca) and np.isfinite(cb)):
        return float("inf")
    alpha = math.acos(max(-1.0, min(1.0, ca)))
    beta = math.acos(max(-1.0, min(1.0, cb)))
    beta_plus = beta + px_error_angle
    gamma_plus = math.pi - alpha - beta_plus
    if not (gamma_plus > 0.0):
        return float("inf")
    z_plus = nt * math.sin(beta_plus) / math.sin(gamma_plus)
    tau_z = z_plus - float(z_ref)
    tau_rho = 0.5 * (1.0 / max(float(z_ref) - tau_z, 1e-7) - 1.0 / (float(z_ref) + tau_z))
    return float(tau_rho * tau_rho)


# --------------------------------------------------------------------------- #
# C1(f). the update (Vogiatzis and Hernandez, as SVO's updateSeed)
# --------------------------------------------------------------------------- #
def update_one(seed, rho_meas, tau2):
    """
    Fold the measurement x = ``rho_meas`` with variance ``tau2`` into the seed's
    (mu, sigma2, a, b): the parametric approximation of the Gaussian + uniform
    mixture posterior, exactly SVO's updateSeed with U = 1 / rho_range the uniform
    density on (0, rho_range].  A measurement outside (0, rho_range] is an OUTLIER-
    ONLY update (C1 = 0, C2 = 1; spec C1(d)), as is one whose mixture weights both
    underflow.  Stores the state, increments n_updates, appends a Measurement (the
    provenance fields are placeholders that update_seeds fills in) with resp = C1,
    and returns C1.
    """
    x = float(rho_meas)
    tau2 = float(tau2)
    mu, s2, a, b = seed.mu, seed.sigma2, seed.a, seed.b
    rho_range = seed.rho_range
    in_support = bool(np.isfinite(x) and 0.0 < x <= rho_range)
    if in_support:
        if not (tau2 > 0.0):
            tau2 = 1e-300              # a zero variance would make the Gaussian a delta: 0/0 below
        norm2 = s2 + tau2
        s2_post = 1.0 / (1.0 / s2 + 1.0 / tau2)
        m = s2_post * (mu / s2 + x / tau2)
        pdf = math.exp(-0.5 * (x - mu) ** 2 / norm2) / math.sqrt(2.0 * math.pi * norm2)
        C1 = a / (a + b) * pdf
        C2 = b / (a + b) * (1.0 / rho_range)
        ssum = C1 + C2
        if ssum > 0.0 and math.isfinite(ssum):
            C1 /= ssum
            C2 /= ssum
        else:
            C1, C2 = 0.0, 1.0
    else:
        C1, C2 = 0.0, 1.0
        s2_post, m = s2, mu
    f = C1 * (a + 1.0) / (a + b + 1.0) + C2 * a / (a + b + 1.0)
    e = (C1 * (a + 1.0) * (a + 2.0) / ((a + b + 1.0) * (a + b + 2.0))
         + C2 * a * (a + 1.0) / ((a + b + 1.0) * (a + b + 2.0)))
    mu_new = C1 * m + C2 * mu
    # SVO writes C1 (s2_post + m^2) + C2 (s2 + mu^2) - mu_new^2; with C1 + C2 = 1 and
    # mu_new = C1 m + C2 mu this equals the law-of-total-variance form below, which
    # does not cancel two large squares (so the C1 = 0 update returns s2 bit for bit).
    s2_new = C1 * (s2_post + (m - mu_new) ** 2) + C2 * (s2 + (mu - mu_new) ** 2)
    s2_new = max(s2_new, (1e-6 * rho_range) ** 2)
    den = f - e / f
    if den != 0.0 and math.isfinite(den):
        a_new = (e - f) / den
        b_new = a_new * (1.0 - f) / f
    else:
        a_new, b_new = a, b            # degenerate (b = 0 only); keep the Beta as it is
    seed.mu = float(mu_new)
    seed.sigma2 = float(s2_new)
    seed.a = float(a_new)
    seed.b = float(b_new)
    seed.n_updates += 1
    seed.measurements.append(Measurement(-1, None, -1, x, tau2, C1, float("nan")))
    return float(C1)


# --------------------------------------------------------------------------- #
# C1. one measurement per seed in a posed frame
# --------------------------------------------------------------------------- #
COUNTER_KEYS = ("seeds", "measured", "no_match", "no_translation", "behind", "no_disparity",
                "outside_support", "inlier", "outlier")


def _pair_desc_dists(desc_s, desc_f, I, J, norm_type):
    """Descriptor distances of the pairs (I, J) with _desc_dist_matrix's semantics (Hamming
    for uint8 ORB by XOR + popcount, L2 otherwise), without the dense matrix."""
    if len(I) == 0:
        return np.zeros(0, dtype=np.float64)
    if norm_type == cv2.NORM_HAMMING:
        A = np.ascontiguousarray(desc_s, dtype=np.uint8)[I]
        B = np.ascontiguousarray(desc_f, dtype=np.uint8)[J]
        return _POPCOUNT[np.bitwise_xor(A, B)].sum(axis=1).astype(np.float64)
    A = np.asarray(desc_s, dtype=np.float64)[I]
    B = np.asarray(desc_f, dtype=np.float64)[J]
    return np.sqrt(np.maximum(((A - B) ** 2).sum(axis=1), 0.0))


def update_seeds(seeds, frame, traj_idx, params, refine=None):
    """
    At most ONE measurement per seed in ``frame`` (trajectory index ``traj_idx``),
    spec C1 (a) to (f), for every seed of the list in order.  The frame's keypoint
    pixels, level sigmas and unlinked index set are built once; the band test is a
    seeds x keypoints boolean (chunked over seeds); descriptor distances are computed
    on the in-band pairs only.  Returns the counters: seeds, measured (updates
    performed, outlier-only ones included), no_match, no_translation, behind,
    no_disparity, outside_support, inlier (resp >= responsibility_inlier), outlier;
    measured == inlier + outlier and seeds == measured + no_match + no_translation +
    behind + no_disparity.

    Stage 04k (epipolar_refine.py, spec S-e(ii)): with ``refine`` a RefineParams
    (--epi_refine) the returned dict also carries the REFINE_COUNTER_KEYS counters
    (zeros from the start), and each matched seed's keypoint is refined along the
    seed's band segment (direction p_near - p_far) against the seed's reference
    keypoint patch before step (d); a refused refinement removes the seed from the
    matched set (it counts in no_match), a refined one is triangulated from the
    refined pixel.  measurement_tau2 keeps the keypoint's sigma and Measurement.kp
    stays the keypoint index.  With refine None nothing changes.
    """
    c = {k: 0 for k in COUNTER_KEYS}
    if refine is not None:
        for key in REFINE_COUNTER_KEYS:
            c[key] = 0
    n = len(seeds)
    c["seeds"] = n
    if n == 0:
        return c
    s_band = float(params.search_band_sigmas)
    chi2 = float(params.epipolar_chi2)
    max_d = float(params.max_desc_dist)
    thr = float(params.responsibility_inlier)

    # the frame, read once
    xy = _kp_xy(frame)
    sig = _keypoint_sigmas(frame)
    linked = frame.keypoint_to_point
    unl = np.array([j for j in range(len(xy)) if j not in linked], dtype=int)
    desc_f = getattr(frame, "Descriptors", None)
    C_F = np.asarray(frame.world_T_cam[:3, 3], dtype=np.float64)
    R_wc_F = np.asarray(frame.world_T_cam[:3, :3], dtype=np.float64)

    # the seeds, as arrays
    centres = np.array([s.centre for s in seeds], dtype=np.float64)
    rays = np.array([s.ray for s in seeds], dtype=np.float64)
    mu = np.array([s.mu for s in seeds], dtype=np.float64)
    sigma2 = np.array([s.sigma2 for s in seeds], dtype=np.float64)

    # (a) relative geometry: ||t|| per seed (one computation per distinct reference frame)
    t_norm = np.empty(n, dtype=np.float64)
    cache = {}
    for k, s in enumerate(seeds):
        key = id(s.ref)
        if key not in cache:
            cache[key] = float(np.linalg.norm(_translation_in_ref(s.ref, frame)))
        t_norm[k] = cache[key]
    has_t = np.isfinite(t_norm) & (t_norm > _ZERO_TRANSLATION)
    c["no_translation"] = int(np.sum(~has_t))

    # (b) the band segments
    p_far, p_near, ok = _band_segments(centres, rays, mu, sigma2, frame, s_band)
    c["behind"] = int(np.sum(has_t & ~ok))
    active = has_t & ok
    if len(unl) == 0 or desc_f is None or not active.any():
        c["no_match"] = int(active.sum())
        return c

    # (c) candidates: unlinked keypoints within the chi-square band of the segment,
    #     tested for all (seed, keypoint) pairs, chunked over seeds
    xu = xy[unl, 0][None, :]
    yu = xy[unl, 1][None, :]
    sig2_u = (chi2 * sig[unl] ** 2)[None, :]
    act_idx = np.where(active)[0]
    v = p_near - p_far
    vv = (v * v).sum(axis=1)
    inv_vv = np.where(vv > 0.0, 1.0 / np.where(vv > 0.0, vv, 1.0), 0.0)     # 0 for a degenerate segment
    I_list, J_list = [], []
    for start in range(0, len(act_idx), _SEED_CHUNK):
        idx = act_idx[start:start + _SEED_CHUNK]
        vx = v[idx, 0][:, None]
        vy = v[idx, 1][:, None]
        wx = xu - p_far[idx, 0][:, None]                                     # (n, m) components of
        wy = yu - p_far[idx, 1][:, None]                                     # keypoint - p_far
        tpar = (wx * vx + wy * vy) * inv_vv[idx][:, None]                    # segment parameter,
        np.clip(tpar, 0.0, 1.0, out=tpar)                                    # clamped to [0, 1]
        wx -= tpar * vx
        wy -= tpar * vy
        band = (wx * wx + wy * wy) <= sig2_u
        ii, jj = np.nonzero(band)
        I_list.append(idx[ii])
        J_list.append(unl[jj])
    I = np.concatenate(I_list) if I_list else np.zeros(0, dtype=int)
    J = np.concatenate(J_list) if J_list else np.zeros(0, dtype=int)

    matched = {}
    if len(I):
        desc_s = np.array([seeds[k].desc for k in range(n)]) if all(s.desc is not None for s in seeds) else None
        if desc_s is not None:
            D = _pair_desc_dists(desc_s, desc_f, I, J, _norm_type(frame))
            order = np.lexsort((J, D, I))            # by seed, then distance, then keypoint index
            I_o, J_o, D_o = I[order], J[order], D[order]
            first = np.concatenate([[True], I_o[1:] != I_o[:-1]])
            for k, j, d in zip(I_o[first], J_o[first], D_o[first]):
                if d <= max_d:
                    matched[int(k)] = (int(j), float(d))
    # stage 04k: sub-pixel refinement of each match along its band segment (switch on only)
    refined_xy = None
    if refine is not None and matched:
        refined_xy = {}
        img_F, D_F = getattr(frame, "img", None), getattr(frame, "distortion", None)
        ref_xy = {}
        for k in sorted(matched):
            seed = seeds[k]
            j = matched[k][0]
            key = id(seed.ref)
            if key not in ref_xy:
                ref_xy[key] = _kp_xy(seed.ref)
            xy_new, status, _peak, _t = refine_along_epipolar(
                getattr(seed.ref, "img", None), ref_xy[key][seed.kp], seed.ref.K,
                getattr(seed.ref, "distortion", None), img_F, xy[j], frame.K, D_F,
                p_near[k] - p_far[k], refine)
            c["refine_attempted"] += 1
            c[STATUS_KEY[status]] += 1
            if status == "refined":
                refined_xy[k] = xy_new
            else:
                del matched[k]
    c["no_match"] = int(len(act_idx) - len(matched))
    if not matched:
        return c

    # (d) triangulation along the reference ray for the matched seeds, vectorised
    ks = np.array(sorted(matched), dtype=int)
    js = np.array([matched[k][0] for k in ks], dtype=int)
    if refined_xy is None:
        d_F = _unit_rays_cam(frame.K, xy[js]) @ R_wc_F.T                   # unit world rays of F
    else:
        d_F = _unit_rays_cam(frame.K, np.array([refined_xy[int(k)] for k in ks])) @ R_wc_F.T
    d = rays[ks]
    bvec = C_F[None, :] - centres[ks]
    cdot = (d * d_F).sum(axis=1)
    p = (d * bvec).sum(axis=1)
    q = (d_F * bvec).sum(axis=1)
    det = 1.0 - cdot * cdot
    for idx_m, k in enumerate(ks):
        seed = seeds[int(k)]
        j, dd = matched[int(k)]
        if not (np.isfinite(det[idx_m]) and abs(det[idx_m]) > _SINGULAR_DET):
            c["no_disparity"] += 1
            continue
        r_R = float((p[idx_m] - cdot[idx_m] * q[idx_m]) / det[idx_m])
        if r_R > 0.0:
            rho_meas = 1.0 / r_R
            # (e) the geometric variance at the measured range, the matched keypoint's sigma
            tau2 = measurement_tau2(seed, frame, r_R, float(sig[j]))
        else:
            rho_meas = 1.0 / r_R if r_R < 0.0 else float("-inf")
            tau2 = float("inf")
        outside = not (r_R > 0.0 and rho_meas <= seed.rho_range)
        if not outside and not np.isfinite(tau2):
            c["no_disparity"] += 1
            continue
        # (f) the update; an outside-support measurement is outlier-only inside update_one
        resp = update_one(seed, rho_meas, tau2)
        meas = seed.measurements[-1]
        meas.traj_idx = int(traj_idx)
        meas.frame = frame
        meas.kp = int(j)
        meas.desc_dist = float(dd)
        c["measured"] += 1
        if outside:
            c["outside_support"] += 1
        if resp >= thr:
            c["inlier"] += 1
        else:
            c["outlier"] += 1
    return c


# --------------------------------------------------------------------------- #
# D. promotion, retirement
# --------------------------------------------------------------------------- #
def promotable(seed, params):
    """
    The theta_min statement (module docstring): (ok, theta_acc_deg, theta_max_deg).
    (i) sqrt(sigma2) / mu <= sigma_0 / (fx_R * theta_min); (ii) the widest parallax at
    the posterior point X(mu) between the reference centre and an INLIER measurement's
    centre is >= theta_min * sigma_px / sigma_0.  theta_acc_deg is the parallax a single
    level-0 measurement would need for this posterior ((i) is theta_acc >= theta_min);
    theta_max_deg is -1 when there is no inlier measurement; mu <= 0 is not promotable.
    """
    theta_min = math.radians(float(params.min_parallax_deg))
    fx_R = float(np.asarray(seed.ref.K, dtype=np.float64)[0, 0])
    eps1 = SIGMA_0 / (fx_R * theta_min)
    mu = float(seed.mu)
    # (ii): the widest inlier parallax at the posterior point
    theta_max = -1.0
    if mu > 0.0 and np.isfinite(mu):
        X = seed_point(seed)
        r_R = X - seed.centre
        n_R = float(np.linalg.norm(r_R))
        for m in seed.measurements:
            if m.resp < float(params.responsibility_inlier) or m.frame is None:
                continue
            r_F = X - np.asarray(m.frame.world_T_cam[:3, 3], dtype=np.float64)
            n_F = float(np.linalg.norm(r_F))
            if n_R <= 0.0 or n_F <= 0.0:
                continue
            cos_a = float(np.dot(r_R, r_F) / (n_R * n_F))
            theta_max = max(theta_max, math.acos(max(-1.0, min(1.0, cos_a))))
    theta_max_deg = math.degrees(theta_max) if theta_max >= 0.0 else -1.0
    if not (mu > 0.0 and np.isfinite(mu)):
        return False, 0.0, theta_max_deg
    sigma_rel = math.sqrt(max(float(seed.sigma2), 0.0)) / mu
    cond_i = sigma_rel <= eps1
    cond_ii = theta_max >= 0.0 and theta_max >= theta_min * float(seed.sigma_px) / SIGMA_0
    theta_acc_deg = math.degrees(SIGMA_0 / (fx_R * sigma_rel)) if sigma_rel > 0.0 else float("inf")
    return bool(cond_i and cond_ii), float(theta_acc_deg), float(theta_max_deg)


def retire_seeds(seeds, current_kf_pos, params):
    """
    (kept, n_retired): drops the seeds whose reference keyframe is max_seed_keyframes
    or more keyframes old (ref_kf_pos <= current_kf_pos - max_seed_keyframes) and the
    seeds whose reference keypoint has become linked meanwhile; order preserved.
    """
    limit = int(current_kf_pos) - int(params.max_seed_keyframes)
    kept = []
    for s in seeds:
        if s.ref_kf_pos <= limit or s.kp in s.ref.keypoint_to_point:
            continue
        kept.append(s)
    return kept, len(seeds) - len(kept)
