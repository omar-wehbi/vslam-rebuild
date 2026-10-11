"""
Stage 04-1a: the default point-source arm of the slot table
(design.md#slots-default-arms-and-tested-alternatives, row "Point source"):
keyframe-pair triangulation between the new keyframe K_n and its covisibility
neighbours K_c with the four gates of papers/orb-slam.md section VI.C,
observation insertion into the neighbours, and observation-count culling
(section VI.B, second condition).  It is the ONLY creator of landmarks since
the per-frame chain-era triangulation was retired with this stage.

The module imports numpy, cv2, geometry, Map/Point helpers and
mapping.local_map_track only.  It never reads a dataset, an image file or any
truth file (the static no-leak rule of stage 03c applies to it).

Entry point (called by pipeline.py once per keyframe insertion, spec A6):

    process_new_keyframe(kf_idx, kf, keyframes, world_map, frames, params, stats)
        -> dict with the counters of the [LOCAL-MAP] line plus 'pairs'

Log lines (spec A7; the harness parses them, formats are exact):

    >> [LOCAL-MAP-PAIR] kf=<k> partner=<c> created=<p>
    >> [LOCAL-MAP-PAIR] kf=<k> partner=<c> created=0 reason=<baseline_ratio|zero_translation>
    >> [LOCAL-MAP] kf=<k> neighbours=<n> pairs_used=<u> pairs_skipped_baseline=<s>
       candidates=<c> created=<p> rej_cheirality=<a> rej_parallax=<b> rej_chi2=<d>
       rej_scale=<e> obs_inserted=<o> culled=<q>

and one diagnostic line (stage 04b spec E4; recorded, never a bar, the harness
ignores it), printed BEFORE the [LOCAL-MAP-PAIR] line of every pair that
reached triangulation (not skipped by A3(i) or by zero translation):

    >> [LOCAL-MAP-PARALLAX] kf=<k> partner=<c> n=<m> median_deg=<x.xxx> frac_over=<0.xxx> baseline=<b>

    m: candidates whose triangulation is finite with positive camera-axis
    depth in both cameras; median_deg: their median ray parallax (0.000 when
    m == 0); frac_over: the fraction of them at or above min_parallax_deg;
    baseline: || C_c - C_n || in map units (%.6g).

and one provenance line per landmark created by create_points (stage 04c spec
P1), printed after create_points returns for that pair and BEFORE the pair's
[LOCAL-MAP-PAIR] line:

    >> [LOCAL-MAP-POINT] id=<point_id> kf=<k> partner=<c> kp_n=<i> kp_c=<j>
       u_n=<%.2f> v_n=<%.2f> u_c=<%.2f> v_c=<%.2f> level_n=<o_n> level_c=<o_c>
       hamming=<d> hamming2=<d2> parallax_deg=<%.3f> z_n=<%.6g> z_c=<%.6g>
       baseline=<%.6g>

    point_id the created Point's id (map_export writes the same integer);
    kp_n, kp_c the keypoint indices in K_n and K_c the landmark is linked to;
    u_n v_n u_c v_c the pixel coordinates of those keypoints (KeyPoints[i].pt:
    raw image pixels with --undistort off, undistorted pixels with it on;
    the harness scores them under the matching --camera-model); level_n,
    level_c the pyramid octaves as _keypoint_sigmas decodes them; hamming the
    descriptor distance of the match (match_pair; integer for ORB, %.3g for
    SIFT/L2), hamming2 the second smallest finite in-band distance of kp_n's
    row in match_pair's masked matrix (-1 when there is no second candidate);
    parallax_deg the per-candidate ray parallax triangulate_pair returned;
    z_n, z_c the camera-axis depths of the point in K_n and K_c at creation
    time; baseline the pair baseline as on the [LOCAL-MAP-PARALLAX] line.
    This line is a diagnostic for the harness's creation attribution (task H,
    creation_attribution.py): it is recorded, never a bar, and changes
    nothing about what the mapper creates, culls or logs elsewhere.

BOOTSTRAP OF THE NEIGHBOURHOOD (stage 04b spec E1; amends A2).  The paper's
covisibility neighbourhood (orb-slam.md, sections III.D / VI.C) assumes an
initialized map, which its initializer supplies and which milestone 04-3 will
supply here.  With an empty map the rule of A2 yields the preceding keyframe
only, and under FORWARD motion one interval of baseline is below the parallax
gate for most keypoints, so nothing is created, covisibility never forms and
wider pairs are never tried (04a attempt 1: neighbours=1 on every [LOCAL-MAP]
line).  This module therefore FILLS the neighbourhood: after the covisibility
selection, the remaining slots up to max_neighbours are taken by the most
recent earlier keyframes not already chosen, newest first.  With an empty map
the neighbours are the last five keyframes; once landmarks exist, covisible
keyframes take the slots first and the fill supplies the rest.  Regime
derivation: a pair n keyframes apart offers a keypoint at angle alpha from
the translation direction a parallax of about (n * b_kf / z) * sin(alpha); at
z / b_kf of 25 to 35 (a forward-moving camera in a room-sized scene) the 1.1
deg gate needs n >= 0.019 * (z / b_kf) / sin(alpha), i.e. n of about 2 at
alpha = 20 deg and 3 to 4 at alpha = 10 deg, which a window of five reaches;
under lateral motion (sin(alpha) near 1) one interval suffices and the extra
pairs cost matching time only.  This fill is NOT the paper's rule: it is the
04b contract's bootstrap of the default arm, recorded as such.  No new
parameter: max_neighbours is the window size.  The pairs are processed in
DESCENDING order of baseline (spec E2; ties: the more recent partner first),
so a keypoint is triangulated from the widest available pair and the
narrower pairs find it linked (A3(v)).

PAIR-LEVEL PARALLAX GATE (stage 04b spec E3): the baseline / depth gate of
A3(i) measured on the pair's own candidates.  A3(i) still skips a pair
against the map's median depth before matching.  For a pair that reaches
matching and triangulation, BEFORE any per-point gate of A3(iv) is applied
or counted, the ray parallax angle (the quantity the A3(iv) parallax gate
tests) is computed over the candidates with finite triangulation and
positive camera-axis depth in BOTH cameras; if at least one exists and their
MEDIAN is below min_parallax_deg, the pair is parallax-starved and skipped
as a whole: no landmark, nothing added to candidates / rej_*, logged with
reason=baseline_ratio and counted in pairs_skipped_baseline exactly like an
A3(i) skip.  The reason string is deliberately the same: this IS the
baseline / depth gate (to first order the median parallax of the true
correspondences equals b / z_med), evaluated on the pair's candidates instead
of the map's median depth; the [LOCAL-MAP-PARALLAX] line tells the two forms
apart.  Why: a per-point parallax gate applied to a pair whose typical
candidate cannot pass it does not select true points; it selects the
candidates whose keypoint error (sigma_px * scale / f, 0.11 deg at level 0
and 0.4 deg at level 7) or mismatch inflated the parallax, and those sit at a
fraction of their true depth (04a attempt 1: the survivors of such pairs had
a median depth of 46 percent of truth; a point accepted at 1.1 deg whose true
parallax is 0.5 deg is placed at 0.5 / 1.1 of its depth).  A pair skipped by
the median is retried from the same keyframe against its wider partners (E1,
E2).  At a keyframe pair with a sub-millimetre true baseline the true
correspondences carry only noise parallax (their triangulated depth is about
f * b / sigma_px, so the parallax is about sigma_px * scale / f whatever b
is), so the median rejects the pair even when no landmark exists yet to give
A3(i) a median depth; this closes the bootstrap gap through which 04a
attempt 1 created two landmarks from pairs below the 1 mm stationary
threshold.  The threshold is the per-point gate's own constant,
min_parallax_deg: one regime quantity, no new parameter, no tuning.

Constants (every one is a CLI flag with the --lm_ prefix; regime derivations
or paper provenance as design.md#stage-04-what-map-based-tracking-means-here
requires -- regime quantities, never frame numbers of the primary sequence):

  min_shared_points = 15, max_neighbours = 5
      Neighbourhood: earlier keyframes sharing >= 15 landmarks with K_n, best
      five by count.  The paper's covisibility graph uses a 15-point edge
      (section III.C); five neighbours is the paper's local-BA neighbourhood
      size used by this repo's keyframe-local BA (keyframe_local_topk).  The
      immediately preceding keyframe is always a neighbour: under the fixed
      interval-20 policy it is the only guaranteed partner.  Slots left free
      by covisibility are filled with the most recent earlier keyframes
      (04b spec E1, the bootstrap paragraph above), so max_neighbours is also
      the size of the recent-keyframe window.
  min_baseline_depth_ratio = 0.01
      Baseline / median-depth gate (paper section VI.C: "discard if the ratio
      of baseline to median depth is too small").  At ratio 0.01 the parallax
      at the median depth is atan(0.01) = 0.57 deg, half the parallax gate, so
      such a pair can yield accepted points only in the near field; the
      paper's value is the starting point.
  epipolar_chi2 = 3.84
      Epipolar band: squared point-to-line distance <= 3.84 sigma^2, the
      chi-square 1-dof 95 percent quantile (ORB-SLAM's CheckDistEpipolarLine).
  epipole_radius_px = 10.0
      Candidates closer than 10 * sqrt(sigma) px to the epipole are excluded
      (ORB-SLAM's rule: squared distance < 100 * scale factor); near the
      epipole the band degenerates and every point is "on the line".
  max_desc_dist = 50
      ORB Hamming distance for a keyframe-pair match (ORB-SLAM's TH_LOW = 50).
  min_parallax_deg = 1.1
      Relative depth error ~ sigma_px / (f * theta) for small parallax theta,
      so a 10 percent single-pixel error at f ~ 521 px (DEFAULT_K) needs
      theta >= 1 / (521 * 0.10) rad = 1.1 deg.  The paper's cos < 0.9998
      (1.15 deg) is the starting point.  Regime note: the parallax a pair
      offers is theta ~ (b / z) * sin(alpha) with alpha the angle between the
      viewing ray and the translation direction, so under FORWARD motion
      (epipole inside the image) it is largest at the image border, sin(alpha)
      <= 0.5 at f = 521 px, and a pair one interval apart reaches the gate
      only where z / b < ~26; under lateral motion (sin(alpha) ~ 1) the same
      gate is met at z / b < 52.
  reproj_chi2 = 5.991
      Squared reprojection error in EACH image <= 5.991 sigma^2 (chi-square,
      2 dof, 95 percent), sigma = scale_factor ** octave of that image's
      keypoint (the paper's per-level covariance, Appendix eq. 5-7).
  scale_ratio_factor = 1.8
      Scale consistency: with r = d_c / d_n the distances from the point to the
      two camera centres, require (sigma_n / sigma_c) / factor <= r <=
      (sigma_n / sigma_c) * factor.  The paper's rule is factor = 1.5 *
      scale_factor = 1.5 * 1.2 = 1.8 (ORB-SLAM's ratioFactor); the value is
      recorded as such, not derived from a regime.
  insert_search_radius_px = 8.0
      Observation insertion window around a landmark's projection into a
      neighbour keyframe (the paper's "projected into other connected
      keyframes"; the pixel radius is this contract's choice: two to three
      chi-square radii at level 0, so the chi-square gate, not the window,
      decides).
  cull_after_keyframes = 3, min_keyframe_obs = 3
      Observation-count culling: a landmark created at a keyframe that is >= 3
      entries before K_n in the keyframe list and observed from fewer than 3
      keyframes is removed (paper section VI.B, second condition: observed
      from >= 3 keyframes).  The WAIT of 3 keyframes is a DEPARTURE from the
      paper's note, which judges a point once more than one keyframe has
      passed since its creation: this contract gives one extra keyframe of
      grace under the interval policy, so a landmark has had three insertion
      opportunities before it is judged.  The paper's first condition (found
      in > 25 percent of the frames where it is predicted visible) needs the
      04-2 tracker's visibility prediction and is NOT built here.
  confirm = False
      Stage 04e switch (the design's multi-baseline verification principle,
      design.md#04-1-after-04c-the-mechanism-and-the-route, first arm): a
      gate-accepted candidate of the pair (K_n, K_c) becomes a landmark only
      after confirm_candidates re-finds it, within the paper's per-level
      chi-square window (reproj_chi2 * sigma^2, the keypoint's own level), in
      at least one FURTHER neighbour keyframe whose translational disparity
      for that candidate differs from the creating pair's (see
      confirm_disparity_ratio); the confirming keypoint is linked as an
      observation.  Off in the reference configuration: with the switch off
      this module is byte-for-byte the 04d mapper (no new line, no new link).
  confirm_disparity_ratio = 0.3
      Eligibility of a third view, applied on BOTH sides of the disparity
      ratio r = d_j / d_pair: r >= ratio and |r - 1| >= ratio.  Regime
      derivation (04e contract): a wrong repeat at pitch p in the pair
      triangulates a point whose disparity is d + p instead of d; in a third
      view at disparity ratio r the wrong point predicts r * (d + p) from the
      infinity point while the true feature sits at r * d and the repeat at
      r * d + p, so the chi-square test refuses it only if p * |r - 1| >=
      3 sigma_L (distance to the repeat) AND p * r >= 3 sigma_L (distance to
      the true feature, which would otherwise falsely confirm).  With p = 10
      to 20 px (the repeat-texture pitch of this hall) and sigma_L = 1.73 px
      (level 3, the median created level) both read r >= 0.3 in the limit:
      the one number 3 sigma_L / p, recorded as such, no tuning.

Log lines of the switch (stage 04e spec C4, C5; printed ONLY with confirm on;
the harness parses them, formats are exact).  One per confirming view,
immediately after the landmark's [LOCAL-MAP-POINT] line:

    >> [LOCAL-MAP-THIRD] id=<pid> kf=<k> partner=<c> third=<j> kp_third=<ki>
       d_pair=<%.2f> d_third=<%.2f> ratio=<%.3f> err_px=<%.2f> hamming=<d>

    j the third view's trajectory index, ki the confirming keypoint, d_pair /
    d_third the translational disparities (translational_disparity) of the
    candidate in K_c and K_j, ratio = d_third / d_pair, err_px the residual of
    the confirming keypoint from the projection, hamming its descriptor
    distance to K_n's keypoint (integer for ORB, %.3g otherwise).

and once per call, immediately BEFORE the [LOCAL-MAP] line:

    >> [LOCAL-MAP-CONFIRM] kf=<k> pending=<n> no_third_view=<v> unconfirmed=<u> confirmed=<c>

    pending: gate-accepted tuples over all triangulated pairs of the call;
    no_third_view: tuples with no eligible third view; confirmed: landmarks
    created; unconfirmed = pending - no_third_view - confirmed.

Stage 04k (ruling S5; epipolar_refine.py, spec S-e(i)): with a RefineParams
passed as ``refine`` to process_new_keyframe (the --epi_refine switch), each
candidate of match_pair is refined in K_c along the epipolar line of its K_n
keypoint (refine_pair_candidates); a refused refinement drops the candidate.
Triangulation, the pair-level parallax gate, the per-point gates and the 04e
confirmation read the refined pixel through a _KeypointOverrideView of K_c;
creation and linking use the real frames and keypoint indices.  Two lines,
printed ONLY with the switch on: before the pair's [LOCAL-MAP-PARALLAX] line

    >> [LOCAL-MAP-REFINE-PAIR] kf=<k> partner=<c> attempted=<n> refined=<n>
       ambiguous=<n> low_peak=<n> edge=<n> oob=<n> no_image=<n> degenerate=<n>

and, per created landmark, after its [LOCAL-MAP-POINT] line and any
[LOCAL-MAP-THIRD] lines of it

    >> [LOCAL-MAP-REFINE] id=<pid> kf=<k> partner=<c> kp_n=<i> kp_c=<j>
       u_c=<%.2f> v_c=<%.2f> ur_c=<%.3f> vr_c=<%.3f> t=<%.3f> peak=<%.3f>

    u_c, v_c the keypoint pixel (as on the point line), ur_c, vr_c the refined
    pixel the point was triangulated from, t the signed move along the unit
    direction and peak the ZNCC maximum.  With refine None nothing changes.

Conventions (conventions.md#pose-convention): cam_T_world maps world to
camera, world_T_cam is its inverse, the camera centre is world_T_cam[:3, 3],
the projection matrix is geometry.camera_pose_to_projection(cam_T_world, K).
The relative pose used for guided matching is T_cn = K_c.cam_T_world @
K_n.world_T_cam (camera-n coordinates to camera-c coordinates), giving the
fundamental matrix F = K_c^-T [t_cn]_x R_cn K_n^-1 with x_c^T F x_n = 0
(tests/test_local_mapping.py::test_epipolar_band_direction pins it).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np

from epipolar_refine import REFINE_COUNTER_KEYS, STATUS_KEY, refine_along_epipolar
from geometry import camera_pose_to_projection
from mapping import local_map_track
from Point import Point


@dataclass
class LocalMappingParams:
    """Constants of the keyframe-pair source; see the module docstring for each."""
    min_shared_points: int = 15
    max_neighbours: int = 5
    min_baseline_depth_ratio: float = 0.01
    epipolar_chi2: float = 3.84
    epipole_radius_px: float = 10.0
    max_desc_dist: float = 50.0
    min_parallax_deg: float = 1.1
    reproj_chi2: float = 5.991
    scale_ratio_factor: float = 1.8
    insert_search_radius_px: float = 8.0
    cull_after_keyframes: int = 3
    min_keyframe_obs: int = 3
    confirm: bool = False
    confirm_disparity_ratio: float = 0.3

    @classmethod
    def from_args(cls, args, prefix="lm_"):
        """Build from an argparse namespace whose flags carry ``prefix`` (main.py)."""
        kw = {}
        for f in fields(cls):
            v = getattr(args, prefix + f.name, None)
            if v is not None:
                kw[f.name] = v      # argparse already applied the field's type
        return cls(**kw)


COUNTER_KEYS = ("neighbours", "pairs_used", "pairs_skipped_baseline", "candidates",
                "created", "rej_cheirality", "rej_parallax", "rej_chi2", "rej_scale",
                "obs_inserted", "culled")

_MIN_DEPTHS_FOR_MEDIAN = 10   # linked landmarks with positive depth needed for z_med (spec A3(i))
_ZERO_TRANSLATION = 1e-9      # ||t_cn|| at or below this is "numerically zero" (spec A3(ii))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _scale_factor(frame):
    cfg = getattr(frame, "detector_config", None) or {}
    try:
        return float((cfg.get("orb") or {}).get("scaleFactor", 1.2))
    except Exception:
        return 1.2


def _keypoint_levels(frame):
    """Decoded pyramid octave per keypoint (ORB: kp.octave; SIFT: the decoded low byte)."""
    kps = getattr(frame, "KeyPoints", None) or []
    method = getattr(frame, "method", "ORB")
    out = np.zeros(len(kps), dtype=np.int64)
    for i, kp in enumerate(kps):
        octv = int(getattr(kp, "octave", 0) or 0)
        if method == "SIFT":
            octv = octv & 255
            if octv >= 128:
                octv -= 256
        out[i] = octv
    return out


def _keypoint_sigmas(frame):
    """sigma = scale_factor ** octave per keypoint (SIFT: 2.0 ** octave); octave from _keypoint_levels."""
    method = getattr(frame, "method", "ORB")
    base = 2.0 if method == "SIFT" else _scale_factor(frame)
    levels = np.maximum(_keypoint_levels(frame), 0).astype(np.float64)
    return np.power(base, levels)


def _kp_xy(frame):
    kps = getattr(frame, "KeyPoints", None) or []
    if len(kps) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in kps], dtype=np.float64)


def _norm_type(frame):
    return cv2.NORM_HAMMING if getattr(frame, "method", "ORB") == "ORB" else cv2.NORM_L2


def _desc_dist_matrix(da, db, norm_type):
    """Pairwise descriptor distances (n, m): Hamming for uint8 ORB, L2 otherwise."""
    if da.shape[0] == 0 or db.shape[0] == 0:
        return np.zeros((da.shape[0], db.shape[0]), dtype=np.float64)
    if norm_type == cv2.NORM_HAMMING:
        A = np.unpackbits(np.ascontiguousarray(da, dtype=np.uint8), axis=1).astype(np.float32)
        B = np.unpackbits(np.ascontiguousarray(db, dtype=np.uint8), axis=1).astype(np.float32)
        D = A @ (1.0 - B).T + (1.0 - A) @ B.T
        return np.rint(D).astype(np.float64)
    A = np.asarray(da, dtype=np.float64)
    B = np.asarray(db, dtype=np.float64)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _skew(t):
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]], dtype=np.float64)


def fundamental_matrix(K_c, K_n, T_cn):
    """F with x_c^T F x_n = 0 for T_cn = cam_c_T_cam_n (camera-n coords to camera-c coords)."""
    R = T_cn[:3, :3]
    t = T_cn[:3, 3]
    return np.linalg.inv(K_c).T @ _skew(t) @ R @ np.linalg.inv(K_n)


def epipolar_sq_dist(F, xy_n, xy_c):
    """Squared distance of each x_c (m, 2) to the line F x_n (n, 2) -> (n, m); inf for a null line."""
    Xn = np.column_stack([xy_n, np.ones(len(xy_n))])
    Xc = np.column_stack([xy_c, np.ones(len(xy_c))])
    L = Xn @ F.T                                    # (n, 3): l_i = F x_i
    num = L @ Xc.T                                  # (n, m)
    den = L[:, 0] ** 2 + L[:, 1] ** 2               # (n,)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = (num * num) / den[:, None]
    d2[~np.isfinite(d2)] = np.inf
    d2[den <= 1e-18, :] = np.inf
    return d2


def _camera_depths(frame, world_map):
    """Camera-axis depths (positive only) of the landmarks linked in ``frame``."""
    R = frame.cam_T_world[:3, :3]
    t = frame.cam_T_world[:3, 3]
    out = []
    for pid in frame.keypoint_to_point.values():
        pt = world_map.get_point(pid)
        if pt is None:
            continue
        z = float(R[2] @ np.asarray(pt.point, dtype=np.float64) + t[2])
        if np.isfinite(z) and z > 0.0:
            out.append(z)
    return out


def _median_depth(kf_c, kf_n, world_map):
    """z_med of spec A3(i): K_c's linked landmarks, else K_n's, else None (gate skipped)."""
    for fr in (kf_c, kf_n):
        d = _camera_depths(fr, world_map)
        if len(d) >= _MIN_DEPTHS_FOR_MEDIAN:
            return float(np.median(d))
    return None


# --------------------------------------------------------------------------- #
# A2. neighbourhood
# --------------------------------------------------------------------------- #
def select_neighbours(kf_pos, kf, keyframes, params):
    """
    Earlier keyframes sharing >= min_shared_points landmarks with ``kf``, by
    shared count descending (ties: the more recent keyframe first), at most
    max_neighbours; the immediately preceding keyframe is always included.

    Fill (04b spec E1, the module docstring's bootstrap paragraph): if fewer
    than max_neighbours were chosen, the remaining slots are taken by the
    most recent earlier keyframes not already chosen, newest first, until
    max_neighbours is reached or no earlier keyframe remains.  With an empty
    map the neighbours are therefore the last max_neighbours keyframes; once
    landmarks exist, covisible keyframes take the slots first.  This fill is
    the contract's bootstrap of the default arm, not the paper's rule (the
    paper's covisibility neighbourhood assumes an initialized map).
    Returns a list of (position_in_keyframes, frame_idx, frame).
    """
    if kf_pos <= 0:
        return []
    mine = set(kf.keypoint_to_point.values())
    ranked = []
    for pos in range(kf_pos):
        idx, fr = keyframes[pos]
        shared = len(mine & set(fr.keypoint_to_point.values()))
        ranked.append((shared, pos, idx, fr))
    ranked.sort(key=lambda r: (-r[0], -r[1]))
    prev_pos = kf_pos - 1
    chosen = [r for r in ranked if r[0] >= int(params.min_shared_points) or r[1] == prev_pos]
    cap = max(1, int(params.max_neighbours))
    if len(chosen) > cap:
        head = chosen[:cap]
        if not any(r[1] == prev_pos for r in head):
            head[-1] = next(r for r in chosen if r[1] == prev_pos)
        chosen = head
    # E1 fill: the most recent earlier keyframes not already chosen, newest first.
    taken = {r[1] for r in chosen}
    for pos in range(kf_pos - 1, -1, -1):
        if len(chosen) >= cap:
            break
        if pos in taken:
            continue
        idx, fr = keyframes[pos]
        chosen.append((0, pos, idx, fr))
        taken.add(pos)
    return [(pos, idx, fr) for (_, pos, idx, fr) in chosen]


# --------------------------------------------------------------------------- #
# A3. one keyframe pair
# --------------------------------------------------------------------------- #
def match_pair(kf_n, kf_c, sig_n, sig_c, params):
    """
    A3(ii): guided matching of the UNLINKED keypoints of K_n against the
    unlinked keypoints of K_c inside the epipolar band, mutual best Hamming
    (L2 for SIFT) distance <= max_desc_dist, candidates near K_c's epipole
    excluded.  Returns (skip_reason, list of (i_n, j_c), list of (hamming,
    hamming2) aligned with the candidate list); skip_reason is
    'zero_translation' when ||t_cn|| is numerically zero (pure rotation:
    conventions.md#observability-under-pure-rotation), else None.  hamming2
    (stage 04c spec P2) is the second smallest finite in-band distance of the
    candidate's K_n row in the masked distance matrix, -1 when there is none.
    """
    T_cn = kf_c.cam_T_world @ kf_n.world_T_cam
    t_cn = T_cn[:3, 3]
    nt = float(np.linalg.norm(t_cn))
    if not np.isfinite(nt) or nt <= _ZERO_TRANSLATION:
        return "zero_translation", [], []

    desc_n = getattr(kf_n, "Descriptors", None)
    desc_c = getattr(kf_c, "Descriptors", None)
    if desc_n is None or desc_c is None or len(kf_n.KeyPoints) == 0 or len(kf_c.KeyPoints) == 0:
        return None, [], []
    idx_n = np.array([i for i in range(len(kf_n.KeyPoints)) if i not in kf_n.keypoint_to_point], dtype=int)
    idx_c = np.array([j for j in range(len(kf_c.KeyPoints)) if j not in kf_c.keypoint_to_point], dtype=int)
    if len(idx_n) == 0 or len(idx_c) == 0:
        return None, [], []

    xy_n = _kp_xy(kf_n)[idx_n]
    xy_c = _kp_xy(kf_c)[idx_c]
    F = fundamental_matrix(kf_c.K, kf_n.K, T_cn)
    d2 = epipolar_sq_dist(F, xy_n, xy_c)                       # (n, m)
    s_c = sig_c[idx_c]
    band = d2 <= float(params.epipolar_chi2) * (s_c ** 2)[None, :]

    # Epipole in K_c: the projection of K_n's centre (t_cn in camera-c coordinates).
    e = kf_c.K @ t_cn
    if abs(float(e[2])) > 1e-12:
        e_px = e[:2] / e[2]
        de2 = ((xy_c - e_px[None, :]) ** 2).sum(1)
        near = de2 < (float(params.epipole_radius_px) ** 2) * s_c
        band[:, near] = False

    if not band.any():
        return None, [], []
    D = _desc_dist_matrix(desc_n[idx_n], desc_c[idx_c], _norm_type(kf_n))
    D = np.where(band, D, np.inf)
    best_j = np.argmin(D, axis=1)
    best_i = np.argmin(D, axis=0)
    out = []
    dists = []
    for a, j in enumerate(best_j):
        dist = D[a, j]
        if not np.isfinite(dist) or dist > float(params.max_desc_dist):
            continue
        if best_i[j] != a:
            continue
        row_sorted = np.sort(D[a][np.isfinite(D[a])])
        second = float(row_sorted[1]) if len(row_sorted) >= 2 else -1.0
        out.append((int(idx_n[a]), int(idx_c[j])))
        dists.append((float(dist), second))
    return None, out, dists


def triangulate_pair(kf_n, kf_c, cands):
    """
    A3(iii): linear triangulation of each candidate pair (cv2.triangulatePoints
    on the two projection matrices).  Returns (X, parallax_deg) with X (N, 3)
    world points (non-finite rows where the triangulation failed) and
    parallax_deg (N,) the ray parallax angle in degrees between the
    world-frame rays from the two centres to the point -- the quantity the
    A3(iv) parallax gate tests -- for the candidates whose triangulation is
    finite with positive camera-axis depth in BOTH cameras, NaN for the rest.
    The pair-level gate of 04b spec E3 reads the median of the finite entries.
    """
    if not cands:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    P_c = camera_pose_to_projection(kf_c.cam_T_world, kf_c.K)
    P_n = camera_pose_to_projection(kf_n.cam_T_world, kf_n.K)
    xy_n_all = _kp_xy(kf_n)
    xy_c_all = _kp_xy(kf_c)
    i_n = np.array([c[0] for c in cands], dtype=int)
    j_c = np.array([c[1] for c in cands], dtype=int)
    pts_n = xy_n_all[i_n]
    pts_c = xy_c_all[j_c]
    X4 = cv2.triangulatePoints(P_c, P_n, pts_c.T.copy(), pts_n.T.copy())
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X4[:3] / X4[3]).T                                  # (N, 3)

    R_c, t_c = kf_c.cam_T_world[:3, :3], kf_c.cam_T_world[:3, 3]
    R_n, t_n = kf_n.cam_T_world[:3, :3], kf_n.cam_T_world[:3, 3]
    C_c = kf_c.world_T_cam[:3, 3]
    C_n = kf_n.world_T_cam[:3, 3]
    parallax = np.full(len(cands), np.nan, dtype=np.float64)
    for q in range(len(cands)):
        xw = X[q]
        if not np.all(np.isfinite(xw)):
            continue
        z_c = float(R_c[2] @ xw + t_c[2])
        z_n = float(R_n[2] @ xw + t_n[2])
        if not (z_c > 0.0 and z_n > 0.0):
            continue
        r_c = xw - C_c
        r_n = xw - C_n
        d_c = float(np.linalg.norm(r_c))
        d_n = float(np.linalg.norm(r_n))
        if d_c <= 0.0 or d_n <= 0.0:
            continue
        cos_a = float(np.dot(r_c, r_n) / (d_c * d_n))
        parallax[q] = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return X, parallax


def pair_parallax_summary(parallax_deg, min_parallax_deg):
    """
    The numbers of the [LOCAL-MAP-PARALLAX] line (04b spec E4): (m, median_deg,
    frac_over) over the finite entries of ``parallax_deg``; (0, 0.0, 0.0) when
    none is finite.
    """
    v = np.asarray(parallax_deg, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0, 0.0, 0.0
    return int(len(v)), float(np.median(v)), float(np.mean(v >= float(min_parallax_deg)))


def gate_candidates(kf_n, kf_c, cands, X, sig_n, sig_c, params, counters):
    """
    A3(iv): the four gates (cheirality, parallax, chi-square reprojection in
    each image, scale consistency) on the triangulated candidates ``X`` of
    triangulate_pair, each counted in ``counters`` when it rejects (the first
    failing gate is the one counted).  Returns list of (i_n, j_c, xw, q) with
    q the candidate's index into ``cands`` / ``X`` (stage 04c spec P2: lets
    the caller look up the candidate's parallax and per-camera depths).
    """
    if not cands:
        return []
    xy_n_all = _kp_xy(kf_n)
    xy_c_all = _kp_xy(kf_c)
    i_n = np.array([c[0] for c in cands], dtype=int)
    j_c = np.array([c[1] for c in cands], dtype=int)
    pts_n = xy_n_all[i_n]
    pts_c = xy_c_all[j_c]

    R_c, t_c = kf_c.cam_T_world[:3, :3], kf_c.cam_T_world[:3, 3]
    R_n, t_n = kf_n.cam_T_world[:3, :3], kf_n.cam_T_world[:3, 3]
    C_c = kf_c.world_T_cam[:3, 3]
    C_n = kf_n.world_T_cam[:3, 3]
    cos_min = float(np.cos(np.radians(float(params.min_parallax_deg))))
    chi2 = float(params.reproj_chi2)
    factor = float(params.scale_ratio_factor)

    accepted = []
    for q in range(len(cands)):
        xw = X[q]
        if not np.all(np.isfinite(xw)):
            continue
        xc = R_c @ xw + t_c
        xn = R_n @ xw + t_n
        if not (xc[2] > 0.0 and xn[2] > 0.0):
            counters["rej_cheirality"] += 1
            continue
        r_c = xw - C_c
        r_n = xw - C_n
        d_c = float(np.linalg.norm(r_c))
        d_n = float(np.linalg.norm(r_n))
        if d_c <= 0.0 or d_n <= 0.0:
            counters["rej_parallax"] += 1
            continue
        cos_a = float(np.dot(r_c, r_n) / (d_c * d_n))
        if cos_a > cos_min:
            counters["rej_parallax"] += 1
            continue
        u_c = kf_c.K @ xc
        u_n = kf_n.K @ xn
        e_c = float(np.sum((u_c[:2] / u_c[2] - pts_c[q]) ** 2))
        e_n = float(np.sum((u_n[:2] / u_n[2] - pts_n[q]) ** 2))
        s_n = float(sig_n[i_n[q]])
        s_c = float(sig_c[j_c[q]])
        if e_c > chi2 * s_c * s_c or e_n > chi2 * s_n * s_n:
            counters["rej_chi2"] += 1
            continue
        ratio_sigma = s_n / s_c
        r = d_c / d_n
        if r < ratio_sigma / factor or r > ratio_sigma * factor:
            counters["rej_scale"] += 1
            continue
        accepted.append((int(i_n[q]), int(j_c[q]), np.array(xw, dtype=np.float64), q))
    return accepted


def triangulate_candidates(kf_n, kf_c, cands, sig_n, sig_c, params, counters):
    """
    A3(iii)-(iv) in one call (triangulate_pair then gate_candidates), WITHOUT
    the pair-level gate of E3; process_new_keyframe applies E3 between the
    two steps and does not use this wrapper.  Returns list of (i_n, j_c, xw)
    (the q of gate_candidates' accepted tuples is dropped here so this
    wrapper's return shape is unchanged, stage 04c spec P2).
    """
    X, _ = triangulate_pair(kf_n, kf_c, cands)
    accepted = gate_candidates(kf_n, kf_c, cands, X, sig_n, sig_c, params, counters)
    return [(i, j, xw) for i, j, xw, _q in accepted]


def create_points(kf_n, kf_c, accepted, world_map):
    """
    A3(v): one landmark per accepted candidate (i_n, j_c, xw, q) with the 01b
    link discipline (no landmark on a keypoint that became linked meanwhile;
    no half-linked landmark).  Returns the list of (Point, accepted_tuple)
    pairs created, by position (stage 04c spec P2: a tuple is skipped, not
    replaced, when its keypoint became linked meanwhile, so the caller pairs
    each created Point with the tuple it came from).
    """
    created = []
    for tup in accepted:
        i, j, xw, _q = tup
        if kf_n.keypoint_to_point.get(i) is not None or kf_c.keypoint_to_point.get(j) is not None:
            continue
        desc = np.array(kf_n.Descriptors[i], copy=True)
        p = Point(point=xw, descriptor=desc)
        world_map.add_point(p, frames=[kf_c, kf_n])
        if not (world_map.link(kf_c, j, p) and world_map.link(kf_n, i, p)):
            world_map.remove_point(p.id, frames=[kf_c, kf_n])
            continue
        p.created_fid = int(kf_n.id)
        p.last_seen_fid = int(kf_n.id)
        created.append((p, tup))
    return created


# --------------------------------------------------------------------------- #
# Stage 04e. third-view chi-square confirmation (behind params.confirm)
# --------------------------------------------------------------------------- #
def ray_infinity_pixel(kf_n, xy_n, kf_b):
    """
    04e spec B1: the pixel in keyframe B where a point at INFINITE depth along
    K_n's ray through the pixel ``xy_n`` would appear (the rotation-only
    prediction, the epipolar line's point at infinity; no translation enters).
    Returns (u_inf (2,), ok); (nan, nan) and False when the ray's direction
    has non-positive camera-axis component in B.
    """
    u, v = float(xy_n[0]), float(xy_n[1])
    r_n = np.linalg.inv(np.asarray(kf_n.K, dtype=np.float64)) @ np.array([u, v, 1.0])
    w = kf_n.world_T_cam[:3, :3] @ r_n
    v_b = kf_b.cam_T_world[:3, :3] @ w
    if not (np.isfinite(v_b[2]) and v_b[2] > 1e-12):
        return np.array([np.nan, np.nan]), False
    uvw = np.asarray(kf_b.K, dtype=np.float64) @ v_b
    return np.array([uvw[0] / v_b[2], uvw[1] / v_b[2]], dtype=np.float64), True


def translational_disparity(kf_n, xy_n, X, kf_b):
    """
    04e spec B2: || pi_b(X) - u_inf || in pixels, the displacement of the world
    point ``X`` in B from the rotation-only prediction of K_n's ray through
    ``xy_n`` (ray_infinity_pixel).  NaN when the infinity point is not ok,
    when X has non-positive camera-axis depth in B, or when X is not finite.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape != (3,) or not np.all(np.isfinite(X)):
        return float("nan")
    u_inf, ok = ray_infinity_pixel(kf_n, xy_n, kf_b)
    if not ok:
        return float("nan")
    xb = kf_b.cam_T_world[:3, :3] @ X + kf_b.cam_T_world[:3, 3]
    if not (np.isfinite(xb[2]) and xb[2] > 0.0):
        return float("nan")
    uvw = np.asarray(kf_b.K, dtype=np.float64) @ xb
    uv = uvw[:2] / uvw[2]
    return float(np.linalg.norm(uv - u_inf))


def confirm_candidates(kf_n, kf_c, accepted, third_views, sigma_cache, params):
    """
    04e spec C1: the third-view confirmation of the gate-accepted candidates
    ``accepted`` (the (i_n, j_c, xw, q) tuples of gate_candidates for the pair
    (kf_n, kf_c)) against ``third_views``, a list of (traj_idx, frame) for
    every neighbour of this call other than kf_c, in the given order.

    For each candidate: d_pair = translational_disparity in K_c (NaN or <= 1e-9:
    counted in n_no_third_view); a third view K_j is ELIGIBLE iff its
    disparity d_j is finite, r = d_j / d_pair satisfies r >= ratio and
    |r - 1| >= ratio (params.confirm_disparity_ratio, both sides) and the
    projection lies inside K_j's image.  In an eligible view the UNLINKED,
    unreserved keypoints inside the chi-square window (params.reproj_chi2 *
    sigma_j[ki]^2, the keypoint's OWN level, no other search radius) compete
    by descriptor distance to kf_n.Descriptors[i_n]; the nearest confirms iff
    its distance is <= params.max_desc_dist, and is then RESERVED for this
    call (one keypoint confirms one candidate).  Returns (confirmed,
    n_no_third_view) with confirmed a list of (tuple, views), views =
    [(traj_idx_j, kf_j, ki, d_pair, d_j, r, err_px, desc_dist), ...] over
    every confirming view.  Candidates with an eligible view and no
    confirming keypoint are unconfirmed (not returned); candidates with no
    eligible view are counted in n_no_third_view.  Reads the map through the
    frames' link tables only; never creates, links or removes anything.
    """
    confirmed = []
    n_no_third_view = 0
    if not accepted:
        return confirmed, n_no_third_view
    ratio_min = float(params.confirm_disparity_ratio)
    chi2 = float(params.reproj_chi2)
    max_d = float(params.max_desc_dist)
    norm_type = _norm_type(kf_n)
    xy_n_all = _kp_xy(kf_n)
    desc_n = getattr(kf_n, "Descriptors", None)

    views_geom = []
    for traj_idx_j, kf_j in third_views:
        if kf_j is kf_c or kf_j is kf_n:
            continue
        if getattr(kf_j, "Descriptors", None) is None or len(kf_j.KeyPoints) == 0:
            continue
        if kf_j.id not in sigma_cache:
            sigma_cache[kf_j.id] = _keypoint_sigmas(kf_j)
        h, w = kf_j.img_shape[:2]
        views_geom.append((int(traj_idx_j), kf_j, _kp_xy(kf_j), sigma_cache[kf_j.id],
                           kf_j.cam_T_world[:3, :3], kf_j.cam_T_world[:3, 3],
                           np.asarray(kf_j.K, dtype=np.float64), float(w), float(h)))

    reserved = set()      # (id(kf_j), ki) keypoints that confirmed a candidate of this call
    for tup in accepted:
        i_n, _j_c, xw, _q = tup
        xy_n = xy_n_all[i_n]
        d_pair = translational_disparity(kf_n, xy_n, xw, kf_c)
        if not np.isfinite(d_pair) or d_pair <= 1e-9:
            n_no_third_view += 1
            continue
        n_eligible = 0
        views = []
        for traj_idx_j, kf_j, xy_j, sig_j, R_j, t_j, K_j, w, h in views_geom:
            d_j = translational_disparity(kf_n, xy_n, xw, kf_j)
            if not np.isfinite(d_j):
                continue
            r = d_j / d_pair
            if r < ratio_min or abs(r - 1.0) < ratio_min:
                continue
            xj = R_j @ xw + t_j
            uvw = K_j @ xj
            uv = uvw[:2] / uvw[2]
            if not (0.0 <= uv[0] < w and 0.0 <= uv[1] < h):
                continue
            n_eligible += 1
            d2 = ((xy_j - uv[None, :]) ** 2).sum(1)
            within = np.where(d2 <= chi2 * sig_j * sig_j)[0]
            best_ki, best_d = -1, np.inf
            for ki in within:
                ki = int(ki)
                if ki in kf_j.keypoint_to_point or (id(kf_j), ki) in reserved:
                    continue
                d = float(cv2.norm(desc_n[i_n], kf_j.Descriptors[ki], norm_type))
                if d < best_d:
                    best_d, best_ki = d, ki
            if best_ki >= 0 and best_d <= max_d:
                reserved.add((id(kf_j), best_ki))
                views.append((traj_idx_j, kf_j, best_ki, float(d_pair), float(d_j), float(r),
                              float(np.sqrt(d2[best_ki])), best_d))
        if n_eligible == 0:
            n_no_third_view += 1
        elif views:
            confirmed.append((tup, views))
    return confirmed, n_no_third_view


# --------------------------------------------------------------------------- #
# Stage 04k. sub-pixel refinement along the epipolar line (behind refine)
# --------------------------------------------------------------------------- #
class _KeypointOverrideView:
    """
    Read-only view of ``frame`` whose KeyPoints list has entry j replaced, for
    every j in ``overrides`` (j -> (u, v)), by a new cv2.KeyPoint at (u, v)
    carrying the original's size, angle, response, octave and class_id; every
    other entry is the original object and every other attribute is read from
    the frame.  The frame is never mutated.
    """

    def __init__(self, frame, overrides):
        self._frame = frame
        kps = list(getattr(frame, "KeyPoints", None) or [])
        for j, (u, v) in overrides.items():
            kp = kps[j]
            kps[j] = cv2.KeyPoint(x=float(u), y=float(v), size=kp.size, angle=kp.angle,
                                  response=kp.response, octave=kp.octave, class_id=kp.class_id)
        self.KeyPoints = kps

    def __getattr__(self, name):
        if name == "_frame":
            raise AttributeError(name)
        return getattr(self._frame, name)


def refine_pair_candidates(kf_n, kf_c, cands, dists, refine):
    """
    04k spec F2(iii): each candidate (i, j) of match_pair refined in K_c along
    the epipolar line of x_n (l = F x_n, direction (l[1], -l[0]); F as
    match_pair builds it).  Returns (cands_kept, dists_kept, view_c, recs,
    counts): the refined candidates in their original order with their dists
    entries; view_c a _KeypointOverrideView of K_c at the refined pixels (a
    K_c keypoint appears at most once per pair: match_pair is mutual-best);
    recs (i, j) -> (u_c, v_c, ur_c, vr_c, t, peak); counts the
    REFINE_COUNTER_KEYS counters.
    """
    counts = {key: 0 for key in REFINE_COUNTER_KEYS}
    counts["refine_attempted"] = len(cands)
    T_cn = kf_c.cam_T_world @ kf_n.world_T_cam
    F = fundamental_matrix(kf_c.K, kf_n.K, T_cn)
    xy_n = _kp_xy(kf_n)
    xy_c = _kp_xy(kf_c)
    img_n, D_n = getattr(kf_n, "img", None), getattr(kf_n, "distortion", None)
    img_c, D_c = getattr(kf_c, "img", None), getattr(kf_c, "distortion", None)
    kept, kept_d, overrides, recs = [], [], {}, {}
    for (i, j), dd in zip(cands, dists):
        l = F @ np.array([xy_n[i][0], xy_n[i][1], 1.0])
        direction = np.array([l[1], -l[0]], dtype=np.float64)
        xy_new, status, peak, t = refine_along_epipolar(img_n, xy_n[i], kf_n.K, D_n,
                                                        img_c, xy_c[j], kf_c.K, D_c, direction, refine)
        counts[STATUS_KEY[status]] += 1
        if status != "refined":
            continue
        kept.append((i, j))
        kept_d.append(dd)
        overrides[j] = (float(xy_new[0]), float(xy_new[1]))
        recs[(i, j)] = (float(xy_c[j][0]), float(xy_c[j][1]), float(xy_new[0]), float(xy_new[1]),
                        float(t), float(peak))
    return kept, kept_d, _KeypointOverrideView(kf_c, overrides), recs, counts


# --------------------------------------------------------------------------- #
# A4(b). insertion of the new landmarks into the other neighbours
# --------------------------------------------------------------------------- #
def insert_new_points_into_neighbours(new_points, neighbours, world_map, params, sigma_cache):
    """
    Project every landmark created in this call into every neighbour other
    than its own pair partner; link the best-descriptor unlinked keypoint
    within insert_search_radius_px whose descriptor distance is <=
    max_desc_dist and whose squared reprojection error passes the chi-square
    gate at its level.  Returns the number of links written.
    """
    n_added = 0
    chi2 = float(params.reproj_chi2)
    radius = float(params.insert_search_radius_px)
    max_d = float(params.max_desc_dist)
    for _, _, nb in neighbours:
        if getattr(nb, "Descriptors", None) is None or len(nb.KeyPoints) == 0:
            continue
        xy = _kp_xy(nb)
        sig = sigma_cache[nb.id]
        R, t = nb.cam_T_world[:3, :3], nb.cam_T_world[:3, 3]
        K = nb.K
        h, w = nb.img_shape[:2]
        norm_type = _norm_type(nb)
        for p, partner in new_points:
            if partner is nb or world_map.get_point(p.id) is None:
                continue
            if nb.id in p.observations:
                continue
            xc = R @ np.asarray(p.point, dtype=np.float64) + t
            if not (np.isfinite(xc[2]) and xc[2] > 0.0):
                continue
            uvw = K @ xc
            uv = uvw[:2] / uvw[2]
            if not (-radius <= uv[0] < w + radius and -radius <= uv[1] < h + radius):
                continue
            d2 = ((xy - uv[None, :]) ** 2).sum(1)
            within = np.where(d2 <= radius * radius)[0]
            best_ki, best_d = -1, np.inf
            for ki in within:
                ki = int(ki)
                if ki in nb.keypoint_to_point:
                    continue
                if d2[ki] > chi2 * sig[ki] * sig[ki]:
                    continue
                d = float(cv2.norm(p.descriptor, nb.Descriptors[ki], norm_type))
                if d < best_d:
                    best_d, best_ki = d, ki
            if best_ki >= 0 and best_d <= max_d:
                if world_map.link(nb, best_ki, p):
                    n_added += 1
    return n_added


# --------------------------------------------------------------------------- #
# A5. observation-count culling
# --------------------------------------------------------------------------- #
def cull_points(kf_pos, keyframes, world_map, frames, params):
    """
    Remove every landmark whose created_fid is a keyframe >= cull_after_keyframes
    entries before the new keyframe and whose number of KEYFRAME observations
    is < min_keyframe_obs (paper VI.B second condition; the wait is this
    contract's departure, see the module docstring).  Landmarks whose
    created_fid is not a keyframe are left alone.  Returns the count removed.
    """
    kf_fids = {}
    for pos, (_, fr) in enumerate(keyframes):
        kf_fids[int(fr.id)] = pos
    wait = int(params.cull_after_keyframes)
    min_obs = int(params.min_keyframe_obs)
    culled = 0
    for pt in list(world_map.points):
        pos = kf_fids.get(int(getattr(pt, "created_fid", -1)))
        if pos is None or kf_pos - pos < wait:
            continue
        n_kf = sum(1 for fid in pt.observations if fid in kf_fids)
        if n_kf < min_obs:
            world_map.remove_point(pt.id, frames=frames)
            culled += 1
    return culled


# --------------------------------------------------------------------------- #
# the per-keyframe entry point
# --------------------------------------------------------------------------- #
def process_new_keyframe(kf_idx, kf, keyframes, world_map, frames, params=None, stats=None,
                         refine=None):
    """
    Local mapping for the new keyframe ``kf`` (trajectory index ``kf_idx``):
    neighbourhood (A2), triangulation with every neighbour (A3), observation
    insertion (A4), culling (A5), the [LOCAL-MAP-PAIR] / [LOCAL-MAP] lines
    (A7) and the cumulative stats counters.  ``keyframes`` is the loop
    closer's list of (frame_idx, frame) INCLUDING ``kf`` as its last entry.

    Returns the counters of the [LOCAL-MAP] line under the same keys plus
    'kf' and 'pairs': [(partner_idx, created, reason), ...] in print order
    (reason None for a pair that was triangulated).  The pairs are processed
    in descending order of baseline (04b spec E2); the neighbours count and
    the A4 insertion set are all neighbours regardless of order.  Stage 04e:
    with params.confirm on, a pair's gate-accepted candidates pass through
    confirm_candidates (third views: the other neighbours, widest first)
    before create_points, and the dict's four keys pending / no_third_view /
    unconfirmed / confirmed carry the [LOCAL-MAP-CONFIRM] counters (zeros
    with the switch off, where nothing else differs from the 04d mapper).
    Stage 04k: with ``refine`` a RefineParams (--epi_refine), each pair's
    candidates pass through refine_pair_candidates before triangulation (the
    module docstring's 04k paragraph); with refine None nothing changes.
    """
    if params is None:
        params = LocalMappingParams()
    c = {k: 0 for k in COUNTER_KEYS}
    pairs_out = []

    kf_pos = len(keyframes) - 1
    for pos in range(len(keyframes) - 1, -1, -1):
        if keyframes[pos][1] is kf:
            kf_pos = pos
            break

    neighbours = select_neighbours(kf_pos, kf, keyframes, params)
    c["neighbours"] = len(neighbours)
    sigma_cache = {kf.id: _keypoint_sigmas(kf)}
    for _, _, nb in neighbours:
        if nb.id not in sigma_cache:
            sigma_cache[nb.id] = _keypoint_sigmas(nb)
    sig_n = sigma_cache[kf.id]

    # E2: widest baseline first (ties: the more recent partner, i.e. the larger position, first).
    C_n = kf.world_T_cam[:3, 3]
    baselines = {pos: float(np.linalg.norm(C_n - nb.world_T_cam[:3, 3])) for pos, _, nb in neighbours}
    ordered = sorted(neighbours, key=lambda r: (-baselines[r[0]], -r[0]))

    min_par = float(params.min_parallax_deg)
    is_orb_n = _norm_type(kf) == cv2.NORM_HAMMING
    xy_n_all = _kp_xy(kf)
    lev_n_all = _keypoint_levels(kf)
    R_n, t_n = kf.cam_T_world[:3, :3], kf.cam_T_world[:3, 3]
    new_points = []   # (Point, partner frame)
    # stage 04e: the confirmation counters (zeros with the switch off)
    confirm_on = bool(getattr(params, "confirm", False))
    pending = no_third_view = confirmed_total = 0
    for pos, c_idx, kf_c in ordered:
        reason = None
        # (i) baseline / median-depth gate
        b = baselines[pos]
        z_med = _median_depth(kf_c, kf, world_map)
        if z_med is not None and z_med > 0.0 and b / z_med < float(params.min_baseline_depth_ratio):
            reason = "baseline_ratio"
        created_here = 0
        if reason is None:
            # (ii) guided matching; skips the pair under zero translation
            reason, cands, dists = match_pair(kf, kf_c, sig_n, sigma_cache[kf_c.id], params)
            if reason is None:
                # stage 04k: the geometric steps below read K_c through geo_c, the refined view
                # (the real frame with the switch off); creation and linking use kf_c itself.
                geo_c = kf_c
                recs = None
                if refine is not None:
                    cands, dists, geo_c, recs, rc = refine_pair_candidates(kf, kf_c, cands, dists, refine)
                    print(f"  >> [LOCAL-MAP-REFINE-PAIR] kf={int(kf_idx)} partner={int(c_idx)} "
                          f"attempted={rc['refine_attempted']} refined={rc['refine_ok']} "
                          f"ambiguous={rc['refine_ambiguous']} low_peak={rc['refine_low_peak']} "
                          f"edge={rc['refine_edge']} oob={rc['refine_oob']} no_image={rc['refine_no_image']} "
                          f"degenerate={rc['refine_degenerate']}")
                # (iii) triangulation, then the E3 pair-level gate on the pair's own candidates
                X, parallax = triangulate_pair(kf, geo_c, cands)
                m, med_deg, frac_over = pair_parallax_summary(parallax, min_par)
                print(f"  >> [LOCAL-MAP-PARALLAX] kf={int(kf_idx)} partner={int(c_idx)} n={m} "
                      f"median_deg={med_deg:.3f} frac_over={frac_over:.3f} baseline={b:.6g}")
                if m >= 1 and med_deg < min_par:
                    reason = "baseline_ratio"     # parallax-starved pair: skipped as a whole (E3)
                else:
                    # (iv)-(v) per-point gates and creation, counted exactly as built
                    c["candidates"] += len(cands)
                    accepted = gate_candidates(kf, geo_c, cands, X, sig_n, sigma_cache[kf_c.id], params, c)
                    third_of = {}
                    if confirm_on:
                        # stage 04e spec C2: a gate-accepted candidate is created only after
                        # confirm_candidates re-finds it in an eligible further neighbour; the
                        # confirming keypoint is linked as a creation observation.
                        third_views = [(int(j_idx), kf_j) for _, j_idx, kf_j in ordered if kf_j is not kf_c]
                        confirmed, n_none = confirm_candidates(kf, geo_c, accepted, third_views, sigma_cache, params)
                        made = create_points(kf, kf_c, [tup for tup, _views in confirmed], world_map)
                        by_tup = {id(tup): views for tup, views in confirmed}
                        for p, tup in made:
                            third_of[p.id] = by_tup.get(id(tup), [])
                            for (j_idx, kf_j, ki, _dp, _dj, _r, _err, _dd) in third_of[p.id]:
                                if not world_map.link(kf_j, ki, p):
                                    print(f"  >> [LOCAL-MAP-ERROR] confirm link refused id={p.id} "
                                          f"kf={int(kf_idx)} third={int(j_idx)}")
                        pending += len(accepted)
                        no_third_view += n_none
                        confirmed_total += len(made)
                    else:
                        made = create_points(kf, kf_c, accepted, world_map)
                    new_points.extend((p, kf_c) for p, _tup in made)
                    created_here = len(made)
                    # stage 04c spec P1: one [LOCAL-MAP-POINT] line per created landmark,
                    # printed before this pair's [LOCAL-MAP-PAIR] line.
                    if made:
                        xy_c_all = _kp_xy(kf_c)
                        lev_c_all = _keypoint_levels(kf_c)
                        R_c, t_c = kf_c.cam_T_world[:3, :3], kf_c.cam_T_world[:3, 3]
                        for p, (i, j, xw, q) in made:
                            hamming, hamming2 = dists[q]
                            z_n = float(R_n[2] @ xw + t_n[2])
                            z_c = float(R_c[2] @ xw + t_c[2])
                            u_n, v_n = xy_n_all[i]
                            u_c, v_c = xy_c_all[j]
                            ham_str = f"{int(round(hamming))}" if is_orb_n else f"{hamming:.3g}"
                            if hamming2 < 0.0:
                                ham2_str = "-1"
                            else:
                                ham2_str = f"{int(round(hamming2))}" if is_orb_n else f"{hamming2:.3g}"
                            print(
                                f"  >> [LOCAL-MAP-POINT] id={p.id} kf={int(kf_idx)} partner={int(c_idx)} "
                                f"kp_n={i} kp_c={j} u_n={u_n:.2f} v_n={v_n:.2f} u_c={u_c:.2f} v_c={v_c:.2f} "
                                f"level_n={int(lev_n_all[i])} level_c={int(lev_c_all[j])} "
                                f"hamming={ham_str} hamming2={ham2_str} parallax_deg={parallax[q]:.3f} "
                                f"z_n={z_n:.6g} z_c={z_c:.6g} baseline={b:.6g}"
                            )
                            # stage 04e spec C4: one [LOCAL-MAP-THIRD] line per confirming view,
                            # only with the switch on, immediately after the point line.
                            for (j_idx, _kf_j, ki, d_pair, d_j, r, err_px, dd) in third_of.get(p.id, []):
                                dd_str = f"{int(round(dd))}" if is_orb_n else f"{dd:.3g}"
                                print(
                                    f"  >> [LOCAL-MAP-THIRD] id={p.id} kf={int(kf_idx)} partner={int(c_idx)} "
                                    f"third={int(j_idx)} kp_third={int(ki)} d_pair={d_pair:.2f} d_third={d_j:.2f} "
                                    f"ratio={r:.3f} err_px={err_px:.2f} hamming={dd_str}"
                                )
                            # stage 04k: the refined pixel the point was triangulated from,
                            # only with the switch on, after the point's other lines.
                            if recs is not None:
                                ku_c, kv_c, ur_c, vr_c, t_ref, peak = recs[(i, j)]
                                print(
                                    f"  >> [LOCAL-MAP-REFINE] id={p.id} kf={int(kf_idx)} partner={int(c_idx)} "
                                    f"kp_n={i} kp_c={j} u_c={ku_c:.2f} v_c={kv_c:.2f} ur_c={ur_c:.3f} "
                                    f"vr_c={vr_c:.3f} t={t_ref:.3f} peak={peak:.3f}"
                                )
        if reason is None:
            c["pairs_used"] += 1
            print(f"  >> [LOCAL-MAP-PAIR] kf={int(kf_idx)} partner={int(c_idx)} created={created_here}")
        else:
            c["pairs_skipped_baseline"] += 1
            print(f"  >> [LOCAL-MAP-PAIR] kf={int(kf_idx)} partner={int(c_idx)} created=0 reason={reason}")
        pairs_out.append((int(c_idx), created_here, reason))
    c["created"] = len(new_points)

    # A4(a): the neighbours' landmarks into K_n; (b): the new landmarks into the neighbours.
    if neighbours:
        c["obs_inserted"] += int(local_map_track(
            kf, world_map, [nb for _, _, nb in neighbours],
            search_radius=float(params.insert_search_radius_px),
            max_desc_dist=float(params.max_desc_dist),
            max_depth_ratio=None,
        ))
        c["obs_inserted"] += insert_new_points_into_neighbours(new_points, neighbours, world_map, params, sigma_cache)

    # A5
    c["culled"] = cull_points(kf_pos, keyframes, world_map, frames, params)

    # stage 04e spec C5: the confirmation summary, only with the switch on, before [LOCAL-MAP].
    unconfirmed = pending - no_third_view - confirmed_total
    if confirm_on:
        print(f"  >> [LOCAL-MAP-CONFIRM] kf={int(kf_idx)} pending={pending} no_third_view={no_third_view} "
              f"unconfirmed={unconfirmed} confirmed={confirmed_total}")
    print(
        f"  >> [LOCAL-MAP] kf={int(kf_idx)} neighbours={c['neighbours']} pairs_used={c['pairs_used']} "
        f"pairs_skipped_baseline={c['pairs_skipped_baseline']} candidates={c['candidates']} "
        f"created={c['created']} rej_cheirality={c['rej_cheirality']} rej_parallax={c['rej_parallax']} "
        f"rej_chi2={c['rej_chi2']} rej_scale={c['rej_scale']} obs_inserted={c['obs_inserted']} "
        f"culled={c['culled']}"
    )
    if stats is not None:
        stats["lm_keyframes"] = stats.get("lm_keyframes", 0) + 1
        stats["lm_created"] = stats.get("lm_created", 0) + c["created"]
        stats["lm_obs_inserted"] = stats.get("lm_obs_inserted", 0) + c["obs_inserted"]
        stats["lm_culled"] = stats.get("lm_culled", 0) + c["culled"]
    out = dict(c)
    out["kf"] = int(kf_idx)
    out["pairs"] = pairs_out
    out["pending"] = int(pending)
    out["no_third_view"] = int(no_third_view)
    out["unconfirmed"] = int(unconfirmed)
    out["confirmed"] = int(confirmed_total)
    return out
