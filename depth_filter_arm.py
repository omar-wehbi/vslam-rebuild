"""
Stage 04g: the SVO depth filter (depth_filter.py, the module verified in 04f)
wired into the product as the --point_source depth_filter arm, the tested
alternative of the slot "Point source" (design.md#slots-default-arms-and-
tested-alternatives; design.md#04-1-after-04c-the-mechanism-and-the-route,
second arm).  Off by default: with --point_source keyframe_pair nothing in
this module runs and the product is byte for byte the 04c / 04d tree.
Stage 04i adds the R2 seed lifetime behind one switch (--df_lifetime r2;
the default 'age' is the verified 04h driver byte for byte).

This module is the DRIVER.  It owns the live seed list, the Map side of the
filter (the module has no Map argument: it creates, links and removes
nothing) and the promotion TIMING; the module owns the measurement, the
posterior and the promotion TEST (depth_filter.promotable); under r2 the
state machine's tests are the pure functions of depth_filter_lifetime.py.
It imports numpy, dataclasses, the module, the lifetime module, five readers
of local_mapping (select_neighbours, cull_points,
insert_new_points_into_neighbours, _keypoint_sigmas, _camera_depths),
mapping.local_map_track and Point.  It never reads a dataset, an image file
or any truth file (the 03c no-leak rule); local_mapping.py itself is not
edited and gains no function.

Three entry points, called by pipeline.py for every frame with a valid pose
(on_frame, BEFORE the keyframe decision of that frame, after the frame's
sliding-window BA), for every keyframe insertion (on_keyframe, in place of
local_mapping.process_new_keyframe, after on_frame ran for the same frame)
and once after the frame loop (on_end, stage 04i):

    DepthFilterArm(df_params, arm_params, lm_params)
        .on_frame(k, frame, keyframes, world_map, frames)    -> dict (spec C)
        .on_keyframe(k, kf, keyframes, world_map, frames)    -> dict (spec D)
        .on_end(k_last, keyframes, world_map, frames)        -> dict (04i spec E; {} under age)
        .tentative_points()                                   -> [(X, sigma_rel, sigma2, seed)] (04i C6)
    scene_range(kf, kf_pos, keyframes, world_map, arm_params, df_params)
        -> (z_init or nan, source in {"links", "map", "window", "none"})

`keyframes` is the loop closer's list of (frame_idx, frame) including the new
keyframe as its last entry; `frames` the pipeline's frame list; k the
trajectory index of the frame (Frame.id - 1, the integer the log lines carry).

THE SEEDING RULE (spec D, once per keyframe insertion).  After the slot's
keyframe-time observation insertion (A4(a): the neighbours' landmarks
projected into the new keyframe, exactly the call process_new_keyframe makes)
and the observation-count culling (A5), the seeds whose reference keyframe is
max_seed_keyframes old or whose reference keypoint became linked are retired
(depth_filter.retire_seeds; under r2 by linkage only, D2 below), and one seed
is made on every keypoint of the new keyframe that is still unlinked
(depth_filter.make_seeds), at the keyframe's scene range z_init (the design's
"keyframe's mean scene depth"; scene_range below).  A keyframe whose scene
range cannot be read (the first keyframe; a keyframe whose window has not
moved) makes no seeds.

THE PROMOTION RULE UNDER 'age' (spec C, once per posed frame).  Every live
seed takes at most one measurement in the frame (depth_filter.update_seeds);
then, in seeding order, a seed whose reference keypoint got linked meanwhile
is dropped (no landmark), and a seed is PROMOTED iff

    promotable(seed) is ok   AND   theta_acc_deg >= promote_factor * min_parallax_deg

with theta_acc the module's own accumulated-parallax reading (the parallax a
single level-0 measurement would need to reach the posterior).  Promotion
makes a landmark at the posterior point X(mu) with the reference keypoint's
descriptor, links the reference keypoint and every INLIER measurement
(responsibility >= responsibility_inlier) as observations under the 01b
discipline (first link wins: a keypoint already linked elsewhere is not
stolen, the refusal is counted and no error is raised), stamps created_fid
with the newest keyframe at promotion (the creation keyframe in the paper's
culling sense, so cull_points judges the landmark after cull_after_keyframes
further keyframes exactly as it judges a pair-created one) and inserts the
new landmark into the recent keyframes (the slot's insertion rule, the
paper's "project into connected keyframes": local_mapping's
insert_new_points_into_neighbours over the last max_neighbours keyframes,
which links only the best-descriptor unlinked keypoint inside the
chi-square window at the keypoint's level).  The default arm's
keyframe-pair triangulation does not run under this arm, so the arm is the
only creator of landmarks.  The landmark-creation block is _create_landmark,
shared by both lifetimes.

THE LIFETIME UNDER 'r2' (stage 04i, ruling R2; depth_filter_lifetime.py
states the tests and what each reads).  Per posed frame (on_frame):
  C1  every live seed whose reference keypoint is linked is dropped
      ([DF-RETIRE] reason=linked);
  C2  the rest are classified (classify): dormant (credible interval outside
      the image or behind the camera: dormant_frames += 1, nothing else),
      skipped (predicted disparity below match_error_px: nothing) or eligible;
  C3  the eligible sub-list measures (update_seeds).  A matched seed resets
      its failure count and its last-match centre; an unmatched one takes an
      outlier observation (b + 1), counts an eligible failure, and is retired
      when hopeless (P(pi < eta) >= conf) or dead (travel since the last match
      >= death_travel_ratio z_init; a tentative dead seed whose convergence
      still holds is CONFIRMED reason=dead instead);
  C4  every matched seed is re-tested (converged: promotable AND N distinct
      baselines); it becomes tentative when ok (the public handle,
      tentative_points), loses the handle when not; a tentative seed with
      theta_acc >= promote_factor theta_min is CONFIRMED reason=early;
  C5  confirmation = _create_landmark + [DF-CONFIRM]; the landmark is a map
      point from then on, culled like any other;
  C7  the frame's new landmarks are inserted into the recent keyframes, then
      [DF-LIFETIME] and [DF-FRAME] are printed.
Per keyframe (on_keyframe): D1 as age; D2 retirement by linkage only (no age
retirement, retire_seeds is not called); D3 the scene range with the
bootstrap window arm_params.bootstrap_window; D4 seeding with the per-seed
state initialised; D5 the cap: while more than seed_cap seeds are live, the
older seeds are evicted in eviction_order (a tentative one still converged
is CONFIRMED reason=evicted); D6 [DF-LIFETIME-KF] then [DF-KEYFRAME].  At the
end of the run (on_end): every tentative seed still converged is CONFIRMED
reason=end, every other live seed retired reason=end, then [DF-END].

Constants (DepthFilterArmParams; every one is a --df_ flag of main.py):

  promote_factor = 1.0
      The k of the promotion-timing ruling (design.md#04-1-after-04c-the-
      mechanism-and-the-route, item 2, human 2026-09-20).  The filter's
      geometric variance assumes sigma_0 = 1 px of disparity noise; on this
      hall the along-line matching error is 5 to 10 px (04c), so the filter
      believes it has converged before it has, and promotion at first
      convergence lands at about theta_min-equivalent precision.  Deferring
      promotion to theta_acc >= k * theta_min with k = epsilon / sigma_0
      (epsilon the measured along-line error scale) is EXACTLY the statement
      "the posterior would just have met condition (i) had the variance used
      epsilon instead of 1 px": it corrects the mis-sized noise model at the
      one place the ruling identifies, in the ruling's own regime quantity,
      and does not depend on the keyframe interval.  1.0 is the paper's
      timing (promote at first convergence; the clause is then implied by
      promotable's own (i)) and is the REFERENCE arm; the tested alternative
      runs with k = 9.5 (04d: und_attr_o3plus_match_err_px_p50 = 9.51 px over
      the 1 px the variance assumes, so the alternative promotes at theta_acc
      >= 9.5 * 1.1 = 10.45 deg; condition (ii) unchanged), passed by the
      harness as --df_promote_factor 9.5, never a product default.  Under r2
      the same clause is the early-confirmation trigger of a tentative seed.
  bootstrap_depth_ratio = 5.0
      Scene range of a keyframe before any landmark is visible in it (source
      "window" of scene_range), as a ratio to the widest baseline of the
      bootstrap window (the previous bootstrap_window keyframes): the scene
      is taken as five window-baselines deep.  Regime statement: on this hall
      the widest-first creating pairs of the default arm sit at 0.66 m p50
      (04d) against 2.5 to 3 m of scene depth, ratio 4 to 4.5; the prior then
      admits ranges from bootstrap_depth_ratio * init_depth_ratio_min = half
      a window baseline to infinity, and an error of a factor two either way
      in the centre is well inside the prior's 6-sigma width.  Stated as a
      ratio of two map-unit lengths, never a metric depth: the map's unit is
      not known to the product.
  lifetime = "age"            age (04h, above) | r2 (stage 04i, above).
  bootstrap_window = 5        the window of scene_range's "window" source, under
      both lifetimes: the value max_seed_keyframes gave it in 04h (R2
      decouples it from the seed lifetime; --df_max_seed_keyframes is now
      age-lifetime retirement only).
  match_error_px = 9.5, death_travel_ratio = 0.3, outlier_eta = 0.05,
  outlier_conf = 0.99, distinct_baselines = 3, seed_cap = 25000
      the R2 constants; their derivations are in depth_filter_lifetime.py.
  trace_path = ""             off when empty; under r2 one line per matched
      measurement is appended to this file (--df_trace; L5 below).

Log lines (spec E of 04g, spec L of 04i; the harness regexes are anchored at
both ends; the formats are exact).  Once per keyframe insertion, at the end
of on_keyframe:

    >> [DF-KEYFRAME] kf=<k> pos=<kf_pos> neighbours=<n> obs_inserted=<o> culled=<q>
       retired=<r> retired_linked=<rl> retired_converged=<rc> unlinked=<u>
       seeds_made=<s> z_init=<%.6g|nan> z_source=<links|map|window|none> live=<l>

once per posed frame, at the end of on_frame:

    >> [DF-FRAME] frame=<k> live=<n> measured=<m> inlier=<i> outlier=<o> no_match=<a>
       no_translation=<b> behind=<c> no_disparity=<d> outside_support=<e> converged=<q>
       promoted=<p> dropped_linked=<x> links=<l> links_refused=<r> inserted=<ins>

(live is the module's `seeds` counter, so the module's two identities live ==
measured + no_match + no_translation + behind + no_disparity and measured ==
inlier + outlier hold on every line; under r2 live is the ELIGIBLE count;
converged = seeds for which the test was ok in this frame, promoted ones
included); with the stage 04k switch (--epi_refine, ``refine`` a RefineParams)
on, immediately after every [DF-FRAME] line, from that frame's update_seeds
counters (zeros if update_seeds was not called):

    >> [DF-REFINE] frame=<k> attempted=<n> refined=<n> ambiguous=<n> low_peak=<n> edge=<n>
       oob=<n> no_image=<n> degenerate=<n>

and one per promoted landmark under age, before the frame's [DF-FRAME] line:

    >> [DF-PROMOTE] id=<pid> ref=<k_ref> kp=<i> u=<%.2f> v=<%.2f> level=<L> frame=<k>
       converged_at=<k_c> widest=<k_w> theta_max_deg=<%.3f> theta_acc_deg=<%.3f>
       n_meas=<n> n_inlier=<m> inlier_ratio=<%.3f> sigma_rel=<%.4g> range=<%.6g>
       z_ref=<%.6g> created_kf=<k_kf> links=<l> links_refused=<r>

pid the Point id (map_points.txt writes the same integer); ref the reference
keyframe's trajectory index (Frame.id - 1); kp, u, v, level the reference
keypoint (pixels in the frame's own convention, raw or undistorted; level as
_keypoint_sigmas decodes it); frame the promotion frame; converged_at the
first frame at which promotable was ok (== frame under the reference arm);
widest the trajectory index of the inlier measurement with the widest
parallax at the posterior point (promotable's (ii) reading; theta_max_deg is
that parallax); n_meas / n_inlier the measurement counts; inlier_ratio the
Beta mean a / (a + b); sigma_rel = sqrt(sigma2) / mu; range = 1 / mu along
the ray (map units); z_ref the camera-axis depth of X in the reference
keyframe; created_kf the trajectory index of the keyframe whose id is
created_fid; links / links_refused the link counts of this landmark (links
includes the reference link).  Under r2 [DF-PROMOTE] is never printed and the
lines below are; under age none of them is:

    >> [DF-CONFIRM] <the 20 [DF-PROMOTE] fields, same names, formats and order>
       reason=<early|dead|evicted|end> n_distinct=<d> last_inlier=<k_l>
       eligible_failures=<ef> travel_since_match=<%.6g> dormant_frames=<df> first_at_mean=<0|1>
    >> [DF-RETIRE] ref=<k_ref> kp=<i> u=<%.2f> v=<%.2f> level=<L> frame=<k>
       reason=<linked|hopeless|dead|evicted|end> converged=<0|1> converged_at=<k_c>
       theta_max_deg=<%.3f> theta_acc_deg=<%.3f> widest=<k_w> n_meas=<n> n_inlier=<m>
       n_distinct=<d> inlier_ratio=<%.3f> a=<%.3f> b=<%.3f> sigma_rel=<%.4g> range=<%.6g>
       z_ref=<%.6g> last_inlier=<k_l> eligible_failures=<ef> travel_since_match=<%.6g>
       dormant_frames=<df> first_at_mean=<0|1>
    >> [DF-LIFETIME] frame=<k> live=<n> linked_dropped=<x> dormant=<d> skipped=<s>
       eligible=<e> matched=<m> failed=<f> degenerate=<g> hopeless=<h> dead_retired=<dr>
       dead_confirmed=<dc> tentative_new=<tn> early_confirmed=<ec> tentative=<t> first_at_mean=<fm>
    >> [DF-LIFETIME-KF] kf=<k> live_before=<n> evicted_dormant=<a> evicted_active=<b>
       evicted_confirmed=<c> live_after=<m> active=<x> dormant=<y> tentative=<z> cap=<CAP>
       arm_seconds=<%.1f>
    >> [DF-END] frame=<k_last> live=<n> confirmed=<c> retired=<r>

[DF-CONFIRM]: converged_at the first frame converged() was ok, theta_max_deg /
theta_acc_deg / n_distinct converged()'s values at confirmation,
travel_since_match = || C_F - last_match_centre || at confirmation (map
units), last_inlier the last frame with an inlier measurement (-1 when none),
first_at_mean 1 when the seed's first measurement was taken under the
initial-mean disparity reading.  [DF-RETIRE]: converged = tentative at
retirement; theta_* / widest / n_distinct from the last convergence
evaluation (a seed that never matched: promotable evaluated once, theta_max
-1.000, widest -1); converged_at -1 when none; a, b the Beta state; z_ref the
camera-axis depth of X(mu) in the reference keyframe; range = 1 / mu.
[DF-LIFETIME] (before the frame's [DF-FRAME]): live = len(live) at entry;
live == linked_dropped + dormant + skipped + eligible; eligible == matched +
failed; the frame's [DF-FRAME] live == eligible; degenerate = the module's
no_translation + behind + no_disparity (inside failed); tentative = the
tentative seeds after the frame; first_at_mean = eligible seeds classified
under the initial-mean reading this frame.  [DF-LIFETIME-KF] (before
[DF-KEYFRAME]): live_before = len(live) after seeding, before eviction;
live_after == active + dormant; tentative <= live_after; arm_seconds = the
cumulative perf_counter time inside on_frame and on_keyframe.  [DF-END]:
n == c + r.  Errors: pipeline.py wraps the entry points and prints
">> [DF-ERROR] frame <k>: <e>" (">> [DF-ERROR] end: <e>" for on_end); the run
continues.

The trace (L5; arm_params.trace_path non-empty, r2 only): a text file with
one header line "# frame ref kp kp_f level state d_pred d_far d_meas resid
rho_meas tau2 resp desc_dist mu_before sigma_rel_before first_at_mean" and
one space-separated line per MATCHED measurement: the measuring frame, the
seed (ref, kp, level), the matched keypoint kp_f, state 1 when the seed was
tentative before the measurement, d_pred the predicted disparity at the
posterior mean BEFORE the update, d_far the skip quantity used (the initial-
mean value when first_at_mean = 1), d_meas the signed along-line position of
the matched keypoint (kp_F - p_inf) . u with u the unit vector from p_inf
toward p_near (from p_far when the infinity point is behind the camera),
resid = d_meas - d_pred, rho_meas / tau2 / resp / desc_dist from the
Measurement, the posterior before the update; floats %.6g; flushed and
closed at on_end.

Recorded, not changed: a seed's centre and ray are fixed at seeding from the
reference keyframe's pose at that time (04f spec B1); if BA later moves that
keyframe the seed is not refreshed (under O3+ BA accepted 3 of 25 windows and
moves poses by millimetres: second order here).

Conventions (conventions.md#pose-convention): cam_T_world maps world to
camera, world_T_cam is its inverse, the camera centre is world_T_cam[:3, 3].
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, fields

import numpy as np

import depth_filter as df
import depth_filter_lifetime as dfl
from epipolar_refine import REFINE_COUNTER_KEYS
from local_mapping import (
    _MIN_DEPTHS_FOR_MEDIAN,
    LocalMappingParams,
    _camera_depths,
    _keypoint_sigmas,
    _kp_xy,
    cull_points,
    insert_new_points_into_neighbours,
    select_neighbours,
)
from mapping import local_map_track
from Point import Point

_ZERO_BASELINE = 1e-9         # a window baseline at or below this has "not moved" (spec D3)


@dataclass
class DepthFilterArmParams:
    """Constants of the arm's own policy; see the module docstring for each.  Every field is a
    --df_<field> flag of main.py (--df_trace for trace_path, dest df_trace_path)."""
    promote_factor: float = 1.0
    bootstrap_depth_ratio: float = 5.0
    # --- stage 04i (ruling R2) ---
    lifetime: str = "age"
    bootstrap_window: int = 5
    match_error_px: float = 9.5
    death_travel_ratio: float = 0.3
    outlier_eta: float = 0.05
    outlier_conf: float = 0.99
    distinct_baselines: int = 3
    seed_cap: int = 25000
    trace_path: str = ""

    @classmethod
    def from_args(cls, args, prefix="df_"):
        """Build from an argparse namespace whose flags carry ``prefix`` (main.py)."""
        kw = {}
        for f in fields(cls):
            v = getattr(args, prefix + f.name, None)
            if v is not None:
                kw[f.name] = v      # argparse already applied the field's type
        return cls(**kw)


FRAME_COUNTER_KEYS = ("live", "measured", "inlier", "outlier", "no_match", "no_translation", "behind",
                      "no_disparity", "outside_support", "converged", "promoted", "dropped_linked",
                      "links", "links_refused", "inserted")
KEYFRAME_COUNTER_KEYS = ("kf", "pos", "neighbours", "obs_inserted", "culled", "retired", "retired_linked",
                         "retired_converged", "unlinked", "seeds_made", "z_init", "z_source", "live")
LIFETIME_COUNTER_KEYS = ("frame", "live", "linked_dropped", "dormant", "skipped", "eligible", "matched", "failed",
                         "degenerate", "hopeless", "dead_retired", "dead_confirmed", "tentative_new", "early_confirmed",
                         "tentative", "first_at_mean")
LIFETIME_KF_COUNTER_KEYS = ("kf", "live_before", "evicted_dormant", "evicted_active", "evicted_confirmed", "live_after",
                            "active", "dormant", "tentative", "cap", "arm_seconds")
TRACE_COLUMNS = ("frame", "ref", "kp", "kp_f", "level", "state", "d_pred", "d_far", "d_meas", "resid", "rho_meas",
                 "tau2", "resp", "desc_dist", "mu_before", "sigma_rel_before", "first_at_mean")


# --------------------------------------------------------------------------- #
# D3. the keyframe's scene range
# --------------------------------------------------------------------------- #
def scene_range(kf, kf_pos, keyframes, world_map, arm_params, df_params):
    """
    (z_init, source) for the seeds of ``kf`` (position ``kf_pos`` in ``keyframes``),
    in this order:

      links   the median camera-axis depth (positive only) of the landmarks linked
              in kf, when at least _MIN_DEPTHS_FOR_MEDIAN (10) exist;
      map     else the median camera-axis depth in kf over every landmark of the
              map with finite coordinates, positive camera-axis depth in kf and a
              projection inside [0, w) x [0, h) of kf.img_shape, when at least 10
              exist (vectorised: the points are stacked once);
      window  else arm_params.bootstrap_depth_ratio * b_win, b_win the widest
              || C_kf - C_j || over the previous min(arm_params.bootstrap_window,
              kf_pos) keyframes, when kf_pos >= 1 and b_win > 1e-9;
      none    else nan: no seeds for this keyframe.

    The window is arm_params.bootstrap_window under BOTH lifetimes (stage 04i, ruling
    R2: the bootstrap depth window is decoupled from the seed lifetime and fixed at 5,
    the value df_params.max_seed_keyframes gave it in 04h, so the age arm is
    byte-identical).  Camera-axis depth stands in for the range along the ray (the
    difference is a factor cos of the ray angle, at most 1.2 at the image corner, far
    inside the prior's width).
    """
    depths = _camera_depths(kf, world_map)
    if len(depths) >= _MIN_DEPTHS_FOR_MEDIAN:
        return float(np.median(depths)), "links"

    pts = list(getattr(world_map, "points", None) or [])
    if len(pts) >= _MIN_DEPTHS_FOR_MEDIAN:
        Xw = np.array([np.asarray(p.point, dtype=np.float64).reshape(-1)[:3] for p in pts], dtype=np.float64)
        R = np.asarray(kf.cam_T_world[:3, :3], dtype=np.float64)
        t = np.asarray(kf.cam_T_world[:3, 3], dtype=np.float64)
        finite = np.all(np.isfinite(Xw), axis=1)
        xc = np.where(finite[:, None], Xw, 0.0) @ R.T + t
        z = xc[:, 2]
        ok = finite & np.isfinite(z) & (z > 0.0)
        uvw = xc @ np.asarray(kf.K, dtype=np.float64).T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = uvw[:, 0] / uvw[:, 2]
            v = uvw[:, 1] / uvw[:, 2]
        h, w = kf.img_shape[:2]
        ok &= np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u < float(w)) & (v >= 0.0) & (v < float(h))
        if int(ok.sum()) >= _MIN_DEPTHS_FOR_MEDIAN:
            return float(np.median(z[ok])), "map"

    if int(kf_pos) >= 1:
        n_win = min(int(arm_params.bootstrap_window), int(kf_pos))
        C = np.asarray(kf.world_T_cam[:3, 3], dtype=np.float64)
        b_win = 0.0
        for j in range(int(kf_pos) - n_win, int(kf_pos)):
            C_j = np.asarray(keyframes[j][1].world_T_cam[:3, 3], dtype=np.float64)
            b_win = max(b_win, float(np.linalg.norm(C - C_j)))
        if np.isfinite(b_win) and b_win > _ZERO_BASELINE:
            return float(arm_params.bootstrap_depth_ratio) * b_win, "window"

    return float("nan"), "none"


def _parallax_at(X, C_a, C_b):
    """The angle (rad) at X between the rays from the centres C_a and C_b, as promotable reads it."""
    r_a = X - C_a
    r_b = X - C_b
    n_a = float(np.linalg.norm(r_a))
    n_b = float(np.linalg.norm(r_b))
    if n_a <= 0.0 or n_b <= 0.0:
        return -1.0
    cos_a = float(np.dot(r_a, r_b) / (n_a * n_b))
    return math.acos(max(-1.0, min(1.0, cos_a)))


def _widest_view(seed, params):
    """(traj_idx, parallax_rad) of the inlier measurement with the widest parallax at X(mu);
    (-1, -1.0) when there is none."""
    X = np.asarray(df.seed_point(seed), dtype=np.float64)
    widest_idx, widest_rad = -1, -1.0
    for m in df.inlier_measurements(seed, params):
        if m.frame is None:
            continue
        ang = _parallax_at(X, seed.centre, np.asarray(m.frame.world_T_cam[:3, 3], dtype=np.float64))
        if ang > widest_rad:
            widest_rad, widest_idx = ang, int(m.traj_idx)
    return widest_idx, widest_rad


def _z_ref(seed, X):
    R_ref = np.asarray(seed.ref.cam_T_world[:3, :3], dtype=np.float64)
    t_ref = np.asarray(seed.ref.cam_T_world[:3, 3], dtype=np.float64)
    return float(R_ref[2] @ X + t_ref[2])


def _sigma_rel(seed):
    return math.sqrt(max(float(seed.sigma2), 0.0)) / float(seed.mu)


# --------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------- #
class DepthFilterArm:
    """The live seeds, the three parameter sets, the cumulative counters (spec B2)."""

    def __init__(self, df_params=None, arm_params=None, lm_params=None, refine=None):
        self.df_params = df_params if df_params is not None else df.DepthFilterParams()
        self.refine = refine                 # stage 04k: epipolar_refine.RefineParams or None (switch off)
        self.arm_params = arm_params if arm_params is not None else DepthFilterArmParams()
        self.lm_params = lm_params if lm_params is not None else LocalMappingParams()
        self.live = []                       # seeds, in seeding order
        self.totals = dict(df_keyframes=0, df_seeds_made=0, df_promoted=0, df_retired=0,
                           df_obs_inserted=0, df_culled=0, df_frames=0, df_links=0,
                           df_links_refused=0, df_inserted=0, df_dropped_linked=0)
        self.last_promotions = []            # the driver's records of the last on_frame call
        self._sigma_cache = {}               # frame id -> _keypoint_sigmas (keypoints never change)
        # stage 04-m0 (spec D1, D2): when True (the pipeline under --live_map on), every live seed's
        # centre and ray follow its reference keyframe's current pose (age lifetime)
        self.follow_ref_pose = False
        self._ray_cam_cache = {}             # reference frame id -> unit camera rays of its keypoints
        # --- stage 04i (r2) ---
        self.r2 = str(self.arm_params.lifetime) == "r2"
        self.tentative = {}                  # id(seed) -> seed: the public handles (C6)
        self.arm_seconds = 0.0               # cumulative perf_counter time inside on_frame / on_keyframe
        self.ended = False
        self._static = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))   # centres, rays, z_init aligned with live
        self._tab_R, self._tab_t, self._tab_K = [], [], []                   # the frame table of the batched B6
        self._tab_row = {}                                                   # traj idx -> row
        self._tab_frames = []                                                # stage 04y: the frame of each row
        self._tab_arrays = None
        self._trace = None
        if self.r2 and self.arm_params.trace_path:
            self._trace = open(str(self.arm_params.trace_path), "w", encoding="utf-8")
            self._trace.write("# " + " ".join(TRACE_COLUMNS) + "\n")

    # ----------------------------------------------------------------- helpers
    def _sigmas(self, frame):
        s = self._sigma_cache.get(frame.id)
        if s is None:
            s = _keypoint_sigmas(frame)
            self._sigma_cache[frame.id] = s
        return s

    def _promotion_threshold_deg(self):
        return float(self.arm_params.promote_factor) * float(self.df_params.min_parallax_deg)

    def tentative_points(self):
        """C6: [(X (3,), sigma_rel, sigma2, seed)] of the tentative seeds, read from each seed's
        CURRENT posterior (the point's depth stays the filter posterior until confirmation; the
        weight is its variance).  Not Map landmarks: nothing consumes them here."""
        out = []
        for s in self.tentative.values():
            out.append((np.asarray(df.seed_point(s), dtype=np.float64), _sigma_rel(s), float(s.sigma2), s))
        return out

    # ------------------------------------------------ the landmark creation (both lifetimes)
    def _create_landmark(self, s, k, keyframes, world_map, frames, out):
        """The landmark of seed ``s`` at frame ``k`` (age: promotion; r2: confirmation): X =
        seed_point, Point, add_point, the reference link with its guard, the inlier links
        (first link wins), created_fid = the newest keyframe, last_seen_fid, the widest view,
        the link counts on ``out`` and the record of the log line; None (links_refused += 1)
        when the reference link is refused."""
        P = self.df_params
        X = np.asarray(df.seed_point(s), dtype=np.float64)
        desc = np.array(s.desc, copy=True) if s.desc is not None else None
        p = Point(point=X, descriptor=desc)
        world_map.add_point(p, frames=[s.ref])
        if not world_map.link(s.ref, s.kp, p):
            # cannot happen after the linked drop; the guard keeps the map free of half-linked landmarks
            world_map.remove_point(p.id, frames=frames)
            out["links_refused"] += 1
            return None
        n_links, n_refused = 1, 0
        inliers = df.inlier_measurements(s, P)
        widest_idx, widest_rad = -1, -1.0
        for m in inliers:
            if m.frame is None:
                continue
            if world_map.link(m.frame, m.kp, p):
                n_links += 1
            else:
                n_refused += 1
            ang = _parallax_at(X, s.centre, np.asarray(m.frame.world_T_cam[:3, 3], dtype=np.float64))
            if ang > widest_rad:
                widest_rad, widest_idx = ang, int(m.traj_idx)
        p.created_fid = int(keyframes[-1][1].id)
        p.last_seen_fid = int(max(int(fid) for fid in p.observations))
        out["links"] += n_links
        out["links_refused"] += n_refused
        out["promoted"] += 1
        u, v = s.ref.KeyPoints[s.kp].pt
        rec = dict(id=int(p.id), ref=int(s.ref.id) - 1, kp=int(s.kp), u=float(u), v=float(v),
                   level=int(s.level), frame=int(k), widest=widest_idx, theta_max_driver_deg=math.degrees(widest_rad),
                   n_meas=len(s.measurements), n_inlier=len(inliers), inlier_ratio=df.inlier_ratio(s),
                   sigma_rel=_sigma_rel(s), range=1.0 / float(s.mu), z_ref=_z_ref(s, X),
                   created_kf=int(p.created_fid) - 1, links=n_links, links_refused=n_refused, point=p, seed=s)
        return rec

    # ------------------------------------------- stage 04-m0: the seeds follow their keyframe
    def _refresh_ref_poses(self):
        """Spec D2 (04-m0): recompute every live seed's centre and ray from its reference keyframe's
        CURRENT pose with make_seeds' formula (centre = world_T_cam[:3, 3]; ray = r_cam @ R_wc.T,
        r_cam the unit camera ray of the seed's keypoint pixel).  mu, sigma2, a, b, rho_range,
        n_updates and the measurements are unchanged: the inverse range along the reference ray is
        the filter's state, and the ray follows its keyframe when bundle adjustment moves it."""
        refs = {}
        for s in self.live:
            refs.setdefault(id(s.ref), (s.ref, []))[1].append(s)
        cache = {}
        for ref, seeds in refs.values():
            rays_cam = self._ray_cam_cache.get(ref.id)
            if rays_cam is None:
                rays_cam = df._unit_rays_cam(ref.K, df._kp_xy(ref))
            cache[ref.id] = rays_cam
            W = np.asarray(ref.world_T_cam, dtype=np.float64)
            R_wc = W[:3, :3]
            centre = np.array(W[:3, 3], dtype=np.float64)
            for s in seeds:
                s.centre = centre.copy()
                s.ray = rays_cam[s.kp] @ R_wc.T
        self._ray_cam_cache = cache

    # ------------------------------------------------------------ C. one frame
    def on_frame(self, k, frame, keyframes, world_map, frames):
        """Measurement (C1), promotion (C2), insertion of the new landmarks (C3), the
        [DF-FRAME] line (C4).  Returns the line's counters plus 'promotions' (and, under
        r2, 'lifetime')."""
        if self.r2:
            return self._on_frame_r2(k, frame, keyframes, world_map, frames)
        if self.follow_ref_pose:
            self._refresh_ref_poses()
        P = self.df_params
        k = int(k)
        c = df.update_seeds(self.live, frame, k, P, refine=self.refine)
        thr_deg = self._promotion_threshold_deg()
        out = {key: 0 for key in FRAME_COUNTER_KEYS}
        for key in ("measured", "inlier", "outlier", "no_match", "no_translation", "behind",
                    "no_disparity", "outside_support"):
            out[key] = int(c[key])
        out["live"] = int(c["seeds"])

        new_points = []       # (Point, reference frame): the partner insert_new_points_into_neighbours skips
        records = []
        kept = []
        for s in self.live:
            # (a) the reference keypoint got linked by an insertion since seeding: no landmark
            if s.kp in s.ref.keypoint_to_point:
                out["dropped_linked"] += 1
                continue
            # (b) the module's test; the first convergence is the driver's record
            ok, theta_acc, theta_max = df.promotable(s, P)
            if ok:
                out["converged"] += 1
                if getattr(s, "first_ok_traj", None) is None:
                    s.first_ok_traj = k
            # (c) the timing clause
            if not (ok and theta_acc >= thr_deg):
                kept.append(s)
                continue
            rec = self._create_landmark(s, k, keyframes, world_map, frames, out)
            if rec is None:
                continue
            rec.update(converged_at=int(s.first_ok_traj), theta_max_deg=float(theta_max), theta_acc_deg=float(theta_acc))
            new_points.append((rec["point"], s.ref))
            records.append(rec)
            print(
                f"  >> [DF-PROMOTE] id={rec['id']} ref={rec['ref']} kp={rec['kp']} u={rec['u']:.2f} v={rec['v']:.2f} "
                f"level={rec['level']} frame={rec['frame']} converged_at={rec['converged_at']} widest={rec['widest']} "
                f"theta_max_deg={rec['theta_max_deg']:.3f} theta_acc_deg={rec['theta_acc_deg']:.3f} "
                f"n_meas={rec['n_meas']} n_inlier={rec['n_inlier']} inlier_ratio={rec['inlier_ratio']:.3f} "
                f"sigma_rel={rec['sigma_rel']:.4g} range={rec['range']:.6g} z_ref={rec['z_ref']:.6g} "
                f"created_kf={rec['created_kf']} links={rec['links']} links_refused={rec['links_refused']}"
            )
        self.live = kept

        # C3: the new landmarks into the recent keyframes (the slot's insertion rule)
        if new_points:
            out["inserted"] = self._insert_recent(new_points, keyframes, world_map)

        self._print_frame_line(k, out)
        if self.refine is not None:
            self._print_refine_line(k, c)
        T = self.totals
        T["df_frames"] += 1
        T["df_promoted"] += out["promoted"]
        T["df_links"] += out["links"]
        T["df_links_refused"] += out["links_refused"]
        T["df_inserted"] += out["inserted"]
        T["df_dropped_linked"] += out["dropped_linked"]
        self.last_promotions = records
        out["frame"] = k
        out["promotions"] = records
        return out

    def _insert_recent(self, new_points, keyframes, world_map):
        """C3 / C7: insert_new_points_into_neighbours over the last max_neighbours keyframes."""
        n_nb = min(int(self.lm_params.max_neighbours), len(keyframes))
        start = len(keyframes) - n_nb
        neighbours = [(pos, keyframes[pos][0], keyframes[pos][1]) for pos in range(start, len(keyframes))]
        sigma_cache = {fr.id: self._sigmas(fr) for _, _, fr in neighbours}
        self._sigma_cache = {fid: sigma_cache[fid] for fid in sigma_cache}     # keep the recent keyframes only
        return int(insert_new_points_into_neighbours(new_points, neighbours, world_map, self.lm_params, sigma_cache))

    @staticmethod
    def _print_frame_line(k, out):
        print(
            f"  >> [DF-FRAME] frame={k} live={out['live']} measured={out['measured']} inlier={out['inlier']} "
            f"outlier={out['outlier']} no_match={out['no_match']} no_translation={out['no_translation']} "
            f"behind={out['behind']} no_disparity={out['no_disparity']} outside_support={out['outside_support']} "
            f"converged={out['converged']} promoted={out['promoted']} dropped_linked={out['dropped_linked']} "
            f"links={out['links']} links_refused={out['links_refused']} inserted={out['inserted']}"
        )

    @staticmethod
    def _print_refine_line(k, c):
        """Stage 04k: the [DF-REFINE] line from update_seeds' refinement counters (0 when absent)."""
        r = {key: int((c or {}).get(key, 0)) for key in REFINE_COUNTER_KEYS}
        print(
            f"  >> [DF-REFINE] frame={k} attempted={r['refine_attempted']} refined={r['refine_ok']} "
            f"ambiguous={r['refine_ambiguous']} low_peak={r['refine_low_peak']} edge={r['refine_edge']} "
            f"oob={r['refine_oob']} no_image={r['refine_no_image']} degenerate={r['refine_degenerate']}"
        )

    # --------------------------------------------------------- D. one keyframe
    def on_keyframe(self, k, kf, keyframes, world_map, frames, stats=None):
        """A4(a) insertion and A5 culling (D1), retirement (D2), the scene range (D3),
        seeding (D4), the [DF-KEYFRAME] line and the stats (D5).  Returns the line's
        counters."""
        if self.r2:
            return self._on_keyframe_r2(k, kf, keyframes, world_map, frames, stats)
        if self.follow_ref_pose:
            self._refresh_ref_poses()
        P = self.df_params
        k = int(k)
        kf_pos, neighbours, obs_inserted, culled = self._keyframe_step_d1(kf, keyframes, world_map, frames)

        # D2: retirement, with the ruling's accounting of the converged-but-never-promoted seeds
        limit = kf_pos - int(P.max_seed_keyframes)
        retired_linked = 0
        retired_converged = 0
        for s in self.live:
            if s.kp in s.ref.keypoint_to_point:
                retired_linked += 1
            elif s.ref_kf_pos <= limit and df.promotable(s, P)[0]:
                retired_converged += 1
        self.live, n_retired = df.retire_seeds(self.live, kf_pos, P)

        # D3, D4
        z_init, source = scene_range(kf, kf_pos, keyframes, world_map, self.arm_params, P)
        seeds = df.make_seeds(kf, kf_pos, z_init, P) if source != "none" else []
        self.live.extend(seeds)
        n_kp = len(getattr(kf, "KeyPoints", None) or [])
        linked = kf.keypoint_to_point
        unlinked = sum(1 for i in range(n_kp) if i not in linked)

        out = dict(kf=k, pos=int(kf_pos), neighbours=len(neighbours), obs_inserted=obs_inserted, culled=culled,
                   retired=int(n_retired), retired_linked=retired_linked, retired_converged=retired_converged,
                   unlinked=int(unlinked), seeds_made=len(seeds), z_init=float(z_init), z_source=source,
                   live=len(self.live))
        self._print_keyframe_line(out)
        self._keyframe_totals(out, stats)
        return out

    def _keyframe_step_d1(self, kf, keyframes, world_map, frames):
        """D1: the keyframe's position, A4(a) exactly as process_new_keyframe makes it, then A5."""
        L = self.lm_params
        kf_pos = len(keyframes) - 1
        for pos in range(len(keyframes) - 1, -1, -1):
            if keyframes[pos][1] is kf:
                kf_pos = pos
                break
        neighbours = select_neighbours(kf_pos, kf, keyframes, L)
        obs_inserted = 0
        if neighbours:
            obs_inserted = int(local_map_track(
                kf, world_map, [nb for _, _, nb in neighbours],
                search_radius=float(L.insert_search_radius_px),
                max_desc_dist=float(L.max_desc_dist),
                max_depth_ratio=None,
            ))
        culled = int(cull_points(kf_pos, keyframes, world_map, frames, L))
        return kf_pos, neighbours, obs_inserted, culled

    @staticmethod
    def _print_keyframe_line(out):
        print(
            f"  >> [DF-KEYFRAME] kf={out['kf']} pos={out['pos']} neighbours={out['neighbours']} "
            f"obs_inserted={out['obs_inserted']} culled={out['culled']} retired={out['retired']} "
            f"retired_linked={out['retired_linked']} retired_converged={out['retired_converged']} "
            f"unlinked={out['unlinked']} seeds_made={out['seeds_made']} z_init={out['z_init']:.6g} "
            f"z_source={out['z_source']} live={out['live']}"
        )

    def _keyframe_totals(self, out, stats):
        T = self.totals
        T["df_keyframes"] += 1
        T["df_seeds_made"] += int(out["seeds_made"])
        T["df_retired"] += int(out["retired"])
        T["df_obs_inserted"] += int(out["obs_inserted"])
        T["df_culled"] += int(out["culled"])
        if stats is not None:
            for key in ("df_keyframes", "df_seeds_made", "df_promoted", "df_retired", "df_obs_inserted", "df_culled"):
                stats[key] = int(T[key])

    # ------------------------------------------------------------ E. the end of the run
    def on_end(self, k_last, keyframes, world_map, frames):
        """Stage 04i spec E1.  Under 'age': nothing (returns {}).  Under r2: the end of the
        recording is the end of observability: every tentative seed still converged is
        CONFIRMED reason=end, every other live seed retired reason=end, then [DF-END]."""
        if not self.r2:
            return {}
        return self._on_end_r2(k_last, keyframes, world_map, frames)

    # =========================================================================== #
    # stage 04i: the r2 lifetime
    # =========================================================================== #
    # ---------------------------------------------------- the aligned static arrays
    def _filter_live(self, keep):
        """Keep the seeds where the bool array ``keep`` is True (the static arrays follow)."""
        keep = np.asarray(keep, dtype=bool)
        if keep.all():
            return
        self.live = [s for s, kp in zip(self.live, keep.tolist()) if kp]
        cen, ray, zi = self._static
        self._static = (cen[keep], ray[keep], zi[keep])

    def _extend_live(self, seeds):
        if not seeds:
            return
        n = len(seeds)
        cen = np.array([s.centre for s in seeds], dtype=np.float64).reshape(n, 3)
        ray = np.array([s.ray for s in seeds], dtype=np.float64).reshape(n, 3)
        zi = np.fromiter((float(s.z_init) for s in seeds), dtype=np.float64, count=n)
        c0, r0, z0 = self._static
        self._static = (np.concatenate([c0, cen]), np.concatenate([r0, ray]), np.concatenate([z0, zi]))
        self.live.extend(seeds)

    def _live_arrays(self):
        n = len(self.live)
        cen, ray, zi = self._static
        assert len(zi) == n
        mu = np.fromiter((s.mu for s in self.live), dtype=np.float64, count=n)
        sigma2 = np.fromiter((s.sigma2 for s in self.live), dtype=np.float64, count=n)
        return cen, ray, mu, sigma2, zi

    # ---------------------------------------------------- the frame table (batched B6)
    def _register_frame(self, k, frame):
        row = self._tab_row.get(int(k))
        if row is None:
            row = len(self._tab_R)
            self._tab_row[int(k)] = row
            self._tab_R.append(np.asarray(frame.cam_T_world[:3, :3], dtype=np.float64))
            self._tab_t.append(np.asarray(frame.cam_T_world[:3, 3], dtype=np.float64))
            self._tab_K.append(np.asarray(frame.K, dtype=np.float64))
            self._tab_frames.append(frame)
            self._tab_arrays = None
        return row

    def _table(self):
        if self._tab_arrays is None:
            self._tab_arrays = (np.array(self._tab_R).reshape(-1, 3, 3), np.array(self._tab_t).reshape(-1, 3),
                                np.array(self._tab_K).reshape(-1, 3, 3))
        return self._tab_arrays

    def _refresh_rows(self, rows):
        """
        Stage 04y (pose vintage, 04i B6): every row in ``rows`` equals its registered frame's
        current cam_T_world and K, bitwise, so the batched count reads the poses the per-seed
        path (depth_filter_lifetime.distinct_baselines) reads.  A row is rewritten only when
        its stored values differ from the frame's; with no pose change nothing is touched.
        """
        R, t, K = self._table()
        for r in np.unique(np.asarray(rows, dtype=np.int64)).tolist():
            f = self._tab_frames[r]
            T = f.cam_T_world
            Rn = np.asarray(T[:3, :3], dtype=np.float64)
            tn = np.asarray(T[:3, 3], dtype=np.float64)
            Kn = np.asarray(f.K, dtype=np.float64)
            if (R[r].tobytes() == np.ascontiguousarray(Rn).tobytes()
                    and t[r].tobytes() == np.ascontiguousarray(tn).tobytes()
                    and K[r].tobytes() == np.ascontiguousarray(Kn).tobytes()):
                continue
            R[r], t[r], K[r] = Rn, tn, Kn
            self._tab_R[r], self._tab_t[r], self._tab_K[r] = Rn.copy(), tn.copy(), Kn.copy()

    def _distinct_batch(self, seeds):
        """n_distinct (B6) for a list of seeds at once: the pairs (seed, inlier frame row)."""
        n = len(seeds)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        counts = np.fromiter((len(s.inl_rows) for s in seeds), dtype=np.int64, count=n)
        total = int(counts.sum())
        if total == 0:
            return np.zeros(n, dtype=np.int64)
        rows = np.fromiter(itertools.chain.from_iterable(s.inl_rows for s in seeds), dtype=np.int64, count=total)
        group = np.repeat(np.arange(n), counts)
        X = np.array([df.seed_point(s) for s in seeds], dtype=np.float64).reshape(n, 3)
        rays = np.array([s.ray for s in seeds], dtype=np.float64).reshape(n, 3)
        self._refresh_rows(rows)
        R, t, K = self._table()
        disp = dfl.point_disparities(X[group], rays[group], R[rows], t[rows], K[rows])
        return dfl.distinct_counts(disp, group, n, float(self.arm_params.match_error_px))

    # ---------------------------------------------------- the per-seed state (C0)
    @staticmethod
    def _init_seed_state(s, k, z_init):
        s.z_init = float(z_init)
        s.state = "active"
        s.dormant_since = None
        s.dormant_frames = 0
        s.tentative = False
        s.converged_at = None
        s.eligible_failures = 0
        s.last_match_centre = np.array(s.centre, dtype=np.float64, copy=True)
        s.last_match_traj = int(k)
        s.last_inlier_traj = -1
        s.first_at_mean = False
        s.n_distinct = 0
        s.theta_acc = 0.0
        s.theta_max = -1.0
        s.inl_rows = []

    def _converged(self, s):
        """converged() with the batched B6 skipped: one seed."""
        return dfl.converged(s, self.df_params, self.arm_params)

    def _drop_handle(self, s):
        self.tentative.pop(id(s), None)

    # ---------------------------------------------------- retirement and confirmation (L3, L4)
    def _retire(self, s, k, C_F, reason):
        """[DF-RETIRE] for seed ``s`` at frame ``k`` (camera centre ``C_F``, None at the end
        when no pose is at hand); removes the handle.  The caller removes the seed."""
        P = self.df_params
        converged = 1 if s.tentative else 0
        self._drop_handle(s)
        if s.n_updates == 0 and not s.measurements:
            _, theta_acc, theta_max = df.promotable(s, P)
            widest = -1
            n_distinct = 0
        else:
            theta_acc, theta_max = float(s.theta_acc), float(s.theta_max)
            widest, _ = _widest_view(s, P)
            n_distinct = int(s.n_distinct)
        X = np.asarray(df.seed_point(s), dtype=np.float64)
        u, v = s.ref.KeyPoints[s.kp].pt
        travel = dfl.travel_since_match(s, C_F) if C_F is not None else float("nan")
        conv_at = int(s.converged_at) if s.converged_at is not None else -1
        n_inl = len(df.inlier_measurements(s, P))
        print(
            f"  >> [DF-RETIRE] ref={int(s.ref.id) - 1} kp={int(s.kp)} u={float(u):.2f} v={float(v):.2f} level={int(s.level)} "
            f"frame={int(k)} reason={reason} converged={converged} converged_at={conv_at} "
            f"theta_max_deg={float(theta_max):.3f} theta_acc_deg={float(theta_acc):.3f} widest={int(widest)} "
            f"n_meas={len(s.measurements)} n_inlier={n_inl} n_distinct={int(n_distinct)} "
            f"inlier_ratio={df.inlier_ratio(s):.3f} a={float(s.a):.3f} b={float(s.b):.3f} sigma_rel={_sigma_rel(s):.4g} "
            f"range={1.0 / float(s.mu):.6g} z_ref={_z_ref(s, X):.6g} last_inlier={int(s.last_inlier_traj)} "
            f"eligible_failures={int(s.eligible_failures)} travel_since_match={travel:.6g} "
            f"dormant_frames={int(s.dormant_frames)} first_at_mean={1 if s.first_at_mean else 0}"
        )

    def _confirm(self, s, k, C_F, reason, keyframes, world_map, frames, out, ok_vals):
        """C5: the landmark of a converged tentative seed and its [DF-CONFIRM] line; returns
        the record (None when the reference link was refused: the seed is then retired
        reason=linked so that it still ends exactly one way)."""
        ok, theta_acc, theta_max, n_distinct = ok_vals
        s.theta_acc, s.theta_max, s.n_distinct = float(theta_acc), float(theta_max), int(n_distinct)
        rec = self._create_landmark(s, k, keyframes, world_map, frames, out)
        if rec is None:
            self._retire(s, k, C_F, "linked")
            return None
        self._drop_handle(s)
        travel = dfl.travel_since_match(s, C_F) if C_F is not None else float("nan")
        rec.update(converged_at=int(s.converged_at) if s.converged_at is not None else int(k),
                   theta_max_deg=float(theta_max), theta_acc_deg=float(theta_acc), reason=reason,
                   n_distinct=int(n_distinct), last_inlier=int(s.last_inlier_traj),
                   eligible_failures=int(s.eligible_failures), travel_since_match=travel,
                   dormant_frames=int(s.dormant_frames), first_at_mean=1 if s.first_at_mean else 0)
        print(
            f"  >> [DF-CONFIRM] id={rec['id']} ref={rec['ref']} kp={rec['kp']} u={rec['u']:.2f} v={rec['v']:.2f} "
            f"level={rec['level']} frame={rec['frame']} converged_at={rec['converged_at']} widest={rec['widest']} "
            f"theta_max_deg={rec['theta_max_deg']:.3f} theta_acc_deg={rec['theta_acc_deg']:.3f} "
            f"n_meas={rec['n_meas']} n_inlier={rec['n_inlier']} inlier_ratio={rec['inlier_ratio']:.3f} "
            f"sigma_rel={rec['sigma_rel']:.4g} range={rec['range']:.6g} z_ref={rec['z_ref']:.6g} "
            f"created_kf={rec['created_kf']} links={rec['links']} links_refused={rec['links_refused']} "
            f"reason={rec['reason']} n_distinct={rec['n_distinct']} last_inlier={rec['last_inlier']} "
            f"eligible_failures={rec['eligible_failures']} travel_since_match={rec['travel_since_match']:.6g} "
            f"dormant_frames={rec['dormant_frames']} first_at_mean={rec['first_at_mean']}"
        )
        return rec

    # ---------------------------------------------------- the trace (L5)
    def _trace_line(self, k, s, m, tentative_before, d_pred, d_far, d_meas, mu_before, sigma_rel_before, fam):
        if self._trace is None:
            return
        resid = d_meas - d_pred
        self._trace.write(
            f"{int(k)} {int(s.ref.id) - 1} {int(s.kp)} {int(m.kp)} {int(s.level)} {1 if tentative_before else 0} "
            f"{d_pred:.6g} {d_far:.6g} {d_meas:.6g} {resid:.6g} {m.rho:.6g} {m.tau2:.6g} {m.resp:.6g} {m.desc_dist:.6g} "
            f"{mu_before:.6g} {sigma_rel_before:.6g} {1 if fam else 0}\n"
        )

    # ---------------------------------------------------- C. one frame under r2
    def _on_frame_r2(self, k, frame, keyframes, world_map, frames):
        t_start = time.perf_counter()
        P, A = self.df_params, self.arm_params
        k = int(k)
        thr_deg = self._promotion_threshold_deg()
        thr_resp = float(P.responsibility_inlier)
        N = int(A.distinct_baselines)
        C_F = np.asarray(frame.world_T_cam[:3, 3], dtype=np.float64)
        out = {key: 0 for key in FRAME_COUNTER_KEYS}
        lt = {key: 0 for key in LIFETIME_COUNTER_KEYS}
        lt["frame"] = k
        lt["live"] = len(self.live)
        records = []
        new_points = []

        # C1: the linked drop
        keep = np.ones(len(self.live), dtype=bool)
        for i, s in enumerate(self.live):
            if s.kp in s.ref.keypoint_to_point:
                keep[i] = False
                lt["linked_dropped"] += 1
                out["dropped_linked"] += 1
                self._retire(s, k, C_F, "linked")
        self._filter_live(keep)

        # C2: classification
        arrays = self._live_arrays()
        seg = dfl.credible_segments(self.live, frame, float(P.search_band_sigmas), arrays=arrays)
        state, d_skip, d_mean, fam = dfl.classify(self.live, frame, A, P, segments=seg)
        st_list = state.tolist()
        fam_list = fam.tolist()
        eligible = []
        elig_idx = []
        for i, s in enumerate(self.live):
            st = st_list[i]
            if st == dfl.STATE_DORMANT:
                if s.state != "dormant":
                    s.state = "dormant"
                    s.dormant_since = k
                s.dormant_frames += 1
                lt["dormant"] += 1
            elif st == dfl.STATE_SKIPPED:
                s.state = "active"
                s.dormant_since = None
                lt["skipped"] += 1
            else:
                s.state = "active"
                s.dormant_since = None
                eligible.append(s)
                elig_idx.append(i)
                if fam_list[i]:
                    lt["first_at_mean"] += 1
        lt["eligible"] = len(eligible)

        # C3: the measurement of the eligible sub-list
        n_before = [s.n_updates for s in eligible]
        if self._trace is not None:
            tent_before = [s.tentative for s in eligible]
            mu_before = [s.mu for s in eligible]
            srel_before = [_sigma_rel(s) for s in eligible]
        c = df.update_seeds(eligible, frame, k, P, refine=self.refine)
        for key in ("measured", "inlier", "outlier", "no_match", "no_translation", "behind",
                    "no_disparity", "outside_support"):
            out[key] = int(c[key])
        out["live"] = int(c["seeds"])
        lt["degenerate"] = int(c["no_translation"]) + int(c["behind"]) + int(c["no_disparity"])
        row = self._register_frame(k, frame)
        xy_F = _kp_xy(frame) if self._trace is not None else None
        matched = []
        remove = set()
        for j, s in enumerate(eligible):
            i = elig_idx[j]
            if s.n_updates > n_before[j]:
                lt["matched"] += 1
                m = s.measurements[-1]
                s.eligible_failures = 0
                s.last_match_centre = C_F.copy()
                s.last_match_traj = k
                if m.resp >= thr_resp:
                    s.last_inlier_traj = k
                    s.inl_rows.append(row)
                if fam_list[i] and s.n_updates == 1:
                    s.first_at_mean = True
                if self._trace is not None:
                    p_inf = seg["p_inf"][i]
                    origin = p_inf if seg["inf_ok"][i] else seg["p_far"][i]
                    u_vec = seg["p_near"][i] - origin
                    nu = float(np.linalg.norm(u_vec))
                    d_meas = float(np.dot(xy_F[m.kp] - origin, u_vec) / nu) if nu > 0.0 else float("nan")
                    self._trace_line(k, s, m, tent_before[j], float(d_mean[i]), float(d_skip[i]), d_meas,
                                     float(mu_before[j]), float(srel_before[j]), fam_list[i])
                matched.append(s)
            else:
                lt["failed"] += 1
                s.b = float(s.b) + 1.0          # an eligible no-match is an outlier observation (V&H 4.2)
                s.eligible_failures += 1
                if dfl.hopeless(s.a, s.b, float(A.outlier_eta), float(A.outlier_conf)):
                    lt["hopeless"] += 1
                    self._retire(s, k, C_F, "hopeless")
                    remove.add(id(s))
                elif dfl.dead(s, C_F, A):
                    vals = self._converged(s) if s.tentative else None
                    if vals is not None and vals[0]:
                        rec = self._confirm(s, k, C_F, "dead", keyframes, world_map, frames, out, vals)
                        if rec is not None:
                            lt["dead_confirmed"] += 1
                            records.append(rec)
                            new_points.append((rec["point"], s.ref))
                    else:
                        lt["dead_retired"] += 1
                        self._retire(s, k, C_F, "dead")
                    remove.add(id(s))

        # C4: convergence of the matched seeds
        n_dist = self._distinct_batch(matched).tolist()
        for j, s in enumerate(matched):
            ok_p, theta_acc, theta_max = df.promotable(s, P)
            nd = int(n_dist[j])
            ok = bool(ok_p and nd >= N)
            s.theta_acc, s.theta_max, s.n_distinct = float(theta_acc), float(theta_max), nd
            if ok:
                out["converged"] += 1
                if not s.tentative:
                    s.tentative = True
                    if s.converged_at is None:
                        s.converged_at = k
                    self.tentative[id(s)] = s
                    lt["tentative_new"] += 1
            elif s.tentative:
                s.tentative = False
                self._drop_handle(s)
            if s.tentative and theta_acc >= thr_deg:
                rec = self._confirm(s, k, C_F, "early", keyframes, world_map, frames, out, (ok, theta_acc, theta_max, nd))
                if rec is not None:
                    lt["early_confirmed"] += 1
                    records.append(rec)
                    new_points.append((rec["point"], s.ref))
                remove.add(id(s))
        if remove:
            self._filter_live(np.fromiter((id(s) not in remove for s in self.live), dtype=bool, count=len(self.live)))
        lt["tentative"] = len(self.tentative)

        # C7: the new landmarks into the recent keyframes, then the two lines
        if new_points:
            out["inserted"] = self._insert_recent(new_points, keyframes, world_map)
        self._print_lifetime_line(lt)
        self._print_frame_line(k, out)
        if self.refine is not None:
            self._print_refine_line(k, c)
        T = self.totals
        T["df_frames"] += 1
        T["df_promoted"] += out["promoted"]
        T["df_retired"] += lt["linked_dropped"] + lt["hopeless"] + lt["dead_retired"]
        T["df_links"] += out["links"]
        T["df_links_refused"] += out["links_refused"]
        T["df_inserted"] += out["inserted"]
        T["df_dropped_linked"] += out["dropped_linked"]
        self.last_promotions = records
        out["frame"] = k
        out["promotions"] = records
        out["lifetime"] = lt
        self.arm_seconds += time.perf_counter() - t_start
        return out

    @staticmethod
    def _print_lifetime_line(lt):
        print(
            f"  >> [DF-LIFETIME] frame={lt['frame']} live={lt['live']} linked_dropped={lt['linked_dropped']} "
            f"dormant={lt['dormant']} skipped={lt['skipped']} eligible={lt['eligible']} matched={lt['matched']} "
            f"failed={lt['failed']} degenerate={lt['degenerate']} hopeless={lt['hopeless']} "
            f"dead_retired={lt['dead_retired']} dead_confirmed={lt['dead_confirmed']} "
            f"tentative_new={lt['tentative_new']} early_confirmed={lt['early_confirmed']} "
            f"tentative={lt['tentative']} first_at_mean={lt['first_at_mean']}"
        )

    # ---------------------------------------------------- D. one keyframe under r2
    def _on_keyframe_r2(self, k, kf, keyframes, world_map, frames, stats=None):
        t_start = time.perf_counter()
        P, A = self.df_params, self.arm_params
        k = int(k)
        C_kf = np.asarray(kf.world_T_cam[:3, 3], dtype=np.float64)
        kf_pos, neighbours, obs_inserted, culled = self._keyframe_step_d1(kf, keyframes, world_map, frames)

        # D2: retirement by linkage only
        retired_linked = 0
        retired_converged = 0
        keep = np.ones(len(self.live), dtype=bool)
        for i, s in enumerate(self.live):
            if s.kp in s.ref.keypoint_to_point:
                keep[i] = False
                retired_linked += 1
                if s.tentative:
                    retired_converged += 1
                self._retire(s, k, C_kf, "linked")
        self._filter_live(keep)
        n_retired = retired_linked

        # D3, D4
        z_init, source = scene_range(kf, kf_pos, keyframes, world_map, A, P)
        seeds = df.make_seeds(kf, kf_pos, z_init, P) if source != "none" else []
        for s in seeds:
            self._init_seed_state(s, k, z_init)
        n_old = len(self.live)
        self._extend_live(seeds)
        live_before = len(self.live)

        # D5: the cap (the new keyframe's own seeds are exempt)
        ev = dict(evicted_dormant=0, evicted_active=0, evicted_confirmed=0)
        n_over = live_before - int(A.seed_cap)
        new_points = []
        out_c = dict(links=0, links_refused=0, promoted=0)
        if n_over > 0 and n_old > 0:
            order = dfl.eviction_order(self.live[:n_old], k)
            victims = order[:min(n_over, n_old)]
            remove = set()
            for s in victims:
                remove.add(id(s))
                vals = self._converged(s) if s.tentative else None
                if vals is not None and vals[0]:
                    rec = self._confirm(s, k, C_kf, "evicted", keyframes, world_map, frames, out_c, vals)
                    if rec is not None:
                        ev["evicted_confirmed"] += 1
                        new_points.append((rec["point"], s.ref))
                        continue
                    n_retired += 1
                    continue
                if s.tentative:
                    retired_converged += 1
                if s.state == "dormant":
                    ev["evicted_dormant"] += 1
                else:
                    ev["evicted_active"] += 1
                n_retired += 1
                self._retire(s, k, C_kf, "evicted")
            self._filter_live(np.fromiter((id(s) not in remove for s in self.live), dtype=bool, count=len(self.live)))
        if new_points:
            self.totals["df_inserted"] += self._insert_recent(new_points, keyframes, world_map)
        self.totals["df_promoted"] += out_c["promoted"]
        self.totals["df_links"] += out_c["links"]
        self.totals["df_links_refused"] += out_c["links_refused"]

        n_kp = len(getattr(kf, "KeyPoints", None) or [])
        linked = kf.keypoint_to_point
        unlinked = sum(1 for i in range(n_kp) if i not in linked)
        n_dormant = sum(1 for s in self.live if s.state == "dormant")
        self.arm_seconds += time.perf_counter() - t_start
        ltk = dict(kf=k, live_before=live_before, evicted_dormant=ev["evicted_dormant"], evicted_active=ev["evicted_active"],
                   evicted_confirmed=ev["evicted_confirmed"], live_after=len(self.live), active=len(self.live) - n_dormant,
                   dormant=n_dormant, tentative=len(self.tentative), cap=int(A.seed_cap), arm_seconds=self.arm_seconds)
        print(
            f"  >> [DF-LIFETIME-KF] kf={ltk['kf']} live_before={ltk['live_before']} evicted_dormant={ltk['evicted_dormant']} "
            f"evicted_active={ltk['evicted_active']} evicted_confirmed={ltk['evicted_confirmed']} live_after={ltk['live_after']} "
            f"active={ltk['active']} dormant={ltk['dormant']} tentative={ltk['tentative']} cap={ltk['cap']} "
            f"arm_seconds={ltk['arm_seconds']:.1f}"
        )
        out = dict(kf=k, pos=int(kf_pos), neighbours=len(neighbours), obs_inserted=obs_inserted, culled=culled,
                   retired=int(n_retired), retired_linked=retired_linked, retired_converged=retired_converged,
                   unlinked=int(unlinked), seeds_made=len(seeds), z_init=float(z_init), z_source=source,
                   live=len(self.live))
        self._print_keyframe_line(out)
        self._keyframe_totals(out, stats)
        out["lifetime_kf"] = ltk
        return out

    # ---------------------------------------------------- E. the end of the run under r2
    def _on_end_r2(self, k_last, keyframes, world_map, frames):
        t_start = time.perf_counter()
        k_last = int(k_last)
        C_F = None
        if frames is not None and 0 <= k_last < len(frames) and getattr(frames[k_last], "world_T_cam", None) is not None:
            C_F = np.asarray(frames[k_last].world_T_cam[:3, 3], dtype=np.float64)
        n = len(self.live)
        out = dict(links=0, links_refused=0, promoted=0)
        records = []
        new_points = []
        confirmed = retired = 0
        for s in list(self.live):
            vals = self._converged(s) if s.tentative else None
            if vals is not None and vals[0] and keyframes:
                rec = self._confirm(s, k_last, C_F, "end", keyframes, world_map, frames, out, vals)
                if rec is not None:
                    confirmed += 1
                    records.append(rec)
                    new_points.append((rec["point"], s.ref))
                    continue
                retired += 1
                continue
            retired += 1
            self._retire(s, k_last, C_F, "end")
        if new_points:
            self.totals["df_inserted"] += self._insert_recent(new_points, keyframes, world_map)
        self.totals["df_promoted"] += out["promoted"]
        self.totals["df_retired"] += retired
        self.totals["df_links"] += out["links"]
        self.totals["df_links_refused"] += out["links_refused"]
        self.live = []
        self._static = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        self.tentative = {}
        self.ended = True
        if self._trace is not None:
            self._trace.flush()
            self._trace.close()
            self._trace = None
        print(f"  >> [DF-END] frame={k_last} live={n} confirmed={confirmed} retired={retired}")
        self.arm_seconds += time.perf_counter() - t_start
        self.last_promotions = records
        return dict(frame=k_last, live=n, confirmed=confirmed, retired=retired, promotions=records)
