"""
Stage 04k (ruling S5, 2026-09-22, workshop/20): sub-pixel refinement of a
descriptor match along the epipolar line, the one bounded method addition the
ruling grants.  One function, refine_along_epipolar, used by both point
sources (local_mapping's pair mapper and depth_filter's measurement) behind
one switch, --epi_refine, off by default.  Sources: Vogiatzis and Hernandez,
sections 3 and 4.2 (the sub-pixel maximum of a correlation score along the
epipolar line; papers/vogiatzis-hernandez-video-mvs.md), SVO section IV.B-C
(papers/svo.md#depth-filter).  The warped, level-aware form of SVO IV.B is NOT
built (S5).

S-a WHAT IS REFINED.  A descriptor match (reference keypoint x_R in image R,
    matched keypoint x_T in image T) gives a position x_T whose error along
    the epipolar line is the depth defect 04i located.  The refinement keeps
    the match's identity and replaces x_T by the position on the epipolar line
    through x_T (unit direction u in T, pinhole pixels) that maximises the
    zero-mean normalized cross-correlation (ZNCC) of a square patch around x_R
    in R with a patch around x_T + s u in T, searched at integer s in
    [-search_px, search_px], with a parabola through the maximum and its two
    neighbours for the sub-pixel position.  It moves x_T only along u, by at
    most search_px + 0.5 px.

S-b THE CONSTANTS, FROZEN (the workshop prototype's, workshop/20 addendum 2):
      patch_half = 4        a 9 x 9 patch
      search_px = 6         13 samples, 1 px apart
      min_peak = 0.6        a best ZNCC below it is no match
      ambiguity_ratio = 0.9 a second local maximum above 0.9 of the best is a
                            repeat: the measurement is dropped, not guessed.
    Regime reading, stated not tuned: 6 px covers SIFT's correct-class
    along-line scatter on this hall (|along| p50 1.30 to 1.79 px at octaves -1
    and 0, which hold 91 percent of its matches; workshop/20 addendum 1); a
    9 x 9 patch is valid unwarped at SIFT's finest octaves and invalid at
    coarse ORB levels, which is why the stage is built on SIFT (addendum 2:
    ORB + this refinement removed a quarter of ORB's offset).  No constant is
    exposed as a flag; the switch is the one flag.

S-c IMAGES STAY RAW.  Keypoints are undistorted at extraction (04d) and images
    are not; every patch is sampled on a grid of PINHOLE pixels (the
    keypoints' frame), each grid point mapped to the raw image with the frozen
    fr2 distortion (the frame's own distortion attribute, None without
    --undistort) by distort_pixels, then the raw image sampled bilinearly
    (cv2.remap, INTER_LINEAR).  The patch is thus an exact local
    undistortion; no image is ever undistorted as a whole (frame.py and
    camera.py are not edited).

S-d WHAT THE REFINEMENT DOES NOT CHANGE.  The matched keypoint index, the
    descriptor distance, the level sigma used by the chi-square gates and by
    the depth filter's measurement variance (SIFT: 1 px at octaves -1 and 0;
    04j S-a), the map's observations (a landmark still links keypoint
    indices, and local BA still reads keypoint pixels: BA outlier handling is
    filed for 04-4 by S7), and every existing log line (new information goes
    on new lines only).

S-e WHERE IT ACTS.  (i) Pair mapper: after match_pair, each candidate is
    refined in the partner keyframe along the epipolar line of the new
    keyframe's keypoint; triangulation, the pair-level parallax gate, the
    per-point gates and the 04e confirmation read the refined pixel; creation
    and linking use the real keypoint.  (ii) Depth filter: after a seed's
    descriptor match in the posed frame, the match is refined along the seed's
    band segment; the measured inverse range is triangulated from the refined
    pixel.  A refused refinement (edge, low peak, ambiguous, out of image, no
    image, degenerate direction) drops the candidate or the measurement; in
    the depth filter it counts as no_match, so every counter identity holds.

Imports math, dataclasses, numpy and cv2 only; no product module.
Deterministic: no randomness, no global state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RefineParams:
    """The four frozen constants of S-b (the prototype's); not flags."""
    patch_half: int = 4
    search_px: int = 6
    min_peak: float = 0.6
    ambiguity_ratio: float = 0.9


REFINE_COUNTER_KEYS = ("refine_attempted", "refine_ok", "refine_ambiguous", "refine_low_peak",
                       "refine_edge", "refine_oob", "refine_no_image", "refine_degenerate")
STATUS_KEY = {"refined": "refine_ok", "ambiguous": "refine_ambiguous",
              "low_peak": "refine_low_peak", "edge": "refine_edge", "oob": "refine_oob",
              "no_image": "refine_no_image", "degenerate": "refine_degenerate"}


def distort_pixels(xy, K, D):
    """
    Pinhole pixels ``xy`` (n, 2) -> raw pixels (n, 2) float64 under OpenCV's
    five-coefficient model D = (k1, k2, p1, p2, k3): the forward map whose
    inverse is camera.undistort_points (cv2.undistortPoints with P = K).
    D None returns a float64 copy of ``xy``; an empty input gives (0, 2).
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if D is None:
        return xy.copy()
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    k1, k2, p1, p2, k3 = (float(v) for v in np.asarray(D, dtype=np.float64).ravel()[:5])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xy[:, 0] - cx) / fx
    y = (xy[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([fx * xd + cx, fy * yd + cy])


def zncc(a, b):
    """Zero-mean normalized cross-correlation of two arrays of one shape; -1.0 when either
    is constant (the denominator is <= 1e-9)."""
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    d = math.sqrt(float(np.sum(a0 * a0)) * float(np.sum(b0 * b0)))
    if d > 1e-9:
        return float(np.sum(a0 * b0)) / d
    return -1.0


def sample_patches(img, K, D, centres, half):
    """
    For each centre c (pinhole pixels) the (2 half + 1) x (2 half + 1) grid
    q = c + (dx, dy), dx, dy in -half..half (row index dy, column index dx),
    mapped to raw pixels by distort_pixels and sampled bilinearly from ``img``
    with one cv2.remap call (float32 image and maps, INTER_LINEAR).  Returns a
    float64 array (m, 2 half + 1, 2 half + 1), or None when any raw
    coordinate is outside [0, W - 1] x [0, H - 1] or not finite.

    The remap runs on the smallest integer-aligned window of ``img`` holding
    every raw coordinate (the maps shifted by the window's integer origin,
    an exact float32 subtraction): the samples are those of the whole image,
    without converting the whole image to float32 at every call.
    """
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    half = int(half)
    side = 2 * half + 1
    m = centres.shape[0]
    offs = np.arange(-half, half + 1, dtype=np.float64)
    gx = centres[:, 0][:, None, None] + offs[None, None, :]          # (m, 1, side): column = dx
    gy = centres[:, 1][:, None, None] + offs[None, :, None]          # (m, side, 1): row = dy
    q = np.column_stack([np.broadcast_to(gx, (m, side, side)).ravel(),
                         np.broadcast_to(gy, (m, side, side)).ravel()])
    raw = distort_pixels(q, K, D)
    H, W = img.shape[:2]
    if raw.shape[0] == 0 or not np.all(np.isfinite(raw)):
        return None
    rx = raw[:, 0].astype(np.float32)
    ry = raw[:, 1].astype(np.float32)
    if rx.min() < 0.0 or ry.min() < 0.0 or rx.max() > W - 1 or ry.max() > H - 1:
        return None
    x0 = int(math.floor(float(rx.min())))
    y0 = int(math.floor(float(ry.min())))
    x1 = min(W - 1, int(math.floor(float(rx.max()))) + 1)
    y1 = min(H - 1, int(math.floor(float(ry.max()))) + 1)
    roi = np.asarray(img[y0:y1 + 1, x0:x1 + 1], dtype=np.float32)
    map_x = (rx - np.float32(x0)).reshape(m * side, side)
    map_y = (ry - np.float32(y0)).reshape(m * side, side)
    out = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
    return np.asarray(out, dtype=np.float64).reshape(m, side, side)


def refine_along_epipolar(img_ref, xy_ref, K_ref, D_ref, img_tgt, xy_tgt, K_tgt, D_tgt,
                          direction, params):
    """
    S-a: (xy_new, status, peak, t).  The reference patch at ``xy_ref`` in
    ``img_ref`` against the 2R + 1 target patches at xy_tgt + s u, s = -R..R
    (R = search_px, u = direction / |direction|) in ``img_tgt``; ZNCC scores
    sc; k = argmax.  Tests in order: an image missing -> no_image; direction
    not finite or |direction| < 1e-9 -> degenerate; a patch outside its image
    -> oob; k at either end -> edge; sc[k] < min_peak -> low_peak; another
    interior local maximum above ambiguity_ratio * sc[k] -> ambiguous; else
    the parabola through sc[k-1], sc[k], sc[k+1] (vertex offset 0 when the
    curvature is not negative) gives t = (k - R) + dt and xy_new = xy_tgt + t u,
    status 'refined'.  xy_new is None on every refusal.
    """
    if img_ref is None or img_tgt is None:
        return None, "no_image", -1.0, 0.0
    d = np.asarray(direction, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(d))
    if not (np.all(np.isfinite(d)) and math.isfinite(n)) or n < 1e-9:
        return None, "degenerate", -1.0, 0.0
    u = d / n
    R = int(params.search_px)
    h = int(params.patch_half)
    xy_tgt = np.asarray(xy_tgt, dtype=np.float64).reshape(2)
    ref = sample_patches(img_ref, K_ref, D_ref, np.asarray(xy_ref, dtype=np.float64).reshape(1, 2), h)
    if ref is None:
        return None, "oob", -1.0, 0.0
    s = np.arange(-R, R + 1, dtype=np.float64)
    tgt = sample_patches(img_tgt, K_tgt, D_tgt, xy_tgt[None, :] + s[:, None] * u[None, :], h)
    if tgt is None:
        return None, "oob", -1.0, 0.0
    sc = np.array([zncc(ref[0], tgt[i]) for i in range(2 * R + 1)], dtype=np.float64)
    k = int(np.argmax(sc))
    peak = float(sc[k])
    if k == 0 or k == 2 * R:
        return None, "edge", peak, 0.0
    if peak < float(params.min_peak):
        return None, "low_peak", peak, 0.0
    thr = float(params.ambiguity_ratio) * peak
    for i in range(1, 2 * R):
        if i != k and sc[i] >= sc[i - 1] and sc[i] >= sc[i + 1] and sc[i] > thr:
            return None, "ambiguous", peak, 0.0
    den = float(sc[k - 1] - 2.0 * sc[k] + sc[k + 1])
    dt = 0.5 * float(sc[k - 1] - sc[k + 1]) / den if den < 0.0 else 0.0
    t = float(k - R) + dt
    return np.asarray(xy_tgt + t * u, dtype=np.float64), "refined", peak, t
