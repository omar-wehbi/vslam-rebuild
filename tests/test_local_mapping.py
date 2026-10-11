"""
Stage 04-1a (spec B): the keyframe-pair point source of local_mapping.py on
synthetic scenes; no dataset, no pipeline run.

Frames follow the _GeoFrame / _scene pattern of tests/test_map_link_invariants.py
(a frame-like object with K, world_T_cam, cam_T_world, img_shape, method,
KeyPoints, Descriptors, keypoint_to_point, id, detector_config = {}), here with
a general pose and DEFAULT_K-like intrinsics (f = 521, 640x480).  Every test
ends with Map.check_invariants() all zero.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Point as point_module  # noqa: E402
from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from local_mapping import (  # noqa: E402
    LocalMappingParams,
    epipolar_sq_dist,
    fundamental_matrix,
    process_new_keyframe,
)
from mapping import local_map_track  # noqa: E402

ZERO = {
    "inv_link_to_missing_point": 0,
    "inv_link_without_observation": 0,
    "inv_observation_without_link": 0,
    "inv_frame_double_link": 0,
    "inv_frames_obs_mismatch": 0,
}

F_PX, W, H = 521.0, 640, 480
K = np.array([[F_PX, 0.0, W / 2], [0.0, F_PX, H / 2], [0.0, 0.0, 1.0]])
BASELINE_M, YAW_DEG = 0.18, 5.0
N_POINTS = 300

RX_PAIR = re.compile(r"^\s*>> \[LOCAL-MAP-PAIR\] kf=(\d+) partner=(\d+) created=(\d+)(?: reason=(baseline_ratio|zero_translation))?$")
RX_LM = re.compile(r"^\s*>> \[LOCAL-MAP\] kf=(\d+) neighbours=(\d+) pairs_used=(\d+) pairs_skipped_baseline=(\d+) candidates=(\d+) created=(\d+) "
                   r"rej_cheirality=(\d+) rej_parallax=(\d+) rej_chi2=(\d+) rej_scale=(\d+) obs_inserted=(\d+) culled=(\d+)$")


def _assert_consistent(world, frames):
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# frames and scenes
# --------------------------------------------------------------------------- #
class _KF:
    """Frame-like object with a general pose; keypoints at the (perturbed) projections."""

    def __init__(self, fid, world_T_cam, pts_world, descs, noise_px=0.0, rng=None, octaves=None):
        self.id = fid
        self.K = K
        self.world_T_cam = np.array(world_T_cam, dtype=np.float64)
        self.cam_T_world = np.linalg.inv(self.world_T_cam)
        self.img_shape = (H, W)
        self.method = "ORB"
        self.match_type = "BF"
        self.detector_config = {}
        self.keypoint_to_point = {}
        self.KeyPoints = []
        self.true_uv = []
        for i, X in enumerate(pts_world):
            xc = self.cam_T_world[:3, :3] @ X + self.cam_T_world[:3, 3]
            uv = K @ xc
            u, v = float(uv[0] / uv[2]), float(uv[1] / uv[2])
            self.true_uv.append((u, v))
            if noise_px > 0.0:
                u += float(rng.normal(0.0, noise_px))
                v += float(rng.normal(0.0, noise_px))
            kp = cv2.KeyPoint(u, v, 7.0)
            kp.octave = int(octaves[i]) if octaves is not None else 0
            self.KeyPoints.append(kp)
        self.Descriptors = np.array(descs, dtype=np.uint8, copy=True) if descs is not None else None
        self.true_uv = np.array(self.true_uv, dtype=np.float64)


def _yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _poses(baseline=BASELINE_M, yaw_deg=YAW_DEG):
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = _yaw(yaw_deg)
    T1[:3, 3] = [baseline, 0.0, 0.0]
    return T0, T1


def _regime_points(rng, T0, T1, n=N_POINTS):
    """n random points at depths 1.5-4.0 m visible in both cameras (the harness's construction)."""
    P0 = K @ np.linalg.inv(T0)[:3]
    P1 = K @ np.linalg.inv(T1)[:3]
    pts = []
    while len(pts) < n:
        z = rng.uniform(1.5, 4.0)
        x = rng.uniform(-0.6, 0.6) * z
        y = rng.uniform(-0.45, 0.45) * z
        X = np.array([x, y, z, 1.0])
        u0 = P0 @ X
        u1 = P1 @ X
        u0 = u0[:2] / u0[2]
        u1 = u1[:2] / u1[2]
        if 0 <= u0[0] < W and 0 <= u0[1] < H and 0 <= u1[0] < W and 0 <= u1[1] < H:
            pts.append(X[:3])
    return np.array(pts)


def _regime_scene(seed=2, noise_px=1.0, T1=None):
    """B1 regime scene: two keyframes, baseline 0.18 m along x with a 5 deg yaw, 300 points,
    1 px Gaussian noise in each image, distinct random descriptors identical across the pair.

    Creation figure (04b spec F1: >= 75 percent, 04a said 85): the A3(ii) band uses sigma_c
    only while B1 perturbs BOTH images by 1 px, so a true pair's point-to-line distance has
    variance about 2 px^2 and the expected in-band fraction at 3.84 sigma^2 is about 83
    percent (04a attempt 1 measured 79 to 87 percent over 20 seeds).
    test_b1_loss_is_the_epipolar_band_not_the_gates pins the mechanism."""
    rng = np.random.default_rng(seed)
    T0, T1_default = _poses()
    pts = _regime_points(rng, T0, T1_default)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=noise_px, rng=rng)
    k1 = _KF(2, T1_default if T1 is None else T1, pts, descs, noise_px=noise_px, rng=rng)
    return pts, descs, k0, k1


def _depth_in(frame, X):
    return float((frame.cam_T_world[:3, :3] @ np.asarray(X) + frame.cam_T_world[:3, 3])[2])


def _run(kf_idx, kf, keyframes, world, frames, params=None, stats=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = process_new_keyframe(kf_idx, kf, keyframes, world, frames, params or LocalMappingParams(), stats)
    return out, buf.getvalue()


# --------------------------------------------------------------------------- #
# B1. regime scene
# --------------------------------------------------------------------------- #
def test_b1_regime_scene_creates_most_points_with_small_depth_error():
    pts, descs, k0, k1 = _regime_scene()
    frames = [k0, k1]
    world = Map(frames=frames)
    assert k0.keypoint_to_point == {}
    stats = {}
    out, log = _run(1, k1, [(0, k0), (1, k1)], world, frames, stats=stats)

    assert out["neighbours"] == 1 and out["pairs_used"] == 1 and out["pairs_skipped_baseline"] == 0
    assert out["pairs"] == [(0, out["created"], None)]
    assert out["created"] == len(world.points)
    assert out["created"] >= 0.75 * N_POINTS, out      # 04b spec F1 (see _regime_scene)
    rel = []
    for p in world.points:
        i = p.observations[k1.id]
        assert p.observations == {k0.id: i, k1.id: i}   # true pair matched, both keyframes only
        assert sorted(p.frames) == [k0.id, k1.id]
        assert p.created_fid == k1.id and p.last_seen_fid == k1.id
        rel.append(abs(_depth_in(k1, p.point) - _depth_in(k1, pts[i])) / _depth_in(k1, pts[i]))
    assert np.median(rel) <= 0.06, np.median(rel)
    # log lines and the per-call identities of spec A7(a)
    pair_lines = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
    lm_lines = [RX_LM.match(l) for l in log.splitlines() if "[LOCAL-MAP]" in l]
    assert len(pair_lines) == 1 and pair_lines[0] is not None and len(lm_lines) == 1 and lm_lines[0] is not None
    assert int(pair_lines[0].group(3)) == out["created"] and pair_lines[0].group(4) is None
    assert log.index("[LOCAL-MAP-PAIR]") < log.index("[LOCAL-MAP] ")
    g = lm_lines[0].groups()
    assert [int(x) for x in g] == [1, 1, 1, 0, out["candidates"], out["created"], out["rej_cheirality"],
                                   out["rej_parallax"], out["rej_chi2"], out["rej_scale"], out["obs_inserted"], out["culled"]]
    assert stats == {"lm_keyframes": 1, "lm_created": out["created"], "lm_obs_inserted": out["obs_inserted"], "lm_culled": 0}
    _assert_consistent(world, frames)


def test_b1_loss_is_the_epipolar_band_not_the_gates():
    """Every candidate inside the band passes the four gates on the regime scene, and the
    band's own pass rate is the noise model's: with noise in K_c only (K_n exact) the
    distance variance is 1 px^2 and >= 93 percent (theory 95) of the pairs are created."""
    for seed in (2, 4, 7):
        pts, descs, k0, k1 = _regime_scene(seed=seed)
        world = Map(frames=[k0, k1])
        out, _ = _run(1, k1, [(0, k0), (1, k1)], world, [k0, k1])
        T_cn = k0.cam_T_world @ k1.world_T_cam
        F = fundamental_matrix(k0.K, k1.K, T_cn)
        d2 = np.diag(epipolar_sq_dist(F, np.array([kp.pt for kp in k1.KeyPoints]),
                                      np.array([kp.pt for kp in k0.KeyPoints])))
        in_band = int(np.sum(d2 <= LocalMappingParams().epipolar_chi2))
        assert out["candidates"] == in_band == out["created"], (seed, out, in_band)
        assert out["rej_cheirality"] == out["rej_parallax"] == out["rej_chi2"] == out["rej_scale"] == 0
        _assert_consistent(world, [k0, k1])
    rng = np.random.default_rng(9)
    T0, T1 = _poses()
    pts = _regime_points(rng, T0, T1)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=1.0, rng=rng)
    k1 = _KF(2, T1, pts, descs)                       # exact: the line is exact, only x_c is noisy
    world = Map(frames=[k0, k1])
    out, _ = _run(1, k1, [(0, k0), (1, k1)], world, [k0, k1])
    assert out["created"] >= 0.93 * N_POINTS, out["created"]
    _assert_consistent(world, [k0, k1])


# --------------------------------------------------------------------------- #
# B2. no points under rotation
# --------------------------------------------------------------------------- #
def test_b2_no_points_under_rotation():
    """Pure rotation: the second camera at the same centre, rotated 5 deg.  ||t_cn|| is
    numerically zero, so the A3(ii) skip fires by design and nothing is created."""
    T_rot = np.eye(4)
    T_rot[:3, :3] = _yaw(YAW_DEG)
    pts, descs, k0, k1 = _regime_scene(T1=T_rot)
    frames = [k0, k1]
    world = Map(frames=frames)
    id_before = point_module.ID
    out, log = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert out["created"] == 0 and len(world.points) == 0
    assert out["pairs_skipped_baseline"] == 1 and out["pairs_used"] == 0 and out["neighbours"] == 1
    assert out["pairs"] == [(0, 0, "zero_translation")]
    assert out["candidates"] == 0
    assert point_module.ID == id_before
    assert k0.keypoint_to_point == {} and k1.keypoint_to_point == {}
    m = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
    assert len(m) == 1 and m[0] is not None and m[0].group(4) == "zero_translation" and m[0].group(3) == "0"
    _assert_consistent(world, frames)


def test_b2_tiny_baseline_is_skipped_by_the_median_depth_gate():
    """Centre displaced by 1e-4 m: not numerically zero, but b / z_med = 4e-5 < 0.01 once
    K_0 has 10 linked landmarks at 2.5 m for the median-depth gate to run."""
    T_tiny = np.eye(4)
    T_tiny[:3, :3] = _yaw(YAW_DEG)
    T_tiny[:3, 3] = [1e-4, 0.0, 0.0]
    pts, descs, k0, k1 = _regime_scene(T1=T_tiny)
    frames = [k0, k1]
    world = Map(frames=frames)
    linked = []
    for i in range(10):
        X = np.array([(k0.KeyPoints[i].pt[0] - W / 2) / F_PX * 2.5, (k0.KeyPoints[i].pt[1] - H / 2) / F_PX * 2.5, 2.5])
        p = Point(point=X, descriptor=descs[i])
        world.add_point(p)
        assert world.link(k0, i, p)
        linked.append(p)
    n_before = len(world.points)
    out, _ = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert out["pairs"] == [(0, 0, "baseline_ratio")]
    assert out["pairs_skipped_baseline"] == 1 and out["created"] == 0 and out["candidates"] == 0
    assert len(world.points) == n_before
    _assert_consistent(world, frames)


RX_PARALLAX = re.compile(r"^\s*>> \[LOCAL-MAP-PARALLAX\] kf=(\d+) partner=(\d+) n=(\d+) median_deg=(\d+\.\d{3}) frac_over=(\d\.\d{3}) baseline=(\S+)$")
RX_POINT = re.compile(
    r"^\s*>> \[LOCAL-MAP-POINT\] id=(\d+) kf=(\d+) partner=(\d+) kp_n=(\d+) kp_c=(\d+) "
    r"u_n=(-?\d+\.\d{2}) v_n=(-?\d+\.\d{2}) u_c=(-?\d+\.\d{2}) v_c=(-?\d+\.\d{2}) "
    r"level_n=(-?\d+) level_c=(-?\d+) hamming=(\S+) hamming2=(\S+) "
    r"parallax_deg=(\d+\.\d{3}) z_n=(\S+) z_c=(\S+) baseline=(\S+)$"
)


def test_b2_without_median_depth_the_tiny_baseline_is_skipped_by_the_pair_level_gate():
    """04b spec F3: the same 1e-4 m displacement with no linked landmarks anywhere (A3(i)
    cannot run) is not numerically zero, so the pair reaches matching and triangulation,
    where the E3 pair-level gate skips it: the median parallax of its candidates is the
    noise parallax, below min_parallax_deg."""
    T_tiny = np.eye(4)
    T_tiny[:3, :3] = _yaw(YAW_DEG)
    T_tiny[:3, 3] = [1e-4, 0.0, 0.0]
    pts, descs, k0, k1 = _regime_scene(T1=T_tiny)
    frames = [k0, k1]
    world = Map(frames=frames)
    assert k0.keypoint_to_point == {} and k1.keypoint_to_point == {}
    out, log = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert out["pairs"] == [(0, 0, "baseline_ratio")]
    assert out["pairs_skipped_baseline"] == 1 and out["pairs_used"] == 0
    assert out["created"] == 0 and len(world.points) == 0
    assert out["candidates"] == 0
    assert out["rej_cheirality"] == out["rej_parallax"] == out["rej_chi2"] == out["rej_scale"] == 0
    par = [RX_PARALLAX.match(l) for l in log.splitlines() if "[LOCAL-MAP-PARALLAX]" in l]
    assert len(par) == 1 and par[0] is not None
    assert int(par[0].group(3)) > 0                       # the pair did reach triangulation
    assert float(par[0].group(4)) < LocalMappingParams().min_parallax_deg
    assert abs(float(par[0].group(6)) - 1e-4) < 1e-9
    assert log.index("[LOCAL-MAP-PARALLAX]") < log.index("[LOCAL-MAP-PAIR]") < log.index("[LOCAL-MAP] ")
    m = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
    assert len(m) == 1 and m[0] is not None and m[0].group(4) == "baseline_ratio" and m[0].group(3) == "0"
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# B3. wrong correspondences
# --------------------------------------------------------------------------- #
def _far_apart_pairs(frame, n_pairs, min_dv=100.0):
    """Pairs of keypoint indices far apart vertically (off each other's near-horizontal epipolar line)."""
    uv = np.array([kp.pt for kp in frame.KeyPoints])
    order = np.argsort(uv[:, 1])
    lo, hi = list(order[: len(order) // 2]), list(order[len(order) // 2:])
    pairs = []
    while lo and hi and len(pairs) < n_pairs:
        a, b = int(lo.pop(0)), int(hi.pop())
        if abs(uv[a, 1] - uv[b, 1]) >= min_dv:
            pairs.append((a, b))
    assert len(pairs) == n_pairs
    return pairs


def test_b3_swapped_descriptors_create_no_landmark_and_leave_others_unaffected():
    pts, descs, k0, k1 = _regime_scene()
    ref_frames = [k0, k1]
    ref_world = Map(frames=ref_frames)
    ref_out, _ = _run(1, k1, [(0, k0), (1, k1)], ref_world, ref_frames)
    ref_created = {p.observations[k1.id] for p in ref_world.points}

    pts2, descs2, s0, s1 = _regime_scene()     # identical scene (same seed)
    pairs = _far_apart_pairs(s1, 15)
    swapped = {i for ab in pairs for i in ab}
    assert len(swapped) == 30
    for a, b in pairs:
        s1.Descriptors[[a, b]] = s1.Descriptors[[b, a]]
    frames = [s0, s1]
    world = Map(frames=frames)
    out, _ = _run(1, s1, [(0, s0), (1, s1)], world, frames)

    created = {p.observations[s1.id]: p for p in world.points}
    for i in swapped:
        assert i not in created, i
        assert i not in s1.keypoint_to_point and i not in s0.keypoint_to_point
    assert set(created) == ref_created - swapped
    for i, p in created.items():
        assert p.observations == {s0.id: i, s1.id: i}
    assert out["created"] == len(ref_created - swapped)
    assert len(ref_created & swapped) > 0     # the swap removed points that the clean run created
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# B4. convention check (direction of T_cn / F)
# --------------------------------------------------------------------------- #
def test_b4_epipolar_band_direction():
    pts, descs, k0, k1 = _regime_scene(noise_px=0.0)
    kf_n, kf_c = k1, k0
    T_cn = kf_c.cam_T_world @ kf_n.world_T_cam
    F = fundamental_matrix(kf_c.K, kf_n.K, T_cn)
    xy_n = np.array([kp.pt for kp in kf_n.KeyPoints])
    xy_c = np.array([kp.pt for kp in kf_c.KeyPoints])
    chi2 = LocalMappingParams().epipolar_chi2
    d2 = np.diag(epipolar_sq_dist(F, xy_n, xy_c))
    assert np.all(d2 <= chi2), d2.max()
    assert np.all(d2 < 1e-6)                       # exact projections lie on the line
    # F transposed (= T_cn taken in the wrong direction), 04b spec F2: on the B1 poses (5 deg
    # yaw) E^T is close to -E near the principal point and the near-horizontal lines barely
    # move (30 percent measured inside the band), so the transpose scene keeps the B1 baseline
    # (0.18 m along x) and uses a 20 deg yaw, where fewer than 20 percent of the true
    # correspondences may fall inside the band with F transposed.
    d2_t = np.diag(epipolar_sq_dist(F.T, xy_n, xy_c))
    assert np.median(d2_t) > 3 * chi2, np.median(d2_t)
    assert np.max(d2_t) > 100.0
    rng = np.random.default_rng(3)
    T0, T20 = _poses(baseline=BASELINE_M, yaw_deg=20.0)
    pts20 = _regime_points(rng, T0, T20)
    descs20 = rng.integers(0, 256, size=(len(pts20), 32), dtype=np.uint8)
    c20 = _KF(1, T0, pts20, descs20)
    n20 = _KF(2, T20, pts20, descs20)
    T_cn20 = c20.cam_T_world @ n20.world_T_cam
    F20 = fundamental_matrix(c20.K, n20.K, T_cn20)
    xy_n20 = np.array([kp.pt for kp in n20.KeyPoints])
    xy_c20 = np.array([kp.pt for kp in c20.KeyPoints])
    assert np.all(np.diag(epipolar_sq_dist(F20, xy_n20, xy_c20)) < 1e-6)
    d2_t20 = np.diag(epipolar_sq_dist(F20.T, xy_n20, xy_c20))
    assert np.mean(d2_t20 <= chi2) < 0.20, np.mean(d2_t20 <= chi2)
    # x_c^T F x_n = 0 written out
    Xn = np.column_stack([xy_n, np.ones(len(xy_n))])
    Xc = np.column_stack([xy_c, np.ones(len(xy_c))])
    assert np.max(np.abs(np.einsum("ij,jk,ik->i", Xc, F, Xn))) < 1e-6   # cv2.KeyPoint.pt is float32
    # and the pipeline-facing sign: T_cn maps camera-n coordinates to camera-c coordinates
    Xw = pts[0]
    xn = kf_n.cam_T_world[:3, :3] @ Xw + kf_n.cam_T_world[:3, 3]
    xc = kf_c.cam_T_world[:3, :3] @ Xw + kf_c.cam_T_world[:3, 3]
    np.testing.assert_allclose(T_cn[:3, :3] @ xn + T_cn[:3, 3], xc, atol=1e-9)


# --------------------------------------------------------------------------- #
# B5. link discipline
# --------------------------------------------------------------------------- #
def test_b5_keypoint_linked_by_local_map_track_is_not_recreated():
    pts, descs, k0, k1 = _regime_scene()
    T_old = np.eye(4)
    T_old[:3, 3] = [-0.15, 0.0, 0.0]
    k_old = _KF(3, T_old, pts, descs)          # an older frame observing landmark 0 (not a keyframe)
    frames = [k_old, k0, k1]
    world = Map(frames=frames)
    old = Point(point=pts[0].copy(), descriptor=descs[0])
    world.add_point(old)
    assert world.link(k_old, 0, old)
    assert local_map_track(k1, world, [k_old], search_radius=8.0, max_desc_dist=50.0, max_depth_ratio=None) == 1
    assert k1.keypoint_to_point == {0: old.id}

    id_before = point_module.ID
    out, _ = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert k1.keypoint_to_point[0] == old.id
    assert old.observations == {k_old.id: 0, k1.id: 0}
    assert 0 not in k0.keypoint_to_point            # K_c's keypoint stays free: no landmark on it
    assert all(p.observations.get(k1.id) != 0 for p in world.points if p is not old)
    assert len(world.points) == 1 + out["created"]
    assert point_module.ID - id_before == out["created"]   # skipped candidates consume no id
    _assert_consistent(world, frames)


def test_b5_candidate_with_linked_kc_keypoint_is_skipped():
    pts, descs, k0, k1 = _regime_scene()
    frames = [k0, k1]
    world = Map(frames=frames)
    q = Point(point=pts[5].copy(), descriptor=descs[5])
    world.add_point(q)
    assert world.link(k0, 5, q)
    id_before = point_module.ID
    out, _ = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert k0.keypoint_to_point[5] == q.id
    new_points = [p for p in world.points if p is not q]
    assert all(p.observations.get(k0.id) != 5 for p in new_points)
    assert all(p.observations.get(k1.id) != 5 for p in new_points)
    # observation insertion (A4(a)) may add K_n's observation to q, never a new landmark
    assert k1.keypoint_to_point.get(5) in (None, q.id)
    assert len(new_points) == out["created"]
    assert point_module.ID - id_before == out["created"]
    _assert_consistent(world, frames)


def test_b5_refused_link_leaves_no_half_linked_landmark():
    pts, descs, k0, k1 = _regime_scene()
    frames = [k0, k1]

    class _RefusingMap(Map):
        """Refuses the K_n side of the link for keypoint 7 once (a refusal the pre-check
        cannot foresee), so the no-half-linked-landmark branch of A3(v) runs."""
        refused = 0

        def link(self, frame, kp_idx, point):
            if frame is k1 and int(kp_idx) == 7:
                self.refused += 1
                return False
            return super().link(frame, kp_idx, point)

    world = _RefusingMap(frames=frames)
    id_before = point_module.ID
    out, log = _run(1, k1, [(0, k0), (1, k1)], world, frames)
    assert world.refused == 1
    assert 7 not in k1.keypoint_to_point and 7 not in k0.keypoint_to_point
    assert all(p.observations.get(k0.id) != 7 for p in world.points)
    assert all(7 not in p.observations.values() for p in world.points)
    assert len(world.points) == out["created"]
    # the refused landmark was created then removed: it is not in the pair's created count
    m = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
    assert int(m[0].group(3)) == out["created"] == len(world.points)
    # ids: one per landmark in the map plus the single refused attempt, none for skips
    assert point_module.ID - id_before == out["created"] + 1
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# B6. culling
# --------------------------------------------------------------------------- #
def test_b6_observation_count_culling():
    T = [np.eye(4) for _ in range(5)]
    for i in range(5):
        T[i][:3, 3] = [0.1 * i, 0.0, 0.0]
    pts = np.array([[0.0, 0.0, 3.0], [0.3, 0.1, 3.5], [-0.2, 0.2, 2.5], [0.1, -0.1, 4.0]])
    rng = np.random.default_rng(0)
    descs = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
    kfs = [_KF(10 + i, T[i], pts, descs) for i in range(4)]
    k4 = _KF(14, T[4], [], None)                 # the new keyframe: nothing to match, nothing to insert
    frames = kfs + [k4]
    keyframes = [(20 * i, f) for i, f in enumerate(frames)]
    world = Map(frames=frames)

    def _make(pt_idx, created_kf, obs_frames):
        p = Point(point=pts[pt_idx].copy(), descriptor=descs[pt_idx])
        world.add_point(p)
        for f in obs_frames:
            assert world.link(f, pt_idx, p)
        p.created_fid = created_kf.id
        return p

    two = _make(0, kfs[1], [kfs[1], kfs[2]])            # created at K_1, 2 keyframe obs -> culled
    three = _make(1, kfs[1], [kfs[1], kfs[2], kfs[3]])  # 3 obs -> kept
    four = _make(2, kfs[1], [kfs[0], kfs[1], kfs[2], kfs[3]])
    young = _make(3, kfs[3], [kfs[2], kfs[3]])          # created at K_3: younger than the wait
    _assert_consistent(world, frames)

    out, log = _run(80, k4, keyframes, world, frames)
    assert out["culled"] == 1
    assert world.get_point(two.id) is None
    assert two.observations == {} and two.frames == []
    assert 0 not in kfs[1].keypoint_to_point and 0 not in kfs[2].keypoint_to_point
    assert world.get_point(three.id) is three and world.get_point(four.id) is four
    assert world.get_point(young.id) is young and young.observations == {kfs[2].id: 3, kfs[3].id: 3}
    # E1 fill: every earlier keyframe is a neighbour; E2: widest baseline first.  K_4 has no
    # keypoints, so every pair is used with created=0 (n=0 on its [LOCAL-MAP-PARALLAX] line).
    assert out["created"] == 0 and out["neighbours"] == 4
    assert out["pairs"] == [(0, 0, None), (20, 0, None), (40, 0, None), (60, 0, None)]
    par = [RX_PARALLAX.match(l) for l in log.splitlines() if "[LOCAL-MAP-PARALLAX]" in l]
    assert len(par) == 4 and all(par) and all(m.group(3) == "0" and m.group(4) == "0.000" for m in par)
    assert "culled=1" in log
    _assert_consistent(world, frames)


def test_b6_culling_ignores_non_keyframe_created_fid_and_counts_keyframe_observations_only():
    T = [np.eye(4) for _ in range(4)]
    for i in range(4):
        T[i][:3, 3] = [0.1 * i, 0.0, 0.0]
    pts = np.array([[0.0, 0.0, 3.0], [0.3, 0.1, 3.5]])
    descs = np.zeros((2, 32), dtype=np.uint8)
    kfs = [_KF(10 + i, T[i], pts, descs) for i in range(3)]
    extra = [_KF(20 + i, T[1], pts, descs) for i in range(3)]   # ordinary frames, not keyframes
    k3 = _KF(13, T[3], [], None)
    frames = kfs + extra + [k3]
    keyframes = [(0, kfs[0]), (20, kfs[1]), (40, kfs[2]), (60, k3)]
    world = Map(frames=frames)
    # created at K_0 (3 keyframes before K_3), one keyframe obs + three ordinary-frame obs -> culled
    a = Point(point=pts[0].copy(), descriptor=descs[0])
    world.add_point(a)
    for f in [kfs[0]] + extra:
        assert world.link(f, 0, a)
    a.created_fid = kfs[0].id
    # created by a non-keyframe frame id with one observation -> left alone
    b = Point(point=pts[1].copy(), descriptor=descs[1])
    world.add_point(b)
    assert world.link(extra[0], 1, b)
    b.created_fid = extra[0].id
    out, _ = _run(60, k3, keyframes, world, frames)
    assert out["culled"] == 1
    assert world.get_point(a.id) is None and world.get_point(b.id) is b
    assert all(0 not in f.keypoint_to_point for f in [kfs[0]] + extra)
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# extra: multi-keyframe run, neighbourhood, insertion in both directions
# --------------------------------------------------------------------------- #
def test_three_keyframes_neighbourhood_insertion_and_pair_identities():
    rng = np.random.default_rng(11)
    T0, T1 = _poses()
    T2 = np.eye(4)
    T2[:3, :3] = _yaw(2 * YAW_DEG)
    T2[:3, 3] = [2 * BASELINE_M, 0.0, 0.0]
    pts = _regime_points(rng, T0, T2)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=0.5, rng=rng)
    k1 = _KF(2, T1, pts, descs, noise_px=0.5, rng=rng)
    k2 = _KF(3, T2, pts, descs, noise_px=0.5, rng=rng)
    frames = [k0, k1, k2]
    world = Map(frames=frames)
    stats = {}
    out1, log1 = _run(20, k1, [(0, k0), (20, k1)], world, frames, stats=stats)
    assert out1["created"] > 200 and out1["neighbours"] == 1
    # The chain's frame-to-frame link transfer gives the new keyframe some links before the
    # mapper runs; emulate it for 30 of the landmarks (>= min_shared_points with K_0 and K_1).
    old = [p for p in world.points if p.created_fid == k1.id]
    for p in old[:30]:
        assert world.link(k2, p.observations[k1.id], p)
    out2, log2 = _run(40, k2, [(0, k0), (20, k1), (40, k2)], world, frames, stats=stats)
    # K_1 (preceding) and K_0 (shares >= 15 landmarks) are both neighbours; the pairs are
    # processed widest baseline first (E2): K_0 at 0.36 m before K_1 at 0.18 m
    assert out2["neighbours"] == 2 and out2["pairs_used"] == 2
    assert [p[0] for p in out2["pairs"]] == [0, 20]
    assert out2["created"] > 0
    # A4(a): the other landmarks of the K_0/K_1 pair gained K_2 observations (insertion, not re-creation)
    rest = old[30:]
    with_k2 = sum(1 for p in rest if k2.id in p.observations)
    assert with_k2 > 0.8 * len(rest)
    assert out2["obs_inserted"] >= with_k2
    assert all(p.observations[k2.id] == p.observations[k1.id] for p in rest if k2.id in p.observations)
    # identities of A7(a) over both calls
    for log, out in ((log1, out1), (log2, out2)):
        pl = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
        lm = [RX_LM.match(l) for l in log.splitlines() if "[LOCAL-MAP] " in l]
        assert all(pl) and len(lm) == 1
        assert len(pl) == out["neighbours"]
        assert sum(int(m.group(3)) for m in pl) == out["created"]
        assert sum(m.group(4) is not None for m in pl) == out["pairs_skipped_baseline"]
        assert max(log.index(m.group(0)) for m in pl) < log.index(lm[0].group(0))
    assert stats["lm_keyframes"] == 2 and stats["lm_created"] == out1["created"] + out2["created"]
    # no landmark carries two keypoints of one frame and every link is mirrored
    _assert_consistent(world, frames)


def test_new_landmarks_are_inserted_into_the_other_neighbour():
    """A4(b): a landmark created from the widest pair (K_2, K_0; E2 order) is projected into
    K_1 and linked there; the landmarks the (K_2, K_1) pair then creates from the keypoints
    K_0 had taken cannot be inserted into K_0."""
    rng = np.random.default_rng(5)
    T0, T1 = _poses()
    T2 = np.eye(4)
    T2[:3, :3] = _yaw(2 * YAW_DEG)
    T2[:3, 3] = [2 * BASELINE_M, 0.0, 0.0]
    pts = _regime_points(rng, T0, T2, n=60)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs)
    k1 = _KF(2, T1, pts, descs)
    k2 = _KF(3, T2, pts, descs)
    frames = [k0, k1, k2]
    world = Map(frames=frames)
    # Landmarks 0..19 are linked in all three frames (>= min_shared_points), so K_0 and K_1 are
    # both neighbours of K_2; keypoints 20..59 are free in K_1 and K_2.
    for i in range(20):
        p = Point(point=pts[i].copy(), descriptor=descs[i])
        world.add_point(p)
        assert world.link(k0, i, p) and world.link(k1, i, p) and world.link(k2, i, p)
        p.created_fid = k1.id
    # K_0's keypoints 20..39 are taken by unrelated landmarks (far away, unmatchable descriptors);
    # 40..59 stay free.  The (K_2, K_0) pair comes first (0.36 m against 0.18 m) and creates
    # 40..59; the (K_2, K_1) pair then creates 20..39 from the K_2 keypoints left unlinked.
    # A4(b) projects the first pair's landmarks into K_1 (free there: linked) and the second
    # pair's into K_0 (taken there: not linked).
    for i in range(20, 40):
        b = Point(point=np.array([5.0 + i, 5.0, 50.0]), descriptor=np.full(32, 255, dtype=np.uint8))
        world.add_point(b)
        assert world.link(k0, i, b)
    out, _ = _run(40, k2, [(0, k0), (20, k1), (40, k2)], world, frames)
    assert [p[0] for p in out["pairs"]] == [0, 20]
    assert out["pairs"][0][1] >= 15 and out["pairs"][1][1] >= 15
    assert out["pairs"][0][1] + out["pairs"][1][1] == out["created"]
    new = [p for p in world.points if p.created_fid == k2.id]
    assert len(new) == out["created"]
    from_k0 = [p for p in new if 40 <= p.observations[k2.id] < 60]
    from_k1 = [p for p in new if 20 <= p.observations[k2.id] < 40]
    assert len(from_k0) + len(from_k1) == len(new)
    assert all(k0.id in p.observations and p.observations[k0.id] == p.observations[k2.id] for p in from_k0)
    assert all(k1.id in p.observations and p.observations[k1.id] == p.observations[k2.id] for p in from_k1)
    inserted = [p for p in from_k0 if k1.id in p.observations]       # A4(b) into K_1
    assert len(inserted) >= 15
    assert all(p.observations[k1.id] == p.observations[k2.id] for p in inserted)
    assert all(k0.id not in p.observations for p in from_k1)          # K_0's 20..39 are taken
    assert out["obs_inserted"] >= len(inserted)
    _assert_consistent(world, frames)


def test_first_keyframe_has_no_neighbours_and_creates_nothing():
    pts, descs, k0, k1 = _regime_scene()
    frames = [k0]
    world = Map(frames=frames)
    out, log = _run(0, k0, [(0, k0)], world, frames)
    assert out["neighbours"] == 0 and out["created"] == 0 and out["pairs"] == []
    assert "[LOCAL-MAP-PAIR]" not in log and RX_LM.match(log.strip()) is not None
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# F4 / F5 (stage 04b). window bootstrap, pair order, pair-level gate, fill vs covisibility
# --------------------------------------------------------------------------- #
def _window_points(rng, poses, n=N_POINTS, z_lo=1.5, z_hi=2.5):
    """n random points at depths z_lo..z_hi visible in every pose (rejection-sampled as B1)."""
    Ps = [K @ np.linalg.inv(T)[:3] for T in poses]
    pts = []
    while len(pts) < n:
        z = rng.uniform(z_lo, z_hi)
        x = rng.uniform(-0.6, 0.6) * z
        y = rng.uniform(-0.45, 0.45) * z
        X = np.array([x, y, z, 1.0])
        ok = True
        for P in Ps:
            u = P @ X
            u = u[:2] / u[2]
            if not (0 <= u[0] < W and 0 <= u[1] < H):
                ok = False
                break
        if ok:
            pts.append(X[:3])
    return np.array(pts)


def _window_scene(n_kf, step_x, yaw_step_deg=0.0, seed=21, noise_px=1.0, n=N_POINTS):
    """n_kf keyframes K_0..K_{n_kf-1} with the B1 intrinsics, centres at x = step_x * i,
    yaw = yaw_step_deg * i, n points at depths 1.5..2.5 m visible in all, noise_px Gaussian
    noise in every image, distinct descriptors with the true correspondences identical."""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n_kf):
        T = np.eye(4)
        T[:3, :3] = _yaw(yaw_step_deg * i)
        T[:3, 3] = [step_x * i, 0.0, 0.0]
        poses.append(T)
    pts = _window_points(rng, poses, n=n)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    kfs = [_KF(100 + i, poses[i], pts, descs, noise_px=noise_px, rng=rng) for i in range(n_kf)]
    return pts, descs, kfs


def _parse_log(log):
    par = [RX_PARALLAX.match(l) for l in log.splitlines() if "[LOCAL-MAP-PARALLAX]" in l]
    pl = [RX_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-PAIR]" in l]
    lm = [RX_LM.match(l) for l in log.splitlines() if "[LOCAL-MAP] " in l]
    assert all(par) and all(pl) and len(lm) == 1 and lm[0] is not None
    return par, pl, lm[0]


def _assert_a7_identities(log, out):
    par, pl, lm = _parse_log(log)
    assert len(pl) == out["neighbours"] == int(lm.group(2))
    assert sum(int(m.group(3)) for m in pl) == out["created"] == int(lm.group(6))
    assert sum(m.group(4) is not None for m in pl) == out["pairs_skipped_baseline"] == int(lm.group(4))
    assert sum(m.group(4) is None for m in pl) == out["pairs_used"] == int(lm.group(3))
    assert max(log.index(m.group(0)) for m in pl) < log.index(lm.group(0))
    assert [int(m.group(2)) for m in pl] == [p[0] for p in out["pairs"]]
    # every [LOCAL-MAP-PARALLAX] line precedes the [LOCAL-MAP-PAIR] line of its partner
    for m in par:
        mate = next(q for q in pl if q.group(2) == m.group(2))
        assert log.index(m.group(0)) < log.index(mate.group(0))
    return par, pl, lm


def test_f4_window_bootstrap_fills_the_neighbourhood_and_orders_pairs_by_baseline():
    """04b spec F4: five keyframes 0.03 m apart, no links; the fill gives K_4 four neighbours,
    pairs go widest first, the consecutive pair (median parallax ~0.03 / 2.0 rad = 0.86 deg) is
    skipped by the pair-level gate, and the wider pairs create most points once each."""
    pts, descs, kfs = _window_scene(5, 0.03)
    frames = list(kfs)
    world = Map(frames=frames)
    keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
    assert all(f.keypoint_to_point == {} for f in kfs)
    k4 = kfs[4]
    out, log = _run(80, k4, keyframes, world, frames)

    assert out["neighbours"] == 4
    assert [p[0] for p in out["pairs"]] == [0, 20, 40, 60]
    assert out["pairs"][3] == (60, 0, "baseline_ratio")
    assert all(p[2] is None for p in out["pairs"][:3])
    assert out["pairs_used"] == 3 and out["pairs_skipped_baseline"] == 1
    created = [p[1] for p in out["pairs"]]
    assert created[0] >= 0.75 * N_POINTS, created
    assert created[0] == max(created) and sum(created) <= N_POINTS and sum(created) == out["created"]
    assert len(world.points) == out["created"]
    # every landmark is linked to exactly one K_4 keypoint, and K_4's keypoints are used once
    k4_kps = [p.observations[k4.id] for p in world.points]
    assert len(k4_kps) == len(set(k4_kps)) and all(k4.id in p.observations for p in world.points)
    # partners 1 and 2 created only from K_4 keypoints the K_0 pair left unlinked, and their
    # observations are of the true correspondence (identical keypoint index in every frame)
    for p in world.points:
        i = p.observations[k4.id]
        assert all(v == i for v in p.observations.values()), p.observations
    # A4(b) into the non-partner neighbours: median keyframe observations >= 3
    kf_ids = {f.id for f in kfs}
    n_obs = [sum(1 for fid in p.observations if fid in kf_ids) for p in world.points]
    assert np.median(n_obs) >= 3, np.median(n_obs)
    par, pl, lm = _assert_a7_identities(log, out)
    assert len(par) == 4 and [int(m.group(2)) for m in par] == [0, 20, 40, 60]
    meds = [float(m.group(4)) for m in par]
    assert meds[3] < LocalMappingParams().min_parallax_deg and 0.6 < meds[3] < 1.1, meds
    assert meds[0] > meds[1] > meds[2] > meds[3], meds        # widest pair, largest parallax
    assert all(int(m.group(3)) > 0 for m in par)
    assert float(par[0].group(5)) > 0.9 and float(par[3].group(5)) < 0.5
    bases = [float(m.group(6)) for m in par]
    np.testing.assert_allclose(bases, [0.12, 0.09, 0.06, 0.03], atol=1e-9)
    _assert_consistent(world, frames)


def test_f4_near_stationary_window_is_skipped_pairwise_by_the_parallax_median():
    """04b spec F4, second scenario: centres 0.2 mm apart (below the 1 mm stationary threshold,
    not numerically zero) with 0.5 deg yaw steps and no links: every pair reaches triangulation
    and is skipped by the pair-level gate; nothing reaches the per-point gates."""
    pts, descs, kfs = _window_scene(5, 0.0002, yaw_step_deg=0.5, seed=22)
    frames = list(kfs)
    world = Map(frames=frames)
    keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
    id_before = point_module.ID
    out, log = _run(80, kfs[4], keyframes, world, frames)
    assert out["neighbours"] == 4
    assert out["pairs"] == [(0, 0, "baseline_ratio"), (20, 0, "baseline_ratio"),
                            (40, 0, "baseline_ratio"), (60, 0, "baseline_ratio")]
    assert out["pairs_skipped_baseline"] == 4 and out["pairs_used"] == 0
    assert out["created"] == 0 and len(world.points) == 0 and point_module.ID == id_before
    assert out["candidates"] == 0
    assert out["rej_cheirality"] == out["rej_parallax"] == out["rej_chi2"] == out["rej_scale"] == 0
    assert all(f.keypoint_to_point == {} for f in kfs)
    par, pl, lm = _assert_a7_identities(log, out)
    assert len(par) == 4 and all(int(m.group(3)) > 0 for m in par)      # they DID reach triangulation
    assert all(float(m.group(4)) < LocalMappingParams().min_parallax_deg for m in par)
    _assert_consistent(world, frames)


def test_f5_fill_versus_covisibility():
    """04b spec F5: nine keyframes on the F4 geometry.  With no links select_neighbours for K_8
    is the five most recent earlier keyframes; with 20 landmarks shared by K_2 and K_8, K_2
    comes first (covisible) followed by the four most recent fill keyframes, and the pairs
    start with partner 2 (largest baseline, 0.18 m)."""
    from local_mapping import select_neighbours
    pts, descs, kfs = _window_scene(9, 0.03, seed=23)
    frames = list(kfs)
    world = Map(frames=frames)
    keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
    params = LocalMappingParams()
    k8 = kfs[8]
    nb = select_neighbours(8, k8, keyframes, params)
    assert [pos for pos, _, _ in nb] == [7, 6, 5, 4, 3]
    assert [idx for _, idx, _ in nb] == [140, 120, 100, 80, 60]
    assert nb[0][2] is kfs[7]
    # 20 shared landmarks between K_2 and K_8 (true positions, linked on both sides)
    for i in range(20):
        p = Point(point=pts[i].copy(), descriptor=descs[i])
        world.add_point(p)
        assert world.link(kfs[2], i, p) and world.link(k8, i, p)
        p.created_fid = kfs[2].id
    _assert_consistent(world, frames)
    nb = select_neighbours(8, k8, keyframes, params)
    assert [pos for pos, _, _ in nb] == [2, 7, 6, 5, 4]
    out, log = _run(160, k8, keyframes, world, frames)
    assert out["neighbours"] == 5
    assert [p[0] for p in out["pairs"]] == [40, 80, 100, 120, 140]    # E2: baseline descending
    assert out["pairs"][0][2] is None and out["pairs"][0][1] > 0
    par, pl, lm = _assert_a7_identities(log, out)
    np.testing.assert_allclose(float(par[0].group(6)), 0.18, atol=1e-9)
    _assert_consistent(world, frames)


# --------------------------------------------------------------------------- #
# T1-T2. stage 04c: the [LOCAL-MAP-POINT] provenance line
# --------------------------------------------------------------------------- #
def test_t1_provenance_line_identity_on_f4_window_scene():
    """04c spec T1: on the F4 window scene, every [LOCAL-MAP-POINT] line matches the exact
    format, its (kf, partner) count equals the pair's created and the total equals the
    returned created, every id is distinct and names a landmark linked at kp_n on K_4 and at
    kp_c on the partner, every parallax_deg is at or above the gate the point passed, every
    z_n/z_c is positive, hamming respects max_desc_dist, hamming2 is -1 or >= hamming, and the
    line precedes its pair's [LOCAL-MAP-PAIR] line."""
    pts, descs, kfs = _window_scene(5, 0.03)
    frames = list(kfs)
    world = Map(frames=frames)
    keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
    idx_to_frame = {20 * i: kfs[i] for i in range(5)}
    k4 = kfs[4]
    out, log = _run(80, k4, keyframes, world, frames)

    lines = log.splitlines()
    raw_point_lines = [l for l in lines if "[LOCAL-MAP-POINT]" in l]
    point_matches = [RX_POINT.match(l) for l in raw_point_lines]
    assert raw_point_lines and all(point_matches), raw_point_lines
    pair_lines = [RX_PAIR.match(l) for l in lines if "[LOCAL-MAP-PAIR]" in l]
    assert all(pair_lines)

    params = LocalMappingParams()
    ids_seen = set()
    by_partner = {}
    for m in point_matches:
        pid, kf_i, partner, kp_n, kp_c = (int(m.group(k)) for k in (1, 2, 3, 4, 5))
        u_n, v_n, u_c, v_c = (float(m.group(k)) for k in (6, 7, 8, 9))
        hamming, hamming2, parallax_deg, z_n, z_c = (
            float(m.group(k)) for k in (12, 13, 14, 15, 16)
        )
        assert pid not in ids_seen, "duplicate landmark id in the provenance lines"
        ids_seen.add(pid)
        assert kf_i == 80
        by_partner[partner] = by_partner.get(partner, 0) + 1
        assert parallax_deg >= params.min_parallax_deg
        assert z_n > 0.0 and z_c > 0.0
        assert hamming <= params.max_desc_dist
        assert hamming2 == -1.0 or hamming2 >= hamming
        # the landmark is linked at kp_n on K_4 and at kp_c on the partner, to the same point
        assert k4.keypoint_to_point.get(kp_n) == pid
        partner_frame = idx_to_frame[partner]
        assert partner_frame.keypoint_to_point.get(kp_c) == pid
        assert u_n == pytest.approx(k4.KeyPoints[kp_n].pt[0], abs=1e-2)
        assert v_n == pytest.approx(k4.KeyPoints[kp_n].pt[1], abs=1e-2)
        assert u_c == pytest.approx(partner_frame.KeyPoints[kp_c].pt[0], abs=1e-2)
        assert v_c == pytest.approx(partner_frame.KeyPoints[kp_c].pt[1], abs=1e-2)
        # the line precedes its pair's [LOCAL-MAP-PAIR] line
        mate = next(pm for pm in pair_lines if int(pm.group(2)) == partner)
        assert log.index(m.group(0)) < log.index(mate.group(0))

    for pm in pair_lines:
        partner = int(pm.group(2))
        created_on_pair = int(pm.group(3))
        assert by_partner.get(partner, 0) == created_on_pair
    assert sum(by_partner.values()) == out["created"] > 0
    _assert_consistent(world, frames)


def test_t2_provenance_line_adds_output_only():
    """04c spec T2: on the F4 window scene, the [LOCAL-MAP], [LOCAL-MAP-PAIR] and
    [LOCAL-MAP-PARALLAX] lines and the returned dict are unaffected by the new provenance
    lines: filtering [LOCAL-MAP-POINT] out of the captured text reproduces exactly the 04b
    F4 assertions, and a second, independently constructed run of the same scene gives the
    same filtered log and the same returned dict (the provenance lines only add output)."""

    def run_once():
        pts, descs, kfs = _window_scene(5, 0.03)
        frames = list(kfs)
        world = Map(frames=frames)
        keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
        out, log = _run(80, kfs[4], keyframes, world, frames)
        filtered = "\n".join(l for l in log.splitlines() if "[LOCAL-MAP-POINT]" not in l)
        return out, log, filtered

    out, log, filtered = run_once()
    assert any("[LOCAL-MAP-POINT]" in l for l in log.splitlines())

    # the 04b F4 assertions hold, unchanged, on the filtered log
    assert out["neighbours"] == 4
    assert [p[0] for p in out["pairs"]] == [0, 20, 40, 60]
    assert out["pairs"][3] == (60, 0, "baseline_ratio")
    assert all(p[2] is None for p in out["pairs"][:3])
    assert out["pairs_used"] == 3 and out["pairs_skipped_baseline"] == 1
    par, pl, lm = _assert_a7_identities(filtered, out)
    assert len(par) == 4 and [int(m.group(2)) for m in par] == [0, 20, 40, 60]
    meds = [float(m.group(4)) for m in par]
    assert meds[3] < LocalMappingParams().min_parallax_deg and 0.6 < meds[3] < 1.1, meds
    assert meds[0] > meds[1] > meds[2] > meds[3], meds
    bases = [float(m.group(6)) for m in par]
    np.testing.assert_allclose(bases, [0.12, 0.09, 0.06, 0.03], atol=1e-9)

    # a second, independent run of the identical scene: same filtered log, same returned dict
    out2, _, filtered2 = run_once()
    assert filtered == filtered2
    assert out == out2


# --------------------------------------------------------------------------- #
# B7. parser
# --------------------------------------------------------------------------- #
def test_b7_parser_point_source_and_retired_flags(tmp_path):
    import json
    import main
    parser = main.get_parser()
    a = parser.parse_args(["--dataset", "rgb"])
    assert a.point_source == "keyframe_pair"
    for fl, extra in {"--min_parallax_deg": ["1"], "--triangulation_max_depth_ratio": ["1"],
                      "--use_multi_frame_triangulation": [], "--triangulation_span": ["3"],
                      "--triangulation_mode": ["all"], "--triangulation_num_refs": ["2"],
                      "--prune_low_obs": [], "--no_prune_low_obs": [], "--prune_1obs_age": ["1"],
                      "--prune_2obs_age": ["1"], "--protect_keyframe_points": [],
                      "--no_protect_keyframe_points": []}.items():
        with pytest.raises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--dataset", "rgb", fl] + extra)
    a = parser.parse_args(["--dataset", "rgb", "--lm_min_parallax_deg", "1.1"])
    assert a.lm_min_parallax_deg == 1.1
    a3 = parser.parse_args(["--dataset", "rgb", "--point_source", "depth_filter"])
    assert a3.point_source == "depth_filter"
    # every LocalMappingParams field is an --lm_ flag with the dataclass default
    defaults = LocalMappingParams()
    for name in defaults.__dataclass_fields__:
        assert getattr(a, "lm_" + name) == getattr(defaults, name), name
    assert LocalMappingParams.from_args(a) == defaults
    a2 = parser.parse_args(["--dataset", "rgb", "--lm_cull_after_keyframes", "5", "--lm_max_desc_dist", "40"])
    p2 = LocalMappingParams.from_args(a2)
    assert p2.cull_after_keyframes == 5 and p2.max_desc_dist == 40.0
    out = tmp_path / "resolved_settings.json"
    with contextlib.redirect_stdout(io.StringIO()):
        main.write_resolved_settings(a, out)
    rs = json.loads(out.read_text())
    assert rs["point_source"] == "keyframe_pair" and rs["lm_min_parallax_deg"] == 1.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
