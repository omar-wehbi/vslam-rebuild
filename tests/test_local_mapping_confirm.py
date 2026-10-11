"""
Stage 04e (spec T): third-view chi-square confirmation of the keyframe-pair
source (local_mapping.confirm_candidates behind LocalMappingParams.confirm)
on synthetic scenes; no dataset, no pipeline run.

The frames, scenes and log regexes are those of tests/test_local_mapping.py
(imported).  Every test ends with Map.check_invariants() all zero.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import Point as point_module  # noqa: E402
from Map import Map  # noqa: E402
from local_mapping import (  # noqa: E402
    LocalMappingParams,
    confirm_candidates,
    fundamental_matrix,
    ray_infinity_pixel,
    translational_disparity,
    triangulate_pair,
)
from test_local_mapping import (  # noqa: E402
    BASELINE_M,
    F_PX,
    H,
    K,
    N_POINTS,
    RX_LM,
    RX_PAIR,
    RX_POINT,
    W,
    YAW_DEG,
    ZERO,
    _KF,
    _assert_a7_identities,
    _poses,
    _regime_points,
    _run,
    _window_points,
    _window_scene,
    _yaw,
)

RX_CONFIRM = re.compile(r"^\s*>> \[LOCAL-MAP-CONFIRM\] kf=(\d+) pending=(\d+) no_third_view=(\d+) unconfirmed=(\d+) confirmed=(\d+)$")
RX_THIRD = re.compile(r"^\s*>> \[LOCAL-MAP-THIRD\] id=(\d+) kf=(\d+) partner=(\d+) third=(\d+) kp_third=(\d+) "
                      r"d_pair=(\d+\.\d{2}) d_third=(\d+\.\d{2}) ratio=(\d+\.\d{3}) err_px=(\d+\.\d{2}) hamming=(\S+)$")


def _project(frame, X):
    xc = frame.cam_T_world[:3, :3] @ np.asarray(X, dtype=np.float64) + frame.cam_T_world[:3, 3]
    uv = frame.K @ xc
    return uv[:2] / uv[2]


def _depth_in(frame, X):
    return float((frame.cam_T_world[:3, :3] @ np.asarray(X) + frame.cam_T_world[:3, 3])[2])


def _confirm_lines(log):
    conf = [RX_CONFIRM.match(l) for l in log.splitlines() if "[LOCAL-MAP-CONFIRM]" in l]
    third = [RX_THIRD.match(l) for l in log.splitlines() if "[LOCAL-MAP-THIRD]" in l]
    point = [RX_POINT.match(l) for l in log.splitlines() if "[LOCAL-MAP-POINT]" in l]
    assert all(conf) and all(third) and all(point)
    return conf, third, point


def _assert_confirm_identities(log, out):
    """Spec C5 / C3 / C4: one [LOCAL-MAP-CONFIRM] line before [LOCAL-MAP], the pending
    identity, confirmed == created, and the THIRD ids equal the POINT ids."""
    conf, third, point = _confirm_lines(log)
    assert len(conf) == 1
    lm = [l for l in log.splitlines() if "[LOCAL-MAP] " in l]
    assert len(lm) == 1
    assert log.index(conf[0].group(0)) < log.index(lm[0])
    kf, pending, none, unc, confirmed = (int(conf[0].group(k)) for k in range(1, 6))
    assert kf == out["kf"]
    assert pending == none + unc + confirmed
    assert (pending, none, unc, confirmed) == (out["pending"], out["no_third_view"], out["unconfirmed"], out["confirmed"])
    assert confirmed == out["created"]
    third_ids = {int(m.group(1)) for m in third}
    point_ids = {int(m.group(1)) for m in point}
    assert third_ids == point_ids
    # every THIRD line follows its POINT line and precedes the pair's [LOCAL-MAP-PAIR] line
    lines = log.splitlines()
    for m in third:
        pid = int(m.group(1))
        i_third = next(i for i, l in enumerate(lines) if l == m.group(0))
        i_point = next(i for i, l in enumerate(lines) if f"[LOCAL-MAP-POINT] id={pid} " in l)
        assert i_point < i_third
        i_pair = next(i for i in range(i_third, len(lines)) if "[LOCAL-MAP-PAIR]" in lines[i])
        assert int(RX_PAIR.match(lines[i_pair]).group(2)) == int(m.group(3))
    return conf[0], third, point


# --------------------------------------------------------------------------- #
# T1. geometry
# --------------------------------------------------------------------------- #
def test_t1_ray_infinity_pixel_and_translational_disparity():
    T0, T1 = _poses()
    X = np.array([0.05, 0.02, 2.5])
    k0 = _KF(1, T0, [X], np.zeros((1, 32), dtype=np.uint8))
    k1 = _KF(2, T1, [X], np.zeros((1, 32), dtype=np.uint8))
    xy0 = np.array(k0.KeyPoints[0].pt, dtype=np.float64)
    C0 = k0.world_T_cam[:3, 3]
    ray = (X - C0) / np.linalg.norm(X - C0)
    u_inf, ok = ray_infinity_pixel(k0, xy0, k1)
    assert ok and np.all(np.isfinite(u_inf))
    # the spec's far point (1e6 m) sits f * b / 1e6 ~ 1e-4 px from the infinity point; a point
    # at 1e9 m is within 1e-6 px of it (the contract's tolerance)
    np.testing.assert_allclose(u_inf, _project(k1, X + 1e9 * ray), atol=1e-6)
    np.testing.assert_allclose(u_inf, _project(k1, X + 1e6 * ray), atol=1e-3)
    d = translational_disparity(k0, xy0, X, k1)
    assert d == pytest.approx(float(np.linalg.norm(_project(k1, X) - u_inf)), abs=1e-9)
    small_angle = F_PX * BASELINE_M / 2.5
    assert abs(d - small_angle) <= 0.05 * small_angle, (d, small_angle)
    # pure rotation: K_1 at K_0's centre, 5 deg yaw -> zero disparity for every point
    T_rot = np.eye(4)
    T_rot[:3, :3] = _yaw(YAW_DEG)
    rng = np.random.default_rng(1)
    pts = _regime_points(rng, T0, T_rot, n=50)
    k_rot = _KF(3, T_rot, pts, np.zeros((len(pts), 32), dtype=np.uint8))
    k0b = _KF(4, T0, pts, np.zeros((len(pts), 32), dtype=np.uint8))
    for i, P in enumerate(pts):
        # the exact projection (cv2.KeyPoint.pt is float32: 1e-5 px of rounding on its own)
        dr = translational_disparity(k0b, k0b.true_uv[i], P, k_rot)
        assert np.isfinite(dr) and dr < 1e-6, dr
    # a point behind B -> NaN (the ray's infinity point is fine, the point is not)
    assert np.isnan(translational_disparity(k0, xy0, np.array([0.0, 0.0, -3.0]), k1))
    assert np.isnan(translational_disparity(k0, xy0, np.array([np.nan, 0.0, 3.0]), k1))
    # a ray whose infinity point is behind B (B rotated by 120 deg) -> ok False / NaN
    T_120 = np.eye(4)
    T_120[:3, :3] = _yaw(120.0)
    T_120[:3, 3] = [BASELINE_M, 0.0, 0.0]
    k_120 = _KF(5, T_120, [], None)
    u_bad, ok_bad = ray_infinity_pixel(k0, xy0, k_120)
    assert ok_bad is False and np.all(np.isnan(u_bad))
    assert np.isnan(translational_disparity(k0, xy0, X, k_120))
    assert Map(frames=[k0, k1]).check_invariants([k0, k1]) == ZERO


# --------------------------------------------------------------------------- #
# T2. switch off is the old mapper
# --------------------------------------------------------------------------- #
def _f4_run(params):
    pts, descs, kfs = _window_scene(5, 0.03)
    frames = list(kfs)
    world = Map(frames=frames)
    keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
    point_module.ID = 1000
    out, log = _run(80, kfs[4], keyframes, world, frames, params=params)
    assert world.check_invariants(frames) == ZERO
    return out, log


def _three_kf_run(params):
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
    point_module.ID = 5000
    out1, log1 = _run(20, k1, [(0, k0), (20, k1)], world, frames, params=params)
    old = [p for p in world.points if p.created_fid == k1.id]
    for p in old[:30]:
        assert world.link(k2, p.observations[k1.id], p)
    out2, log2 = _run(40, k2, [(0, k0), (20, k1), (40, k2)], world, frames, params=params)
    assert world.check_invariants(frames) == ZERO
    return (out1, out2), log1 + log2


def test_t2_switch_off_is_the_old_mapper():
    assert LocalMappingParams().confirm is False
    assert LocalMappingParams().confirm_disparity_ratio == 0.3
    for runner in (_f4_run, _three_kf_run):
        out_a, log_a = runner(LocalMappingParams())
        out_b, log_b = runner(LocalMappingParams(confirm=False))
        assert out_a == out_b
        assert log_a == log_b
        assert "[LOCAL-MAP-POINT]" in log_a
        assert "[LOCAL-MAP-CONFIRM]" not in log_a and "[LOCAL-MAP-THIRD]" not in log_a
        for out in (out_a if isinstance(out_a, tuple) else (out_a,)):
            assert (out["pending"], out["no_third_view"], out["unconfirmed"], out["confirmed"]) == (0, 0, 0, 0)
            assert out["created"] > 0


# --------------------------------------------------------------------------- #
# T3. true correspondences confirm
# --------------------------------------------------------------------------- #
def test_t3_true_correspondences_confirm_on_the_window_scene():
    def run(confirm):
        pts, descs, kfs = _window_scene(5, 0.03, noise_px=0.5)
        frames = list(kfs)
        world = Map(frames=frames)
        keyframes = [(20 * i, f) for i, f in enumerate(kfs)]
        out, log = _run(80, kfs[4], keyframes, world, frames, params=LocalMappingParams(confirm=confirm))
        assert world.check_invariants(frames) == ZERO
        return out, log, world, kfs

    # off: the F4 structure (four neighbours by the fill, widest pair first, most points from
    # it; at 0.5 px noise the widest pair takes every keypoint, so the narrower pairs have no
    # candidate left and the consecutive pair is not reached by the parallax median)
    out_off, log_off, world_off, _ = run(False)
    assert out_off["neighbours"] == 4 and [p[0] for p in out_off["pairs"]] == [0, 20, 40, 60]
    assert out_off["pairs"][0][1] >= 0.75 * N_POINTS
    assert out_off["pairs"][0][1] == max(p[1] for p in out_off["pairs"])
    assert out_off["created"] == len(world_off.points) == sum(p[1] for p in out_off["pairs"])
    assert "[LOCAL-MAP-CONFIRM]" not in log_off and "[LOCAL-MAP-THIRD]" not in log_off
    _assert_a7_identities(log_off, out_off)

    out_on, log_on, world_on, kfs = run(True)
    k4 = kfs[4]
    assert out_on["neighbours"] == 4 and [p[0] for p in out_on["pairs"]] == [0, 20, 40, 60]
    # the first pair's candidates and gate counts are those of the switch-off run (nothing
    # before confirmation changes); its created count is at least 80 percent of the off run's
    assert out_on["candidates"] >= out_off["pairs"][0][1]
    assert out_on["pairs"][0][1] >= 0.8 * out_off["pairs"][0][1], (out_on["pairs"], out_off["pairs"])
    conf, third, point = _assert_confirm_identities(log_on, out_on)
    assert out_on["pending"] > 0 and out_on["confirmed"] == out_on["created"] == len(world_on.points) > 0
    # the widest pair (K_4, K_0): third views K_1, K_2, K_3 at r ~ 0.75, 0.50, 0.25 -> only K_2
    first_pair = [m for m in third if int(m.group(3)) == 0]
    assert first_pair and len(first_pair) == out_on["pairs"][0][1]
    for m in first_pair:
        assert int(m.group(4)) == 40, m.group(0)
        r = float(m.group(8))
        assert 0.3 <= r <= 0.7, r
        assert abs(r - 0.5) < 0.05, r
        assert float(m.group(9)) <= np.sqrt(LocalMappingParams().reproj_chi2) + 1e-6
        assert int(m.group(10)) == 0            # identical descriptors of the true correspondence
    # the other triangulated pairs see r in the eligibility set too
    for m in third:
        r = float(m.group(8))
        assert r >= 0.3 and abs(r - 1.0) >= 0.3, m.group(0)
    # every created landmark: >= 3 keyframe observations, all of the true correspondence
    kf_ids = {f.id for f in kfs}
    for p in world_on.points:
        i = p.observations[k4.id]
        assert sum(1 for fid in p.observations if fid in kf_ids) >= 3, p.observations
        assert all(v == i for v in p.observations.values()), p.observations
        assert k4.keypoint_to_point[i] == p.id
    # the confirming keypoint is linked: every THIRD line's (third, kp_third) is an observation of id
    idx_to_frame = {20 * i: kfs[i] for i in range(5)}
    for m in third:
        p = world_on.get_point(int(m.group(1)))
        fr = idx_to_frame[int(m.group(4))]
        assert p is not None and p.observations.get(fr.id) == int(m.group(5))
        assert fr.keypoint_to_point.get(int(m.group(5))) == p.id
    _assert_a7_identities(log_on, out_on)


# --------------------------------------------------------------------------- #
# T4 / T5. a planted repeat is refused; eligibility on both sides
# --------------------------------------------------------------------------- #
def _repeat_scene(p_px, k1_x=BASELINE_M, k1_yaw=YAW_DEG, seed=11, noise_px=0.5, z_range=None):
    """K_0 at the origin, K_1 at (k1_x, yaw k1_yaw), K_2 at (0.36 m, 10 deg yaw), 0.5 px noise,
    no links; the K_0 keypoints of every third point are displaced by p_px along their
    epipolar line for the K_2 keypoint, on the side that increases the disparity from K_2's
    ray infinity point.  Points: the B1 regime points (1.5..4 m, visible in K_0 and K_2) or,
    with z_range, _window_points at those depths visible in all three.
    Returns (frames, displaced index set, undisplaced index set)."""
    rng = np.random.default_rng(seed)
    T0, _ = _poses()
    T1 = np.eye(4)
    T1[:3, :3] = _yaw(k1_yaw)
    T1[:3, 3] = [k1_x, 0.0, 0.0]
    T2 = np.eye(4)
    T2[:3, :3] = _yaw(2 * YAW_DEG)
    T2[:3, 3] = [2 * BASELINE_M, 0.0, 0.0]
    if z_range is None:
        pts = _regime_points(rng, T0, T2)
    else:
        pts = _window_points(rng, [T0, T1, T2], n=N_POINTS, z_lo=z_range[0], z_hi=z_range[1])
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=noise_px, rng=rng)
    k1 = _KF(2, T1, pts, descs, noise_px=noise_px, rng=rng)
    k2 = _KF(3, T2, pts, descs, noise_px=noise_px, rng=rng)
    T_02 = k0.cam_T_world @ k2.world_T_cam            # camera-2 coordinates to camera-0 coordinates
    F = fundamental_matrix(k0.K, k2.K, T_02)          # x_0^T F x_2 = 0
    displaced, undisplaced = set(), set()
    for i in range(len(pts)):
        if i % 3 != 0:
            undisplaced.add(i)
            continue
        x2 = np.array([k2.KeyPoints[i].pt[0], k2.KeyPoints[i].pt[1], 1.0])
        l = F @ x2
        direction = np.array([l[1], -l[0]]) / np.hypot(l[0], l[1])
        x0 = np.array(k0.KeyPoints[i].pt, dtype=np.float64)
        u_inf, ok = ray_infinity_pixel(k2, x2[:2], k0)
        assert ok
        plus, minus = x0 + p_px * direction, x0 - p_px * direction
        new = plus if np.linalg.norm(plus - u_inf) > np.linalg.norm(minus - u_inf) else minus
        if not (0 <= new[0] < W and 0 <= new[1] < H):
            undisplaced.add(i)                     # would leave the image: left where it is
            continue
        k0.KeyPoints[i].pt = (float(new[0]), float(new[1]))
        displaced.add(i)
    assert len(displaced) >= 0.8 * (len(pts) // 3)
    return [k0, k1, k2], displaced, undisplaced


def _created_by_k2_keypoint(world, k2):
    return {p.observations[k2.id]: p for p in world.points if k2.id in p.observations}


def test_t4_planted_repeat_is_refused_by_the_third_view():
    # the displacement lowers the triangulated depth (the sign the pair gates admit)
    frames, displaced, undisplaced = _repeat_scene(15.0)
    k0, k1, k2 = frames
    i = min(displaced)
    clean, _, _ = _repeat_scene(0.0)
    X_clean, _ = triangulate_pair(clean[2], clean[0], [(i, i)])
    X_disp, _ = triangulate_pair(k2, k0, [(i, i)])
    assert _depth_in(k2, X_disp[0]) < 0.95 * _depth_in(k2, X_clean[0]), (X_disp, X_clean)
    d_clean = translational_disparity(clean[2], np.array(clean[2].KeyPoints[i].pt), X_clean[0], clean[0])
    d_disp = translational_disparity(k2, np.array(k2.KeyPoints[i].pt), X_disp[0], k0)
    assert d_disp > d_clean + 10.0, (d_disp, d_clean)

    def run(confirm):
        fr, disp, undisp = _repeat_scene(15.0)
        world = Map(frames=fr)
        keyframes = [(0, fr[0]), (20, fr[1]), (40, fr[2])]
        out, log = _run(40, fr[2], keyframes, world, fr, params=LocalMappingParams(confirm=confirm))
        assert world.check_invariants(fr) == ZERO
        return out, log, world, fr, disp, undisp

    # off: the displaced candidates pass the pair gates and are created from the (K_2, K_0) pair
    out, log, world, fr, disp, undisp = run(False)
    assert [p[0] for p in out["pairs"]] == [0, 20]
    by_kp = _created_by_k2_keypoint(world, fr[2])
    made_disp = [i for i in disp if i in by_kp and by_kp[i].observations.get(fr[0].id) == i]
    assert len(made_disp) >= 0.8 * len(disp), (len(made_disp), len(disp))
    assert "[LOCAL-MAP-CONFIRM]" not in log
    # on: at most 5 percent of the displaced are created, >= 80 percent of the undisplaced are,
    # and no K_1 observation is of a wrong keypoint
    out, log, world, fr, disp, undisp = run(True)
    assert [p[0] for p in out["pairs"]] == [0, 20]
    by_kp = _created_by_k2_keypoint(world, fr[2])
    n_disp = sum(1 for i in disp if i in by_kp)
    n_undisp = sum(1 for i in undisp if i in by_kp)
    assert n_disp <= 0.05 * len(disp), (n_disp, len(disp))
    assert n_undisp >= 0.8 * len(undisp), (n_undisp, len(undisp))
    for i, p in by_kp.items():
        if fr[1].id in p.observations:
            assert p.observations[fr[1].id] == i, p.observations
    conf, third, point = _assert_confirm_identities(log, out)
    assert out["unconfirmed"] >= 0.8 * len(disp)
    first_pair = [m for m in third if int(m.group(3)) == 0]
    assert first_pair and all(int(m.group(4)) == 20 for m in first_pair)
    ratios = [float(m.group(8)) for m in first_pair]
    assert 0.4 < np.median(ratios) < 0.6, np.median(ratios)
    _assert_a7_identities(log, out)


# T5 placements of K_1.  The disparity ratio r of spec C1 is measured from K_n's (K_2's) ray,
# so r ~ b_21 / b_20 on this translation: r ~ 0.94 needs K_1 0.34 m from K_2, i.e. at x = 0.02 m
# (the spec's "0.34 m"), and r ~ 0.2 needs K_1 0.072 m from K_2, i.e. at x = 0.288 m (the
# spec's "0.072 m").  Depths 4..8 m so that the (K_2, K_1) pair of placement (b), 0.072 m of
# baseline, is parallax-starved (0.072 / 6 = 0.69 deg < 1.1) and skipped as a whole: the
# (K_2, K_0) pair is then the only source of pending candidates, and the [LOCAL-MAP-CONFIRM]
# line reads on it alone (with the 1.5..4 m points that pair is used and K_0 is an eligible
# third view for it at r ~ 5, which is not what T5 tests).
T5_Z_RANGE = (4.0, 8.0)


@pytest.mark.parametrize("k1_x, r_lo, r_hi", [(0.36 - 0.34, 0.85, 1.0), (0.36 - 0.072, 0.15, 0.25)])
def test_t5_eligibility_both_sides(k1_x, r_lo, r_hi):
    fr, disp, undisp = _repeat_scene(10.0, k1_x=k1_x, z_range=T5_Z_RANGE)
    world = Map(frames=fr)
    keyframes = [(0, fr[0]), (20, fr[1]), (40, fr[2])]
    out, log = _run(40, fr[2], keyframes, world, fr, params=LocalMappingParams(confirm=True))
    assert world.check_invariants(fr) == ZERO
    assert [p[0] for p in out["pairs"]] == [0, 20]
    conf, third, point = _assert_confirm_identities(log, out)
    assert out["pending"] >= 200
    assert out["no_third_view"] == out["pending"], out
    assert out["created"] == 0 and len(world.points) == 0 and not third and not point
    assert int(conf.group(3)) == int(conf.group(2)) and int(conf.group(5)) == 0
    assert all(f.keypoint_to_point == {} for f in fr)
    # the ratio K_1 offers on the (K_2, K_0) pair, from the geometry directly: outside the
    # eligibility set on the stated side
    i = min(undisp)
    X, _ = triangulate_pair(fr[2], fr[0], [(i, i)])
    xy2 = np.array(fr[2].KeyPoints[i].pt)
    r = translational_disparity(fr[2], xy2, X[0], fr[1]) / translational_disparity(fr[2], xy2, X[0], fr[0])
    assert r_lo < r < r_hi, r
    assert r < 0.3 or abs(r - 1.0) < 0.3
    _assert_a7_identities(log, out)


def test_t5b_control_lower_bound_a_near_zero_disparity_view_confirms_the_repeat():
    """With confirm_disparity_ratio = 0.1 the K_1 view at r ~ 0.2 becomes eligible and the
    wrong points are confirmed by the TRUE K_1 keypoint: their prediction lands p * r = 2 px
    from it, inside the 2.45 px level-0 window (the lower bound's motivation)."""
    fr, disp, undisp = _repeat_scene(10.0, k1_x=0.36 - 0.072, z_range=T5_Z_RANGE)
    world = Map(frames=fr)
    keyframes = [(0, fr[0]), (20, fr[1]), (40, fr[2])]
    params = LocalMappingParams(confirm=True, confirm_disparity_ratio=0.1)
    out, log = _run(40, fr[2], keyframes, world, fr, params=params)
    assert world.check_invariants(fr) == ZERO
    conf, third, point = _assert_confirm_identities(log, out)
    assert out["no_third_view"] == 0 and out["confirmed"] > 0
    by_kp = _created_by_k2_keypoint(world, fr[2])
    wrongly = [i for i in disp if i in by_kp and by_kp[i].observations.get(fr[1].id) == i
               and by_kp[i].observations.get(fr[0].id) == i]
    assert len(wrongly) >= 0.5 * len(disp), (len(wrongly), len(disp))
    first_pair = [m for m in third if int(m.group(3)) == 0]
    assert first_pair and all(int(m.group(4)) == 20 for m in first_pair)
    ratios = [float(m.group(8)) for m in first_pair]
    assert 0.1 <= min(ratios) and np.median(ratios) < 0.3, (min(ratios), np.median(ratios))
    # and the confirming residual is the predicted p * r ~ 2 px, inside the level-0 window
    errs = [float(m.group(9)) for m in first_pair if int(m.group(1)) in {by_kp[i].id for i in wrongly}]
    assert errs and 1.0 < np.median(errs) < np.sqrt(LocalMappingParams().reproj_chi2), np.median(errs)
    _assert_a7_identities(log, out)


# --------------------------------------------------------------------------- #
# reservation: confirm_candidates itself
# --------------------------------------------------------------------------- #
def test_confirm_candidates_reserves_a_keypoint_once_and_never_links():
    pts, descs, kfs = _window_scene(5, 0.03, noise_px=0.5)
    k4, k0, k2 = kfs[4], kfs[0], kfs[2]
    frames = list(kfs)
    world = Map(frames=frames)
    from local_mapping import _keypoint_sigmas, gate_candidates, match_pair
    params = LocalMappingParams(confirm=True)
    sigma_cache = {f.id: _keypoint_sigmas(f) for f in kfs}
    _, cands, _ = match_pair(k4, k0, sigma_cache[k4.id], sigma_cache[k0.id], params)
    X, _ = triangulate_pair(k4, k0, cands)
    counters = {k: 0 for k in ("rej_cheirality", "rej_parallax", "rej_chi2", "rej_scale")}
    accepted = gate_candidates(k4, k0, cands, X, sigma_cache[k4.id], sigma_cache[k0.id], params, counters)
    assert len(accepted) > 100
    # a duplicate of the first candidate competes for the same K_2 keypoint: the second copy
    # sees it reserved and (with no other keypoint in its window) is unconfirmed
    dup = list(accepted[:1]) + list(accepted)
    third_views = [(20, kfs[1]), (40, k2), (60, kfs[3])]
    confirmed, n_none = confirm_candidates(k4, k0, dup, third_views, sigma_cache, params)
    kps = [(v[0], v[2]) for _t, views in confirmed for v in views]
    assert len(kps) == len(set(kps))                       # one keypoint confirms one candidate
    assert all(v[0] == 40 for _t, views in confirmed for v in views)   # only K_2 is eligible
    assert all(0.3 <= v[5] <= 0.7 for _t, views in confirmed for v in views)
    assert n_none == 0
    firsts = [t for t, _v in confirmed if t is accepted[0]]
    assert len(firsts) == 1
    assert len(confirmed) >= 0.8 * len(accepted)
    # nothing was created or linked
    assert len(world.points) == 0 and all(f.keypoint_to_point == {} for f in kfs)
    # a candidate whose projection leaves every eligible image has no third view
    far = (accepted[0][0], accepted[0][1], np.array([50.0, 0.0, 2.0]), accepted[0][3])
    c2, n2 = confirm_candidates(k4, k0, [far], third_views, sigma_cache, params)
    assert c2 == [] and n2 == 1
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T6. parser and params
# --------------------------------------------------------------------------- #
def test_t6_parser_and_params(tmp_path):
    import main
    parser = main.get_parser()
    a = parser.parse_args(["--dataset", "rgb"])
    assert a.lm_confirm is False and a.lm_confirm_disparity_ratio == 0.3
    assert LocalMappingParams.from_args(a) == LocalMappingParams()
    b = parser.parse_args(["--dataset", "rgb", "--lm_confirm"])
    assert b.lm_confirm is True
    pb = LocalMappingParams.from_args(b)
    assert pb.confirm is True and pb.confirm_disparity_ratio == 0.3
    c = parser.parse_args(["--dataset", "rgb", "--lm_confirm", "--lm_confirm_disparity_ratio", "0.25"])
    assert c.lm_confirm_disparity_ratio == 0.25
    assert LocalMappingParams.from_args(c) == LocalMappingParams(confirm=True, confirm_disparity_ratio=0.25)
    out = tmp_path / "resolved_settings.json"
    with contextlib.redirect_stdout(io.StringIO()):
        main.write_resolved_settings(b, out)
    rs = json.loads(out.read_text())
    assert rs["lm_confirm"] is True and rs["lm_confirm_disparity_ratio"] == 0.3
    with contextlib.redirect_stdout(io.StringIO()):
        main.write_resolved_settings(a, out)
    assert json.loads(out.read_text())["lm_confirm"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
