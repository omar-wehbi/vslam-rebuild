"""
Stage 04k (spec T): epipolar_refine.py and its two hooks (local_mapping's pair
mapper, depth_filter.update_seeds), the switch and its hand-off.  Synthetic
textures and stub frames only; no dataset, no image file.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import camera  # noqa: E402
import depth_filter as df  # noqa: E402
import epipolar_refine as er  # noqa: E402
import local_mapping as lm  # noqa: E402
from epipolar_refine import (  # noqa: E402
    REFINE_COUNTER_KEYS,
    STATUS_KEY,
    RefineParams,
    distort_pixels,
    refine_along_epipolar,
    sample_patches,
    zncc,
)
from frame import DEFAULT_K  # noqa: E402  (the matrix main.py uses for --default_k)
from Map import Map  # noqa: E402
from test_depth_filter import _assert_counter_identities, sweep  # noqa: E402
from test_local_mapping import F_PX, H, K, W, _KF, _regime_scene  # noqa: E402

P = RefineParams()
EYE = np.eye(3)


def _texture(seed=5, shape=(200, 200), sigma=2.0):
    """Gaussian-blurred uniform noise, float32."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0.0, 255.0, shape).astype(np.float32), (0, 0), sigma)


def _shifted(img, shift_xy):
    """``img`` translated by ``shift_xy``: content at p in img appears at p + shift_xy."""
    M = np.float32([[1.0, 0.0, shift_xy[0]], [0.0, 1.0, shift_xy[1]]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# T1. distort_pixels
# --------------------------------------------------------------------------- #
def test_t1_distort_pixels_none_roundtrip_and_empty():
    q = np.array([[10.0, 20.0], [300.5, 200.25]])
    out = distort_pixels(q, DEFAULT_K, None)
    assert out.dtype == np.float64 and np.array_equal(out, q) and out is not q
    us, vs = np.meshgrid(np.linspace(20.0, 620.0, 20), np.linspace(20.0, 460.0, 15))
    grid = np.column_stack([us.ravel(), vs.ravel()])
    raw = distort_pixels(grid, DEFAULT_K, camera.DISTORTION_FR2)
    assert raw.shape == grid.shape
    assert np.max(np.abs(raw - grid)) > 1.0              # the lens moves the corners by pixels
    back = camera.undistort_points(raw, DEFAULT_K, camera.DISTORTION_FR2)
    # cv2.undistortPoints' default iterations leave up to 5.9e-3 px at the corners (spec T1)
    assert np.max(np.linalg.norm(back - grid, axis=1)) <= 1e-2
    assert distort_pixels(np.zeros((0, 2)), DEFAULT_K, camera.DISTORTION_FR2).shape == (0, 2)
    assert distort_pixels(np.zeros((0, 2)), DEFAULT_K, None).shape == (0, 2)


# --------------------------------------------------------------------------- #
# T2. zncc
# --------------------------------------------------------------------------- #
def test_t2_zncc_self_negative_constant():
    a = _texture()[50:59, 50:59].astype(np.float64)
    assert zncc(a, a) == pytest.approx(1.0, abs=1e-12)
    assert zncc(a, -a) == pytest.approx(-1.0, abs=1e-12)
    assert zncc(a, np.full_like(a, 7.0)) == -1.0
    assert zncc(np.full_like(a, 3.0), a) == -1.0


def test_sample_patches_grid_layout_and_bounds():
    img = np.add.outer(np.arange(60.0) * 100.0, np.arange(80.0)).astype(np.float32)   # value = 100 y + x
    p = sample_patches(img, EYE, None, np.array([[30.0, 20.0], [40.25, 30.5]]), 2)
    assert p.shape == (2, 5, 5) and p.dtype == np.float64
    assert p[0, 0, 0] == pytest.approx(100 * 18 + 28) and p[0, 0, 4] == pytest.approx(100 * 18 + 32)
    assert p[0, 4, 0] == pytest.approx(100 * 22 + 28)                   # row index = dy
    assert p[1, 2, 2] == pytest.approx(100 * 30.5 + 40.25, abs=0.1)      # bilinear
    assert sample_patches(img, EYE, None, np.array([[1.5, 20.0]]), 2) is None
    assert sample_patches(img, EYE, None, np.array([[77.0, 20.0]]), 2) is not None    # 79 = W - 1: inside
    assert sample_patches(img, EYE, None, np.array([[77.01, 20.0]]), 2) is None
    assert sample_patches(img, EYE, None, np.array([[30.0, 57.5]]), 2) is None
    assert sample_patches(img, EYE, None, np.array([[np.nan, 20.0]]), 2) is None
    # the window crop samples exactly as the whole image does
    tex = _texture(seed=9)
    c = np.array([[60.3, 70.7], [61.1, 71.9]])
    q = np.column_stack([np.repeat(c[:, 0], 1), np.repeat(c[:, 1], 1)])
    full = cv2.remap(tex, np.float32([[c[0, 0]]]), np.float32([[c[0, 1]]]), cv2.INTER_LINEAR)
    assert sample_patches(tex, EYE, None, q, 0)[0, 0, 0] == float(full[0, 0])


def test_sample_patches_distortion_samples_the_raw_image_at_the_distorted_grid():
    tex = _texture(seed=11, shape=(480, 640))
    c = np.array([[100.0, 80.0]])
    got = sample_patches(tex, DEFAULT_K, camera.DISTORTION_FR2, c, 1)
    offs = np.arange(-1, 2, dtype=np.float64)
    q = np.array([[c[0, 0] + dx, c[0, 1] + dy] for dy in offs for dx in offs])
    raw = distort_pixels(q, DEFAULT_K, camera.DISTORTION_FR2).astype(np.float32)
    want = cv2.remap(tex, raw[:, 0].reshape(3, 3), raw[:, 1].reshape(3, 3), cv2.INTER_LINEAR)
    np.testing.assert_array_equal(got[0], want.astype(np.float64))
    assert not np.array_equal(got, sample_patches(tex, DEFAULT_K, None, c, 1))


# --------------------------------------------------------------------------- #
# T3. refinement of a known sub-pixel shift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", [(0.8, 0.6), (1.0, 0.0), (-0.28, 0.96)])
def test_t3_recovers_a_known_subpixel_shift_along_the_direction(direction):
    tex = _texture()
    u = np.asarray(direction, dtype=np.float64)
    u = u / np.linalg.norm(u)
    xy_ref = np.array([100.0, 100.0])
    for s in (0.25, 0.5, -0.3, 1.7):
        tgt = _shifted(tex, s * u)
        true = xy_ref + s * u
        for off in (-2, 0, 2):
            xy_tgt = true + off * u
            xy_new, status, peak, t = refine_along_epipolar(tex, xy_ref, EYE, None, tgt, xy_tgt, EYE, None,
                                                            7.0 * u, P)       # the direction's length is free
            assert status == "refined", (s, off, status, peak)
            assert np.linalg.norm(xy_new - true) <= 0.15, (s, off, xy_new, true)
            assert peak >= P.min_peak and t == pytest.approx(-off, abs=0.15)
            # the move is along u only
            d = xy_new - xy_tgt
            assert abs(d[0] * u[1] - d[1] * u[0]) <= 1e-9


def test_t3_parabola_vertex_is_the_documented_formula():
    """The sub-pixel step: dt = 0.5 (s[k-1] - s[k+1]) / (s[k-1] - 2 s[k] + s[k+1]) (a wrong sign
    would move the answer away from the true shift; T3 above catches it on real textures)."""
    tex = _texture(seed=2)
    u = np.array([1.0, 0.0])
    tgt = _shifted(tex, 0.4 * u)
    xy_ref = np.array([100.0, 100.0])
    xy_new, status, peak, t = refine_along_epipolar(tex, xy_ref, EYE, None, tgt, xy_ref, EYE, None, u, P)
    ref = sample_patches(tex, EYE, None, xy_ref[None, :], 4)[0]
    s = np.arange(-6, 7, dtype=np.float64)
    tp = sample_patches(tgt, EYE, None, xy_ref[None, :] + s[:, None] * u[None, :], 4)
    sc = np.array([zncc(ref, tp[i]) for i in range(13)])
    k = int(np.argmax(sc))
    dt = 0.5 * (sc[k - 1] - sc[k + 1]) / (sc[k - 1] - 2 * sc[k] + sc[k + 1])
    assert status == "refined" and k == 6 and peak == sc[k]
    assert t == pytest.approx((k - 6) + dt, abs=1e-12)
    assert 0.2 < t < 0.6


# --------------------------------------------------------------------------- #
# T4. statuses
# --------------------------------------------------------------------------- #
def test_t4_stripes_with_a_4px_period_are_ambiguous():
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float64)
    stripes = (128.0 + 100.0 * np.cos(2.0 * np.pi * xx / 4.0)).astype(np.float32)
    xy = np.array([100.0, 100.0])
    xy_new, status, peak, t = refine_along_epipolar(stripes, xy, EYE, None, stripes, xy, EYE, None,
                                                    np.array([1.0, 0.0]), P)
    # maxima at s = -4, 0, 4 with equal scores: a repeat, dropped rather than guessed
    assert status == "ambiguous" and xy_new is None and peak > 0.99
    # the same stripes seen across the direction are not ambiguous along it... they are constant
    # along it, so every patch correlates perfectly: the second maximum test still refuses
    xy_new, status, _, _ = refine_along_epipolar(stripes, xy, EYE, None, stripes, xy, EYE, None,
                                                 np.array([0.0, 1.0]), P)
    assert xy_new is None and status in ("ambiguous", "edge")


def test_t4_true_position_beyond_the_search_is_edge():
    tex = _texture()
    u = np.array([1.0, 0.0])
    xy_ref = np.array([100.0, 100.0])
    # true position 7 px from xy_tgt: the best sample is s = 6, the end of the search -> edge
    xy_new, status, peak, t = refine_along_epipolar(tex, xy_ref, EYE, None, tex, xy_ref - 7.0 * u, EYE, None, u, P)
    assert status == "edge" and xy_new is None and t == 0.0
    xy_new, status, _, _ = refine_along_epipolar(tex, xy_ref, EYE, None, tex, xy_ref + 7.0 * u, EYE, None, u, P)
    assert status == "edge"
    # true position 5.5 px away: s = 5 and s = 6 sit 0.5 px either side of it, so their scores are
    # (nearly) equal and the argmax may land on either. On s = 6 (index 2R) it is 'edge'; on s = 5 the
    # parabola through s = 4, 5, 6 puts the vertex near 5.5 and the result is 'refined' at the truth.
    tgt = _shifted(tex, 0.5 * u)
    true = xy_ref + 0.5 * u
    xy_new, status, _, t = refine_along_epipolar(tex, xy_ref, EYE, None, tgt, true - 5.5 * u, EYE, None, u, P)
    assert status in ("edge", "refined")
    if status == "refined":
        assert np.linalg.norm(xy_new - true) <= 0.15


def test_t4_oob_no_image_degenerate_and_unrelated():
    tex = _texture()
    u = np.array([1.0, 0.0])
    xy = np.array([100.0, 100.0])
    near_border = np.array([5.0, 100.0])       # within 5 px of the border along u: 6 + 4 px needed
    assert refine_along_epipolar(tex, xy, EYE, None, tex, near_border, EYE, None, u, P)[:2] == (None, "oob")
    # the reference patch alone (4 px half-width) outside its image
    assert refine_along_epipolar(tex, np.array([3.0, 100.0]), EYE, None, tex, xy, EYE, None, u, P)[:2] == (None, "oob")
    assert refine_along_epipolar(tex, xy, EYE, None, tex, np.array([196.0, 100.0]), EYE, None, u, P)[1] == "oob"
    assert refine_along_epipolar(None, xy, EYE, None, tex, xy, EYE, None, u, P) == (None, "no_image", -1.0, 0.0)
    assert refine_along_epipolar(tex, xy, EYE, None, None, xy, EYE, None, u, P) == (None, "no_image", -1.0, 0.0)
    assert refine_along_epipolar(tex, xy, EYE, None, tex, xy, EYE, None, (0.0, 0.0), P) == \
        (None, "degenerate", -1.0, 0.0)
    assert refine_along_epipolar(tex, xy, EYE, None, tex, xy, EYE, None, (np.nan, 1.0), P)[1] == "degenerate"
    # an unrelated texture as target: no refined position.  Fine-grained textures (blur sigma
    # 0.7 px) are used here: a 9 x 9 patch of a sigma-2 texture holds only a handful of independent
    # samples, and two unrelated sigma-2 textures reach ZNCC >= 0.6 somewhere among 13 positions for
    # about a quarter of the positions (measured while building this test; a property of the
    # frozen constants on very smooth texture, recorded, not tuned).
    tex_f, other = _texture(seed=5, sigma=0.7), _texture(seed=99, sigma=0.7)
    for y in range(40, 170, 13):
        for x in range(40, 170, 17):
            p = np.array([float(x), float(y)])
            xy_new, status, peak, _ = refine_along_epipolar(tex_f, p, EYE, None, other, p, EYE, None, u, P)
            assert xy_new is None and status in ("low_peak", "edge", "ambiguous"), (x, y, status, peak)
    # and the same fine texture against itself shifted is refined
    xy_new, status, _, _ = refine_along_epipolar(tex_f, xy, EYE, None, _shifted(tex_f, 0.5 * u), xy, EYE, None, u, P)
    assert status == "refined" and np.linalg.norm(xy_new - (xy + 0.5 * u)) <= 0.15


def test_t4_low_peak_is_refused_before_the_ambiguity_test():
    tex = _texture()
    u = np.array([1.0, 0.0])
    xy = np.array([100.0, 100.0])
    strict = RefineParams(min_peak=1.01)
    xy_new, status, peak, t = refine_along_epipolar(tex, xy, EYE, None, _shifted(tex, 0.3 * u), xy, EYE, None, u, strict)
    assert status == "low_peak" and xy_new is None and 0.9 < peak < 1.01


def test_refine_with_distortion_recovers_a_shift_rendered_in_raw_pixels():
    """S-c: the images are raw; keypoints are pinhole.  Render a raw target whose pinhole content
    is the reference's shifted by s along u, refine in pinhole pixels with D = fr2."""
    tex = _texture(seed=4, shape=(480, 640))
    D = camera.DISTORTION_FR2
    Kd = DEFAULT_K
    s, u = 0.6, np.array([0.6, 0.8])
    # pinhole image of the reference: tex itself in pinhole coordinates; the raw reference is its
    # distortion (raw(r) = pin(undistort(r))), and the raw target is the same with a shift.
    ys, xs = np.mgrid[0:480, 0:640].astype(np.float64)
    r = np.column_stack([xs.ravel(), ys.ravel()])
    pin = camera.undistort_points(r, Kd, D)
    mx = pin[:, 0].reshape(480, 640).astype(np.float32)
    my = pin[:, 1].reshape(480, 640).astype(np.float32)
    raw_ref = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    raw_tgt = cv2.remap(tex, mx - np.float32(s * u[0]), my - np.float32(s * u[1]), cv2.INTER_LINEAR)
    xy_ref = np.array([150.0, 120.0])          # off-centre, where the lens is not the identity
    true = xy_ref + s * u
    xy_new, status, _, _ = refine_along_epipolar(raw_ref, xy_ref, Kd, D, raw_tgt, true + 2.0 * u, Kd, D, u, P)
    assert status == "refined" and np.linalg.norm(xy_new - true) <= 0.15


# --------------------------------------------------------------------------- #
# T5. depth_filter.update_seeds with the refine keyword
# --------------------------------------------------------------------------- #
def _seed_state(seeds):
    return [(s.mu, s.sigma2, s.a, s.b, s.n_updates,
             [(m.traj_idx, m.kp, m.rho, m.tau2, m.resp, m.desc_dist) for m in s.measurements]) for s in seeds]


def _run_sweep(refine_kw):
    pts, poses, frames, _ = sweep(0.009, n_frames=3, noise_px=1.0)
    Pdf = df.DepthFilterParams()
    seeds = df.make_seeds(frames[0], 0, 2.5, Pdf)
    counters = []
    for k in range(1, 4):
        if refine_kw is None:
            c = df.update_seeds(seeds, frames[k], k, Pdf)
        else:
            c = df.update_seeds(seeds, frames[k], k, Pdf, **refine_kw)
        _assert_counter_identities(c)
        counters.append(c)
    return pts, poses, frames, seeds, counters


def test_t5_refine_none_is_the_call_without_the_keyword():
    _, _, _, s0, c0 = _run_sweep(None)
    _, _, _, s1, c1 = _run_sweep({"refine": None})
    assert c0 == c1 and all(set(c) == set(df.COUNTER_KEYS) for c in c1)
    assert _seed_state(s0) == _seed_state(s1)
    assert sum(c["measured"] for c in c0) > 0


def test_t5_stub_frames_without_images_refuse_every_match_as_no_image():
    _, _, _, _, c_off = _run_sweep(None)
    _, _, _, seeds, c_on = _run_sweep({"refine": RefineParams()})
    for c in c_on:
        assert all(key in c for key in REFINE_COUNTER_KEYS)
    # frame 1: the same seed states as the switch-off run, so the same matches were attempted
    c = c_on[0]
    assert c["refine_attempted"] == c["refine_no_image"] > 0
    assert c["measured"] == 0 and c["inlier"] == 0 and c["outlier"] == 0
    assert c["no_match"] == c_off[0]["no_match"] + c_off[0]["measured"] + c_off[0]["no_disparity"]
    assert c["no_match"] - c_off[0]["no_match"] == c["refine_no_image"]
    assert all(s.n_updates == 0 for s in seeds)
    assert sum(c["refine_attempted"] for c in c_on) == sum(c["refine_no_image"] for c in c_on)
    # the early return keeps the refinement counters (no unlinked keypoint in the frame)
    Pdf = df.DepthFilterParams()
    pts, poses, frames, _ = sweep(0.009, n_frames=1)
    empty = _KF(60, poses[1], [], None)
    c = df.update_seeds(df.make_seeds(frames[0], 0, 2.5, Pdf), empty, 1, Pdf, refine=RefineParams())
    assert all(c[key] == 0 for key in REFINE_COUNTER_KEYS) and c["no_match"] == 300


def _true_pixel_refiner(frames_by_id):
    """A stand-in for refine_along_epipolar returning the target frame's noise-free projection of
    the matched keypoint (true_uv), found by position; lets the test see which pixel step (d) reads."""
    def fake(img_ref, xy_ref, K_ref, D_ref, img_tgt, xy_tgt, K_tgt, D_tgt, direction, params):
        fr = frames_by_id[img_tgt]
        j = int(np.argmin(np.linalg.norm(np.array([kp.pt for kp in fr.KeyPoints]) - xy_tgt, axis=1)))
        return np.array(fr.true_uv[j], dtype=np.float64), "refined", 0.95, 0.5
    return fake


def test_t5_a_refined_measurement_is_triangulated_from_the_refined_pixel(monkeypatch):
    pts, poses, frames, _ = sweep(0.009, n_frames=3, noise_px=1.0)
    for i, fr in enumerate(frames):
        fr.img = i                              # a token: the stand-in maps it back to the frame
    monkeypatch.setattr(df, "refine_along_epipolar", _true_pixel_refiner({i: fr for i, fr in enumerate(frames)}))
    Pdf = df.DepthFilterParams()
    seeds = df.make_seeds(frames[0], 0, 2.5, Pdf)
    c = df.update_seeds(seeds, frames[3], 3, Pdf, refine=RefineParams())
    _assert_counter_identities(c)
    assert c["refine_attempted"] == c["refine_ok"] > 100
    n_checked = 0
    for s in seeds:
        for m in s.measurements:
            if not (m.rho > 0.0):
                continue
            j = m.kp
            assert m.kp == s.kp                  # the matched keypoint index, not a new one
            # the reference ray against the ray through the frame's TRUE pixel of keypoint j
            d = s.ray
            xyt = frames[3].true_uv[j]
            dF = np.linalg.inv(K) @ np.array([xyt[0], xyt[1], 1.0])
            dF = frames[3].world_T_cam[:3, :3] @ (dF / np.linalg.norm(dF))
            b = frames[3].world_T_cam[:3, 3] - s.centre
            cd, p, q = float(d @ dF), float(d @ b), float(dF @ b)
            r_R = (p - cd * q) / (1.0 - cd * cd)
            assert 1.0 / m.rho == pytest.approx(r_R, rel=1e-9)
            n_checked += 1
    assert n_checked > 100
    # tau2 kept the keypoint's sigma: equal to measurement_tau2 at the measured range, sigma 1 px
    s = next(s for s in seeds if s.measurements and s.measurements[0].rho > 0.0 and np.isfinite(s.measurements[0].tau2))
    m = s.measurements[0]
    assert m.tau2 == pytest.approx(df.measurement_tau2(s, frames[3], 1.0 / m.rho, 1.0), rel=1e-12)


def test_t5_identity_refinement_reproduces_the_switch_off_state(monkeypatch):
    """A refinement that returns xy_tgt itself changes nothing but the counters (rules out a
    refined path that reads anything but the refined pixel); a refused one drops the seed."""
    _, _, _, s_off, c_off = _run_sweep(None)
    monkeypatch.setattr(df, "refine_along_epipolar",
                        lambda ir, xr, Kr, Dr, it, xt, Kt, Dt, d, p: (np.array(xt, dtype=np.float64), "refined", 1.0, 0.0))
    _, _, _, s_on, c_on = _run_sweep({"refine": RefineParams()})
    assert _seed_state(s_off) == _seed_state(s_on)
    for a, b in zip(c_off, c_on):
        assert {k: b[k] for k in df.COUNTER_KEYS} == a
    calls = []

    def half(ir, xr, Kr, Dr, it, xt, Kt, Dt, d, p):
        calls.append(np.asarray(d, dtype=np.float64))
        return ((np.array(xt, dtype=np.float64), "refined", 1.0, 0.0) if len(calls) % 2
                else (None, "ambiguous", 0.95, 0.0))
    monkeypatch.setattr(df, "refine_along_epipolar", half)
    _, _, _, _, c_half = _run_sweep({"refine": RefineParams()})
    c = c_half[0]
    assert c["refine_attempted"] == c["refine_ok"] + c["refine_ambiguous"] and c["refine_ambiguous"] > 0
    assert c["no_match"] == c_off[0]["no_match"] + c["refine_ambiguous"]
    assert all(np.linalg.norm(d) > 0.0 for d in calls)


# --------------------------------------------------------------------------- #
# T6. _KeypointOverrideView
# --------------------------------------------------------------------------- #
def test_t6_keypoint_override_view():
    _, _, k0, _ = _regime_scene()
    k0.KeyPoints[3].angle = 12.5
    k0.KeyPoints[3].response = 0.25
    k0.KeyPoints[3].octave = 255
    k0.KeyPoints[3].class_id = 7
    before = [(kp.pt, kp.size) for kp in k0.KeyPoints]
    originals = list(k0.KeyPoints)
    k0.keypoint_to_point[9] = 42
    view = lm._KeypointOverrideView(k0, {3: (101.25, 55.5)})
    kp = view.KeyPoints[3]
    assert kp.pt == pytest.approx((101.25, 55.5))
    assert (kp.size, kp.angle, kp.response, kp.octave, kp.class_id) == \
        (originals[3].size, originals[3].angle, originals[3].response, originals[3].octave, originals[3].class_id)
    assert kp is not originals[3]
    assert all(view.KeyPoints[i] is originals[i] for i in range(len(originals)) if i != 3)
    assert view.K is k0.K and view.id == k0.id and view.cam_T_world is k0.cam_T_world
    assert view.keypoint_to_point is k0.keypoint_to_point and view.Descriptors is k0.Descriptors
    assert k0.KeyPoints is not view.KeyPoints and [(kp.pt, kp.size) for kp in k0.KeyPoints] == before
    assert all(k0.KeyPoints[i] is originals[i] for i in range(len(originals)))
    np.testing.assert_allclose(lm._kp_xy(view)[3], [101.25, 55.5])


# --------------------------------------------------------------------------- #
# the pair mapper hook (spec F2)
# --------------------------------------------------------------------------- #
RX_REFINE_PAIR = re.compile(r"^\s*>> \[LOCAL-MAP-REFINE-PAIR\] kf=(\d+) partner=(\d+) attempted=(\d+) refined=(\d+) "
                            r"ambiguous=(\d+) low_peak=(\d+) edge=(\d+) oob=(\d+) no_image=(\d+) degenerate=(\d+)$")
RX_REFINE = re.compile(r"^\s*>> \[LOCAL-MAP-REFINE\] id=(\d+) kf=(\d+) partner=(\d+) kp_n=(\d+) kp_c=(\d+) "
                       r"u_c=(-?\d+\.\d\d) v_c=(-?\d+\.\d\d) ur_c=(-?\d+\.\d{3}) vr_c=(-?\d+\.\d{3}) "
                       r"t=(-?\d+\.\d{3}) peak=(-?\d+\.\d{3})$")
RX_POINT = re.compile(r"^\s*>> \[LOCAL-MAP-POINT\] id=(\d+) kf=(\d+) partner=(\d+) kp_n=(\d+) kp_c=(\d+) "
                      r"u_n=(\S+) v_n=(\S+) u_c=(\S+) v_c=(\S+) ")


def _run_pair(refine, confirm=False, scene=None):
    pts, descs, k0, k1 = scene if scene is not None else _regime_scene()
    frames = [k0, k1]
    world = Map(frames=frames)
    buf = io.StringIO()
    params = lm.LocalMappingParams(confirm=confirm)
    with contextlib.redirect_stdout(buf):
        out = lm.process_new_keyframe(1, k1, [(0, k0), (1, k1)], world, frames, params, None, refine=refine)
    return pts, k0, k1, world, out, buf.getvalue()


def test_pair_refine_none_is_the_call_without_the_keyword():
    pts, descs, k0, k1 = _regime_scene()
    frames = [k0, k1]
    world = Map(frames=frames)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out0 = lm.process_new_keyframe(1, k1, [(0, k0), (1, k1)], world, frames, lm.LocalMappingParams(), None)
    _, _, _, world1, out1, log1 = _run_pair(None)
    ids = lambda log: re.sub(r"id=\d+", "id=#", log)
    assert out0 == out1 and ids(buf.getvalue()) == ids(log1) and "REFINE" not in log1
    assert [p.point.tolist() for p in world.points] == [p.point.tolist() for p in world1.points]


def test_pair_stub_frames_without_images_drop_every_candidate():
    _, _, _, _, out_off, _ = _run_pair(None)
    _, _, _, world, out, log = _run_pair(RefineParams())
    rows = [RX_REFINE_PAIR.match(l) for l in log.splitlines() if "[LOCAL-MAP-REFINE-PAIR]" in l]
    assert len(rows) == 1 and rows[0] is not None
    g = [int(x) for x in rows[0].groups()]
    assert g[:2] == [1, 0] and g[2] == g[8] > 0.75 * out_off["candidates"] and g[3] == 0
    assert out["candidates"] == 0 and out["created"] == 0 and len(world.points) == 0
    assert log.index("[LOCAL-MAP-REFINE-PAIR]") < log.index("[LOCAL-MAP-PARALLAX]")


def test_pair_refined_pixel_feeds_triangulation_and_the_keypoint_feeds_the_map(monkeypatch):
    pts, descs, k0, k1 = _regime_scene()
    k0.img, k1.img = 0, 1
    fake = _true_pixel_refiner({0: k0, 1: k1})
    monkeypatch.setattr(lm, "refine_along_epipolar", fake)
    kp_before = [kp.pt for kp in k0.KeyPoints]
    _, _, _, world, out, log = _run_pair(RefineParams(), scene=(pts, descs, k0, k1))
    assert [kp.pt for kp in k0.KeyPoints] == kp_before                   # K_c never mutated
    lines = log.splitlines()
    point_rows = {int(m.group(1)): m for m in (RX_POINT.match(l) for l in lines) if m}
    ref_rows = [RX_REFINE.match(l) for l in lines if "[LOCAL-MAP-REFINE]" in l]
    assert all(r is not None for r in ref_rows)
    assert out["created"] > 0.7 * len(pts) and len(ref_rows) == len(point_rows) == out["created"]
    for r in ref_rows:
        pid, kf, partner, i, j = (int(x) for x in r.groups()[:5])
        pr = point_rows[pid]
        assert (kf, partner, i, j) == (1, 0, int(pr.group(4)), int(pr.group(5)))
        assert r.group(6) == pr.group(8) and r.group(7) == pr.group(9)        # u_c v_c: the keypoint pixel
        assert float(r.group(6)) == pytest.approx(k0.KeyPoints[j].pt[0], abs=0.006)
        assert float(r.group(8)) == pytest.approx(k0.true_uv[j][0], abs=6e-4)  # ur_c: the refined pixel
        assert float(r.group(9)) == pytest.approx(k0.true_uv[j][1], abs=6e-4)
        assert float(r.group(10)) == 0.5 and float(r.group(11)) == 0.95
        # each [LOCAL-MAP-REFINE] line follows its [LOCAL-MAP-POINT] line
        assert log.index(f"[LOCAL-MAP-REFINE] id={pid} ") > log.index(f"[LOCAL-MAP-POINT] id={pid} ")
    # the map links keypoint indices (the real frames) and the point was triangulated from the
    # true K_c pixel: its reprojection into K_c sits on the truth, not on the noisy keypoint
    res_true, res_kp = [], []
    for p in world.points:
        j = p.observations[k0.id]
        assert k0.keypoint_to_point[j] == p.id and k1.keypoint_to_point[p.observations[k1.id]] == p.id
        xc = k0.cam_T_world[:3, :3] @ p.point + k0.cam_T_world[:3, 3]
        uv = (K @ xc)[:2] / (K @ xc)[2]
        res_true.append(np.linalg.norm(uv - k0.true_uv[j]))
        res_kp.append(np.linalg.norm(uv - np.array(k0.KeyPoints[j].pt)))
    assert np.median(res_true) < 0.5 * np.median(res_kp)


def test_pair_refusals_drop_candidates_and_the_confirmation_reads_the_view(monkeypatch):
    pts, descs, k0, k1 = _regime_scene()
    seen = []

    def odd_refused(img_ref, xy_ref, K_ref, D_ref, img_tgt, xy_tgt, K_tgt, D_tgt, direction, params):
        seen.append((np.asarray(xy_ref).copy(), np.asarray(direction, dtype=np.float64)))
        if len(seen) % 2 == 0:
            return None, "low_peak", 0.3, 0.0
        return np.array(xy_tgt, dtype=np.float64), "refined", 0.9, 0.0
    monkeypatch.setattr(lm, "refine_along_epipolar", odd_refused)
    _, _, _, world, out, log = _run_pair(RefineParams(), scene=(pts, descs, k0, k1))
    g = [int(x) for x in RX_REFINE_PAIR.match(next(l for l in log.splitlines() if "REFINE-PAIR" in l)).groups()]
    assert g[2] == len(seen) and g[3] == (len(seen) + 1) // 2 and g[5] == len(seen) // 2
    assert out["candidates"] == g[3]
    # the direction is the epipolar line of x_n in K_c: F x_n . (x_c + t direction) stays 0
    F = lm.fundamental_matrix(k0.K, k1.K, k0.cam_T_world @ k1.world_T_cam)
    for xy_n, d in seen[:20]:
        l = F @ np.array([xy_n[0], xy_n[1], 1.0])
        assert abs(l[0] * d[0] + l[1] * d[1]) <= 1e-9 * np.linalg.norm(l[:2]) * np.linalg.norm(d)


# --------------------------------------------------------------------------- #
# end to end on rendered images: a textured fronto-parallel plane
# --------------------------------------------------------------------------- #
def _plane_scene(noise_px=0.7, seed=13, z=2.0, b=0.1):
    """Two keyframes 0.1 m apart along x looking at a textured plane at z = 2 m: the new keyframe's
    image is the partner's shifted by f b / z px.  Keypoints at the true projections + noise."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.uniform(0.0, 255.0, (H, W)).astype(np.float32), (0, 0), 2.0)
    shift = F_PX * b / z
    img1 = _shifted(tex, (-shift, 0.0))
    T0, T1 = np.eye(4), np.eye(4)
    T1[0, 3] = b
    pts = []
    for v in np.linspace(40.0, H - 40.0, 12):
        for u in np.linspace(40.0, W - 40.0 - shift, 16):
            pts.append([(u - K[0, 2]) * z / F_PX + b, (v - K[1, 2]) * z / F_PX, z])
    pts = np.array(pts)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=noise_px, rng=rng)
    k1 = _KF(2, T1, pts, descs, noise_px=noise_px, rng=rng)
    k0.img, k1.img = tex.astype(np.uint8), img1.astype(np.uint8)
    k0.distortion = k1.distortion = None
    return pts, descs, k0, k1, shift


def test_pair_end_to_end_on_a_rendered_plane_moves_the_partner_pixel_to_the_correspondence():
    pts, descs, k0, k1, shift = _plane_scene()
    _, _, _, w_off, out_off, _ = _run_pair(None, scene=(pts, descs, k0, k1))
    pts, descs, k0, k1, shift = _plane_scene()
    _, _, _, w_on, out_on, log = _run_pair(RefineParams(), scene=(pts, descs, k0, k1))
    g = [int(x) for x in RX_REFINE_PAIR.match(next(l for l in log.splitlines() if "REFINE-PAIR" in l)).groups()]
    assert g[3] >= 0.8 * g[2] and g[8] == 0
    rows = [RX_REFINE.match(l) for l in log.splitlines() if "[LOCAL-MAP-REFINE]" in l]
    err = []
    for r in rows:
        i, j = int(r.group(4)), int(r.group(5))
        # the correspondence of the new keyframe's (noisy) keypoint i on the plane is x_n + shift in u
        err.append(abs(float(r.group(8)) - (k1.KeyPoints[i].pt[0] + shift)))
    assert len(rows) == out_on["created"] > 0.7 * len(pts)
    assert np.median(err) <= 0.15, np.median(err)

    def rel(world):
        return np.median([abs(p.point[2] - 2.0) / 2.0 for p in world.points])
    # the disparity is now the true one: depth on the plane within a fraction of the unrefined error
    assert rel(w_on) < 0.4 * rel(w_off) and rel(w_on) < 0.01, (rel(w_on), rel(w_off))


# --------------------------------------------------------------------------- #
# T7. the parser and the hand-off
# --------------------------------------------------------------------------- #
def test_t7_parser_and_run_pipeline_hand_off(monkeypatch):
    import main as main_mod
    p = main_mod.get_parser()
    assert p.parse_args(["--dataset", "rgb"]).epi_refine is False
    assert p.parse_args(["--dataset", "rgb", "--epi_refine"]).epi_refine is True
    calls = []
    monkeypatch.setattr(main_mod, "iter_frames", lambda *a, **kw: (None, 0, iter([])))
    monkeypatch.setattr(main_mod, "show_trajectory", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(main_mod, "write_resolved_settings", lambda args, out_path: None)
    main_mod.run_pipeline(p.parse_args(["--dataset", "rgb", "--no_display"]))
    main_mod.run_pipeline(p.parse_args(["--dataset", "rgb", "--no_display", "--epi_refine"]))
    assert calls[0]["refine_params"] is None
    assert isinstance(calls[1]["refine_params"], RefineParams) and calls[1]["refine_params"] == RefineParams()
    # a hand-built namespace without the attribute is off
    ns = p.parse_args(["--dataset", "rgb", "--no_display"])
    del ns.epi_refine
    main_mod.run_pipeline(ns)
    assert calls[2]["refine_params"] is None


def test_t7_resolved_settings_record_the_switch(tmp_path):
    import json
    import main as main_mod
    a = main_mod.get_parser().parse_args(["--dataset", "rgb", "--epi_refine"])
    with contextlib.redirect_stdout(io.StringIO()):
        main_mod.write_resolved_settings(a, tmp_path / "rs.json")
    assert json.load(open(tmp_path / "rs.json"))["epi_refine"] is True


# --------------------------------------------------------------------------- #
# T8. constants, keys, no-leak tokens
# --------------------------------------------------------------------------- #
def test_t8_params_keys_and_tokens():
    R = RefineParams()
    assert (R.patch_half, R.search_px, R.min_peak, R.ambiguity_ratio) == (4, 6, 0.6, 0.9)
    assert list(RefineParams.__dataclass_fields__) == ["patch_half", "search_px", "min_peak", "ambiguity_ratio"]
    with pytest.raises(Exception):
        R.patch_half = 5                         # frozen
    assert REFINE_COUNTER_KEYS == ("refine_attempted", "refine_ok", "refine_ambiguous", "refine_low_peak",
                                   "refine_edge", "refine_oob", "refine_no_image", "refine_degenerate")
    assert STATUS_KEY == {"refined": "refine_ok", "ambiguous": "refine_ambiguous", "low_peak": "refine_low_peak",
                          "edge": "refine_edge", "oob": "refine_oob", "no_image": "refine_no_image",
                          "degenerate": "refine_degenerate"}
    src = (ROOT / "epipolar_refine.py").read_text()
    for tok in ("import local_mapping", "from local_mapping", "import depth_filter", "from depth_filter",
                "import pipeline", "from pipeline", "groundtruth", "depth.txt", "import dataset", "from dataset",
                "cv2.imread", "world_map"):
        assert tok not in src, tok
    dsrc = (ROOT / "depth_filter.py").read_text()
    for tok in ("lifetime", "betainc", "dormant", "world_map", "groundtruth", "depth.txt", "depth/"):
        assert tok not in dsrc, tok


def test_df_arm_prints_the_refine_line_only_with_the_switch(monkeypatch):
    import depth_filter_arm as dfa
    pts, poses, frames, _ = sweep(0.009, n_frames=2)
    lines = {}
    for tag, refine in (("off", None), ("on", RefineParams())):
        arm = dfa.DepthFilterArm(refine=refine)
        assert arm.refine is refine
        arm.live = df.make_seeds(frames[0], 0, 2.5, arm.df_params)
        world = Map(frames=frames)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            arm.on_frame(1, frames[1], [(0, frames[0])], world, frames)
        lines[tag] = buf.getvalue().splitlines()
    assert not any("[DF-REFINE]" in l for l in lines["off"])
    on = lines["on"]
    i = next(n for n, l in enumerate(on) if "[DF-FRAME]" in l)
    m = re.match(r"^\s*>> \[DF-REFINE\] frame=1 attempted=(\d+) refined=(\d+) ambiguous=(\d+) low_peak=(\d+) "
                 r"edge=(\d+) oob=(\d+) no_image=(\d+) degenerate=(\d+)$", on[i + 1])
    assert m is not None
    g = [int(x) for x in m.groups()]
    assert g[0] == sum(g[1:]) == g[6] > 0          # stub frames have no image
