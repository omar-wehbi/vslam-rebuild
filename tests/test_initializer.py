"""Stage 04s: the two-view initializer (initializer.py), not wired into the pipeline.

I1 InitParams and InitResult; I2 the motion hypotheses; I3 the H and F scores; I4 check_rt;
I5 a general scene is accepted with F; I6 pure rotation is refused with no point; I7 distinctness
(spec E1); I8 spatial support (spec E7); I9 determinism; I10 the [INIT-PAIR] line; I11
initialize_frames on stub frames equals initialize on the same arrays. Plus: the H branch on a
plane scene, per-keypoint sigma in the scores, and the refusal reasons matches and scores.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import initializer as ini  # noqa: E402

K = np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])
KI = np.linalg.inv(K)
W, H = 640, 480


def rot(axis, deg):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    S = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * S + (1.0 - math.cos(th)) * S @ S


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def dir_err_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(math.acos(min(1.0, max(-1.0, float(a @ b)))))


def project(X):
    p = X @ K.T
    return p[:, :2] / p[:, 2:3]


def scene(seed, R, t, n=300, noise=0.5, out_frac=0.2, plane=False, band_u=None):
    """n matches of points uniform in image a at 2-8 m (or on the floor y = 1 m), visible in b."""
    rng = np.random.default_rng(seed)
    X, ua, ub = [], [], []
    while len(X) < n:
        if plane:
            P = np.array([rng.uniform(-3.0, 3.0), 1.0, rng.uniform(2.0, 8.0)])
            pa = project(P[None])[0]
            if not (0 <= pa[0] < W and 0 <= pa[1] < H):
                continue
        else:
            pa = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            P = rng.uniform(2.0, 8.0) * (KI @ np.array([pa[0], pa[1], 1.0]))
        Pb = R @ P + t
        if Pb[2] <= 0.1:
            continue
        pb = project(Pb[None])[0]
        if not (0 <= pb[0] < W and 0 <= pb[1] < H):
            continue
        na, nb = rng.normal(0.0, noise, 2), rng.normal(0.0, noise, 2)
        if band_u is not None and pb[0] + nb[0] >= band_u:
            continue
        X.append(P)
        ua.append(pa + na)
        ub.append(pb + nb)
    X, ua, ub = np.array(X), np.array(ua), np.array(ub)
    k = int(out_frac * n)
    out = rng.choice(n, k, replace=False)
    ub[out] = rng.uniform((0.0, 0.0), (band_u if band_u is not None else W, H), (k, 2))
    outlier = np.zeros(n, dtype=bool)
    outlier[out] = True
    return dict(X=X, uv_a=ua, uv_b=ub, sigma=np.ones(n), desc=rng.uniform(0.0, 1.0, n), outlier=outlier,
                R=R, t=t)


R_GEN = rot([0.2, 1.0, 0.1], 10.0)
T_GEN = 0.5 * np.array([0.9, 0.1, 0.3]) / np.linalg.norm([0.9, 0.1, 0.3])


@pytest.fixture(scope="module")
def gen():
    return scene(12, R_GEN, T_GEN)


@pytest.fixture(scope="module")
def gen_result(gen):
    s = gen["sigma"]
    return ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H))


# --------------------------------------------------------------------------- I1
def test_i1_params_and_result_fields():
    want = [("iterations", int, 200), ("seed", int, 0), ("min_matches", int, 100), ("chi2_h", float, 5.991),
            ("chi2_f", float, 3.841), ("score_gamma", float, 5.991), ("rh_select", float, 0.45),
            ("reproj_sq_sigmas", float, 4.0), ("infinite_cos", float, 0.99998), ("min_triangulated", int, 50),
            ("good_frac", float, 0.9), ("f_similar_frac", float, 0.7), ("h_second_frac", float, 0.75),
            ("min_parallax_deg", float, 1.0), ("h_degenerate_ratio", float, 1.00001),
            ("theta_min_deg", float, 1.1), ("ba_iterations", int, 20), ("min_points", int, 50),
            ("grid_cols", int, 4), ("grid_rows", int, 3), ("min_rows", int, 2), ("min_cols", int, 2),
            ("dup_tol_px", float, 0.001)]
    fields = dataclasses.fields(ini.InitParams)
    assert [f.name for f in fields] == [w[0] for w in want]
    P = ini.InitParams()
    for name, typ, val in want:
        got = getattr(P, name)
        assert type(got) is typ and got == val, name
    assert [f.name for f in dataclasses.fields(ini.InitResult)] == [
        "status", "reason", "model", "rh", "score_h", "score_f", "n_matches", "n_inliers", "good", "chosen",
        "parallax51_deg", "n_triangulated", "R", "t", "points", "match_idx", "kp_a", "kp_b", "parallax_deg",
        "rows", "cols", "cells", "ba_cost_initial", "ba_cost_final", "ba_iterations"]


# --------------------------------------------------------------------------- I2
def test_i2_essential_hypotheses_contain_truth():
    R0 = rot([0.3, -0.5, 0.8], 12.0)
    t0 = np.array([0.4, -0.1, 0.9])
    hyps = ini.essential_hypotheses(skew(t0) @ R0)
    assert len(hyps) == 4
    for R, _ in hyps:
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
    tu = t0 / np.linalg.norm(t0)
    best = min(max(math.radians(rot_err_deg(R, R0)), np.linalg.norm(t - tu)) for R, t in hyps)
    assert best < 1e-9
    # the order of DecomposeE: (R1, t), (R2, t), (R1, -t), (R2, -t)
    assert np.array_equal(hyps[0][0], hyps[2][0]) and np.array_equal(hyps[1][0], hyps[3][0])
    assert np.array_equal(hyps[0][1], -hyps[2][1]) and np.array_equal(hyps[0][1], hyps[1][1])


def test_i2_homography_hypotheses_plane_and_rotation():
    R0 = rot([0.0, 1.0, 0.0], 8.0)
    t0 = np.array([0.4, 0.05, 0.2])
    n_, d = np.array([0.0, 1.0, 0.0]), 1.0
    Hm = K @ (R0 + np.outer(t0, n_) / d) @ KI
    for scale in (1.0 / Hm[2, 2], -2.5):          # the decomposition does not depend on the scale of H
        hyps = ini.homography_hypotheses(scale * Hm, K)
        assert len(hyps) == 8
        tu = t0 / np.linalg.norm(t0)
        best = min(max(math.radians(rot_err_deg(R, R0)), np.linalg.norm(t - tu)) for R, t in hyps)
        assert best < 1e-6                          # the harness probe's tolerance (square roots of small differences)
        for _, t in hyps:
            assert abs(np.linalg.norm(t) - 1.0) < 1e-12
    assert ini.homography_hypotheses(K @ rot([1.0, 2.0, 3.0], 15.0) @ KI, K) == []


# --------------------------------------------------------------------------- I3
def test_i3_scores_with_true_models():
    P = ini.InitParams()
    g = scene(3, R_GEN, T_GEN, noise=0.0, out_frac=0.0)
    n = len(g["uv_a"])
    F = KI.T @ skew(T_GEN) @ R_GEN @ KI
    sf, mf = ini.score_fundamental(F, g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], P)
    assert np.all(mf) and abs(sf - 2 * n * P.score_gamma) <= 1e-6 * 2 * n * P.score_gamma
    Rp, tp = rot([0.0, 1.0, 0.0], -6.0), np.array([0.3, 0.0, 0.2])
    pl = scene(4, Rp, tp, noise=0.0, out_frac=0.0, plane=True)
    Hm = K @ (Rp + np.outer(tp, [0.0, 1.0, 0.0])) @ KI
    sh, mh = ini.score_homography(Hm, pl["uv_a"], pl["uv_b"], pl["sigma"], pl["sigma"], P)
    assert np.all(mh) and abs(sh - 2 * n * P.chi2_h) <= 1e-6 * 2 * n * P.chi2_h


def test_scores_use_per_keypoint_sigma():
    """B(2): e_a is divided by sigma_a^2 and e_b by sigma_b^2; sigma 1 everywhere is ORB-SLAM's score."""
    P = ini.InitParams()
    ua = np.array([[100.0, 100.0], [200.0, 150.0]])
    Hm = np.eye(3)
    ub = ua + np.array([[2.0, 0.0], [0.0, 1.0]])          # squared transfer distances 4 and 1 in both images
    one = np.ones(2)
    s1, m1 = ini.score_homography(Hm, ua, ub, one, one, P)
    assert np.allclose(s1, 2 * (P.chi2_h - 4.0) + 2 * (P.chi2_h - 1.0)) and m1.tolist() == [True, True]
    s2, m2 = ini.score_homography(Hm, ua, ub, one, np.array([2.0, 1.0]), P)
    assert np.allclose(s2, (P.chi2_h - 4.0) + (P.chi2_h - 1.0) + (P.chi2_h - 1.0) + (P.chi2_h - 1.0))
    s3, m3 = ini.score_homography(Hm, ua, ub * np.array([1.0, 1.0]) + [[1.0, 0.0], [0.0, 0.0]], one, one, P)
    assert m3.tolist() == [False, True] and np.allclose(s3, 2 * (P.chi2_h - 1.0))
    # F: a pure x-translation (epipolar lines horizontal); a vertical offset of 1.5 px
    F = skew([1.0, 0.0, 0.0])
    ub = ua + np.array([[30.0, 1.5], [10.0, 0.5]])
    s4, m4 = ini.score_fundamental(F, ua, ub, one, one, P)
    assert m4.tolist() == [True, True]
    assert np.allclose(s4, 2 * (P.score_gamma - 2.25) + 2 * (P.score_gamma - 0.25))
    s5, m5 = ini.score_fundamental(F, ua, ub, np.array([1.0, 1.0]), np.array([0.5, 1.0]), P)
    assert m5.tolist() == [False, True]                   # 2.25 / 0.25 = 9 > chi2_f in image b
    assert np.allclose(s5, (P.score_gamma - 2.25) + 2 * (P.score_gamma - 0.25))


# --------------------------------------------------------------------------- I4
def test_i4_check_rt_true_pose():
    P = ini.InitParams()
    g = scene(5, R_GEN, T_GEN, noise=0.0, out_frac=0.0)
    n = len(g["uv_a"])
    n_good, X, tri, cp, p51 = ini.check_rt(R_GEN, T_GEN, g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], K,
                                           np.ones(n, dtype=bool), P)
    assert n_good == n
    assert np.all(np.abs(X - g["X"]) <= 1e-6 * np.abs(g["X"]).max(axis=1, keepdims=True))
    assert np.array_equal(tri, cp < P.infinite_cos) and p51 > 1.0
    # a masked-out match is not good, and its row is nan
    mask = np.ones(n, dtype=bool)
    mask[:10] = False
    n2, X2, tri2, _, _ = ini.check_rt(R_GEN, T_GEN, g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], K, mask, P)
    assert n2 == n - 10 and np.all(np.isnan(X2[:10])) and not np.any(tri2[:10])
    # the mirrored pose puts the points behind: nothing is good
    n3, *_ = ini.check_rt(R_GEN, -T_GEN, g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], K,
                          np.ones(n, dtype=bool), P)
    assert n3 < n // 10


# --------------------------------------------------------------------------- I5
def test_i5_general_scene_accepted(gen, gen_result):
    r = gen_result
    assert r.status == "accepted" and r.reason == "none" and r.model == "F", (r.status, r.reason, r.model)
    assert rot_err_deg(r.R, R_GEN) < 0.5
    assert dir_err_deg(r.t, T_GEN) < 2.0
    assert abs(float(np.median(r.points[:, 2])) - 1.0) <= 1e-9
    assert np.all(r.parallax_deg >= 1.1)
    m = len(r.points)
    assert m >= 50 and r.match_idx.shape == (m,) and r.parallax_deg.shape == (m,)
    assert np.array_equal(r.kp_a, r.match_idx) and np.array_equal(r.kp_b, r.match_idx)
    assert r.rh <= 0.45 and len(r.good) == 4 and 0 <= r.chosen < 4
    assert r.ba_iterations >= 1 and r.ba_cost_final <= r.ba_cost_initial
    assert not np.any(gen["outlier"][r.match_idx])
    # published points reproject within 2 sigma and lie in front of both cameras (E4)
    Xb = r.points @ r.R.T + r.t
    assert np.all(r.points[:, 2] > 0) and np.all(Xb[:, 2] > 0)
    assert np.all(np.sum((project(r.points) - gen["uv_a"][r.match_idx]) ** 2, axis=1) <= 4.0)
    assert np.all(np.sum((project(Xb) - gen["uv_b"][r.match_idx]) ** 2, axis=1) <= 4.0)
    assert r.rows >= 2 and r.cols >= 2


# --------------------------------------------------------------------------- I6
def test_i6_pure_rotation_refused():
    g = scene(12, rot([0.4, 1.0, -0.2], 15.0), np.zeros(3))
    r = ini.initialize(g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], g["desc"], K, (W, H))
    assert r.status == "refused" and r.reason != "none"
    assert r.R is None and r.t is None
    assert r.points.shape == (0, 3) and len(r.match_idx) == 0 and len(r.kp_a) == 0 and len(r.kp_b) == 0
    assert len(r.parallax_deg) == 0


# --------------------------------------------------------------------------- I7
def test_i7_distinct_matches_rules():
    P = ini.InitParams()
    base = np.array([[10.0 * i + 5.0, 7.0 * i + 3.0] for i in range(12)])
    ua, ub = base.copy(), base.copy() + 50.0
    kp_a = np.arange(12)
    kp_b = np.arange(100, 112)
    desc = np.linspace(0.1, 0.9, 12)
    kp_a[5] = 0; desc[0], desc[5] = 0.5, 0.3                 # kp_a 0 twice: keep 5 (smaller distance)
    kp_a[2] = 1; desc[1] = desc[2] = 0.2                     # kp_a 1 twice, tie: keep the lower index 1
    kp_b[4] = kp_b[3]; desc[3], desc[4] = 0.45, 0.44         # kp_b twice: keep 4
    ua[6] = ua[7] + 0.0005; desc[6], desc[7] = 0.7, 0.6      # same pixel in image a: keep 7
    ub[9] = ub[8] - 0.0008; desc[8], desc[9] = 0.65, 0.75    # same pixel in image b: keep 8
    ua[10] = ua[11] + 0.002                                  # 2e-3 px apart: both kept
    keep = ini.distinct_matches(ua, ub, desc, kp_a, kp_b, P)
    assert keep.tolist() == [1, 4, 5, 7, 8, 10, 11]
    # the kp_b pass runs on what the kp_a pass kept: row 0 (dropped above, distance 0.5) shares its
    # kp_b id with row 7 (0.6); a pass over every row would drop row 7, the rule keeps it
    kp_b2 = kp_b.copy(); kp_b2[0] = kp_b[7]
    keep2 = ini.distinct_matches(ua, ub, desc, kp_a, kp_b2, P)
    assert keep2.tolist() == keep.tolist()


def test_i7_initialize_removes_duplicates(gen, gen_result):
    rng = np.random.default_rng(7)
    n = len(gen["uv_a"])
    di = rng.choice(n, 90, replace=False)
    ua = np.concatenate([gen["uv_a"], gen["uv_a"][di]])
    ub = np.concatenate([gen["uv_b"], gen["uv_b"][di]])
    s = np.ones(n + 90)
    desc = np.concatenate([gen["desc"], gen["desc"][di] + 1.0])  # the copies lose every tie
    kp = np.arange(n + 90)
    r = ini.initialize(ua, ub, s, s, desc, K, (W, H), kp_a=kp, kp_b=kp)
    assert r.n_matches == n
    assert r.status == "accepted" and np.all(r.match_idx < n)
    assert np.array_equal(r.points, gen_result.points)          # the same n matches as without copies
    # repeated keypoint ids (same kp id on both sides, different pixels): one match per id survives
    kp2 = np.concatenate([np.arange(n), di])
    ub2 = np.concatenate([gen["uv_b"], gen["uv_b"][di] + 5.0])
    r2 = ini.initialize(ua, ub2, s, s, desc, K, (W, H), kp_a=kp2, kp_b=kp2)
    assert r2.n_matches == n and len(np.unique(r2.kp_a)) == len(r2.kp_a)


# --------------------------------------------------------------------------- I8
def test_i8_grid_support():
    P = ini.InitParams()
    uv = np.array([[10.0, 10.0], [170.0, 10.0], [10.0, 170.0]])
    assert ini.grid_support(uv, (W, H), P) == (2, 2, 3)
    assert ini.grid_support(np.array([[-3.0, 500.0], [645.0, -1.0]]), (W, H), P) == (2, 2, 2)
    assert ini.grid_support(np.array([[159.9, 0.0], [0.0, 479.0]]), (W, H), P) == (3 - 0 - 1, 1, 2)
    assert ini.grid_support(np.zeros((0, 2)), (W, H), P) == (0, 0, 0)


def test_i8_one_column_band_never_accepted(gen):
    # the general scene restricted to the matches whose uv_b has u < 160
    sel = np.flatnonzero(gen["uv_b"][:, 0] < 160.0)
    s = np.ones(len(sel))
    r = ini.initialize(gen["uv_a"][sel], gen["uv_b"][sel], s, s, gen["desc"][sel], K, (W, H))
    assert r.status == "refused"
    # 300 matches generated inside the band (the harness's conc regime): never accepted
    for seed in range(21, 26):
        b = scene(seed, R_GEN, T_GEN, band_u=160.0)
        r2 = ini.initialize(b["uv_a"], b["uv_b"], b["sigma"], b["sigma"], b["desc"], K, (W, H))
        assert r2.status == "refused" and r2.points.shape == (0, 3) and r2.R is None


def test_i8_support_gate_refuses(gen, gen_result):
    """E7 on a scene that passes every earlier step: with min_cols above the grid's column count the
    same result is refused with reason support, and carries rows, cols and cells but no geometry."""
    assert gen_result.status == "accepted" and gen_result.cols == 4
    s = gen["sigma"]
    r = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H), params=ini.InitParams(min_cols=5))
    assert r.status == "refused" and r.reason == "support"
    assert (r.rows, r.cols, r.cells) == (gen_result.rows, gen_result.cols, gen_result.cells)
    assert r.R is None and r.points.shape == (0, 3)
    r = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H), params=ini.InitParams(min_rows=4))
    assert r.status == "refused" and r.reason == "support"


# --------------------------------------------------------------------------- I9
def test_i9_determinism(gen):
    s = gen["sigma"]
    r1 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H))
    r2 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H))
    for f in dataclasses.fields(ini.InitResult):
        a, b = getattr(r1, f.name), getattr(r2, f.name)
        if a is None or isinstance(a, str):
            assert a == b, f.name
        else:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), f.name


# --------------------------------------------------------------------------- I10
def test_i10_format_init_line():
    acc = ini.InitResult(status="accepted", reason="none", model="F", rh=0.123456789, score_h=1.0, score_f=2.0,
                         n_matches=300, n_inliers=231, good=(3, 212, 5, 0), chosen=1, parallax51_deg=4.5,
                         n_triangulated=210, R=np.eye(3), t=np.array([1.0, 0.0, 0.0]),
                         points=np.ones((3, 3)), match_idx=np.arange(3), kp_a=np.arange(3), kp_b=np.arange(3),
                         parallax_deg=np.array([2.0, 5.0, 3.0]), rows=3, cols=4, cells=11,
                         ba_cost_initial=120.5, ba_cost_final=99.25, ba_iterations=6)
    assert ini.format_init_line(3, 40, acc) == (
        "  >> [INIT-PAIR] a=3 b=40 status=accepted reason=none model=F rh=0.123457 matches=300 inliers=231 "
        "good=3/212/5/0 chosen=1 parallax51_deg=4.500000 triangulated=210 published=3 rows=3 cols=4 "
        "median_parallax_deg=3.000000 ba_cost0=120.500000 ba_cost=99.250000")
    ref = ini.InitResult(status="refused", reason="matches", model="none", rh=math.nan, score_h=math.nan,
                         score_f=math.nan, n_matches=42, n_inliers=0, good=(), chosen=-1, parallax51_deg=math.nan,
                         n_triangulated=0, R=None, t=None, points=np.zeros((0, 3)),
                         match_idx=np.zeros(0, dtype=np.int64), kp_a=np.zeros(0, dtype=np.int64),
                         kp_b=np.zeros(0, dtype=np.int64), parallax_deg=np.zeros(0), rows=0, cols=0, cells=0,
                         ba_cost_initial=math.nan, ba_cost_final=math.nan, ba_iterations=-1)
    assert ini.format_init_line(0, 7, ref) == (
        "  >> [INIT-PAIR] a=0 b=7 status=refused reason=matches model=none rh=nan matches=42 inliers=0 "
        "good=- chosen=-1 parallax51_deg=nan triangulated=0 published=0 rows=0 cols=0 "
        "median_parallax_deg=nan ba_cost0=nan ba_cost=nan")


# --------------------------------------------------------------------------- I11
def _stub_frame(uv, octaves, desc, img_shape=(480, 640)):
    kps = [cv2.KeyPoint(float(u), float(v), 5.0, -1.0, 0.0, int(o)) for (u, v), o in zip(uv, octaves)]
    return SimpleNamespace(KeyPoints=kps, Descriptors=desc.astype(np.float32), method="SIFT", match_type="KNN",
                           K=K.copy(), img_shape=img_shape, detector_config=None)


def test_i11_initialize_frames_equals_initialize(gen):
    rng = np.random.default_rng(99)
    n = len(gen["uv_a"])
    perm = rng.permutation(n)                       # frame b keypoint perm[i] is the match of frame a's i
    inv = np.argsort(perm)
    uv_b_frame = np.empty_like(gen["uv_b"])
    uv_b_frame[perm] = gen["uv_b"]
    oct_a = rng.choice([0, 1, 255], n)              # SIFT low byte 255 = octave -1 (sigma 1)
    oct_b = rng.choice([0, 1, 2], n)
    da = rng.normal(0.0, 1.0, (n, 128))
    db = np.empty_like(da)
    db[perm] = da + rng.normal(0.0, 0.01, (n, 128))
    fa = _stub_frame(gen["uv_a"], oct_a, da)
    fb = _stub_frame(uv_b_frame, oct_b, db)
    r = ini.initialize_frames(fa, fb)
    # the expected arrays, built independently of the matcher
    qi = np.arange(n)
    ti = perm
    dist = np.linalg.norm(fa.Descriptors.astype(np.float64) - fb.Descriptors.astype(np.float64)[ti], axis=1)
    sig_a = 2.0 ** np.maximum(np.where(oct_a >= 128, oct_a - 256, oct_a), 0)
    sig_b = 2.0 ** oct_b[ti]
    uv_b = np.array([fb.KeyPoints[j].pt for j in ti])
    uv_a = np.array([kp.pt for kp in fa.KeyPoints])
    e = ini.initialize(uv_a, uv_b, sig_a, sig_b, dist, K, (640, 480), kp_a=qi, kp_b=ti)
    assert e.status == "accepted"
    assert np.array_equal(inv[r.kp_b], r.kp_a)      # every published match is a true match
    for f in dataclasses.fields(ini.InitResult):
        a, b = getattr(r, f.name), getattr(e, f.name)
        if a is None or isinstance(a, str):
            assert a == b, f.name
        else:
            assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f"), f.name


def test_initialize_frames_too_few_matches():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 1.0, (5, 128))
    uv = rng.uniform(0, 400, (5, 2))
    r = ini.initialize_frames(_stub_frame(uv, [0] * 5, d), _stub_frame(uv, [0] * 5, d + 0.001))
    assert r.status == "refused" and r.reason == "matches" and r.n_matches == 5 and r.model == "none"


# --------------------------------------------------------------------------- the other branches
def test_matches_refusal_below_min_matches(gen):
    s = np.ones(99)
    r = ini.initialize(gen["uv_a"][:99], gen["uv_b"][:99], s, s, gen["desc"][:99], K, (W, H))
    assert r.status == "refused" and r.reason == "matches" and r.n_matches == 99
    assert math.isnan(r.rh) and r.model == "none" and r.good == () and r.ba_iterations == -1
    s = np.ones(100)
    r = ini.initialize(gen["uv_a"][:100], gen["uv_b"][:100], s, s, gen["desc"][:100], K, (W, H))
    assert r.reason != "matches"


def test_plane_scene_selects_h():
    Rp, tp = rot([0.0, 1.0, 0.0], 7.0), 0.5 * np.array([0.8, 0.0, 0.6])
    pl = scene(8, Rp, tp, plane=True)
    r = ini.initialize(pl["uv_a"], pl["uv_b"], pl["sigma"], pl["sigma"], pl["desc"], K, (W, H))
    assert r.model == "H" and r.rh > 0.45
    assert len(r.good) in (0, 8)                    # 8 hypotheses unless degenerate
    if r.status == "accepted":
        assert rot_err_deg(r.R, Rp) < 1.0


def test_refusal_carries_no_geometry(gen):
    """A refusal after E3 (the parallax gate raised above what the scene has) keeps the statistics
    of D but no pose and no point."""
    s = gen["sigma"]
    r = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H),
                       params=ini.InitParams(theta_min_deg=30.0))
    assert r.status == "refused" and r.reason == "points"
    assert r.R is None and r.points.shape == (0, 3) and len(r.parallax_deg) == 0
    assert r.model == "F" and r.ba_iterations >= 1 and math.isfinite(r.ba_cost_final)


# --------------------------------------------------------------------------- 04t: the refit switch (ruling R37-2)

def _unit_frob(M):
    """Frobenius-normalize and fix the sign on the entry of largest magnitude (spec C1/T2)."""
    M = np.asarray(M, dtype=np.float64)
    M = M / np.linalg.norm(M)
    i = np.unravel_index(np.argmax(np.abs(M)), M.shape)
    return M if M[i] > 0 else -M


# the noise-free plane scene of I2 (test_i2_homography_hypotheses_plane_and_rotation): R0, t0, n = (0, 1, 0), d = 1
R0_I2 = rot([0.0, 1.0, 0.0], 8.0)
T0_I2 = np.array([0.4, 0.05, 0.2])


def test_t1_refit_off_bitwise_identical(gen):
    s = gen["sigma"]
    r0 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H))
    r1 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H), refit=False)
    for f in dataclasses.fields(ini.InitResult):
        a, b = getattr(r0, f.name), getattr(r1, f.name)
        if a is None or isinstance(a, str):
            assert a == b, f.name
        else:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), f.name


def test_t2_refit_model_recovers_true_geometry():
    P = ini.InitParams()
    g = scene(3, R_GEN, T_GEN, noise=0.0, out_frac=0.0)
    n = len(g["uv_a"])
    F_true = KI.T @ skew(T_GEN) @ R_GEN @ KI
    M, score, m = ini.refit_model("F", g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], np.ones(n, dtype=bool), P)
    assert M is not None and math.isfinite(score)
    assert np.max(np.abs(_unit_frob(M) - _unit_frob(F_true))) <= 1e-6
    assert np.all(m)

    pl = scene(4, R0_I2, T0_I2, noise=0.0, out_frac=0.0, plane=True)
    npl = len(pl["uv_a"])
    H_true = K @ (R0_I2 + np.outer(T0_I2, [0.0, 1.0, 0.0]) / 1.0) @ KI
    M2, score2, m2 = ini.refit_model("H", pl["uv_a"], pl["uv_b"], pl["sigma"], pl["sigma"],
                                     np.ones(npl, dtype=bool), P)
    assert M2 is not None and math.isfinite(score2)
    assert np.max(np.abs(_unit_frob(M2) - _unit_frob(H_true))) <= 1e-6
    assert np.all(m2)


def test_t3_refit_model_too_few_inliers(gen):
    P = ini.InitParams()
    n = len(gen["uv_a"])
    mask = np.zeros(n, dtype=bool)
    mask[:7] = True
    M, score, m = ini.refit_model("F", gen["uv_a"], gen["uv_b"], gen["sigma"], gen["sigma"], mask, P)
    assert M is None and math.isnan(score) and np.array_equal(m, mask) and m is not mask


def test_t4_pure_rotation_refused_with_refit():
    g = scene(12, rot([0.4, 1.0, -0.2], 15.0), np.zeros(3))
    r = ini.initialize(g["uv_a"], g["uv_b"], g["sigma"], g["sigma"], g["desc"], K, (W, H), refit=True)
    assert r.status == "refused" and r.reason != "none"
    assert r.R is None and r.t is None
    assert r.points.shape == (0, 3) and len(r.match_idx) == 0 and len(r.kp_a) == 0 and len(r.kp_b) == 0
    assert len(r.parallax_deg) == 0


def test_t5_refit_on_deterministic(gen):
    s = gen["sigma"]
    r1 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H), refit=True)
    r2 = ini.initialize(gen["uv_a"], gen["uv_b"], s, s, gen["desc"], K, (W, H), refit=True)
    for f in dataclasses.fields(ini.InitResult):
        a, b = getattr(r1, f.name), getattr(r2, f.name)
        if a is None or isinstance(a, str):
            assert a == b, f.name
        else:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), f.name


def test_t6_initialize_frames_refit_equals_initialize(gen):
    rng = np.random.default_rng(99)
    n = len(gen["uv_a"])
    perm = rng.permutation(n)
    uv_b_frame = np.empty_like(gen["uv_b"])
    uv_b_frame[perm] = gen["uv_b"]
    oct_a = rng.choice([0, 1, 255], n)
    oct_b = rng.choice([0, 1, 2], n)
    da = rng.normal(0.0, 1.0, (n, 128))
    db = np.empty_like(da)
    db[perm] = da + rng.normal(0.0, 0.01, (n, 128))
    fa = _stub_frame(gen["uv_a"], oct_a, da)
    fb = _stub_frame(uv_b_frame, oct_b, db)
    r = ini.initialize_frames(fa, fb, refit=True)
    qi = np.arange(n)
    ti = perm
    dist = np.linalg.norm(fa.Descriptors.astype(np.float64) - fb.Descriptors.astype(np.float64)[ti], axis=1)
    sig_a = 2.0 ** np.maximum(np.where(oct_a >= 128, oct_a - 256, oct_a), 0)
    sig_b = 2.0 ** oct_b[ti]
    uv_b = np.array([fb.KeyPoints[j].pt for j in ti])
    uv_a = np.array([kp.pt for kp in fa.KeyPoints])
    e = ini.initialize(uv_a, uv_b, sig_a, sig_b, dist, K, (640, 480), kp_a=qi, kp_b=ti, refit=True)
    assert e.status == "accepted"
    for f in dataclasses.fields(ini.InitResult):
        a, b = getattr(r, f.name), getattr(e, f.name)
        if a is None or isinstance(a, str):
            assert a == b, f.name
        else:
            assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f"), f.name
