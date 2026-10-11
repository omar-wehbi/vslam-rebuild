"""
Stage 01c: the loop-closure geometric-verification PnP must no longer be seeded
from the drifted current pose (conventions.md#loop-closure-relocalization-seed,
design.md foundation items 5 and 8).

LoopCloser.detect factors its PnP call into module-level verify_loop_candidate(),
which calls cv2.solvePnPRansac with useExtrinsicGuess=False and SOLVEPNP_EPNP
(RANSAC hypotheses), then refines the RANSAC result (never an external guess)
with cv2.solvePnPRefineLM.  relocalize() is untouched: it still seeds from the
matched keyframe's pose, which is explicitly allowed by the convention.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loop_closure as lc  # noqa: E402

DEFAULT_K = np.array([
    [520.9,   0.0, 325.1],
    [  0.0, 521.0, 249.7],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


# --------------------------------------------------------------------------- #
# C1. Call-signature assertion: verify_loop_candidate must call
#     cv2.solvePnPRansac unseeded (useExtrinsicGuess=False, SOLVEPNP_EPNP, no
#     rvec/tvec guess) and must refine from the RANSAC result via
#     cv2.solvePnPRefineLM, never from an externally supplied pose.
# --------------------------------------------------------------------------- #

def test_verify_loop_candidate_call_signature(monkeypatch):
    n = 20
    pts_3d = np.random.default_rng(0).uniform(-1, 1, size=(n, 3)) + np.array([0, 0, 8.0])
    pts_2d = np.random.default_rng(1).uniform(0, 640, size=(n, 2))

    ransac_calls = []
    fake_rvec = np.array([[0.1], [0.2], [0.3]])
    fake_tvec = np.array([[0.0], [0.0], [8.0]])
    fake_inliers = np.arange(15).reshape(-1, 1)

    def fake_solve_pnp_ransac(**kwargs):
        ransac_calls.append(kwargs)
        return True, fake_rvec.copy(), fake_tvec.copy(), fake_inliers.copy()

    refine_calls = []

    def fake_refine_lm(**kwargs):
        refine_calls.append(kwargs)
        return kwargs["rvec"], kwargs["tvec"]

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve_pnp_ransac)
    monkeypatch.setattr(cv2, "solvePnPRefineLM", fake_refine_lm)

    ok, cam_T_world, n_inliers = lc.verify_loop_candidate(pts_3d, pts_2d, DEFAULT_K)

    assert ok is True
    assert n_inliers == 15
    assert cam_T_world is not None

    assert len(ransac_calls) == 1, "verify_loop_candidate must call solvePnPRansac exactly once"
    kw = ransac_calls[0]
    assert kw.get("useExtrinsicGuess") is False, "loop PnP must not be seeded"
    assert kw.get("flags") == cv2.SOLVEPNP_EPNP, "RANSAC hypotheses must use EPnP"
    assert kw.get("rvec") is None, "no external rvec guess may be passed"
    assert kw.get("tvec") is None, "no external tvec guess may be passed"
    assert kw.get("iterationsCount") == 500
    assert kw.get("reprojectionError") == 4.0
    assert kw.get("confidence") == 0.999

    assert len(refine_calls) == 1, "refinement must run on the RANSAC result"
    rkw = refine_calls[0]
    # The refinement must start from the RANSAC result, not from any external pose.
    assert np.array_equal(rkw["rvec"], fake_rvec)
    assert np.array_equal(rkw["tvec"], fake_tvec)


def test_verify_loop_candidate_rejects_below_inlier_bar(monkeypatch):
    """ok=True but < 10 inliers must still be rejected (unchanged bar)."""
    n = 20
    pts_3d = np.zeros((n, 3), dtype=np.float64)
    pts_2d = np.zeros((n, 2), dtype=np.float64)

    def fake_solve_pnp_ransac(**kwargs):
        return True, np.zeros((3, 1)), np.zeros((3, 1)), np.arange(9).reshape(-1, 1)

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve_pnp_ransac)

    ok, cam_T_world, n_inliers = lc.verify_loop_candidate(pts_3d, pts_2d, DEFAULT_K)
    assert ok is False
    assert cam_T_world is None
    assert n_inliers == 0


# --------------------------------------------------------------------------- #
# C1 (detect-level). Build minimal stand-in frames/keyframes so LoopCloser.detect
# reaches the PnP call, and prove it never touches curr_frame.cam_T_world: the
# stand-in current frame simply has no such attribute, so any access would
# raise AttributeError and fail the test.
# --------------------------------------------------------------------------- #

class _MiniKeyPoint:
    def __init__(self, x, y):
        self.pt = (float(x), float(y))


class _MiniPoint:
    def __init__(self, xyz):
        self.point = np.asarray(xyz, dtype=np.float64)


class _MiniMap:
    def __init__(self, points_by_id):
        self._points = points_by_id

    def get_point(self, pid):
        return self._points.get(pid)


def test_detect_no_seed_from_current_pose(monkeypatch):
    n = 30
    rng = np.random.default_rng(42)

    # Keyframe: has keypoint_to_point links to map points 0..n-1.
    kf = SimpleNamespace(
        Descriptors=None,
        method="ORB",
        keypoint_to_point={i: i for i in range(n)},
    )

    # Current frame: deliberately NO cam_T_world attribute. If detect() (or
    # anything it calls) reads curr_frame.cam_T_world, this test fails with
    # AttributeError instead of silently passing.
    curr_frame = SimpleNamespace(
        Descriptors=None,
        method="ORB",
        K=DEFAULT_K,
        KeyPoints=[_MiniKeyPoint(rng.uniform(0, 640), rng.uniform(0, 480)) for _ in range(n)],
    )
    assert not hasattr(curr_frame, "cam_T_world")

    world_map = _MiniMap({i: _MiniPoint([rng.uniform(-1, 1), rng.uniform(-1, 1), 8.0 + i * 0.1])
                           for i in range(n)})

    fake_matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(n)]
    monkeypatch.setattr(lc, "match_frames", lambda kf_, cf_, method, **kw: fake_matches)

    ransac_calls = []
    fake_rvec = np.array([[0.0], [0.0], [0.0]])
    fake_tvec = np.array([[0.0], [0.0], [8.0]])
    fake_inliers = np.arange(15).reshape(-1, 1)

    def fake_solve_pnp_ransac(**kwargs):
        ransac_calls.append(kwargs)
        return True, fake_rvec.copy(), fake_tvec.copy(), fake_inliers.copy()

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve_pnp_ransac)
    monkeypatch.setattr(cv2, "solvePnPRefineLM", lambda **kw: (kw["rvec"], kw["tvec"]))

    closer = lc.LoopCloser()
    closer.keyframes = [(0, kf)]

    loop_kf_idx, loop_kf, cam_T_world, n_inliers = closer.detect(curr_idx=100, curr_frame=curr_frame, world_map=world_map)

    assert loop_kf_idx == 0
    assert loop_kf is kf
    assert cam_T_world is not None
    assert n_inliers == 15

    assert len(ransac_calls) == 1
    kw = ransac_calls[0]
    assert kw.get("useExtrinsicGuess") is False
    assert kw.get("flags") == cv2.SOLVEPNP_EPNP
    assert kw.get("rvec") is None
    assert kw.get("tvec") is None


def test_relocalize_keeps_keyframe_seed():
    """Out of scope for this stage: relocalize must still seed from kf.cam_T_world."""
    import inspect
    source = inspect.getsource(lc.LoopCloser.relocalize)
    assert "kf.cam_T_world" in source
    assert "useExtrinsicGuess=True" in source
    assert "SOLVEPNP_ITERATIVE" in source


# --------------------------------------------------------------------------- #
# C2. Synthetic recovery: unseeded PnP must recover the true pose despite
# outliers, without regard to how far the (unused) drifted seed is from it.
# The OLD seeded call is also run on the same data for comparison; its
# result is only printed / included in the failure message, never asserted.
# --------------------------------------------------------------------------- #

def _rotation_angle_deg(R):
    """Angle (degrees) of the rotation matrix R (angle of R relative to identity)."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(np.arccos(cos_theta))


def test_synthetic_recovery_unseeded_vs_seeded():
    rng = np.random.default_rng(7)
    n_points = 60
    n_outliers = 12  # 20%

    K = DEFAULT_K
    img_w, img_h = 640.0, 480.0

    # True pose: identity (world coordinates chosen to already be camera-frame
    # coordinates at the true pose), points 5-15 units in front of the camera.
    R_true = np.eye(3)
    t_true = np.zeros(3)

    us = rng.uniform(0.0, img_w, size=n_points)
    vs = rng.uniform(0.0, img_h, size=n_points)
    zs = rng.uniform(5.0, 15.0, size=n_points)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    pts_3d = np.stack([xs, ys, zs], axis=1)  # == camera-frame coords at truth
    pts_2d = np.stack([us, vs], axis=1).copy()

    median_depth = float(np.median(zs))

    # Corrupt 20% of the 2D observations with uncorrelated pixel positions.
    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
    pts_2d[outlier_idx, 0] = rng.uniform(0.0, img_w, size=n_outliers)
    pts_2d[outlier_idx, 1] = rng.uniform(0.0, img_h, size=n_outliers)

    # Drifted pose: 10 degrees and 0.5 units away from the truth (used ONLY
    # for the old seeded call below, never for the unseeded path).
    axis = np.array([1.0, 1.0, 0.0])
    axis /= np.linalg.norm(axis)
    theta = np.radians(10.0)
    R_drift, _ = cv2.Rodrigues(axis * theta)
    t_drift = np.array([0.5, 0.0, 0.0])
    drift_rvec, _ = cv2.Rodrigues(R_drift @ R_true)
    drift_tvec = (t_true + t_drift).reshape(3, 1)

    # --- Unseeded path (this stage) ---
    ok, cam_T_world, n_inliers = lc.verify_loop_candidate(pts_3d, pts_2d, K)
    assert ok, "unseeded PnP should succeed on 80%-inlier synthetic data"
    R_est = cam_T_world[:3, :3]
    t_est = cam_T_world[:3, 3]

    unseeded_angle_err = _rotation_angle_deg(R_true.T @ R_est)
    unseeded_trans_err = float(np.linalg.norm(t_est - t_true))

    # --- Old seeded call, for comparison only (no assertion on its result) ---
    ok_seeded, rvec_seeded, tvec_seeded, inliers_seeded = cv2.solvePnPRansac(
        objectPoints=pts_3d,
        imagePoints=pts_2d,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=500,
        reprojectionError=4.0,
        confidence=0.999,
        useExtrinsicGuess=True,
        rvec=drift_rvec.copy(),
        tvec=drift_tvec.copy(),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok_seeded:
        R_seeded, _ = cv2.Rodrigues(rvec_seeded)
        seeded_angle_err = _rotation_angle_deg(R_true.T @ R_seeded)
        seeded_trans_err = float(np.linalg.norm(tvec_seeded.flatten() - t_true))
        seeded_n_inliers = 0 if inliers_seeded is None else len(inliers_seeded)
    else:
        seeded_angle_err = seeded_trans_err = float("nan")
        seeded_n_inliers = 0

    msg = (
        f"unseeded: angle_err={unseeded_angle_err:.4f}deg trans_err={unseeded_trans_err:.4f} "
        f"({100 * unseeded_trans_err / median_depth:.3f}% of median depth {median_depth:.3f}) "
        f"n_inliers={n_inliers} | "
        f"seeded (drifted guess): angle_err={seeded_angle_err:.4f}deg trans_err={seeded_trans_err:.4f} "
        f"n_inliers={seeded_n_inliers}"
    )
    print(msg)

    assert unseeded_angle_err <= 0.5, msg
    assert unseeded_trans_err <= 0.01 * median_depth, msg
    assert n_inliers >= 40, msg
