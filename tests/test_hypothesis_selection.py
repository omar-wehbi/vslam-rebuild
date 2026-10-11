"""Stage 03b (spec C): select_essential_hypothesis on synthetic correspondences (no images).
OpenCV convention: X2 = R X1 + t, x2^T E x1 = 0, E = [t]x R; camera 1 = prev, camera 2 = curr."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking import (  # noqa: E402
    select_essential_hypothesis,
    twoview_hypothesis_votes,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
AXIS = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
R_TRUE = Rotation.from_rotvec(np.deg2rad(2.0) * AXIS).as_matrix()
T_TRUE = np.array([0.6, -0.2, 0.77]) / np.linalg.norm([0.6, -0.2, 0.77])
N = 200


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_about(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return Rotation.from_rotvec(np.deg2rad(deg) * axis).as_matrix()


def _project(X, Kmat):
    p = (Kmat @ X.T).T
    return p[:, :2] / p[:, 2:3]


def _make_scene(zmin, zmax, t_sign=+1.0, seed=3):
    """Points at depth [zmin, zmax] in camera 1, second view X2 = R_true X1 + t_sign * t_true."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(zmin, zmax, N)
    u = rng.uniform(40.0, 600.0, N)
    v = rng.uniform(40.0, 440.0, N)
    X1 = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    X2 = (R_TRUE @ X1.T).T + t_sign * T_TRUE
    assert np.all(X2[:, 2] > 0)
    pts1 = _project(X1, K)
    pts2 = _project(X2, K)
    E = _skew(t_sign * T_TRUE) @ R_TRUE  # equals +/- ([t_true]x R_true): same E up to sign
    assert abs(float(AXIS @ T_TRUE)) < 0.99  # generic axis, not parallel to t
    return dict(pts1=pts1, pts2=pts2, E=E)


@pytest.fixture(scope="module")
def near():
    return _make_scene(2.0, 8.0)


@pytest.fixture(scope="module")
def far():
    # Hundreds of baselines away (||t|| = 1), the pioneer_slam3 regime of Stage 03a.
    return _make_scene(200.0, 800.0)


def _check_true_pose(R, t, t_expected):
    assert R.shape == (3, 3) and t.shape == (3, 1)
    assert R.dtype == np.float64 and t.dtype == np.float64
    assert np.linalg.norm(R - R_TRUE) < 1e-6, R
    assert np.linalg.norm(t - np.asarray(t_expected).reshape(3, 1)) < 1e-6, t
    assert abs(float(np.linalg.norm(t)) - 1.0) < 1e-9


def test_c1_near_scene_selects_true_pose_with_all_votes(near):
    R, t, votes_all, votes_near, chosen = select_essential_hypothesis(near["E"], near["pts1"], near["pts2"], K)
    _check_true_pose(R, t, T_TRUE)
    assert len(votes_all) == 4 and len(votes_near) == 4
    assert votes_all[chosen] == N, votes_all
    assert chosen == int(np.argmax(votes_all))
    # No other hypothesis explains every point.
    assert all(votes_all[i] < N for i in range(4) if i != chosen), votes_all


def test_c2_far_scene_defect_regime_still_selects_true_pose(far):
    R, t, votes_all, votes_near, chosen = select_essential_hypothesis(far["E"], far["pts1"], far["pts2"], K)
    _check_true_pose(R, t, T_TRUE)
    assert votes_all[chosen] == N, votes_all
    assert chosen == int(np.argmax(votes_all))
    # What the old cutoff saw on this scene: not a single "near" point for any hypothesis,
    # so its vote was 0/0/0/0 and its tie rule returned hypothesis 0 whatever the data said.
    diag = twoview_hypothesis_votes(far["E"], far["pts1"], far["pts2"], K, R, t, near_thresh=50.0)
    assert diag["votes_near"] == [0, 0, 0, 0], diag
    assert diag["votes_all"] == votes_all
    assert votes_near == [0, 0, 0, 0]
    assert diag["chosen"] == chosen


def test_c3_agrees_with_recoverpose_where_recoverpose_works(near):
    R, t, _, _, _ = select_essential_hypothesis(near["E"], near["pts1"], near["pts2"], K)
    _, R_cv, t_cv, _ = cv2.recoverPose(near["E"], near["pts1"], near["pts2"], cameraMatrix=K)
    assert R_cv.shape == R.shape and t_cv.shape == t.shape
    assert np.linalg.norm(R - R_cv) < 1e-6
    assert np.linalg.norm(t - t_cv) < 1e-6


def test_c4_sign_of_t_is_decided_by_the_correspondences():
    neg = _make_scene(2.0, 8.0, t_sign=-1.0)
    pos = _make_scene(2.0, 8.0, t_sign=+1.0)
    # E is the same matrix up to sign: the decomposition cannot tell the two apart.
    assert np.linalg.norm(neg["E"] + pos["E"]) < 1e-12
    R, t, votes_all, _, chosen = select_essential_hypothesis(neg["E"], neg["pts1"], neg["pts2"], K)
    _check_true_pose(R, t, -T_TRUE)
    assert votes_all[chosen] == N
    # Feeding the +t correspondences with the SAME (negated) E returns +t: the points decide.
    R2, t2, _, _, _ = select_essential_hypothesis(neg["E"], pos["pts1"], pos["pts2"], K)
    _check_true_pose(R2, t2, T_TRUE)


def test_c5_twisted_pair_never_wins_on_near_scene(near):
    R, t, votes_all, _, chosen = select_essential_hypothesis(near["E"], near["pts1"], near["pts2"], K)
    R_twist = _rot_about(T_TRUE, 180.0) @ R_TRUE
    assert np.linalg.norm(R - R_twist) > 1.0
    # The twisted pair is one of the four hypotheses and it has fewer votes than the winner.
    diag = twoview_hypothesis_votes(near["E"], near["pts1"], near["pts2"], K, R_twist, T_TRUE)
    assert diag["chosen"] is not None and diag["chosen"] != chosen
    assert votes_all[diag["chosen"]] < votes_all[chosen]


def test_c6_hypothesis_order_and_chosen_index(near):
    R, t, votes_all, votes_near, chosen = select_essential_hypothesis(near["E"], near["pts1"], near["pts2"], K)
    R1, R2, tt = cv2.decomposeEssentialMat(near["E"])
    tt = tt.reshape(3)
    order = [(R1, tt), (R1, -tt), (R2, tt), (R2, -tt)]
    matches = [i for i, (Ri, ti) in enumerate(order)
               if np.linalg.norm(Ri - R) < 1e-6 and np.linalg.norm(ti - t.reshape(3)) < 1e-6]
    assert matches == [chosen], (matches, chosen)
    diag = twoview_hypothesis_votes(near["E"], near["pts1"], near["pts2"], K, R, t)
    assert diag["chosen"] == chosen
    assert diag["votes_all"] == votes_all and diag["votes_near"] == votes_near


def test_selection_never_raises_on_degenerate_input():
    # All five correspondences identical in both views: the selector must not raise and must
    # still name the first argmax (spec A1/A4).
    E = _skew(T_TRUE) @ R_TRUE
    p = np.tile(np.array([[320.0, 240.0]]), (5, 1))
    R, t, votes_all, votes_near, chosen = select_essential_hypothesis(E, p, p, K)
    assert chosen == int(np.argmax(votes_all))
    assert R.shape == (3, 3) and t.shape == (3, 1)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(t))


def test_zero_votes_everywhere_returns_hypothesis_zero():
    # A non-essential E (identity) on shifted points: every hypothesis triangulates non-finite or
    # non-positive depths, so the vote is 0/0/0/0 and the tie rule returns hypothesis 0 (R1, +t),
    # exactly what the old OpenCV call returned in that case.
    E = np.eye(3)
    p = np.tile(np.array([[320.0, 240.0]]), (5, 1))
    R, t, votes_all, votes_near, chosen = select_essential_hypothesis(E, p + 1.0, p, K)
    assert votes_all == [0, 0, 0, 0], votes_all
    assert votes_near == [0, 0, 0, 0], votes_near
    assert chosen == 0
    R1, _, tt = cv2.decomposeEssentialMat(E)
    assert np.linalg.norm(R - R1) < 1e-12 and np.linalg.norm(t - tt.reshape(3, 1)) < 1e-12
    assert R.shape == (3, 3) and t.shape == (3, 1)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(t))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
