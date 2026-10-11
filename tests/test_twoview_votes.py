"""Stage 03a (spec E2): the two-view decomposition diagnostic in tracking.py on synthetic
correspondences (no images). OpenCV convention: X2 = R X1 + t, x2^T E x1 = 0, E = [t]x R."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking import (  # noqa: E402
    _print_twoview_diagnostic,
    format_twoview_line,
    rotation_angle_deg,
    twoview_hypothesis_votes,
)

TWOVIEW_RE = re.compile(
    r"^\s*>> \[TWOVIEW\] frame=(\d+) inliers=(\d+) rot_deg=(\S+) t_cam=(\S+),(\S+),(\S+) "
    r"votes_all=(\d+)/(\d+)/(\d+)/(\d+) votes_near=(\d+)/(\d+)/(\d+)/(\d+) chosen=(\d|none)$"
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
AXIS = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
R_TRUE = Rotation.from_rotvec(np.deg2rad(2.0) * AXIS).as_matrix()
T_TRUE = np.array([0.6, -0.2, 0.77]) / np.linalg.norm([0.6, -0.2, 0.77])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_about(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return Rotation.from_rotvec(np.deg2rad(deg) * axis).as_matrix()


def _project(X, Kmat):
    p = (Kmat @ X.T).T
    return p[:, :2] / p[:, 2:3]


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    n = 200
    z = rng.uniform(2.0, 8.0, n)
    u = rng.uniform(40.0, 600.0, n)
    v = rng.uniform(40.0, 440.0, n)
    X1 = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    X2 = (R_TRUE @ X1.T).T + T_TRUE
    assert np.all(X2[:, 2] > 0)
    pts1 = _project(X1, K)
    pts2 = _project(X2, K)
    E = _skew(T_TRUE) @ R_TRUE
    # The test's generic axis must not be parallel to t (the spec's precondition for the
    # twisted-pair statement below).
    assert abs(float(AXIS @ T_TRUE)) < 0.99
    return dict(pts1=pts1, pts2=pts2, E=E, n=n)


def test_true_pose_is_a_hypothesis_with_all_votes(scene):
    votes = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, T_TRUE)
    assert votes["chosen"] is not None
    c = votes["chosen"]
    assert votes["votes_all"][c] == scene["n"]
    for i in range(4):
        if i != c:
            assert votes["votes_all"][i] < scene["n"], votes
    assert abs(votes["rot_deg"] - 2.0) <= 1e-3
    # Every point is within the default far cutoff here, so votes_near agrees with votes_all.
    assert votes["votes_near"][c] == scene["n"]


def test_near_threshold_below_every_depth_starves_votes_near(scene):
    base = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, T_TRUE)
    starved = twoview_hypothesis_votes(
        scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, T_TRUE, near_thresh=1.0
    )
    assert starved["votes_near"] == [0, 0, 0, 0]
    assert starved["votes_all"] == base["votes_all"]
    assert starved["chosen"] == base["chosen"]


def test_twisted_pair_is_a_hypothesis_but_not_the_argmax(scene):
    R_twist = _rot_about(T_TRUE, 180.0) @ R_TRUE  # half-turn about the baseline, on the LEFT
    votes = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_twist, T_TRUE)
    assert votes["chosen"] is not None, votes
    assert votes["chosen"] != int(np.argmax(votes["votes_all"])), votes
    assert abs(votes["rot_deg"] - rotation_angle_deg(R_twist)) < 1e-9
    # The right-multiplied form is NOT one of the four hypotheses for a generic axis.
    R_wrong = R_TRUE @ _rot_about(T_TRUE, 180.0)
    votes_wrong = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_wrong, T_TRUE)
    assert votes_wrong["chosen"] is None


def test_negated_baseline_is_a_different_hypothesis(scene):
    votes = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, -T_TRUE)
    good = twoview_hypothesis_votes(scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, T_TRUE)
    assert votes["chosen"] is not None and votes["chosen"] != good["chosen"]
    assert votes["votes_all"] == good["votes_all"]  # the scores do not depend on the choice
    assert votes["votes_all"][votes["chosen"]] < scene["n"]


def test_unrelated_pose_gives_chosen_none(scene):
    votes = twoview_hypothesis_votes(
        scene["E"], scene["pts1"], scene["pts2"], K, _rot_about([1, 0, 0], 45.0), T_TRUE
    )
    assert votes["chosen"] is None
    assert sum(votes["votes_all"]) > 0


def test_format_twoview_line_matches_regex():
    votes = dict(votes_all=[0, 0, 103, 324], votes_near=[0, 0, 0, 1], chosen=3, rot_deg=0.0)
    line = format_twoview_line(2, 427, 0.0, np.array([[0.2383], [0.5796], [-0.7792]]), votes)
    m = TWOVIEW_RE.match(line)
    assert m is not None, line
    assert m.groups() == ("2", "427", "0.0000", "0.2383", "0.5796", "-0.7792",
                          "0", "0", "103", "324", "0", "0", "0", "1", "3")
    votes["chosen"] = None
    line = format_twoview_line(9, 12, 179.99996, [1.0, 0.0, 0.0], votes)
    m = TWOVIEW_RE.match(line)
    assert m is not None, line
    assert m.group(15) == "none" and m.group(3) == "180.0000"


def test_diagnostic_prints_on_real_scene(scene, capsys):
    _print_twoview_diagnostic(scene["E"], scene["pts1"], scene["pts2"], K, R_TRUE, T_TRUE.reshape(3, 1), 42)
    out = [l for l in capsys.readouterr().out.splitlines() if "[TWOVIEW]" in l]
    assert len(out) == 1
    m = TWOVIEW_RE.match(out[0])
    assert m is not None, out[0]
    assert m.group(1) == "42" and m.group(2) == "200" and m.group(15) != "none"
    assert int(m.group(7 + int(m.group(15)))) == 200


def test_diagnostic_never_raises_and_prints_zero_votes(capsys):
    # A malformed E makes the vote computation raise; the line must still be printed with
    # votes 0/0/0/0 and chosen=none, and the caller must not see the exception.
    _print_twoview_diagnostic(None, np.zeros((5, 2)), np.zeros((5, 2)), K, R_TRUE, T_TRUE, 7)
    out = [l for l in capsys.readouterr().out.splitlines() if "[TWOVIEW]" in l]
    assert len(out) == 1
    m = TWOVIEW_RE.match(out[0])
    assert m is not None, out[0]
    assert m.groups()[6:14] == ("0",) * 8
    assert m.group(15) == "none"
    assert abs(float(m.group(3)) - 2.0) < 1e-3  # rot_deg still comes from the chosen R


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
