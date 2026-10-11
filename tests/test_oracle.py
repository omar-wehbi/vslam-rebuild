"""Stage 03c: the oracle ladder pose rungs (oracle.py, main.py flags, frame.py injection).

Synthetic ground truth (spec D): 8 poses, fixed seed; rotations of 2 to 6 deg about generic
axes; translations of 5 to 20 mm; pair (0, 1) STATIONARY (identical positions, small
rotation) so that D4's scale pair is (1, 2) at 5 mm; frame 5's timestamp lies 0.3 s from
every row inside a 1.0 s mocap gap (no truth); timestamps 33 ms apart otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geometry  # noqa: E402
import oracle as oracle_mod  # noqa: E402
from oracle import Oracle, quat_to_R, rotation_angle_deg  # noqa: E402
from tracking import MotionEstimate  # noqa: E402

RX_ORACLE = re.compile(
    r"^\s*>> \[ORACLE\] frame=(\d+) rung=(o1|o2|o3|o4) action=(inject|hold|skip) "
    r"reason=(ok|bridge|no_truth|vo_fail|homography) "
    r"class=(translating|near_stationary|stationary|undefined) gt_step_m=(\S+) inj_rot_deg=(\S+) "
    r"inj_step=(\S+) est_rot_deg=(\S+) est_step=(\S+) rot_err_deg=(\S+) dir_err_deg=(\S+) accepted=(0|1|-)$"
)
RX_SETUP = re.compile(
    r"^\s*>> \[ORACLE-SETUP\] rung=(o1|o2|o3|o4) stationary=(estimator|zero) frames=(\d+) "
    r"frames_with_truth=(\d+) frames_interp=(\d+) scale_pair=(\d+),(\d+) gt_step_m=(\S+) s=(\S+) anchor=(\d+)$"
)
FLOAT_RX = re.compile(r"^(-?\d+\.\d{6}|nan)$")

N_FRAMES = 8
NO_TRUTH_K = 5


def _rotvec_to_R(v):
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-15:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _R_to_quat(R):
    """(qx, qy, qz, qw), scalar last (TUM)."""
    m = R; tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2; w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w]); return q / np.linalg.norm(q)


def make_synthetic_gt(tmp_path):
    """Returns (gt_path, timestamps, G: list of world_T_cam or None per frame)."""
    rng = np.random.default_rng(20260918)
    # per-step relative motions in the camera frame (frame k-1 -> k), frames 1..7
    steps = []
    for k in range(1, N_FRAMES):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = math.radians(rng.uniform(2.0, 6.0))
        R = _rotvec_to_R(axis * ang)
        if k == 1:
            t = np.zeros(3)                       # STATIONARY pair (0, 1): identical positions
        elif k == 2:
            d = rng.normal(size=3); d /= np.linalg.norm(d); t = 0.005 * d   # 5 mm: the scale pair
        else:
            d = rng.normal(size=3); d /= np.linalg.norm(d); t = rng.uniform(0.008, 0.020) * d
        steps.append((R, t))
    # frame times: 33 ms apart, except frame 5 which sits 0.3 s after row 4 inside a 1.0 s gap
    ts = [1000.0 + 0.033 * k for k in range(NO_TRUTH_K)]
    ts.append(ts[-1] + 0.3)
    t6 = ts[NO_TRUTH_K - 1] + 1.0
    ts += [t6, t6 + 0.033]
    # world poses by composition (T_k = T_{k-1} @ step_k as a camera-frame increment)
    T = np.eye(4); G_all = [T.copy()]
    for R, t in steps:
        S = np.eye(4); S[:3, :3] = R; S[:3, 3] = t
        T = T @ S; G_all.append(T.copy())
    rows = []
    for k in range(N_FRAMES):
        if k == NO_TRUTH_K:
            continue  # no mocap row for frame 5 (gap 1.0 s between rows of frames 4 and 6)
        q = _R_to_quat(G_all[k][:3, :3]); p = G_all[k][:3, 3]
        rows.append(f"{ts[k]:.6f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    gt = tmp_path / "groundtruth_synth.txt"
    gt.write_text("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(rows) + "\n")
    # the truth the oracle should see: re-read from the file rows (quaternion-normalised)
    orc = Oracle("o3", str(gt), ts)
    G = [orc.gt_world_T_cam(k) for k in range(N_FRAMES)]
    return gt, ts, G


def _dummy_motion(R=None, t=None, model="ESSENTIAL_MATRIX", inlier_matches=None):
    return MotionEstimate(
        R=np.eye(3) if R is None else np.asarray(R, dtype=np.float64),
        t=np.array([[1.0], [0.0], [0.0]]) if t is None else np.asarray(t, dtype=np.float64).reshape(3, 1),
        inlier_matches=[] if inlier_matches is None else inlier_matches, all_matches=["m1", "m2", "m3"], model=model,
        inliers_E=10, inliers_H=2, epipolar_err_before=0.7, epipolar_err_after=0.3,
    )


def _random_est_motion(rng):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    R = _rotvec_to_R(axis * math.radians(rng.uniform(1.0, 5.0)))
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    return _dummy_motion(R, u)


def _scaled_rel(G, a, b, s):
    T = np.linalg.inv(G[a]) @ G[b]
    T = T.copy(); T[:3, 3] *= s
    return T


def _o3_chain(gt, ts):
    """T_k chained through geometry.rt_to_t_inv from override_motion, exactly as update_pose does."""
    orc = Oracle("o3", str(gt), ts)
    rng = np.random.default_rng(1)
    T = np.eye(4); chain = [T.copy()]; recs = []
    for k in range(1, N_FRAMES):
        mo, rec = orc.override_motion(k, _random_est_motion(rng))
        T = T @ geometry.rt_to_t_inv(mo.R, mo.t)
        chain.append(T.copy()); recs.append(rec)
    return orc, chain, recs


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth(tmp_path):
    return make_synthetic_gt(tmp_path)


# --------------------------------------------------------------------------- #
# A3 / A4 / A5: truth at frame times, scale, anchor
# --------------------------------------------------------------------------- #
def test_truth_status_and_no_truth_frame(synth):
    gt, ts, G = synth
    orc = Oracle("o3", str(gt), ts)
    for k in range(N_FRAMES):
        assert orc.has_truth(k) == (k != NO_TRUTH_K)
    assert not orc.has_truth(NO_TRUTH_K)
    assert orc.frames_with_truth == N_FRAMES - 1
    assert orc.frames_interp == 0
    assert orc.anchor_k == 0
    assert G[NO_TRUTH_K] is None


def test_interpolation_between_close_rows(tmp_path):
    """A frame 10 ms off a row is 'exact'; one halfway between rows 0.1 s apart is 'interp';
    one 0.3 s from every row inside a 1.0 s gap has no truth."""
    rows = []
    T = np.eye(4)
    for i, t in enumerate([100.0, 100.1, 100.2, 101.2]):
        T = T.copy(); T[0, 3] = 0.01 * i
        q = _R_to_quat(T[:3, :3]); rows.append(f"{t:.6f} {T[0,3]:.6f} 0 0 {q[0]} {q[1]} {q[2]} {q[3]}")
    gt = tmp_path / "gt.txt"; gt.write_text("\n".join(rows) + "\n")
    ts = [100.01, 100.15, 100.5]
    orc = Oracle("o1", str(gt), ts)
    assert orc.status(0) == "exact" and orc.has_truth(0)
    assert orc.status(1) == "interp" and orc.has_truth(1)
    assert abs(orc.gt_world_T_cam(1)[0, 3] - 0.015) < 1e-12
    assert orc.status(2) == "none" and not orc.has_truth(2)
    assert orc.frames_interp == 1


def test_scale_pair_skips_stationary_first_pair(synth):
    """D4: s = 1 / m_gt of the first pair with m_gt >= 3 mm; pair (0, 1) is stationary."""
    gt, ts, G = synth
    orc = Oracle("o1", str(gt), ts)
    _, _, m01 = orc.gt_relative(0, 1)
    _, _, m12 = orc.gt_relative(1, 2)
    assert m01 < oracle_mod.STATIONARY_M
    assert abs(m12 - 0.005) < 1e-9
    assert orc.scale_pair == (1, 2)
    assert abs(orc.scale_s - 1.0 / m12) < 1e-9
    assert abs(orc.scale_gt_step_m - m12) < 1e-12


def test_no_scale_pair_raises(tmp_path):
    rows = []
    for i in range(4):
        rows.append(f"{10.0 + 0.033 * i:.6f} 0 0 0 0 0 0 1")
    gt = tmp_path / "gt_static.txt"; gt.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        Oracle("o3", str(gt), [10.0 + 0.033 * i for i in range(4)])


def test_bad_rung_and_stationary_rejected(synth):
    gt, ts, _ = synth
    with pytest.raises(ValueError):
        Oracle("o5", str(gt), ts)
    with pytest.raises(ValueError):
        Oracle("o2", str(gt), ts, stationary="median")


def test_gt_relative_convention(synth):
    """gt_relative(a, b) is in the MotionEstimate convention: rt_to_t_inv(R, t) == G_a^-1 G_b."""
    gt, ts, G = synth
    orc = Oracle("o3", str(gt), ts)
    for a, b in ((0, 1), (1, 2), (2, 3), (4, 6), (0, 7)):
        R, t, m = orc.gt_relative(a, b)
        assert R.shape == (3, 3) and t.shape == (3, 1)
        Rel = np.linalg.inv(G[a]) @ G[b]
        assert np.abs(geometry.rt_to_t_inv(R, t) - Rel).max() < 1e-12
        assert abs(m - np.linalg.norm(Rel[:3, 3])) < 1e-12
        # x_b = R x_a + t maps the camera-a origin to its coordinates in camera b
        assert np.abs((R @ np.zeros((3, 1)) + t).ravel() - (np.linalg.inv(Rel)[:3, 3])).max() < 1e-12
    with pytest.raises(ValueError):
        orc.gt_relative(4, NO_TRUTH_K)


# --------------------------------------------------------------------------- #
# D1: convention and composition with rung o3 (hold on no-truth, bridge after it)
# --------------------------------------------------------------------------- #
def test_d1_o3_chain_reproduces_scaled_truth(synth):
    gt, ts, G = synth
    orc, chain, recs = _o3_chain(gt, ts)
    s = orc.scale_s
    for k in range(N_FRAMES):
        if k == NO_TRUTH_K:
            continue
        expect = _scaled_rel(G, 0, k, s)
        assert np.linalg.norm(chain[k][:3, :3] - expect[:3, :3]) < 1e-9, k
        assert np.abs(chain[k][:3, 3] - expect[:3, 3]).max() < 1e-9, k
    # the no-truth frame held the previous pose
    assert np.abs(chain[NO_TRUTH_K] - chain[NO_TRUTH_K - 1]).max() == 0.0
    actions = [(r["action"], r["reason"]) for r in recs]
    assert actions[NO_TRUTH_K - 1] == ("hold", "no_truth")
    assert actions[NO_TRUTH_K] == ("inject", "bridge")
    assert all(a == ("inject", "ok") for i, a in enumerate(actions) if i not in (NO_TRUTH_K - 1, NO_TRUTH_K))
    assert orc.counts == {"inject": 6, "hold": 1, "skip": 0, "bridge": 1, "accepted": 0}


def test_d1_harness_style_dummy_motion(synth):
    """The harness self-test's form: identity dummy motion, unit x translation."""
    gt, ts, G = synth
    orc = Oracle("o3", str(gt), ts)
    T = np.eye(4)
    for k in range(1, NO_TRUTH_K):
        mo, _ = orc.override_motion(k, _dummy_motion())
        T = T @ geometry.rt_to_t_inv(mo.R, mo.t)
        R0, t0, _ = orc.gt_relative(0, k)
        Gk = geometry.rt_to_t_inv(R0, t0); Gk[:3, 3] *= orc.scale_s
        assert np.abs(T - Gk).max() < 1e-9


# --------------------------------------------------------------------------- #
# D2: o1 and o2 rules, the record's error columns
# --------------------------------------------------------------------------- #
def test_d2_o1_keeps_estimator_rotation_and_direction(synth):
    gt, ts, G = synth
    orc = Oracle("o1", str(gt), ts)
    rng = np.random.default_rng(2)
    for k in (1, 2, 3, 4, 6, 7):
        est = _random_est_motion(rng)
        est.inlier_matches = ["m1", "m2"]
        mo, rec = orc.override_motion(k, est)
        a = k - 1 if k != 6 else 4
        R_gt, t_gt, m_gt = orc.gt_relative(a, k)
        assert np.abs(mo.R - est.R).max() == 0.0
        assert abs(np.linalg.norm(mo.t) - orc.scale_s * m_gt) < 1e-12
        if m_gt > 0:
            cos = float(mo.t.ravel() @ est.t.ravel()) / (np.linalg.norm(mo.t) * np.linalg.norm(est.t))
            assert abs(cos - 1.0) < 1e-12
        else:
            assert np.linalg.norm(mo.t) == 0.0
        assert rec["action"] == "inject"
        assert abs(rec["rot_err_deg"] - rotation_angle_deg(est.R.T @ R_gt)) < 1e-9
        assert abs(rec["est_rot_deg"] - rotation_angle_deg(est.R)) < 1e-9
        assert abs(rec["est_step"] - 1.0) < 1e-12
        assert abs(rec["gt_step_m"] - m_gt) < 1e-12
        assert abs(rec["inj_step"] - np.linalg.norm(mo.t)) < 1e-12
        # kept from the estimator: everything but R and t
        assert mo.inlier_matches is est.inlier_matches and mo.all_matches is est.all_matches
        assert mo.model == est.model and mo.inliers_E == est.inliers_E and mo.inliers_H == est.inliers_H
        assert mo.epipolar_err_before == est.epipolar_err_before and mo.epipolar_err_after == est.epipolar_err_after
        assert mo.R.shape == (3, 3) and mo.t.shape == (3, 1) and mo.t.dtype == np.float64
    # the stationary pair (0, 1) has class stationary and a nan direction error
    rec1 = orc.override_motion(1, _random_est_motion(rng))[1]
    assert rec1["class"] == "stationary" and math.isnan(rec1["dir_err_deg"])
    assert not math.isnan(rec1["rot_err_deg"])


def test_d2_o2_true_rotation_and_direction(synth):
    gt, ts, G = synth
    rng = np.random.default_rng(3)
    for mode in ("estimator", "zero"):
        orc = Oracle("o2", str(gt), ts, stationary=mode)
        for k in (2, 3, 4, 7):
            est = _random_est_motion(rng)
            mo, rec = orc.override_motion(k, est)
            R_gt, t_gt, m_gt = orc.gt_relative(k - 1, k)
            assert rec["class"] == "translating"
            assert np.abs(mo.R - R_gt).max() < 1e-15
            assert abs(np.linalg.norm(mo.t) - 1.0) < 1e-12
            assert np.abs(mo.t - t_gt / m_gt).max() < 1e-12
            u_est = est.t.ravel() / np.linalg.norm(est.t)
            u_gt = (t_gt / m_gt).ravel()
            ang = math.degrees(math.atan2(np.linalg.norm(np.cross(u_est, u_gt)), u_est @ u_gt))
            assert abs(rec["dir_err_deg"] - ang) < 1e-9
            assert abs(rec["rot_err_deg"] - rotation_angle_deg(est.R.T @ R_gt)) < 1e-9
            assert abs(rec["inj_rot_deg"] - rotation_angle_deg(R_gt)) < 1e-9
        # the stationary pair (0, 1)
        est = _random_est_motion(rng)
        mo, rec = orc.override_motion(1, est)
        R_gt, _, m_gt = orc.gt_relative(0, 1)
        assert rec["class"] == "stationary"
        assert np.abs(mo.R - R_gt).max() < 1e-15
        assert math.isnan(rec["dir_err_deg"])
        if mode == "estimator":
            assert np.abs(mo.t - est.t / np.linalg.norm(est.t)).max() < 1e-12
            assert abs(rec["inj_step"] - 1.0) < 1e-12
        else:
            assert np.linalg.norm(mo.t) == 0.0 and rec["inj_step"] == 0.0


def test_near_stationary_class_gets_unit_true_direction(tmp_path):
    """A 2 mm pair is near_stationary: o2 still injects u_gt and reports dir_err."""
    rows = []
    T = np.eye(4)
    for i, dx in enumerate([0.0, 0.002, 0.012]):
        q = _R_to_quat(np.eye(3)); rows.append(f"{50.0 + 0.033 * i:.6f} {dx:.6f} 0 0 {q[0]} {q[1]} {q[2]} {q[3]}")
    gt = tmp_path / "gt.txt"; gt.write_text("\n".join(rows) + "\n")
    orc = Oracle("o2", str(gt), [50.0, 50.033, 50.066])
    assert orc.scale_pair == (1, 2)
    mo, rec = orc.override_motion(1, _dummy_motion(t=[0.0, 1.0, 0.0]))
    assert rec["class"] == "near_stationary"
    # the camera moved +x in the world, so in x_b = R x_a + t the true t points along -x
    assert np.abs(mo.t.ravel() - [-1.0, 0.0, 0.0]).max() < 1e-12
    assert abs(rec["dir_err_deg"] - 90.0) < 1e-9


# --------------------------------------------------------------------------- #
# skip paths, no mutation, undefined rungs
# --------------------------------------------------------------------------- #
def test_override_motion_skips_vo_fail_and_homography(synth):
    gt, ts, _ = synth
    for rung in ("o1", "o2", "o3"):
        orc = Oracle(rung, str(gt), ts)
        mo, rec = orc.override_motion(2, None)
        assert mo is None and rec["action"] == "skip" and rec["reason"] == "vo_fail" and rec["accepted"] == "-"
        assert math.isnan(rec["rot_err_deg"]) and math.isnan(rec["dir_err_deg"]) and math.isnan(rec["est_rot_deg"])
        est = _dummy_motion(model="HOMOGRAPHY")
        mo, rec = orc.override_motion(2, est)
        assert mo is est and rec["action"] == "skip" and rec["reason"] == "homography"
        assert math.isnan(rec["rot_err_deg"]) and math.isnan(rec["dir_err_deg"])
        assert orc.counts["skip"] == 2 and orc.counts["inject"] == 0


def test_override_motion_never_mutates_input(synth):
    gt, ts, _ = synth
    for rung in ("o1", "o2", "o3"):
        orc = Oracle(rung, str(gt), ts)
        est = _random_est_motion(np.random.default_rng(9))
        R0, t0 = est.R.copy(), est.t.copy()
        for k in (2, NO_TRUTH_K, 6):
            mo, _ = orc.override_motion(k, est)
            assert mo is not est
            assert np.abs(est.R - R0).max() == 0.0 and np.abs(est.t - t0).max() == 0.0
            assert not np.shares_memory(mo.t, est.t)


def test_hold_on_no_truth_is_identity_motion(synth):
    gt, ts, _ = synth
    for rung in ("o1", "o2", "o3"):
        orc = Oracle(rung, str(gt), ts)
        est = _random_est_motion(np.random.default_rng(4))
        mo, rec = orc.override_motion(NO_TRUTH_K, est)
        assert np.abs(mo.R - np.eye(3)).max() == 0.0 and np.linalg.norm(mo.t) == 0.0
        assert mo.t.shape == (3, 1)
        assert rec["action"] == "hold" and rec["reason"] == "no_truth" and rec["class"] == "undefined"
        assert rec["inj_rot_deg"] == 0.0 and rec["inj_step"] == 0.0
        assert not math.isnan(rec["est_rot_deg"])
        assert mo.inlier_matches is est.inlier_matches and mo.model == est.model


def test_rung_methods_undefined_for_other_rungs(synth):
    gt, ts, _ = synth
    with pytest.raises(ValueError):
        Oracle("o4", str(gt), ts).override_motion(1, _dummy_motion())
    with pytest.raises(ValueError):
        Oracle("o3", str(gt), ts).absolute_pose(1)


# --------------------------------------------------------------------------- #
# D3: o4 shares the o3 gauge
# --------------------------------------------------------------------------- #
def test_d3_o4_absolute_pose_matches_o3_chain(synth):
    gt, ts, G = synth
    _, chain, _ = _o3_chain(gt, ts)
    orc4 = Oracle("o4", str(gt), ts)
    assert orc4.anchor_k == 0
    assert np.abs(orc4.anchor_pose() - np.eye(4)).max() == 0.0
    assert abs(orc4.scale_s - Oracle("o3", str(gt), ts).scale_s) < 1e-15
    for k in range(N_FRAMES):
        if k == NO_TRUTH_K:
            assert orc4.absolute_pose(k) is None
            continue
        cam_T_world = orc4.absolute_pose(k)
        assert np.abs(np.linalg.inv(cam_T_world) - chain[k]).max() < 1e-9, k


def test_o4_anchor_beyond_frame_zero(tmp_path):
    """anchor_k > 0: absolute_pose needs the captured pose E and maps through it."""
    rows = []
    for i in range(1, 4):
        rows.append(f"{70.0 + 0.033 * i:.6f} {0.01 * i:.6f} 0 0 0 0 0 1")
    gt = tmp_path / "gt.txt"; gt.write_text("\n".join(rows) + "\n")
    ts = [70.0 - 2.0, 70.033, 70.066, 70.099]   # frame 0 is 2 s before the first row: no truth
    orc = Oracle("o4", str(gt), ts)
    assert orc.anchor_k == 1 and not orc.has_truth(0) and orc.scale_pair == (1, 2)
    assert orc.anchor_pose() is None
    with pytest.raises(RuntimeError):
        orc.absolute_pose(2)
    E = np.eye(4); E[:3, :3] = _rotvec_to_R([0, 0, math.radians(30)]); E[:3, 3] = [1, 2, 3]
    orc.set_anchor_pose(E)
    T2 = np.linalg.inv(orc.absolute_pose(2))
    rel = np.linalg.inv(orc.gt_world_T_cam(1)) @ orc.gt_world_T_cam(2); rel[:3, 3] *= orc.scale_s
    assert np.abs(T2 - E @ rel).max() < 1e-12
    assert np.abs(np.linalg.inv(orc.absolute_pose(1)) - E).max() < 1e-12


def test_o4_record_counts_and_fields(synth):
    gt, ts, _ = synth
    orc = Oracle("o4", str(gt), ts)
    rec = orc.o4_record(NO_TRUTH_K, "skip", "no_truth")
    assert rec["action"] == "skip" and rec["accepted"] == 0 and rec["class"] == "undefined"
    prev = np.eye(4); chained = np.eye(4); chained[:3, 3] = [1, 0, 0]
    inj = np.linalg.inv(orc.absolute_pose(3))
    rec = orc.o4_record(3, "inject", "ok", prev, chained, inj, accepted=1)
    assert rec["accepted"] == 1 and rec["class"] == "translating"
    assert abs(rec["est_step"] - 1.0) < 1e-12 and rec["est_rot_deg"] == 0.0
    assert abs(rec["inj_step"] - np.linalg.norm(inj[:3, 3])) < 1e-12
    assert abs(rec["inj_rot_deg"] - rotation_angle_deg(inj[:3, :3])) < 1e-9
    assert math.isnan(rec["rot_err_deg"]) and math.isnan(rec["dir_err_deg"])
    assert orc.counts["inject"] == 1 and orc.counts["skip"] == 1 and orc.counts["accepted"] == 1


# --------------------------------------------------------------------------- #
# D5: line formats
# --------------------------------------------------------------------------- #
def test_d5_format_line_and_setup_line(synth):
    gt, ts, _ = synth
    orc = Oracle("o3", str(gt), ts)
    rng = np.random.default_rng(5)
    lines = [
        orc.format_line(orc.override_motion(2, _random_est_motion(rng))[1]),        # inject
        orc.format_line(orc.override_motion(NO_TRUTH_K, _random_est_motion(rng))[1]),  # hold
        orc.format_line(orc.override_motion(3, None)[1]),                           # skip
    ]
    orc4 = Oracle("o4", str(gt), ts)
    lines.append(orc4.format_line(orc4.o4_record(3, "inject", "ok", np.eye(4), np.eye(4), np.linalg.inv(orc4.absolute_pose(3)), accepted=1)))
    lines.append(orc4.format_line(orc4.o4_record(NO_TRUTH_K, "skip", "no_truth")))
    expect = [("inject", "ok", "-"), ("hold", "no_truth", "-"), ("skip", "vo_fail", "-"), ("inject", "ok", "1"), ("skip", "no_truth", "0")]
    for line, (action, reason, acc) in zip(lines, expect):
        m = RX_ORACLE.match(line)
        assert m is not None, line
        assert m.group(3) == action and m.group(4) == reason and m.group(13) == acc
        for g in range(6, 13):
            assert FLOAT_RX.match(m.group(g)), (line, m.group(g))
    assert "gt_step_m=nan" in lines[1] and "rot_err_deg=nan" in lines[1] and "dir_err_deg=nan" in lines[1]
    m = RX_SETUP.match(orc.setup_line())
    assert m is not None, orc.setup_line()
    assert m.group(1) == "o3" and m.group(2) == "estimator"
    assert int(m.group(3)) == N_FRAMES and int(m.group(4)) == N_FRAMES - 1 and int(m.group(5)) == 0
    assert (int(m.group(6)), int(m.group(7))) == (1, 2)
    assert abs(float(m.group(9)) - orc.scale_s) < 1e-5 and int(m.group(10)) == 0
    z = Oracle("o2", str(gt), ts, stationary="zero")
    assert RX_SETUP.match(z.setup_line()).group(2) == "zero"
    assert re.match(r"^  Oracle: rung=o3 inject=\d+ hold=\d+ skip=\d+ bridge=\d+ accepted=\d+$", orc.summary_line())


# --------------------------------------------------------------------------- #
# D6: CLI and resolved settings
# --------------------------------------------------------------------------- #
def test_d6_cli_defaults_and_resolved_settings(tmp_path):
    import main as main_mod
    p = main_mod.get_parser()
    a = p.parse_args(["--dataset", "rgb"])
    assert a.oracle == "none" and a.oracle_stationary == "estimator"
    b = p.parse_args(["--dataset", "rgb", "--oracle", "o3", "--oracle_stationary", "zero"])
    assert b.oracle == "o3" and b.oracle_stationary == "zero"
    with pytest.raises(SystemExit):
        p.parse_args(["--oracle", "o9"])
    out = tmp_path / "resolved_settings.json"
    main_mod.write_resolved_settings(b, out)
    data = json.loads(out.read_text())
    assert data["oracle"] == "o3" and data["oracle_stationary"] == "zero"
    main_mod.write_resolved_settings(a, out)
    data = json.loads(out.read_text())
    assert data["oracle"] == "none" and data["oracle_stationary"] == "estimator"


def test_main_oracle_error_exits_2_without_ground_truth(monkeypatch, tmp_path, capsys):
    import main as main_mod
    calls = []
    monkeypatch.setattr(main_mod, "iter_frames", lambda *a, **kw: (None, 3, iter([])))
    monkeypatch.setattr(main_mod, "show_trajectory", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(main_mod, "write_resolved_settings", lambda args, out_path: None)
    monkeypatch.setattr(main_mod, "load_timestamps_for_images", lambda *a, **kw: None)
    p = main_mod.get_parser()
    args = p.parse_args(["--dataset", "rgb", "--no_display", "--save_estimated", str(tmp_path / "e.txt"), "--oracle", "o3"])
    with pytest.raises(SystemExit) as ex:
        main_mod.run_pipeline(args)
    assert ex.value.code == 2
    assert ">> ORACLE ERROR:" in capsys.readouterr().out
    assert calls == []   # no frame processed
    # with a ground-truth file but no usable timestamps: still exit 2
    gt = tmp_path / "gt.txt"; gt.write_text("1 0 0 0 0 0 0 1\n2 0.01 0 0 0 0 0 1\n")
    args = p.parse_args(["--dataset", "rgb", "--no_display", "--save_estimated", str(tmp_path / "e.txt"),
                         "--ground_truth", str(gt), "--oracle", "o1"])
    with pytest.raises(SystemExit) as ex:
        main_mod.run_pipeline(args)
    assert ex.value.code == 2 and ">> ORACLE ERROR:" in capsys.readouterr().out and calls == []


def test_main_oracle_none_passes_no_oracle(monkeypatch, tmp_path):
    import main as main_mod
    calls = []
    monkeypatch.setattr(main_mod, "iter_frames", lambda *a, **kw: (None, 0, iter([])))
    monkeypatch.setattr(main_mod, "show_trajectory", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(main_mod, "write_resolved_settings", lambda args, out_path: None)
    p = main_mod.get_parser()
    args = p.parse_args(["--dataset", "rgb", "--no_display", "--ground_truth", "nonexistent_gt.txt"])
    main_mod.run_pipeline(args)
    assert len(calls) == 1 and calls[0]["oracle"] is None


def test_main_builds_oracle_with_flag(monkeypatch, tmp_path, synth):
    gt, ts, _ = synth
    import main as main_mod
    calls = []
    monkeypatch.setattr(main_mod, "iter_frames", lambda *a, **kw: (None, N_FRAMES, iter([])))
    monkeypatch.setattr(main_mod, "show_trajectory", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(main_mod, "write_resolved_settings", lambda args, out_path: None)
    monkeypatch.setattr(main_mod, "load_timestamps_for_images", lambda *a, **kw: [(t, f"{i}.png") for i, t in enumerate(ts)])
    rgb = tmp_path / "rgb.txt"; rgb.write_text("# fake\n")
    p = main_mod.get_parser()
    args = p.parse_args(["--dataset", "rgb", "--no_display", "--save_estimated", str(tmp_path / "e.txt"),
                         "--timestamp_file", str(rgb), "--ground_truth", str(gt), "--oracle", "o2", "--oracle_stationary", "zero"])
    main_mod.run_pipeline(args)
    orc = calls[0]["oracle"]
    assert isinstance(orc, Oracle) and orc.rung == "o2" and orc.stationary == "zero"
    assert orc.n_frames == N_FRAMES and orc.scale_pair == (1, 2)


# --------------------------------------------------------------------------- #
# frame.py injection points with a stubbed estimator
# --------------------------------------------------------------------------- #
def _frames_pair():
    from frame import Frame
    img = np.zeros((48, 64), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    prev = Frame(img, K, "ORB", "BF")
    curr = Frame(img, K, "ORB", "BF")
    prev.set_world_pose(np.eye(4))
    return prev, curr


def test_update_pose_injection_point_1_o3(monkeypatch, capsys, synth):
    """The injected motion is what update_pose composes; one [ORACLE] line before [SCALE-LOG]."""
    import frame as frame_mod
    gt, ts, G = synth
    prev, curr = _frames_pair()
    k = curr.id - 1   # Frame.id is a global counter: build truth for frames 0..k (generic motion)
    rng = np.random.default_rng(11)
    T = np.eye(4); rows = []
    for i in range(k + 1):
        if i:
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            S = np.eye(4); S[:3, :3] = _rotvec_to_R(axis * math.radians(3.0)); S[:3, 3] = rng.normal(size=3) * 0.01
            T = T @ S
        q = _R_to_quat(T[:3, :3])
        rows.append(f"{4000.0 + 0.033 * i:.6f} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    gt2 = gt.with_name("gt_ip1.txt"); gt2.write_text("\n".join(rows) + "\n")
    orc = Oracle("o3", str(gt2), [4000.0 + 0.033 * i for i in range(k + 1)])
    assert orc.has_truth(k) and orc.has_truth(k - 1), (k, orc.n_frames)
    est = _random_est_motion(rng)
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: est)
    curr.update_pose(prev, world_map=None, use_relative_scale=False, oracle=orc)   # reference configuration
    out = capsys.readouterr().out.splitlines()
    oracle_lines = [l for l in out if "[ORACLE]" in l]
    assert len(oracle_lines) == 1 and RX_ORACLE.match(oracle_lines[0])
    i_or = next(i for i, l in enumerate(out) if "[ORACLE]" in l)
    i_sl = next(i for i, l in enumerate(out) if "[SCALE-LOG]" in l)
    assert i_or < i_sl
    R_gt, t_gt, m_gt = orc.gt_relative(k - 1, k)
    expect = geometry.rt_to_t_inv(R_gt, orc.scale_s * t_gt)
    assert np.abs(curr.world_T_cam - expect).max() < 1e-9
    assert curr._last_motion is not est and np.abs(curr._last_motion.R - R_gt).max() < 1e-15
    sl = next(l for l in out if "[SCALE-LOG]" in l)
    assert f"norm_t={orc.scale_s * m_gt:.6f}" in sl and "scale=1.000000" in sl and "reason=unit_disabled" in sl


def test_update_pose_without_oracle_prints_nothing(monkeypatch, capsys):
    import frame as frame_mod
    prev, curr = _frames_pair()
    est = _random_est_motion(np.random.default_rng(12))
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: est)
    curr.update_pose(prev, world_map=None)
    out = capsys.readouterr().out
    assert "[ORACLE" not in out
    assert np.abs(curr.world_T_cam - geometry.rt_to_t_inv(est.R, est.t)).max() < 1e-12


def test_update_pose_o4_skips_pnp_without_truth(monkeypatch, capsys, synth):
    """o4 on a PnP frame without truth: no pnp_localize call, no gate, one skip line."""
    import frame as frame_mod
    gt, ts, _ = synth
    prev, curr = _frames_pair()
    k = curr.id - 1
    # a truth file with rows only far away from frame k's timestamp: k has no truth
    ts2 = [1000.0 + 0.033 * i for i in range(k + 1)]
    rows = [f"{2000.0 + 0.033 * i:.6f} {0.01 * i:.6f} 0 0 0 0 0 1" for i in range(3)]
    gt2 = gt.with_name("gt_far.txt"); gt2.write_text("\n".join(rows) + "\n")
    orc = Oracle("o4", str(gt2), ts2 + [2000.0, 2000.033, 2000.066])
    assert not orc.has_truth(k)
    est = _random_est_motion(np.random.default_rng(13))
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: est)
    called = []
    monkeypatch.setattr(frame_mod, "pnp_localize", lambda *a, **kw: called.append(1) or (False, None, 0, None))
    from Map import Map
    wm = Map(points=[], frames=[prev, curr], max_points=100)
    stats = {"pnp_success": 0, "pnp_fail": 0, "pnp_solver_wins": {"oracle": 0}, "homography": 0, "essential": 0}
    curr.update_pose(prev, world_map=wm, use_pnp=True, stats=stats, oracle=orc)
    out = capsys.readouterr().out
    assert called == []
    assert "[PnP-GATE]" not in out and stats["pnp_success"] == 0 and stats["pnp_fail"] == 0
    lines = [l for l in out.splitlines() if "[ORACLE]" in l]
    assert len(lines) == 1
    m = RX_ORACLE.match(lines[0]); assert m and m.group(2) == "o4" and m.group(3) == "skip" and m.group(4) == "no_truth" and m.group(13) == "0"


def test_update_pose_o4_injects_through_gate(monkeypatch, capsys, synth):
    """o4 on a PnP frame with truth: pnp_localize is not called, the gate runs as built and
    the injected pose (chained from prev with the true increment) is accepted."""
    import frame as frame_mod
    gt, ts, _ = synth
    prev, curr = _frames_pair()
    k = curr.id - 1
    # truth: frames 0..k, 1 cm steps along x, anchor 0 = identity
    rows = [f"{3000.0 + 0.033 * i:.6f} {0.01 * i:.6f} 0 0 0 0 0 1" for i in range(k + 1)]
    gt2 = gt.with_name("gt_o4.txt"); gt2.write_text("\n".join(rows) + "\n")
    orc = Oracle("o4", str(gt2), [3000.0 + 0.033 * i for i in range(k + 1)])
    assert orc.has_truth(k) and orc.scale_pair == (0, 1) and abs(orc.scale_s - 100.0) < 1e-9
    # the estimator's chain sits where the truth says frame k-1 is; its unit step is along x
    prev.set_world_pose(np.linalg.inv(orc.absolute_pose(k - 1)))
    est = _dummy_motion(np.eye(3), [-1.0, 0.0, 0.0])   # x_curr = x_prev - 1 -> camera moved +x by 1 unit
    monkeypatch.setattr(frame_mod, "estimate_motion", lambda p, c, use_homography=True: est)
    monkeypatch.setattr(frame_mod, "pnp_localize", lambda *a, **kw: pytest.fail("pnp_localize must not run under o4"))
    from Map import Map
    wm = Map(points=[], frames=[prev, curr], max_points=100)
    stats = {"pnp_success": 0, "pnp_fail": 0, "pnp_solver_wins": {"oracle": 0}, "homography": 0, "essential": 0,
             "recent_vo_steps": [1.0] * 5, "pnp_consecutive_failures": 0}
    curr.update_pose(prev, world_map=wm, use_pnp=True, stats=stats, oracle=orc)
    out = capsys.readouterr().out
    assert "[PnP-GATE]" in out and "PnP localization succeeded" in out
    assert stats["pnp_success"] == 1 and stats["pnp_fail"] == 0 and stats["pnp_solver_wins"]["oracle"] == 1
    assert np.abs(curr.cam_T_world - orc.absolute_pose(k)).max() < 1e-9
    lines = [l for l in out.splitlines() if "[ORACLE]" in l]
    assert len(lines) == 1
    m = RX_ORACLE.match(lines[0])
    assert m and m.group(3) == "inject" and m.group(13) == "1"
    assert abs(float(m.group(8)) - 1.0) < 1e-6 and abs(float(m.group(10)) - 1.0) < 1e-6   # inj_step, est_step
    assert m.group(11) == "nan" and m.group(12) == "nan"
    # the line is printed after the gate decision
    assert out.index("[PnP-GATE]") < out.index("[ORACLE]")
    assert orc.counts["accepted"] == 1 and orc.counts["inject"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
