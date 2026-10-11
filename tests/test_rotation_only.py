"""Stage 04r: the pre-map rotation-only mode (rotation_only.py) and its flag (main.py).

R1 the --tracking rotation choice and validate_tracking_args; R2 wahba; R3 solve_rotation on a
synthetic scene with outliers; R4 its two failure reasons; R5 view_coverage; R6 the tracker on
synthetic yawing frames (orientation, switch rule, zero centres, [ROT-KF] lines); R7 the failure
path (constant-angular-velocity prediction, the keyframe stays) and the error path; R8 the no-leak
tokens; R9 the pipeline keywords.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
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

import main  # noqa: E402
import pipeline  # noqa: E402
import rotation_only as ro  # noqa: E402
from frame import DEFAULT_K  # noqa: E402
from optimization import so3_exp  # noqa: E402

K = np.asarray(DEFAULT_K, dtype=np.float64)
W, H = 640, 480
F = float(K[0, 0])

ROT_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
            "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba", "--no_point_fusion"]

# the harness's line formats (premap_rotation_score.py), copied verbatim
P = r"^\s*>>\s+"
RX_KF = re.compile(P + r"\[ROT-KF\] frame=(\d+) prev=(-?\d+) n=(\d+)\s*$")
RX_ONLY = re.compile(P + r"\[ROT-ONLY\] frame=(\d+) state=(measured|unsupported) kf=(\d+) matches=(\d+) inliers=(\d+) "
                     r"rel_rot_deg=(\S+) coverage=(\S+) reason=(none|matches|inliers|error)\s*$")
RX_ERR = re.compile(P + r"\[ROT-ONLY-ERROR\] frame=(\d+) error=(.*)$")
RX_SUM = re.compile(P + r"\[ROT-SUMMARY\] frames=(\d+) measured=(\d+) unsupported=(\d+) keyframes=(\d+) map_points=(\d+)\s*$")
SETUP_LINE = "  >> [ROT-SETUP] hypotheses=512 seed=43 gate_sigmas=3.000000 min_inliers=5 grid=4x3 min_coverage=0.750000"


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def angle(R):
    return ro.rotation_angle_deg(R)


# --------------------------------------------------------------------------- R1
def _refused(argv):
    args = main.get_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.validate_tracking_args(args)
        return None
    except SystemExit as e:
        return str(e)


def test_r1_choice_parses_and_defaults_unchanged():
    pr = main.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.tracking == "chain" and a.m1_map is None and a.map_reloc == "none" and a.reloc_solver == "epnp"
    assert a.oracle == "none" and a.point_source == "keyframe_pair"
    assert pr.parse_args(ROT_ARGV).tracking == "rotation"
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        pr.parse_args(["--dataset", "rgb", "--tracking", "pnp"])


def test_r1_validate_accepts_the_flag_set():
    assert _refused(ROT_ARGV) is None
    # the front end is not constrained
    assert _refused([x for x in ROT_ARGV if x not in ("--front_end", "sift")]) is None
    # chain and map behave as before
    assert _refused(["--dataset", "rgb"]) is None
    assert "--m1_map" in _refused(["--dataset", "rgb", "--m1_map", "/tmp/m1.txt"])


@pytest.mark.parametrize("flag", ["--default_k", "--undistort", "--no_loop_closure", "--no_relocalization",
                                  "--no_ba", "--no_point_fusion"])
def test_r1_refuses_a_missing_required_flag(flag):
    msg = _refused([x for x in ROT_ARGV if x != flag])
    assert msg == f"--tracking rotation requires {flag}"


def test_r1_refuses_pnp_interval():
    argv = list(ROT_ARGV)
    argv[argv.index("--pnp_interval") + 1] = "10"
    assert _refused(argv) == "--tracking rotation requires --pnp_interval 0"


@pytest.mark.parametrize("extra,flag", [
    (["--m1_map", "/tmp/m1.txt"], "--m1_map"), (["--map_reloc", "pnp"], "--map_reloc"),
    (["--reloc_solver", "sqpnp"], "--reloc_solver"), (["--reloc_solver", "newton"], "--reloc_solver"),
    (["--reloc_shadow"], "--reloc_shadow"), (["--epi_refine"], "--epi_refine"), (["--lm_confirm"], "--lm_confirm"),
    (["--oracle", "o3"], "--oracle"), (["--point_source", "depth_filter"], "--point_source")])
def test_r1_refuses_each_refused_flag(extra, flag):
    msg = _refused(ROT_ARGV + extra)
    assert msg is not None and msg.startswith("--tracking rotation refuses") and flag in msg


# --------------------------------------------------------------------------- R2
def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_r2_wahba_exact_on_noise_free_bearings():
    rng = np.random.default_rng(1)
    a = _unit(rng.normal(size=(20, 3)))
    R = so3_exp(rng.normal(size=3) * 0.4)
    b = a @ R.T
    Rh = ro.wahba(a, b)
    err = math.radians(angle(Rh.T @ R))
    assert err < 1e-9
    assert abs(np.linalg.det(Rh) - 1.0) < 1e-12


def test_r2_wahba_never_returns_a_reflection():
    rng = np.random.default_rng(2)
    a = _unit(rng.normal(size=(20, 3)))
    b = a @ np.diag([1.0, 1.0, -1.0])        # a mirror image: the unconstrained fit is the reflection
    U, _S, Vt = np.linalg.svd(b.T @ a)
    assert np.linalg.det(U @ Vt) < 0          # the input does invite a reflection
    Rh = ro.wahba(a, b)
    assert abs(np.linalg.det(Rh) - 1.0) < 1e-12
    assert np.allclose(Rh.T @ Rh, np.eye(3), atol=1e-12)


# --------------------------------------------------------------------------- R3 / R4
def _scene(seed=11, n=300, outlier_frac=0.3, noise_px=0.5):
    rng = np.random.default_rng(seed)
    Ki = np.linalg.inv(K)
    ax = _unit(rng.normal(size=(1, 3)))[0]
    R = so3_exp(ax * math.radians(rng.uniform(5.0, 25.0)))
    px = np.column_stack((rng.uniform(0, W, n), rng.uniform(0, H, n), np.ones(n)))
    a = _unit(px @ Ki.T)
    pb = (a @ R.T) @ K.T
    pb = pb[:, :2] / pb[:, 2:] + rng.normal(0.0, noise_px, (n, 2))
    no = int(outlier_frac * n)
    out = rng.choice(n, no, replace=False)
    pb[out] = np.column_stack((rng.uniform(0, W, no), rng.uniform(0, H, no)))
    b = _unit(np.column_stack((pb, np.ones(n))) @ Ki.T)
    return a, b, R, out


def test_r3_solve_rotation_with_outliers():
    a, b, R, out = _scene()
    s = np.ones(len(a))
    Rh, mask, reason = ro.solve_rotation(a, b, s, s, F, ro.RotOnlyParams())
    assert reason == "none" and Rh is not None
    assert angle(Rh.T @ R) < 0.1
    assert np.mean(~mask[out]) >= 0.95
    assert mask.sum() >= 0.6 * len(a)
    Rh2, mask2, reason2 = ro.solve_rotation(a, b, s, s, F, ro.RotOnlyParams())
    assert reason2 == reason and Rh2.tobytes() == Rh.tobytes() and mask2.tobytes() == mask.tobytes()


def test_r3_gate_uses_the_keypoint_sigmas():
    """A 5 px error is inside the gate at sigma 2 (3 * sqrt(8) = 8.5 px) and outside at sigma 0.25
    (3 * sqrt(0.125) = 1.06 px, less than half the error, so no hypothesis between the true
    rotation and the shifted points keeps both): the per-match gate reads sigma_a and sigma_b."""
    a, b, R, _ = _scene(seed=5, outlier_frac=0.0, noise_px=0.0)
    Ki = np.linalg.inv(K)
    pb = b @ K.T
    pb = pb[:, :2] / pb[:, 2:]
    pb[:10, 0] += 5.0
    b2 = _unit(np.column_stack((pb, np.ones(len(pb)))) @ Ki.T)
    wide = np.full(len(a), 2.0)
    narrow = np.full(len(a), 0.25)
    _R1, m1, r1 = ro.solve_rotation(a, b2, wide, wide, F, ro.RotOnlyParams())
    _R2, m2, r2 = ro.solve_rotation(a, b2, narrow, narrow, F, ro.RotOnlyParams())
    assert r1 == r2 == "none"
    assert m1[:10].all() and not m2[:10].any()


def test_r4_too_few_matches():
    a, b, _R, _ = _scene(n=4, outlier_frac=0.0)
    s = np.ones(4)
    Rh, mask, reason = ro.solve_rotation(a, b, s, s, F, ro.RotOnlyParams())
    assert Rh is None and reason == "matches" and mask.shape == (4,) and not mask.any()
    Rh, mask, reason = ro.solve_rotation(np.zeros((0, 3)), np.zeros((0, 3)), [], [], F, ro.RotOnlyParams())
    assert Rh is None and reason == "matches" and mask.shape == (0,)


def test_r4_unrelated_bearings():
    rng = np.random.default_rng(3)
    a = _unit(rng.normal(size=(60, 3)))
    b = _unit(rng.normal(size=(60, 3)))
    s = np.ones(60)
    Rh, mask, reason = ro.solve_rotation(a, b, s, s, F, ro.RotOnlyParams())
    assert Rh is None and reason == "inliers"
    assert 0 < mask.sum() < 5 or mask.sum() == 0


# --------------------------------------------------------------------------- R5
def test_r5_view_coverage():
    p = ro.RotOnlyParams()
    assert ro.view_coverage(K, np.eye(3), W, H, p) == 1.0
    assert ro.view_coverage(K, rot_y(10.0), W, H, p) == 0.75
    assert ro.view_coverage(K, rot_y(-10.0), W, H, p) == 0.75
    assert ro.view_coverage(K, rot_y(24.0), W, H, p) == 0.5
    assert ro.view_coverage(K, rot_y(-24.0), W, H, p) == 0.5
    assert ro.view_coverage(K, rot_y(180.0), W, H, p) == 0.0


def test_r5_view_coverage_convention():
    """Keyframe at the identity, current frame with cam_T_world rotation R (so R = R_(k,P)): a cell
    centre counts when its world direction, world_T_cam(k) applied to its bearing, projects inside
    the keyframe image. For R = rot_y(21) rot_x(5) that is 9 of 12 centres; the flipped convention
    (R instead of R.T) gives 8, so this pins the direction of the mapping."""
    p = ro.RotOnlyParams()
    R = rot_y(21.0) @ rot_x(5.0)
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_wc = np.linalg.inv(T_cw)
    count = 0
    for j in range(3):
        for i in range(4):
            d = np.linalg.solve(K, np.array([(i + 0.5) * W / 4, (j + 0.5) * H / 3, 1.0]))
            q = T_wc[:3, :3] @ d
            if q[2] > 0:
                u, v, _ = K @ (q / q[2])
                count += int(0 <= u < W and 0 <= v < H)
    assert count == 9
    assert ro.view_coverage(K, R, W, H, p) == count / 12
    assert ro.view_coverage(K, R.T, W, H, p) != count / 12


# --------------------------------------------------------------------------- R6 / R7 stubs
class _Map:
    def __init__(self):
        self.points = []


class _Frame:
    """Frame-like stub: SIFT-like keypoints at the exact projections of the scene directions it sees,
    one float32 descriptor per scene direction."""

    def __init__(self, fid, R_cw, dirs, desc, empty=False):
        self.id = fid
        self.K = K
        self.method = "SIFT"
        self.match_type = "knn"
        self.img_shape = (H, W)
        self.pose_valid = True
        self.tracking_state = "OK"
        self.cam_T_world = np.eye(4)
        self.world_T_cam = np.eye(4)
        self.KeyPoints = []
        self.Descriptors = None
        if empty:
            return
        q = dirs @ R_cw.T
        z = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (q @ K.T)[:, :2] / z[:, None]
        vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        for u, v in uv[vis]:
            kp = cv2.KeyPoint(float(u), float(v), 3.0)
            kp.octave = 0
            self.KeyPoints.append(kp)
        self.Descriptors = np.asarray(desc, dtype=np.float32)[vis]

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _yaw_scene(n_frames, empty=(), seed=0, n_dirs=3000, pitch_per_frame=0.0):
    """Frame k's cam_T_world rotation is (rot_y(k) rot_x(pitch_per_frame * k)).T (1 deg of yaw per
    frame, the optical axis at azimuth +k deg; with pitch the rotations do not commute); the scene
    directions cover azimuth -50..100 deg and elevation +-55 deg."""
    rng = np.random.default_rng(seed)
    az = np.radians(rng.uniform(-50.0, 100.0, n_dirs))
    el = np.radians(rng.uniform(-55.0, 55.0, n_dirs))
    dirs = np.column_stack((np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)))
    desc = rng.uniform(0.0, 255.0, (n_dirs, 128)).astype(np.float32)
    truth = [(rot_y(float(k)) @ rot_x(pitch_per_frame * k)).T for k in range(n_frames)]
    frames = [_Frame(k, truth[k], dirs, desc, empty=(k in empty)) for k in range(n_frames)]
    return frames, truth


def _run(frames):
    tr = ro.RotationOnlyTracker()
    tr.world_map = _Map()
    buf = io.StringIO()
    states = {}
    with contextlib.redirect_stdout(buf):
        print(tr.setup_line())
        tr.start(frames)
        for k in range(1, len(frames)):
            states[k] = tr.on_frame(k, frames)
        print(tr.summary_line())
    return tr, states, buf.getvalue().splitlines()


def _expected_keyframes(truth):
    """The switch rule on the true rotations: frame k becomes a keyframe when the coverage of the
    current keyframe's view (R_(k,P) = R_k R_P^T) is below min_coverage."""
    p = ro.RotOnlyParams()
    kfs = [0]
    for k in range(1, len(truth)):
        if ro.view_coverage(K, truth[k] @ truth[kfs[-1]].T, W, H, p) < p.min_coverage:
            kfs.append(k)
    return kfs


def test_r6_tracker_on_synthetic_yaw():
    frames, truth = _yaw_scene(40)
    tr, states, lines = _run(frames)
    assert all(s == "measured" for s in states.values())
    for k in range(40):
        R_est = frames[k].cam_T_world[:3, :3]
        assert math.radians(angle(R_est.T @ truth[k])) < 1e-6, k
        assert np.linalg.norm(frames[k].world_T_cam[:3, 3]) == 0.0
        assert frames[k].tracking_state == ("OK" if k == 0 else "ROT")
        assert frames[k].pose_valid
    # the first switch: the first frame whose coverage against frame 0 is below 0.75
    p = ro.RotOnlyParams()
    first = next(k for k in range(1, 40) if ro.view_coverage(K, truth[k] @ truth[0].T, W, H, p) < 0.75)
    assert 15 <= first <= 30
    kfs = _expected_keyframes(truth)
    assert kfs[1] == first and len(kfs) >= 2
    kf_lines = [RX_KF.match(l).groups() for l in lines if RX_KF.match(l)]
    assert kf_lines == [(str(f), str(prev), str(i + 1)) for i, (f, prev) in enumerate(zip(kfs, [-1] + kfs[:-1]))]
    # line order: [ROT-ONLY] of the switching frame immediately precedes its [ROT-KF]
    for f in kfs[1:]:
        i = next(i for i, l in enumerate(lines) if RX_ONLY.match(l) and RX_ONLY.match(l).group(1) == str(f))
        assert RX_KF.match(lines[i + 1]).group(1) == str(f)
        assert float(RX_ONLY.match(lines[i]).group(7)) < 0.75
    # every [ROT-ONLY] line parses, names the keyframe solved against, and its rel_rot_deg is the
    # angle of R_(k,kf) from the written orientations
    kf = 0
    for l in lines:
        m = RX_KF.match(l)
        if m:
            kf = int(m.group(1))
            continue
        m = RX_ONLY.match(l)
        if m:
            k = int(m.group(1))
            assert int(m.group(3)) == kf or int(m.group(1)) in kfs
            Rk = frames[k].cam_T_world[:3, :3]
            Rkf = frames[int(m.group(3))].cam_T_world[:3, :3]
            assert abs(float(m.group(6)) - angle(Rk @ Rkf.T)) < 1e-6
            assert m.group(2) == "measured" and m.group(8) == "none" and int(m.group(5)) >= 5
    assert lines[0] == SETUP_LINE
    assert RX_SUM.match(lines[-1]).groups() == ("40", "39", "0", str(len(kfs)), "0")


def test_r6_composition_across_switches_with_noncommuting_rotations():
    """Yaw plus pitch: R_k = R_(k,P) R_P must compose in that order across keyframe switches."""
    frames, truth = _yaw_scene(50, pitch_per_frame=0.4, n_dirs=4000)
    tr, states, lines = _run(frames)
    assert all(s == "measured" for s in states.values())
    assert tr.n_kf >= 3
    assert _expected_keyframes(truth) == [int(RX_KF.match(l).group(1)) for l in lines if RX_KF.match(l)]
    for k in range(50):
        assert math.radians(angle(frames[k].cam_T_world[:3, :3].T @ truth[k])) < 1e-6, k


def test_r7_prediction_with_noncommuting_rotations():
    frames, truth = _yaw_scene(6, empty=(3,), pitch_per_frame=3.0)
    _tr, states, _lines = _run(frames)
    assert states[3] == "unsupported"
    R1, R2 = frames[1].cam_T_world[:3, :3], frames[2].cam_T_world[:3, :3]
    assert np.allclose(frames[3].cam_T_world[:3, :3], (R2 @ R1.T) @ R2, atol=1e-12)
    # an increment taken the wrong way round (V = R_(k-2) R_(k-1)^T) would give R_(k-2)-like poses
    assert not np.allclose(frames[3].cam_T_world[:3, :3], (R1 @ R2.T) @ R2, atol=1e-6)


def test_r6_summary_counts_the_map_and_exposes_a_missing_handoff():
    tr = ro.RotationOnlyTracker()
    assert tr.world_map is None
    assert tr.summary_line().endswith("map_points=-1")
    tr.world_map = _Map()
    tr.world_map.points = [object(), object()]
    assert tr.summary_line().endswith("map_points=2")


def test_r7_frame_without_descriptors_is_unsupported():
    frames, truth = _yaw_scene(6, empty=(3,))
    tr, states, lines = _run(frames)
    assert states[3] == "unsupported" and all(states[k] == "measured" for k in (1, 2, 4, 5))
    only = {int(m.group(1)): m.groups() for m in (RX_ONLY.match(l) for l in lines) if m}
    k3 = only[3]
    assert k3[1] == "unsupported" and k3[2] == "0" and k3[3] == "0" and k3[4] == "0"
    assert k3[5] == "nan" and k3[6] == "nan" and k3[7] == "matches"
    # the pose is the constant-angular-velocity prediction from frames 1 and 2, not the identity
    R1, R2 = frames[1].cam_T_world[:3, :3], frames[2].cam_T_world[:3, :3]
    R_pred = (R2 @ R1.T) @ R2
    assert np.allclose(frames[3].cam_T_world[:3, :3], R_pred, atol=1e-12)
    assert angle(frames[3].cam_T_world[:3, :3]) > 2.9
    assert np.linalg.norm(frames[3].world_T_cam[:3, 3]) == 0.0
    assert frames[3].pose_valid is False and frames[3].tracking_state == "LOST"
    # the keyframe stays: frame 4 is solved against frame 0, and no [ROT-KF] follows frame 3
    assert only[4][2] == "0" and tr.kf == 0
    assert [l for l in lines if RX_KF.match(l)] == ["  >> [ROT-KF] frame=0 prev=-1 n=1"]
    assert math.radians(angle(frames[4].cam_T_world[:3, :3].T @ truth[4])) < 1e-6
    assert RX_SUM.match(lines[-1]).groups() == ("6", "4", "1", "1", "0")


def test_r7_prediction_at_frame_1_is_frame_0():
    """V is the identity at k = 1: frame 1 carries frame 0's rotation (the identity in a real run,
    where frame 0 is the gauge; a non-identity frame 0 here separates the rule from a constant)."""
    frames, _truth = _yaw_scene(2, empty=(1,))
    _tr, states, _lines = _run(frames)
    assert states[1] == "unsupported"
    assert np.allclose(frames[1].cam_T_world, np.eye(4))
    frames, _truth = _yaw_scene(2, empty=(1,))
    T0 = np.eye(4)
    T0[:3, :3] = rot_y(5.0) @ rot_x(2.0)
    frames[0].set_camera_pose(T0)
    _tr, states, _lines = _run(frames)
    assert states[1] == "unsupported"
    assert np.allclose(frames[1].cam_T_world[:3, :3], T0[:3, :3], atol=1e-12)


def test_r7_exception_is_an_error_line_and_unsupported():
    frames, _truth = _yaw_scene(4)
    frames[2].K = None          # the bearing construction raises on frame 2
    tr, states, lines = _run(frames)
    assert states[2] == "unsupported" and states[3] == "measured"
    i = next(i for i, l in enumerate(lines) if RX_ERR.match(l))
    assert RX_ERR.match(lines[i]).group(1) == "2" and "\n" not in lines[i]
    m = RX_ONLY.match(lines[i + 1])
    assert m and m.group(1) == "2" and m.group(2) == "unsupported" and m.group(8) == "error"
    R0, R1 = frames[0].cam_T_world[:3, :3], frames[1].cam_T_world[:3, :3]
    assert np.allclose(frames[2].cam_T_world[:3, :3], (R1 @ R0.T) @ R1, atol=1e-12)
    assert frames[2].tracking_state == "LOST"


# --------------------------------------------------------------------------- R8 / R9
def test_r8_no_leak_tokens():
    src = Path(ro.__file__).read_text()
    for tok in ("groundtruth", "depth.txt", "depth/", "cv2.imread"):
        assert tok not in src, tok
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not mods & {"oracle", "dataset", "pipeline"}, mods


def test_r9_pipeline_keywords():
    s = inspect.signature(pipeline.show_trajectory).parameters
    assert "rot_tracker" in s and s["rot_tracker"].default is None
    assert "m1_tracker" in s and s["m1_tracker"].default is None
