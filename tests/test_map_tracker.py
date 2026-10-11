"""Stage 04l: the map-based tracker against a perfect map (map_tracker.py) and its flags (main.py).

T1 flags and validate_tracking_args; T2 the motion-only Jacobian against a central finite difference
through retract_pose; T3 motion-only BA recovery with and without gross outliers; T4 predict_pose;
T5 load_m1_map; T6 MapTracker on stub frames (causal release, m1b uses exactly the C lines, pixel
mismatch, LOST with the prediction); T7 the log line formats of spec E.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import map_tracker as mt  # noqa: E402
from Map import Map  # noqa: E402
from frame import DEFAULT_K  # noqa: E402
from optimization import so3_exp  # noqa: E402

K = np.asarray(DEFAULT_K, dtype=np.float64)
W, H = 640, 480

M1_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "map", "--m1_map", "/tmp/m1.txt",
           "--m1_mode", "m1b", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba",
           "--no_point_fusion"]

# the harness's line formats (spec E), copied verbatim from the check's regexes
RX_SETUP = re.compile(r"\[M1-SETUP\] mode=(m1a|m1b) keyframes=(\d+) points=(\d+) corr=(\d+) front_end=(orb|sift) interval=(\d+) scale_s=(\S+)")
RX_RELEASE = re.compile(r"\[M1-RELEASE\] frame=(\d+) points=(\d+) shadowed=(\d+) refused=(\d+)\s*$")
RX_M1ERR = re.compile(r"\[M1-ERROR\] frame=(-?\d+) kind=(pixel|unreleased|header) detail=(\S+)\s*$")
RX_TRACK = re.compile(r"\[MAP-TRACK\] frame=(\d+) state=(tracked|lost) mode=(m1a|m1b) pred_rot=(twoview|cv) step_a=(-?\d+) "
                      r"step_a_inliers=(-?\d+) step_a_wide=([01]) step_b=(-?\d+) obs=(\d+) inliers=(\d+) rmse_px=(\S+)\s*$")


# --------------------------------------------------------------------------- T1
def _refused(argv):
    args = main.get_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.validate_tracking_args(args)
        return None
    except SystemExit as e:
        return str(e)


def test_t1_flags_parse_with_defaults_and_choices():
    pr = main.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.tracking == "chain" and a.m1_map is None and a.m1_mode == "m1a"
    b = pr.parse_args(M1_ARGV)
    assert b.tracking == "map" and b.m1_map == "/tmp/m1.txt" and b.m1_mode == "m1b"
    # --tracker_delay_ms still parses (the new flag is not a prefix of it)
    assert pr.parse_args(["--dataset", "rgb", "--tracker_delay_ms", "5"]).tracker_delay_ms == 5
    for bad in (["--tracking", "pnp"], ["--m1_mode", "m1c"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            pr.parse_args(["--dataset", "rgb"] + bad)


def test_t1_validate_accepts_defaults_and_the_m1_set():
    assert _refused(["--dataset", "rgb"]) is None
    assert _refused(["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift"]) is None
    assert _refused(M1_ARGV) is None


@pytest.mark.parametrize("drop,flag", [
    (("--m1_map", "/tmp/m1.txt"), "--m1_map"), (("--default_k",), "--default_k"), (("--undistort",), "--undistort"),
    (("--no_loop_closure",), "--no_loop_closure"), (("--no_relocalization",), "--no_relocalization"),
    (("--no_ba",), "--no_ba"), (("--no_point_fusion",), "--no_point_fusion")])
def test_t1_validate_refuses_a_missing_flag(drop, flag):
    msg = _refused([x for x in M1_ARGV if x not in drop])
    assert msg is not None and flag in msg


def test_t1_validate_refuses_the_other_cases():
    argv = list(M1_ARGV)
    argv[argv.index("--pnp_interval") + 1] = "10"
    assert "--pnp_interval" in _refused(argv)
    assert "--epi_refine" in _refused(M1_ARGV + ["--epi_refine"])
    assert "--lm_confirm" in _refused(M1_ARGV + ["--lm_confirm"])
    assert "--m1_map" in _refused(["--dataset", "rgb", "--m1_map", "/tmp/m1.txt"])


def test_t1_run_pipeline_validates_first(monkeypatch):
    """run_pipeline refuses before anything reads args (no frame loader call)."""
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("iter_frames called")))
    args = main.get_parser().parse_args([x for x in M1_ARGV if x != "--no_ba"])
    with pytest.raises(SystemExit):
        main.run_pipeline(args)


# --------------------------------------------------------------------------- scenes
def _rand_pose(rng, rot_deg=20.0, t=0.5):
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * math.radians(rng.uniform(5, rot_deg))
    T = np.eye(4)
    T[:3, :3] = so3_exp(w)
    T[:3, 3] = rng.uniform(-t, t, 3)
    return T


def _scene(rng, T, n):
    Tinv = np.linalg.inv(T)
    X = []
    while len(X) < n:
        z = rng.uniform(1.5, 6.0)
        u, v = rng.uniform(20, 620), rng.uniform(20, 460)
        xc = z * np.linalg.solve(K, [u, v, 1.0])
        X.append(Tinv[:3, :3] @ xc + Tinv[:3, 3])
    X = np.array(X)
    xc = X @ T[:3, :3].T + T[:3, 3]
    p = xc @ K.T
    return X, p[:, :2] / p[:, 2:3], rng.choice([1.0, 2.0, 4.0], size=n)


def _rot_err(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2)))


# --------------------------------------------------------------------------- T2
def test_t2_jacobian_matches_central_difference():
    rng = np.random.default_rng(3)
    for _ in range(3):
        T = _rand_pose(rng)
        X, uv, sg = _scene(rng, T, 40)
        uv = uv + rng.normal(0, 1.0, uv.shape)
        J = mt.motion_only_jacobian(K, T, X, uv, sg)
        assert J.shape == (80, 6)
        eps = 1e-6
        Jfd = np.zeros_like(J)
        for j in range(6):
            e = np.zeros(6)
            e[j] = eps
            Jfd[:, j] = (mt.motion_only_residuals(K, mt.retract_pose(T, e), X, uv, sg)
                         - mt.motion_only_residuals(K, mt.retract_pose(T, -e), X, uv, sg)) / (2 * eps)
        for blk in (slice(0, 3), slice(3, 6)):
            rel = np.linalg.norm(Jfd[:, blk] - J[:, blk]) / np.linalg.norm(Jfd[:, blk])
            assert rel <= 1e-6, rel
        # a sign-flipped rotation block would be caught by the same probe
        Jbad = J.copy()
        Jbad[:, :3] *= -1
        assert np.linalg.norm(Jfd[:, :3] - Jbad[:, :3]) / np.linalg.norm(Jfd[:, :3]) > 0.5


def test_t2_behind_camera_rows():
    rng = np.random.default_rng(4)
    T = _rand_pose(rng)
    X, uv, sg = _scene(rng, T, 5)
    Tinv = np.linalg.inv(T)
    X[2] = Tinv[:3, :3] @ np.array([0.1, 0.2, -2.0]) + Tinv[:3, 3]
    r = mt.motion_only_residuals(K, T, X, uv, sg)
    J = mt.motion_only_jacobian(K, T, X, uv, sg)
    assert np.allclose(r[4:6], 100.0) and np.allclose(J[4:6], 0.0) and not np.allclose(J[0:2], 0.0)


# --------------------------------------------------------------------------- T3
def test_t3_recovery_noise_free_and_with_outliers():
    rng = np.random.default_rng(5)
    P = mt.MapTrackerParams()
    T = _rand_pose(rng)
    X, uv, sg = _scene(rng, T, 200)
    zmean = float(np.mean((X @ T[:3, :3].T + T[:3, 3])[:, 2]))
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * math.radians(5.0)
    dt = rng.normal(size=3)
    dt = dt / np.linalg.norm(dt) * 0.1 * zmean
    T0 = mt.retract_pose(T, np.concatenate([w, dt]))
    Te, mask, info = mt.motion_only_ba(K, T0, X, uv, sg, P)
    assert info["status"] == "ok" and info["n_obs"] == 200 and info["n_inliers"] == 200 and mask.all()
    assert _rot_err(Te[:3, :3], T[:3, :3]) <= 1e-6
    assert np.linalg.norm(Te[:3, 3] - T[:3, 3]) / zmean <= 1e-6
    # 20 percent of the observations moved by 20 to 60 px
    out = rng.choice(200, 40, replace=False)
    uv2 = uv.copy()
    ang = rng.uniform(0, 2 * math.pi, 40)
    mag = rng.uniform(20, 60, 40)
    uv2[out] += np.stack([np.cos(ang), np.sin(ang)], 1) * mag[:, None]
    Te, mask, info = mt.motion_only_ba(K, T0, X, uv2, sg, P)
    expect = np.ones(200, dtype=bool)
    expect[out] = False
    assert math.degrees(_rot_err(Te[:3, :3], T[:3, :3])) <= 0.01
    assert np.array_equal(mask, expect) and info["n_inliers"] == 160


def test_t3_too_few():
    rng = np.random.default_rng(6)
    T = _rand_pose(rng)
    X, uv, sg = _scene(rng, T, 2)
    Tf, mask, info = mt.motion_only_ba(K, T, X, uv, sg, mt.MapTrackerParams())
    assert info["status"] == "too_few" and np.allclose(Tf, T) and mask.shape == (2,) and not mask.any()


# --------------------------------------------------------------------------- T4
def test_t4_predict_pose():
    rng = np.random.default_rng(7)
    T1 = _rand_pose(rng)
    assert np.allclose(mt.predict_pose(T1, None, None), T1)
    R = so3_exp([0.01, -0.02, 0.03])
    Tp = mt.predict_pose(T1, None, R)
    assert np.allclose(Tp[:3, :3], R @ T1[:3, :3])
    T0 = _rand_pose(rng)
    Tp = mt.predict_pose(T1, T0, R)
    assert np.allclose(Tp[:3, :3], R @ T1[:3, :3])
    # constant velocity reproduces a uniformly moving camera exactly
    V = _rand_pose(rng, rot_deg=6.0, t=0.05)
    A = _rand_pose(rng)
    B = V @ A
    C = V @ B
    assert np.allclose(mt.predict_pose(B, A, None), C, atol=1e-12)
    # with the true two-view rotation too
    assert np.allclose(mt.predict_pose(B, A, V[:3, :3]), C, atol=1e-12)


# --------------------------------------------------------------------------- T5
def test_t5_load_m1_map(tmp_path):
    p = tmp_path / "m1.txt"
    p.write_text("# comment\n"
                 "H front_end=sift method=SIFT undistort=1 keyframe_interval=20 scale_s=1.5 anchor_k=0 n_keyframes=1 n_points=2 n_corr=1\n"
                 "P 0 0 3 10.500000 20.250000 0.1 0.2 3.0\n"
                 "P 1 0 7 30.000000 40.000000 -0.1 0.0 2.5\n"
                 "C 5 11 1 31.000000 41.000000\n")
    m = mt.load_m1_map(p)
    assert m.header["front_end"] == "sift" and m.header["keyframe_interval"] == "20"
    assert m.points == [(0, 0, 3, 10.5, 20.25, 0.1, 0.2, 3.0), (1, 0, 7, 30.0, 40.0, -0.1, 0.0, 2.5)]
    assert isinstance(m.points[0][0], int) and isinstance(m.points[0][3], float)
    assert m.corr == {5: [(11, 1, 31.0, 41.0)]}
    bad = tmp_path / "bad.txt"
    bad.write_text("H front_end=sift\nP 0 0 x 1 2 3 4 5\n")
    with pytest.raises(ValueError, match="line 2"):
        mt.load_m1_map(bad)
    bad.write_text("# a\nH front_end=sift\nP 0 0 1 1 2 3 4 5\nC 1 2 3 4\n")
    with pytest.raises(ValueError, match="line 4"):
        mt.load_m1_map(bad)


# --------------------------------------------------------------------------- T6 stub frames
class _Frame:
    """Frame-like stub: SIFT keypoints at the exact projections of the visible scene points."""

    def __init__(self, fid, cam_T_world, X, desc, pids):
        self.id = fid
        self.K = K
        self.method = "SIFT"
        self.img_shape = (H, W)
        self.keypoint_to_point = {}
        self.cam_T_world = np.array(cam_T_world, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.pose_valid = True
        self.tracking_state = "OK"
        uv, z = mt._project(K, self.cam_T_world, X)
        vis = mt._in_image(uv, z, W, H)
        self.pid_of_kp = [int(p) for p in np.asarray(pids)[vis]]
        self.KeyPoints = []
        for u, v in uv[vis]:
            kp = cv2.KeyPoint(float(u), float(v), 3.0)
            kp.octave = 0
            self.KeyPoints.append(kp)
        self.Descriptors = np.asarray(desc, dtype=np.float32)[vis]

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _world(n_frames=5, step=0.01, seed=11):
    """Points 0..99 belong to keyframe 0, points 100..199 to keyframe 2. Camera moves +x by `step`."""
    rng = np.random.default_rng(seed)
    T_true = []
    for k in range(n_frames):
        T = np.eye(4)
        T[0, 3] = -step * k            # cam_T_world of a camera at x = step * k
        T_true.append(T)
    X = []
    for T in (T_true[0], T_true[2]):
        Tinv = np.linalg.inv(T)
        for _ in range(100):
            z = rng.uniform(1.5, 6.0)
            u, v = rng.uniform(40, 600), rng.uniform(40, 440)
            X.append(Tinv[:3, :3] @ (z * np.linalg.solve(K, [u, v, 1.0])) + Tinv[:3, 3])
    X = np.array(X)
    desc = rng.uniform(0, 100, size=(len(X), 128))
    pids = np.arange(len(X))
    frames = [_Frame(k + 1, T_true[k], X, desc, pids) for k in range(n_frames)]
    for f in frames[1:]:
        f.set_camera_pose(np.eye(4))   # only frame 0 starts posed; the tracker sets the rest
    return frames, X, T_true


def _m1map(frames, X, corr=None):
    pts = []
    for kf, lo in ((0, 0), (2, 100)):
        fr = frames[kf]
        for kp, pid in enumerate(fr.pid_of_kp):
            if lo <= pid < lo + 100:
                u, v = fr.KeyPoints[kp].pt
                pts.append((pid, kf, kp, u, v, *X[pid]))
    pts.sort()
    header = dict(front_end="sift", method="SIFT", undistort="1", keyframe_interval="2", scale_s="1.0",
                  anchor_k="0", n_keyframes="2", n_points=str(len(pts)), n_corr=str(sum(len(v) for v in (corr or {}).values())))
    return mt.M1Map(header=header, points=pts, corr=dict(corr or {}))


@pytest.fixture
def no_twoview(monkeypatch):
    monkeypatch.setattr(mt.tracking, "estimate_motion", lambda a, b, use_homography=False: None)


def _run(tracker, frames, wmap, upto):
    tracker.release(0, frames, wmap)
    states = []
    for k in range(1, upto + 1):
        states.append(tracker.on_frame(k, frames, wmap))
        tracker.release(k, frames, wmap)
    return states


def test_t6_m1a_causal_release_and_tracking(no_twoview, capsys):
    frames, X, T_true = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    assert tr.release(0, frames, wmap) == 100 and len(wmap.points) == 100
    assert tr.on_frame(1, frames, wmap) == "tracked"
    assert np.allclose(frames[1].cam_T_world, T_true[1], atol=1e-6)
    assert tr.release(1, frames, wmap) == 0
    # keyframe 2 is tracked against the points released before it only
    assert tr.on_frame(2, frames, wmap) == "tracked"
    assert len(wmap.points) == 100
    tracked_links = dict(frames[2].keypoint_to_point)
    assert len(tracked_links) >= 30 and all(frames[2].pid_of_kp[kp] < 100 for kp in tracked_links)
    assert tr.release(2, frames, wmap) == 100 and len(wmap.points) == 200
    # keyframe 2's own points are linked at release, on their own keypoints
    own = {tr.points_by_pid[p].id: p for p in range(100, 200)}
    rel_links = {kp: own[pid] for kp, pid in frames[2].keypoint_to_point.items() if pid in own}
    assert len(rel_links) == 100 and all(frames[2].pid_of_kp[kp] == p for kp, p in rel_links.items())
    assert tr.on_frame(3, frames, wmap) == "tracked"
    assert np.allclose(frames[3].cam_T_world, T_true[3], atol=1e-6)
    # every link frame 3 made joins a keypoint to its own point
    ids = {pt.id: pid for pid, pt in tr.points_by_pid.items()}
    assert frames[3].keypoint_to_point and all(frames[3].pid_of_kp[kp] == ids[pid]
                                               for kp, pid in frames[3].keypoint_to_point.items())
    assert wmap.check_invariants(frames) == {k: 0 for k in wmap.check_invariants(frames)}
    out = capsys.readouterr().out
    assert "[M1-ERROR]" not in out and "[MAP-TRACK-ERROR]" not in out


def _unlinked_frame0(frames, X):
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.release(0, frames, wmap)
    for kp in list(frames[0].keypoint_to_point):
        wmap.unlink(frames[0], kp)
    return tr, wmap


def test_t6_step_b_local_map_when_step_a_has_nothing(no_twoview, capsys):
    """Frame 0 carries no links: step A finds nothing, step B searches the released map at the
    wide radius from the prediction and tracks the frame."""
    frames, X, T_true = _world()
    tr, wmap = _unlinked_frame0(frames, X)
    capsys.readouterr()
    assert tr.on_frame(1, frames, wmap) == "tracked"
    t = RX_TRACK.search(capsys.readouterr().out)
    assert t.group(5) == "0" and t.group(6) == "0" and t.group(7) == "1" and int(t.group(8)) >= 30
    assert np.allclose(frames[1].cam_T_world, T_true[1], atol=1e-6)


def test_t6_step_b_distance_ratio_filter(no_twoview, capsys):
    """The same frame with every creating keyframe centre moved 50 m away: the distance-ratio test
    removes every candidate and the frame is LOST with no links."""
    frames, X, _T = _world()
    tr, wmap = _unlinked_frame0(frames, X)
    tr._ckf_list = [np.array([0.0, 0.0, -50.0]) for _ in tr._ckf_list]
    tr._dirty = True
    capsys.readouterr()
    assert tr.on_frame(1, frames, wmap) == "lost"
    t = RX_TRACK.search(capsys.readouterr().out)
    assert t.group(8) == "0" and frames[1].keypoint_to_point == {}


def test_t6_m1b_uses_exactly_the_c_lines(no_twoview, capsys):
    frames, X, T_true = _world()
    fr1 = frames[1]
    chosen = [kp for kp, pid in enumerate(fr1.pid_of_kp) if pid < 100][:40]
    corr = {1: [(kp, fr1.pid_of_kp[kp], *fr1.KeyPoints[kp].pt) for kp in chosen]}
    # one C line whose pixel does not match the product keypoint, one naming an unreleased point
    bad_kp = [kp for kp, pid in enumerate(fr1.pid_of_kp) if pid < 100][45]
    u, v = fr1.KeyPoints[bad_kp].pt
    corr[1].append((bad_kp, fr1.pid_of_kp[bad_kp], u + 0.01, v))
    other_kp = [kp for kp, pid in enumerate(fr1.pid_of_kp) if pid >= 100][0]
    corr[1].append((other_kp, fr1.pid_of_kp[other_kp], *fr1.KeyPoints[other_kp].pt))
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.release(0, frames, wmap)
    capsys.readouterr()
    assert tr.on_frame(1, frames, wmap) == "tracked"
    out = capsys.readouterr().out
    errs = [RX_M1ERR.search(l) for l in out.splitlines() if "[M1-ERROR]" in l]
    assert [e.group(2) for e in errs] == ["pixel", "unreleased"]
    tl = RX_TRACK.search(out)
    assert tl.group(9) == "40" and tl.group(10) == "40" and tl.group(5) == "-1" and tl.group(8) == "-1"
    assert sorted(frames[1].keypoint_to_point) == sorted(chosen)
    assert np.allclose(frames[1].cam_T_world, T_true[1], atol=1e-6)


def test_t6_too_few_inliers_is_lost_with_prediction(no_twoview):
    frames, X, _T = _world()
    fr1 = frames[1]
    chosen = [kp for kp, pid in enumerate(fr1.pid_of_kp) if pid < 100][:10]
    corr = {1: [(kp, fr1.pid_of_kp[kp], *fr1.KeyPoints[kp].pt) for kp in chosen]}
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.release(0, frames, wmap)
    T_pred = mt.predict_pose(frames[0].cam_T_world, None, None)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert tr.on_frame(1, frames, wmap) == "lost"
    assert np.allclose(frames[1].cam_T_world, T_pred)
    assert frames[1].pose_valid is False and frames[1].tracking_state == "LOST"
    assert frames[1].keypoint_to_point == {}
    assert RX_TRACK.search(buf.getvalue()).group(10) == "10"


def test_t6_release_pixel_mismatch_is_refused(capsys):
    frames, X, _T = _world()
    mp = _m1map(frames, X)
    pid, kf, kp, u, v, *xyz = mp.points[3]
    mp.points[3] = (pid, kf, kp, u + 0.5, v, *xyz)
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(mp, "m1a", None)
    assert tr.release(0, frames, wmap) == 99
    out = capsys.readouterr().out
    errs = [l for l in out.splitlines() if "[M1-ERROR]" in l]
    assert len(errs) == 1 and RX_M1ERR.search(errs[0]).group(2) == "pixel"
    assert RX_RELEASE.search(out).groups() == ("0", "99", "0", "1")


def test_t6_shadowed_point_keeps_the_older_link(no_twoview, capsys):
    frames, X, _T = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    _run(tr, frames, wmap, 1)
    tr.on_frame(2, frames, wmap)
    kp_old = sorted(frames[2].keypoint_to_point)[0]
    old_pid = frames[2].keypoint_to_point[kp_old]
    # a P line of keyframe 2 on a keypoint the tracker already linked to an older point
    tr._p_by_kf[2] = [(999, 2, kp_old, *frames[2].KeyPoints[kp_old].pt, 0.0, 0.0, 3.0)]
    capsys.readouterr()
    assert tr.release(2, frames, wmap) == 1
    assert frames[2].keypoint_to_point[kp_old] == old_pid
    assert RX_RELEASE.search(capsys.readouterr().out).groups() == ("2", "1", "1", "0")
    assert tr.points_by_pid[999].observations == {}


def test_t6_exception_is_lost_with_error_line(no_twoview, monkeypatch, capsys):
    frames, X, _T = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.release(0, frames, wmap)
    monkeypatch.setattr(tr, "_track_m1a", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert tr.on_frame(1, frames, wmap) == "lost"
    out = capsys.readouterr().out
    assert "[MAP-TRACK-ERROR] frame=1 error=RuntimeError: boom" in out
    assert RX_TRACK.search(out).group(2) == "lost" and frames[1].pose_valid is False


# --------------------------------------------------------------------------- T7
def test_t7_line_formats(no_twoview, monkeypatch, capsys):
    frames, X, _T = _world()
    monkeypatch.setattr(mt.tracking, "estimate_motion",
                        lambda a, b, use_homography=False: types.SimpleNamespace(R=np.eye(3)))
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    print(tr.setup_line())
    _run(tr, frames, wmap, 4)
    out = capsys.readouterr().out.splitlines()
    setup = [RX_SETUP.search(l) for l in out if "[M1-SETUP]" in l]
    assert len(setup) == 1 and setup[0] is not None and setup[0].group(1) == "m1a" and setup[0].group(3) == "200"
    rel = [RX_RELEASE.search(l) for l in out if "[M1-RELEASE]" in l]
    assert [r.group(1) for r in rel] == ["0", "2"] and all(r is not None for r in rel)
    track = [RX_TRACK.search(l) for l in out if "[MAP-TRACK]" in l]
    assert len(track) == 4 and all(t is not None for t in track)
    assert [t.group(1) for t in track] == ["1", "2", "3", "4"]
    assert all(t.group(2) == "tracked" and t.group(4) == "twoview" for t in track)
    assert all(float(t.group(11)) < 1e-3 for t in track)


def test_t7_header_check_prints_header_error(capsys):
    frames, X, _T = _world()
    tr = mt.MapTracker(_m1map(frames, X), "m1b", None)
    assert tr.check_header("sift", "SIFT", True, 2)
    assert not tr.check_header("sift", "SIFT", True, 20)
    errs = [RX_M1ERR.search(l) for l in capsys.readouterr().out.splitlines() if "[M1-ERROR]" in l]
    assert len(errs) == 1 and errs[0].group(2) == "header"


# =========================================================================== stage 04m (spec E)
RX_RELOC = re.compile(r"^  >> \[MAP-RELOC\] frame=(\d+) mode=(m1a|m1b) status=(ok|fail) "
                      r"reason=(none|too_few|pnp|inliers|error) matches=(\d+) ransac_inliers=(-?\d+) inliers=(\d+)$")


# --------------------------------------------------------------------------- R1
def test_r1_map_reloc_flag_and_refusals():
    pr = main.get_parser()
    assert pr.parse_args(["--dataset", "rgb"]).map_reloc == "none"
    assert pr.parse_args(M1_ARGV + ["--map_reloc", "pnp"]).map_reloc == "pnp"
    assert pr.parse_args(M1_ARGV + ["--map_reloc", "none"]).map_reloc == "none"
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        pr.parse_args(["--dataset", "rgb", "--map_reloc", "epnp"])
    assert "--map_reloc" in _refused(["--dataset", "rgb", "--map_reloc", "pnp"])
    assert _refused(["--dataset", "rgb", "--map_reloc", "none"]) is None
    assert _refused(M1_ARGV + ["--map_reloc", "pnp"]) is None
    # every 04l refusal stays, --no_relocalization included
    assert "--no_relocalization" in _refused([x for x in M1_ARGV if x != "--no_relocalization"] + ["--map_reloc", "pnp"])


def test_r1_run_pipeline_sets_the_tracker_attributes(monkeypatch):
    """main.run_pipeline hands --map_reloc and MapRelocParams() to the tracker right after construction."""
    seen = {}
    real = mt.MapTracker

    class _Spy(real):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(main, "MapTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda p: mt.M1Map(header={}, points=[], corr={}))
    for val in ("pnp", "none"):
        args = main.get_parser().parse_args(M1_ARGV + ["--map_reloc", val])
        with pytest.raises(SystemExit):
            main.run_pipeline(args)
        t = seen.pop("tracker")
        assert t.reloc == val and t.reloc_params == mt.MapRelocParams()
    assert real(mt.M1Map(header={}, points=[], corr={}), "m1b").reloc == "none"


# --------------------------------------------------------------------------- R2
def _pose_90_2m(axis):
    T = np.eye(4)
    a = np.asarray(axis, dtype=np.float64)
    T[:3, :3] = so3_exp(a / np.linalg.norm(a) * math.radians(90.0))
    T[:3, 3] = [0.0, 0.0, 2.0]
    return T


def test_r2_params_defaults():
    import dataclasses
    f = dataclasses.fields(mt.MapRelocParams)
    assert [x.name for x in f] == ["ransac_iterations", "ransac_reproj_px", "ransac_confidence", "ransac_min_inliers",
                                   "min_matches_m1a", "ratio_m1a", "search_radius"]
    p = mt.MapRelocParams()
    assert (p.ransac_iterations, p.ransac_reproj_px, p.ransac_confidence, p.ransac_min_inliers, p.min_matches_m1a,
            p.ratio_m1a, p.search_radius) == (300, 2.4477, 0.99, 10, 15, 0.75, 10.0)
    assert abs(p.ransac_reproj_px - math.sqrt(5.991)) < 1e-4


@pytest.mark.parametrize("axis", [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.8, 0.5]])
def test_r2_relocalize_from_correspondences(axis):
    rng = np.random.default_rng(21)
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    T = _pose_90_2m(axis)
    X, uv, sg = _scene(rng, T, 200)
    Te, mask, info = mt.relocalize_from_correspondences(K, X, uv, sg, P, RP)
    assert info["status"] == "ok" and info["n_obs"] == 200 and info["n_inliers"] == 200 and mask.all()
    assert info["ransac_inliers"] >= RP.ransac_min_inliers
    assert _rot_err(Te[:3, :3], T[:3, :3]) <= 1e-6
    assert np.linalg.norm(Te[:3, 3] - T[:3, 3]) / np.linalg.norm(T[:3, 3]) <= 1e-6
    # deterministic
    Te2, mask2, info2 = mt.relocalize_from_correspondences(K, X, uv, sg, P, RP)
    assert np.array_equal(Te, Te2) and np.array_equal(mask, mask2) and info == info2
    # 20 percent of the observations moved by 20 to 60 px: recovered, exactly the moved ones flagged
    out = rng.choice(200, 40, replace=False)
    uv2 = uv.copy()
    ang = rng.uniform(0, 2 * math.pi, 40)
    uv2[out] += np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(20, 60, 40)[:, None]
    To, masko, infoo = mt.relocalize_from_correspondences(K, X, uv2, sg, P, RP)
    expect = np.ones(200, dtype=bool)
    expect[out] = False
    assert infoo["status"] == "ok" and math.degrees(_rot_err(To[:3, :3], T[:3, :3])) <= 0.01
    assert np.array_equal(masko, expect) and infoo["n_inliers"] == 160


def test_r2_seeded_solve_does_not_recover():
    """The known fail behind R2: motion-only BA seeded at the identity does not reach the 90 deg pose,
    so the recovery above needs the unseeded step."""
    rng = np.random.default_rng(21)
    T = _pose_90_2m([0.0, 1.0, 0.0])
    X, uv, sg = _scene(rng, T, 200)
    Ts, _m, _i = mt.motion_only_ba(K, np.eye(4), X, uv, sg, mt.MapTrackerParams())
    assert math.degrees(_rot_err(Ts[:3, :3], T[:3, :3])) > 1.0 or np.linalg.norm(Ts[:3, 3] - T[:3, 3]) > 0.2


def test_r2_too_few_and_pnp():
    rng = np.random.default_rng(22)
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    T = _pose_90_2m([0.0, 1.0, 0.0])
    X, uv, sg = _scene(rng, T, 200)
    Tf, maskf, infof = mt.relocalize_from_correspondences(K, X[:3], uv[:3], sg[:3], P, RP)
    assert Tf is None and maskf.shape == (3,) and not maskf.any()
    assert infof["status"] == "too_few" and infof["ransac_inliers"] == -1
    uvr = np.stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)], 1)
    Tr, maskr, infor = mt.relocalize_from_correspondences(K, X, uvr, sg, P, RP)
    assert Tr is None and maskr.shape == (200,) and not maskr.any() and infor["status"] == "pnp"


# --------------------------------------------------------------------------- R3, R4 (stub frames)
def _c_lines(fr, pids_lo, pids_hi, n):
    chosen = [kp for kp, pid in enumerate(fr.pid_of_kp) if pids_lo <= pid < pids_hi][:n]
    return [(kp, fr.pid_of_kp[kp], *fr.KeyPoints[kp].pt) for kp in chosen]


def _m1b_lost_at_1(frame2_lines=60):
    """M1b world where frame 1 has 10 C lines (LOST) and frames 2..4 are supported by 60."""
    frames, X, T_true = _world()
    corr = {1: _c_lines(frames[1], 0, 100, 10), 2: _c_lines(frames[2], 0, 100, frame2_lines)}
    for k in (3, 4):
        corr[k] = _c_lines(frames[k], 0, 200, 60)
    return frames, X, T_true, corr


def _run_lines(tr, frames, wmap, upto):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        states = _run(tr, frames, wmap, upto)
    return states, buf.getvalue().splitlines()


def test_r3_relocalizes_only_after_a_lost_frame_m1b(no_twoview):
    frames, X, T_true, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    states, out = _run_lines(tr, frames, wmap, 4)
    assert states == ["lost", "tracked", "tracked", "tracked"]
    rel = [l for l in out if "[MAP-RELOC]" in l]
    assert len(rel) == 1
    g = RX_RELOC.match(rel[0])
    assert g is not None and g.groups()[:5] == ("2", "m1b", "ok", "none", "60") and int(g.group(7)) == 60
    # the line comes right before frame 2's [MAP-TRACK] line, which carries the relocalized fields
    i = out.index(rel[0])
    t = RX_TRACK.search(out[i + 1])
    assert t.group(1) == "2" and t.group(2) == "tracked"
    assert (t.group(5), t.group(6), t.group(7), t.group(8), t.group(9), t.group(10)) == ("-1", "-1", "0", "-1", "60", "60")
    assert np.allclose(frames[2].cam_T_world, T_true[2], atol=1e-6)
    assert frames[2].pose_valid is True and frames[2].tracking_state == "OK" and len(frames[2].keypoint_to_point) >= 60
    assert np.allclose(frames[4].cam_T_world, T_true[4], atol=1e-6)


def test_r3_reloc_none_prints_no_line(no_twoview):
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    assert tr.reloc == "none"
    states, out = _run_lines(tr, frames, wmap, 4)
    assert states[0] == "lost" and not any("[MAP-RELOC]" in l for l in out)
    t2 = [RX_TRACK.search(l) for l in out if "[MAP-TRACK] frame=2 " in l][0]
    assert t2.group(9) == "60"                          # 04l tracking from the prediction, no relocalization


def test_r3_failed_relocalization_is_lost_with_prediction(no_twoview):
    frames, X, _T, corr = _m1b_lost_at_1(frame2_lines=3)
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    states, out = _run_lines(tr, frames, wmap, 3)
    assert states == ["lost", "lost", "tracked"]
    rel = [RX_RELOC.match(l) for l in out if "[MAP-RELOC]" in l]
    assert [r.groups() for r in rel] == [("2", "m1b", "fail", "too_few", "3", "-1", "0"),
                                         ("3", "m1b", "ok", "none", "60", rel[1].group(6), "60")]
    # frame 2 carries its prediction: frame 1 (lost) carried frame 0's pose, so the constant velocity
    # from frames 0 and 1 is the identity and the prediction is frame 0's pose (no two-view rotation)
    # no tracking link: the only links on keyframe 2 are its own released points (file pids >= 100)
    own = {tr.points_by_pid[p].id for p in range(100, 200) if p in tr.points_by_pid}
    assert frames[2].pose_valid is False and set(frames[2].keypoint_to_point.values()) <= own
    assert frames[2].tracking_state == "LOST"
    assert np.allclose(frames[2].cam_T_world, frames[0].cam_T_world)
    T_frame2 = [l for l in out if "[MAP-TRACK] frame=2 " in l]
    assert RX_TRACK.search(T_frame2[0]).group(2) == "lost"


def test_r3_relocalization_error_is_lost_with_error_line(no_twoview, monkeypatch):
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    monkeypatch.setattr(mt, "relocalize_with_solver",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    states, out = _run_lines(tr, frames, wmap, 2)
    assert states == ["lost", "lost"]
    assert any("[MAP-TRACK-ERROR] frame=2 error=RuntimeError: boom" in l for l in out)
    g = [RX_RELOC.match(l) for l in out if "[MAP-RELOC]" in l]
    assert len(g) == 1 and g[0].groups() == ("2", "m1b", "fail", "error", "60", "-1", "0")
    t = [RX_TRACK.search(l) for l in out if "[MAP-TRACK] frame=2 " in l][0]
    assert t.group(5) == "-1" and t.group(6) == "-1" and t.group(7) == "0"


def test_r3_relocalizes_m1a_by_global_search(no_twoview, monkeypatch):
    frames, X, T_true = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.reloc = "pnp"
    real = tr._track_m1a

    def forced(k, fr, T):
        if k == 1:
            raise RuntimeError("forced")
        return real(k, fr, T)
    monkeypatch.setattr(tr, "_track_m1a", forced)
    states, out = _run_lines(tr, frames, wmap, 4)
    assert states == ["lost", "tracked", "tracked", "tracked"]
    rel = [RX_RELOC.match(l) for l in out if "[MAP-RELOC]" in l]
    assert len(rel) == 1 and rel[0].group(1) == "2" and rel[0].group(3) == "ok"
    n_visible0 = sum(1 for pid in frames[2].pid_of_kp if pid < 100)
    assert int(rel[0].group(5)) == n_visible0           # every visible released point, matched by descriptor
    t = [RX_TRACK.search(l) for l in out if "[MAP-TRACK] frame=2 " in l][0]
    assert t.group(5) == "-1" and t.group(6) == "-1" and t.group(7) == "0" and int(t.group(8)) >= 0
    assert int(t.group(10)) == int(rel[0].group(7))
    assert np.allclose(frames[2].cam_T_world, T_true[2], atol=1e-6)
    # the tracked frame 3 after it is tracked by the 04l steps (no [MAP-RELOC] line)
    t3 = [RX_TRACK.search(l) for l in out if "[MAP-TRACK] frame=3 " in l][0]
    assert t3.group(5) != "-1"


def test_r3_m1a_guided_search_adds_matches(no_twoview, monkeypatch):
    """The global match keeps only 20 points; the guided search at the first pose finds the other
    visible ones, and the final BA runs on both."""
    frames, X, T_true = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.release(0, frames, wmap)
    frames[1].tracking_state = "LOST"
    tr.reloc = "pnp"
    real = mt.global_match
    monkeypatch.setattr(mt, "global_match", lambda *a, **k: real(*a, **k)[:20])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr.on_frame(2, frames, wmap) == "tracked"
    out = buf.getvalue()
    g = [RX_RELOC.match(l) for l in out.splitlines() if "[MAP-RELOC]" in l][0]
    t = RX_TRACK.search(out)
    n_visible0 = sum(1 for pid in frames[2].pid_of_kp if pid < 100)
    assert g.group(5) == "20" and int(t.group(8)) == n_visible0 - 20 and int(t.group(9)) == n_visible0
    assert np.allclose(frames[2].cam_T_world, T_true[2], atol=1e-6)


def test_r4_motion_model_restarts_after_a_loss(no_twoview, monkeypatch):
    seq = []
    real = mt.predict_pose

    def spy(T_prev, T_prev2, R):
        seq.append(T_prev2 is None)
        return real(T_prev, T_prev2, R)
    monkeypatch.setattr(mt, "predict_pose", spy)
    # k=1: no k-2; k=2: frame 0 counts as tracked; k=3: frame 1 LOST -> None under pnp; k=4: frame 2 OK
    for reloc, expect in (("pnp", [True, False, True, False]), ("none", [True, False, False, False])):
        frames, X, _T, corr = _m1b_lost_at_1()
        wmap = Map(points=[], frames=frames)
        tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
        tr.reloc = reloc
        seq.clear()
        states, _out = _run_lines(tr, frames, wmap, 4)
        assert states[0] == "lost" and seq == expect, (reloc, seq)


# --------------------------------------------------------------------------- R5
def test_r5_global_match_one_to_one_and_ratio():
    norm = cv2.NORM_L2
    M = np.zeros((5, 4))
    M[:, 0] = [0.0, 10.0, 20.0, 30.0, 40.0]            # map rows along one axis
    kp = np.zeros((7, 4))
    kp[:, 0] = [0.0,   # kp0 -> row 0 at 0 (second 10): kept
                1.0,   # kp1 -> row 0 at 1: loses row 0 to kp0 (larger distance)
                10.0,  # kp2 -> row 1 at 0: kept
                25.0,  # kp3 -> rows 2 / 3 at 5 / 5: ratio 1 >= 0.75, dropped
                31.0,  # kp4 -> row 3 at 1: kept
                40.0,  # kp5 -> row 4 at 0: tie with kp6, the lower keypoint index wins
                40.0]  # kp6
    got = mt.global_match(kp, M, norm, bound=100.0, ratio=0.75)
    assert [(j, r) for j, r, _d in got] == [(0, 0), (2, 1), (4, 3), (5, 4)]
    # one point per keypoint and one keypoint per point
    assert len({j for j, _r, _d in got}) == len(got) == len({r for _j, r, _d in got})
    # the ratio is applied: at ratio 1.01 keypoint 3 is kept (ties to the lower row)
    got2 = mt.global_match(kp, M, norm, bound=100.0, ratio=1.01)
    assert (3, 2) in [(j, r) for j, r, _d in got2]
    # the bound is applied
    kp_far = kp.copy()
    kp_far[:, 1] = 50.0
    assert mt.global_match(kp_far, M, norm, bound=49.0, ratio=0.75) == []
    # a single map row: no second distance, only the bound applies
    assert [(j, r) for j, r, _d in mt.global_match(kp[:1], M[:1], norm, bound=1.0, ratio=0.75)] == [(0, 0)]
    # chunking does not change the result
    assert mt.global_match(kp, M, norm, bound=100.0, ratio=0.75, chunk=2) == got


# --------------------------------------------------------------------------- R6
def test_r6_reloc_line_format():
    tr = mt.MapTracker(mt.M1Map(header={}, points=[], corr={}), "m1a")
    line = tr._reloc_line(7, "fail", "pnp", 23, 4, 0)
    assert line == "  >> [MAP-RELOC] frame=7 mode=m1a status=fail reason=pnp matches=23 ransac_inliers=4 inliers=0"
    assert RX_RELOC.match(line) is not None
    assert RX_RELOC.match(tr._reloc_line(3, "ok", "none", 60, -1, 60)).group(6) == "-1"


# =========================================================================== stage 04q (spec F)
SOLVERS = ("epnp", "sqpnp", "newton")
_F = r"(?:-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|-?inf|nan)"
_P = rf"(none|{_F}(?:,{_F}){{11}})"
RX_ATTEMPT = re.compile(
    rf"^  >> \[MAP-RELOC-ATTEMPT\] frame=(?P<frame>\d+) mode=(?P<mode>m1a|m1b) shadow=(?P<shadow>[01]) "
    rf"solver=(?P<solver>epnp|sqpnp|newton) "
    rf"outcome=(?P<outcome>too_few|ransac|refit|cheirality|ba_inliers|inliers|error|ok) matches=(?P<matches>\d+) "
    rf"ransac_inliers=(?P<ransac_inliers>-?\d+) behind=(?P<behind>-?\d+) ba1_inliers=(?P<ba1>-?\d+) "
    rf"guided=(?P<guided>-?\d+) inliers=(?P<inliers>\d+) hull=(?P<hull>{_F}) refit_ms=(?P<refit_ms>{_F}) "
    rf"npnp_converged=(?P<conv>-1|0|1) npnp_gap_rel=(?P<gap>{_F}) refit_pose={_P} pose={_P}$")
RX_SET = re.compile(rf"^  >> \[RELOC-SET\] frame=(\d+) mode=(m1a|m1b) shadow=([01]) n=(\d+) "
                    rf"points=({_F}(?:,{_F})*) uv=({_F}(?:,{_F})*)$")
RX_SOLVE = re.compile(rf"^  >> \[RELOC-SOLVE\] frame=(\d+) shadow=([01]) solver=(epnp|sqpnp|newton) "
                      rf"npnp_gamma=({_F}) npnp_cost=({_F}) npnp_outer=(-?\d+) npnp_newton=(-?\d+) "
                      rf"npnp_ls_evals=(-?\d+) refit_peak_b=(-?\d+)$")
OUT2REASON = {"too_few": "too_few", "ransac": "pnp", "refit": "pnp", "cheirality": "pnp", "ba_inliers": "inliers",
              "inliers": "inliers", "error": "error", "ok": "none"}


def _att(line):
    g = RX_ATTEMPT.match(line)
    assert g is not None, line
    d = g.groupdict()
    d["refit_pose"], d["pose"] = g.group(16), g.group(17)
    return d


def _pose_of(s):
    if s == "none":
        return None
    T = np.eye(4)
    T[:3, :4] = np.array([float(x) for x in s.split(",")]).reshape(3, 4)
    return T


def _scene_90_2m(noise=False, seed=21):
    rng = np.random.default_rng(seed)
    T = _pose_90_2m([0.0, 1.0, 0.0])
    X, uv, sg = _scene(rng, T, 200)
    if noise:
        out = rng.choice(200, 40, replace=False)
        ang = rng.uniform(0, 2 * math.pi, 40)
        uv = uv.copy()
        uv[out] += np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(20, 60, 40)[:, None]
    return T, X, uv, sg


def _ransac_pose(X, uv):
    n_r, cons, T_r = mt._ransac_consensus(K, X, uv, mt.MapRelocParams())
    assert T_r is not None and n_r == int(cons.sum())
    return T_r, cons


# --------------------------------------------------------------------------- Q1
def test_q1_solver_and_shadow_flags():
    pr = main.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.reloc_solver == "epnp" and a.reloc_shadow is False
    PNP = M1_ARGV + ["--map_reloc", "pnp"]
    for s in SOLVERS:
        assert pr.parse_args(PNP + ["--reloc_solver", s]).reloc_solver == s
        assert _refused(PNP + ["--reloc_solver", s]) is None
    assert pr.parse_args(PNP + ["--reloc_shadow"]).reloc_shadow is True
    assert _refused(PNP + ["--reloc_shadow", "--reloc_solver", "newton"]) is None
    for bad in ("npnp", "p3p", "EPNP"):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            pr.parse_args(["--dataset", "rgb", "--reloc_solver", bad])
    for extra, flag in ((["--reloc_solver", "sqpnp"], "--reloc_solver"), (["--reloc_solver", "newton"], "--reloc_solver"),
                        (["--reloc_shadow"], "--reloc_shadow")):
        assert flag in _refused(M1_ARGV + ["--map_reloc", "none"] + extra)
        assert flag in _refused(M1_ARGV + extra)
        assert flag in _refused(["--dataset", "rgb"] + extra)
    # epnp stated explicitly is the default and refused nowhere; every earlier refusal stays
    assert _refused(["--dataset", "rgb", "--reloc_solver", "epnp"]) is None
    assert _refused(M1_ARGV + ["--reloc_solver", "epnp"]) is None
    assert "--map_reloc" in _refused(["--dataset", "rgb", "--map_reloc", "pnp", "--reloc_shadow"])
    assert "--no_ba" in _refused([x for x in PNP if x != "--no_ba"] + ["--reloc_shadow"])
    helps = {a.dest: a.help for a in pr._actions}
    assert "stage 04q, D1 part C" in helps["reloc_solver"] and "stage 04q, D1 part C" in helps["reloc_shadow"]


def test_q1_run_pipeline_hands_solver_and_shadow(monkeypatch):
    seen = {}
    real = mt.MapTracker

    class _Spy(real):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(main, "MapTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda p: mt.M1Map(header={}, points=[], corr={}))
    for extra, solver, shadow in (([], "epnp", False), (["--reloc_solver", "newton"], "newton", False),
                                  (["--reloc_solver", "sqpnp", "--reloc_shadow"], "sqpnp", True)):
        args = main.get_parser().parse_args(M1_ARGV + ["--map_reloc", "pnp"] + extra)
        with pytest.raises(SystemExit):
            main.run_pipeline(args)
        t = seen.pop("tracker")
        assert t.reloc == "pnp" and t.reloc_solver == solver and t.reloc_shadow is shadow
    t = real(mt.M1Map(header={}, points=[], corr={}), "m1a")
    assert t.reloc_solver == "epnp" and t.reloc_shadow is False


# --------------------------------------------------------------------------- Q2
def test_q2_refit_consensus_recovers_the_pose():
    T, X, uv, _sg = _scene_90_2m()
    T_r, cons = _ransac_pose(X, uv)
    for s in SOLVERS:
        Tf, info = mt.refit_consensus(K, X[cons], uv[cons], T_r, s)
        assert Tf is not None and _rot_err(Tf[:3, :3], T[:3, :3]) <= 1e-6
        assert np.linalg.norm(Tf[:3, 3] - T[:3, 3]) / np.linalg.norm(T[:3, 3]) <= 1e-6
        assert info["refit_ms"] >= 0.0
        rec = [info[k] for k in ("npnp_gamma", "npnp_cost", "npnp_outer", "npnp_newton", "npnp_ls_evals")]
        if s == "newton":
            assert info["npnp_converged"] == 1 and info["npnp_gap_rel"] <= 1e-5
            assert all(np.isfinite(rec[:2])) and min(rec[2:]) >= 0
        else:
            assert info["npnp_converged"] == -1 and math.isnan(info["npnp_gap_rel"])
            assert math.isnan(rec[0]) and math.isnan(rec[1]) and rec[2:] == [-1, -1, -1]
    # epnp returns T_ransac bitwise, as a copy
    Te, _i = mt.refit_consensus(K, X[cons], uv[cons], T_r, "epnp")
    assert Te.tobytes() == T_r.tobytes() and Te is not T_r


def test_q2_newton_scale():
    T, X, uv, _sg = _scene_90_2m()
    T_r, cons = _ransac_pose(X, uv)
    T1, i1 = mt.refit_consensus(K, X[cons], uv[cons], T_r, "newton")
    T2, i2 = mt.refit_consensus(K, X[cons] * 331.0, uv[cons], T_r, "newton", 331.0)
    assert i2["npnp_converged"] == 1 and i2["npnp_gap_rel"] <= 1e-5
    assert _rot_err(T2[:3, :3], T1[:3, :3]) <= 1e-9
    assert np.linalg.norm(T2[:3, 3] - 331.0 * T1[:3, 3]) / np.linalg.norm(331.0 * T1[:3, 3]) <= 1e-9
    # the scale is live: at scale 1 on the same points the translation comes out 331 times larger
    T3, _i3 = mt.refit_consensus(K, X[cons] * 331.0, uv[cons], T_r, "newton", 1.0)
    assert np.linalg.norm(T3[:3, 3] - 331.0 * T1[:3, 3]) / np.linalg.norm(331.0 * T1[:3, 3]) <= 1e-6
    T4, _i4 = mt.refit_consensus(K, X[cons], uv[cons], T_r, "newton", 331.0)
    assert np.linalg.norm(T4[:3, 3] - T1[:3, 3]) / np.linalg.norm(T1[:3, 3]) <= 1e-6


def _replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_q2_unknown_solver_and_refit_failures(monkeypatch):
    T, X, uv, sg = _scene_90_2m()
    T_r, cons = _ransac_pose(X, uv)
    with pytest.raises(ValueError):
        mt.refit_consensus(K, X[cons], uv[cons], T_r, "p3p")
    with pytest.raises(ValueError):
        mt.relocalize_with_solver(K, X, uv, sg, mt.MapTrackerParams(), mt.MapRelocParams(), "p3p")
    import npnp
    real = npnp.solve
    # an exception inside the refit is a refit failure, with no solve record
    monkeypatch.setattr(npnp, "solve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    Tf, info = mt.refit_consensus(K, X[cons], uv[cons], T_r, "newton")
    assert Tf is None and info["npnp_converged"] == -1 and info["npnp_outer"] == -1 and info["refit_ms"] >= 0
    # a solve that does not converge is a refit failure, its record kept
    monkeypatch.setattr(npnp, "solve", lambda P, V: _replace(real(P, V), converged=False))
    Tf, info = mt.refit_consensus(K, X[cons], uv[cons], T_r, "newton")
    assert Tf is None and info["npnp_converged"] == 0 and np.isfinite(info["npnp_gamma"]) and info["npnp_newton"] >= 0
    monkeypatch.setattr(npnp, "solve", real)
    # a non-finite pose is a failure
    Tn = T_r.copy()
    Tn[0, 3] = np.nan
    assert mt.refit_consensus(K, X[cons], uv[cons], Tn, "epnp")[0] is None


def test_q2_npnp_not_imported_at_module_level():
    import ast
    tree = ast.parse(Path(mt.__file__).read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module)
    assert "npnp" not in top and "tracemalloc" not in top


# --------------------------------------------------------------------------- Q3
def test_q3_behind_count():
    pts = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1e-9]])
    assert mt.behind_count(np.eye(4), pts) == 2
    T = np.eye(4)
    T[2, 3] = -1.0                                       # every depth minus 1: z = 1 counts now
    assert mt.behind_count(T, pts) == 4
    T = np.eye(4)
    T[:3, :3] = so3_exp([0.0, math.pi, 0.0])             # half turn about y: z -> -z
    assert mt.behind_count(T, pts) == 3


# --------------------------------------------------------------------------- Q4
@pytest.mark.parametrize("noise", [False, True])
def test_q4_epnp_bitwise_and_consensus_shared(noise):
    T, X, uv, sg = _scene_90_2m(noise=noise)
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    Ta, ma, ia = mt.relocalize_from_correspondences(K, X, uv, sg, P, RP)
    Tb, mb, ib = mt.relocalize_with_solver(K, X, uv, sg, P, RP, "epnp")
    assert ib["status"] == "ok" and Ta.tobytes() == Tb.tobytes() and ma.tobytes() == mb.tobytes()
    assert ib["ransac_inliers"] == ia["ransac_inliers"] and ib["n_inliers"] == ia["n_inliers"]
    infos = [mt.relocalize_with_solver(K, X, uv, sg, P, RP, s)[2] for s in SOLVERS]
    for inf in infos:
        assert inf["status"] == "ok" and np.array_equal(inf["consensus"], infos[0]["consensus"])
        assert inf["ransac_inliers"] == int(inf["consensus"].sum()) == ia["ransac_inliers"]
        assert inf["behind"] == 0 and inf["refit_pose"] is not None
    assert infos[2]["npnp_converged"] == 1 and infos[0]["npnp_converged"] == -1
    for s in ("sqpnp", "newton"):
        Ts, _m, _i = mt.relocalize_with_solver(K, X, uv, sg, P, RP, s)
        assert math.degrees(_rot_err(Ts[:3, :3], T[:3, :3])) <= (0.01 if noise else 1e-6)


def test_q4_cheirality_rule(monkeypatch):
    T, X, uv, sg = _scene_90_2m()
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    zc = np.sort((X @ T[:3, :3].T + T[:3, 3])[:, 2])
    got = {}
    for nb in (100, 101, 200):
        Ts = T.copy()
        Ts[2, 3] -= 0.5 * (zc[nb - 1] + zc[nb]) if nb < 200 else zc[-1] + 1.0
        monkeypatch.setattr(mt, "refit_consensus", lambda *a, _T=Ts, **k: (_T.copy(), dict(refit_ms=0.0)))
        got[nb] = mt.relocalize_with_solver(K, X, uv, sg, P, RP, "sqpnp")
    # exactly half behind is not rejected
    assert got[100][2]["status"] == "ok" and got[100][2]["behind"] == 100 and got[100][0] is not None
    for nb in (101, 200):
        Tc, mc, ic = got[nb]
        assert Tc is None and not mc.any() and ic["status"] == "cheirality" and ic["behind"] == nb
        assert ic["ransac_inliers"] == 200 and ic["refit_pose"] is not None and ic["n_inliers"] == 0


def test_q4_refit_status(monkeypatch):
    T, X, uv, sg = _scene_90_2m()
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    for stub in (lambda *a, **k: (None, dict(refit_ms=0.0)),
                 lambda *a, **k: (_ for _ in ()).throw(RuntimeError("refit boom"))):
        monkeypatch.setattr(mt, "refit_consensus", stub)
        Tc, mc, ic = mt.relocalize_with_solver(K, X, uv, sg, P, RP, "newton")
        assert Tc is None and mc.shape == (200,) and not mc.any() and ic["status"] == "refit"
        assert ic["ransac_inliers"] == 200 and ic["consensus"].sum() == 200 and ic["behind"] == -1
        assert ic["refit_pose"] is None and ic["refit_ms"] >= 0.0


def test_q4_too_few_and_ransac():
    rng = np.random.default_rng(22)
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    T = _pose_90_2m([0.0, 1.0, 0.0])
    X, uv, sg = _scene(rng, T, 200)
    for s in SOLVERS:
        Tf, maskf, infof = mt.relocalize_with_solver(K, X[:3], uv[:3], sg[:3], P, RP, s)
        assert Tf is None and maskf.shape == (3,) and not maskf.any()
        assert infof["status"] == "too_few" and infof["ransac_inliers"] == -1 and infof["refit_ms"] == -1.0
        uvr = np.stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)], 1)
        Tr, maskr, infor = mt.relocalize_with_solver(K, X, uvr, sg, P, RP, s)
        assert Tr is None and maskr.shape == (200,) and not maskr.any() and infor["status"] == "ransac"
        assert not infor["consensus"].any() and infor["refit_pose"] is None and infor["behind"] == -1
        assert infor["npnp_converged"] == -1 and infor["refit_ms"] == -1.0


# --------------------------------------------------------------------------- Q5, Q7, Q8 (real attempts)
def _lines_with(out, tag):
    return [i for i, l in enumerate(out) if tag in l]


def _check_real_attempts(out, mode, solver):
    """One shadow=0 attempt line right before each [MAP-RELOC] line, fields and mapping as D / C4; the
    D2 group ([RELOC-SET] when a consensus set exists, [RELOC-SOLVE]) contiguous right before it."""
    idx = _lines_with(out, "[MAP-RELOC]")
    atts = _lines_with(out, "[MAP-RELOC-ATTEMPT]")
    assert len(idx) == len(atts) >= 1
    res = []
    for i in idx:
        r = RX_RELOC.match(out[i])
        a = _att(out[i - 1])
        assert a["shadow"] == "0" and a["solver"] == solver and a["mode"] == mode and a["frame"] == r.group(1)
        assert a["matches"] == r.group(5) and a["ransac_inliers"] == r.group(6)
        assert (a["ba1"] if a["outcome"] == "ba_inliers" else a["inliers"]) == r.group(7)
        assert OUT2REASON[a["outcome"]] == r.group(4)
        assert (a["outcome"] == "ok") == (r.group(3) == "ok") == (int(a["inliers"]) >= 30)
        s = RX_SOLVE.match(out[i - 2])
        assert s is not None and s.group(1) == a["frame"] and s.group(2) == "0" and s.group(3) == solver
        assert s.group(9) == "-1"
        has_set = a["outcome"] in ("refit", "cheirality", "ba_inliers", "inliers", "ok")
        st = RX_SET.match(out[i - 3])
        assert (st is not None) == has_set
        if has_set:
            assert st.group(1) == a["frame"] and st.group(2) == mode and st.group(3) == "0"
            assert int(st.group(4)) == int(a["ransac_inliers"])
        res.append((a, s, st))
    return res


@pytest.mark.parametrize("solver", SOLVERS)
def test_q5_real_attempt_line_m1b(no_twoview, solver):
    frames, X, T_true, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    tr.reloc_solver = solver
    states, out = _run_lines(tr, frames, wmap, 4)
    assert states == ["lost", "tracked", "tracked", "tracked"]
    (a, s, st), = _check_real_attempts(out, "m1b", solver)
    assert a["frame"] == "2" and a["outcome"] == "ok" and a["matches"] == "60" and a["inliers"] == "60"
    assert a["ba1"] == "60" and a["guided"] == "-1" and a["behind"] == "0" and float(a["refit_ms"]) >= 0
    assert 0.0 < float(a["hull"]) <= 1.0
    assert (a["conv"] == "1") == (solver == "newton") and (a["conv"] == "-1") == (solver != "newton")
    assert np.allclose(frames[2].cam_T_world, T_true[2], atol=1e-6)
    assert np.allclose(_pose_of(a["pose"]), frames[2].cam_T_world, atol=1e-12)
    assert np.allclose(_pose_of(a["refit_pose"]), T_true[2], atol=1e-6)
    assert not any("shadow=1" in l for l in out)


def test_q5_real_failures_map_to_the_reasons(no_twoview, monkeypatch):
    # too_few
    frames, X, _T, corr = _m1b_lost_at_1(frame2_lines=3)
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    _s, out = _run_lines(tr, frames, wmap, 2)
    (a, _so, st), = _check_real_attempts(out, "m1b", "epnp")
    assert a["outcome"] == "too_few" and a["ransac_inliers"] == "-1" and a["hull"] == "-1.0" and st is None
    assert a["refit_pose"] == "none" and a["pose"] == "none" and a["refit_ms"] == "-1.0"
    # cheirality: a refit pose with every consensus point behind is a failed attempt (reason pnp)
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    Tb = np.eye(4)
    Tb[2, 3] = -100.0
    monkeypatch.setattr(mt, "refit_consensus", lambda *a, **k: (Tb.copy(), dict(refit_ms=0.5)))
    states, out = _run_lines(tr, frames, wmap, 2)
    assert states == ["lost", "lost"] and frames[2].tracking_state == "LOST"
    (a, _so, st), = _check_real_attempts(out, "m1b", "epnp")
    assert a["outcome"] == "cheirality" and int(a["behind"]) == int(a["ransac_inliers"]) == 60 and st is not None
    assert a["pose"] == "none" and a["ba1"] == "-1" and _pose_of(a["refit_pose"])[2, 3] == -100.0
    # refit
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    monkeypatch.setattr(mt, "refit_consensus", lambda *a, **k: (None, dict(refit_ms=0.5)))
    states, out = _run_lines(tr, frames, wmap, 2)
    (a, _so, st), = _check_real_attempts(out, "m1b", "epnp")
    assert states == ["lost", "lost"] and a["outcome"] == "refit" and a["refit_pose"] == "none" and st is not None


def test_q5_real_error_attempt_line(no_twoview, monkeypatch):
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    monkeypatch.setattr(mt, "relocalize_with_solver", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    _s, out = _run_lines(tr, frames, wmap, 2)
    i = _lines_with(out, "[MAP-RELOC]")[0]
    assert "[MAP-TRACK-ERROR] frame=2" in out[i - 3]
    a = _att(out[i - 1])
    assert a["outcome"] == "error" and a["matches"] == "60" and a["ransac_inliers"] == "-1" and a["inliers"] == "0"
    assert a["pose"] == "none" and a["refit_pose"] == "none" and a["hull"] == "-1.0"
    assert RX_SOLVE.match(out[i - 2]).group(1) == "2"


def test_q5_m1a_ba_inliers_outcome(no_twoview, monkeypatch):
    """A first BA under ransac_min_inliers: outcome ba_inliers, [MAP-RELOC] reason inliers with the
    first BA's count, no guided search and no pose."""
    frames, X, _T = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.release(0, frames, wmap)
    frames[1].tracking_state = "LOST"
    tr.reloc = "pnp"
    real = mt.motion_only_ba

    def few(K_, T0, pts, uv, sg, params):
        T, mask, info = real(K_, T0, pts, uv, sg, params)
        mask = mask.copy()
        mask[np.flatnonzero(mask)[5:]] = False
        return T, mask, info
    monkeypatch.setattr(mt, "motion_only_ba", few)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr.on_frame(2, frames, wmap) == "lost"
    out = buf.getvalue().splitlines()
    (a, _s, st), = _check_real_attempts(out, "m1a", "epnp")
    assert a["outcome"] == "ba_inliers" and a["ba1"] == "5" and a["inliers"] == "5" and a["guided"] == "-1"
    assert a["pose"] == "none" and a["refit_pose"] != "none" and st is not None


def test_q5_m1a_real_attempt_guided(no_twoview, monkeypatch):
    frames, X, T_true = _world()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    tr.release(0, frames, wmap)
    frames[1].tracking_state = "LOST"
    tr.reloc = "pnp"
    tr.reloc_solver = "newton"
    real = mt.global_match
    monkeypatch.setattr(mt, "global_match", lambda *a, **k: real(*a, **k)[:20])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr.on_frame(2, frames, wmap) == "tracked"
    out = buf.getvalue().splitlines()
    (a, _s, _st), = _check_real_attempts(out, "m1a", "newton")
    n_visible0 = sum(1 for pid in frames[2].pid_of_kp if pid < 100)
    assert a["outcome"] == "ok" and a["matches"] == "20" and a["ba1"] == "20"
    assert int(a["guided"]) == n_visible0 - 20 and int(a["inliers"]) == n_visible0
    assert np.allclose(frames[2].cam_T_world, T_true[2], atol=1e-6)


def test_q7_attempt_line_format():
    tr = mt.MapTracker(mt.M1Map(header={}, points=[], corr={}), "m1b")
    a = tr._new_attempt(12, 1, "newton", 44)
    line = tr._attempt_line(a)
    assert line == ("  >> [MAP-RELOC-ATTEMPT] frame=12 mode=m1b shadow=1 solver=newton outcome=error matches=44 "
                    "ransac_inliers=-1 behind=-1 ba1_inliers=-1 guided=-1 inliers=0 hull=-1.0 refit_ms=-1.0 "
                    "npnp_converged=-1 npnp_gap_rel=nan refit_pose=none pose=none")
    T = _pose_90_2m([0.3, -0.8, 0.5])
    T[0, 3] = -0.0
    a.update(outcome="ok", ransac_inliers=40, behind=0, ba1_inliers=38, inliers=38, hull=0.123456789012345,
             refit_ms=3.25, npnp_converged=1, npnp_gap_rel=1.5e-9, refit_pose=T, pose=T)
    d = _att(tr._attempt_line(a))
    assert d["outcome"] == "ok" and float(d["hull"]) == 0.123456789012345 and float(d["gap"]) == 1.5e-9
    assert _pose_of(d["pose"]).tobytes() == T.tobytes() and _pose_of(d["refit_pose"]).tobytes() == T.tobytes()
    assert "[MAP-RELOC]" not in line and "[MAP-TRACK" not in line
    # the D2 lines
    a["set"] = (np.array([[1.0, 2.0, 3.0], [0.1, -0.2, 1e-17]]), np.array([[10.5, 20.25], [300.0, 400.0]]))
    st = RX_SET.match(tr._set_line(a))
    assert st is not None and st.group(4) == "2" and st.group(5) == "1.0,2.0,3.0,0.1,-0.2,1e-17"
    so = RX_SOLVE.match(tr._solve_line(a))
    assert so is not None and so.group(4) == "nan" and so.group(9) == "-1"
    for l in (tr._set_line(a), tr._solve_line(a)):
        assert "[MAP-RELOC" not in l and "[MAP-TRACK" not in l


def test_q8_d2_lines_real_attempt_m1b(no_twoview, monkeypatch):
    import npnp
    frames, X, _T, corr = _m1b_lost_at_1()
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    tr.reloc_solver = "newton"
    seen = {}
    real = mt.relocalize_with_solver

    def spy(*a):
        seen["args"] = a
        return real(*a)
    monkeypatch.setattr(mt, "relocalize_with_solver", spy)
    _s, out = _run_lines(tr, frames, wmap, 2)
    (a, so, st), = _check_real_attempts(out, "m1b", "newton")
    n = int(st.group(4))
    Pp = np.array([float(x) for x in st.group(5).split(",")]).reshape(n, 3)
    Up = np.array([float(x) for x in st.group(6).split(",")]).reshape(n, 2)
    # the consensus rows, bitwise, in ascending input order
    Kc, pts, uv = seen["args"][:3]
    pts, uv = np.asarray(pts, dtype=np.float64), np.asarray(uv, dtype=np.float64)
    _nr, cons, _Tr = mt._ransac_consensus(Kc, pts, uv, mt.MapRelocParams())
    assert Pp.tobytes() == pts[cons].tobytes() and Up.tobytes() == uv[cons].tobytes()
    V = np.column_stack([Up, np.ones(n)]) @ np.linalg.inv(K).T
    r = npnp.solve(Pp, V)
    Mf = float(np.linalg.norm(npnp.rotation_cost_matrix(Pp, V)))
    assert abs(float(so.group(4)) - r.gamma) <= 1e-12 * Mf and abs(float(so.group(5)) - r.cost) <= 1e-12 * Mf
    assert int(so.group(6)) == r.outer_iters and int(so.group(7)) == r.newton_iters and int(so.group(8)) >= 0
    assert abs(float(a["gap"]) - r.gap / Mf) <= 1e-9 * abs(r.gap / Mf) + 1e-15
    # the logged hull and behind reproduce from the set
    assert float(a["hull"]) == mt._hull_share(Up, W, H)
    assert int(a["behind"]) == mt.behind_count(_pose_of(a["refit_pose"]), Pp)


def test_q8_refit_info_record():
    _T, X, uv, sg = _scene_90_2m(noise=True)
    P, RP = mt.MapTrackerParams(), mt.MapRelocParams()
    info = mt.relocalize_with_solver(K, X, uv, sg, P, RP, "newton")[2]
    assert info["npnp_converged"] == 1 and np.isfinite(info["npnp_gamma"]) and np.isfinite(info["npnp_cost"])
    assert min(info["npnp_outer"], info["npnp_newton"], info["npnp_ls_evals"]) >= 0
    for s in ("epnp", "sqpnp"):
        info = mt.relocalize_with_solver(K, X, uv, sg, P, RP, s)[2]
        assert math.isnan(info["npnp_gamma"]) and math.isnan(info["npnp_cost"]) and math.isnan(info["npnp_gap_rel"])
        assert (info["npnp_converged"], info["npnp_outer"], info["npnp_newton"], info["npnp_ls_evals"]) == (-1, -1, -1, -1)


# --------------------------------------------------------------------------- Q6, Q8 (the shadow)
def _long_world(mode, n_frames=12, lost_at=None):
    frames, X, T_true = _world(n_frames=n_frames, step=0.005)
    corr = None
    if mode == "m1b":
        corr = {k: _c_lines(frames[k], 0, 200 if k > 2 else 100, 60) for k in range(1, n_frames)}
        if lost_at is not None:
            corr[lost_at] = corr[lost_at][:10]
    return frames, X, T_true, corr


RX_MS = re.compile(r"refit_ms=\S+")


def _no_shadow_lines(out):
    """The lines of a run without the shadow lines, the wall-time field masked (it is a timing)."""
    return [RX_MS.sub("refit_ms=T", l) for l in out if "shadow=1" not in l]


def _snapshot(frames, wmap):
    idx = {p.id: i for i, p in enumerate(wmap.points)}
    fr = [(f.cam_T_world.tobytes(), f.pose_valid, f.tracking_state,
           sorted((kp, idx.get(pid, -1)) for kp, pid in f.keypoint_to_point.items())) for f in frames]
    pts = [(np.asarray(p.point, dtype=np.float64).tobytes(), sorted(p.observations.items())) for p in wmap.points]
    return fr, pts


def _shadow_run(mode, shadow, monkeypatch, lost_at=1, solver="epnp", n_frames=12):
    frames, X, T_true, corr = _long_world(mode, n_frames=n_frames, lost_at=lost_at if mode == "m1b" else None)
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), mode, None)
    tr.reloc = "pnp"
    tr.reloc_solver = solver
    tr.reloc_shadow = shadow
    if mode == "m1a" and lost_at is not None:
        real = tr._track_m1a

        def forced(k, fr, T):
            if k == lost_at:
                raise RuntimeError("forced")
            return real(k, fr, T)
        monkeypatch.setattr(tr, "_track_m1a", forced)
    states, out = _run_lines(tr, frames, wmap, n_frames - 1)
    return states, out, _snapshot(frames, wmap), T_true


@pytest.mark.parametrize("mode", ["m1a", "m1b"])
def test_q6_shadow_lines_and_no_leak(no_twoview, monkeypatch, mode):
    s0, out0, snap0, _T = _shadow_run(mode, False, monkeypatch)
    s1, out1, snap1, T_true = _shadow_run(mode, True, monkeypatch)
    assert s0 == s1 and s0[0] == "lost" and s0.count("tracked") == len(s0) - 1
    assert snap0 == snap1
    # the shadow run is the no-shadow run with shadow lines inserted, nothing else
    assert _no_shadow_lines(out1) == _no_shadow_lines(out0)
    assert not any("shadow=1" in l for l in out0)
    # after each tracked frame's [MAP-TRACK] line: its [RELOC-SET], then [RELOC-SOLVE] / attempt pairs for
    # epnp, sqpnp, newton; nothing after a lost frame's
    n_tracked = 0
    for i, l in enumerate(out1):
        t = RX_TRACK.search(l) if "[MAP-TRACK]" in l else None
        if t is None:
            continue
        nxt = out1[i + 1:i + 8]
        if t.group(2) == "lost":
            assert not nxt or "shadow=1" not in nxt[0]
            continue
        n_tracked += 1
        st = RX_SET.match(nxt[0])
        assert st is not None and st.group(1) == t.group(1) and st.group(3) == "1" and st.group(2) == mode
        atts = []
        for j, s in enumerate(SOLVERS):
            so = RX_SOLVE.match(nxt[1 + 2 * j])
            a = _att(nxt[2 + 2 * j])
            assert so.group(1) == a["frame"] == t.group(1) and so.group(2) == a["shadow"] == "1"
            assert so.group(3) == a["solver"] == s and a["outcome"] == "ok"
            assert int(st.group(4)) == int(a["ransac_inliers"])
            atts.append(a)
        assert len({(a["matches"], a["ransac_inliers"], a["hull"]) for a in atts}) == 1
        assert len(nxt) == 7 or "shadow=1" not in nxt[7]
        for a in atts:                                   # every solver finds the frame's true pose
            assert np.allclose(_pose_of(a["pose"]), T_true[int(t.group(1))], atol=1e-6)
    assert n_tracked == len(s1) - 1


@pytest.mark.parametrize("mode", ["m1a", "m1b"])
def test_q8_shadow_d2_and_peak(no_twoview, monkeypatch, mode):
    _s, out, _snap, _T = _shadow_run(mode, True, monkeypatch)
    sets = [RX_SET.match(l) for l in out if "[RELOC-SET]" in l and "shadow=1" in l]
    tracked = [RX_TRACK.search(l).group(1) for l in out if "[MAP-TRACK]" in l and "state=tracked" in l]
    assert [g.group(1) for g in sets] == tracked
    peaks = {}
    for l in out:
        so = RX_SOLVE.match(l)
        if so is None:
            continue
        k, sh, peak = int(so.group(1)), so.group(2), int(so.group(9))
        if sh == "0" or k % 10 != 0:
            assert peak == -1
        else:
            peaks[(k, so.group(3))] = peak
        if so.group(3) == "newton":
            assert np.isfinite(float(so.group(4))) and int(so.group(7)) >= 0
        else:
            assert so.group(4) == so.group(5) == "nan" and so.group(6) == so.group(7) == so.group(8) == "-1"
    assert sorted(peaks) == [(10, s) for s in sorted(SOLVERS)] and all(v > 0 for v in peaks.values())


def test_q6_shadow_exception_prints_error_and_changes_nothing(no_twoview, monkeypatch):
    s0, out0, snap0, _T = _shadow_run("m1b", False, monkeypatch, lost_at=None, n_frames=6)
    # one solver raising after the shared RANSAC step
    real = mt._solve_from_consensus

    def boom(*a, **k):
        if a[7] == "sqpnp":
            raise RuntimeError("shadow boom")
        return real(*a, **k)
    monkeypatch.setattr(mt, "_solve_from_consensus", boom)
    s1, out1, snap1, _T = _shadow_run("m1b", True, monkeypatch, lost_at=None, n_frames=6)
    assert s0 == s1 and snap0 == snap1 and _no_shadow_lines(out1) == _no_shadow_lines(out0)
    atts = [_att(l) for l in out1 if "[MAP-RELOC-ATTEMPT]" in l]
    assert len(atts) == 3 * 5
    assert all((a["outcome"] == "error") == (a["solver"] == "sqpnp") for a in atts)
    assert all(a["pose"] == "none" for a in atts if a["solver"] == "sqpnp")
    # the shared step raising: every solver's line prints error, no set
    monkeypatch.setattr(mt, "_solve_from_consensus", real)
    monkeypatch.setattr(mt, "_ransac_consensus", lambda *a: (_ for _ in ()).throw(RuntimeError("ransac boom")))
    s2, out2, snap2, _T = _shadow_run("m1b", True, monkeypatch, lost_at=None, n_frames=6)
    assert s0 == s2 and snap0 == snap2 and _no_shadow_lines(out2) == _no_shadow_lines(out0)
    atts = [_att(l) for l in out2 if "[MAP-RELOC-ATTEMPT]" in l]
    assert len(atts) == 15 and all(a["outcome"] == "error" and a["matches"] == "60" for a in atts)
    assert not any("[RELOC-SET]" in l for l in out2)


def test_q6_no_shadow_without_pnp(no_twoview):
    frames, X, _T, corr = _long_world("m1b", n_frames=4)
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc_shadow = True                 # reloc stays none: the shadow needs --map_reloc pnp
    _s, out = _run_lines(tr, frames, wmap, 3)
    assert not any("[MAP-RELOC-ATTEMPT]" in l or "[RELOC-" in l for l in out)
