"""Stage 04u: the opening driver (opening.py), its flags (main.py) and the pipeline hook.

O1 the flags, defaults and refusals (and the 04r refusals under --init twoview); O2 the
run_pipeline hand-off; O3 refusals leave the 04r rotation run unchanged and the pair lines name the
candidates of D3 (2); O4 an accepted result: frame b's pose, the world points, the lines; O5 the
survival check and the false start; O6 the refit arms (shadow, off, live); O7 the best-pair choice;
O8 the no-leak tokens and imports; O9 the renewal of the candidate list (R43-1).
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

import initializer  # noqa: E402
import local_mapping  # noqa: E402
import main  # noqa: E402
import map_tracker  # noqa: E402
import opening  # noqa: E402
import pipeline  # noqa: E402
import rotation_only as ro  # noqa: E402
from frame import DEFAULT_K  # noqa: E402
from Map import Map  # noqa: E402

K = np.asarray(DEFAULT_K, dtype=np.float64)
W, H = 640, 480

ROT_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
            "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba", "--no_point_fusion"]
M1_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "map", "--m1_map", "/tmp/m1.txt",
           "--m1_mode", "m1a", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba",
           "--no_point_fusion"]

P = r"^\s*>>\s+"
RX_KF = re.compile(P + r"\[ROT-KF\] frame=(\d+) prev=(-?\d+) n=(\d+)\s*$")
RX_ONLY = re.compile(P + r"\[ROT-ONLY\] frame=(\d+) state=(measured|unsupported) kf=(\d+) .*$")
RX_PAIR = re.compile(P + r"\[INIT-PAIR\] a=(\d+) b=(\d+) status=(\S+) reason=(\S+) ")
RX_SHADOW = re.compile(P + r"\[INIT-PAIR-SHADOW\] a=(\d+) b=(\d+) status=(\S+) reason=(\S+) ")
RX_POSE = re.compile(P + r"\[INIT-POSE\] a=(\d+) b=(\d+) arm=(live|shadow) R=(\S+) t=(\S+)$")
RX_ACC = re.compile(P + r"\[INIT-ACCEPT\] map=(\d+) a=(\d+) b=(\d+) published=(\d+) model=(\S+)$")
RX_PT = re.compile(P + r"\[INIT-POINT\] map=(\d+) i=(\d+) kp_a=(\d+) kp_b=(\d+) ua=(\S+) va=(\S+) ub=(\S+) vb=(\S+) "
                   r"x=(\S+) y=(\S+) z=(\S+) par_deg=(\S+) sig_a=(\S+) sig_b=(\S+)$")
RX_RESET = re.compile(P + r"\[INIT-RESET\] map=(\d+) a=(\d+) b=(\d+) frame=(\d+) tracked=(\d+) removed=(\d+)$")
RX_SURV = re.compile(P + r"\[INIT-SURVIVED\] map=(\d+) a=(\d+) b=(\d+) frame=(\d+)$")
RX_CAND = re.compile(P + r"\[INIT-CAND\] frame=(\d+) reason=(\S+)$")
RX_ERR = re.compile(P + r"\[INIT-ERROR\] a=(\d+) b=(\d+) arm=(live|shadow) error=(.*)$")
RX_REL = re.compile(P + r"\[M1-RELEASE\] frame=(\d+) points=(\d+) shadowed=(\d+) refused=(\d+)$")


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# --------------------------------------------------------------------------- O1
def _refused(argv):
    args = main.get_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            main.validate_tracking_args(args)
        return None
    except SystemExit as e:
        return str(e)


def test_o1_flags_and_defaults():
    pr = main.get_parser()
    a = pr.parse_args(["--dataset", "rgb"])
    assert a.init == "none" and a.init_refit == "off"
    for v in ("off", "shadow", "live"):
        assert pr.parse_args(ROT_ARGV + ["--init", "twoview", "--init_refit", v]).init_refit == v
    for bad in (["--init", "sfm"], ["--init_refit", "on"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            pr.parse_args(["--dataset", "rgb"] + bad)
    helps = {act.dest: act.help for act in pr._actions}
    assert "04u" in helps["init"]
    assert "04u" in helps["init_refit"] and "R41-1" in helps["init_refit"]


def test_o1_refusals():
    for extra in ([], ["--init", "none"], ["--init", "twoview"], ["--init", "twoview", "--init_refit", "off"],
                  ["--init", "twoview", "--init_refit", "shadow"], ["--init", "twoview", "--init_refit", "live"]):
        assert _refused(ROT_ARGV + extra) is None, extra
    assert _refused(["--dataset", "rgb", "--init", "twoview"]) == "--init twoview is only valid with --tracking rotation"
    assert _refused(M1_ARGV + ["--init", "twoview"]) == "--init twoview is only valid with --tracking rotation"
    for v in ("shadow", "live"):
        assert _refused(ROT_ARGV + ["--init_refit", v]) == f"--init_refit {v} is only valid with --init twoview"
        assert _refused(["--dataset", "rgb", "--init_refit", v]) == f"--init_refit {v} is only valid with --init twoview"
        assert _refused(ROT_ARGV + ["--init", "none", "--init_refit", v]) is not None
    # the existing rules stand
    assert _refused(["--dataset", "rgb"]) is None and _refused(M1_ARGV) is None


@pytest.mark.parametrize("flag", ["--default_k", "--undistort", "--no_loop_closure", "--no_relocalization",
                                  "--no_ba", "--no_point_fusion"])
def test_o1_04r_requirements_hold_with_init_twoview(flag):
    assert _refused([x for x in ROT_ARGV if x != flag] + ["--init", "twoview"]) == f"--tracking rotation requires {flag}"


@pytest.mark.parametrize("extra", [["--map_reloc", "pnp"], ["--m1_map", "/tmp/m1.txt"], ["--reloc_shadow"],
                                   ["--reloc_solver", "sqpnp"], ["--oracle", "o3"], ["--point_source", "depth_filter"],
                                   ["--epi_refine"], ["--lm_confirm"]])
def test_o1_04r_refusals_hold_with_init_twoview(extra):
    msg = _refused(ROT_ARGV + ["--init", "twoview"] + extra)
    assert msg is not None and msg.startswith("--tracking rotation refuses")


def test_o1_tracker_interface():
    assert [f for f in opening.OpeningParams.__dataclass_fields__] == ["window", "survival_frames"]
    p = opening.OpeningParams()
    assert (p.window, p.survival_frames) == (2, 10)
    with pytest.raises(ValueError):
        opening.OpeningTracker(None, refit="on")
    tr = opening.OpeningTracker()
    assert tr.world_map is None and tr.use_homography is False and tr.refit == "off"
    for v in ("off", "shadow", "live"):
        assert opening.OpeningTracker(None, refit=v).setup_line() == \
            f"  >> [INIT-SETUP] window=2 survival_frames=10 refit={v}"
    s = inspect.signature(pipeline.show_trajectory).parameters
    assert "open_tracker" in s and s["open_tracker"].default is None


# --------------------------------------------------------------------------- O2
def _capture_run_pipeline(monkeypatch, argv):
    seen = {}

    def fake_show(frame_iter, **kw):
        seen.update(kw)

    monkeypatch.setattr(main, "show_trajectory", fake_show)
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (K, 0, iter(())))
    args = main.get_parser().parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(args)
    return seen


@pytest.mark.parametrize("refit", ["off", "shadow", "live"])
def test_o2_run_pipeline_passes_the_opening_tracker(monkeypatch, refit):
    kw = _capture_run_pipeline(monkeypatch, ROT_ARGV + ["--init", "twoview", "--init_refit", refit])
    assert isinstance(kw["open_tracker"], opening.OpeningTracker)
    assert kw["open_tracker"].refit == refit
    assert kw["rot_tracker"] is None and kw["m1_tracker"] is None


def test_o2_run_pipeline_without_twoview(monkeypatch):
    kw = _capture_run_pipeline(monkeypatch, ROT_ARGV)
    assert kw["open_tracker"] is None and isinstance(kw["rot_tracker"], ro.RotationOnlyTracker)
    kw = _capture_run_pipeline(monkeypatch, ROT_ARGV + ["--init", "none"])
    assert kw["open_tracker"] is None and isinstance(kw["rot_tracker"], ro.RotationOnlyTracker)
    kw = _capture_run_pipeline(monkeypatch, ["--dataset", "rgb"])
    assert kw["open_tracker"] is None and kw["rot_tracker"] is None


def test_o2_namespace_without_the_new_flags(monkeypatch):
    """Existing tests build namespaces that lack --init and --init_refit (getattr defaults)."""
    args = main.get_parser().parse_args(ROT_ARGV)
    del args.init
    del args.init_refit
    with contextlib.redirect_stdout(io.StringIO()):
        main.validate_tracking_args(args)


# --------------------------------------------------------------------------- the synthetic scene
class _Frame:
    """Frame-like stub as in tests/test_rotation_only.py, with the link table the map needs."""

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
        self.keypoint_to_point = {}
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


def _yaw_scene(n_frames, seed=0, n_dirs=3000):
    rng = np.random.default_rng(seed)
    az = np.radians(rng.uniform(-50.0, 100.0, n_dirs))
    el = np.radians(rng.uniform(-55.0, 55.0, n_dirs))
    dirs = np.column_stack((np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)))
    desc = rng.uniform(0.0, 255.0, (n_dirs, 128)).astype(np.float32)
    truth = [rot_y(float(k)).T for k in range(n_frames)]
    return [_Frame(k, truth[k], dirs, desc) for k in range(n_frames)], truth


def _plain_run(frames):
    tr = ro.RotationOnlyTracker()
    tr.world_map = Map()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(tr.setup_line())
        tr.start(frames)
        for k in range(1, len(frames)):
            tr.on_frame(k, frames)
    return [f.cam_T_world.copy() for f in frames], buf.getvalue().splitlines()


def _open_run(frames, refit="off", n=None):
    tr = opening.OpeningTracker(None, refit=refit)
    tr.world_map = Map(frames=frames)
    buf = io.StringIO()
    states = {}
    with contextlib.redirect_stdout(buf):
        print(tr.setup_line())
        tr.start(frames)
        for k in range(1, len(frames) if n is None else n):
            states[k] = tr.on_frame(k, frames)
        print(tr.summary_line())
    return tr, states, buf.getvalue().splitlines()


def _refusal(reason, n_matches=0):
    st = initializer._new_state()
    st["n_matches"] = n_matches
    return initializer._refused(reason, st)


def _accepted(frames, a, b, n, R=None, t=None, seed=1):
    """An accepted InitResult on keypoints 0..n-1 of frames a and b (points in camera a)."""
    rng = np.random.default_rng(seed)
    R = rot_x(3.0) @ rot_y(-2.0) if R is None else R
    t = np.array([0.1, -0.2, 0.05]) if t is None else np.asarray(t, dtype=np.float64)
    pts = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.5, 2.0, n)))
    idx = np.arange(n, dtype=np.int64)
    return initializer.InitResult(
        status="accepted", reason="none", model="F", rh=0.2, score_h=1.0, score_f=2.0, n_matches=n + 10,
        n_inliers=n + 5, good=(n, 1, 0, 0), chosen=0, parallax51_deg=2.5, n_triangulated=n, R=R, t=t,
        points=pts, match_idx=idx.copy(), kp_a=idx.copy(), kp_b=idx[::-1].copy(),
        parallax_deg=rng.uniform(1.2, 5.0, n), rows=3, cols=4, cells=12, ba_cost_initial=1.0,
        ba_cost_final=0.5, ba_iterations=3)


def _pair_lines(lines, rx=RX_PAIR):
    return [(int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)) for m in (rx.match(l) for l in lines) if m]


def _rot_lines(lines):
    return [l for l in lines if RX_ONLY.match(l) or RX_KF.match(l) or "[ROT-SETUP]" in l]


def _expected_candidates(plain_lines, n):
    """D3 (2) without renewals: the last two keyframes below k."""
    kfs = []
    out = {}
    for l in plain_lines:
        m = RX_ONLY.match(l)
        if m:
            k = int(m.group(1))
            pass
        m2 = RX_KF.match(l)
        if m2:
            kfs.append(int(m2.group(1)))
            continue
        if m:
            out[k] = None
    # rebuild per frame: keyframe k is known from frame k on, used as a candidate from frame k+1
    for k in range(1, n):
        out[k] = [a for a in kfs if a < k][-2:]
    return out


# --------------------------------------------------------------------------- O3
def test_o3_refusals_leave_the_rotation_run_unchanged(monkeypatch):
    calls = []

    def refuse(fa, fb, params=None, refit=False):
        calls.append((fa.id, fb.id, refit))
        return _refusal("parallax", 120)

    monkeypatch.setattr(initializer, "initialize_frames", refuse)
    n = 60                                   # 1 deg of yaw per frame: keyframes about every 23 frames
    frames0, _truth = _yaw_scene(n)
    poses_plain, plain = _plain_run(frames0)
    frames, _truth = _yaw_scene(n)
    tr, states, lines = _open_run(frames)
    assert len(frames) == len(poses_plain)
    assert all(np.array_equal(f.cam_T_world, T) for f, T in zip(frames, poses_plain))
    assert _rot_lines(lines) == _rot_lines(plain)
    kfs = [int(RX_KF.match(l).group(1)) for l in plain if RX_KF.match(l)]
    assert len(kfs) >= 3                     # at least two switches, so the window slides past keyframe 0
    want = []
    for k in range(1, n):
        for a in [x for x in kfs if x < k][-2:]:
            want.append((a, k))
    assert [(a, b) for a, b, _s, _r in _pair_lines(lines)] == want
    assert [(a, b) for a, b, _r in calls] == want and all(r is False for _a, _b, r in calls)
    assert all(s in ("measured", "unsupported") for s in states.values())
    assert not any(RX_CAND.match(l) for l in lines)
    assert not any("[INIT-ACCEPT]" in l for l in lines)
    assert lines[1] == "  >> [INIT-SETUP] window=2 survival_frames=10 refit=off" or \
        lines[0] == "  >> [INIT-SETUP] window=2 survival_frames=10 refit=off"


# --------------------------------------------------------------------------- O4
def test_o4_accepted_result_pose_points_and_lines(monkeypatch):
    frames, _truth = _yaw_scene(4)
    T0 = np.eye(4)
    T0[:3, :3] = rot_y(4.0) @ rot_x(-2.0)
    T0[:3, 3] = [0.3, -0.1, 0.7]
    n = 60
    res = _accepted(frames, 0, 1, n)
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False: res if (fa.id, fb.id) == (0, 1)
                        else _refusal("parallax"))
    monkeypatch.setattr(map_tracker.MapTracker, "on_frame", lambda self, k, fr, wm: "tracked")
    tr = opening.OpeningTracker(None)
    tr.world_map = Map(frames=frames)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.start(frames)
        frames[0].set_camera_pose(T0)       # a non-trivial anchor pose: rotation and translation
        state = tr.on_frame(1, frames)
    lines = buf.getvalue().splitlines()
    assert state == "init" and tr.state == "SURVIVAL"
    Rt = np.eye(4)
    Rt[:3, :3], Rt[:3, 3] = res.R, res.t
    assert np.allclose(frames[1].cam_T_world, Rt @ T0, atol=1e-12)
    assert not np.allclose(frames[1].cam_T_world, T0 @ Rt, atol=1e-6)     # the order matters here
    assert frames[1].pose_valid is True and frames[1].tracking_state == "OK"
    # the world points: X_w = R_a^T (X_a - t_a), and they reproject to camera b as R X_a + t
    Xw = (res.points - T0[:3, 3]) @ T0[:3, :3]
    got = np.array([p.point for p in tr.world_map.points])
    assert len(tr.world_map.points) == n and np.allclose(got, Xw, atol=1e-12)
    Xb = got @ frames[1].cam_T_world[:3, :3].T + frames[1].cam_T_world[:3, 3]
    assert np.allclose(Xb, res.points @ res.R.T + res.t, atol=1e-9)
    # every point linked to its keypoint kp_b of frame b
    for i, p in enumerate(tr.world_map.points):
        assert p.observations == {1: int(res.kp_b[i])}
        assert frames[1].keypoint_to_point[int(res.kp_b[i])] == p.id
    acc = [RX_ACC.match(l).groups() for l in lines if RX_ACC.match(l)]
    assert acc == [("1", "0", "1", str(n), "F")]
    pts = [RX_PT.match(l).groups() for l in lines if RX_PT.match(l)]
    assert len(pts) == n
    sa, sb = local_mapping._keypoint_sigmas(frames[0]), local_mapping._keypoint_sigmas(frames[1])
    for i, g in enumerate(pts):
        ka, kb = int(res.kp_a[i]), int(res.kp_b[i])
        assert g[:4] == ("1", str(i), str(ka), str(kb))
        assert g[4:8] == tuple("%.6f" % x for x in (*frames[0].KeyPoints[ka].pt, *frames[1].KeyPoints[kb].pt))
        assert g[8:12] == tuple("%.9f" % x for x in (*res.points[i], res.parallax_deg[i]))
        assert g[12:] == ("%.6f" % sa[ka], "%.6f" % sb[kb])
    poses = [RX_POSE.match(l).groups() for l in lines if RX_POSE.match(l)]
    assert poses == [("0", "1", "live", ",".join("%.9f" % x for x in res.R.reshape(-1)),
                      ",".join("%.9f" % x for x in res.t))]
    rel = [RX_REL.match(l).groups() for l in lines if RX_REL.match(l)]
    assert rel == [("1", str(n), "0", "0")]
    # the line order: pair, pose, accept, points, M1 setup, release
    order = [next(i for i, l in enumerate(lines) if tag in l)
             for tag in ("[INIT-PAIR]", "[INIT-POSE]", "[INIT-ACCEPT]", "[INIT-POINT]", "[M1-SETUP]",
                         "[M1-RELEASE]")]
    assert order == sorted(order)
    # the tracker carries the relocalizing state
    t = tr.tracker
    assert (t.mode, t.reloc, t.reloc_solver, t.reloc_shadow) == ("m1a", "pnp", "epnp", False)
    assert isinstance(t.reloc_params, map_tracker.MapRelocParams)


# --------------------------------------------------------------------------- O5
def _tracker_stub(truth, lost_at=()):
    def on_frame(self, k, frames, world_map):
        T = np.eye(4)
        T[:3, :3] = truth[k]
        frames[k].set_camera_pose(T)
        if k in lost_at:
            frames[k].pose_valid = False
            frames[k].tracking_state = "LOST"
            print(f"  >> [MAP-TRACK] frame={k} state=lost")
            return "lost"
        frames[k].pose_valid = True
        frames[k].tracking_state = "OK"
        print(f"  >> [MAP-TRACK] frame={k} state=tracked")
        return "tracked"
    return on_frame


def _accept_at(frames, a, b, n=60):
    res = _accepted(frames, a, b, n)
    return lambda fa, fb, params=None, refit=False: res if (fa.id, fb.id) == (a, b) else _refusal("parallax")


def test_o5_ten_tracked_frames_survive(monkeypatch):
    b = 5
    frames, truth = _yaw_scene(30)
    monkeypatch.setattr(initializer, "initialize_frames", _accept_at(frames, 0, b))
    monkeypatch.setattr(map_tracker.MapTracker, "on_frame", _tracker_stub(truth))
    tr, states, lines = _open_run(frames)
    assert states[b] == "init"
    assert all(states[k] == "tracked" for k in range(b + 1, 30))
    assert [RX_SURV.match(l).groups() for l in lines if RX_SURV.match(l)] == [("1", "0", str(b), str(b + 10))]
    i_surv = next(i for i, l in enumerate(lines) if RX_SURV.match(l))
    assert "[MAP-TRACK] frame=15 state=tracked" in lines[i_surv - 1]
    assert max(int(m.group(2)) for m in (RX_PAIR.match(l) for l in lines) if m) == b
    assert not any(RX_RESET.match(l) or RX_ONLY.match(l) and int(RX_ONLY.match(l).group(1)) > b for l in lines)
    assert tr.state == "MAP" and len(tr.world_map.points) == 60
    s = lines[-1]
    # frames 1..b are oriented by the segment (frame b too, before its attempt), the rest by the map
    assert s.startswith("  >> [INIT-SUMMARY] frames=30 rot_segments=1 rot_measured=5 rot_unsupported=0 rot_keyframes=1 ")
    assert ("attempts=5 accepted_live=1 accepted_shadow=0 errors=0 maps=1 resets=0 renewals=0 survived=1 "
            "first_accept_b=5 survived_b=5 tracked=24 lost=0 map_points=60") in s


def test_o5_lost_frame_is_a_false_start(monkeypatch):
    b = 5
    frames, truth = _yaw_scene(30)
    monkeypatch.setattr(initializer, "initialize_frames", _accept_at(frames, 0, b))
    monkeypatch.setattr(map_tracker.MapTracker, "on_frame", _tracker_stub(truth, lost_at=(b + 3,)))
    tr, states, lines = _open_run(frames)
    assert states[b + 1] == states[b + 2] == "tracked"
    resets = [RX_RESET.match(l).groups() for l in lines if RX_RESET.match(l)]
    assert resets == [("1", "0", str(b), str(b + 3), "2", "60")]
    assert len(tr.world_map.points) == 0
    assert all(not f.keypoint_to_point for f in frames)
    i = next(i for i, l in enumerate(lines) if RX_RESET.match(l))
    assert lines[i - 1].startswith(f"  >> [MAP-TRACK] frame={b + 3} state=lost")
    assert lines[i + 1] == f"  >> [ROT-KF] frame={b + 2} prev=-1 n=1"
    m = RX_ONLY.match(lines[i + 2])
    assert m and m.group(1) == str(b + 3) and m.group(3) == str(b + 2) and m.group(2) == "measured"
    # frame b+3 oriented by the new segment against frame b+2 (whose pose the tracker had set)
    assert states[b + 3] == "measured" and frames[b + 3].tracking_state == "ROT"
    R_est = frames[b + 3].cam_T_world[:3, :3]
    assert math.degrees(math.acos(min(1.0, (np.trace(R_est.T @ truth[b + 3]) - 1) / 2))) < 1e-6
    # no attempt at frame b+3; attempts resume at b+4 against the new segment's candidate b+2
    assert not any(int(x[1]) == b + 3 for x in _pair_lines(lines))
    assert [(a, k) for a, k, _s, _r in _pair_lines(lines) if k == b + 4] == [(b + 2, b + 4)]
    assert tr.state == "ROTATION" and tr.tracker is None
    s = lines[-1]
    assert " rot_segments=2 " in s and " resets=1 " in s and " survived=0 " in s and " tracked=2 lost=0 " in s
    assert s.endswith("map_points=0")


# --------------------------------------------------------------------------- O6
def _refit_stub(frames, calls):
    acc = _accepted(frames, 0, 1, 60)

    def stub(fa, fb, params=None, refit=False):
        calls.append(refit)
        return acc if refit else _refusal("few_good", 150)
    return stub


def test_o6_shadow_arm_is_logged_never_used(monkeypatch):
    frames, truth = _yaw_scene(12)
    calls = []
    monkeypatch.setattr(initializer, "initialize_frames", _refit_stub(frames, calls))
    tr, states, lines = _open_run(frames, refit="shadow")
    live, shadow = _pair_lines(lines), _pair_lines(lines, RX_SHADOW)
    assert len(live) == 11 and [(a, b) for a, b, _s, _r in live] == [(a, b) for a, b, _s, _r in shadow]
    assert all(s == "refused" for _a, _b, s, _r in live) and all(s == "accepted" for _a, _b, s, _r in shadow)
    # each shadow line right after its live line, its pose line right after it
    for i, l in enumerate(lines):
        if RX_PAIR.match(l):
            assert RX_SHADOW.match(lines[i + 1])
            assert RX_POSE.match(lines[i + 2]).group(3) == "shadow"
    assert not any("[INIT-ACCEPT]" in l for l in lines) and len(tr.world_map.points) == 0
    assert all(s in ("measured", "unsupported") for s in states.values())
    assert calls == [False, True] * 11
    assert " accepted_live=0 accepted_shadow=11 " in lines[-1]


def test_o6_off_arm_has_no_shadow_line(monkeypatch):
    frames, _truth = _yaw_scene(12)
    calls = []
    monkeypatch.setattr(initializer, "initialize_frames", _refit_stub(frames, calls))
    _tr, _states, lines = _open_run(frames, refit="off")
    assert not any("[INIT-PAIR-SHADOW]" in l or "arm=shadow" in l for l in lines)
    assert calls == [False] * 11


def test_o6_live_arm_calls_with_refit_only(monkeypatch):
    frames, truth = _yaw_scene(12)
    calls = []
    monkeypatch.setattr(initializer, "initialize_frames", _refit_stub(frames, calls))
    monkeypatch.setattr(map_tracker.MapTracker, "on_frame", _tracker_stub(truth))
    tr, states, lines = _open_run(frames, refit="live")
    assert calls == [True]
    assert states[1] == "init" and [RX_ACC.match(l).groups()[:3] for l in lines if RX_ACC.match(l)] == [("1", "0", "1")]
    assert not any("[INIT-PAIR-SHADOW]" in l for l in lines)


# --------------------------------------------------------------------------- O7
def _two_candidate_frame():
    frames, truth = _yaw_scene(40)
    _poses, plain = _plain_run(_yaw_scene(40)[0])
    kfs = [int(RX_KF.match(l).group(1)) for l in plain if RX_KF.match(l)]
    assert len(kfs) >= 2
    return frames, truth, kfs[0], kfs[1], kfs[1] + 1


@pytest.mark.parametrize("n_first,n_second,want", [(60, 80, "second"), (80, 60, "first"), (70, 70, "first")])
def test_o7_best_pair_most_published_ties_to_lower_a(monkeypatch, n_first, n_second, want):
    frames, truth, a1, a2, b = _two_candidate_frame()
    res = {a1: _accepted(frames, a1, b, n_first, seed=2), a2: _accepted(frames, a2, b, n_second, seed=3)}
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False: res[fa.id] if fb.id == b else _refusal("parallax"))
    monkeypatch.setattr(map_tracker.MapTracker, "on_frame", _tracker_stub(truth))
    tr, states, lines = _open_run(frames, n=b + 1)
    assert [(a, k) for a, k, _s, _r in _pair_lines(lines) if k == b] == [(a1, b), (a2, b)]
    a = a1 if want == "first" else a2
    n = n_first if want == "first" else n_second
    assert [RX_ACC.match(l).groups() for l in lines if RX_ACC.match(l)] == [("1", str(a), str(b), str(n), "F")]
    Rt = np.eye(4)
    Rt[:3, :3], Rt[:3, 3] = res[a].R, res[a].t
    assert len(tr.world_map.points) == n
    assert states[b] == "init"


# --------------------------------------------------------------------------- O8
def test_o8_no_leak_tokens_and_imports():
    src = Path(opening.__file__).read_text()
    for tok in ("groundtruth", "gt_path", "ground_truth", "depth.txt", "depth/", "cv2.imread"):
        assert tok not in src, tok
    assert not re.search(r"['\"]depth['\"]", src)
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    ok = {"numpy", "rotation_only", "initializer", "map_tracker", "local_mapping", "keypoints_match", "pose_log"}
    assert not {m for m in mods if m not in ok and m not in sys.stdlib_module_names}, mods
    assert not mods & {"oracle", "dataset", "pipeline", "npnp"}


# --------------------------------------------------------------------------- O9
def _cands_by_frame(lines):
    out = {}
    for a, b, _s, _r in _pair_lines(lines):
        out.setdefault(b, []).append(a)
    return out


def test_o9_renewal_on_matches_refusals(monkeypatch):
    k = 3
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False:
                        _refusal("matches", 20) if fb.id == k else _refusal("parallax", 150))
    frames, _truth = _yaw_scene(8)
    assert len(frames[k].KeyPoints) > initializer.InitParams().min_matches
    poses_plain, plain = _plain_run(_yaw_scene(8)[0])
    tr, _states, lines = _open_run(frames)
    cands = [RX_CAND.match(l).groups() for l in lines if RX_CAND.match(l)]
    assert cands == [(str(k), "matches")]
    i = next(i for i, l in enumerate(lines) if RX_CAND.match(l))
    assert RX_PAIR.match(lines[i - 1]) and int(RX_PAIR.match(lines[i - 1]).group(2)) == k
    assert "[POSE-ONLINE]" not in lines[i - 1]
    c = _cands_by_frame(lines)
    assert c[k] == [0] and c[k + 1] == [0, k] and c[k + 2] == [0, k]
    assert tr.seg_cands == [0, k] and tr.seg_kfs == [0]
    assert _rot_lines(lines) == _rot_lines(plain)
    assert all(np.array_equal(f.cam_T_world, T) for f, T in zip(frames, poses_plain))
    assert " renewals=1 " in lines[-1]


def test_o9_no_renewal_when_one_pair_refused_for_another_reason(monkeypatch):
    k = 4
    frames, _truth = _yaw_scene(8)
    # renew at frame 2 first so that frame k has two candidates (0 and 2)
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False:
                        _refusal("matches", 20) if (fb.id == 2 or (fb.id == k and fa.id == 0))
                        else _refusal("parallax", 150))
    _tr, _states, lines = _open_run(frames)
    c = _cands_by_frame(lines)
    assert c[k] == [0, 2] and c[k + 1] == [0, 2]
    assert [RX_CAND.match(l).group(1) for l in lines if RX_CAND.match(l)] == ["2"]


def test_o9_no_renewal_with_100_keypoints_or_fewer(monkeypatch):
    k = 3
    frames, _truth = _yaw_scene(8)
    frames[k].KeyPoints = frames[k].KeyPoints[:100]
    frames[k].Descriptors = frames[k].Descriptors[:100]
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False: _refusal("matches", 20))
    frames_p, _ = _yaw_scene(8)
    frames_p[k].KeyPoints = frames_p[k].KeyPoints[:100]
    frames_p[k].Descriptors = frames_p[k].Descriptors[:100]
    poses_plain, plain = _plain_run(frames_p)
    _tr, _states, lines = _open_run(frames)
    renewed = [int(RX_CAND.match(l).group(1)) for l in lines if RX_CAND.match(l)]
    assert k not in renewed and k - 1 in renewed and k + 1 in renewed
    assert _cands_by_frame(lines)[k + 1] == [1, 2]
    assert _rot_lines(lines) == _rot_lines(plain)
    assert all(np.array_equal(f.cam_T_world, T) for f, T in zip(frames, poses_plain))


def test_o9_no_renewal_when_the_live_call_raises(monkeypatch):
    k = 3

    def stub(fa, fb, params=None, refit=False):
        if fb.id == k:
            raise RuntimeError("boom\nsecond line")
        return _refusal("parallax", 150)

    monkeypatch.setattr(initializer, "initialize_frames", stub)
    frames, _truth = _yaw_scene(8)
    tr, _states, lines = _open_run(frames)
    errs = [RX_ERR.match(l).groups() for l in lines if RX_ERR.match(l)]
    assert errs == [("0", str(k), "live", "RuntimeError: boom second line")]
    assert not any(RX_CAND.match(l) for l in lines)
    assert not any(int(b) == k for _a, b, _s, _r in _pair_lines(lines))
    assert " attempts=7 " in lines[-1] and " errors=1 " in lines[-1]


def test_o9_live_error_next_to_a_matches_refusal_blocks_the_renewal(monkeypatch):
    """Two candidates at frame k: one refused with reason matches, the other raising. The renewal
    needs every live candidate answered by a matches refusal, so the error line prevents it."""
    k = 4

    def stub(fa, fb, params=None, refit=False):
        if fb.id == 2:
            return _refusal("matches", 20)          # renews frame 2, so frame k has candidates 0 and 2
        if fb.id == k:
            if fa.id == 2:
                raise RuntimeError("boom")
            return _refusal("matches", 20)
        return _refusal("parallax", 150)

    monkeypatch.setattr(initializer, "initialize_frames", stub)
    frames, _truth = _yaw_scene(8)
    tr, _states, lines = _open_run(frames)
    assert [(a, b, arm) for a, b, arm, _e in (RX_ERR.match(l).groups() for l in lines if RX_ERR.match(l))] == \
        [("2", str(k), "live")]
    assert [(a, b) for a, b, _s, _r in _pair_lines(lines) if b == k] == [(0, k)]
    assert [RX_CAND.match(l).group(1) for l in lines if RX_CAND.match(l)] == ["2"]
    assert _cands_by_frame(lines)[k + 1] == [0, 2] and tr.seg_cands == [0, 2]


def test_o9_shadow_matches_refusal_never_renews(monkeypatch):
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False:
                        _refusal("matches", 20) if refit else _refusal("parallax", 150))
    frames, _truth = _yaw_scene(8)
    _tr, _states, lines = _open_run(frames, refit="shadow")
    assert len(_pair_lines(lines, RX_SHADOW)) == 7
    assert all(r == "matches" for _a, _b, _s, r in _pair_lines(lines, RX_SHADOW))
    assert not any(RX_CAND.match(l) for l in lines)


def test_o9_renewal_line_follows_the_shadow_lines(monkeypatch):
    k = 2
    monkeypatch.setattr(initializer, "initialize_frames",
                        lambda fa, fb, params=None, refit=False:
                        _refusal("matches", 20) if fb.id == k else _refusal("parallax", 150))
    frames, _truth = _yaw_scene(6)
    _tr, _states, lines = _open_run(frames, refit="shadow")
    i = next(i for i, l in enumerate(lines) if RX_CAND.match(l))
    assert RX_SHADOW.match(lines[i - 1]) and int(RX_SHADOW.match(lines[i - 1]).group(2)) == k
    assert _cands_by_frame(lines)[k + 1] == [0, k]
