"""Stage 04-m1: registered risk 1's residual-consistency test (residual_test.py), its sites in the map
tracker (map_tracker.py, map_rotation.py) and the --residual_test switch (main.py).

T1 critical_mean (reference values, monotone in n, ValueError, a Monte Carlo of the rejection share);
T2 evaluate; T3 the attribute off; T4 log; T5 on, tracking; T6 on, relocalization; T7 the r52 exit under
on; T8 the 04q shadow under log; T9 main (refusals, None for off, the hand-off of the attribute).
The stand-ins are copied from tests/test_map_tracker.py (not imported).
"""
from __future__ import annotations

import argparse
import contextlib
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
import map_rotation as mr  # noqa: E402
import map_tracker as mt  # noqa: E402
import residual_test as rt  # noqa: E402
from Map import Map  # noqa: E402
from frame import DEFAULT_K  # noqa: E402

K = np.asarray(DEFAULT_K, dtype=np.float64)
W, H = 640, 480
CUT = 5.991

# spec B2's reference values at cutoff 5.991 (the orchestrator's convolution)
REF = {2: 3.5327, 3: 3.1716, 5: 2.8092, 10: 2.4623, 20: 2.2261, 30: 2.1237, 50: 2.0224, 100: 1.9218,
       300: 1.8206, 1000: 1.7588}
CLOSED_N1 = -2.0 * math.log(1.0 - 0.95 * (1.0 - math.exp(-CUT / 2.0)))

_F = r"(?:-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|-?inf|nan)"
RX_TEST = re.compile(
    rf"^  >> \[RESID-TEST\] frame=(?P<frame>\d+) mode=(?P<mode>m1a|m1b) kind=(?P<kind>track|reloc|exit|shadow) "
    rf"shadow=(?P<shadow>[01]) solver=(?P<solver>none|epnp|sqpnp|newton) n=(?P<n>\d+) stat=(?P<stat>{_F}) "
    rf"crit=(?P<crit>{_F}) max_e2=(?P<max_e2>{_F}) pass=(?P<pass>[01]) applied=(?P<applied>[01]) "
    rf"hull=(?P<hull>{_F}) pose=(?P<pose>none|{_F}(?:,{_F}){{11}})$")
RX_RSET = re.compile(
    rf"^  >> \[RESID-SET\] frame=(?P<frame>\d+) kind=(?P<kind>track|reloc|exit|shadow) shadow=(?P<shadow>[01]) "
    rf"solver=(?P<solver>none|epnp|sqpnp|newton) n=(?P<n>\d+) points=(?P<points>{_F}(?:,{_F})*) "
    rf"uv=(?P<uv>{_F}(?:,{_F})*) sigma=(?P<sigma>{_F}(?:,{_F})*)$")
RX_TRACK = re.compile(r"\[MAP-TRACK\] frame=(\d+) state=(tracked|lost) mode=(m1a|m1b) pred_rot=(twoview|cv) "
                      r"step_a=(-?\d+) step_a_inliers=(-?\d+) step_a_wide=([01]) step_b=(-?\d+) obs=(\d+) "
                      r"inliers=(\d+) rmse_px=(\S+)\s*$")
RX_RELOC = re.compile(r"^  >> \[MAP-RELOC\] frame=(\d+) mode=(m1a|m1b) status=(ok|fail) "
                      r"reason=(none|too_few|pnp|inliers|error|residual) matches=(\d+) ransac_inliers=(-?\d+) "
                      r"inliers=(\d+)$")
RX_ATT = re.compile(r"^  >> \[MAP-RELOC-ATTEMPT\] frame=(\d+) mode=(m1a|m1b) shadow=([01]) solver=(\w+) "
                    r"outcome=(\w+) .* pose=(\S+)$")
RX_MS = re.compile(r"refit_ms=\S+")


# =========================================================================== T1 critical_mean
def test_t1_reference_values_and_closed_form():
    assert abs(rt.critical_mean(1, CUT) - CLOSED_N1) <= 2e-3
    assert abs(CLOSED_N1 - 4.655579) <= 1e-6
    for n, v in REF.items():
        assert abs(rt.critical_mean(n, CUT) - v) <= 2e-3, (n, rt.critical_mean(n, CUT), v)
    # the normal approximation (R63-7's orientation figures) is not what is computed
    for n, v in ((5, 2.73), (10, 2.42), (30, 2.11), (100, 1.92)):
        approx = 1.684607 + 1.6448536 * math.sqrt(2.011008 / n)
        assert abs(approx - v) <= 6e-3                       # the figures are the normal approximation
    assert abs(rt.critical_mean(5, CUT) - 2.73) > 0.05
    assert rt.LEVEL == 0.95


def test_t1_strictly_decreasing_and_deterministic():
    ns = (1, 2, 3, 5, 10, 20, 30, 50, 100)
    seq = [rt.critical_mean(n, CUT) for n in ns]
    assert all(a > b for a, b in zip(seq, seq[1:])), seq
    # every value lies between the one-draw mean and the cutoff
    assert all(1.684607 < x < CUT for x in seq)
    rt._critical_mean_cached.cache_clear()
    assert [rt.critical_mean(n, CUT) for n in ns] == seq


def test_t1_value_errors():
    for bad in (0, -1, 2.5, True, None):
        with pytest.raises(ValueError):
            rt.critical_mean(bad, CUT)
    with pytest.raises(ValueError):
        rt.critical_mean(5, 0.0)
    assert isinstance(rt.critical_mean(np.int64(7), CUT), float)


def _truncated_draws(rng, size):
    """Inverse-CDF draws of chi-square(2) truncated to [0, CUT]."""
    z = 1.0 - math.exp(-CUT / 2.0)
    return -2.0 * np.log1p(-rng.uniform(0.0, 1.0, size) * z)


@pytest.mark.parametrize("n", [5, 30])
def test_t1_monte_carlo_rejection_share(n):
    rng = np.random.default_rng(0)
    means = _truncated_draws(rng, (200_000, n)).mean(1)
    share = float(np.mean(means > rt.critical_mean(n, CUT)))
    assert abs(share - 0.05) <= 0.004, share
    # the normal approximation's point rejects a visibly different share at n = 5
    if n == 5:
        assert abs(float(np.mean(means > 2.73)) - 0.05) > 0.004


# =========================================================================== T2 evaluate
def test_t2_evaluate():
    e0 = rt.evaluate(np.zeros(0), CUT)
    assert e0["n"] == 0 and e0["passed"] is False
    assert all(math.isnan(e0[k]) for k in ("stat", "crit", "max_e2"))
    e1 = rt.evaluate(np.full(30, 1.0), CUT)
    assert e1["passed"] is True and e1["n"] == 30
    e3 = rt.evaluate(np.full(30, 3.0), CUT)
    assert e3["passed"] is False
    v = np.array([0.5, 2.0, 1.25, 5.5, 0.0, 3.0])
    e = rt.evaluate(v, CUT)
    assert type(e["n"]) is int and all(type(e[k]) is float for k in ("stat", "crit", "max_e2"))
    assert type(e["passed"]) is bool
    assert e["stat"] == float(np.mean(v)) and e["max_e2"] == 5.5
    assert e["crit"] == rt.critical_mean(6, CUT)
    assert e["passed"] == (e["stat"] <= e["crit"])
    # the decision at the boundary: stat == crit passes
    c = rt.critical_mean(4, CUT)
    assert rt.evaluate(np.full(4, c), CUT)["passed"] is True
    assert rt.evaluate(np.full(4, np.nextafter(c, 10.0)), CUT)["passed"] is False


# =========================================================================== stand-ins (from test_map_tracker)
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
        T[0, 3] = -step * k
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
        f.set_camera_pose(np.eye(4))
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
                  anchor_k="0", n_keyframes="2", n_points=str(len(pts)),
                  n_corr=str(sum(len(v) for v in (corr or {}).values())))
    return mt.M1Map(header=header, points=pts, corr=dict(corr or {}))


def _c_lines(fr, pids_lo, pids_hi, n):
    chosen = [kp for kp, pid in enumerate(fr.pid_of_kp) if pids_lo <= pid < pids_hi][:n]
    return [(kp, fr.pid_of_kp[kp], *fr.KeyPoints[kp].pt) for kp in chosen]


def _inflate(fr, mag=1.8):
    """Move every keypoint of fr by mag px in a well-spread direction (golden-angle sequence): the residuals
    of a fitted pose stay inside the 5.991 gate (mag^2 < 5.991) while their mean is far above the test's
    critical point (about 2 for n of order 60)."""
    g = math.pi * (3.0 - math.sqrt(5.0))
    kps = []
    for i, kp in enumerate(fr.KeyPoints):
        u, v = kp.pt
        nk = cv2.KeyPoint(float(u + mag * math.cos(g * i)), float(v + mag * math.sin(g * i)), float(kp.size))
        nk.octave = kp.octave
        kps.append(nk)
    fr.KeyPoints = kps


@pytest.fixture
def no_twoview(monkeypatch):
    monkeypatch.setattr(mt.tracking, "estimate_motion", lambda a, b, use_homography=False: None)


def _capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue().splitlines()


def _links(fr, wmap):
    """The frame's links as (keypoint, index of the point in the map): point ids are global counters."""
    idx = {p.id: i for i, p in enumerate(wmap.points)}
    return sorted((kp, idx.get(pid, -1)) for kp, pid in fr.keypoint_to_point.items())


def _frame1_run(mode_attr, inflate, delete_attr=False):
    """M1a: frame 0 released, then frame 1 tracked once (optionally with inflated residuals)."""
    frames, X, T_true = _world()
    if inflate:
        _inflate(frames[1])
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X), "m1a", None)
    if delete_attr:
        del tr.residual_test
    else:
        tr.residual_test = mode_attr
    with contextlib.redirect_stdout(io.StringIO()):
        tr.release(0, frames, wmap)
    state, out = _capture(tr.on_frame, 1, frames, wmap)
    fr = frames[1]
    snap = (fr.cam_T_world.tobytes(), fr.pose_valid, fr.tracking_state, _links(fr, wmap))
    return state, out, snap, frames, T_true


def _pose_of(s):
    T = np.eye(4)
    T[:3, :4] = np.array([float(x) for x in s.split(",")]).reshape(3, 4)
    return T


def _floats(s):
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def _recompute(test_line, set_line):
    """The statistic from the [RESID-SET] fields at the [RESID-TEST] pose, by an independent projection."""
    t = RX_TEST.match(test_line)
    s = RX_RSET.match(set_line)
    assert t is not None and s is not None, (test_line, set_line)
    n = int(s.group("n"))
    P = _floats(s.group("points")).reshape(n, 3)
    UV = _floats(s.group("uv")).reshape(n, 2)
    S = _floats(s.group("sigma")).reshape(n)
    T = _pose_of(t.group("pose"))
    Xc = P @ T[:3, :3].T + T[:3, 3]
    proj = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
    e2 = (((proj - UV) / S[:, None]) ** 2).sum(1)
    return float(np.mean(e2)), float(np.max(e2)), n


def _resid_lines(out):
    return [l for l in out if "[RESID-" in l]


# =========================================================================== T3 off
@pytest.mark.parametrize("inflate", [False, True])
def test_t3_off_equals_absent(no_twoview, inflate):
    s_abs, out_abs, snap_abs, _f, _T = _frame1_run(None, inflate, delete_attr=True)
    s_off, out_off, snap_off, _f, _T = _frame1_run("off", inflate)
    assert mt.MapTracker(mt.M1Map(header={}, points=[], corr={}), "m1a").residual_test == "off"
    assert s_abs == s_off == "tracked"
    assert snap_abs == snap_off and out_abs == out_off
    assert not _resid_lines(out_off)


def test_t3_off_resid_accept_computes_nothing(monkeypatch):
    tr = mt.MapTracker(mt.M1Map(header={}, points=[], corr={}), "m1a")
    monkeypatch.setattr(mt.residual_test, "evaluate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr._resid_accept(3, "track", None, None, [1], [2], [True]) is True
    assert buf.getvalue() == ""


# =========================================================================== T4 log
@pytest.mark.parametrize("inflate", [False, True])
def test_t4_log_changes_nothing_and_logs_one_pair(no_twoview, inflate):
    s_off, out_off, snap_off, _f, _T = _frame1_run("off", inflate)
    s_log, out_log, snap_log, _f, _T = _frame1_run("log", inflate)
    assert s_off == s_log == "tracked" and snap_off == snap_log
    rl = _resid_lines(out_log)
    assert [l for l in out_log if l not in rl] == out_off
    assert len(rl) == 2
    t = RX_TEST.match(rl[0])
    s = RX_RSET.match(rl[1])
    assert t is not None and s is not None
    assert (t.group("frame"), t.group("mode"), t.group("kind"), t.group("shadow"), t.group("solver"),
            t.group("applied")) == ("1", "m1a", "track", "0", "none", "0")
    assert (s.group("frame"), s.group("kind"), s.group("shadow"), s.group("solver")) == ("1", "track", "0", "none")
    # the pair comes before the frame's [MAP-TRACK] line; n is the frame's inlier count
    i_track = [i for i, l in enumerate(out_log) if "[MAP-TRACK]" in l][0]
    assert out_log.index(rl[1]) == out_log.index(rl[0]) + 1 < i_track
    assert int(t.group("n")) == int(s.group("n")) == int(RX_TRACK.search(out_log[i_track]).group(10))
    stat, max_e2, n = _recompute(rl[0], rl[1])
    lstat = float(t.group("stat"))
    assert abs(lstat - stat) <= 1e-12 * max(1.0, abs(stat))
    assert abs(float(t.group("max_e2")) - max_e2) <= 1e-12 * max(1.0, max_e2)
    assert float(t.group("crit")) == rt.critical_mean(n, CUT)
    assert t.group("pass") == ("0" if inflate else "1")
    assert float(t.group("hull")) > 0.1
    if inflate:
        assert lstat > 1.5 * rt.critical_mean(n, CUT) and max_e2 <= CUT


# =========================================================================== T5 on, tracking
def test_t5_on_rejects_an_inflated_frame(no_twoview):
    s_log, out_log, _snap, _f, _T = _frame1_run("log", True)
    s_on, out_on, snap_on, frames, _T = _frame1_run("on", True)
    assert s_log == "tracked" and s_on == "lost"
    fr = frames[1]
    T_pred = mt.predict_pose(frames[0].cam_T_world, None, None)
    assert np.array_equal(fr.cam_T_world, T_pred) and fr.pose_valid is False and fr.tracking_state == "LOST"
    assert fr.keypoint_to_point == {}
    t = RX_TEST.match(_resid_lines(out_on)[0])
    assert t.group("pass") == "0" and t.group("applied") == "1" and t.group("kind") == "track"
    # the lost line carries the same obs, inliers and rmse fields as the count-accepted line
    tl_on = RX_TRACK.search([l for l in out_on if "[MAP-TRACK]" in l][0])
    tl_log = RX_TRACK.search([l for l in out_log if "[MAP-TRACK]" in l][0])
    assert tl_on.group(2) == "lost" and tl_log.group(2) == "tracked"
    assert tl_on.groups()[2:] == tl_log.groups()[2:]
    assert not any("[MAP-RELOC" in l for l in out_on)


def test_t5_on_tracks_a_clean_frame_as_off(no_twoview):
    s_off, out_off, snap_off, _f, T_true = _frame1_run("off", False)
    s_on, out_on, snap_on, frames, _T = _frame1_run("on", False)
    assert s_off == s_on == "tracked" and snap_on == snap_off
    rl = _resid_lines(out_on)
    assert [l for l in out_on if l not in rl] == out_off
    t = RX_TEST.match(rl[0])
    assert t.group("pass") == "1" and t.group("applied") == "1"
    assert np.allclose(frames[1].cam_T_world, T_true[1], atol=1e-6)


def test_t5_on_computation_error_refuses(no_twoview, monkeypatch):
    monkeypatch.setattr(mt.residual_test, "evaluate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad\nthing")))
    s_on, out_on, _snap, _f, _T = _frame1_run("on", False)
    s_log, out_log, _snap, _f, _T = _frame1_run("log", False)
    assert s_on == "lost" and s_log == "tracked"
    for out in (out_on, out_log):
        err = [l for l in out if "[RESID-" in l]
        assert err == ["  >> [RESID-ERROR] frame=1 kind=track error=RuntimeError: bad thing"]


# =========================================================================== T6 on, relocalization
def _reloc_run(mode_attr, monkeypatch=None):
    """M1b: frame 1 has 10 C lines (LOST); frame 2's keypoints are inflated and it has 60 C lines, so its
    relocalization passes the count; frame 3 is clean."""
    frames, X, T_true = _world()
    _inflate(frames[2])
    corr = {1: _c_lines(frames[1], 0, 100, 10), 2: _c_lines(frames[2], 0, 100, 60), 3: _c_lines(frames[3], 0, 200, 60)}
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), "m1b", None)
    tr.reloc = "pnp"
    tr.residual_test = mode_attr
    buf = io.StringIO()
    states = []
    with contextlib.redirect_stdout(buf):
        tr.release(0, frames, wmap)
        for k in (1, 2):
            states.append(tr.on_frame(k, frames, wmap))
            tr.release(k, frames, wmap)
    return states, buf.getvalue().splitlines(), frames


def test_t6_on_rejected_relocalization(no_twoview):
    s_log, out_log, _fr = _reloc_run("log")
    s_on, out_on, frames = _reloc_run("on")
    assert s_log == ["lost", "tracked"] and s_on == ["lost", "lost"]
    rel_log = [RX_RELOC.match(l) for l in out_log if "[MAP-RELOC]" in l]
    rel_on = [RX_RELOC.match(l) for l in out_on if "[MAP-RELOC]" in l]
    assert len(rel_log) == len(rel_on) == 1
    assert rel_log[0].group(3) == "ok" and rel_log[0].group(4) == "none"
    assert rel_on[0].group(3) == "fail" and rel_on[0].group(4) == "residual"
    assert rel_on[0].groups()[4:] == rel_log[0].groups()[4:]
    att_on = [RX_ATT.match(l) for l in out_on if "[MAP-RELOC-ATTEMPT]" in l]
    att_log = [RX_ATT.match(l) for l in out_log if "[MAP-RELOC-ATTEMPT]" in l]
    assert len(att_on) == 1 and att_on[0].group(5) == "residual" and att_log[0].group(5) == "ok"
    assert att_on[0].group(6) != "none" and att_on[0].group(6) == att_log[0].group(6)   # the pose stays
    # the test line comes before the attempt and [MAP-RELOC] lines, kind reloc, the run's solver
    t = [RX_TEST.match(l) for l in out_on if "[RESID-TEST]" in l]
    assert len(t) == 1 and t[0].group("kind") == "reloc" and t[0].group("solver") == "epnp"
    assert t[0].group("pass") == "0" and t[0].group("applied") == "1"
    i_t = [i for i, l in enumerate(out_on) if "[RESID-TEST]" in l][0]
    i_att = [i for i, l in enumerate(out_on) if "[MAP-RELOC-ATTEMPT]" in l][0]
    assert i_t < i_att
    # the frame stays lost, with no tracking link (keyframe 2's own released points aside)
    fr = frames[2]
    assert fr.tracking_state == "LOST" and fr.pose_valid is False
    tl = RX_TRACK.search([l for l in out_on if "[MAP-TRACK] frame=2 " in l][0])
    assert tl.group(2) == "lost"
    # outside the [RESID-*] lines and the decision, the two runs print the same
    strip = lambda out: [RX_MS.sub("refit_ms=T", l) for l in out if "[RESID-" not in l]
    diff = [(a, b) for a, b in zip(strip(out_log), strip(out_on)) if a != b]
    assert len(strip(out_log)) == len(strip(out_on)) and len(diff) == 3   # attempt, [MAP-RELOC], [MAP-TRACK]


# =========================================================================== T7 r52 exit under on
class _Seg:
    kf = 0
    n_holds = 0

    def on_frame(self, k, frames):
        return "measured"


def _r52_inside(mode_attr, inflate, monkeypatch):
    frames, X, T_true = _world()
    if inflate:
        _inflate(frames[1])
    wmap = Map(points=[], frames=frames)
    tr = mr.MapRotationTracker(_m1map(frames, X), "m1a", None)
    tr.variant = "r52"
    tr.reloc = "pnp"
    tr.residual_test = mode_attr
    with contextlib.redirect_stdout(io.StringIO()):
        tr.release(0, frames, wmap)
    tr._anchor = 0
    tr._seg = _Seg()
    calls = []

    def parent(self, k, frames_, world_map):          # positional, as test_map_rotation's stubs
        calls.append(k)
        return "lost"
    monkeypatch.setattr(mt.MapTracker, "on_frame", parent)
    state, out = _capture(tr.on_frame, 1, frames, wmap)
    return state, out, calls, tr, frames


def test_t7_r52_rejected_exit_goes_to_relocalization(no_twoview, monkeypatch):
    s_off, out_off, calls_off, tr_off, _f = _r52_inside("off", True, monkeypatch)
    assert s_off == "tracked" and calls_off == [] and tr_off.n_exits_track == 1
    s_on, out_on, calls_on, tr_on, frames = _r52_inside("on", True, monkeypatch)
    assert calls_on == [1] and s_on == "rotation" and tr_on.n_exits_track == 0 and tr_on._anchor == 0
    t = [RX_TEST.match(l) for l in out_on if "[RESID-TEST]" in l]
    assert len(t) == 1 and t[0].group("kind") == "exit" and t[0].group("pass") == "0"
    assert t[0].group("applied") == "1" and t[0].group("solver") == "none"
    assert any("[MAP-ROT] frame=1 event=stay" in l for l in out_on)
    assert frames[1].keypoint_to_point == {}
    # under log the same exit is taken, with its line
    s_log, out_log, calls_log, _tr, _f = _r52_inside("log", True, monkeypatch)
    assert s_log == "tracked" and calls_log == []
    assert [l for l in out_log if "[RESID-" not in l] == out_off


def test_t7_r52_clean_exit_under_on(no_twoview, monkeypatch):
    s_on, out_on, calls_on, tr_on, _f = _r52_inside("on", False, monkeypatch)
    assert s_on == "tracked" and calls_on == [] and tr_on.n_exits_track == 1
    t = [RX_TEST.match(l) for l in out_on if "[RESID-TEST]" in l]
    assert len(t) == 1 and t[0].group("pass") == "1" and t[0].group("kind") == "exit"


def test_t7_r52_raising_computation_refuses_the_exit(no_twoview, monkeypatch):
    monkeypatch.setattr(mt.residual_test, "evaluate", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    s_on, out_on, calls_on, tr_on, _f = _r52_inside("on", False, monkeypatch)
    assert calls_on == [1] and s_on == "rotation" and tr_on.n_exits_track == 0
    assert [l for l in out_on if "[RESID-" in l] == ["  >> [RESID-ERROR] frame=1 kind=exit error=ValueError: x"]
    assert not any("[MAP-ROT-ERROR]" in l for l in out_on)


# =========================================================================== T8 shadow under log
def _shadow_run(mode, attr, n_frames=8, lost_at=1, monkeypatch=None):
    frames, X, T_true = _world(n_frames=n_frames, step=0.005)
    corr = None
    if mode == "m1b":
        corr = {k: _c_lines(frames[k], 0, 200 if k > 2 else 100, 60) for k in range(1, n_frames)}
        corr[lost_at] = corr[lost_at][:10]
    wmap = Map(points=[], frames=frames)
    tr = mt.MapTracker(_m1map(frames, X, corr), mode, None)
    tr.reloc = "pnp"
    tr.reloc_shadow = True
    tr.residual_test = attr
    if mode == "m1a":
        real = tr._track_m1a

        def forced(k, fr, T):
            if k == lost_at:
                raise RuntimeError("forced")
            return real(k, fr, T)
        tr._track_m1a = forced
    buf = io.StringIO()
    states = []
    with contextlib.redirect_stdout(buf):
        tr.release(0, frames, wmap)
        for k in range(1, n_frames):
            states.append(tr.on_frame(k, frames, wmap))
            tr.release(k, frames, wmap)
    snap = [(f.cam_T_world.tobytes(), f.pose_valid, f.tracking_state, _links(f, wmap)) for f in frames]
    return states, buf.getvalue().splitlines(), snap


@pytest.mark.parametrize("mode", ["m1a", "m1b"])
def test_t8_shadow_lines_under_log(no_twoview, mode):
    s0, out0, snap0 = _shadow_run(mode, "off")
    s1, out1, snap1 = _shadow_run(mode, "log")
    assert s0 == s1 and snap0 == snap1 and s0[0] == "lost" and "tracked" in s0
    # the attempt lines are identical to the run with the attribute off (the timing field masked)
    att = lambda out: [RX_MS.sub("refit_ms=T", l) for l in out if "[MAP-RELOC-ATTEMPT]" in l]
    assert att(out0) == att(out1)
    assert [RX_MS.sub("refit_ms=T", l) for l in out1 if "[RESID-" not in l] == \
        [RX_MS.sub("refit_ms=T", l) for l in out0]
    assert not _resid_lines(out0)
    n_shadow = 0
    for i, l in enumerate(out1):
        a = RX_ATT.match(l)
        if a is None or a.group(3) != "1" or a.group(4) != "newton":
            continue
        k = a.group(1)
        ok_solvers = [RX_ATT.match(x).group(4) for x in out1[:i + 1]
                      if RX_ATT.match(x) and RX_ATT.match(x).group(1) == k and RX_ATT.match(x).group(3) == "1"
                      and RX_ATT.match(x).group(5) == "ok"]
        after = out1[i + 1:i + 1 + 2 * len(ok_solvers)]
        tests = [RX_TEST.match(x) for x in after[0::2]]
        sets = [RX_RSET.match(x) for x in after[1::2]]
        assert all(t is not None for t in tests) and all(s is not None for s in sets)
        assert [t.group("solver") for t in tests] == ok_solvers
        for t, s, tl, sl in zip(tests, sets, after[0::2], after[1::2]):
            assert t.group("frame") == s.group("frame") == k and t.group("kind") == "shadow"
            assert t.group("shadow") == "1" and t.group("applied") == "0"
            stat, _mx, _n = _recompute(tl, sl)
            assert abs(float(t.group("stat")) - stat) <= 1e-12 * max(1.0, stat)
        nxt = out1[i + 1 + 2 * len(ok_solvers)] if i + 1 + 2 * len(ok_solvers) < len(out1) else ""
        assert "shadow=1" not in nxt
        n_shadow += len(ok_solvers)
    assert n_shadow == 3 * s1.count("tracked")
    # one non-shadow line per tracked frame (kind track or reloc), none for a lost frame
    ns = [RX_TEST.match(l) for l in out1 if "[RESID-TEST]" in l and "shadow=0" in l]
    assert len(ns) == s1.count("tracked")
    assert [t.group("kind") for t in ns][0] == "reloc"


def test_t8_shadow_attempt_never_changes(no_twoview, monkeypatch):
    """The shadow evaluation raising prints [RESID-ERROR] lines only."""
    monkeypatch.setattr(mt.residual_test, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("s")))
    s0, out0, snap0 = _shadow_run("m1b", "off")
    s1, out1, snap1 = _shadow_run("m1b", "log")
    assert s0 == s1 and snap0 == snap1
    assert [RX_MS.sub("refit_ms=T", l) for l in out1 if "[RESID-" not in l] == \
        [RX_MS.sub("refit_ms=T", l) for l in out0]
    errs = [l for l in out1 if "[RESID-ERROR]" in l]
    assert sum("kind=shadow" in l for l in errs) == 3 * s1.count("tracked")


# =========================================================================== T9 main
def _ns(**kw):
    d = dict(residual_test="off", tracking="chain", live_map="off", map_rot_state="off")
    d.update(kw)
    return argparse.Namespace(**d)


def _msg(args):
    try:
        main.validate_residual_args(args)
        return None
    except SystemExit as e:
        return str(e)


def test_t9_flag_and_refusals():
    pr = main.get_parser()
    assert pr.parse_args(["--dataset", "rgb"]).residual_test == "off"
    for v in ("off", "log", "on"):
        assert pr.parse_args(["--dataset", "rgb", "--residual_test", v]).residual_test == v
    for bad in ("ON", "yes", ""):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            pr.parse_args(["--dataset", "rgb", "--residual_test", bad])
    helps = {a.dest: a.help for a in pr._actions}
    assert "R63-1" in helps["residual_test"]
    V = "is only valid with --tracking map or --live_map on"
    for v in ("log", "on"):
        assert _msg(_ns(residual_test=v)) == f"--residual_test {v} {V}"
        assert _msg(_ns(residual_test=v, tracking="rotation")) == f"--residual_test {v} {V}"
        # (i) before (ii)
        assert _msg(_ns(residual_test=v, map_rot_state="on")) == f"--residual_test {v} {V}"
        assert _msg(_ns(residual_test=v, tracking="map", map_rot_state="on")) == \
            f"--residual_test {v} refuses --map_rot_state on"
        assert _msg(_ns(residual_test=v, tracking="rotation", live_map="on", map_rot_state="on")) == \
            f"--residual_test {v} refuses --map_rot_state on"
        for rs in ("off", "r52"):
            assert _msg(_ns(residual_test=v, tracking="map", map_rot_state=rs)) is None
            assert _msg(_ns(residual_test=v, tracking="rotation", live_map="on", map_rot_state=rs)) is None
    # getattr defaults: a bare namespace is off
    assert main.validate_residual_args(argparse.Namespace()) is None
    assert _msg(argparse.Namespace(residual_test="on")) == f"--residual_test on {V}"


def test_t9_off_is_none_in_every_combination():
    for tracking in ("chain", "map", "rotation"):
        for live in ("off", "on"):
            for rs in ("off", "on", "r52"):
                assert _msg(_ns(tracking=tracking, live_map=live, map_rot_state=rs)) is None


def test_t9_run_pipeline_order():
    import inspect
    src = inspect.getsource(main.run_pipeline)
    assert src.index("validate_ba_args(args)") < src.index("validate_residual_args(args)")
    assert src.index("validate_residual_args(args)") < src.index("write_resolved_settings(")


M0_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
           "--init", "twoview", "--live_map", "on", "--map_rot_state", "r52", "--pnp_interval", "0",
           "--no_loop_closure", "--no_relocalization", "--point_source", "depth_filter", "--df_promote_factor", "1.0",
           "--ba_outlier_reject", "on", "--ba_jump_gate", "off", "--ba_free_cameras", "supported"]
M1_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "map", "--m1_map", "/tmp/m1.txt",
           "--m1_mode", "m1a", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba",
           "--no_point_fusion", "--map_reloc", "pnp"]


def _capture_run_pipeline(monkeypatch, argv):
    seen = {}
    monkeypatch.setattr(main, "show_trajectory", lambda frame_iter, **kw: seen.update(kw))
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (K, 0, iter(())))
    monkeypatch.setattr(main, "load_timestamps_for_images", lambda *a, **k: [])
    with contextlib.redirect_stderr(io.StringIO()):
        args = main.get_parser().parse_args(main.front_end.expand_argv(argv))
    with contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(args)
    return seen


@pytest.mark.parametrize("value", ["off", "log", "on"])
@pytest.mark.parametrize("rot", ["r52", "off"])
def test_t9_factory_carries_the_attribute(monkeypatch, tmp_path, value, rot):
    argv = list(M0_ARGV) + ["--save_estimated", str(tmp_path / "estimated.txt")]
    argv[argv.index("--map_rot_state") + 1] = rot
    extra = [] if value == "off" else ["--residual_test", value]
    kw = _capture_run_pipeline(monkeypatch, argv + extra)
    tr = kw["open_tracker"].tracker_factory(mt.M1Map(header={}, points=[], corr={}))
    assert tr.residual_test == value and tr.live_map is True
    assert isinstance(tr, mr.MapRotationTracker) == (rot == "r52")


@pytest.mark.parametrize("value", ["off", "log", "on"])
def test_t9_m1_tracker_carries_the_attribute(monkeypatch, value):
    seen = {}
    real = mt.MapTracker

    class _Spy(real):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(main, "MapTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda p: mt.M1Map(header={}, points=[], corr={}))
    extra = [] if value == "off" else ["--residual_test", value]
    args = main.get_parser().parse_args(M1_ARGV + extra)
    with pytest.raises(SystemExit):
        main.run_pipeline(args)
    assert seen["tracker"].residual_test == value


def test_t9_refused_before_any_frame(monkeypatch):
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("iter_frames")))
    args = main.get_parser().parse_args(["--dataset", "rgb", "--residual_test", "log"])
    with pytest.raises(SystemExit) as e:
        main.run_pipeline(args)
    assert str(e.value) == "--residual_test log is only valid with --tracking map or --live_map on"
