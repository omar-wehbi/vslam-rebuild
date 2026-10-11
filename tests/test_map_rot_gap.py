"""Stage 04-r04 (ruling R65-4): the post-gap label rule of the r52 fallback behind --map_rot_gap.

T1 the parse; T2 validate_gap_args (and run_pipeline's call and wiring, spec A2/A3); T3 _post_gap;
T4 _gap_relabel; T5 the setup and summary lines; T6 the stay site (spec B5 (b)); T7 the entry site
(spec B5 (a)).

Imports only the standard library, numpy, pytest, main, map_rotation and map_tracker (spec D): the
segment's fit and the parent's tracking are stubbed through map_rotation and map_tracker attributes.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import map_rotation as mr  # noqa: E402
import map_tracker as mt  # noqa: E402

K = np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])
W, H = 640, 480

M1_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--tracking", "map", "--m1_map", "/tmp/m1.txt",
           "--m1_mode", "m1a", "--pnp_interval", "0", "--no_loop_closure", "--no_relocalization", "--no_ba",
           "--no_point_fusion", "--map_reloc", "pnp"]
M0_ARGV = ["--dataset", "rgb", "--default_k", "--undistort", "--front_end", "sift", "--tracking", "rotation",
           "--init", "twoview", "--live_map", "on", "--map_rot_state", "r52", "--pnp_interval", "0",
           "--no_loop_closure", "--no_relocalization", "--point_source", "depth_filter", "--df_promote_factor", "1.0",
           "--ba_outlier_reject", "on", "--ba_jump_gate", "off", "--ba_free_cameras", "supported",
           "--residual_test", "on"]
REFUSAL = "--map_rot_gap lost requires --map_rot_state r52"

P = r"^\s*>>\s+"
RX_GAP = re.compile(P + r"\[MAP-ROT-GAP\] frame=(\d+) event=(enter|stay) gap_s=(\d+\.\d{6}) anchor=(-?\d+) "
                    r"kf=(-?\d+)$")
RX_ROT = re.compile(P + r"\[MAP-ROT\] frame=(\d+) event=(enter|stay|exit|refused) label=(bridged|rotation|lost|tracked) "
                    r"anchor=(-?\d+) age=(-?\d+) kf=(-?\d+) rot=(measured|unsupported) exit=(none|track|reloc) "
                    r"track_inliers=(-?\d+) twist=(\S+),(\S+),(\S+)\s*$")


def _parse(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        return main.get_parser().parse_args(main.front_end.expand_argv(argv))


def _refusal(args):
    try:
        main.validate_gap_args(args)
        return None
    except SystemExit as e:
        return str(e)


def _empty_map():
    return mt.M1Map(header={}, points=[], corr={})


# --------------------------------------------------------------------------- T1
def test_t1_parse_default_values_and_refusal():
    assert _parse(["--dataset", "rgb"]).map_rot_gap == "carry"
    for v in ("carry", "lost"):
        assert _parse(["--dataset", "rgb", "--map_rot_gap", v]).map_rot_gap == v
    for bad in ("hold", "LOST", "on", ""):
        with pytest.raises(SystemExit):
            _parse(["--dataset", "rgb", "--map_rot_gap", bad])


# --------------------------------------------------------------------------- T2
def test_t2_validate_gap_args():
    # carry: None for every map_rot_state (and with nothing set at all)
    for state in ("off", "on", "r52"):
        assert _refusal(types.SimpleNamespace(map_rot_gap="carry", map_rot_state=state)) is None
    assert _refusal(types.SimpleNamespace()) is None
    assert _refusal(_parse(["--dataset", "rgb"])) is None
    # lost on r52: with --tracking map, and with --live_map on and --tracking rotation
    assert _refusal(_parse(M1_ARGV + ["--map_rot_state", "r52", "--map_rot_gap", "lost"])) is None
    assert _refusal(_parse(M0_ARGV + ["--map_rot_gap", "lost"])) is None
    # lost elsewhere: the exact A2 message
    assert _refusal(_parse(["--dataset", "rgb", "--map_rot_gap", "lost"])) == REFUSAL
    assert _refusal(_parse(M1_ARGV + ["--map_rot_state", "on", "--map_rot_gap", "lost"])) == REFUSAL
    assert _refusal(types.SimpleNamespace(map_rot_gap="lost", map_rot_state="off")) == REFUSAL
    assert _refusal(types.SimpleNamespace(map_rot_gap="lost", map_rot_state="on")) == REFUSAL
    assert _refusal(types.SimpleNamespace(map_rot_gap="lost")) == REFUSAL


def test_t2_run_pipeline_calls_it_after_the_residual_rules(monkeypatch):
    def unreached(*a, **k):
        raise AssertionError("iter_frames reached before the refusal")
    monkeypatch.setattr(main, "iter_frames", unreached)
    with pytest.raises(SystemExit) as ex:
        main.run_pipeline(_parse(["--dataset", "rgb", "--map_rot_gap", "lost"]))
    assert str(ex.value) == REFUSAL
    # an existing rule still refuses first, with its own message
    with pytest.raises(SystemExit) as ex:
        main.run_pipeline(_parse(["--dataset", "rgb", "--residual_test", "on", "--map_rot_gap", "lost"]))
    assert str(ex.value) == "--residual_test on is only valid with --tracking map or --live_map on"


def test_t2_run_pipeline_hands_the_rule_to_the_m1_tracker(monkeypatch):
    seen = {}

    class _Spy(mr.MapRotationTracker):
        def check_header(self, *a, **k):
            seen["tracker"] = self
            raise SystemExit("stop")
    monkeypatch.setattr(mr, "MapRotationTracker", _Spy)
    monkeypatch.setattr(main, "load_m1_map", lambda path, **kw: _empty_map())
    for extra, want in (([], "carry"), (["--map_rot_gap", "carry"], "carry"), (["--map_rot_gap", "lost"], "lost")):
        with pytest.raises(SystemExit):
            main.run_pipeline(_parse(M1_ARGV + ["--map_rot_state", "r52"] + extra))
        t = seen.pop("tracker")
        assert t.variant == "r52" and t.gap_rule == want and t.n_gap_relabelled == 0
    # the 04v path never receives the rule (validate refuses lost; carry leaves the default)
    with pytest.raises(SystemExit):
        main.run_pipeline(_parse(M1_ARGV + ["--map_rot_state", "on"]))
    t = seen.pop("tracker")
    assert t.variant == "on" and t.gap_rule == "carry"


def _capture_run_pipeline(monkeypatch, argv, tmp_path):
    seen = {}

    def fake_show(frame_iter, **kw):
        seen.update(kw)
    monkeypatch.setattr(main, "show_trajectory", fake_show)
    monkeypatch.setattr(main, "iter_frames", lambda *a, **k: (K, 0, iter(())))
    monkeypatch.setattr(main, "load_timestamps_for_images", lambda *a, **k: [])
    with contextlib.redirect_stdout(io.StringIO()):
        main.run_pipeline(_parse(argv + ["--save_estimated", str(tmp_path / "estimated.txt")]))
    return seen


@pytest.mark.parametrize("extra,want", [([], "carry"), (["--map_rot_gap", "lost"], "lost")])
def test_t2_run_pipeline_hands_the_rule_to_the_m0_factory(monkeypatch, tmp_path, extra, want):
    kw = _capture_run_pipeline(monkeypatch, M0_ARGV + extra, tmp_path)
    tr = kw["open_tracker"].tracker_factory(_empty_map())
    assert type(tr) is mr.MapRotationTracker and tr.variant == "r52" and tr.gap_rule == want
    assert tr.residual_test == "on" and tr.live_map is True
    assert json.load(open(tmp_path / "resolved_settings.json"))["map_rot_gap"] == want


# --------------------------------------------------------------------------- T3
def test_t3_post_gap():
    tr = mr.MapRotationTracker(_empty_map(), "m1a")
    assert mr.POST_GAP_S == 0.150
    assert tr._post_gap(1) is False                      # timestamps None
    # intervals 0.03125, 0.15625, 0.140625 s (exact in binary)
    tr.timestamps = [0.0, 0.03125, 0.1875, 0.328125]
    assert [tr._post_gap(k) for k in range(4)] == [False, False, True, False]
    assert tr._post_gap(4) is False and tr._post_gap(-1) is False
    # strictly over 0.150: an interval of exactly 0.150 is not a gap (as jumps.classify_pairs)
    tr.timestamps = [0.0, 0.15]
    assert tr._post_gap(1) is False


# --------------------------------------------------------------------------- T4
def _probe_frames(n=4):
    W0 = np.eye(4)
    W0[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    W0[:3, 3] = [0.1, 0.2, 0.3]
    C0 = np.linalg.inv(W0)
    return [types.SimpleNamespace(pose_valid=True, tracking_state="ROT", cam_T_world=C0.copy(),
                                  world_T_cam=W0.copy()) for _ in range(n)], C0, W0


def _probe_tracker(rule):
    tr = mr.MapRotationTracker(_empty_map(), "m1a")
    tr.variant = "r52"
    tr.gap_rule = rule
    tr.timestamps = [0.0, 0.03125, 0.1875, 0.328125]
    tr._anchor = 1
    tr._seg = types.SimpleNamespace(kf=1)
    return tr


def test_t4_gap_relabel_under_carry_is_a_no_op():
    frs, C0, W0 = _probe_frames()
    tr = _probe_tracker("carry")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr._gap_relabel(2, frs, "stay") is False
        assert tr._gap_relabel(3, frs, "stay") is False
    assert buf.getvalue() == "" and tr.n_gap_relabelled == 0
    assert all(f.tracking_state == "ROT" and f.pose_valid is True for f in frs)
    assert all(np.array_equal(f.cam_T_world, C0) and np.array_equal(f.world_T_cam, W0) for f in frs)
    # the attribute missing reads as carry
    del tr.gap_rule
    assert tr._gap_relabel(2, frs, "stay") is False


def test_t4_gap_relabel_under_lost():
    frs, C0, W0 = _probe_frames()
    tr = _probe_tracker("lost")
    seg = tr._seg
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr._gap_relabel(3, frs, "stay") is False          # an ordinary frame (0.140625 s)
        assert tr._gap_relabel(1, frs, "stay") is False
        assert tr._gap_relabel(2, frs, "stay") is True           # the post-gap frame (0.15625 s)
    lines = buf.getvalue().splitlines()
    assert lines == ["  >> [MAP-ROT-GAP] frame=2 event=stay gap_s=0.156250 anchor=1 kf=1"]
    assert RX_GAP.match(lines[0]).groups() == ("2", "stay", "0.156250", "1", "1")
    assert frs[2].tracking_state == "LOST" and frs[2].pose_valid is False
    assert frs[3].tracking_state == "ROT" and frs[1].tracking_state == "ROT"
    assert np.array_equal(frs[2].cam_T_world, C0) and np.array_equal(frs[2].world_T_cam, W0)
    assert tr.n_gap_relabelled == 1 and tr._anchor == 1 and tr._seg is seg and seg.kf == 1
    # anchor and kf print -1 when absent
    tr._anchor, tr._seg = None, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tr._gap_relabel(2, frs, "enter") is True
    assert buf.getvalue() == "  >> [MAP-ROT-GAP] frame=2 event=enter gap_s=0.156250 anchor=-1 kf=-1\n"
    tr._seg = types.SimpleNamespace(kf=None)
    with contextlib.redirect_stdout(io.StringIO()) as b2:
        tr._gap_relabel(2, frs, "stay")
    assert b2.getvalue().endswith("anchor=-1 kf=-1\n") and tr.n_gap_relabelled == 3


# --------------------------------------------------------------------------- T5
def test_t5_setup_and_summary_lines():
    tr = mr.MapRotationTracker(_empty_map(), "m1a")
    assert tr.gap_rule == "carry" and tr.n_gap_relabelled == 0
    tr.variant = "r52"
    s0, m0 = tr.setup_line(), tr.summary_line()
    assert "GAP" not in s0 and "GAP" not in m0
    assert s0.split("\n")[-1].startswith("  >> [MAP-ROT-R52-SETUP] ")
    assert m0.split("\n")[-1].startswith("  >> [MAP-ROT-R52-SUMMARY] ")
    tr.gap_rule = "lost"
    assert tr.setup_line() == s0 + "\n  >> [MAP-ROT-GAP-SETUP] rule=lost post_gap_s=0.150000"
    assert tr.summary_line() == m0 + "\n  >> [MAP-ROT-GAP-SUMMARY] relabelled=0"
    tr.n_gap_relabelled = 5
    assert tr.summary_line().endswith("\n  >> [MAP-ROT-GAP-SUMMARY] relabelled=5")
    tr.gap_rule = "carry"
    assert tr.setup_line() == s0 and tr.summary_line() == m0


# --------------------------------------------------------------------------- stub scene for T6, T7
def _Ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class _F:
    def __init__(self, fid, T, state):
        self.id = fid
        self.K = K
        self.img_shape = (H, W)
        self.keypoint_to_point = {}
        self.cam_T_world = np.array(T, dtype=np.float64)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        self.tracking_state = state
        self.pose_valid = state == "OK"

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64, copy=True)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


class _Pt:
    def __init__(self, pid, X):
        self.id = pid
        self.x, self.y, self.z = (float(v) for v in X)


class _PtMap:
    def __init__(self):
        self.points = []
        self.point_by_id = {}

    def add(self, X):
        p = _Pt(1000 + len(self.points), X)
        self.points.append(p)
        self.point_by_id[p.id] = p
        return p

    def link(self, frame, kp, point, **kw):
        frame.keypoint_to_point[int(kp)] = point.id
        return True


C0 = np.array([0.3, -0.1, 1.2])
DT = 0.03125
GAP = 0.15625


def _scene(monkeypatch, rule, gap_before, n=30, e=20):
    """Frames 0..e tracked, turning 1 deg per frame about y around C0 (rotation without parallax: the
    entry is detected), the anchor e with three linked points at depths 2, 3, 4. The rgb interval before
    frame gap_before is GAP s, every other one DT s. The parent tracker always loses the frame, normal
    tracking never accepts, and the segment's fit is measured: a 1 deg yaw step about the panorama
    keyframe's centre."""
    frames = []
    for k in range(n):
        if k <= e:
            Wk = np.eye(4)
            Wk[:3, :3] = _Ry(1.0 * k)
            Wk[:3, 3] = C0
            frames.append(_F(k + 1, np.linalg.inv(Wk), "OK"))
        else:
            frames.append(_F(k + 1, np.eye(4), "LOST"))
    times, t = [], 0.0
    for k in range(n):
        if k:
            t += GAP if k == gap_before else DT
        times.append(t)
    tr = mr.MapRotationTracker(_empty_map(), "m1a")
    tr.reloc = "pnp"
    tr.variant = "r52"
    tr.gap_rule = rule
    tr.timestamps = times
    wmap = _PtMap()
    Wf = frames[e].world_T_cam
    for i, z in enumerate((2.0, 3.0, 4.0)):
        Xc = np.array([0.1 * i - 0.2, 0.05 * i, z])
        wmap.link(frames[e], i, wmap.add(Wf[:3, :3] @ Xc + Wf[:3, 3]))
    tr._record_inliers(e, frames, wmap)

    def parent_on_frame(self_, k, frames_, world_map):
        T = np.eye(4)
        T[:3, :3] = _Ry(17.0)
        T[:3, 3] = [-5.0, 1.0, 2.0]
        frames_[k].set_camera_pose(T)
        frames_[k].pose_valid, frames_[k].tracking_state = False, "LOST"
        print(f"  >> [MAP-TRACK] frame={k} state=lost mode=m1a pred_rot=cv step_a=0 step_a_inliers=0 "
              f"step_a_wide=1 step_b=0 obs=0 inliers=0 rmse_px=nan")
        return "lost"

    def seg_on_frame(self_, k, frames_):
        c_P = np.asarray(frames_[self_.kf].world_T_cam)[:3, 3]
        R = _Ry(-1.0) @ np.asarray(frames_[k - 1].cam_T_world)[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ c_P
        frames_[k].set_camera_pose(T)
        frames_[k].pose_valid, frames_[k].tracking_state = True, "ROT"
        print(f"  >> [ROT-ONLY] frame={k} state=measured kf={self_.kf}")
        return "measured"

    def track_m1a(k, frames_, T_pred):
        counts = dict(step_a=0, step_a_inliers=0, step_a_wide=1, step_b=3)
        return None, [0, 1, 2], [0, 1, 2], np.zeros(3, dtype=bool), counts

    monkeypatch.setattr(mt.MapTracker, "on_frame", parent_on_frame)
    monkeypatch.setattr(mr.PanoramaSegment, "on_frame", seg_on_frame)
    monkeypatch.setattr(tr, "_track_m1a", track_m1a)
    return frames, tr, wmap


def _run(tr, frames, wmap, ks):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        states = [tr.on_frame(k, frames, wmap) for k in ks]
    return states, buf.getvalue().splitlines()


def _rot(out, k):
    hits = [i for i, l in enumerate(out) if (RX_ROT.match(l) and RX_ROT.match(l).group(1) == str(k))]
    assert len(hits) == 1, (k, hits)
    return hits[0], RX_ROT.match(out[hits[0]])


# --------------------------------------------------------------------------- T6
def test_t6_stay_site(monkeypatch):
    """Entry at 21 (ordinary), a post-gap stay frame 22 and an ordinary stay frame 23, rot=measured."""
    res = {}
    for rule in ("carry", "lost"):
        frames, tr, wmap = _scene(monkeypatch, rule, gap_before=22)
        states, out = _run(tr, frames, wmap, [21, 22, 23])
        res[rule] = (frames, tr, states, out)
    frames_c, tr_c, states_c, out_c = res["carry"]
    frames_l, tr_l, states_l, out_l = res["lost"]
    # carry: today's labels, no GAP line
    assert states_c == ["rotation", "rotation", "rotation"]
    assert not any("[MAP-ROT-GAP" in l for l in out_c)
    assert frames_c[22].tracking_state == "ROT" and frames_c[22].pose_valid is True
    assert tr_c.n_gap_relabelled == 0 and tr_c.n_rotation == 3 and tr_c.n_lost_in_state == 0
    # lost: frame 22 relabelled, frame 23 not
    assert states_l == ["rotation", "lost", "rotation"]
    assert frames_l[22].tracking_state == "LOST" and frames_l[22].pose_valid is False
    assert frames_l[23].tracking_state == "ROT" and frames_l[23].pose_valid is True
    assert frames_l[21].tracking_state == "ROT"
    i22, m22 = _rot(out_l, 22)
    assert m22.group(2, 3, 4, 6, 7) == ("stay", "lost", "20", "20", "measured")
    assert RX_GAP.match(out_l[i22 - 1]).groups() == ("22", "stay", "0.156250", "20", "20")
    gap_lines = [l for l in out_l if "[MAP-ROT-GAP]" in l]
    assert len(gap_lines) == 1
    _, m23 = _rot(out_l, 23)
    assert m23.group(2, 3, 7) == ("stay", "rotation", "measured")
    _, m22c = _rot(out_c, 22)
    assert m22c.group(2, 3, 7) == ("stay", "rotation", "measured")
    assert tr_l.n_gap_relabelled == 1 and tr_l.n_rotation == 2 and tr_l.n_lost_in_state == 1
    assert tr_l._anchor == 20 and tr_l._seg.kf == 20
    assert tr_l.summary_line().endswith("\n  >> [MAP-ROT-GAP-SUMMARY] relabelled=1")
    # the rule changes the label only: every pose is bitwise the carry run's
    for k in range(len(frames_c)):
        assert np.array_equal(frames_c[k].cam_T_world, frames_l[k].cam_T_world), k
        assert np.array_equal(frames_c[k].world_T_cam, frames_l[k].world_T_cam), k
    # and every other line is the carry run's
    assert [l for l in out_l if "[MAP-ROT-GAP]" not in l] == [
        l.replace("frame=22 event=stay label=rotation", "frame=22 event=stay label=lost") for l in out_c]


# --------------------------------------------------------------------------- T7
def test_t7_entry_site(monkeypatch):
    """The entry frame 21 is post-gap; stay frame 22 is ordinary."""
    res = {}
    for rule in ("carry", "lost"):
        frames, tr, wmap = _scene(monkeypatch, rule, gap_before=21)
        states, out = _run(tr, frames, wmap, [21, 22])
        res[rule] = (frames, tr, states, out)
    frames_c, tr_c, states_c, out_c = res["carry"]
    frames_l, tr_l, states_l, out_l = res["lost"]
    assert states_c == ["rotation", "rotation"] and not any("[MAP-ROT-GAP" in l for l in out_c)
    assert states_l == ["lost", "rotation"]
    # the stint is entered all the same
    for tr in (tr_c, tr_l):
        assert tr._anchor == 20 and isinstance(tr._seg, mr.PanoramaSegment) and tr._seg.kf == 20
        assert tr.n_entered == 1 and tr.n_stints == 1 and tr.n_attempts == 1
    assert frames_l[21].tracking_state == "LOST" and frames_l[21].pose_valid is False
    assert frames_c[21].tracking_state == "ROT" and frames_c[21].pose_valid is True
    i21, m21 = _rot(out_l, 21)
    assert m21.group(2, 3, 4, 6, 7) == ("enter", "lost", "20", "20", "measured")
    _, m21c = _rot(out_c, 21)
    assert m21c.group(2, 3, 7) == ("enter", "rotation", "measured")
    i_gap = [i for i, l in enumerate(out_l) if "[MAP-ROT-GAP]" in l]
    i_entry = [i for i, l in enumerate(out_l) if "[MAP-ROT-ENTRY]" in l]
    assert len(i_gap) == 1 and len(i_entry) == 1 and i_gap[0] < i_entry[0] < i21
    assert RX_GAP.match(out_l[i_gap[0]]).groups() == ("21", "enter", "0.156250", "20", "20")
    assert out_l[i_entry[0]].endswith("detected=1 fit=measured result=enter")
    assert tr_l.n_gap_relabelled == 1 and tr_l.n_lost == 1 and tr_l.n_lost_in_state == 1
    assert tr_l.n_rotation == 1 and tr_c.n_rotation == 2
    _, m22 = _rot(out_l, 22)
    assert m22.group(2, 3) == ("stay", "rotation") and frames_l[22].tracking_state == "ROT"
    for k in range(len(frames_c)):
        assert np.array_equal(frames_c[k].cam_T_world, frames_l[k].cam_T_world), k
        assert np.array_equal(frames_c[k].world_T_cam, frames_l[k].world_T_cam), k
