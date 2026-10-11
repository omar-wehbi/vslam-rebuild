"""
Stage 04i (spec T): the R2 seed lifetime (depth_filter_lifetime.py, the r2 paths of
depth_filter_arm.py, the flags, the pipeline's on_end call) on the synthetic sweeps of
tests/test_depth_filter.py and the helpers of tests/test_local_mapping.py.  No dataset,
no pipeline run.

The clean sweep (test_depth_filter.sweep): K_0 at the origin, frames 1..n at x = 0.009 k,
300 points at 1.5 to 4.0 m, 1 px noise in every image, distinct random descriptors
identical across frames, keypoint index == point index; 12 landmarks linked in K_0 so
z_init comes from "links".
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betainc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import depth_filter as df  # noqa: E402
import depth_filter_arm as dfa  # noqa: E402
import depth_filter_lifetime as dfl  # noqa: E402
from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from depth_filter import DepthFilterParams, Measurement  # noqa: E402
from depth_filter_arm import DepthFilterArm, DepthFilterArmParams  # noqa: E402
from local_mapping import LocalMappingParams  # noqa: E402
from test_depth_filter import sweep  # noqa: E402
from test_depth_filter_arm import RX_FR, RX_KF, RX_PR, _capture, _link_true_landmarks, _run_sweep  # noqa: E402
from test_local_mapping import H, K, W, ZERO, _KF, _yaw  # noqa: E402

THETA_MIN_DEG = 1.1
DEFERRED_K = 9.5
EPS_MATCH = 9.5
N_DISTINCT = 3

# the harness formats (system/verify/checks/depth_filter_lifetime_score.py), anchored at both ends
_PR_BODY = RX_PR.pattern[len(r"^\s*>> \[DF-PROMOTE\] "):-1]
RX_CONFIRM = re.compile(r"^\s*>> \[DF-CONFIRM\] " + _PR_BODY +
                        r" reason=(early|dead|evicted|end) n_distinct=(\d+) last_inlier=(-?\d+) eligible_failures=(\d+) "
                        r"travel_since_match=(\S+) dormant_frames=(\d+) first_at_mean=([01])$")
RX_RETIRE = re.compile(r"^\s*>> \[DF-RETIRE\] ref=(\d+) kp=(\d+) u=(\S+) v=(\S+) level=(\d+) frame=(\d+) reason=(linked|hopeless|dead|evicted|end) "
                       r"converged=([01]) converged_at=(-?\d+) theta_max_deg=(\S+) theta_acc_deg=(\S+) widest=(-?\d+) n_meas=(\d+) n_inlier=(\d+) "
                       r"n_distinct=(\d+) inlier_ratio=(\S+) a=(\S+) b=(\S+) sigma_rel=(\S+) range=(\S+) z_ref=(\S+) last_inlier=(-?\d+) "
                       r"eligible_failures=(\d+) travel_since_match=(\S+) dormant_frames=(\d+) first_at_mean=([01])$")
RX_LT = re.compile(r"^\s*>> \[DF-LIFETIME\] frame=(\d+) live=(\d+) linked_dropped=(\d+) dormant=(\d+) skipped=(\d+) eligible=(\d+) matched=(\d+) "
                   r"failed=(\d+) degenerate=(\d+) hopeless=(\d+) dead_retired=(\d+) dead_confirmed=(\d+) tentative_new=(\d+) early_confirmed=(\d+) "
                   r"tentative=(\d+) first_at_mean=(\d+)$")
RX_LTKF = re.compile(r"^\s*>> \[DF-LIFETIME-KF\] kf=(\d+) live_before=(\d+) evicted_dormant=(\d+) evicted_active=(\d+) evicted_confirmed=(\d+) "
                     r"live_after=(\d+) active=(\d+) dormant=(\d+) tentative=(\d+) cap=(\d+) arm_seconds=(\S+)$")
RX_END = re.compile(r"^\s*>> \[DF-END\] frame=(\d+) live=(\d+) confirmed=(\d+) retired=(\d+)$")
NEW_FLAGS = dict(lifetime="age", bootstrap_window=5, match_error_px=9.5, death_travel_ratio=0.3, outlier_eta=0.05,
                 outlier_conf=0.99, distinct_baselines=3, seed_cap=25000, trace_path="")
LT_KEYS = ("frame", "live", "linked_dropped", "dormant", "skipped", "eligible", "matched", "failed", "degenerate", "hopeless",
           "dead_retired", "dead_confirmed", "tentative_new", "early_confirmed", "tentative", "first_at_mean")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _arm_r2(factor=1.0, **arm_kw):
    return DepthFilterArm(DepthFilterParams(), DepthFilterArmParams(promote_factor=factor, lifetime="r2", **arm_kw),
                          LocalMappingParams())


def _parse_r2(log):
    """Every [DF-...] line of ``log`` parsed with the harness formats; a line that does not match
    its format fails the test."""
    out = dict(kf=[], fr=[], pr=[], cf=[], rt=[], lt=[], ltkf=[], end=[])
    tags = (("[DF-KEYFRAME]", RX_KF, "kf"), ("[DF-FRAME]", RX_FR, "fr"), ("[DF-PROMOTE]", RX_PR, "pr"), ("[DF-CONFIRM]", RX_CONFIRM, "cf"),
            ("[DF-RETIRE]", RX_RETIRE, "rt"), ("[DF-LIFETIME-KF]", RX_LTKF, "ltkf"), ("[DF-LIFETIME]", RX_LT, "lt"), ("[DF-END]", RX_END, "end"))
    for line in log.splitlines():
        for tag, rx, key in tags:
            if tag in line:
                m = rx.match(line)
                assert m, f"a {tag} line does not match its format: {line!r}"
                out[key].append(m)
                break
        else:
            assert "[DF-ERROR]" not in line, line
    return out


def _lt(m):
    return dict(zip(LT_KEYS, [int(x) for x in m.groups()]))


def _assert_lifetime_identities(parsed):
    fr_live = {int(m.group(1)): int(m.group(2)) for m in parsed["fr"]}
    for m in parsed["lt"]:
        t = _lt(m)
        assert t["live"] == t["linked_dropped"] + t["dormant"] + t["skipped"] + t["eligible"], m.group(0)
        assert t["eligible"] == t["matched"] + t["failed"], m.group(0)
        assert fr_live[t["frame"]] == t["eligible"], m.group(0)
    for m in parsed["fr"]:
        g = [int(x) for x in m.groups()]
        assert g[2] == g[3] + g[4] and g[1] == g[2] + g[5] + g[6] + g[7] + g[8], m.group(0)
    for m in parsed["ltkf"]:
        g = [int(x) for x in m.groups()[:10]]
        assert g[5] == g[6] + g[7] and g[8] <= g[5], m.group(0)


def _run_sweep_r2(factor, n_frames=40, frames_override=None, arm=None, end=True, **arm_kw):
    """The clean sweep through the r2 driver: on_keyframe(0), on_frame(1..n), on_end(n).
    Returns (pts, frames, world, arm, kf_out, frame_outs, end_out, log)."""
    pts, poses, frames, _ = sweep(0.009, n_frames=n_frames, noise_px=1.0)
    if frames_override is not None:
        frames = frames_override(pts, poses, frames)
    K0 = frames[0]
    world = Map(frames=frames)
    _link_true_landmarks(world, K0, pts, range(12))
    arm = arm if arm is not None else _arm_r2(factor, **arm_kw)
    keyframes = [(0, K0)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kf_out = arm.on_keyframe(0, K0, keyframes, world, frames)
        outs = [arm.on_frame(k, frames[k], keyframes, world, frames) for k in range(1, n_frames + 1)]
        end_out = arm.on_end(n_frames, keyframes, world, frames) if end else None
    return pts, frames, world, arm, kf_out, outs, end_out, buf.getvalue()


def _rel_err(rec, pts):
    r_true = float(np.linalg.norm(pts[rec["kp"]]))
    return abs(rec["range"] - r_true) / r_true


def _confirm_records(outs, end_out):
    recs = [r for o in outs for r in o["promotions"]]
    if end_out:
        recs += end_out["promotions"]
    return recs


# --------------------------------------------------------------------------- #
# T1. parser and params
# --------------------------------------------------------------------------- #
def test_t1_flags_defaults_from_args_and_resolved_settings(tmp_path):
    import main
    parser = main.get_parser()
    a = parser.parse_args(["--dataset", "rgb"])
    A = DepthFilterArmParams()
    for f, want in NEW_FLAGS.items():
        assert getattr(A, f) == want, f
        assert getattr(a, "df_" + f) == getattr(A, f), f
        assert type(getattr(a, "df_" + f)) is type(getattr(A, f)), f
    assert a.df_promote_factor == 1.0 and a.df_bootstrap_depth_ratio == 5.0 and a.point_source == "keyframe_pair"
    assert a.lm_confirm is False and a.undistort is False
    assert DepthFilterArmParams.from_args(a) == A and A.lifetime == "age"
    b = parser.parse_args(["--dataset", "rgb", "--df_lifetime", "r2", "--df_promote_factor", "9.5", "--df_trace", "/tmp/x.txt"])
    B = DepthFilterArmParams.from_args(b)
    assert B.lifetime == "r2" and B.promote_factor == 9.5 and B.trace_path == "/tmp/x.txt"
    assert b.df_trace_path == "/tmp/x.txt"
    assert B.seed_cap == 25000 and B.distinct_baselines == 3 and B.match_error_px == 9.5 and B.bootstrap_window == 5
    assert DepthFilterParams.from_args(b) == DepthFilterParams()
    with pytest.raises(SystemExit):
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--dataset", "rgb", "--df_lifetime", "other"])
    out = tmp_path / "resolved_settings.json"
    with contextlib.redirect_stdout(io.StringIO()):
        main.write_resolved_settings(b, out)
    settings = json.loads(out.read_text())
    assert settings["df_lifetime"] == "r2" and settings["df_seed_cap"] == 25000 and settings["df_trace_path"] == "/tmp/x.txt"
    assert settings["df_promote_factor"] == 9.5 and settings["df_distinct_baselines"] == 3 and settings["df_match_error_px"] == 9.5
    assert settings["df_min_parallax_deg"] == 1.1 and settings["lm_confirm"] is False
    # the max_seed_keyframes flag still reaches the module params (age retirement only)
    c = parser.parse_args(["--dataset", "rgb", "--df_max_seed_keyframes", "3", "--df_bootstrap_window", "2"])
    assert DepthFilterParams.from_args(c).max_seed_keyframes == 3 and DepthFilterArmParams.from_args(c).bootstrap_window == 2


# --------------------------------------------------------------------------- #
# T2. the age path is unchanged
# --------------------------------------------------------------------------- #
def _module_reference(z_init, n_frames):
    """The 04f module driven by hand with the paper's timing (promote at the first frame
    promotable says ok) on a fresh copy of the sweep (same construction, same rng) with K_0's
    first 12 keypoints marked linked: the records and the per-frame counters the age driver
    must reproduce."""
    P = DepthFilterParams()
    pts, poses, frames, _ = sweep(0.009, n_frames=n_frames, noise_px=1.0)
    frames[0].keypoint_to_point = {i: -1 for i in range(12)}
    seeds = df.make_seeds(frames[0], 0, z_init, P)
    live = list(seeds)
    recs, counters = [], []
    for k in range(1, n_frames + 1):
        c = df.update_seeds(live, frames[k], k, P)
        counters.append(dict(c))
        keep = []
        conv = 0
        for s in live:
            ok, th_acc, th_max = df.promotable(s, P)
            if ok:
                conv += 1
                recs.append((int(s.kp), k, float(th_acc), float(th_max), 1.0 / float(s.mu)))
            else:
                keep.append(s)
        counters[-1]["converged"] = conv
        live = keep
    return len(seeds), recs, counters


def test_t2_age_path_unchanged_and_silent_on_the_new_lines():
    pts, frames, world, arm, kf_out, outs, log = _run_sweep(1.0)
    assert arm.arm_params.lifetime == "age" and arm.r2 is False
    for tok in ("[DF-LIFETIME]", "[DF-LIFETIME-KF]", "[DF-CONFIRM]", "[DF-RETIRE]", "[DF-END]"):
        assert tok not in log, tok
    n_seeds, ref_recs, ref_counters = _module_reference(kf_out["z_init"], 40)
    assert kf_out["seeds_made"] == n_seeds == 288
    got = [(r["kp"], r["frame"], r["theta_acc_deg"], r["theta_max_deg"], r["range"]) for o in outs for r in o["promotions"]]
    assert got == ref_recs and len(got) >= 0.8 * 288
    for o, c in zip(outs, ref_counters):
        assert o["live"] == c["seeds"] and o["converged"] == c["converged"] == o["promoted"]
        for key in ("measured", "inlier", "outlier", "no_match", "no_translation", "behind", "no_disparity", "outside_support"):
            assert o[key] == c[key], key
    # the values test_t3 of tests/test_depth_filter_arm.py asserts
    recs = [r for o in outs for r in o["promotions"]]
    assert all(r["converged_at"] == r["frame"] and r["theta_max_deg"] >= THETA_MIN_DEG and r["theta_acc_deg"] >= THETA_MIN_DEG for r in recs)
    assert np.median([_rel_err(r, pts) for r in recs]) <= 0.095
    # on_end is a no-op under age: prints nothing, returns {}, changes nothing
    n_live = len(arm.live)
    n_points = len(world.points)
    ret, log_end = _capture(arm.on_end, 40, [(0, frames[0])], world, frames)
    assert ret == {} and log_end == "" and len(arm.live) == n_live and len(world.points) == n_points
    assert arm.tentative_points() == []
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T3. classification, the failure test, the distinct-baseline count
# --------------------------------------------------------------------------- #
def _frame_at(fid, x, yaw_deg=0.0, pts=(), descs=None):
    T = np.eye(4)
    T[:3, :3] = _yaw(yaw_deg)
    T[:3, 3] = [x, 0.0, 0.0]
    return _KF(fid, T, list(pts), descs)


def _seeds_at(pts, z_init=2.5):
    """Seeds on K_0 (the identity pose) for ``pts``, with the driver's z_init attribute."""
    descs = np.zeros((len(pts), 32), dtype=np.uint8)
    K0 = _frame_at(100, 0.0, pts=pts, descs=descs)
    seeds = df.make_seeds(K0, 0, z_init, DepthFilterParams())
    for s in seeds:
        s.z_init = float(z_init)
    return K0, seeds


def test_t3_classification_dormant_skipped_eligible():
    A = DepthFilterArmParams(lifetime="r2")
    P = DepthFilterParams()
    pts = [np.array([0.0, 0.0, 2.5]), np.array([0.3, 0.1, 3.0]), np.array([-0.4, -0.2, 2.0])]
    K0, seeds = _seeds_at(pts)
    # a camera turned 90 deg: every credible segment leaves the image -> dormant
    st, d_skip, d_mean, fam = dfl.classify(seeds, _frame_at(1, 0.05, yaw_deg=90.0), A, P)
    assert st.tolist() == [dfl.STATE_DORMANT] * 3
    # a camera ahead of every point looking forward: the seeds are behind it -> dormant
    T_ahead = np.eye(4)
    T_ahead[2, 3] = 10.0
    st, _, _, _ = dfl.classify(seeds, _KF(2, T_ahead, [], None), A, P)
    assert st.tolist() == [dfl.STATE_DORMANT] * 3
    # a fresh seed (far end unbounded): the initial-mean disparity f b / z_init decides
    b_small = 0.01                       # 521 * 0.01 / 2.5 = 2.1 px < 9.5 -> skipped
    st, d_skip, d_mean, fam = dfl.classify(seeds, _frame_at(3, b_small), A, P)
    assert st.tolist() == [dfl.STATE_SKIPPED] * 3 and fam.tolist() == [True] * 3
    assert d_skip[0] == pytest.approx(521.0 * b_small / 2.5, rel=1e-6)
    b_big = 0.1                          # 20.8 px -> eligible, under the initial-mean reading
    st, d_skip, d_mean, fam = dfl.classify(seeds, _frame_at(4, b_big), A, P)
    assert st.tolist() == [dfl.STATE_ELIGIBLE] * 3 and fam.tolist() == [True] * 3
    assert d_skip[0] == pytest.approx(521.0 * b_big / 2.5, rel=1e-6) and d_mean[0] == pytest.approx(d_skip[0], rel=1e-6)
    # a measured seed with a finite far end: skipped when d_far < 9.5 although the mean's disparity is above
    s = seeds[0]
    s.mu = 1.0 / 2.5
    s.sigma2 = (0.1 * s.mu) ** 2          # far end at rho = 0.8 mu: range 3.125 m
    st, d_skip, d_mean, fam = dfl.classify([s], _frame_at(5, 0.05), A, P)
    assert d_mean[0] == pytest.approx(521.0 * 0.05 / 2.5, rel=1e-6) and d_mean[0] > EPS_MATCH
    assert d_skip[0] == pytest.approx(521.0 * 0.05 / 3.125, rel=1e-6) and d_skip[0] < EPS_MATCH
    assert st.tolist() == [dfl.STATE_SKIPPED] and fam.tolist() == [False]
    st, d_skip, _, fam = dfl.classify([s], _frame_at(6, 0.07), A, P)      # 11.7 px at the far end -> eligible
    assert st.tolist() == [dfl.STATE_ELIGIBLE] and fam.tolist() == [False] and d_skip[0] > EPS_MATCH
    # segment_in_image on plain segments
    ok = np.array([True, True, True, True])
    pa = np.array([[-10.0, 10.0], [-10.0, -10.0], [10.0, 10.0], [700.0, 10.0]])
    pb = np.array([[10.0, 10.0], [-5.0, -5.0], [20.0, 20.0], [W + 5.0, 10.0]])
    assert dfl.segment_in_image(pa, pb, ok, W, H).tolist() == [True, False, True, False]
    assert dfl.segment_in_image(pa[:1], pb[:1], np.array([False]), W, H).tolist() == [False]
    # the disparity is unbounded when the infinity point is behind the camera
    assert np.isposinf(dfl.predicted_disparity(np.array([[1.0, 2.0]]), np.array([[np.nan, np.nan]]), np.array([False])))[0]


def test_t3_hopeless_matches_the_beta_cdf():
    assert dfl.hopeless(10, 10, 0.05, 0.99) is False
    assert dfl.hopeless(10, 320, 0.05, 0.99) is False
    assert dfl.hopeless(10, 400, 0.05, 0.99) is True
    assert dfl.hopeless(40, 1000, 0.05, 0.99) is False
    assert betainc(10, 400, 0.05) >= 0.99 > betainc(10, 320, 0.05)
    assert 0.94 < betainc(10, 320, 0.05) < 0.99 and 0.95 < betainc(40, 1000, 0.05) < 0.99
    assert dfl.hopeless(10, 320, 0.05, 0.96) is True                      # the orientation of the CDF


def _seed_with_inliers(baselines_m, z=2.5, extra_frames=()):
    """A seed on the ray (0, 0, 1) at range z with one inlier measurement per lateral baseline
    (frames at x = b, identity rotation) plus ``extra_frames`` (frame objects)."""
    K0, seeds = _seeds_at([np.array([0.0, 0.0, z])], z_init=z)
    s = seeds[0]
    s.mu = 1.0 / z
    s.sigma2 = (0.02 * s.mu) ** 2
    for i, b in enumerate(baselines_m):
        fr = _frame_at(200 + i, b)
        s.measurements.append(Measurement(100 + i, fr, 0, s.mu, 1e-6, 0.9, 10.0))
    for j, fr in enumerate(extra_frames):
        s.measurements.append(Measurement(300 + j, fr, 0, s.mu, 1e-6, 0.9, 10.0))
    s.measurements.append(Measurement(999, _frame_at(999, 0.5), 0, s.mu, 1e-6, 0.1, 40.0))    # an outlier: never counted
    s.n_updates = len(s.measurements)
    return s


def test_t3_distinct_baselines_counts_gaps_of_match_error_px():
    A = DepthFilterArmParams(lifetime="r2")
    P = DepthFilterParams()
    px = lambda b: 521.0 * b / 2.5
    s = _seed_with_inliers([0.05, 0.05 + 3.0 * 2.5 / 521.0])            # 3 px apart -> 1
    n, disp = dfl.distinct_baselines(s, P, A)
    assert n == 1 and len(disp) == 2 and disp[0] == pytest.approx(px(0.05), rel=1e-6)
    s = _seed_with_inliers([0.05, 0.05 + 10.0 * 2.5 / 521.0])           # 10 px apart -> 2
    assert dfl.distinct_baselines(s, P, A)[0] == 2
    s = _seed_with_inliers([0.05, 0.05 + 10.0 * 2.5 / 521.0, 0.05 + 20.0 * 2.5 / 521.0, 0.05 + 25.0 * 2.5 / 521.0])
    assert dfl.distinct_baselines(s, P, A)[0] == 3
    # a camera behind the point looking back at it: the infinity point is behind the camera, +inf counts once
    T = np.eye(4)
    T[:3, :3] = _yaw(180.0)
    T[:3, 3] = [0.0, 0.0, 5.0]
    back = _KF(400, T, [], None)
    s = _seed_with_inliers([0.05, 0.05 + 10.0 * 2.5 / 521.0], extra_frames=[back, back])
    n, disp = dfl.distinct_baselines(s, P, A)
    assert n == 3 and np.isposinf(disp[2]) and np.isposinf(disp[3])
    assert dfl.count_distinct([1.0, 2.0, 12.0, np.inf, np.inf, np.nan], 9.5) == 3
    assert dfl.count_distinct([], 9.5) == 0
    assert dfl.distinct_counts(np.array([1.0, 12.0, 5.0, np.inf, 30.0]), np.array([0, 0, 1, 1, 0]), 3, 9.5).tolist() == [3, 2, 0]
    # converged: promotable's test AND the count
    ok, th_acc, th_max, nd = dfl.converged(s, P, A)
    assert nd == 3 and th_max > 0.0
    A2 = DepthFilterArmParams(lifetime="r2", distinct_baselines=4)
    assert dfl.converged(s, P, A2)[0] is False and dfl.converged(s, P, A2)[3] == 3
    # dead: travel against z_init, pose-only
    s.last_match_centre = np.zeros(3)
    assert dfl.dead(s, np.array([0.3 * 2.5 - 1e-6, 0.0, 0.0]), A) is False
    assert dfl.dead(s, np.array([0.3 * 2.5, 0.0, 0.0]), A) is True
    assert dfl.travel_since_match(s, np.array([0.0, 0.4, 0.0])) == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# T4. the R2 reference timing on the clean sweep
# --------------------------------------------------------------------------- #
def test_t4_reference_timing_confirms_at_convergence_with_three_distinct_baselines(monkeypatch):
    # a spy on the classifier: every per-frame classification the driver made, for the skip rule below
    classified = []
    orig_classify = dfl.classify

    def spy(seeds, frame, params_arm, params_df, segments=None):
        state, d_skip, d_mean, fam = orig_classify(seeds, frame, params_arm, params_df, segments=segments)
        far_unbounded = segments["far_unbounded"] if segments is not None else dfl.credible_segments(seeds, frame, 2.0)["far_unbounded"]
        classified.append((int(frame.id) - 100, state.copy(), d_skip.copy(), np.asarray(far_unbounded, dtype=bool).copy()))
        return state, d_skip, d_mean, fam

    monkeypatch.setattr(dfa.dfl, "classify", spy)
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(1.0)
    monkeypatch.setattr(dfa.dfl, "classify", orig_classify)
    _, frames3, _, _, _, outs3, _ = _run_sweep(1.0)
    K0 = frames[0]
    p = _parse_r2(log)
    assert kf_out["seeds_made"] == 288 and kf_out["z_source"] == "links" and len(p["pr"]) == 0
    assert len(p["lt"]) == 40 and len(p["fr"]) == 40 and len(p["ltkf"]) == 1 and len(p["end"]) == 1
    _assert_lifetime_identities(p)
    recs = _confirm_records(outs, end_out)
    early = [r for r in recs if r["reason"] == "early"]
    assert len(p["cf"]) == len(recs) == len(world.points) - 12
    assert len(early) == sum(o["promoted"] for o in outs)
    for r in early:
        assert r["n_distinct"] >= N_DISTINCT and r["theta_max_deg"] >= THETA_MIN_DEG and r["theta_acc_deg"] >= THETA_MIN_DEG
        assert r["converged_at"] == r["frame"]
        assert abs(r["theta_max_driver_deg"] - r["theta_max_deg"]) <= 1e-6
        assert r["travel_since_match"] == 0.0 and r["eligible_failures"] == 0
    for m in p["cf"]:
        assert int(m.group(22)) >= N_DISTINCT and float(m.group(10)) >= THETA_MIN_DEG
        assert m.group(21) in ("early", "end")
    # the R2 reference timing confirms LATER than the age path promotes (N baselines >= 9.5 px apart)
    med_r2 = np.median([r["frame"] for r in early])
    med_age = np.median([r["frame"] for o in outs3 for r in o["promotions"]])
    assert med_r2 > med_age, (med_r2, med_age)
    assert len(recs) >= 0.6 * 288, len(recs)
    # the skip rule, per seed and per frame: a seed is skipped iff its skip disparity is below 9.5 px, and no
    # seed whose far end is finite and whose far-end disparity is >= 9.5 px is skipped.  (On this sweep the
    # mixture update leaves sigma_rel near 0.47 after two measurements at responsibility 0.7 to 0.8, so the far
    # end of many seeds sits near infinity and they ARE skipped for a while past frame 20: the far-end reading
    # of the contract, not a defect of the classifier.)
    assert len(classified) == 40 and [c[0] for c in classified] == list(range(1, 41))
    n_skipped_finite = 0
    for k, state, d_skip, far_unbounded in classified:
        skipped = state == dfl.STATE_SKIPPED
        eligible = state == dfl.STATE_ELIGIBLE
        assert not np.any(state == dfl.STATE_DORMANT)
        assert np.all(d_skip[skipped] < EPS_MATCH) and np.all(d_skip[eligible] >= EPS_MATCH), k
        assert not np.any(skipped & ~far_unbounded & (d_skip >= EPS_MATCH)), k
        n_skipped_finite += int(np.sum(skipped & ~far_unbounded))
    assert n_skipped_finite > 0                                        # the finite-far-end skip branch was exercised
    assert sum(_lt(m)["first_at_mean"] for m in p["lt"]) > 0
    # precision, provenance, invariants, accounting
    assert np.median([_rel_err(r, pts) for r in recs]) <= 0.095
    for r in recs:
        pt = r["point"]
        assert world.get_point(pt.id) is pt
        assert pt.observations[K0.id] == r["kp"] and all(kp == r["kp"] for kp in pt.observations.values())
        assert all(frames[fid - 100].keypoint_to_point[r["kp"]] == pt.id for fid in pt.observations)
        assert pt.created_fid == K0.id
    assert world.check_invariants(frames) == ZERO
    end = [int(x) for x in p["end"][0].groups()]
    assert end == [40, end_out["live"], end_out["confirmed"], end_out["retired"]] and end[1] == end[2] + end[3]
    assert kf_out["seeds_made"] == len(p["cf"]) + len(p["rt"])
    assert arm.live == [] and arm.tentative_points() == [] and arm.ended
    assert arm.totals["df_promoted"] == len(recs) and arm.totals["df_retired"] == len(p["rt"])
    # each confirm line precedes its frame's [DF-LIFETIME] line, which precedes the [DF-FRAME] line
    for r in early:
        line_c = next(m.group(0) for m in p["cf"] if int(m.group(1)) == r["id"])
        line_l = next(m.group(0) for m in p["lt"] if int(m.group(1)) == r["frame"])
        line_f = next(m.group(0) for m in p["fr"] if int(m.group(1)) == r["frame"])
        assert log.index(line_c) < log.index(line_l) < log.index(line_f)


# --------------------------------------------------------------------------- #
# T5. the deferred timing: tentative seeds, handles, end-of-run confirmation
# --------------------------------------------------------------------------- #
def test_t5_deferred_timing_tentative_handles_and_end_confirmations():
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(DEFERRED_K, end=False)
    pts1, _, _, _, _, outs1, end1, _ = _run_sweep_r2(1.0)
    p = _parse_r2(log)
    _assert_lifetime_identities(p)
    bar = DEFERRED_K * THETA_MIN_DEG
    early = [r for o in outs for r in o["promotions"]]
    assert len(early) > 0
    for r in early:
        assert r["reason"] == "early" and r["theta_acc_deg"] >= bar and r["n_distinct"] >= N_DISTINCT and r["frame"] >= r["converged_at"]
    first_early = min(r["frame"] for r in early)
    tent = {int(m.group(1)): _lt(m)["tentative"] for m in p["lt"]}
    assert max(tent[k] for k in range(1, first_early)) > 0 and tent[first_early - 1] >= 1
    # the handles: the current posterior of every tentative seed, as many as the line says, none a landmark
    handles = arm.tentative_points()
    assert len(handles) == tent[40] > 0
    P = DepthFilterParams()
    for X, sigma_rel, sigma2, s in handles:
        assert s.tentative and s in arm.live
        np.testing.assert_allclose(X, df.seed_point(s))
        assert sigma2 == s.sigma2 and sigma_rel == pytest.approx(math.sqrt(s.sigma2) / s.mu)
        assert s.kp not in frames[0].keypoint_to_point            # not a Map landmark
        assert dfl.converged(s, P, arm.arm_params)[0]
    linked_ids = {frames[0].keypoint_to_point[i] for i in range(12)}
    assert {pt.id for pt in world.points} == {r["point"].id for r in early} | linked_ids
    # the end of the run confirms the remaining converged seeds
    n_tent = len(handles)
    end_out, log_end = _capture(arm.on_end, 40, [(0, frames[0])], world, frames)
    pe = _parse_r2(log_end)
    assert len(pe["end"]) == 1 and end_out["confirmed"] == n_tent == len(end_out["promotions"])
    for r in end_out["promotions"]:
        assert r["reason"] == "end" and r["n_distinct"] >= N_DISTINCT and r["theta_max_deg"] >= THETA_MIN_DEG and r["frame"] == 40
        assert r["converged_at"] <= 40 and r["created_kf"] == frames[0].id - 1 and r["ref"] == frames[0].id - 1
    assert all(m.group(7) == "end" and m.group(8) == "0" for m in pe["rt"])
    assert end_out["live"] == end_out["confirmed"] + end_out["retired"] and arm.live == [] and arm.tentative_points() == []
    assert kf_out["seeds_made"] == len(p["cf"]) + len(pe["cf"]) + len(p["rt"]) + len(pe["rt"])
    recs = early + end_out["promotions"]
    assert len(world.points) - 12 == len(recs)
    err_def = np.median([_rel_err(r, pts) for r in recs])
    err_ref = np.median([_rel_err(r, pts1) for r in _confirm_records(outs1, end1)])
    assert err_def <= err_ref
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T6. dormancy, revisit, eviction
# --------------------------------------------------------------------------- #
def _look_away(k_from=10, k_to=20):
    def override(pts, poses, frames):
        out = list(frames)
        for k in range(k_from, k_to + 1):
            T = np.array(poses[k], dtype=float)
            T[:3, :3] = _yaw(90.0)
            out[k] = _KF(100 + k, T, [], None)
        return out
    return override


def test_t6_dormancy_counts_nothing_and_the_seeds_measure_again_on_return():
    arm = _arm_r2(1.0)
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(1.0, n_frames=30, frames_override=_look_away(), arm=arm, end=False)
    p = _parse_r2(log)
    _assert_lifetime_identities(p)
    lts = {int(m.group(1)): _lt(m) for m in p["lt"]}
    for k in range(10, 21):
        t = lts[k]
        assert t["dormant"] == t["live"] > 0 and t["eligible"] == t["skipped"] == t["matched"] == t["failed"] == 0, t
        assert t["hopeless"] == t["dead_retired"] == t["dead_confirmed"] == 0
    assert lts[9]["dormant"] == 0 and lts[21]["dormant"] == 0 and lts[21]["matched"] > 0 and lts[22]["matched"] > 0
    for s in arm.live:
        assert s.state == "active" and s.dormant_frames == 11 and s.dormant_since is None
    # b and eligible_failures did not move during dormancy: replay the sweep and snapshot them at frame 9
    arm2 = _arm_r2(1.0)
    pts2, poses2, frames2, _ = sweep(0.009, n_frames=30, noise_px=1.0)
    frames2 = _look_away()(pts2, poses2, frames2)
    world2 = Map(frames=frames2)
    _link_true_landmarks(world2, frames2[0], pts2, range(12))
    keyframes = [(0, frames2[0])]
    with contextlib.redirect_stdout(io.StringIO()):
        arm2.on_keyframe(0, frames2[0], keyframes, world2, frames2)
        for k in range(1, 10):
            arm2.on_frame(k, frames2[k], keyframes, world2, frames2)
        snap = {id(s): (s.b, s.eligible_failures, s.n_updates, s.dormant_frames, s.state) for s in arm2.live}
        for k in range(10, 21):
            arm2.on_frame(k, frames2[k], keyframes, world2, frames2)
            for s in arm2.live:
                b0, ef0, nu0, df0, _ = snap[id(s)]
                assert s.b == b0 and s.eligible_failures == ef0 and s.n_updates == nu0
                assert s.state == "dormant" and s.dormant_since == 10 and s.dormant_frames == df0 + (k - 9)
    assert len(arm2.live) == len(snap)
    assert world.check_invariants(frames) == ZERO


def test_t6_eviction_order_and_the_cap():
    # the order: dormant (longest first), active (longest since last match), then tentative
    _, seeds = _seeds_at([np.array([0.1 * i, 0.0, 2.5]) for i in range(6)])
    for i, s in enumerate(seeds):
        DepthFilterArm._init_seed_state(s, 0, 2.5)
    seeds[0].state = "dormant"; seeds[0].dormant_since = 7
    seeds[1].state = "dormant"; seeds[1].dormant_since = 3
    seeds[2].last_match_traj = 9
    seeds[3].last_match_traj = 4
    seeds[4].tentative = True; seeds[4].last_match_traj = 1
    seeds[5].tentative = True; seeds[5].state = "dormant"; seeds[5].dormant_since = 2
    order = dfl.eviction_order(seeds, 12)
    assert [seeds.index(s) for s in order] == [1, 0, 3, 2, 5, 4]
    # the cap: two keyframes, the second on_keyframe evicts the excess among the FIRST keyframe's seeds in
    # eviction_order; a tentative evicted seed is confirmed and exported.  The cap is set so that the excess
    # is a strict subset of the older seeds: the scenario is run once with no cap in force to read the two
    # populations (the run is deterministic), then replayed with cap = live_before - N_EVICT.
    N_EVICT = 100      # more than the older seeds that are not tentative, so the eviction reaches the tentative ones

    def _cap_scenario(cap):
        pts, poses, frames, _ = sweep(0.009, n_frames=30, noise_px=1.0)
        K0 = frames[0]
        world = Map(frames=frames)
        _link_true_landmarks(world, K0, pts, range(12))
        arm = _arm_r2(DEFERRED_K, seed_cap=cap)
        keyframes = [(0, K0)]
        with contextlib.redirect_stdout(io.StringIO()):
            out0 = arm.on_keyframe(0, K0, keyframes, world, frames)
            for k in range(1, 30):
                arm.on_frame(k, frames[k], keyframes, world, frames)
        old = list(arm.live)
        keyframes.append((30, frames[30]))
        out1, log1 = _capture(arm.on_keyframe, 30, frames[30], keyframes, world, frames)
        return pts, frames, K0, world, arm, keyframes, out0, old, out1, log1

    *_, out0_m, old_m, out1_m, log1_m = _cap_scenario(10 ** 9)
    live_before_m = int(_parse_r2(log1_m)["ltkf"][0].group(2))
    assert out0_m["seeds_made"] == 288 and out1_m["seeds_made"] > 100
    assert "reason=evicted" not in log1_m and live_before_m == len(old_m) - out1_m["retired_linked"] + out1_m["seeds_made"]
    CAP = live_before_m - N_EVICT
    n_old_live = len(old_m) - out1_m["retired_linked"]
    n_tent_m = int(_parse_r2(log1_m)["ltkf"][0].group(9))
    assert n_old_live - n_tent_m < N_EVICT < n_old_live           # a strict subset, reaching into the tentative seeds

    pts, frames, K0, world, arm, keyframes, out0, old, out1, log1 = _cap_scenario(CAP)
    old_keys = {(id(s.ref), s.kp) for s in old}
    n_tent_old = sum(1 for s in old if s.tentative)
    assert n_tent_old > 0 and out0["seeds_made"] == 288 and len(old) == len(old_m)
    expected = dfl.eviction_order(old, 30)
    p = _parse_r2(log1)
    assert len(p["ltkf"]) == 1 and len(p["kf"]) == 1
    g = [int(x) for x in p["ltkf"][0].groups()[:10]]
    kf, live_before, ev_d, ev_a, ev_c, live_after, active, dormant, tentative, cap = g
    # (the new keyframe's unlinked keypoints are fewer than 300: D1's local_map_track links the
    #  landmarks confirmed so far into it before the seeding)
    assert cap == CAP and out1["seeds_made"] > 100 and live_before == len(old) - out1["retired_linked"] + out1["seeds_made"]
    n_over = live_before - CAP
    assert n_over == N_EVICT and ev_d + ev_a + ev_c == n_over and live_after == CAP == len(arm.live) == active + dormant
    assert out1["retired"] == out1["retired_linked"] + ev_d + ev_a and out1["live"] == CAP
    evicted_rt = [(int(m.group(1)), int(m.group(2))) for m in p["rt"] if m.group(7) == "evicted"]
    evicted_cf = [(int(m.group(2)), int(m.group(3))) for m in p["cf"] if m.group(21) == "evicted"]
    linked_rt = {(int(m.group(1)), int(m.group(2))) for m in p["rt"] if m.group(7) == "linked"}
    assert len(evicted_rt) == ev_d + ev_a and len(evicted_cf) == ev_c > 0
    evicted = set(evicted_rt) | set(evicted_cf)
    assert all(ref == K0.id - 1 for ref, _ in evicted)                     # none of the new keyframe's seeds
    want = [(int(s.ref.id) - 1, s.kp) for s in expected if (int(s.ref.id) - 1, s.kp) not in linked_rt][:n_over]
    assert evicted == set(want)
    assert all(m.group(8) == "0" for m in p["rt"] if m.group(7) == "evicted")
    for m in p["cf"]:
        # reason, n_distinct, the confirmation frame and the creation keyframe (the newest one)
        assert m.group(21) == "evicted" and int(m.group(22)) >= N_DISTINCT
        assert int(m.group(7)) == 30 and int(m.group(18)) == frames[30].id - 1
        pid = int(m.group(1))
        assert world.get_point(pid) is not None and len(world.get_point(pid).observations) >= 2
    assert all(s.ref is frames[30] for s in arm.live[-out1["seeds_made"]:])
    # the evicted-confirmed landmarks are exported by write_map_points
    from map_export import write_map_points
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "map_points.txt"
        with contextlib.redirect_stdout(io.StringIO()):
            write_map_points(path, world, frames, [0, 30])
        ids = {int(l.split()[0]) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}
        assert all(int(m.group(1)) in ids for m in p["cf"])
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T7. descriptor death and the hopeless test
# --------------------------------------------------------------------------- #
def _kill_descriptors(k_from):
    def override(pts, poses, frames):
        rng = np.random.default_rng(99)
        for k in range(k_from, len(frames)):
            frames[k].Descriptors = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
        return frames
    return override


def test_t7_descriptor_death_retires_an_unconverged_seed_and_confirms_a_converged_one():
    # (a) descriptors die at frame 5, before any seed measured: b grows by one per eligible frame and the
    #     seed is retired reason=dead once the travel since the last match (the seeding frame) is 0.3 z_init
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(1.0, n_frames=100, frames_override=_kill_descriptors(5))
    p = _parse_r2(log)
    _assert_lifetime_identities(p)
    lts = [_lt(m) for m in p["lt"]]
    assert sum(t["matched"] for t in lts) == 0 and len(p["cf"]) == 0
    dead = [m for m in p["rt"] if m.group(7) == "dead"]
    assert len(dead) > 0.9 * 288 and all(m.group(8) == "0" and m.group(12) == "-1" and float(m.group(10)) == -1.0 for m in dead)
    z_init = kf_out["z_init"]
    for m in dead:
        k = int(m.group(6))
        travel = float(m.group(24))
        assert travel >= 0.3 * z_init and 0.009 * (k - 1) < 0.3 * z_init
        assert abs(travel - float(np.linalg.norm(frames[k].world_T_cam[:3, 3]))) <= 1e-9
        ef = int(m.group(23))
        assert float(m.group(18)) == pytest.approx(10.0 + ef) and ef >= 1
        assert int(m.group(13)) == 0 and int(m.group(9)) == -1
    frames_dead = sorted({int(m.group(6)) for m in dead})
    assert lts[frames_dead[0] - 1]["dead_retired"] > 0
    # b grew by exactly one per eligible frame: the failed sum over the frames equals the b increments
    assert sum(t["failed"] for t in lts) == sum(int(m.group(23)) for m in p["rt"])
    assert world.check_invariants(frames) == ZERO
    # (b) under k = 9.5 the seeds converge (tentative) before their descriptors die at frame 25: confirmed reason=dead
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(DEFERRED_K, n_frames=120, frames_override=_kill_descriptors(25))
    p = _parse_r2(log)
    _assert_lifetime_identities(p)
    dead_cf = [m for m in p["cf"] if m.group(21) == "dead"]
    assert len(dead_cf) > 0
    for m in dead_cf:
        # n_distinct, theta_max, converged before the descriptors died and confirmed after, travel and failures
        assert int(m.group(22)) >= N_DISTINCT and float(m.group(10)) >= THETA_MIN_DEG and int(m.group(8)) < 25 <= int(m.group(7))
        assert float(m.group(25)) >= 0.3 * kf_out["z_init"] and int(m.group(24)) >= 1
    assert sum(_lt(m)["dead_confirmed"] for m in p["lt"]) == len(dead_cf)
    assert all(m.group(8) == "0" for m in p["rt"] if m.group(7) == "dead")
    assert world.check_invariants(frames) == ZERO


def test_t7_hopeless_retires_on_the_next_eligible_failure():
    pts, poses, frames, _ = sweep(0.009, n_frames=12, noise_px=1.0)
    frames = _kill_descriptors(6)(pts, poses, frames)
    K0 = frames[0]
    world = Map(frames=frames)
    _link_true_landmarks(world, K0, pts, range(12))
    arm = _arm_r2(1.0)
    keyframes = [(0, K0)]
    with contextlib.redirect_stdout(io.StringIO()):
        arm.on_keyframe(0, K0, keyframes, world, frames)
        for k in range(1, 8):
            arm.on_frame(k, frames[k], keyframes, world, frames)
    live = list(arm.live)
    s400, s320 = live[0], live[1]
    s400.a, s400.b = 10.0, 400.0
    s320.a, s320.b = 10.0, 320.0
    out, log = _capture(arm.on_frame, 8, frames[8], keyframes, world, frames)
    p = _parse_r2(log)
    t = _lt(p["lt"][0])
    assert t["eligible"] > 0 and t["hopeless"] == 1
    hop = [m for m in p["rt"] if m.group(7) == "hopeless"]
    assert len(hop) == 1 and int(hop[0].group(2)) == s400.kp and float(hop[0].group(18)) == 401.0 and float(hop[0].group(17)) == 10.0
    assert s400 not in arm.live and s320 in arm.live and s320.b == 321.0 and s320.a == 10.0
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T8. linkage
# --------------------------------------------------------------------------- #
def test_t8_linked_tentative_seed_is_retired_converged_and_counted():
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(DEFERRED_K, n_frames=30, end=False)
    K0 = frames[0]
    tent = [s for s in arm.live if s.tentative]
    assert len(tent) >= 2
    a, b = tent[0], tent[1]
    for s in (a, b):
        pt = Point(point=np.array(pts[s.kp], dtype=np.float64), descriptor=np.zeros(32, dtype=np.uint8))
        world.add_point(pt)
        assert world.link(K0, s.kp, pt)
    n_handles = len(arm.tentative_points())
    # D2: on_keyframe retires both by linkage, counted as retired_converged
    keyframes = [(0, K0), (30, frames[30])]
    out, log1 = _capture(arm.on_keyframe, 30, frames[30], keyframes, world, frames)
    p = _parse_r2(log1)
    linked = [m for m in p["rt"] if m.group(7) == "linked"]
    assert {int(m.group(2)) for m in linked} >= {a.kp, b.kp} and all(m.group(8) == "1" for m in linked if int(m.group(2)) in (a.kp, b.kp))
    assert out["retired_linked"] == len(linked) >= 2 and out["retired_converged"] >= 2 and out["retired"] == out["retired_linked"]
    assert int(p["kf"][0].group(8)) == out["retired_converged"]
    assert a not in arm.live and b not in arm.live and len(arm.tentative_points()) == n_handles - 2
    assert all(h[3] is not a and h[3] is not b for h in arm.tentative_points())
    # C1: a tentative seed linked between frames is dropped by the next on_frame (linked_dropped)
    c = next(s for s in arm.live if s.tentative)
    pt = Point(point=np.array(pts[c.kp], dtype=np.float64), descriptor=np.zeros(32, dtype=np.uint8))
    world.add_point(pt)
    assert world.link(K0, c.kp, pt)
    out2, log2 = _capture(arm.on_frame, 30, frames[30], keyframes, world, frames)
    p2 = _parse_r2(log2)
    t = _lt(p2["lt"][0])
    assert t["linked_dropped"] >= 1 and out2["dropped_linked"] == t["linked_dropped"]
    m = next(m for m in p2["rt"] if m.group(7) == "linked" and int(m.group(2)) == c.kp)
    assert m.group(8) == "1" and c not in arm.live and all(h[3] is not c for h in arm.tentative_points())
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T9. the trace
# --------------------------------------------------------------------------- #
def test_t9_trace_one_line_per_matched_measurement(tmp_path):
    trace = tmp_path / "df_trace.txt"
    pts, frames, world, arm, kf_out, outs, end_out, log = _run_sweep_r2(1.0, n_frames=25, trace_path=str(trace))
    p = _parse_r2(log)
    lines = trace.read_text().splitlines()
    assert lines[0] == "# " + " ".join(dfa.TRACE_COLUMNS) and len(dfa.TRACE_COLUMNS) == 17
    rows = [l.split() for l in lines[1:]]
    assert len(rows) == sum(_lt(m)["matched"] for m in p["lt"]) > 0
    assert all(len(r) == 17 for r in rows)
    n_fam = 0
    n_first = 0
    seen = set()
    for r in rows:
        k, ref, kp, kp_f, level, state = (int(x) for x in r[:6])
        d_pred, d_far, d_meas, resid = (float(x) for x in r[6:10])
        assert ref == frames[0].id - 1 and kp == kp_f and level == 0 and state in (0, 1) and 1 <= k <= 25
        # resid = d_meas - d_pred, up to the %.6g rounding of the three printed values
        assert abs(resid - (d_meas - d_pred)) <= 1e-6 + 1e-5 * (abs(d_meas) + abs(d_pred) + abs(resid))
        fam = int(r[16])
        assert fam in (0, 1)
        first_row = kp not in seen
        seen.add(kp)
        if fam:
            n_fam += 1
            # d_far is the initial-mean disparity: X(z_init) against the ray at infinity in frame k
            seed_ray = df._unit_rays_cam(K, np.array([frames[0].KeyPoints[kp].pt]))[0]
            X0 = seed_ray * kf_out["z_init"]
            fr = frames[k]
            R, t = fr.cam_T_world[:3, :3], fr.cam_T_world[:3, 3]
            p_x = df._project(K, (R @ X0 + t)[None, :])[0]
            p_inf = df._project(K, (R @ seed_ray)[None, :])[0]
            assert d_far == pytest.approx(float(np.linalg.norm(p_x - p_inf)), rel=2e-5)
            if first_row:
                # the seed's very first measurement: mu is still 1 / z_init, so d_pred is the same quantity
                n_first += 1
                assert d_pred == pytest.approx(d_far, rel=2e-5)
                assert float(r[14]) == pytest.approx(1.0 / kf_out["z_init"], rel=1e-6)     # mu_before
        else:
            assert math.isfinite(d_far) and d_far >= EPS_MATCH
        assert 0.0 <= float(r[12]) <= 1.0 and float(r[13]) <= 50.0 and float(r[10]) > 0.0 and float(r[11]) > 0.0
    # the rows under the initial-mean reading are the matched part of the eligible seeds under it
    assert 0 < n_first <= n_fam <= sum(_lt(m)["first_at_mean"] for m in p["lt"])
    assert arm._trace is None                                                 # closed at on_end
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T10. no-leak and boundaries
# --------------------------------------------------------------------------- #
def test_t10_no_leak_and_module_boundaries():
    src_l = Path(dfl.__file__).read_text()
    src_a = Path(dfa.__file__).read_text()
    for tok in ("groundtruth", "depth.txt", "import dataset", "from dataset", "cv2.imread", "depth/"):
        assert tok not in src_l and tok not in src_a, tok
    for tok in ("import Map", "from Map", "import Point", "from Point", "world_map"):
        assert tok not in src_l, tok
    import local_mapping
    for name in ("classify", "hopeless", "converged", "credible_segments", "eviction_order", "DepthFilterArm", "make_seeds"):
        assert not hasattr(local_mapping, name), name
    src_df = Path(df.__file__).read_text()
    for tok in ("lifetime", "betainc", "dormant"):
        assert tok not in src_df, tok
    for tok in ("[DF-LIFETIME]", "[DF-LIFETIME-KF]", "[DF-CONFIRM]", "[DF-RETIRE]", "[DF-END]", "[DF-PROMOTE]", "[DF-FRAME]", "[DF-KEYFRAME]"):
        assert tok in src_a, tok
    import pipeline
    psrc = Path(pipeline.__file__).read_text()
    for tok in (".on_frame(", ".on_keyframe(", ".on_end(", "[DF-ERROR]"):
        assert tok in psrc, tok
    assert psrc.index("frames_estimated = frames[: last_idx + 1]") < psrc.index(".on_end(") < psrc.index("write_map_points(")
    for fn in ("credible_segments", "segment_in_image", "predicted_disparity", "classify", "hopeless", "distinct_baselines", "converged", "dead", "eviction_order"):
        assert callable(getattr(dfl, fn)), fn
    for meth in ("on_frame", "on_keyframe", "on_end", "tentative_points"):
        assert hasattr(DepthFilterArm, meth), meth


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
