"""
Stage 04g (spec T): the depth-filter ARM (depth_filter_arm.py) that drives the
verified module of 04f against a Map, on the synthetic sweeps of
tests/test_depth_filter.py and the three-keyframe scene of
tests/test_local_mapping.py; the parser flags; the no-leak rule.  No dataset,
no pipeline run.

The sweep (test_depth_filter.sweep): K_0 at the origin, frames 1..40 at
x = 0.009 k, 300 points at 1.5 to 4.0 m, 1 px noise in every image, distinct
random descriptors identical across frames, keypoint index == point index.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import depth_filter as df  # noqa: E402
import depth_filter_arm as dfa  # noqa: E402
from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from depth_filter import DepthFilterParams  # noqa: E402
from depth_filter_arm import DepthFilterArm, DepthFilterArmParams, scene_range  # noqa: E402
from local_mapping import LocalMappingParams  # noqa: E402
from test_depth_filter import sweep  # noqa: E402
from test_local_mapping import BASELINE_M, H, K, W, YAW_DEG, ZERO, _KF, _poses, _yaw  # noqa: E402

THETA_MIN_DEG = 1.1
DEFERRED_K = 9.5

RX_KF = re.compile(r"^\s*>> \[DF-KEYFRAME\] kf=(\d+) pos=(\d+) neighbours=(\d+) obs_inserted=(\d+) culled=(\d+) retired=(\d+) "
                   r"retired_linked=(\d+) retired_converged=(\d+) unlinked=(\d+) seeds_made=(\d+) z_init=(\S+) "
                   r"z_source=(links|map|window|none) live=(\d+)$")
RX_FR = re.compile(r"^\s*>> \[DF-FRAME\] frame=(\d+) live=(\d+) measured=(\d+) inlier=(\d+) outlier=(\d+) no_match=(\d+) "
                   r"no_translation=(\d+) behind=(\d+) no_disparity=(\d+) outside_support=(\d+) converged=(\d+) promoted=(\d+) "
                   r"dropped_linked=(\d+) links=(\d+) links_refused=(\d+) inserted=(\d+)$")
RX_PR = re.compile(r"^\s*>> \[DF-PROMOTE\] id=(\d+) ref=(\d+) kp=(\d+) u=(\S+) v=(\S+) level=(\d+) frame=(\d+) converged_at=(\d+) "
                   r"widest=(\d+) theta_max_deg=(\S+) theta_acc_deg=(\S+) n_meas=(\d+) n_inlier=(\d+) inlier_ratio=(\S+) "
                   r"sigma_rel=(\S+) range=(\S+) z_ref=(\S+) created_kf=(\d+) links=(\d+) links_refused=(\d+)$")
DF_FLAGS = ("df_min_parallax_deg", "df_init_depth_ratio_min", "df_search_band_sigmas", "df_epipolar_chi2",
            "df_max_desc_dist", "df_beta_init", "df_max_seed_keyframes", "df_responsibility_inlier")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _arm(factor=1.0, **df_kw):
    return DepthFilterArm(DepthFilterParams(**df_kw), DepthFilterArmParams(promote_factor=factor), LocalMappingParams())


def _capture(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kw)
    return out, buf.getvalue()


def _parse(log):
    kfs = [RX_KF.match(l) for l in log.splitlines() if "[DF-KEYFRAME]" in l]
    frs = [RX_FR.match(l) for l in log.splitlines() if "[DF-FRAME]" in l]
    prs = [RX_PR.match(l) for l in log.splitlines() if "[DF-PROMOTE]" in l]
    assert all(kfs) and all(frs) and all(prs), "a [DF-...] line does not match its format"
    return kfs, frs, prs


def _assert_frame_identities(m):
    g = [int(x) for x in m.groups()]
    frame, live, measured, inlier, outlier, no_match, no_translation, behind, no_disparity = g[:9]
    assert measured == inlier + outlier, m.group(0)
    assert live == measured + no_match + no_translation + behind + no_disparity, m.group(0)


def _depth_in(frame, X):
    return float((frame.cam_T_world[:3, :3] @ np.asarray(X, dtype=np.float64) + frame.cam_T_world[:3, 3])[2])


def _link_true_landmarks(world, kf, pts, idxs):
    """Landmarks at the true positions of ``pts[idxs]`` linked in ``kf`` at those keypoints."""
    made = []
    for i in idxs:
        p = Point(point=np.array(pts[i], dtype=np.float64), descriptor=kf.Descriptors[i])
        world.add_point(p)
        assert world.link(kf, i, p)
        made.append(p)
    return made


def _run_sweep(factor, n_frames=40):
    """T3 / T4: the clean sweep through the driver.  Returns (pts, frames, world, arm, kf_out,
    frame_outs, log)."""
    pts, poses, frames, _ = sweep(0.009, n_frames=n_frames, noise_px=1.0)
    K0 = frames[0]
    world = Map(frames=frames)
    _link_true_landmarks(world, K0, pts, range(12))
    arm = _arm(factor)
    keyframes = [(0, K0)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kf_out = arm.on_keyframe(0, K0, keyframes, world, frames)
        outs = [arm.on_frame(k, frames[k], keyframes, world, frames) for k in range(1, n_frames + 1)]
    return pts, frames, world, arm, kf_out, outs, buf.getvalue()


# --------------------------------------------------------------------------- #
# T1. parser and params
# --------------------------------------------------------------------------- #
def test_t1_parser_flags_params_and_resolved_settings(tmp_path):
    import main
    parser = main.get_parser()
    a = parser.parse_args(["--dataset", "rgb"])
    assert a.point_source == "keyframe_pair"
    P = DepthFilterParams()
    for f in DF_FLAGS:
        assert hasattr(a, f), f
        assert getattr(a, f) == getattr(P, f[3:]), f
        assert type(getattr(a, f)) is type(getattr(P, f[3:])), f
    assert a.df_promote_factor == 1.0 and a.df_bootstrap_depth_ratio == 5.0
    assert DepthFilterParams.from_args(a) == P
    assert DepthFilterArmParams.from_args(a) == DepthFilterArmParams()
    assert DepthFilterArmParams() == DepthFilterArmParams(promote_factor=1.0, bootstrap_depth_ratio=5.0)
    b = parser.parse_args(["--dataset", "rgb", "--point_source", "depth_filter", "--df_promote_factor", "9.5"])
    assert b.point_source == "depth_filter"
    assert DepthFilterArmParams.from_args(b).promote_factor == 9.5
    assert DepthFilterArmParams.from_args(b).bootstrap_depth_ratio == 5.0
    assert DepthFilterParams.from_args(b) == P
    # the module flags reach the module params and the arm flag does not touch them
    c = parser.parse_args(["--dataset", "rgb", "--df_max_seed_keyframes", "3", "--df_min_parallax_deg", "2.0",
                           "--df_bootstrap_depth_ratio", "4.0"])
    Q = DepthFilterParams.from_args(c)
    assert Q.max_seed_keyframes == 3 and Q.min_parallax_deg == 2.0 and Q.beta_init == P.beta_init
    assert DepthFilterArmParams.from_args(c) == DepthFilterArmParams(promote_factor=1.0, bootstrap_depth_ratio=4.0)
    # an unknown arm is refused by the parser and by the pipeline
    with pytest.raises(SystemExit):
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--dataset", "rgb", "--point_source", "svo"])
    import pipeline
    with pytest.raises(ValueError):
        pipeline.show_trajectory([object()], point_source="svo")
    # resolved_settings.json records the arm and the factor
    out = tmp_path / "resolved_settings.json"
    with contextlib.redirect_stdout(io.StringIO()):
        main.write_resolved_settings(b, out)
    settings = json.loads(out.read_text())
    assert settings["point_source"] == "depth_filter" and settings["df_promote_factor"] == 9.5
    assert settings["lm_confirm"] is False and settings["undistort"] is False
    for f in DF_FLAGS:
        assert settings[f] == getattr(P, f[3:]), f


# --------------------------------------------------------------------------- #
# T2. scene_range, four sources
# --------------------------------------------------------------------------- #
def _three_keyframe_scene(noise_px=0.5, seed=11):
    """K_0 at the origin, K_1 at 0.18 m with 5 deg yaw, K_2 at 0.36 m with 10 deg yaw; 300 points."""
    from test_local_mapping import _regime_points
    rng = np.random.default_rng(seed)
    T0, T1 = _poses()
    T2 = np.eye(4)
    T2[:3, :3] = _yaw(2 * YAW_DEG)
    T2[:3, 3] = [2 * BASELINE_M, 0.0, 0.0]
    pts = _regime_points(rng, T0, T2)
    descs = rng.integers(0, 256, size=(len(pts), 32), dtype=np.uint8)
    k0 = _KF(1, T0, pts, descs, noise_px=noise_px, rng=rng)
    k1 = _KF(2, T1, pts, descs, noise_px=noise_px, rng=rng)
    k2 = _KF(3, T2, pts, descs, noise_px=noise_px, rng=rng)
    return pts, descs, [k0, k1, k2]


def _fresh_map(kfs):
    """A new Map over ``kfs`` with the frames' link tables cleared (the tables live on the frames)."""
    for f in kfs:
        f.keypoint_to_point = {}
    return Map(frames=kfs)


def test_t2_scene_range_links_map_window_none():
    pts, descs, kfs = _three_keyframe_scene()
    k0, k1, k2 = kfs
    A = DepthFilterArmParams()
    P = DepthFilterParams()
    keyframes = [(0, k0), (20, k1), (40, k2)]
    # (links): 12 landmarks linked in K_2 at known camera depths
    world = _fresh_map(kfs)
    made = _link_true_landmarks(world, k2, pts, range(12))
    depths = [_depth_in(k2, p.point) for p in made]
    z, src = scene_range(k2, 2, keyframes, world, A, P)
    assert src == "links" and z == pytest.approx(float(np.median(depths)))
    # (map): the same landmarks present but unlinked in K_2 (linked in K_0 instead)
    world = _fresh_map(kfs)
    made = _link_true_landmarks(world, k0, pts, range(12))
    assert k2.keypoint_to_point == {}
    z, src = scene_range(k2, 2, keyframes, world, A, P)
    assert src == "map" and z == pytest.approx(float(np.median(depths)))
    # ... a landmark behind K_2 or outside its image does not count, and 9 are not enough
    world.add_point(Point(point=np.array([0.0, 0.0, -5.0]), descriptor=descs[0]))
    world.add_point(Point(point=np.array([50.0, 0.0, 3.0]), descriptor=descs[0]))
    world.add_point(Point(point=np.array([np.nan, 0.0, 3.0]), descriptor=descs[0]))
    z, src = scene_range(k2, 2, keyframes, world, A, P)
    assert src == "map" and z == pytest.approx(float(np.median(depths)))
    world9 = _fresh_map(kfs)
    _link_true_landmarks(world9, k0, pts, range(9))
    for _ in range(5):
        world9.add_point(Point(point=np.array([0.0, 0.0, -5.0]), descriptor=descs[0]))     # behind: never counted
    z, src = scene_range(k2, 2, keyframes, world9, A, P)
    assert src != "map" and src != "links"
    assert src == "window"
    # (links) beats (map): 10 linked in K_2 and 12 more unlinked at other depths
    world = _fresh_map(kfs)
    linked = _link_true_landmarks(world, k2, pts, range(10))
    for i in range(10, 22):
        world.add_point(Point(point=np.array(pts[i]) * 1.5, descriptor=descs[i]))
    z, src = scene_range(k2, 2, keyframes, world, A, P)
    assert src == "links" and z == pytest.approx(float(np.median([_depth_in(k2, p.point) for p in linked])))
    # (window): empty map, kf_pos 2, the earlier keyframes at 0.18 m and 0.36 m from K_2
    world = _fresh_map(kfs)
    z, src = scene_range(k2, 2, keyframes, world, A, P)
    assert src == "window" and z == pytest.approx(5.0 * 0.36)
    z, src = scene_range(k1, 1, keyframes, world, A, P)
    assert src == "window" and z == pytest.approx(5.0 * 0.18)
    assert scene_range(k2, 2, keyframes, world, DepthFilterArmParams(bootstrap_depth_ratio=4.0), P)[0] == pytest.approx(4.0 * 0.36)
    # the window is the previous bootstrap_window keyframes only (04i: decoupled from max_seed_keyframes)
    far = _KF(9, np.diag([1.0, 1.0, 1.0, 1.0]), [], None)
    far.world_T_cam[:3, 3] = [-10.0, 0.0, 0.0]
    far.cam_T_world = np.linalg.inv(far.world_T_cam)
    kf_far = [(0, far), (10, k0), (20, k1), (40, k2)]
    assert scene_range(k2, 3, kf_far, world, A, P)[0] == pytest.approx(5.0 * 10.36)
    assert scene_range(k2, 3, kf_far, world, DepthFilterArmParams(bootstrap_window=2), P)[0] == pytest.approx(5.0 * 0.36)  # 04i: the window is the previous bootstrap_window keyframes only
    # (none): kf_pos 0, or earlier keyframes at the same centre; make_seeds is then not called
    z, src = scene_range(k0, 0, keyframes, world, A, P)
    assert src == "none" and math.isnan(z)
    k0b = _KF(4, np.eye(4), pts, descs)
    z, src = scene_range(k0b, 1, [(0, k0), (20, k0b)], world, A, P)
    assert src == "none" and math.isnan(z)
    arm = _arm()
    calls = []
    real = df.make_seeds
    try:
        df.make_seeds = lambda *a, **k: calls.append(a) or real(*a, **k)
        out, log = _capture(arm.on_keyframe, 20, k0b, [(0, k0), (20, k0b)], world, kfs + [k0b])
    finally:
        df.make_seeds = real
    assert calls == [] and out["seeds_made"] == 0 and out["z_source"] == "none" and arm.live == []
    assert out["unlinked"] == 300
    assert "z_init=nan z_source=none live=0" in log
    assert world.check_invariants(kfs + [k0b]) == ZERO


# --------------------------------------------------------------------------- #
# T3. the reference arm on the clean sweep
# --------------------------------------------------------------------------- #
def test_t3_reference_arm_promotes_at_first_convergence_and_links_the_true_observations():
    pts, frames, world, arm, kf_out, outs, log = _run_sweep(1.0)
    K0 = frames[0]
    kfs, frs, prs = _parse(log)
    assert len(kfs) == 1 and len(frs) == 40
    # seeding: the 12 linked keypoints are not seeded, z_init from the links
    assert kf_out["seeds_made"] == 288 and kf_out["unlinked"] == 288 and kf_out["z_source"] == "links"
    assert kf_out["z_init"] == pytest.approx(float(np.median([_depth_in(K0, pts[i]) for i in range(12)])))
    assert int(kfs[0].group(10)) == 288 and kfs[0].group(12) == "links" and int(kfs[0].group(13)) == 288
    recs = [r for o in outs for r in o["promotions"]]
    assert len(recs) == len(prs) >= 0.8 * 288, len(recs)
    assert sum(o["promoted"] for o in outs) == len(recs) == len(world.points) - 12
    # the paper's timing: converged_at == frame, theta_max >= theta_min, the driver's widest view
    # reads promotable's theta_max
    for r, m in zip(recs, prs):
        assert r["converged_at"] == r["frame"]
        assert int(m.group(8)) == int(m.group(7)) == r["frame"]
        assert r["theta_max_deg"] >= THETA_MIN_DEG and float(m.group(10)) >= THETA_MIN_DEG
        assert r["theta_acc_deg"] >= THETA_MIN_DEG
        assert abs(r["theta_max_driver_deg"] - r["theta_max_deg"]) <= 1e-6
        assert int(m.group(1)) == r["id"] == r["point"].id
        assert int(m.group(2)) == K0.id - 1 and int(m.group(3)) == r["kp"]
        assert float(m.group(4)) == pytest.approx(K0.KeyPoints[r["kp"]].pt[0], abs=0.006)
        assert int(m.group(6)) == 0 and int(m.group(18)) == K0.id - 1
        assert int(m.group(19)) == r["links"] == len(r["point"].observations)
        assert int(m.group(20)) == r["links_refused"] == 0
        assert int(m.group(12)) == len(r["seed"].measurements) >= int(m.group(13)) >= 1
        assert 0.0 < float(m.group(14)) < 1.0
        # widest is the inlier measurement's frame with the largest parallax at X
        s = r["seed"]
        X = df.seed_point(s)
        angles = {m_.traj_idx: math.degrees(dfa._parallax_at(X, s.centre, m_.frame.world_T_cam[:3, 3]))
                  for m_ in df.inlier_measurements(s, DepthFilterParams()) if m_.frame is not None}
        assert r["widest"] == max(angles, key=angles.get)
        assert angles[r["widest"]] == pytest.approx(r["theta_max_deg"], abs=1e-9)
    # every promoted landmark: K_0's descriptor, an observation in K_0 at the keypoint, further
    # observations at the same keypoint index (the true correspondence), created_fid K_0
    errs = []
    for r in recs:
        p = r["point"]
        assert world.get_point(p.id) is p
        np.testing.assert_array_equal(p.descriptor, K0.Descriptors[r["kp"]])
        assert p.observations[K0.id] == r["kp"]
        assert len(p.observations) >= 2
        assert all(kp == r["kp"] for kp in p.observations.values())
        assert all(frames[fid - 100].keypoint_to_point[r["kp"]] == p.id for fid in p.observations)
        assert p.created_fid == K0.id and p.last_seen_fid == max(p.observations)
        assert r["created_kf"] == K0.id - 1
        r_true = float(np.linalg.norm(pts[r["kp"]]))
        r_est = float(np.linalg.norm(np.asarray(p.point) - K0.world_T_cam[:3, 3]))
        assert r_est == pytest.approx(r["range"], rel=1e-9)
        errs.append(abs(r_est - r_true) / r_true)
    assert np.median(errs) <= 0.095, np.median(errs)
    # the [DF-FRAME] identities on every line; promoted counts sum to the landmarks
    for m in frs:
        _assert_frame_identities(m)
    assert sum(int(m.group(12)) for m in frs) == len(recs)
    assert sum(int(m.group(14)) for m in frs) == sum(r["links"] for r in recs)
    assert all(int(m.group(11)) == int(m.group(12)) for m in frs)       # converged == promoted under k = 1
    assert all(o["dropped_linked"] == 0 and o["inserted"] == 0 for o in outs)   # K_0 is every landmark's partner
    assert [int(m.group(1)) for m in frs] == list(range(1, 41))
    assert int(frs[0].group(2)) == 288 and int(frs[-1].group(2)) == len(arm.live)
    # each promote line precedes its frame's line
    for r in recs:
        line_p = next(m.group(0) for m in prs if int(m.group(1)) == r["id"])
        line_f = next(m.group(0) for m in frs if int(m.group(1)) == r["frame"])
        assert log.index(line_p) < log.index(line_f)
    assert "[LOCAL-MAP" not in log
    assert world.check_invariants(frames) == ZERO


def test_t3_totals_and_stats_accumulate():
    pts, frames, world, arm, kf_out, outs, log = _run_sweep(1.0, n_frames=12)
    n = sum(o["promoted"] for o in outs)
    assert arm.totals["df_promoted"] == n and arm.totals["df_seeds_made"] == 288 and arm.totals["df_keyframes"] == 1
    assert arm.totals["df_frames"] == 12 and arm.totals["df_links"] == sum(o["links"] for o in outs)
    stats = {}
    with contextlib.redirect_stdout(io.StringIO()):
        arm.on_keyframe(12, frames[12], [(0, frames[0]), (12, frames[12])], world, frames, stats)
    assert stats["df_keyframes"] == 2 and stats["df_promoted"] == n and stats["df_seeds_made"] == arm.totals["df_seeds_made"]
    assert set(stats) == {"df_keyframes", "df_seeds_made", "df_promoted", "df_retired", "df_obs_inserted", "df_culled"}
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T4. the deferred arm on the same sweep
# --------------------------------------------------------------------------- #
def test_t4_deferred_arm_promotes_later_at_the_accumulated_parallax_bar_and_more_precisely():
    pts3, frames3, world3, arm3, _, outs3, log3 = _run_sweep(1.0)
    pts, frames, world, arm, kf_out, outs, log = _run_sweep(DEFERRED_K)
    K0 = frames[0]
    kfs, frs, prs = _parse(log)
    recs = [r for o in outs for r in o["promotions"]]
    recs3 = [r for o in outs3 for r in o["promotions"]]
    assert kf_out["seeds_made"] == 288
    assert len(recs) == len(prs) >= 0.3 * 288, len(recs)
    bar = DEFERRED_K * THETA_MIN_DEG
    for r, m in zip(recs, prs):
        assert r["theta_acc_deg"] >= bar and float(m.group(11)) >= bar - 0.0005
        assert r["frame"] >= r["converged_at"]
        assert int(m.group(7)) == r["frame"] and int(m.group(8)) == r["converged_at"]
        assert r["theta_max_deg"] >= THETA_MIN_DEG
        assert abs(r["theta_max_driver_deg"] - r["theta_max_deg"]) <= 1e-6
        # the module's own rule held at the first convergence and still holds
        assert df.promotable(r["seed"], DepthFilterParams())[0]
    later = [r for r in recs if r["frame"] > r["converged_at"]]
    assert len(later) >= 0.9 * len(recs), (len(later), len(recs))
    assert np.median([r["frame"] for r in recs]) > np.median([r["frame"] for r in recs3])
    assert np.median([r["frame"] - r["converged_at"] for r in recs]) >= 1

    def rel_errs(rs):
        return [abs(r["range"] - np.linalg.norm(pts[r["kp"]])) / np.linalg.norm(pts[r["kp"]]) for r in rs]
    assert np.median(rel_errs(recs)) <= np.median(rel_errs(recs3))
    assert np.median(rel_errs(recs)) <= 0.095
    # converged counts the waiting seeds too: on some frame more seeds converged than promoted
    assert any(int(m.group(11)) > int(m.group(12)) for m in frs)
    for m in frs:
        _assert_frame_identities(m)
    assert sum(int(m.group(12)) for m in frs) == len(recs) == len(world.points) - 12
    # the first convergence is recorded on the seed and never moves
    for r in recs:
        assert r["seed"].first_ok_traj == r["converged_at"]
    assert all(getattr(s, "first_ok_traj", None) is None or s.first_ok_traj <= 40 for s in arm.live)
    # observations: more measurements linked per landmark than under the reference arm
    assert np.median([len(r["point"].observations) for r in recs]) > np.median([len(r["point"].observations) for r in recs3])
    for r in recs:
        p = r["point"]
        assert p.observations[K0.id] == r["kp"] and all(kp == r["kp"] for kp in p.observations.values())
        assert p.created_fid == K0.id
    assert world.check_invariants(frames) == ZERO


def test_t4_promote_factor_is_the_only_difference_between_the_arms():
    """A factor below 1 reduces to the module's rule (promotable's (i) implies theta_acc >= theta_min):
    identical promotions to factor 1.0 on the same sweep, frame by frame."""
    _, _, _, _, _, outs1, log1 = _run_sweep(1.0, n_frames=15)
    _, _, _, _, _, outs0, log0 = _run_sweep(0.5, n_frames=15)
    a = [(r["kp"], r["frame"], r["theta_acc_deg"]) for o in outs1 for r in o["promotions"]]
    b = [(r["kp"], r["frame"], r["theta_acc_deg"]) for o in outs0 for r in o["promotions"]]
    assert a == b and len(a) > 100
    # ... and a factor of 2 delays a real fraction of them
    _, _, _, _, _, outs2, _ = _run_sweep(2.0, n_frames=15)
    c = {r["kp"]: r["frame"] for o in outs2 for r in o["promotions"]}
    a_by = {kp: fr for kp, fr, _ in a}
    both = [kp for kp in c if kp in a_by]
    assert len(both) > 50 and all(c[kp] >= a_by[kp] for kp in both)
    assert np.mean([c[kp] > a_by[kp] for kp in both]) > 0.5


# --------------------------------------------------------------------------- #
# T5. keyframes: retirement, culling, the [DF-KEYFRAME] line
# --------------------------------------------------------------------------- #
def test_t5_keyframes_seed_retire_and_cull_through_the_driver():
    pts, descs, kfs = _three_keyframe_scene()
    k0, k1, k2 = kfs
    P = DepthFilterParams()
    L = LocalMappingParams()
    frames = list(kfs)
    world = Map(frames=frames)
    arm = _arm()
    # a landmark created at K_0 with one keyframe observation and a descriptor no keypoint matches:
    # culled once the keyframe list is cull_after_keyframes entries past K_0
    lone = Point(point=np.array([0.05, -0.02, 2.2]), descriptor=np.full(32, 0xAA, dtype=np.uint8))
    world.add_point(lone)
    assert world.link(k0, 0, lone)
    lone.created_fid = k0.id
    # K_0: no seeds (z_source none)
    keyframes = [(0, k0)]
    out0, log0 = _capture(arm.on_keyframe, 0, k0, keyframes, world, frames)
    kfl, _, _ = _parse(log0)
    assert len(kfl) == 1 and out0["z_source"] == "none" and out0["seeds_made"] == 0 and arm.live == []
    assert out0["pos"] == 0 and out0["neighbours"] == 0 and out0["culled"] == 0 and out0["retired"] == 0
    assert out0["unlinked"] == 299 and out0["live"] == 0
    assert math.isnan(out0["z_init"]) and kfl[0].group(11) == "nan" and kfl[0].group(12) == "none"
    # K_1: every unlinked keypoint seeded at 5 x ||C_1 - C_0||
    keyframes.append((20, k1))
    out1, log1 = _capture(arm.on_keyframe, 20, k1, keyframes, world, frames)
    kfl, _, _ = _parse(log1)
    assert len(kfl) == 1 and int(kfl[0].group(1)) == 20 and int(kfl[0].group(2)) == 1
    assert out1["z_source"] == "window" and out1["z_init"] == pytest.approx(5.0 * float(np.linalg.norm(k1.world_T_cam[:3, 3])))
    assert out1["seeds_made"] == out1["unlinked"] == len(k1.KeyPoints) - len(k1.keypoint_to_point) == len(arm.live)
    assert out1["seeds_made"] >= 299
    assert all(s.ref is k1 and s.ref_kf_pos == 1 and s.mu == pytest.approx(1.0 / out1["z_init"]) for s in arm.live)
    assert out1["neighbours"] == 1 and out1["retired"] == 0 and out1["culled"] == 0
    assert int(kfl[0].group(10)) == out1["seeds_made"] and int(kfl[0].group(13)) == out1["live"]
    assert kfl[0].group(11) == f"{out1['z_init']:.6g}"
    n1 = len(arm.live)
    # link two of K_1's keypoints by hand: the next keyframe retires exactly those two as retired_linked
    seeded = [s.kp for s in arm.live]
    a_kp, b_kp = seeded[5], seeded[17]
    for kp in (a_kp, b_kp):
        p = Point(point=np.array(pts[kp], dtype=np.float64), descriptor=np.full(32, 0x55, dtype=np.uint8))
        world.add_point(p)
        assert world.link(k1, kp, p)
    keyframes.append((40, k2))
    out2, log2 = _capture(arm.on_keyframe, 40, k2, keyframes, world, frames)
    assert out2["retired_linked"] == 2 and out2["retired"] == 2 and out2["retired_converged"] == 0
    assert all(s.kp not in (a_kp, b_kp) or s.ref is not k1 for s in arm.live)
    assert out2["z_source"] == "window" and out2["z_init"] == pytest.approx(5.0 * 0.36)
    assert out2["seeds_made"] == out2["unlinked"] and out2["seeds_made"] > 250
    assert out2["live"] == n1 - 2 + out2["seeds_made"] == len(arm.live)
    assert out2["neighbours"] == 2 and out2["culled"] == 0
    n2 = out2["seeds_made"]
    # K_3: three entries past K_0 -> the lone landmark is culled (one keyframe observation)
    T = np.eye(4)
    T[:3, 3] = [3 * BASELINE_M, 0.0, 0.0]
    k3 = _KF(4, T, [], None)
    frames.append(k3)
    keyframes.append((60, k3))
    out3, log3 = _capture(arm.on_keyframe, 60, k3, keyframes, world, frames)
    assert out3["culled"] == 1 and world.get_point(lone.id) is None and 0 not in k0.keypoint_to_point
    assert out3["seeds_made"] == 0 and out3["unlinked"] == 0 and out3["retired"] == 0
    assert out3["live"] == n1 - 2 + n2
    assert out3["z_source"] == "window"          # the window moved; nothing to seed on an empty frame
    # append keyframes until K_1's seeds are max_seed_keyframes old: retired by age
    pos = 4
    culled_by_pos = {}
    while pos < 1 + P.max_seed_keyframes:
        T = np.eye(4)
        T[:3, 3] = [pos * BASELINE_M, 0.0, 0.0]
        kf = _KF(10 + pos, T, [], None)
        frames.append(kf)
        keyframes.append((20 * pos, kf))
        out, log = _capture(arm.on_keyframe, 20 * pos, kf, keyframes, world, frames)
        kfl, _, _ = _parse(log)
        assert len(kfl) == 1 and int(kfl[0].group(2)) == pos
        assert out["retired"] == 0 and out["live"] == n1 - 2 + n2
        culled_by_pos[pos] = out["culled"]
        pos += 1
    T = np.eye(4)
    T[:3, 3] = [pos * BASELINE_M, 0.0, 0.0]
    kf6 = _KF(10 + pos, T, [], None)
    frames.append(kf6)
    keyframes.append((20 * pos, kf6))
    assert pos == 6
    out6, log6 = _capture(arm.on_keyframe, 120, kf6, keyframes, world, frames)
    assert out6["retired"] == n1 - 2 and out6["retired_linked"] == 0
    assert out6["live"] == n2 == len(arm.live) and all(s.ref is k2 for s in arm.live)
    kfl, _, _ = _parse(log6)
    assert int(kfl[0].group(6)) == n1 - 2 and int(kfl[0].group(13)) == n2
    # culled: the lone landmark at K_3, the two hand-made ones (created at K_1, one keyframe
    # observation each) at K_4
    assert culled_by_pos == {4: 2, 5: 0}
    assert arm.totals["df_retired"] == 2 + (n1 - 2) and arm.totals["df_culled"] == 3 and arm.totals["df_keyframes"] == 7
    # the aged seeds were not converged (no measurements): retired_converged stays 0
    assert out6["retired_converged"] == 0
    assert "[LOCAL-MAP" not in log0 + log1 + log2 + log3 + log6
    assert world.check_invariants(frames) == ZERO


def test_t5_retired_converged_counts_converged_seeds_lost_to_age_under_deferral():
    """Under k = 9.5 on a 12-frame sweep some seeds converge under the paper's rule but do not
    reach 10.45 deg; when the reference keyframe ages out they are counted as retired_converged."""
    pts, frames, world, arm, kf_out, outs, log = _run_sweep(DEFERRED_K, n_frames=12)
    P = DepthFilterParams()
    conv_live = [s for s in arm.live if df.promotable(s, P)[0]]
    assert len(conv_live) > 50
    keyframes = [(0, frames[0])]
    for pos in range(1, 6):
        T = np.eye(4)
        T[:3, 3] = [0.009 * 12 + 0.05 * pos, 0.0, 0.0]
        kf = _KF(300 + pos, T, [], None)
        frames.append(kf)
        keyframes.append((12 + pos, kf))
        out, _ = _capture(arm.on_keyframe, 12 + pos, kf, keyframes, world, frames)
    assert out["retired_converged"] == len(conv_live) and out["retired"] == kf_out["seeds_made"] - sum(o["promoted"] for o in outs)
    assert arm.live == []
    assert world.check_invariants(frames) == ZERO


def test_t5_dropped_linked_and_links_refused_never_raise():
    """A seed whose reference keypoint got linked meanwhile is dropped without a landmark; a
    measurement keypoint already linked elsewhere is not stolen (first link wins, counted)."""
    pts, poses, frames, _ = sweep(0.009, n_frames=8, noise_px=1.0)
    K0 = frames[0]
    world = Map(frames=frames)
    _link_true_landmarks(world, K0, pts, range(12))      # the scene range of K_0 (source "links")
    arm = _arm()
    keyframes = [(0, K0)]
    with contextlib.redirect_stdout(io.StringIO()):
        out0 = arm.on_keyframe(0, K0, keyframes, world, frames)
        for k in range(1, 4):
            arm.on_frame(k, frames[k], keyframes, world, frames)
    assert out0["seeds_made"] == 288 and len(arm.live) > 200
    live_kps = [s.kp for s in arm.live]
    # link two reference keypoints and one frame-4 measurement keypoint of a third seed to foreign landmarks
    foreign = []
    for kp in live_kps[:2]:
        p = Point(point=np.array(pts[kp]), descriptor=np.zeros(32, dtype=np.uint8))
        world.add_point(p)
        assert world.link(K0, kp, p)
        foreign.append(p)
    victim = next(s for s in arm.live if s.kp not in live_kps[:2] and len(s.measurements) >= 2)
    thief = Point(point=np.array(pts[victim.kp]), descriptor=np.zeros(32, dtype=np.uint8))
    world.add_point(thief)
    assert world.link(frames[2], victim.kp, thief)
    out, log = _capture(arm.on_frame, 4, frames[4], keyframes, world, frames)
    _, frs, prs = _parse(log)
    assert out["dropped_linked"] == 2 and all(s.kp not in live_kps[:2] for s in arm.live)
    assert all(len(p.observations) == 1 for p in foreign)
    with contextlib.redirect_stdout(io.StringIO()):
        outs = [arm.on_frame(k, frames[k], keyframes, world, frames) for k in range(5, 9)]
    rec = next((r for o in [out] + outs for r in o["promotions"] if r["kp"] == victim.kp), None)
    assert rec is not None, "the victim seed did not promote"
    assert rec["links_refused"] >= 1 and frames[2].keypoint_to_point[victim.kp] == thief.id
    assert frames[2].id not in rec["point"].observations
    assert "[DF-ERROR]" not in log
    assert world.check_invariants(frames) == ZERO


def test_t5_insertion_into_recent_keyframes_links_the_new_landmark():
    """C3: a landmark promoted from K_0's seed whose true keypoint in a later keyframe K_1 was not an
    inlier measurement (K_1 not yet measured) is inserted into K_1 by insert_new_points_into_neighbours."""
    pts, poses, frames, _ = sweep(0.009, n_frames=10, noise_px=0.3)
    K0 = frames[0]
    world = Map(frames=frames)
    _link_true_landmarks(world, K0, pts, range(12))      # the scene range of K_0 (source "links")
    arm = _arm()
    keyframes = [(0, K0), (10, frames[10])]           # K_1 = frame 10 is already a keyframe, never measured
    with contextlib.redirect_stdout(io.StringIO()):
        out0 = arm.on_keyframe(0, K0, [(0, K0)], world, frames)
        outs = [arm.on_frame(k, frames[k], keyframes, world, frames) for k in range(1, 10)]
    assert out0["seeds_made"] == 288
    recs = [r for o in outs for r in o["promotions"]]
    assert len(recs) > 100
    assert sum(o["inserted"] for o in outs) > 0.5 * len(recs)
    with_k1 = [r for r in recs if frames[10].id in r["point"].observations]
    assert len(with_k1) > 0.5 * len(recs)
    assert all(r["point"].observations[frames[10].id] == r["kp"] for r in with_k1)
    assert all(frames[10].id not in (m.frame.id for m in r["seed"].measurements if m.frame is not None) for r in recs)
    assert world.check_invariants(frames) == ZERO


# --------------------------------------------------------------------------- #
# T7. no-leak, and the driver stays out of local_mapping
# --------------------------------------------------------------------------- #
def test_t7_no_leak_and_module_boundaries():
    src = Path(dfa.__file__).read_text()
    for tok in ("groundtruth", "depth.txt", "import dataset", "from dataset", "cv2.imread", "depth/"):
        assert tok not in src, tok
    import local_mapping
    for name in ("DepthFilterArm", "DepthFilterArmParams", "scene_range", "make_seeds", "update_seeds", "promotable"):
        assert not hasattr(local_mapping, name), name
    import pipeline
    psrc = Path(pipeline.__file__).read_text()
    for tok in (".on_frame(", ".on_keyframe(", "process_new_keyframe(", "[DF-ERROR]"):
        assert tok in psrc, tok
    for tok in ("[DF-FRAME]", "[DF-KEYFRAME]", "[DF-PROMOTE]"):
        assert tok in src, tok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
