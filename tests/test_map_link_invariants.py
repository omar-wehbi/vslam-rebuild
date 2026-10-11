"""
Stage 01b: the frame link table (Frame.keypoint_to_point) and the landmark
observation table (Point.observations / Point.frames) are mutual inverses,
maintained by construction through Map.link / unlink / remove_point /
add_point (conventions.md#landmark-and-observation-consistency).

Every test ends by asserting Map.check_invariants() is all zeros, and the
scenario tests cross-check that against the verifier's own
check_map_consistency (system/verify/checks/map_invariants.py).
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from mapping import (  # noqa: E402
    associate_known_points,
    fuse_duplicate_points,
    local_map_track,
    triangulate_and_add_points,
)


def _load_verifier_check():
    path = ROOT / "system" / "verify" / "checks" / "map_invariants.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_map_invariants_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_map_consistency


VERIFIER_CHECK = _load_verifier_check()

ZERO = {
    "inv_link_to_missing_point": 0,
    "inv_link_without_observation": 0,
    "inv_observation_without_link": 0,
    "inv_frame_double_link": 0,
    "inv_frames_obs_mismatch": 0,
}


def _assert_consistent(world, frames):
    counts = world.check_invariants(frames)
    assert counts == ZERO, counts
    if VERIFIER_CHECK is not None:
        assert VERIFIER_CHECK(world, frames) == ZERO


# --------------------------------------------------------------------------
# Minimal frames: a table-only frame for the API tests, a geometric frame
# (keypoints, descriptors, intrinsics, pose) for the mapping.py scenarios.
# --------------------------------------------------------------------------
class _Frame:
    def __init__(self, fid):
        self.id = fid
        self.keypoint_to_point = {}


K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 240.0],
              [0.0, 0.0, 1.0]])


class _GeoFrame(_Frame):
    def __init__(self, fid, cam_centre, pts_world, descs):
        super().__init__(fid)
        self.K = K
        self.world_T_cam = np.eye(4)
        self.world_T_cam[:3, 3] = np.asarray(cam_centre, dtype=np.float64)
        self.cam_T_world = np.linalg.inv(self.world_T_cam)
        self.img_shape = (480, 640)
        self.method = "ORB"
        self.match_type = "BF"
        self.KeyPoints = []
        for X in pts_world:
            xc = self.cam_T_world[:3, :3] @ X + self.cam_T_world[:3, 3]
            uv = K @ xc
            u, v = float(uv[0] / uv[2]), float(uv[1] / uv[2])
            assert 0 <= u < 640 and 0 <= v < 480, (u, v)
            self.KeyPoints.append(cv2.KeyPoint(u, v, 7.0))
        self.Descriptors = np.array(descs, dtype=np.uint8, copy=True)


def _scene(n=12, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-0.8, 0.8, n),
                           rng.uniform(-0.6, 0.6, n),
                           rng.uniform(3.0, 5.0, n)])
    descs = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    # Three cameras along x: parallax between neighbours ~4 deg at depth 4.
    f0 = _GeoFrame(10, [-0.3, 0.0, 0.0], pts, descs)
    f1 = _GeoFrame(11, [0.0, 0.0, 0.0], pts, descs)
    f2 = _GeoFrame(12, [0.3, 0.0, 0.0], pts, descs)
    return pts, descs, f0, f1, f2


def _matches(n):
    return [cv2.DMatch(i, i, 0.0) for i in range(n)]


def _pt(xyz=(0.0, 0.0, 1.0), **kw):
    return Point(point=np.asarray(xyz, dtype=np.float64), **kw)


# --------------------------------------------------------------------------
# A. Map API
# --------------------------------------------------------------------------
def test_link_unlink_round_trip():
    f = _Frame(1)
    world = Map(frames=[f])
    p = _pt()
    world.add_point(p)

    assert world.link(f, 4, p) is True
    assert f.keypoint_to_point == {4: p.id}
    assert p.observations == {1: 4}
    assert p.frames == [1]
    assert p.created_fid == 1 and p.last_seen_fid == 1
    # Idempotent.
    assert world.link(f, 4, p) is True
    assert f.keypoint_to_point == {4: p.id} and p.observations == {1: 4} and p.frames == [1]
    _assert_consistent(world, [f])

    assert world.unlink(f, 4) is True
    assert f.keypoint_to_point == {}
    assert p.observations == {} and p.frames == []
    # No link left: unlink reports False, nothing changes.
    assert world.unlink(f, 4) is False
    assert f.keypoint_to_point == {} and p.observations == {}
    _assert_consistent(world, [f])


def test_first_link_wins_keypoint_already_linked():
    f = _Frame(1)
    world = Map(frames=[f])
    p, q = _pt(), _pt((1.0, 0.0, 1.0))
    world.add_point(p)
    world.add_point(q)
    assert world.link(f, 3, p)
    # Same keypoint, different point: refused, both tables untouched.
    assert world.link(f, 3, q) is False
    assert f.keypoint_to_point == {3: p.id}
    assert p.observations == {1: 3}
    assert q.observations == {} and q.frames == []
    _assert_consistent(world, [f])


def test_first_link_wins_point_already_has_keypoint_in_frame():
    f = _Frame(1)
    world = Map(frames=[f])
    p = _pt()
    world.add_point(p)
    assert world.link(f, 3, p)
    # Same point, different keypoint of the same frame: refused (would be a
    # double link); nothing changes.
    assert world.link(f, 8, p) is False
    assert f.keypoint_to_point == {3: p.id}
    assert p.observations == {1: 3} and p.frames == [1]
    _assert_consistent(world, [f])


def test_link_refuses_point_not_in_map():
    f = _Frame(1)
    world = Map(frames=[f])
    p = _pt()  # never added
    assert world.link(f, 0, p) is False
    assert f.keypoint_to_point == {} and p.observations == {}
    _assert_consistent(world, [f])


def test_remove_point_clears_all_sides():
    f1, f2, f3 = _Frame(1), _Frame(2), _Frame(3)
    frames = [f1, f2, f3]
    world = Map(frames=frames)
    p, q = _pt(), _pt((1.0, 0.0, 1.0))
    world.add_point(p)
    world.add_point(q)
    assert world.link(f1, 0, p) and world.link(f2, 5, p) and world.link(f3, 7, p)
    assert world.link(f1, 1, q) and world.link(f2, 6, q)

    world.remove_point(p.id)
    assert world.get_point(p.id) is None
    assert all(pt.id != p.id for pt in world.points)
    assert f1.keypoint_to_point == {1: q.id}
    assert f2.keypoint_to_point == {6: q.id}
    assert f3.keypoint_to_point == {}
    assert p.observations == {} and p.frames == []
    assert q.observations == {1: 1, 2: 6}
    _assert_consistent(world, frames)


def test_remove_point_resolves_frames_outside_map_list():
    # Frames unknown to the map are resolved through the ``frames`` argument.
    f = _Frame(1)
    world = Map()
    p = _pt()
    world.add_point(p, frames=[f])
    assert world.link(f, 2, p)
    world.remove_point(p.id, frames=[f])
    assert f.keypoint_to_point == {}
    assert world.points == []
    _assert_consistent(world, [f])


def test_add_point_mirrors_constructor_observations_and_drops_unmirrorable():
    f1, f2, f3 = _Frame(1), _Frame(2), _Frame(3)
    frames = [f1, f2, f3]
    world = Map(frames=frames)
    other = _pt()
    world.add_point(other)
    assert world.link(f2, 4, other)  # f2 kp 4 is taken before p arrives

    # frames list deliberately out of sync with observations (3 has no
    # observation; 2 collides with `other`).
    p = _pt((0.5, 0.0, 2.0), frames=[1, 2, 3], observations={1: 9, 2: 4})
    world.add_point(p)

    assert f1.keypoint_to_point == {9: p.id}          # mirrored
    assert f2.keypoint_to_point == {4: other.id}      # first link wins
    assert f3.keypoint_to_point == {}
    assert p.observations == {1: 9}                   # collision dropped
    assert p.frames == [1]                            # (iv) restored
    assert other.observations == {2: 4}
    _assert_consistent(world, frames)


def test_add_point_keeps_observations_of_unknown_frames():
    # A frame the map cannot resolve cannot be mirrored, but also cannot be
    # contradicted by any frame table; the observation stays (this is what
    # tests/test_stage3_map.py::test_fuse_identical_xyz_and_desc relies on).
    world = Map()
    p = _pt(frames=[7], observations={7: 3})
    world.add_point(p)
    assert p.observations == {7: 3} and p.frames == [7]
    assert world.check_invariants([]) == ZERO


def test_eviction_goes_through_remove_point():
    f = _Frame(1)
    world = Map(frames=[f], max_points=2)
    p1, p2, p3 = _pt(), _pt((0, 0, 2.0)), _pt((0, 0, 3.0))
    world.add_point(p1)
    assert world.link(f, 0, p1)
    world.add_point(p2)
    assert world.link(f, 1, p2)
    world.add_point(p3)                       # evicts p1, must clear f[0]
    assert world.get_point(p1.id) is None
    assert p1.observations == {} and p1.frames == []
    assert f.keypoint_to_point == {1: p2.id}
    assert world.link(f, 0, p1) is False      # evicted point cannot be linked
    assert world.link(f, 2, p3)
    world.add_point(_pt((0, 0, 4.0)))         # evicts p2, must clear f[1]
    assert world.get_point(p2.id) is None
    assert f.keypoint_to_point == {2: p3.id}
    assert len(world.points) == 2
    _assert_consistent(world, [f])


def test_frame_by_id_sees_frames_appended_after_construction():
    # The pipeline streams: it appends to the list it built the Map with.
    frames = [_Frame(1)]
    world = Map(frames=frames)
    assert world.frame_by_id(1) is frames[0]
    assert world.frame_by_id(2) is None
    frames.append(_Frame(2))
    assert world.frame_by_id(2) is frames[1]
    p = _pt()
    world.add_point(p)
    assert world.link(frames[1], 0, p)
    world.remove_point(p.id)                  # resolves frame 2 via frame_by_id
    assert frames[1].keypoint_to_point == {}


def test_check_invariants_counts_each_violation():
    """The executor-side copy must flag exactly what the verifier flags."""
    f1, f2 = _Frame(1), _Frame(2)
    frames = [f1, f2]
    world = Map(frames=frames)
    p, q = _pt(), _pt((1, 0, 1))
    world.add_point(p)
    world.add_point(q)
    assert world.link(f1, 0, p) and world.link(f2, 3, q)
    assert world.check_invariants() == ZERO

    # Break the tables by hand, one invariant at a time.
    f1.keypoint_to_point[5] = 999                       # (i) missing point
    f1.keypoint_to_point[6] = q.id                      # (i) link w/o observation
    p.observations[2] = 7                               # (ii) observation w/o link
    f2.keypoint_to_point[8] = q.id                      # (iii) double link on q in f2
    q.frames.append(42)                                 # (iv) frames/obs mismatch
    expected = {
        "inv_link_to_missing_point": 1,
        "inv_link_without_observation": 2,   # f1[6]->q and f2[8]->q
        "inv_observation_without_link": 1,
        "inv_frame_double_link": 1,
        "inv_frames_obs_mismatch": 2,        # p (obs 2, frames 1) and q (obs 1, frames 2)
    }
    assert world.check_invariants() == expected
    if VERIFIER_CHECK is not None:
        assert VERIFIER_CHECK(world, frames) == expected


# --------------------------------------------------------------------------
# B. mapping.py scenarios
# --------------------------------------------------------------------------
def test_triangulation_over_local_map_link_creates_no_duplicate():
    """
    A keypoint of the current frame gets linked by local_map_track (to a
    landmark observed by an older frame); triangulate_and_add_points on the
    same keypoint must add prev's observation to that landmark instead of
    creating a second one, and the invariant must hold.
    """
    pts, descs, f0, f1, f2 = _scene()
    frames = [f0, f1, f2]
    world = Map(frames=frames)

    # Landmark 0 already in the map, observed by the older frame f0 only.
    p0 = Point(point=pts[0].copy(), descriptor=descs[0])
    world.add_point(p0)
    assert world.link(f0, 0, p0)

    n_local = local_map_track(f2, world, [f0], search_radius=25.0)
    assert n_local == 1
    assert f2.keypoint_to_point == {0: p0.id}
    assert f1.keypoint_to_point == {}

    n_before = len(world.points)
    triangulate_and_add_points(f1, f2, _matches(len(pts)), world, min_parallax_deg=1.5)

    # Match 0: curr keypoint taken -> no new point; prev observation added.
    assert len(world.points) == n_before + (len(pts) - 1)
    assert f2.keypoint_to_point[0] == p0.id
    assert f1.keypoint_to_point[0] == p0.id
    assert p0.observations == {f0.id: 0, f1.id: 0, f2.id: 0}
    assert sorted(p0.frames) == [f0.id, f1.id, f2.id]
    # The other matches each produced one fully linked landmark.
    for i in range(1, len(pts)):
        pid = f1.keypoint_to_point[i]
        assert f2.keypoint_to_point[i] == pid
        pt = world.get_point(pid)
        assert pt.observations == {f1.id: i, f2.id: i}
        assert pt.created_fid == f2.id and pt.last_seen_fid == f2.id
    _assert_consistent(world, frames)


def test_triangulation_skips_match_when_prev_keypoint_is_linked():
    pts, descs, f0, f1, f2 = _scene()
    frames = [f0, f1, f2]
    world = Map(frames=frames)

    # prev keypoint 0 -> landmark A; curr keypoint 0 -> landmark B (both taken).
    a = Point(point=pts[0].copy(), descriptor=descs[0])
    b = Point(point=pts[0].copy() + 0.01, descriptor=descs[0])
    world.add_point(a)
    world.add_point(b)
    assert world.link(f1, 0, a) and world.link(f2, 0, b)
    # prev keypoint 1 -> landmark C, curr keypoint 1 free.
    c = Point(point=pts[1].copy(), descriptor=descs[1])
    world.add_point(c)
    assert world.link(f1, 1, c)

    import Point as point_module
    id_before = point_module.ID
    triangulate_and_add_points(f1, f2, _matches(len(pts)), world, min_parallax_deg=1.5)

    # Matches 0 and 1 skipped: no new landmark, no link written on either side.
    assert f1.keypoint_to_point[0] == a.id and f2.keypoint_to_point[0] == b.id
    assert a.observations == {f1.id: 0} and b.observations == {f2.id: 0}
    assert f1.keypoint_to_point[1] == c.id and 1 not in f2.keypoint_to_point
    assert c.observations == {f1.id: 1}
    assert len(world.points) == 3 + (len(pts) - 2)
    # Skipped matches are skipped, not created-then-removed: exactly one
    # landmark id was consumed per accepted match.
    assert point_module.ID - id_before == len(pts) - 2
    # skip_if_curr_already_linked is accepted and ignored; now every prev
    # keypoint is linked, so nothing is created and no id is consumed.
    id_before = point_module.ID
    triangulate_and_add_points(f1, f2, _matches(len(pts)), world,
                               skip_if_curr_already_linked=True)
    assert len(world.points) == 3 + (len(pts) - 2)
    assert point_module.ID == id_before
    _assert_consistent(world, frames)


def test_triangulation_prev_link_refused_when_prev_keypoint_taken_by_other_point():
    """
    Curr keypoint linked to P, prev keypoint linked to Q (a different point):
    first-link-wins keeps both as they are and creates nothing.
    """
    pts, descs, f0, f1, f2 = _scene()
    frames = [f0, f1, f2]
    world = Map(frames=frames)
    p = Point(point=pts[0].copy(), descriptor=descs[0])
    q = Point(point=pts[0].copy() + 0.02, descriptor=descs[0])
    world.add_point(p)
    world.add_point(q)
    assert world.link(f2, 0, p) and world.link(f1, 0, q)
    triangulate_and_add_points(f1, f2, _matches(len(pts)), world)
    assert f1.keypoint_to_point[0] == q.id and f2.keypoint_to_point[0] == p.id
    assert p.observations == {f2.id: 0} and q.observations == {f1.id: 0}
    _assert_consistent(world, frames)


def test_associate_known_points_refused_link_is_not_a_triangulation_candidate():
    f1, f2 = _Frame(1), _Frame(2)
    world = Map(frames=[f1, f2])
    p, q = _pt(), _pt((1, 0, 1))
    world.add_point(p)
    world.add_point(q)
    assert world.link(f1, 0, p)            # prev kp 0 -> p
    assert world.link(f2, 5, q)            # curr kp 5 already taken by q
    assert world.link(f1, 1, q)            # prev kp 1 -> q (q already has kp 5 in curr)
    matches = [cv2.DMatch(0, 5, 0.0),      # refused: curr keypoint taken
               cv2.DMatch(1, 6, 0.0),      # refused: q already has a keypoint in curr
               cv2.DMatch(2, 7, 0.0),      # prev unlinked: candidate
               cv2.DMatch(0, 8, 0.0)]      # p has no keypoint in curr yet: transfers
    unmatched = associate_known_points(f1, f2, matches, world)
    assert [(m.queryIdx, m.trainIdx) for m in unmatched] == [(2, 7)]
    assert f2.keypoint_to_point == {5: q.id, 8: p.id}
    assert p.observations == {1: 0, 2: 8}
    assert q.observations == {2: 5, 1: 1}
    _assert_consistent(world, [f1, f2])


def test_fusion_keeper_already_has_keypoint_in_duplicate_frame():
    """
    Keeper observes frame 1 at kp 3; duplicate observes frame 1 at kp 7 and
    frame 2 at kp 4.  After fusion the keeper has exactly one keypoint in
    frame 1 (its own), kp 7 is free, frame 2's link moved to the keeper.
    """
    f1, f2 = _Frame(1), _Frame(2)
    frames = [f1, f2]
    world = Map(frames=frames)
    desc = np.zeros(32, dtype=np.uint8)
    keeper = Point(point=np.array([1.0, 2.0, 3.0]), descriptor=desc)
    dup = Point(point=np.array([1.001, 2.0, 3.0]), descriptor=desc.copy())
    world.add_point(keeper)
    world.add_point(dup)
    assert world.link(f1, 3, keeper)
    assert world.link(f1, 7, dup) and world.link(f2, 4, dup)
    # Pre-01b weights: keeper's frame count after the union (2), dup's (2).
    xyz_expected = (2 * keeper.point + 2 * dup.point) / 4.0

    n = fuse_duplicate_points(world, frames, max_dist=0.05, max_desc_dist=40.0)
    assert n == 1
    assert [p.id for p in world.points] == [keeper.id]
    assert f1.keypoint_to_point == {3: keeper.id}      # 7 is free now
    assert f2.keypoint_to_point == {4: keeper.id}
    assert keeper.observations == {1: 3, 2: 4}
    assert sorted(keeper.frames) == [1, 2]
    assert dup.observations == {} and dup.frames == []
    np.testing.assert_allclose(keeper.point, xyz_expected)  # weights unchanged
    _assert_consistent(world, frames)


def test_end_to_end_link_fuse_prune_sequence():
    """Random sequence of link / fuse / prune / evict; invariants stay at zero."""
    rng = random.Random(1234)
    frames = [_Frame(i) for i in range(1, 9)]
    world = Map(frames=frames, max_points=60)
    desc = np.zeros(32, dtype=np.uint8)

    # Clusters of near-duplicate landmarks so fusion has work to do.
    points = []
    for c in range(30):
        centre = np.array([c * 1.0, 0.0, 2.0])
        for k in range(3):
            p = Point(point=centre + rng.uniform(0.0, 0.01), descriptor=desc)
            world.add_point(p)
            points.append(p)
    assert len(world.points) == 60

    n_link_ok = n_link_refused = 0
    for _ in range(600):
        f = rng.choice(frames)
        p = rng.choice(points)
        if world.get_point(p.id) is None:
            continue
        if world.link(f, rng.randrange(0, 40), p):
            n_link_ok += 1
        else:
            n_link_refused += 1
    assert n_link_ok > 100 and n_link_refused > 20   # both branches exercised
    _assert_consistent(world, frames)

    n_merged = fuse_duplicate_points(world, frames, max_dist=0.05, max_desc_dist=40.0)
    assert n_merged >= 30
    _assert_consistent(world, frames)

    # Prune (pipeline-style: remove_point on a snapshot) every third point.
    pruned = 0
    for pt in list(world.points)[::3]:
        world.remove_point(pt.id)
        pruned += 1
    assert pruned > 0
    assert all(pt.observations == {} and pt.frames == [] for pt in points
               if world.get_point(pt.id) is None)
    _assert_consistent(world, frames)

    # Some unlinks, then more links, then eviction by max_points.
    for f in frames:
        for kp in list(f.keypoint_to_point)[::2]:
            assert world.unlink(f, kp)
    _assert_consistent(world, frames)
    for i in range(80):
        p = Point(point=np.array([100.0 + i, 0.0, 2.0]), descriptor=desc)
        world.add_point(p)
        for f in rng.sample(frames, 3):
            world.link(f, rng.randrange(0, 40), p)
    assert len(world.points) == 60
    _assert_consistent(world, frames)
    # Nothing links to an evicted point.
    live = set(world.point_by_id)
    for f in frames:
        assert set(f.keypoint_to_point.values()) <= live


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
