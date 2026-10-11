"""Stage 3 unit checks: point fusion (T3.2.1) + Map eviction link cleanup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from mapping import (  # noqa: E402
    compute_map_observation_stats,
    count_duplicate_neighbours,
    fuse_duplicate_points,
)


class _FakeFrame:
    def __init__(self, fid):
        self.id = fid
        self.keypoint_to_point = {}


def test_fuse_identical_xyz_and_desc():
    """T3.2.1: two points same XYZ+desc → one after merge."""
    desc = np.zeros(32, dtype=np.uint8)
    desc[0] = 1
    p1 = Point(point=np.array([1.0, 2.0, 3.0]), frames=[0, 1],
               observations={0: 0, 1: 1}, descriptor=desc)
    p2 = Point(point=np.array([1.001, 2.0, 3.0]), frames=[2, 3],
               observations={2: 0, 3: 1}, descriptor=desc.copy())
    f0, f1, f2, f3 = (_FakeFrame(i) for i in range(4))
    f0.keypoint_to_point[0] = p1.id
    f1.keypoint_to_point[1] = p1.id
    f2.keypoint_to_point[0] = p2.id
    f3.keypoint_to_point[1] = p2.id
    frames = [f0, f1, f2, f3]
    world = Map(max_points=10000)
    world.add_point(p1)
    world.add_point(p2)
    assert len(world.points) == 2
    n = fuse_duplicate_points(world, frames, max_dist=0.05, max_desc_dist=40.0)
    assert n == 1, f"expected 1 merge, got {n}"
    assert len(world.points) == 1
    keeper = world.points[0]
    assert keeper.id == p1.id
    assert 2 in keeper.observations and 3 in keeper.observations
    assert f2.keypoint_to_point[0] == p1.id
    assert f3.keypoint_to_point[1] == p1.id


def test_map_eviction_clears_frame_links():
    """Stage 3.3: Map.add_point eviction cleans keypoint_to_point."""
    frames = [_FakeFrame(0)]
    world = Map(frames=frames, max_points=2)
    p1 = Point(point=np.array([0.0, 0.0, 1.0]), frames=[0], observations={0: 0})
    p2 = Point(point=np.array([0.0, 0.0, 2.0]), frames=[0], observations={0: 1})
    p3 = Point(point=np.array([0.0, 0.0, 3.0]), frames=[0], observations={0: 2})
    frames[0].keypoint_to_point[0] = p1.id
    frames[0].keypoint_to_point[1] = p2.id
    world.add_point(p1, frames=frames)
    world.add_point(p2, frames=frames)
    world.add_point(p3, frames=frames)
    assert len(world.points) == 2
    assert world.get_point(p1.id) is None
    assert 0 not in frames[0].keypoint_to_point
    assert frames[0].keypoint_to_point.get(1) == p2.id


def test_duplicate_neighbour_count_and_obs_stats():
    world = Map(max_points=10000)
    for offset in (0.0, 0.01, 1.0):
        world.add_point(Point(
            point=np.array([offset, 0.0, 1.0]),
            frames=[0, 1] if offset < 0.5 else [0],
            observations={0: 0} if offset >= 0.5 else {0: 0, 1: 1},
            descriptor=np.zeros(32, dtype=np.uint8),
        ))
    pairs, n = count_duplicate_neighbours(world, radius=0.05)
    assert n == 3
    assert pairs >= 1  # first two within 0.05
    stats = compute_map_observation_stats(world)
    assert stats["n_points"] == 3
    assert stats["median_obs"] >= 1.0


if __name__ == "__main__":
    test_fuse_identical_xyz_and_desc()
    test_map_eviction_clears_frame_links()
    test_duplicate_neighbour_count_and_obs_stats()
    print("All Stage 3 unit tests passed.")
