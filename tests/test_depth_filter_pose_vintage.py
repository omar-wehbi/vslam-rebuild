"""Stage 04y spec B: the pose vintage of the depth filter's distinct-baseline count (04i B6).

The batched count (DepthFilterArm._distinct_batch, used at convergence under --df_lifetime
r2) must read each registered frame's CURRENT cam_T_world and K, as the per-seed path
(depth_filter_lifetime.distinct_baselines) does; with no pose change the cached table, and
so every count, is as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import depth_filter as df  # noqa: E402
import depth_filter_lifetime as dfl  # noqa: E402
from depth_filter_arm import DepthFilterArm, DepthFilterArmParams  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


class _Frame:
    def __init__(self, fid, centre):
        self.id = fid
        self.K = K.copy()
        T = np.eye(4)
        T[:3, 3] = -np.asarray(centre, dtype=np.float64)
        self.cam_T_world = T
        self.world_T_cam = np.linalg.inv(T)

    def set_camera_pose(self, T):
        self.cam_T_world = np.array(T, dtype=np.float64, copy=True)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True


def _seed(frames_rows, z=4.0, ray=(0.0, 0.0, 1.0)):
    """A seed at range z on the reference ray from the origin, with one inlier measurement
    per (traj idx, frame, table row)."""
    ray = np.asarray(ray, dtype=np.float64)
    ray = ray / np.linalg.norm(ray)
    s = df.Seed(None, 0, 0, 0, 1.0, None, np.zeros(3), ray, 1.0 / z, 1e-6, 10.0, 10.0, 1.0)
    s.inl_rows = []
    for traj, f, row in frames_rows:
        s.measurements.append(df.Measurement(traj, f, 0, 1.0 / z, 1e-6, 1.0, 0.0))
        s.inl_rows.append(row)
    return s


def _arm_with_frames(xs):
    arm = DepthFilterArm(arm_params=DepthFilterArmParams(lifetime="r2"))
    frames = [_Frame(200 + i, [x, 0.0, 0.0]) for i, x in enumerate(xs)]
    rows = [arm._register_frame(i + 1, f) for i, f in enumerate(frames)]
    return arm, frames, rows


def _per_seed(seed, arm):
    return dfl.distinct_baselines(seed, arm.df_params, arm.arm_params)[0]


def test_v1_batched_count_reads_current_poses():
    # four frames 0.1 m apart: disparities 12.5, 25, 37.5, 50 px at 4 m (gap 9.5 px) -> 4 distinct
    arm, frames, rows = _arm_with_frames([0.1, 0.2, 0.3, 0.4])
    seed = _seed([(i + 1, f, r) for i, (f, r) in enumerate(zip(frames, rows))])
    other = _seed([(1, frames[0], rows[0]), (4, frames[3], rows[3])], z=6.0, ray=(0.05, 0.0, 1.0))
    before = arm._distinct_batch([seed, other]).tolist()
    assert before[0] == 4 == _per_seed(seed, arm)
    assert before[1] == _per_seed(other, arm)
    assert arm._tab_arrays is not None          # the table is cached at this point
    # BA moves three of the registered frames next to the first one: one distinct baseline left
    for f, dx in zip(frames[1:], (1e-4, 2e-4, 3e-4)):
        T = f.cam_T_world.copy()
        T[:3, 3] = -np.array([0.1 + dx, 0.0, 0.0])
        T[:3, :3] = df_rot(0.001 * dx * 1e4) @ T[:3, :3]
        f.set_camera_pose(T)
    after_per_seed = [_per_seed(seed, arm), _per_seed(other, arm)]
    assert after_per_seed[0] == 1                # the old and the new poses give different counts
    after = arm._distinct_batch([seed, other]).tolist()
    assert after == after_per_seed
    # every row the count used now equals its frame's current pose and K, bitwise
    R, t, Kt = arm._table()
    for f, r in zip(frames, rows):
        assert R[r].tobytes() == np.ascontiguousarray(f.cam_T_world[:3, :3]).tobytes()
        assert t[r].tobytes() == np.ascontiguousarray(f.cam_T_world[:3, 3]).tobytes()
        assert Kt[r].tobytes() == f.K.tobytes()


def test_v1_refresh_survives_a_later_registration_and_in_place_edits():
    arm, frames, rows = _arm_with_frames([0.1, 0.2, 0.3])
    seed = _seed([(i + 1, f, r) for i, (f, r) in enumerate(zip(frames, rows))])
    assert arm._distinct_batch([seed]).tolist() == [3]
    frames[2].set_camera_pose(frames[1].cam_T_world)            # rebinding setter
    late = _Frame(299, [0.9, 0.0, 0.0])
    arm._register_frame(10, late)                               # the cache is rebuilt from the lists
    assert arm._distinct_batch([seed]).tolist() == [_per_seed(seed, arm)] == [2]
    frames[1].cam_T_world[:3, 3] = [-0.9, 0.0, 0.0]               # an in-place edit: centre x = 0.9
    assert arm._distinct_batch([seed]).tolist() == [_per_seed(seed, arm)] == [3]


def test_v2_no_pose_change_keeps_the_cached_table():
    arm, frames, rows = _arm_with_frames([0.1, 0.2, 0.3, 0.4])
    seed = _seed([(i + 1, f, r) for i, (f, r) in enumerate(zip(frames, rows))])
    first = arm._distinct_batch([seed]).tolist()
    cached = arm._tab_arrays
    snap = tuple(a.copy() for a in cached)
    again = arm._distinct_batch([seed]).tolist()
    assert again == first == [_per_seed(seed, arm)] == [4]
    assert arm._tab_arrays is cached                             # not rebuilt
    for a, b in zip(arm._tab_arrays, snap):
        assert a.tobytes() == b.tobytes()                        # not rewritten
    # a setter call that writes the same pose changes nothing either
    frames[2].set_camera_pose(frames[2].cam_T_world.copy())
    assert arm._distinct_batch([seed]).tolist() == first
    for a, b in zip(arm._tab_arrays, snap):
        assert a.tobytes() == b.tobytes()


def df_rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
