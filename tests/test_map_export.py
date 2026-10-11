"""
Stage 03d (spec B): map_export.write_map_points writes the final map (points +
observations) in the "# map_points v1" format that system/verify/checks/depth_error.py
parses. These tests build the format's executable definition (a small parser, B2)
and round-trip it, so a corrupted writer is caught by re-reading its own output.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Map import Map  # noqa: E402
from Point import Point  # noqa: E402
from map_export import write_map_points  # noqa: E402


class _FakeKeyPoint:
    def __init__(self, u, v):
        self.pt = (u, v)


class _FakeFrame:
    """Just enough of Frame for write_map_points: .id, .KeyPoints, .keypoint_to_point."""

    def __init__(self, id_, keypoints):
        self.id = id_
        self.KeyPoints = keypoints
        self.keypoint_to_point = {}


def _parse_map_points(path):
    """
    B2: the format's executable definition. Returns (header, {point_id: (xyz,
    {k: (kp, u, v)})}); the harness's depth_error.py parser is equivalent.
    """
    header = {}
    points = {}
    for line in Path(path).read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.startswith("map_points"):
                header["version"] = body.split()[-1]
            elif body.startswith("frames "):
                header["frames"] = int(body.split()[1])
            elif body.startswith("keyframes"):
                rest = body[len("keyframes"):].strip()
                header["keyframes"] = [int(x) for x in rest.split(",") if x.strip()] if rest else []
            elif body.startswith("n_points "):
                header["n_points"] = int(body.split()[1])
            continue
        fields = line.split()
        pid = int(fields[0])
        xyz = (float(fields[1]), float(fields[2]), float(fields[3]))
        created_k, last_seen_k, n_obs = int(fields[4]), int(fields[5]), int(fields[6])
        obs_tokens = fields[7:7 + n_obs]
        assert len(obs_tokens) == n_obs
        obs = {}
        for tok in obs_tokens:
            k_s, kp_s, u_s, v_s = tok.split(":")
            obs[int(k_s)] = (int(kp_s), float(u_s), float(v_s))
        points[pid] = {"xyz": xyz, "created_k": created_k, "last_seen_k": last_seen_k, "obs": obs}
    return header, points


def _build_scenario():
    """
    Two frames (trajectory index 0 -> Frame.id 1, index 1 -> Frame.id 2), three
    points: two observed in both frames, one observed in frame 2 only, plus one
    observation on an unresolvable frame id (999) that must be dropped.
    """
    frame1 = _FakeFrame(1, [_FakeKeyPoint(10.0, 20.0), _FakeKeyPoint(11.0, 21.0)])
    frame2 = _FakeFrame(2, [_FakeKeyPoint(30.0, 40.0), _FakeKeyPoint(31.0, 41.0), _FakeKeyPoint(32.0, 42.0)])
    world_map = Map(points=[], frames=[frame1, frame2])

    point_a = Point([1.0, 2.0, 3.0])
    world_map.add_point(point_a)
    assert world_map.link(frame1, 0, point_a)
    assert world_map.link(frame2, 0, point_a)
    # An observation pointing at a frame the map cannot resolve: must be dropped,
    # not written, and counted (the unknown frame is never in world_map.frames).
    unknown_frame = _FakeFrame(999, [_FakeKeyPoint(0.0, 0.0)])
    assert world_map.link(unknown_frame, 0, point_a)

    point_b = Point([4.0, 5.0, 6.0])
    world_map.add_point(point_b)
    assert world_map.link(frame1, 1, point_b)
    assert world_map.link(frame2, 1, point_b)

    point_c = Point([7.0, 8.0, 9.0])
    world_map.add_point(point_c)
    assert world_map.link(frame2, 2, point_c)

    return world_map, frame1, frame2, point_a, point_b, point_c


def test_map_export_format_and_counts(tmp_path):
    world_map, frame1, frame2, point_a, point_b, point_c = _build_scenario()
    out_path = tmp_path / "map_points.txt"

    n_points, n_obs = write_map_points(
        out_path, world_map, [frame1, frame2], keyframe_indices=[0, 1],
    )

    assert n_points == 3
    # point_a: 2 valid obs, point_b: 2, point_c: 1 -> 5 written; the frame-999 obs dropped.
    assert n_obs == 5

    header, points = _parse_map_points(out_path)
    assert header["version"] == "v1"
    assert header["frames"] == 2
    assert header["keyframes"] == [0, 1]
    assert header["n_points"] == 3
    assert len(points) == 3

    # point_a: frame1 (k=0, kp=0) and frame2 (k=1, kp=0); the dropped obs (frame 999) absent.
    pa = points[point_a.id]
    assert pa["xyz"] == (1.0, 2.0, 3.0)
    assert set(pa["obs"].keys()) == {0, 1}
    assert pa["obs"][0] == (0, 10.0, 20.0)
    assert pa["obs"][1] == (0, 30.0, 40.0)
    assert pa["created_k"] == point_a.created_fid - 1
    assert pa["last_seen_k"] == point_a.last_seen_fid - 1

    pb = points[point_b.id]
    assert set(pb["obs"].keys()) == {0, 1}
    assert pb["obs"][0] == (1, 11.0, 21.0)
    assert pb["obs"][1] == (1, 31.0, 41.0)

    pc = points[point_c.id]
    assert set(pc["obs"].keys()) == {1}
    assert pc["obs"][1] == (2, 32.0, 42.0)


def test_dropped_observation_never_appears_anywhere(tmp_path):
    """Mutation-style check: the frame-999 observation must not survive under any point id."""
    world_map, frame1, frame2, point_a, *_ = _build_scenario()
    out_path = tmp_path / "map_points.txt"
    write_map_points(out_path, world_map, [frame1, frame2], keyframe_indices=[0, 1])

    _, points = _parse_map_points(out_path)
    for pdata in points.values():
        for k in pdata["obs"]:
            assert k != 999


def test_empty_map_writes_zero_points(tmp_path):
    empty_map = Map(points=[], frames=[])
    out_path = tmp_path / "map_points.txt"

    n_points, n_obs = write_map_points(out_path, empty_map, [], keyframe_indices=[])

    assert (n_points, n_obs) == (0, 0)
    header, points = _parse_map_points(out_path)
    assert header["version"] == "v1"
    assert header["n_points"] == 0
    assert header["keyframes"] == []
    assert points == {}
