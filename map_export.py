"""
Stage 03d (spec A1): export the final map (points + their observations) to a text
file next to the trajectory, so the depth protocol (design.md#open-questions) has
something to measure. Standalone: no import of pipeline.py or visualizer.py.

File format (# map_points v1):
    # map_points v1
    # frames <n_frames>
    # keyframes <k1>,<k2>,...
    # n_points <n>
    # columns: point_id x y z created_k last_seen_k n_obs obs...
    # obs: k:kp:u:v (k = trajectory index = Frame.id - 1; kp = keypoint index in
    #                that frame; u v = KeyPoints[kp].pt in the raw image, %.3f)
    <point_id> <x> <y> <z> <created_k> <last_seen_k> <n_obs> <k:kp:u:v> ...
"""
from __future__ import annotations

from pathlib import Path


def write_map_points(path, world_map, frames, keyframe_indices):
    """
    Write the final map (world_map.points, in list order) with its observations.

    An observation whose frame cannot be resolved (Map.resolve_frame against
    `frames`) or whose keypoint index is out of range for that frame is dropped
    (never written) and counted towards the printed summary line.

    Returns (n_points, n_obs): n_obs is the number of observations actually
    written (after drops).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = list(world_map.points)
    body_lines = []
    n_obs_written = 0
    n_dropped = 0

    for pt in points:
        entries = []
        for fid in sorted(pt.observations.keys()):
            kp_idx = pt.observations[fid]
            frame = world_map.resolve_frame(fid, frames)
            if frame is None:
                n_dropped += 1
                continue
            keypoints = getattr(frame, "KeyPoints", None) or []
            if kp_idx < 0 or kp_idx >= len(keypoints):
                n_dropped += 1
                continue
            u, v = keypoints[kp_idx].pt
            k = int(frame.id) - 1
            entries.append((k, int(kp_idx), float(u), float(v)))
        entries.sort(key=lambda e: e[0])

        created_fid = int(getattr(pt, "created_fid", -1))
        last_seen_fid = int(getattr(pt, "last_seen_fid", -1))
        created_k = created_fid - 1 if created_fid >= 0 else -1
        last_seen_k = last_seen_fid - 1 if last_seen_fid >= 0 else -1

        obs_str = " ".join(f"{k}:{kp}:{u:.3f}:{v:.3f}" for k, kp, u, v in entries)
        line = (
            f"{pt.id} {float(pt.x):.9g} {float(pt.y):.9g} {float(pt.z):.9g} "
            f"{created_k} {last_seen_k} {len(entries)}"
        )
        if obs_str:
            line += " " + obs_str
        body_lines.append(line)
        n_obs_written += len(entries)

    n_points = len(points)
    keyframes_csv = ",".join(str(int(k)) for k in keyframe_indices)

    with open(path, "w") as f:
        f.write("# map_points v1\n")
        f.write(f"# frames {len(frames)}\n")
        f.write(f"# keyframes {keyframes_csv}\n")
        f.write(f"# n_points {n_points}\n")
        f.write("# columns: point_id x y z created_k last_seen_k n_obs obs...\n")
        f.write(
            "# obs: k:kp:u:v (k = trajectory index = Frame.id - 1; kp = keypoint "
            "index in that frame; u v = KeyPoints[kp].pt in the raw image, %.3f)\n"
        )
        for line in body_lines:
            f.write(line + "\n")

    print(
        f"  >> Saved map points ({n_points} points, {n_obs_written} observations, "
        f"{n_dropped} dropped) to {path}"
    )
    return n_points, n_obs_written
