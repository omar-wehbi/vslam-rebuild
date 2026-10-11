"""
Reprojection error computation (requirement 6 of תרגיל 3).

For each (frame, 3D-point) observation pair we project the 3D point into
the frame using  uv = K @ cam_T_world[:3,:] @ [X,Y,Z,1]  and compare with
the actually-observed keypoint pixel location.  The L2 distance in pixels
is the reprojection error.

This module is called BEFORE and AFTER Bundle Adjustment so we can report
the improvement.
"""

import numpy as np

from geometry import camera_pose_to_projection, project_point


def compute_reprojection_error(frames_dict, world_map, K):
    """
    Compute the mean reprojection error over all (frame, point) observations.

    Parameters
    ----------
    frames_dict : dict[int, Frame]
        Mapping from frame id to Frame object (only the frames in the
        optimisation window need to be supplied).
    world_map : Map
        The global map of 3D points.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    mean_error : float
        Mean reprojection error in pixels across all observations.
    n_obs : int
        Total number of observations used.
    """
    total_error = 0.0
    n_obs = 0

    for point in world_map.points:
        for fid, kp_idx in point.observations.items():
            frame = frames_dict.get(fid)
            if frame is None:
                continue

            # Build the 3×4 projection matrix for this frame.
            proj = camera_pose_to_projection(frame.cam_T_world, K)

            # Project the 3D world point into the frame.
            uv_proj = project_point(proj, point.point)

            # Observed keypoint location.
            uv_obs = np.array(frame.KeyPoints[kp_idx].pt, dtype=np.float64)

            total_error += np.linalg.norm(uv_proj - uv_obs)
            n_obs += 1

    if n_obs == 0:
        return 0.0, 0
    return total_error / n_obs, n_obs


def compute_per_point_reprojection_error(frames_dict, world_map, K):
    """
    Compute per-point mean reprojection error (in pixels) over observations that fall
    within frames_dict.

    Parameters
    ----------
    frames_dict : dict[int, Frame]
        Frames to consider (typically the current BA window).
    world_map : Map
        Global map.
    K : np.ndarray (3,3)
        Camera intrinsics.

    Returns
    -------
    dict[int, float]
        Mapping point_id -> mean reprojection error in pixels for that point.
        Points with zero usable observations in frames_dict are omitted.
    """
    per_point_sum = {}
    per_point_cnt = {}

    for point in world_map.points:
        pid = getattr(point, "id", None)
        if pid is None:
            continue
        for fid, kp_idx in point.observations.items():
            frame = frames_dict.get(fid)
            if frame is None:
                continue
            proj = camera_pose_to_projection(frame.cam_T_world, K)
            uv_proj = project_point(proj, point.point)
            uv_obs = np.array(frame.KeyPoints[kp_idx].pt, dtype=np.float64)
            err = float(np.linalg.norm(uv_proj - uv_obs))
            per_point_sum[pid] = per_point_sum.get(pid, 0.0) + err
            per_point_cnt[pid] = per_point_cnt.get(pid, 0) + 1

    out = {}
    for pid, s in per_point_sum.items():
        c = per_point_cnt.get(pid, 0)
        if c > 0:
            out[pid] = float(s) / float(c)
    return out
