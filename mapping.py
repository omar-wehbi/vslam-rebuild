import cv2
import numpy as np

from geometry import camera_pose_to_projection, project_point
from Point import Point


def local_map_track(curr_frame, world_map, recent_frames,
                    search_radius=25.0, max_desc_dist=None,
                    max_depth_ratio=5.0):
    """
    Project map points from *recent_frames* into the current frame and
    match by descriptor inside a pixel window around each projection.
    New 2D-3D correspondences are written into
    ``curr_frame.keypoint_to_point``, giving PnP many more inliers than
    frame-to-frame association alone.

    Parameters
    ----------
    curr_frame : Frame
        Current frame (must already have a chained pose estimate).
    world_map : Map
        The global 3D point map.
    recent_frames : list[Frame]
        The last N processed frames (e.g. ``frames[k-10 : k]``).
    search_radius : float
        Pixel radius around the projected location to search for
        descriptor matches. Stage 3 default 25 (was 40; looser admits
        more wrong 2D–3D links — DIAGNOSIS §5).
    max_desc_dist : float | None
        Maximum descriptor distance to accept a match.  If None a
        sensible default is chosen per method (ORB: 48, SIFT: 150).
        Stage 3 tightened ORB from 64 (DIAGNOSIS §5 / package 3.1).
    max_depth_ratio : float | None
        If > 0, reject a match when the candidate's camera depth differs
        from the median depth of already-associated current-frame points
        by more than this factor (and there are enough associated depths).
        Set None/<=0 to disable.

    Returns
    -------
    int
        Number of new correspondences added.
    """
    if (not recent_frames
            or curr_frame.Descriptors is None
            or len(curr_frame.KeyPoints) == 0):
        return 0

    K = curr_frame.K
    R_cw = curr_frame.cam_T_world[:3, :3]
    t_cw = curr_frame.cam_T_world[:3, 3]

    already_associated = set(curr_frame.keypoint_to_point.values())
    assigned_kp = set(curr_frame.keypoint_to_point.keys())

    candidates = {}  # point_id -> (Point, Frame, kp_idx_in_frame)
    for rf in recent_frames:
        if rf.Descriptors is None:
            continue
        for kp_idx, pt_id in rf.keypoint_to_point.items():
            if pt_id in already_associated:
                continue
            pt = world_map.get_point(pt_id)
            if pt is None:
                continue
            candidates[pt_id] = (pt, rf, kp_idx)

    if not candidates:
        return 0

    cand_ids = list(candidates.keys())
    cand_data = [candidates[pid] for pid in cand_ids]
    pts_3d = np.array([c[0].point for c in cand_data], dtype=np.float64)

    pts_cam = (R_cw @ pts_3d.T).T + t_cw
    in_front = pts_cam[:, 2] > 0.1

    uvw = (K @ pts_cam.T).T
    uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    h, w = curr_frame.img_shape[:2]
    visible = (
        in_front
        & (uv[:, 0] >= -search_radius) & (uv[:, 0] < w + search_radius)
        & (uv[:, 1] >= -search_radius) & (uv[:, 1] < h + search_radius)
    )

    if not np.any(visible):
        return 0

    kp_xy = np.array([kp.pt for kp in curr_frame.KeyPoints], dtype=np.float64)

    if max_desc_dist is None:
        max_desc_dist = 48 if curr_frame.method == "ORB" else 150.0
    norm_type = cv2.NORM_HAMMING if curr_frame.method == "ORB" else cv2.NORM_L2

    # Median depth of points already linked to the current frame (depth gate).
    median_depth = None
    if max_depth_ratio is not None and max_depth_ratio > 0:
        depths = []
        for pid in already_associated:
            pt = world_map.get_point(pid)
            if pt is None:
                continue
            xc = R_cw @ np.asarray(pt.point, dtype=np.float64) + t_cw
            if xc[2] > 0.1 and np.isfinite(xc[2]):
                depths.append(float(xc[2]))
        if len(depths) >= 5:
            median_depth = float(np.median(depths))

    vis_idx = np.where(visible)[0]
    vis_idx = sorted(vis_idx,
                     key=lambda j: len(cand_data[j][0].frames), reverse=True)

    n_added = 0
    for i in vis_idx:
        pt_id = cand_ids[i]
        proj = uv[i]
        depth = float(pts_cam[i, 2])
        if median_depth is not None and depth > 0:
            ratio = depth / median_depth if depth > median_depth else median_depth / depth
            if ratio > float(max_depth_ratio):
                continue

        pt_obj, ref_frame, ref_kp_idx = cand_data[i]
        ref_desc = (
            pt_obj.descriptor
            if getattr(pt_obj, "descriptor", None) is not None
            else ref_frame.Descriptors[ref_kp_idx]
        )

        dists_px = np.linalg.norm(kp_xy - proj, axis=1)
        within = np.where(dists_px <= search_radius)[0]
        if len(within) == 0:
            continue

        best_d = float('inf')
        best_ki = -1
        for ki in within:
            if ki in assigned_kp:
                continue
            d = cv2.norm(ref_desc, curr_frame.Descriptors[ki], norm_type)
            if d < best_d:
                best_d = d
                best_ki = int(ki)
        if best_ki >= 0 and best_d <= max_desc_dist:
            # First link wins: False means the keypoint or the point was
            # already taken in this frame; skip the match.
            if not world_map.link(curr_frame, best_ki, pt_obj):
                continue
            assigned_kp.add(best_ki)
            already_associated.add(pt_id)
            n_added += 1

    return n_added


def associate_known_points(prev_frame, curr_frame, inlier_matches, world_map):
    """
    Transfer existing map-point associations from prev_frame to curr_frame.

    Returns
    -------
    list
        Matches that do NOT correspond to an existing map point and are
        candidates for triangulating new 3D points.
    """
    unmatched_matches = []
    for m in inlier_matches:
        existing_id = prev_frame.keypoint_to_point.get(m.queryIdx)
        if existing_id is None:
            unmatched_matches.append(m)
            continue

        existing_point = world_map.get_point(existing_id)
        if existing_point is None:
            unmatched_matches.append(m)
            continue

        # First link wins. A refused link (keypoint or point already taken in
        # curr_frame) is skipped and is NOT a triangulation candidate either.
        world_map.link(curr_frame, m.trainIdx, existing_point)

    return unmatched_matches


def _descriptor_distance(desc_a, desc_b, method="ORB"):
    if desc_a is None or desc_b is None:
        return float("inf")
    norm_type = cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2
    return float(cv2.norm(np.asarray(desc_a), np.asarray(desc_b), norm_type))


def fuse_duplicate_points(world_map, frames,
                          max_dist=0.05, max_desc_dist=40.0, method="ORB"):
    """
    Merge near-duplicate landmarks (Stage 3.2).

    Cluster by 3D distance + descriptor distance. Keep the older id, union
    observations, re-wire ``keypoint_to_point``, delete the duplicate.

    Conservative thresholds by default to limit false merges.
    Uses a coarse spatial hash so cost stays near-linear in map size.

    Returns
    -------
    int
        Number of points merged (removed as duplicates).
    """
    if world_map is None or not getattr(world_map, "points", None):
        return 0

    points = list(world_map.points)
    n = len(points)
    if n < 2:
        return 0

    cell = float(max_dist)
    if cell <= 0:
        cell = 0.05

    # Spatial hash: cell -> indices (sorted by id so older ids come first).
    grid = {}
    for i, p in enumerate(points):
        xyz = np.asarray(p.point, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            continue
        key = (int(np.floor(xyz[0] / cell)),
               int(np.floor(xyz[1] / cell)),
               int(np.floor(xyz[2] / cell)))
        grid.setdefault(key, []).append(i)

    for key in grid:
        grid[key].sort(key=lambda i: int(getattr(points[i], "id", i)))

    alive = [True] * n
    merged = 0
    neighbour_offsets = [(dx, dy, dz)
                         for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1)
                         for dz in (-1, 0, 1)]

    for key, idxs in list(grid.items()):
        for ii, i in enumerate(idxs):
            if not alive[i]:
                continue
            pi = points[i]
            xi = np.asarray(pi.point, dtype=np.float64)
            # Compare against later points in same cell and all neighbour cells.
            candidates = []
            for dx, dy, dz in neighbour_offsets:
                nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                for j in grid.get(nkey, []):
                    if j == i or not alive[j]:
                        continue
                    # Only merge into older (lower id).
                    if int(getattr(points[j], "id", j)) < int(getattr(pi, "id", i)):
                        continue
                    candidates.append(j)
            for j in candidates:
                if not alive[j]:
                    continue
                pj = points[j]
                xj = np.asarray(pj.point, dtype=np.float64)
                if float(np.linalg.norm(xi - xj)) > float(max_dist):
                    continue
                d = _descriptor_distance(
                    getattr(pi, "descriptor", None),
                    getattr(pj, "descriptor", None),
                    method=method,
                )
                if d > float(max_desc_dist):
                    continue
                _merge_point_into(pi, pj, frames, world_map)
                alive[j] = False
                merged += 1

    return merged


def _merge_point_into(keeper, duplicate, frames, world_map):
    """
    Union observations from *duplicate* into *keeper*; rewire; delete dup.

    Stage 01b: every table write goes through Map.link / Map.unlink /
    Map.remove_point.  For each observation (fid, kp) of the duplicate: if the
    keeper already has a keypoint in fid, the keeper's keypoint is kept and the
    duplicate's keypoint is unlinked (it becomes free); otherwise (fid, kp) is
    re-linked to the keeper.  The duplicate is then removed with all its links
    already gone.
    """
    dup_id = int(duplicate.id)

    # Observation count of the duplicate before its links are taken away
    # (weight for the XYZ average below; unchanged from the pre-01b code).
    n_d = max(1, len(duplicate.frames))

    for fid, kp_idx in list(getattr(duplicate, "observations", {}).items()):
        f = world_map.resolve_frame(fid, frames)
        if f is None:
            # Unknown frame: nothing to rewire; the observation dies with dup.
            continue
        world_map.unlink(f, kp_idx)
        if fid not in keeper.observations:
            world_map.link(f, kp_idx, keeper)

    # Average XYZ (observation-weighted lightly toward keeper).
    n_k = max(1, len(keeper.frames))
    w_k = float(n_k)
    w_d = float(n_d)
    new_xyz = (w_k * np.asarray(keeper.point, dtype=np.float64)
               + w_d * np.asarray(duplicate.point, dtype=np.float64)) / (w_k + w_d)
    keeper.point = new_xyz
    keeper.x = float(new_xyz[0])
    keeper.y = float(new_xyz[1])
    keeper.z = float(new_xyz[2])

    if getattr(keeper, "descriptor", None) is None and getattr(duplicate, "descriptor", None) is not None:
        keeper.descriptor = np.array(duplicate.descriptor, copy=True)

    try:
        keeper.last_seen_fid = max(
            int(getattr(keeper, "last_seen_fid", -1)),
            int(getattr(duplicate, "last_seen_fid", -1)),
        )
    except Exception:
        pass

    # Links were rewired above; this drops the duplicate from the map (and
    # clears any observation whose frame could not be resolved).
    world_map.remove_point(dup_id, frames=frames)


def count_duplicate_neighbours(world_map, radius=0.05):
    """
    Count unordered pairs of distinct points within *radius* (DIAGNOSIS §7.C).

    Returns
    -------
    tuple[int, int]
        (pair_count, n_points)
    """
    if world_map is None or not getattr(world_map, "points", None):
        return 0, 0
    pts = [np.asarray(p.point, dtype=np.float64) for p in world_map.points
           if np.all(np.isfinite(getattr(p, "point", [np.nan])))]
    n = len(pts)
    if n < 2:
        return 0, n
    cell = float(radius)
    if cell <= 0:
        cell = 0.05
    grid = {}
    for i, xyz in enumerate(pts):
        key = (int(np.floor(xyz[0] / cell)),
               int(np.floor(xyz[1] / cell)),
               int(np.floor(xyz[2] / cell)))
        grid.setdefault(key, []).append(i)
    r2 = float(radius) ** 2
    pairs = 0
    seen = set()
    neighbour_offsets = [(dx, dy, dz)
                         for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1)
                         for dz in (-1, 0, 1)]
    for key, idxs in grid.items():
        for i in idxs:
            xi = pts[i]
            for dx, dy, dz in neighbour_offsets:
                for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if float(np.sum((pts[j] - xi) ** 2)) <= r2:
                        pairs += 1
    return pairs, n


def compute_map_observation_stats(world_map):
    """
    Return dict with median/mean obs per point and fraction with obs >= 3.
    """
    if world_map is None or not getattr(world_map, "points", None) or len(world_map.points) == 0:
        return {
            "n_points": 0,
            "median_obs": 0.0,
            "mean_obs": 0.0,
            "frac_obs_ge3": 0.0,
            "obs_counts": [],
        }
    counts = [len(getattr(p, "frames", []) or []) for p in world_map.points]
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "n_points": int(len(counts)),
        "median_obs": float(np.median(arr)),
        "mean_obs": float(np.mean(arr)),
        "frac_obs_ge3": float(np.mean(arr >= 3)),
        "obs_counts": counts,
    }


def sample_keyframe_link_counts(keyframes, world_map, min_age_frames=200, current_fid=None):
    """
    For keyframes at least *min_age_frames* older than *current_fid*,
    report live ``keypoint_to_point`` link counts (Stage 3 E3.3 / T3.3.1).

    Returns
    -------
    dict
    """
    if not keyframes:
        return {
            "n_old_keyframes": 0,
            "median_links": 0.0,
            "mean_links": 0.0,
            "min_links": 0,
            "frac_links_ge15": 0.0,
            "frac_links_ge10": 0.0,
            "samples": [],
        }
    if current_fid is None:
        try:
            current_fid = max(int(getattr(kf, "id", 0)) for _, kf in keyframes)
        except Exception:
            current_fid = 0

    samples = []
    for kf_idx, kf in keyframes:
        try:
            fid = int(getattr(kf, "id", kf_idx))
        except Exception:
            fid = int(kf_idx)
        age = int(current_fid) - fid
        if age < int(min_age_frames):
            continue
        links = 0
        for kp, pid in (getattr(kf, "keypoint_to_point", None) or {}).items():
            if world_map is not None and world_map.get_point(pid) is not None:
                links += 1
            elif world_map is None:
                links += 1
        samples.append((fid, links, age))

    if not samples:
        return {
            "n_old_keyframes": 0,
            "median_links": 0.0,
            "mean_links": 0.0,
            "min_links": 0,
            "frac_links_ge15": 0.0,
            "frac_links_ge10": 0.0,
            "samples": [],
        }
    link_arr = np.asarray([s[1] for s in samples], dtype=np.float64)
    return {
        "n_old_keyframes": int(len(samples)),
        "median_links": float(np.median(link_arr)),
        "mean_links": float(np.mean(link_arr)),
        "min_links": int(np.min(link_arr)),
        "frac_links_ge15": float(np.mean(link_arr >= 15)),
        "frac_links_ge10": float(np.mean(link_arr >= 10)),
        "samples": samples,
    }


def triangulate_and_add_points(prev_frame, curr_frame, inlier_matches, world_map,
                              min_parallax_deg=1.5, skip_if_curr_already_linked=False,
                              median_scene_depth=None, max_depth_ratio=10.0,
                              recent_depths=None):
    """
    Triangulate new 3D points from unmatched inlier matches and add to the map.

    Not on the product path since stage 04-1a; kept as the 01b link-discipline
    reference exercised by tests/test_map_link_invariants.py.

    Parameters
    ----------
    min_parallax_deg : float
        Minimum angle (degrees) between the two viewing rays to accept a
        triangulation. Below this, depth uncertainty is very high; default 1.5.
        Use 0 to disable the parallax check (accept all triangulations).
    skip_if_curr_already_linked : bool
        Accepted and ignored (kept for signature compatibility). Since Stage
        01b the behaviour it used to enable is unconditional: a keypoint that
        is already linked in either frame never gets a new landmark; if only
        the current keypoint is linked, prev's observation is added to that
        landmark through Map.link instead of creating a duplicate.
    median_scene_depth : float | None
        If set with max_depth_ratio > 0, reject triangulations whose current-
        camera depth is farther than max_depth_ratio × median (Stage 3.4).
    max_depth_ratio : float
        Depth-sanity factor vs median_scene_depth (and vs baseline×40).
    recent_depths : list | None
        If provided, accepted depths are appended for rolling median updates.
    """
    if len(inlier_matches) < 8:
        return

    K = curr_frame.K
    K_inv = np.linalg.inv(K)
    proj_prev = camera_pose_to_projection(prev_frame.cam_T_world, K)
    proj_curr = camera_pose_to_projection(curr_frame.cam_T_world, K)

    cam_center_prev = prev_frame.world_T_cam[:3, 3]
    cam_center_curr = curr_frame.world_T_cam[:3, 3]
    baseline = np.linalg.norm(cam_center_curr - cam_center_prev)
    if baseline < 1e-6:
        return

    pts_prev_list = []
    pts_curr_list = []
    for m in inlier_matches:
        pts_prev_list.append(prev_frame.KeyPoints[m.queryIdx].pt)
        pts_curr_list.append(curr_frame.KeyPoints[m.trainIdx].pt)
    new_matches = list(inlier_matches)

    if len(new_matches) < 8:
        return

    pts_prev = np.array(pts_prev_list, dtype=np.float64)
    pts_curr = np.array(pts_curr_list, dtype=np.float64)

    points4d = cv2.triangulatePoints(proj_prev, proj_curr, pts_prev.T, pts_curr.T)
    points3d = (points4d[:3] / (points4d[3] + 1e-12)).T

    R_prev_w = prev_frame.world_T_cam[:3, :3]
    R_curr_w = curr_frame.world_T_cam[:3, :3]
    min_parallax_rad = np.radians(min_parallax_deg) if min_parallax_deg > 0 else None

    for i, m in enumerate(new_matches):
        if min_parallax_rad is not None:
            uv_p = pts_prev[i]
            uv_c = pts_curr[i]
            ray_prev = R_prev_w @ (K_inv @ np.array([uv_p[0], uv_p[1], 1.0]))
            ray_curr = R_curr_w @ (K_inv @ np.array([uv_c[0], uv_c[1], 1.0]))
            n_prev = np.linalg.norm(ray_prev)
            n_curr = np.linalg.norm(ray_curr)
            if n_prev < 1e-9 or n_curr < 1e-9:
                continue
            ray_prev = ray_prev / n_prev
            ray_curr = ray_curr / n_curr
            cos_angle = float(np.clip(np.dot(ray_prev, ray_curr), -1.0, 1.0))
            angle = np.arccos(cos_angle)
            if angle < min_parallax_rad:
                continue

        xw = points3d[i]
        if not np.all(np.isfinite(xw)):
            continue

        # Cheirality: in front of both cameras (positive depth).
        x_prev = prev_frame.cam_T_world[:3, :3] @ xw + prev_frame.cam_T_world[:3, 3]
        x_curr = curr_frame.cam_T_world[:3, :3] @ xw + curr_frame.cam_T_world[:3, 3]
        if x_prev[2] <= 1e-6 or x_curr[2] <= 1e-6:
            continue

        # Depth sanity (Stage 3.4): reject absurd depths vs baseline / scene median.
        depth_curr = float(x_curr[2])
        depth_prev = float(x_prev[2])
        if not (np.isfinite(depth_curr) and np.isfinite(depth_prev)):
            continue
        # Relative to baseline (ORB-SLAM-style soft cap); with unit-scale VO
        # baseline≈1 so this bounds runaway depths without needing GT scale.
        if max_depth_ratio is not None and max_depth_ratio > 0 and baseline > 1e-9:
            if depth_curr > float(max_depth_ratio) * 4.0 * baseline:
                continue
            if depth_prev > float(max_depth_ratio) * 4.0 * baseline:
                continue
        if (
            median_scene_depth is not None
            and max_depth_ratio is not None
            and max_depth_ratio > 0
            and median_scene_depth > 1e-6
        ):
            if depth_curr > float(max_depth_ratio) * float(median_scene_depth):
                continue
            if depth_curr < float(median_scene_depth) / float(max_depth_ratio):
                continue

        # Symmetric reprojection (secondary gate; not sufficient alone — §2.6).
        uv_prev = project_point(proj_prev, xw)
        uv_curr = project_point(proj_curr, xw)
        err_prev = np.linalg.norm(uv_prev - pts_prev[i])
        err_curr = np.linalg.norm(uv_curr - pts_curr[i])
        if (err_prev + err_curr) > 6.0:
            continue

        # Stage 01b: never create a landmark on a keypoint that is already
        # linked, in either frame (unconditional; was the
        # skip_if_curr_already_linked=True branch).
        if prev_frame.keypoint_to_point.get(m.queryIdx) is not None:
            continue
        existing_id = curr_frame.keypoint_to_point.get(m.trainIdx)
        if existing_id is not None:
            existing_point = world_map.get_point(existing_id)
            if existing_point is not None:
                # Add prev's observation to the existing landmark instead of a
                # duplicate; link() refuses (no-op) if prev's keypoint is taken.
                world_map.link(prev_frame, m.queryIdx, existing_point)
            continue

        desc = np.array(prev_frame.Descriptors[m.queryIdx], copy=True)
        p = Point(point=xw, descriptor=desc)
        world_map.add_point(p, frames=[prev_frame, curr_frame])
        if not (world_map.link(prev_frame, m.queryIdx, p)
                and world_map.link(curr_frame, m.trainIdx, p)):
            # No half-linked landmark: remove_point unlinks whatever was set.
            world_map.remove_point(p.id, frames=[prev_frame, curr_frame])
            continue
        try:
            p.created_fid = int(curr_frame.id)
            p.last_seen_fid = int(curr_frame.id)
        except Exception:
            pass
        if recent_depths is not None:
            recent_depths.append(depth_curr)
