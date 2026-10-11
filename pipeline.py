import time
import sys
import threading
from collections import defaultdict
from pathlib import Path

from scipy.spatial.transform import Rotation
import numpy as np

from Map import Map
from depth_filter_arm import DepthFilterArm
import local_mapping
from local_mapping import LocalMappingParams, process_new_keyframe
from loop_closure import LoopCloser, correct_trajectory, correct_trajectory_pose_graph
from map_export import write_map_points
from mapping import (
    compute_map_observation_stats,
    count_duplicate_neighbours,
    fuse_duplicate_points,
    sample_keyframe_link_counts,
)
import optimization
from optimization import bundle_adjust
# Stage 03a instrumentation helpers (spec B/C); re-exported here at module level so the tests
# can pin them as pipeline.pose_delta_stats / pipeline.format_* without running the pipeline.
from pose_log import (  # noqa: F401
    format_correction_line,
    format_pose_online_line,
    pose_delta_stats,
    print_correction,
)
from reprojection import compute_reprojection_error, compute_per_point_reprojection_error
from visualizer import (
    build_overlay_image,
    finish_visualizer,
    init_visualizer,
    is_quit_requested,
    update_visualizer_from_state,
)

# Terminal quit: when user types 'q' or 'quit' in the terminal, set this to True
# so the main loop stops; windows stay open for inspection.
_stop_requested = False

# Stage 04v (spec C1): the map tracker's frame label -> ([POSE-ONLINE] source, state). MapTracker
# returns only tracked and lost; bridged and rotation come from map_rotation.MapRotationTracker.
M1_ONLINE_LABELS = {"tracked": ("map", "OK"), "lost": ("predicted", "LOST"),
                    "bridged": ("bridged", "BRIDGED"), "rotation": ("rotation", "ROT")}

# Stage 04-m0 (spec F5): the opening driver's frame label under --live_map on -> ([POSE-ONLINE] source,
# state); the 04u labels plus the r52 fallback's rotation state.
M0_ONLINE_LABELS = {"measured": ("rotation", "ROT"), "unsupported": ("predicted", "LOST"), "lost": ("predicted", "LOST"),
                    "init": ("init", "OK"), "tracked": ("map", "OK"), "rotation": ("rotation", "ROT")}


def m0_writes_map(state):
    """Stage 04-m0 (spec F3): the mapper runs on a frame the map posed (tracked) or on the hand-off
    frame (init), and on no other frame (R52-1: no fallback state writes to the map; registered risk 2:
    no depth update from a frame whose translation is held)."""
    return state in ("init", "tracked")


def format_m0_frame_line(k, state, mapper, kf, points, init_points, inl_init, inl_new):
    """Stage 04-m0 (spec F6): the one [M0-FRAME] line of frame k >= 1."""
    return (f"  >> [M0-FRAME] frame={int(k)} state={state} mapper={1 if mapper else 0} kf={1 if kf else 0} "
            f"points={int(points)} init_points={int(init_points)} new_points={int(points) - int(init_points)} "
            f"inl_init={int(inl_init)} inl_new={int(inl_new)}")


def m0_clear_map(world_map, frames):
    """Stage 04-m0 (spec F2): a false start removes the whole map; every point left in world_map goes
    through Map.remove_point (both link tables). Returns the number removed."""
    removed = 0
    for pt in list(world_map.points):
        world_map.remove_point(pt.id, frames)
        removed += 1
    return removed



class ViewerState:
    """Snapshot of state for the viewer thread (trajectory, points, overlay image)."""
    __slots__ = ('trajectory', 'map_points_display', 'keyframe_indices', 'tracking_state', 'overlay_image')

    def __init__(self, trajectory, map_points_display, keyframe_indices, tracking_state, overlay_image):
        self.trajectory = trajectory
        self.map_points_display = map_points_display
        self.keyframe_indices = keyframe_indices
        self.tracking_state = tracking_state
        self.overlay_image = overlay_image


def _terminal_quit_listener():
    """Background thread: read stdin; on 'q' or 'quit' set _stop_requested."""
    global _stop_requested
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip().lower()
            if line in ('q', 'quit'):
                _stop_requested = True
                print("\n  >> Terminal quit requested; stopping pipeline (windows stay open).")
                break
    except (EOFError, OSError):
        pass


def _camera_center_from_cam_pose(cam_t_world):
    r = cam_t_world[:3, :3]
    t = cam_t_world[:3, 3]
    return -r.T @ t


def _pose_to_tum_line(world_T_cam):
    """Convert 4x4 world_T_cam to TUM line: timestamp tx ty tz qx qy qz qw (timestamp not included here)."""
    tx, ty, tz = world_T_cam[:3, 3]
    R = world_T_cam[:3, :3]
    # Guard: sometimes upstream numeric issues can produce a slightly-invalid rotation.
    # Project to nearest valid rotation so trajectory saving never crashes.
    R = np.asarray(R, dtype=np.float64)
    if not np.all(np.isfinite(R)):
        raise ValueError("Pose rotation contains NaN/inf")
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    r = Rotation.from_matrix(Rn)
    qx, qy, qz, qw = r.as_quat()  # scipy uses (qx, qy, qz, qw)
    return f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}"


def _write_estimated_trajectory_tum(frames, timestamps, out_path, pose_of=None):
    """
    Write estimated trajectory in TUM format (timestamp tx ty tz qx qy qz qw).
    timestamps: list of float (one per frame), or None to use frame index 0,1,2,...
    pose_of: callable frame -> 4x4 world_T_cam to write; defaults to the frame's current pose
    (stage 03a: the online stream passes the per-frame snapshot instead).
    Also writes a companion * _frame_index.txt with lines "timestamp frame_index" so you can
    map evo_ape plot time (x-axis) back to image sequence frame index.
    """
    if pose_of is None:
        pose_of = lambda f: f.pose
    out_path = Path(out_path)
    with open(out_path, "w") as f:
        f.write("# estimated trajectory (TUM format: timestamp tx ty tz qx qy qz qw)\n")
        for i, frame in enumerate(frames):
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else float(i)
            try:
                line = _pose_to_tum_line(pose_of(frame))
                f.write(f"{ts} {line}\n")
            except Exception as e:
                # Skip corrupt poses instead of crashing the whole run.
                f.write(f"# skipped frame {i} (id={getattr(frame, 'id', '?')}): {e}\n")

    # Companion file: timestamp -> frame index (for evo_ape: look up time on x-axis to get frame)
    stem = out_path.stem
    parent = out_path.parent
    frame_index_path = parent / f"{stem}_frame_index.txt"
    with open(frame_index_path, "w") as f:
        f.write("# timestamp frame_index  (use to map evo_ape plot time -> image sequence frame)\n")
        for i, frame in enumerate(frames):
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else float(i)
            f.write(f"{ts} {i}\n")
    return frame_index_path


def _load_gt_timestamps(gt_path):
    """Load timestamps (first column) from a TUM-format trajectory file. Returns (array, t_min)."""
    timestamps = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    timestamps.append(float(parts[0]))
                except ValueError:
                    continue
    if not timestamps:
        return np.array([]), 0.0
    arr = np.array(timestamps)
    return arr, float(np.min(arr))


def _write_evo_frame_index_file(frames_estimated, timestamps, ground_truth_path, save_estimated_path):
    """
    Write a file that maps evo_ape plot x-axis (relative time in sec) -> frame index.
    evo associates by nearest timestamp: for each GT time we find our pose with closest timestamp.
    So the plot's x-axis (relative GT time) can be looked up to get the image frame index.
    Returns the path of the written file, or None if not written.
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.is_file():
        return None
    gt_timestamps, t_gt_min = _load_gt_timestamps(gt_path)
    if len(gt_timestamps) == 0:
        return None
    # Our estimate timestamps (same as in the TUM file we wrote)
    est_ts = np.array([
        timestamps[i] if timestamps is not None and i < len(timestamps) else float(i)
        for i in range(len(frames_estimated))
    ])
    out_path = Path(save_estimated_path)
    stem = out_path.stem
    parent = out_path.parent
    evo_frame_path = parent / f"{stem}_evo_frame_index.txt"
    with open(evo_frame_path, "w") as f:
        f.write("# evo_ape plot x-axis (relative time in sec) -> frame_index (use for spike lookup)\n")
        for t_gt in gt_timestamps:
            rel_t = t_gt - t_gt_min
            i = int(np.argmin(np.abs(est_ts - t_gt)))
            i = min(i, len(frames_estimated) - 1)
            f.write(f"{rel_t:.6f} {i}\n")
    return evo_frame_path


def _print_timing_summary(stats, run_time_bounds):
    """Print timing summary (overall run, avg pose update, avg PnP, avg PnP by solver in multi, avg BA)."""
    start, end = run_time_bounds[0], run_time_bounds[1]
    if start is not None and end is not None:
        total = end - start
        print()
        print("--- Timing summary ---")
        print(f"  Overall run: {total:.2f}s")
    pose_times = stats.get("pose_update_times", [])
    if pose_times:
        avg_pose = sum(pose_times) / len(pose_times)
        print(f"  Pose update: count={len(pose_times)}, avg={avg_pose:.3f}s")
    pnp_times = stats.get("pnp_times", [])
    if pnp_times:
        avg_pnp = sum(t for t, _ in pnp_times) / len(pnp_times)
        print(f"  PnP: count={len(pnp_times)}, avg={avg_pnp:.3f}s")
        multi_times = stats.get("pnp_multi_solver_times", [])
        if multi_times:
            by_solver = defaultdict(list)
            for name, sec in multi_times:
                by_solver[name].append(sec)
            print("  PnP (multi mode) avg per solver:")
            for name in ("iterative", "ap3p", "epnp", "sqpnp"):
                if name in by_solver:
                    vals = by_solver[name]
                    print(f"    {name}: count={len(vals)}, avg={sum(vals)/len(vals):.3f}s")
    ba_times = stats.get("ba_times", [])
    if ba_times:
        avg_ba = sum(t for t, _ in ba_times) / len(ba_times)
        print(f"  Bundle adjustment: count={len(ba_times)}, avg={avg_ba:.3f}s")
        by_method = {}
        for t, method in ba_times:
            by_method.setdefault(method, []).append(t)
        if len(by_method) > 1:
            print("  BA avg per method:")
            for method in sorted(by_method.keys()):
                vals = by_method[method]
                print(f"    {method}: count={len(vals)}, avg={sum(vals)/len(vals):.3f}s")
    lc_times = stats.get("loop_closure_times", [])
    if lc_times:
        print("  Loop closure correction (frame, keyframe, time):")
        for frame_k, lc_kf_idx, duration in lc_times:
            print(f"    frame {frame_k} <-> frame {lc_kf_idx}: {duration:.2f}s")
        total_lc = sum(d for _, _, d in lc_times)
        print(f"  Loop closure: count={len(lc_times)}, total={total_lc:.2f}s, avg={total_lc/len(lc_times):.2f}s")
    print()

BA_COUNTER_KEYS = ("ba_attempted", "ba_skipped", "ba_completed", "ba_accepted",
                   "ba_rejected", "ba_errors")


def record_ba_outcome(stats, status, accepted=None, reason=None):
    """
    Update the BA accounting counters for one attempt (stage 01a, spec A6).

    status: "skipped" (reason required for the per-reason table), "completed" (accepted
    required: True -> ba_accepted, False -> ba_rejected), or "error".
    Identities kept: attempted = skipped + completed + errors; completed = accepted + rejected;
    ba_success = ba_accepted; ba_fail = ba_rejected + ba_errors (aliases the summary line uses).
    """
    for key in BA_COUNTER_KEYS:
        stats.setdefault(key, 0)
    stats.setdefault("ba_skip_reasons", {})
    if status not in ("skipped", "completed", "error"):
        raise ValueError(f"unknown BA outcome status {status!r}")
    if status == "completed" and accepted is None:
        raise ValueError("accepted must be given for a completed BA attempt")
    stats["ba_attempted"] += 1
    if status == "skipped":
        stats["ba_skipped"] += 1
        key = str(reason) if reason is not None else "unspecified"
        stats["ba_skip_reasons"][key] = stats["ba_skip_reasons"].get(key, 0) + 1
    elif status == "completed":
        stats["ba_completed"] += 1
        if accepted:
            stats["ba_accepted"] += 1
        else:
            stats["ba_rejected"] += 1
    else:
        stats["ba_errors"] += 1
    stats["ba_success"] = stats["ba_accepted"]
    stats["ba_fail"] = stats["ba_rejected"] + stats["ba_errors"]


def ba_summary_lines(stats):
    """The two BA summary lines (exact formats parsed by the verifier)."""
    success = stats.get("ba_success", 0)
    fail = stats.get("ba_fail", 0)
    total = success + fail
    ratio = (success / total) if total > 0 else 0.0
    line1 = f"  Bundle adjustment: success={success}, fail={fail}, success/overall={ratio:.2%}"
    line2 = (
        "  Bundle adjustment accounting: attempted=%d, skipped=%d, completed=%d, accepted=%d, "
        "rejected=%d, errors=%d, skip_reasons=%s"
        % (stats.get("ba_attempted", 0), stats.get("ba_skipped", 0), stats.get("ba_completed", 0),
           stats.get("ba_accepted", 0), stats.get("ba_rejected", 0), stats.get("ba_errors", 0),
           repr(dict(stats.get("ba_skip_reasons", {}))))
    )
    return line1, line2


def _restore_ba_snapshot(frames_window, pose_snapshot, point_snapshot, world_map):
    """Roll back window poses (through the setter) and all map point coordinates."""
    for f in frames_window:
        old_cam_T_world = pose_snapshot.get(f.id)
        if old_cam_T_world is not None:
            f.set_camera_pose(old_cam_T_world)
    for p in world_map.points:
        old_xyz = point_snapshot.get(p.id)
        if old_xyz is not None:
            p.point = old_xyz.copy()
            p.x = float(old_xyz[0])
            p.y = float(old_xyz[1])
            p.z = float(old_xyz[2])


def apply_ba_outlier_removal(world_map, frames, outlier_obs):
    """
    Stage 04y (spec A4): unlink the final outliers of an accepted outlier-rejecting BA solve.

    outlier_obs: (frame_id, keypoint_index, point_id) triples.  Each frame is resolved with
    world_map.resolve_frame(frame_id, frames); the link is removed through Map.unlink (both
    sides) only when the keypoint still points at that point.  No point is deleted.  Returns
    the number of links removed.
    """
    removed = 0
    for fid, kp, pid in outlier_obs:
        frame = world_map.resolve_frame(fid, frames)
        if frame is None:
            continue
        if frame.keypoint_to_point.get(int(kp)) == pid:
            if world_map.unlink(frame, int(kp)):
                removed += 1
    return removed


def format_ba_outliers_line(k, n_obs, n_mid, n_end, accepted, removed, inv_total):
    """The one [BA-OUTLIERS] line per completed solve under --ba_outlier_reject on (spec A4)."""
    return (f"  >> [BA-OUTLIERS] frame={int(k)} obs={int(n_obs)} mid={int(n_mid)} end={int(n_end)} "
            f"accepted={1 if accepted else 0} removed={int(removed)} inv_total={int(inv_total)}")


def format_ba_support_line(k, free, held, min_free_rows, p2_ran, held_p2, p2_min_rows, p2_under10):
    """Stage 04z (spec C3): the one [BA-SUPPORT] line per completed solve under --ba_jump_gate off
    or --ba_free_cameras supported (the R57-2 participation counts and the R57-3 phase-2 instrument)."""
    return (f"  >> [BA-SUPPORT] frame={int(k)} free={int(free)} held={int(held)} "
            f"min_free_rows={int(min_free_rows)} p2_ran={1 if p2_ran else 0} held_p2={int(held_p2)} "
            f"p2_min_rows={int(p2_min_rows)} p2_under10={int(p2_under10)}")


def ba_outlier_epilogue(world_map, frames, k, n_obs, info, accepted):
    """
    Stage 04z (spec C4): the 04y outlier epilogue of one completed solve, never raising.

    frames = frames[:k + 1].  Unlinks the final outliers when the solve was accepted (removed = 0
    otherwise), sums world_map.check_invariants(frames) and returns the [BA-OUTLIERS] line.  If
    anything inside raises, returns a [BA-OUTLIER-ERROR] line instead: the solve's accept or
    roll-back decision and its accounting were recorded before and are never revisited; links
    already removed stay removed (Map.unlink keeps the tables mutual inverses link by link).
    """
    try:
        removed = 0
        if accepted:
            removed = apply_ba_outlier_removal(world_map, frames, info.get("outlier_obs", []))
        inv_total = sum(world_map.check_invariants(frames).values())
        return format_ba_outliers_line(k, n_obs, info.get("n_outliers_mid", 0),
                                       info.get("n_outliers_end", 0), accepted, removed, inv_total)
    except Exception as e:
        return f"  >> [BA-OUTLIER-ERROR] frame={int(k)}: {e}"


'''helper function to calculate the maximum step length and the 95th percentile step length in a window of camera poses.
 note : a step is the distance between the center of the current camera pose and the center of the previous camera pose.'''
def _max_step_in_window(cam_poses):
    if len(cam_poses) < 2:
        return 0.0, 0.0
    centers = [_camera_center_from_cam_pose(p) for p in cam_poses]
    steps = [
        float(np.linalg.norm(centers[i] - centers[i - 1]))
        for i in range(1, len(centers))
    ]
    if len(steps) == 0:
        return 0.0, 0.0
    steps_sorted = sorted(steps)
    p95_idx = min(len(steps_sorted) - 1, int(0.95 * (len(steps_sorted) - 1)))
    return float(max(steps)), float(steps_sorted[p95_idx])


def show_trajectory(frames, n_frames=None, camera_spacing=None, show_grid=True,
                    show_traj=True, pnp_interval=10, ba_interval=20,
                    ba_window_size=12, ba_max_points=250,
                    ba_max_allowed_step_factor=3.0, ba_bad_reproj_ratio=1.2,
                    use_bundle_adjustment=True, ba_method='lm',
                    ba_use_analytical_jacobian=True,
                    use_homography=True,
                    show_realtime=True, save_estimated_path=None, timestamps=None,
                    t_norm_log_path=None,
                    local_map_window=10,
                    log_file_path=None,
                    log_invocation=None,
                    pnp_min_inlier_ratio=0.55,
                    pnp_solver="iterative",
                    pnp_refinement=True,
                    pnp_use_initial_guess=True,
                    pnp_multi_inlier_gate=0.8,
                    pnp_reprojection_error=3.0,
                    pnp_adaptive_gate=False,
                    pnp_adaptive_gate_factor=0.2,
                    use_relative_scale=True,
                    ground_truth_path=None,
                    viewer_fps=30,
                    tracker_delay_ms=0,
                    use_loop_closure=True,
                    loop_closure_strategy='pose_graph',
                    loop_detection_strategy='spaced',
                    vocabulary_path=None,
                    num_spaced_candidates=5,
                    min_gap=100,
                    keyframe_strategy='fixed',
                    keyframe_interval=20,
                    keyframe_min_spacing=5,
                    keyframe_max_spacing=50,
                    keyframe_ratio_threshold=0.80,
                    use_relocalization=True,
                    relocalization_match_threshold=25,
                    relocalization_min_2d3d_pairs=10,
                    relocalization_min_pnp_inliers=10,
                    relocalization_ratio_threshold=0.85,
                    relocalization_sanity_check_enabled=True,
                    relocalization_max_distance_from_keyframe=5.0,
                    ba_min_point_observations=3,
                    ba_point_sanity_check=True,
                    # Stage 01a: back-end disposition switches (defaults declared in the contract)
                    ba_max_fixed_observers=0,
                    ba_huber_delta_px=2.4477,
                    ba_min_anchor_points=10,
                    ba_max_iterations=10,
                    # Stage 04y: local-BA outlier rejection (ORB-SLAM section VI.D), "off" or "on"
                    ba_outlier_reject="off",
                    # Stage 04z (R57-2): participation rule ("all" / "supported") and the jump gate ("on" / "off")
                    ba_free_cameras="all",
                    ba_jump_gate="on",
                    prune_interval=30,
                    prune_high_reproj=False,
                    prune_numeric_corruption=False,
                    prune_outlier_hits=False,
                    prune_outlier_hit_count=3,
                    prune_outlier_last_seen_age=20,
                    ba_mode='sliding_window',
                    keyframe_local_topk=5,
                    keyframe_local_min_shared_points=20,
                    # Stage 3 — map continuity
                    local_map_search_radius=25.0,
                    local_map_max_desc_dist=None,
                    local_map_max_depth_ratio=5.0,
                    use_point_fusion=True,
                    fusion_interval=30,
                    fusion_max_dist=0.05,
                    fusion_max_desc_dist=40.0,
                    map_metrics_path=None,
                    step_lengths_path=None,
                    oracle=None,
                    # Stage 04-1a: the point-source arm and the local mapper's constants
                    point_source='keyframe_pair',
                    lm_params=None,
                    # Stage 04g: the depth-filter arm's constants (used only with point_source 'depth_filter')
                    df_params=None,
                    df_arm_params=None,
                    # Stage 04k: sub-pixel refinement along the epipolar line (--epi_refine)
                    refine_params=None,
                    # Stage 04l: the map-based tracker against a perfect map (--tracking map)
                    m1_tracker=None,
                    # Stage 04r: the pre-map rotation-only mode (--tracking rotation)
                    rot_tracker=None,
                    # Stage 04u: the opening driver (--tracking rotation --init twoview)
                    open_tracker=None,
                    # Stage 04-m0: the integrated system M0 ("off" or "on"; --live_map)
                    live_map="off"):
    """
    Main VO pipeline: iterate over frames, estimate poses, and optionally
    visualize and/or save estimated trajectory to file (TUM format).

    Parameters
    ----------
    camera_spacing : int | None
        None = draw frustums only for keyframes (default).
        int N = draw a frustum every N-th frame (legacy every-N mode).
    show_realtime : bool
        If True, show integrated Pangolin visualizer. If False,
        run the pose loop only (no display); useful with save_estimated_path.
    save_estimated_path : str | Path | None
        If set, write estimated trajectory to this file in TUM format.
    timestamps : list of float | None
        One timestamp per frame, for TUM output.
    t_norm_log_path : Path | None
        If set, write one line per frame: frame_id and ||t|| from recoverPose.
    ground_truth_path : str | Path | None
        If set, load and overlay ground-truth trajectory (TUM format).
    viewer_fps : float
        Viewer refresh rate in Hz when show_realtime is True (default 30).
    tracker_delay_ms : float
        Extra delay in milliseconds after each frame in the tracker thread (default 0).
        Increase to slow down trajectory building so the display can keep up.
    oracle : oracle.Oracle | None
        Stage 03c oracle ladder injector, built by main.py only when --oracle is not
        "none". Passed to Frame.update_pose; None (the default) is a no-op everywhere.
    point_source : str
        Stage 04-1a arm switch (design.md slot "Point source"): 'keyframe_pair' (the
        default; landmarks are created at keyframe insertion by
        local_mapping.process_new_keyframe, never per frame) or 'depth_filter' (stage
        04g; the SVO depth filter of depth_filter.py driven by depth_filter_arm.py:
        seeds per keyframe, one measurement per posed frame, promotion to a landmark).
    lm_params : local_mapping.LocalMappingParams | None
        Constants of the keyframe-pair source (the --lm_ flags); None = defaults.
    df_params : depth_filter.DepthFilterParams | None
        Constants of the depth filter (the --df_ flags); None = defaults.
    df_arm_params : depth_filter_arm.DepthFilterArmParams | None
        The arm's own constants (--df_promote_factor, --df_bootstrap_depth_ratio); None = defaults.
    refine_params : epipolar_refine.RefineParams | None
        Stage 04k (--epi_refine): refine every descriptor match of the point source along the
        epipolar line (process_new_keyframe and the depth-filter arm receive it as ``refine``);
        keyframe images are then kept in memory (they are released otherwise). None = off.
    m1_tracker : map_tracker.MapTracker | None
        Stage 04l (--tracking map): the map-based tracker against a perfect map. When set, each
        frame k >= 1 is posed by m1_tracker.on_frame and nothing else runs (no chain pose update,
        relocalization, hold, pruning, fusion, BA, point source, keyframe insertion or loop
        closure); the map file's points are released after each frame's pose is final. None
        (the default) leaves every statement of the chain as it was.
    rot_tracker : rotation_only.RotationOnlyTracker | None
        Stage 04r (--tracking rotation): the pre-map rotation-only mode. When set, each frame
        k >= 1 gets its orientation from rot_tracker.on_frame (translation held, no map) and
        nothing else runs, as under m1_tracker. None (the default) leaves every statement of the
        chain as it was.
    open_tracker : opening.OpeningTracker | None
        Stage 04u (--tracking rotation --init twoview): the rotation-only mode with the two-view
        initializer and the hand-off to the map tracker. When set, each frame k >= 1 is posed by
        open_tracker.on_frame and nothing else runs (rot_tracker is then None). None (the default)
        leaves every statement as it was.
    live_map : str
        Stage 04-m0 (--live_map): "off" (the default) leaves every statement as it was. "on" (the
        integrated system M0; open_tracker is then set) poses frame k with the opening driver, whose
        hand-off tracker tracks the live map (sync_live), and runs the chain's mapper statements
        (pruning, fusion, BA, the point source, the keyframe decision) on the frames the map posed
        and on the hand-off frame only; a false start removes the whole map.
    """
    if point_source not in ('keyframe_pair', 'depth_filter'):
        raise ValueError(f"unknown point_source {point_source!r} (built arms: keyframe_pair, depth_filter)")
    if lm_params is None:
        lm_params = LocalMappingParams()
    # Support both list-of-frames (legacy) and streaming iterator.
    if n_frames is None:
        try:
            n_frames = len(frames)
        except Exception:
            n_frames = None

    frame_iter = None
    if isinstance(frames, list):
        if len(frames) == 0:
            raise RuntimeError("No frames to process")
    else:
        frame_iter = iter(frames)
        first = next(frame_iter, None)
        if first is None:
            raise RuntimeError("No frames to process")
        frames = [first]
        if n_frames is None:
            raise RuntimeError("Streaming mode requires n_frames")

    global _stop_requested
    _stop_requested = False
    ''' if the terminal is a tty, start a thread to listen for terminal quit commands'''
    if sys.stdin.isatty():
        t = threading.Thread(target=_terminal_quit_listener, daemon=True)
        t.start()
        if show_realtime:
            print("Type 'q' or 'quit' in terminal to stop the pipeline (windows stay open).")
        else:
            print("Type 'q' or 'quit' in terminal to stop the pipeline.")

    world_map = Map(points=[], frames=frames, max_points=5000000)#build the world map.
    # Stage 04g (spec F1): the depth-filter arm exists only when selected; with arm None every
    # statement executed below is the keyframe-pair tree's.
    arm = DepthFilterArm(df_params, df_arm_params, lm_params, refine=refine_params) if point_source == 'depth_filter' else None
    ''' if the show_realtime is True, initialize the visualizer'''
    if show_realtime:
        init_visualizer(
            camera_spacing=camera_spacing,
            show_grid=show_grid,
            show_traj=show_traj,
            ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
            stop_requested_check=lambda: _stop_requested,
        )
    else:
        print("Running without real-time display (headless).")
    ''' initialize the loop closer'''
    '''note: keyframe is a frame that is used to anchor the map. and it is also used to detect loop closures.'''
    loop_closer = LoopCloser(
        keyframe_interval=keyframe_interval,
        min_gap=min_gap,
        match_threshold=50,
        keyframe_strategy=keyframe_strategy,
        keyframe_min_spacing=keyframe_min_spacing,
        keyframe_max_spacing=keyframe_max_spacing,
        keyframe_ratio_threshold=keyframe_ratio_threshold,
        loop_detection_strategy=loop_detection_strategy,
        vocabulary_path=vocabulary_path,
        num_spaced_candidates=num_spaced_candidates,
        relocalization_match_threshold=relocalization_match_threshold,
        relocalization_min_2d3d_pairs=relocalization_min_2d3d_pairs,
        relocalization_min_pnp_inliers=relocalization_min_pnp_inliers,
        relocalization_ratio_threshold=relocalization_ratio_threshold,
        relocalization_sanity_check_enabled=relocalization_sanity_check_enabled,
        relocalization_max_distance_from_keyframe=relocalization_max_distance_from_keyframe,
    )
    # Stage 04-m0 (spec F4): under live_map the mapper's keyframes begin at the hand-off, and the
    # depth filter's seeds follow their reference keyframe when BA moves it.
    m0_on = (live_map == "on")
    if not m0_on:
        loop_closer.maybe_add_keyframe(0, frames[0]) #add the first frame as a keyframe.
    if m0_on and arm is not None:
        arm.follow_ref_pose = True

    t_norm_file = None#file to log the t-norm of the frames.
    if t_norm_log_path is not None:
        t_norm_file = open(t_norm_log_path, "w")
        t_norm_file.write("# frame_id norm_t scale ||t_scaled|| reason\n")

    K = frames[0].K  # shared intrinsic matrix
    ba_window_size = max(2, int(ba_window_size))
    ba_max_points = max(50, int(ba_max_points))
    last_idx = 0

    '''stats is a dictionary to track the statistics of the pipeline.'''
    stats = {
        "homography": 0,
        "essential": 0,
        "vo_attempts": 0,
        "vo_fail": 0,
        "pnp_success": 0,
        "pnp_fail": 0,
        "pnp_consecutive_failures": 0,  # for adaptive PnP gate: reset on accept, +1 on non-accept when use_pnp
        "pnp_solver_wins": {"iterative": 0, "ap3p": 0, "epnp": 0, "sqpnp": 0, "oracle": 0},  # "oracle": stage 03c rung o4
        "pnp_fail_reasons": {},  # count per reason: too_few_correspondences, ransac_failed, inliers_below_min, ratio_below_threshold, multi_all_failed, safety_gate, unknown_solver
        "pnp_min_inlier_ratio": pnp_min_inlier_ratio,  # for summary: condition value for ratio_below_threshold
        "loop_closures": 0,
        "loop_closure_times": [],  # list of (frame_k, lc_kf_idx, duration_sec) for each correction
        "relocalizations": 0,
        "relocalization_failures": 0,
        "relocalization_attempts": [],  # list of (frame_k, success, matched_kf_idx|None, n_inliers, fail_reason|None)
        "pose_holds": 0,  # Stage 2: VO+reloc fail → hold/CV pose (not identity)
        "pose_hold_events": [],  # list of (frame_k, policy)
        "scale_fallback_reasons": {},
        "recent_vo_steps": [],
        "pose_update_times": [],
        "pnp_times": [],
        "ba_times": [],
        "ba_success": 0,   # alias of ba_accepted (kept for the summary line)
        "ba_fail": 0,      # alias of ba_rejected + ba_errors
        "ba_attempted": 0,
        "ba_skipped": 0,
        "ba_completed": 0,
        "ba_accepted": 0,
        "ba_rejected": 0,
        "ba_errors": 0,
        "ba_skip_reasons": {},
        "fusion_merges": 0,
        "fusion_runs": 0,
        # Stage 04-1a: local mapper counters (cumulative over keyframes; local_mapping.py adds to them)
        "lm_keyframes": 0,
        "lm_created": 0,
        "lm_obs_inserted": 0,
        "lm_culled": 0,
    }
    run_time_bounds = [None, None]  # [start, end] set by tracker/headless loop
    log_file = None
    _stdout_orig = None 
    ''' if log_file_path is set, open the file and redirect the stdout to the file.'''
    if log_file_path is not None:
        log_file = open(log_file_path, "w")
        log_file.write("# Invocation: " + (log_invocation or "(not recorded)") + "\n")
        log_file.write("#\n")
        _stdout_orig = sys.stdout
        sys.stdout = log_file
    # Stage 04k (spec F5 iii): the refinement's constants, once, before the frame loop.
    if refine_params is not None:
        print(f">> [EPI-REFINE] on patch_half={refine_params.patch_half:g} search_px={refine_params.search_px:g} "
              f"min_peak={refine_params.min_peak:g} ambiguity_ratio={refine_params.ambiguity_ratio:g}")

    def _may_release(k):
        """Stage 04k (spec F5 ii): with the refinement on, a keyframe's image is kept (both
        point sources read it later); every other frame is released as before."""
        if refine_params is None:
            return True
        return not any(idx == k for idx, _ in loop_closer.keyframes)

    def _ba_outcome_since(counters_before):
        """Outcome of the (at most one) BA attempt since counters_before was taken (spec B)."""
        delta = {key: int(stats.get(key, 0)) - counters_before.get(key, 0) for key in BA_COUNTER_KEYS}
        if delta["ba_attempted"] <= 0:
            return "none"
        if delta["ba_errors"] > 0:
            return "error"
        if delta["ba_accepted"] > 0:
            return "accepted"
        if delta["ba_rejected"] > 0:
            return "rejected"
        return "skipped"

    def _print_pose_online_line(k, pnp_accepted, hold_policy, ba_outcome, lc_applied):
        """One [POSE-ONLINE] line for frame k from its online snapshot (spec B); never raises."""
        try:
            fr = frames[k]
            if k == 0:
                source = "init"
            elif getattr(fr, "_relocalized", False):
                source = "reloc"
            elif hold_policy is not None:
                source = hold_policy
            elif pnp_accepted:
                source = "pnp"
            elif hasattr(fr, "_last_motion"):
                source = "vo_homography" if fr._last_motion.model == "HOMOGRAPHY" else "vo_essential"
            else:
                source = "hold_prev_fallback"  # unreachable: the hold policy covers every VO failure
            state = str(getattr(fr, "tracking_state", "OK")).replace(" ", "_")
            if k == 0:
                rot_deg, step = 0.0, 0.0
            else:
                prev = frames[k - 1]
                T_prev = getattr(prev, "online_world_T_cam", None)
                if T_prev is None:
                    T_prev = prev.world_T_cam
                rot_deg, step = pose_delta_stats(T_prev, fr.online_world_T_cam)
            print(format_pose_online_line(k, source, state, rot_deg, step, pnp_accepted, ba_outcome, lc_applied))
        except Exception as e:
            print(f"  >> [POSE-ONLINE-ERROR] frame {k}: {e}")
            print(format_pose_online_line(k, "hold_prev_fallback", "UNKNOWN", float("nan"), float("nan"),
                                          False, "none", False))

    # Stage 03a (spec A1/B): frame 0's online snapshot and provenance line (pose is the identity).
    frames[0].online_world_T_cam = frames[0].world_T_cam.copy()
    _print_pose_online_line(0, pnp_accepted=False, hold_policy=None, ba_outcome="none", lc_applied=False)
    # Stage 03c (spec C2): the oracle setup line, once, before frame 1 is processed.
    if oracle is not None:
        print(oracle.setup_line())
    # Stage 04l (spec E): the perfect map's setup line, then frame 0's keyframe points (its
    # identity pose is final: the anchor of the file's gauge).
    if m1_tracker is not None:
        m1_tracker.use_homography = bool(use_homography)
        print(m1_tracker.setup_line())
        m1_tracker.release(0, frames, world_map)
    # Stage 04r (spec F): the rotation-only mode's setup line and frame 0 (the gauge, the first
    # panorama keyframe); the tracker counts map points from the pipeline's Map at summary time.
    if rot_tracker is not None:
        rot_tracker.world_map = world_map
        print(rot_tracker.setup_line())
        rot_tracker.start(frames)
    # Stage 04u (spec C1): the opening driver's setup line, then its first rotation segment.
    if open_tracker is not None:
        open_tracker.world_map = world_map
        open_tracker.use_homography = bool(use_homography)
        print(open_tracker.setup_line())
        open_tracker.start(frames)
    # Stage 04-m0 (spec F, F7): the integrated system's bookkeeping and its setup line, before frame 1.
    # m0_init_ids: the current map's initial set (the ids in world_map right after the hand-off's
    # release; empty before the first hand-off and after a reset).
    m0_init_ids = set()
    m0_counts = dict(mapper_frames=0, keyframes_added=0, handoffs=0, resets=0)
    m0_camera_supported = ((lambda f: bool(getattr(f, "m0_supported", False)))
                           if m0_on and ba_free_cameras == "supported" else None)
    if m0_on:
        m0_rot_state = getattr(getattr(open_tracker, "tracker_factory", None), "map_rot_state", "off")
        print(f"  >> [M0-SETUP] point_source={point_source} map_rot_state={m0_rot_state} "
              f"ba_free_cameras={ba_free_cameras} ba_jump_gate={ba_jump_gate} ba_outlier_reject={ba_outlier_reject} "
              f"follow_ref_pose={1 if (arm is not None and arm.follow_ref_pose) else 0}")

    def _prune_map_points(current_fid, frames_dict_for_reproj=None):
        """
        Prune map points based on enabled policies (B high_reproj, D numeric_corruption,
        E outlier_hits; all default off).  The chain-era low-observation age prune
        (policy A) and its keyframe protection were retired in stage 04-1a: landmark
        culling is local_mapping.cull_points, by keyframe observation count.
        current_fid uses Frame.id as the time base.
        frames_dict_for_reproj: dict[frame_id -> Frame] for per-point reprojection error, or None.
        Returns count pruned.
        """
        pruned = 0
        if world_map is None or not getattr(world_map, "points", None):
            return 0

        per_point_err = None
        med_err = None
        thresh_err = None
        if prune_high_reproj and frames_dict_for_reproj is not None:
            try:
                per_point_err = compute_per_point_reprojection_error(frames_dict_for_reproj, world_map, K)
                if per_point_err:
                    vals = np.array(list(per_point_err.values()), dtype=np.float64)
                    med_err = float(np.median(vals))
                    # Conservative internal threshold: max(8px, 3x median)
                    thresh_err = max(8.0, 3.0 * med_err)
            except Exception:
                per_point_err = None

        # Iterate over a snapshot list since we may delete points.
        for pt in list(world_map.points):
            pid = getattr(pt, "id", None)
            if pid is None:
                continue

            # Policy D: numeric corruption (toggle)
            if prune_numeric_corruption:
                x = getattr(pt, "point", None)
                if x is None or (not np.all(np.isfinite(x))):
                    world_map.remove_point(pid)
                    pruned += 1
                    continue

            # Policy E: outlier hits + last-seen age (toggle)
            if prune_outlier_hits:
                hits = int(getattr(pt, "outlier_hits", 0) or 0)
                last_seen = int(getattr(pt, "last_seen_fid", -1))
                if last_seen < 0:
                    try:
                        last_seen = int(max(getattr(pt, "frames", []) or [-1]))
                    except Exception:
                        last_seen = -1
                not_seen_age = (int(current_fid) - last_seen) if last_seen >= 0 else 0
                if hits >= int(prune_outlier_hit_count) and not_seen_age >= int(prune_outlier_last_seen_age):
                    world_map.remove_point(pid)
                    pruned += 1
                    continue

            # Policy B: high reprojection error (toggle)
            if per_point_err is not None and thresh_err is not None:
                e = per_point_err.get(pid)
                if e is not None and float(e) > float(thresh_err):
                    world_map.remove_point(pid)
                    pruned += 1
                    continue

        if pruned > 0:
            msg = f"  >> [PRUNE] pruned {pruned} map points"
            if thresh_err is not None and prune_high_reproj:
                msg += f" (high_reproj thresh={thresh_err:.2f}px, median={med_err:.2f}px)"
            print(msg)
        return pruned

    def _maybe_fuse(k):
        """Periodically merge near-duplicate landmarks (Stage 3.2)."""
        if not use_point_fusion:
            return 0
        try:
            interval = int(fusion_interval)
        except Exception:
            interval = 0
        if interval <= 0 or k % interval != 0:
            return 0
        method = getattr(frames[k], "method", "ORB") if k < len(frames) else "ORB"
        n_merged = fuse_duplicate_points(
            world_map, frames,
            max_dist=float(fusion_max_dist),
            max_desc_dist=float(fusion_max_desc_dist),
            method=method,
        )
        stats["fusion_runs"] = stats.get("fusion_runs", 0) + 1
        if n_merged > 0:
            stats["fusion_merges"] = stats.get("fusion_merges", 0) + n_merged
            print(f"  >> [FUSION] merged {n_merged} duplicate map points")
        return n_merged

    def _maybe_prune(k, frames_dict_for_reproj=None):
        """Run pruning on its own cadence (independent of BA)."""
        try:
            interval = int(prune_interval)
        except Exception:
            interval = 0
        if interval <= 0:
            return 0
        if k % interval != 0:
            return 0
        try:
            current_fid = int(getattr(frames[k], "id", k))
        except Exception:
            current_fid = k
        return _prune_map_points(current_fid, frames_dict_for_reproj=frames_dict_for_reproj)

    def _run_one_frame_m1(k):
        """Stage 04l (spec E): the map-based tracker poses frame k, the online snapshot and its
        [POSE-ONLINE] line follow, then the frame's keyframe points (if any) are released."""
        nonlocal last_idx
        state = m1_tracker.on_frame(k, frames, world_map)
        frames[k].online_world_T_cam = frames[k].world_T_cam.copy()
        source, pose_state = M1_ONLINE_LABELS.get(state, ("predicted", "LOST"))
        try:
            T_prev = getattr(frames[k - 1], "online_world_T_cam", None)
            if T_prev is None:
                T_prev = frames[k - 1].world_T_cam
            rot_deg, step = pose_delta_stats(T_prev, frames[k].online_world_T_cam)
        except Exception as e:
            print(f"  >> [POSE-ONLINE-ERROR] frame {k}: {e}")
            rot_deg, step = float("nan"), float("nan")
        print(format_pose_online_line(k, source, pose_state, rot_deg, step, False, "none", False))
        m1_tracker.release(k, frames, world_map)
        last_idx = k

    def _run_one_frame_rot(k):
        """Stage 04r (spec F): the rotation-only mode orients frame k; the online snapshot and its
        [POSE-ONLINE] line follow (the pattern of _run_one_frame_m1). Nothing else runs."""
        nonlocal last_idx
        state = rot_tracker.on_frame(k, frames)
        frames[k].online_world_T_cam = frames[k].world_T_cam.copy()
        source, pose_state = ("rotation", "ROT") if state == "measured" else ("predicted", "LOST")
        try:
            T_prev = getattr(frames[k - 1], "online_world_T_cam", None)
            if T_prev is None:
                T_prev = frames[k - 1].world_T_cam
            rot_deg, step = pose_delta_stats(T_prev, frames[k].online_world_T_cam)
        except Exception as e:
            print(f"  >> [POSE-ONLINE-ERROR] frame {k}: {e}")
            rot_deg, step = float("nan"), float("nan")
        print(format_pose_online_line(k, source, pose_state, rot_deg, step, False, "none", False))
        last_idx = k

    def _run_one_frame_open(k):
        """Stage 04u (spec C2): the opening driver poses frame k; the online snapshot and its
        [POSE-ONLINE] line follow (the pattern of _run_one_frame_rot). Nothing else runs."""
        nonlocal last_idx
        state = open_tracker.on_frame(k, frames)
        frames[k].online_world_T_cam = frames[k].world_T_cam.copy()
        source, pose_state = {"measured": ("rotation", "ROT"), "unsupported": ("predicted", "LOST"),
                              "lost": ("predicted", "LOST"), "init": ("init", "OK"),
                              "tracked": ("map", "OK")}[state]
        try:
            T_prev = getattr(frames[k - 1], "online_world_T_cam", None)
            if T_prev is None:
                T_prev = frames[k - 1].world_T_cam
            rot_deg, step = pose_delta_stats(T_prev, frames[k].online_world_T_cam)
        except Exception as e:
            print(f"  >> [POSE-ONLINE-ERROR] frame {k}: {e}")
            rot_deg, step = float("nan"), float("nan")
        print(format_pose_online_line(k, source, pose_state, rot_deg, step, False, "none", False))
        last_idx = k

    def _m0_error_line(k, e):
        msg = f"{type(e).__name__}: {e}".replace("\r", " ").replace("\n", " ")
        return f"  >> [M0-ERROR] frame={int(k)} error={msg}"

    def _m0_pose(k):
        """Stage 04-m0 (spec F1, F2): the opening driver poses frame k, its hand-off tracker reading
        the live map; frame k's links are counted; a false start removes the whole map. Returns
        (state, inl_init, inl_new)."""
        nonlocal m0_init_ids
        # F1. the pose
        if open_tracker.tracker is not None:
            open_tracker.tracker.sync_live(frames, world_map)
        resets_before = open_tracker.n_resets
        state = open_tracker.on_frame(k, frames)
        if state == "init":
            m0_init_ids = {p.id for p in world_map.points}
        inl_init = inl_new = 0
        for pid in frames[k].keypoint_to_point.values():
            if pid in m0_init_ids:
                inl_init += 1
            else:
                inl_new += 1
        frames[k].m0_supported = (state == "tracked")
        # F2. a false start (the opening has removed the reset map's initial points already)
        if open_tracker.n_resets > resets_before:
            try:
                m0_init_ids = set()
                m0_counts["resets"] += 1
                removed = m0_clear_map(world_map, frames)
                seeds_cleared = 0
                if arm is not None:
                    seeds_cleared = len(arm.live)
                    arm.live = []
                keyframes_cleared = len(loop_closer.keyframes)
                del loop_closer.keyframes[:]
                print(f"  >> [M0-RESET] frame={int(k)} map={int(open_tracker.n_maps)} removed={int(removed)} "
                      f"seeds_cleared={int(seeds_cleared)} keyframes_cleared={int(keyframes_cleared)}")
            except Exception as e:
                print(_m0_error_line(k, e))
        return state, inl_init, inl_new

    def _m0_output(k, state, mapper, kf, inl_init, inl_new, ba_outcome):
        """Stage 04-m0 (spec F5, F6): frame k's online snapshot and [POSE-ONLINE] line, then its
        [M0-FRAME] line (the last line printed for frame k)."""
        source, pose_state = M0_ONLINE_LABELS.get(state, ("predicted", "LOST"))
        rot_deg, step = float("nan"), float("nan")
        try:
            frames[k].online_world_T_cam = frames[k].world_T_cam.copy()
            T_prev = getattr(frames[k - 1], "online_world_T_cam", None)
            if T_prev is None:
                T_prev = frames[k - 1].world_T_cam
            rot_deg, step = pose_delta_stats(T_prev, frames[k].online_world_T_cam)
        except Exception as e:
            print(_m0_error_line(k, e))
        try:
            print(format_pose_online_line(k, source, pose_state, rot_deg, step, False, ba_outcome, False))
        except Exception as e:
            print(_m0_error_line(k, e))
        try:
            points = len(world_map.points)
            init_points = sum(1 for pid in m0_init_ids if pid in world_map.point_by_id)
            print(format_m0_frame_line(k, state, mapper, kf, points, init_points, inl_init, inl_new))
        except Exception as e:
            print(_m0_error_line(k, e))

    def _run_one_frame(k):
        """Run pose update, BA, local mapping (at keyframes) and loop closure for frame k. Returns last_idx (k)."""
        nonlocal last_idx
        if m1_tracker is not None:
            _run_one_frame_m1(k)
            return
        if rot_tracker is not None:
            _run_one_frame_rot(k)
            return
        if open_tracker is not None and not m0_on:
            _run_one_frame_open(k)
            return
        use_pnp = (pnp_interval > 0) and (k % pnp_interval == 0) #use pnp if pnp_interval is greater than 0 and k is a multiple of pnp_interval.
        recent = frames[max(0, k - local_map_window): k] if local_map_window > 0 else None #recent is a list of frames from k - local_map_window to k.
        pnp_correction_window = None
        if use_pnp and pnp_interval > 0:
            # Stage 2 policy B: distribute accepted PnP over last pnp_interval frames.
            pnp_correction_window = frames[max(0, k - pnp_interval): k + 1]
        # Stage 03a (spec B): provenance bookkeeping, read-only on stats.
        pnp_success_before = int(stats.get("pnp_success", 0))
        ba_counters_before = {key: int(stats.get(key, 0)) for key in BA_COUNTER_KEYS}
        hold_policy_applied = None
        lc_applied = False
        m0_kf = False
        if m0_on:
            # Stage 04-m0 (spec F1, F2): the opening driver poses frame k; the mapper statements below
            # run only when the map posed it or it is the hand-off frame (spec F3).
            m0_state, m0_inl_init, m0_inl_new = _m0_pose(k)
            tracking_ok = m0_writes_map(m0_state)
            if tracking_ok:
                m0_counts["mapper_frames"] += 1
        else:
            t_pose_start = time.perf_counter()
            frames[k].update_pose(
                frames[k - 1], world_map,
                use_pnp=use_pnp, use_homography=use_homography,
                t_norm_log_file=t_norm_file,
                recent_frames=recent,
                stats=stats,
                pnp_min_inlier_ratio=pnp_min_inlier_ratio,
                pnp_solver=pnp_solver,
                pnp_refinement=pnp_refinement,
                pnp_use_initial_guess=pnp_use_initial_guess,
                pnp_multi_inlier_gate=pnp_multi_inlier_gate,
                pnp_reprojection_error=pnp_reprojection_error,
                pnp_adaptive_gate=pnp_adaptive_gate,
                pnp_adaptive_gate_factor=pnp_adaptive_gate_factor,
                use_relative_scale=use_relative_scale,
                outlier_hit_accounting=bool(prune_outlier_hits),
                pnp_correction_window=pnp_correction_window,
                local_map_search_radius=local_map_search_radius,
                local_map_max_desc_dist=local_map_max_desc_dist,
                local_map_max_depth_ratio=local_map_max_depth_ratio,
                oracle=oracle,
            )
            t_pose_elapsed = time.perf_counter() - t_pose_start
            stats["pose_update_times"].append(t_pose_elapsed)
            print(f"  >> [TIMING] pose update frame {k}: {t_pose_elapsed:.3f}s")

            # When VO failed (no _last_motion), try relocalization against keyframes.
            if not hasattr(frames[k], "_last_motion") and use_relocalization and len(loop_closer.keyframes) > 0:
                print(f"  >> VO failed at frame {k}; attempting relocalization against keyframes.")
                t_reloc_start = time.perf_counter()
                reloc_kf_idx, reloc_pose, reloc_inliers, reloc_fail_reason = loop_closer.relocalize(k, frames[k], world_map)
                t_reloc_elapsed = time.perf_counter() - t_reloc_start
                if reloc_kf_idx is not None and reloc_pose is not None:
                    frames[k].set_camera_pose(reloc_pose)
                    frames[k]._relocalized = True
                    frames[k].pose_valid = True
                    frames[k].tracking_state = "RELOCALIZED"
                    stats["relocalizations"] = stats.get("relocalizations", 0) + 1
                    stats.setdefault("relocalization_attempts", []).append((k, True, reloc_kf_idx, reloc_inliers, None))
                    print(
                        f"  >> RELOCALIZATION succeeded at frame {k} (matched keyframe {reloc_kf_idx}, "
                        f"{reloc_inliers} inliers, {t_reloc_elapsed:.2f}s)"
                    )
                else:
                    stats["relocalization_failures"] = stats.get("relocalization_failures", 0) + 1
                    stats.setdefault("relocalization_attempts", []).append((k, False, None, 0, reloc_fail_reason))
                    print(f"  >> Relocalization failed at frame {k}: {reloc_fail_reason}.")

            # Stage 2.1: never leave identity after VO (+ optional reloc) failure.
            if not hasattr(frames[k], "_last_motion") and not getattr(frames[k], "_relocalized", False):
                prev = frames[k - 1]
                policy = "hold_prev"
                held = prev.world_T_cam.copy()
                if (
                    k >= 2
                    and getattr(frames[k - 1], "pose_valid", True)
                    and getattr(frames[k - 2], "pose_valid", True)
                ):
                    # Constant-velocity: T_k = T_{k-1} (T_{k-2}^{-1} T_{k-1})
                    T_prev = frames[k - 1].world_T_cam
                    T_prev2 = frames[k - 2].world_T_cam
                    rel = np.linalg.inv(T_prev2) @ T_prev
                    held = T_prev @ rel
                    policy = "constant_velocity"
                if frames[k].set_world_pose(held):
                    frames[k].pose_valid = False
                    frames[k].tracking_state = "LOST"
                    stats["pose_holds"] = stats.get("pose_holds", 0) + 1
                    stats.setdefault("pose_hold_events", []).append((k, policy))
                    hold_policy_applied = policy
                    print(
                        f"  >> [POSE-HOLD] frame {k}: VO+reloc failed; "
                        f"applied {policy} (pose_valid=False, not identity)"
                    )
                else:
                    # Last resort: copy previous pose without CV composition.
                    frames[k].set_world_pose(prev.world_T_cam.copy())
                    frames[k].pose_valid = False
                    frames[k].tracking_state = "LOST"
                    stats["pose_holds"] = stats.get("pose_holds", 0) + 1
                    stats.setdefault("pose_hold_events", []).append((k, "hold_prev_fallback"))
                    hold_policy_applied = "hold_prev_fallback"
                    print(
                        f"  >> [POSE-HOLD] frame {k}: CV pose invalid; "
                        f"held previous pose (pose_valid=False)"
                    )

            # Only run BA and loop closure when frame has a valid pose (VO or relocalization).
            tracking_ok = (
                (hasattr(frames[k], "_last_motion") or getattr(frames[k], "_relocalized", False))
                and getattr(frames[k], "pose_valid", True)
            )

        # Prune on cadence even when BA is disabled (policies A/D/E). Policy B is only
        # evaluated when we have a frames_dict_for_reproj (i.e., after BA below).
        if tracking_ok:
            _maybe_prune(k, frames_dict_for_reproj=None)
            _maybe_fuse(k)

        def _run_ba_attempt(frames_window, label):
            """
            One bundle-adjustment attempt (stage 01a, spec A6), shared by both BA modes.

            Snapshot window poses and all map point coordinates; call bundle_adjust with the
            processed frames as outer observers; on "skipped" count it and change nothing; on
            "completed" apply the reprojection gate (on all of the problem's observations,
            fixed observers included) and the jump gate (window poses), accept or roll back
            poses AND points; on any exception roll back the same snapshot and count an error.
            Returns True iff an accepted solve happened (pruning policy B runs only then).
            """
            frames_window = list(frames_window)
            bao_on = (ba_outlier_reject == "on")
            bsup_on = (ba_jump_gate == "off" or ba_free_cameras == "supported")
            min_cam_pts = (optimization.BA_MIN_FREE_CAMERA_POINTS if ba_free_cameras == "supported" else 0)
            pose_snapshot = {f.id: f.cam_T_world.copy() for f in frames_window}
            point_snapshot = {p.id: p.point.copy() for p in world_map.points}
            accepted = False
            outcome_recorded = False
            try:
                pre_window_poses = [pose_snapshot[f.id] for f in frames_window]
                pre_max_step, pre_p95_step = _max_step_in_window(pre_window_poses)
                t_ba_only_start = time.perf_counter()
                info = bundle_adjust(
                    frames_window, world_map, K,
                    max_points=ba_max_points, method=ba_method,
                    use_analytical_jacobian=ba_use_analytical_jacobian,
                    min_point_observations=ba_min_point_observations,
                    point_sanity_check=ba_point_sanity_check,
                    all_frames=frames[:k + 1],
                    max_fixed_observers=ba_max_fixed_observers,
                    huber_delta_px=ba_huber_delta_px,
                    min_anchor_points=ba_min_anchor_points,
                    max_iterations=ba_max_iterations,
                    reject_outliers=bao_on,
                    sigma_fn=(local_mapping._keypoint_sigmas if bao_on else None),
                    min_camera_points=min_cam_pts,
                    camera_supported=m0_camera_supported,
                )
                t_ba_only_elapsed = time.perf_counter() - t_ba_only_start
                status = info.get("status") if isinstance(info, dict) else None
                if status == "skipped":
                    record_ba_outcome(stats, "skipped", reason=info.get("reason"))
                    return False
                if status != "completed":
                    raise RuntimeError(f"bundle_adjust returned unexpected status {status!r}")
                stats["ba_times"].append((t_ba_only_elapsed, ba_method))
                print(f"  >> [TIMING] {label} frame {k} (method={ba_method}): {t_ba_only_elapsed:.3f}s")
                err_before = float(info["reproj_before_px"])
                err_after = float(info["reproj_after_px"])
                n_obs = int(info.get("n_obs", 0))
                print()
                print(
                    f"  >> [{label}] Reprojection error BEFORE: {err_before:.3f} px "
                    f"({n_obs} observations, gauge={info.get('gauge')}, "
                    f"fixed observers={info.get('n_fixed_observers', 0)})"
                )
                post_window_poses = [f.cam_T_world.copy() for f in frames_window]
                post_max_step, _ = _max_step_in_window(post_window_poses)
                max_allowed_step = max(
                    ba_max_allowed_step_factor * max(pre_p95_step, 1e-6),
                    pre_max_step + 2.0,
                )
                # Two gates: reprojection must not grow past the ratio; no pose may jump.
                bad_reproj = err_before > 0.0 and err_after > ba_bad_reproj_ratio * err_before
                # Stage 04z (spec C2): the jump gate (absolute map units) is a switch.
                bad_jump = (ba_jump_gate == "on") and post_max_step > max_allowed_step
                if bad_reproj or bad_jump:
                    _restore_ba_snapshot(frames_window, pose_snapshot, point_snapshot, world_map)
                    record_ba_outcome(stats, "completed", accepted=False)
                    outcome_recorded = True
                    reason = "error increased" if bad_reproj and not bad_jump else (
                        "jump spike" if bad_jump and not bad_reproj else "error+jump"
                    )
                    print()
                    print(f"  >> [{label}] Update rejected ({reason}); state restored.")
                else:
                    record_ba_outcome(stats, "completed", accepted=True)
                    outcome_recorded = True
                    accepted = True
                    # Stage 03a (spec C2): which already-emitted frames the accepted solve moved
                    # (cam_T_world on both sides; frames[k] itself excluded).
                    print_correction(
                        k, "ba", pose_snapshot,
                        [(f, f.cam_T_world) for f in frames_window],
                        exclude_id=frames[k].id,
                    )
                if bao_on:
                    # Stage 04y (spec A4): unlink the final outliers of an accepted solve, then
                    # one [BA-OUTLIERS] line (also for a rolled-back solve, removed=0).
                    # Stage 04z (spec C4): through the never-raising epilogue.
                    print(ba_outlier_epilogue(world_map, frames[:k + 1], k, n_obs, info, accepted))
                if bsup_on:
                    # Stage 04z (spec C3): one [BA-SUPPORT] line per completed solve.
                    print(format_ba_support_line(
                        k, info.get("n_free", 0), info.get("n_held_build", 0), info.get("free_min_rows", -1),
                        info.get("p2_ran", 0), info.get("n_held_p2", 0), info.get("p2_min_rows", -1),
                        info.get("p2_under10", -1)))
                print()
                print(
                    f"  >> [{label}] Reprojection error AFTER:  {err_after:.3f} px "
                    f"({n_obs} observations)  [{t_ba_only_elapsed:.1f}s]"
                )
            except Exception as e:
                if outcome_recorded:
                    # Stage 04z (spec C4): the accept or roll-back decision and its accounting are
                    # never revisited after they are recorded.
                    print()
                    print(f"  >> [{label}] Warning: Bundle Adjustment epilogue failed at frame {k}: {e}")
                else:
                    _restore_ba_snapshot(frames_window, pose_snapshot, point_snapshot, world_map)
                    record_ba_outcome(stats, "error")
                    print()
                    print(f"  >> [{label}] Warning: Bundle Adjustment failed at frame {k}: {e}")
                    return False
            if accepted:
                # Policy B (high reprojection error) relies on the BA window; evaluate only here.
                frames_dict_for_reproj = {f.id: f for f in frames_window} if prune_high_reproj else None
                _maybe_prune(k, frames_dict_for_reproj=frames_dict_for_reproj)
            return accepted

        '''we implement sliding window bundle adjustment.'''
        if (
            tracking_ok
            and use_bundle_adjustment
            and ba_mode == "sliding_window"
            and ba_interval > 0
            and k > 0
            and k % ba_interval == 0
        ):
            window_start = max(0, k - ba_window_size + 1) #window is current frame index - ba_window_size + 1 to current frame index.
            _run_ba_attempt(frames[window_start: k + 1], "BA")

        ''' loop closure detection and correction.'''
        try:
            if tracking_ok:
                # Stage 04g (spec F2): under the depth-filter arm every posed frame measures the
                # live seeds and promotes the converged ones, after the frame's sliding-window BA
                # and before the keyframe decision; it must never abort the run.
                if arm is not None:
                    try:
                        arm.on_frame(k, frames[k], loop_closer.keyframes, world_map, frames)
                    except Exception as e:
                        print(f"  >> [DF-ERROR] frame {k}: {e}")
                if m0_on and m0_state == "init":
                    # Stage 04-m0 (spec F3 (i)): the hand-off frame is the mapper's first keyframe.
                    del loop_closer.keyframes[:]
                    loop_closer.keyframes.append((k, frames[k]))
                    added = True
                else:
                    added = loop_closer.maybe_add_keyframe(k, frames[k])
                m0_kf = bool(added)
                if (
                    added
                    and use_bundle_adjustment
                    and ba_mode == "keyframe_local_covisibility"
                    and k > 0
                ):
                    try:
                        center_frame = frames[k]
                        center_id = center_frame.id

                        # Map keyframe Frame.id -> Frame object.
                        keyframe_id_to_frame = {
                            kf_frame.id: kf_frame
                            for _, kf_frame in loop_closer.keyframes
                        }

                        neighbor_counts = {}  # neighbor_frame_id -> shared_point_count
                        min_shared = max(0, int(keyframe_local_min_shared_points))
                        topk = max(0, int(keyframe_local_topk))

                        # Count shared map points between center keyframe and other keyframes.
                        for pid in center_frame.keypoint_to_point.values():
                            pt = world_map.get_point(pid)
                            if pt is None:
                                continue
                            for fid2 in (getattr(pt, "frames", None) or pt.observations.keys()):
                                if fid2 == center_id:
                                    continue
                                if fid2 in keyframe_id_to_frame:
                                    neighbor_counts[fid2] = neighbor_counts.get(fid2, 0) + 1

                        neighbors_sorted = sorted(
                            neighbor_counts.items(), key=lambda x: x[1], reverse=True
                        )
                        neighbor_fids = [
                            fid for fid, cnt in neighbors_sorted
                            if cnt >= min_shared and fid != center_id
                        ][:topk]
                        neighbor_frames = [keyframe_id_to_frame[fid] for fid in neighbor_fids]
                        frames_window = [center_frame] + neighbor_frames
                        print()
                        print(f"  >> [BA-keyframe-local] frame {k}: neighbors={len(neighbor_frames)}")
                        # One attempt per keyframe; a 1-frame window is a recorded skip.
                        _run_ba_attempt(frames_window, "BA-keyframe-local")
                    except Exception as e:
                        # Only neighbourhood selection can raise here; the attempt helper
                        # handles (and counts) its own errors.
                        print()
                        print(f"  >> [BA-keyframe-local] Warning: neighbourhood selection failed at frame {k}: {e}")
                # Stage 04-1a (spec A6): the local mapper runs once per keyframe insertion,
                # after the keyframe-local BA block and before loop detection. It is the
                # only point source; it must never abort the run.
                if m0_on and m0_state == "init":
                    m0_counts["handoffs"] += 1
                    print(f"  >> [M0-HANDOFF] frame={int(k)} map={int(open_tracker.n_maps)} "
                          f"init_points={len(world_map.points)}")
                if added:
                    if arm is None:
                        try:
                            process_new_keyframe(k, frames[k], loop_closer.keyframes, world_map,
                                                 frames, lm_params, stats, refine=refine_params)
                        except Exception as e:
                            print(f"  >> [LOCAL-MAP-ERROR] frame {k}: {e}")
                    else:
                        # Stage 04g (spec F3): the arm's keyframe step (insertion, culling,
                        # retirement, seeding) in place of the keyframe-pair source.
                        try:
                            arm.on_keyframe(k, frames[k], loop_closer.keyframes, world_map, frames, stats)
                        except Exception as e:
                            print(f"  >> [DF-ERROR] frame {k}: {e}")
            if tracking_ok and use_loop_closure and k % loop_closer.keyframe_interval == 0: #if its time to check for loop closures.
                ''' detect loop closure.'''
                t_lc_start = time.perf_counter() #time the loop closure detection.
                lc_kf_idx, lc_kf, lc_pose, lc_inliers = loop_closer.detect(
                    k, frames[k], world_map
                )
                t_lc_elapsed = time.perf_counter() - t_lc_start
                if t_lc_elapsed > 0.5:
                    print()
                    print(f"  >> [LC] Loop closure check took {t_lc_elapsed:.1f}s")
                if lc_kf_idx is not None:
                    stats["loop_closures"] = stats.get("loop_closures", 0) + 1
                    # Report to terminal so user sees it even when log is redirected
                    if terminal_stdout is not None:
                        terminal_stdout.write(
                            f"\n  >> LOOP CLOSURE detected at frame {k} <-> frame {lc_kf_idx} "
                            f"({lc_inliers} PnP inliers)\n"
                        )
                        terminal_stdout.flush()
                    print()
                    print(
                        f"  >> LOOP CLOSURE detected: frame {k} <-> frame {lc_kf_idx} "
                        f"({lc_inliers} PnP inliers)"
                    )
                    t_correct_start = time.perf_counter()
                    # Stage 03a (spec C3): snapshot the frames the corrector may rewrite.
                    lc_window = frames[lc_kf_idx + 1: k]
                    lc_snapshot = {f.id: f.world_T_cam.copy() for f in lc_window}
                    if loop_closure_strategy == 'pose_graph':
                        corrected_points = correct_trajectory_pose_graph(
                            frames, lc_kf_idx, k, lc_pose, world_map=world_map
                        )
                        print(
                            f"  >> Trajectory corrected (pose-graph): frames {lc_kf_idx+1}..{k}; "
                            f"map points corrected: {corrected_points}"
                        )
                    else:
                        corrected_points = correct_trajectory(
                            frames, lc_kf_idx, k, lc_pose, world_map=world_map
                        )
                        print(
                            f"  >> Trajectory corrected (linear): frames {lc_kf_idx+1}..{k}; "
                            f"map points corrected: {corrected_points}"
                        )
                    lc_applied = True
                    print_correction(
                        k, "loop_closure", lc_snapshot,
                        [(f, f.world_T_cam) for f in lc_window],
                        exclude_id=frames[k].id,
                    )
                    t_correct_elapsed = time.perf_counter() - t_correct_start
                    stats.setdefault("loop_closure_times", []).append(
                        (k, lc_kf_idx, t_correct_elapsed)
                    )
                    print(f"  >> [LC] Correction took {t_correct_elapsed:.2f}s")
        except Exception as e:
            print()
            print(f"  >> [LC] Warning: Loop closure failed at frame {k}: {e}")

        # Stage 04-m0 (spec F5, F6): the online snapshot after the mapper, [POSE-ONLINE], [M0-FRAME].
        if m0_on:
            if m0_kf:
                m0_counts["keyframes_added"] += 1
            _m0_output(k, m0_state, tracking_ok, m0_kf, m0_inl_init, m0_inl_new,
                       _ba_outcome_since(ba_counters_before))
            last_idx = k
            return

        # Stage 03a (spec A1/B): the ONLINE pose of frame k is its pose as the per-frame
        # processing ends; snapshot it (a copy, never updated afterwards) and print provenance.
        frames[k].online_world_T_cam = frames[k].world_T_cam.copy()
        _print_pose_online_line(
            k,
            pnp_accepted=int(stats.get("pnp_success", 0)) > pnp_success_before,
            hold_policy=hold_policy_applied,
            ba_outcome=_ba_outcome_since(ba_counters_before),
            lc_applied=lc_applied,
        )
        # Stage 03c (spec A5): the O4 gauge anchor, captured once when frame anchor_k finishes
        # (anchor_k == 0 is the identity and needs no capture).
        if oracle is not None and k == oracle.anchor_k:
            oracle.set_anchor_pose(frames[k].world_T_cam)
        last_idx = k

    # Progress line and loop-closure alerts go to the terminal when --log_file is set.
    terminal_stdout = _stdout_orig if _stdout_orig is not None else sys.stdout

    ''' if show_realtime is True (display enabled), run the tracker in the background and update the viewer at viewer_fps.'''
    try:
        if show_realtime:
            # Tracker runs in background; viewer pulls latest state at VIEWER_FPS.
            state_lock = threading.Lock()
            latest_state = [None]
            tracker_finished = [False]
            last_idx_holder = [0]
            ''' initial state is the initial state of the viewer.'''
            initial_state = ViewerState(
                trajectory=[frames[0].pose.copy()],
                map_points_display=[],
                keyframe_indices=[0],
                tracking_state="INITIALIZING",
                overlay_image=build_overlay_image(frames[0], None, "INITIALIZING", keypoints_only=True),
            )
            if hasattr(frames[0], "release_image") and _may_release(0):
                frames[0].release_image()
            with state_lock:
                latest_state[0] = initial_state #set the initial state of the viewer.

            n_frames = int(n_frames)
            ''' loops over frames, runs _run_one_frame for each frame, which updates the pose of the frame and runs bundle adjustment and loop closure.'''
            def run_tracker():
                run_time_bounds[0] = time.perf_counter()
                for k in range(1, n_frames):
                    if _stop_requested:
                        print()
                        print(f"Pipeline stopped at frame {k} (terminal quit).")
                        break
                    # Streaming: append next frame before processing k.
                    if frame_iter is not None and len(frames) <= k:
                        nxt = next(frame_iter, None)
                        if nxt is None:
                            break
                        frames.append(nxt)
                    _run_one_frame(k) #update the pose of the frame and run bundle adjustment and loop closure.
                    last_idx_holder[0] = k #update the last index of the frame.
                    ''' get the pose for each frame from 0 to k. trajectoroy is a list of k+1 poses that the 
                    the visualizer will use this trajectory to draw the trajectory of the camera.''' 
                    trajectory = [frames[i].pose.copy() for i in range(k + 1)] 
                    map_points_display = [(p.x, p.y, p.z, len(p.frames)) for p in world_map.points] #get the map points for the display.
                    keyframe_indices = [idx for idx, _ in loop_closer.keyframes]
                    tracking_state = (
                        "TRACKING OK" if hasattr(frames[k], "_last_motion") else
                        "RELOCALIZED" if getattr(frames[k], "_relocalized", False) else
                        "LOST"
                    )
                    overlay_image = build_overlay_image(
                        frames[k], frames[k - 1], tracking_state, keypoints_only=True
                    )
                    # Free pixel memory after overlay is built.
                    if hasattr(frames[k], "release_image") and _may_release(k):
                        frames[k].release_image()
                    with state_lock:
                        latest_state[0] = ViewerState(
                            trajectory, map_points_display, keyframe_indices,
                            tracking_state, overlay_image,
                        )
                    terminal_stdout.write(f"\rframe pose {k}/{n_frames} updated")
                    terminal_stdout.flush()
                    if tracker_delay_ms > 0:
                        time.sleep(tracker_delay_ms / 1000.0)
                run_time_bounds[1] = time.perf_counter()
                tracker_finished[0] = True

            tracker_thread = threading.Thread(target=run_tracker) #run the tracker in the background.
            tracker_thread.start()  #start the tracker thread.

            viewer_interval = 1.0 / max(1.0, float(viewer_fps))
            ''' while the tracker is not finished, update the viewer.'''
            while not tracker_finished[0]:
                ''' if the user quits the viewer, set the stop_requested flag to True.'''
                if is_quit_requested():
                    _stop_requested = True
                with state_lock: #get the latest state of the viewer.
                    state = latest_state[0]
                if state is not None: #if the state is not None, update the viewer.
                    if not update_visualizer_from_state(state):
                        print()
                        print("User quit (viewer closed).")
                        break
                time.sleep(viewer_interval)

            tracker_thread.join(timeout=120.0)
            last_idx = last_idx_holder[0]
        else:
            # Headless: single-threaded pose loop only.
            n_frames = int(n_frames)
            run_time_bounds[0] = time.perf_counter()
            for k in range(1, n_frames):
                if _stop_requested:
                    print()
                    print(f"Pipeline stopped at frame {k} (terminal quit).")
                    break
                if frame_iter is not None and len(frames) <= k:
                    nxt = next(frame_iter, None)
                    if nxt is None:
                        break
                    frames.append(nxt)
                _run_one_frame(k)
                # In headless mode we never need pixels after feature extraction.
                if hasattr(frames[k], "release_image") and _may_release(k):
                    frames[k].release_image()
                last_idx = k
                terminal_stdout.write(f"\rframe pose {k}/{n_frames} updated")
                terminal_stdout.flush()
            run_time_bounds[1] = time.perf_counter()
        terminal_stdout.write("\n")  # newline after progress line so next output is on its own line
        terminal_stdout.flush()

        if t_norm_file is not None:
            t_norm_file.close()
            print(f"  >> Wrote t-norm log to {t_norm_log_path}")

        frames_estimated = frames[: last_idx + 1]

        # Stage 04i (spec E2): the end of the recording is the end of observability for the
        # depth-filter arm's live seeds (a no-op under the age lifetime); it must never abort.
        if arm is not None:
            try:
                arm.on_end(last_idx, loop_closer.keyframes, world_map, frames)
            except Exception as e:
                print(f"  >> [DF-ERROR] end: {e}")
        # Stage 04r (spec F): the rotation-only mode's summary, after the last frame.
        if rot_tracker is not None:
            print(rot_tracker.summary_line())
        # Stage 04u (spec C3): the opening driver's summary, in the same place.
        if open_tracker is not None:
            print(open_tracker.summary_line())
        # Stage 04-m0 (spec F7): the integrated system's summary, after the opening's.
        if m0_on:
            try:
                print(f"  >> [M0-SUMMARY] frames={int(last_idx) + 1} mapper_frames={m0_counts['mapper_frames']} "
                      f"keyframes_added={m0_counts['keyframes_added']} handoffs={m0_counts['handoffs']} "
                      f"resets={m0_counts['resets']} points={len(world_map.points)} "
                      f"init_points={sum(1 for pid in m0_init_ids if pid in world_map.point_by_id)}")
            except Exception as e:
                print(f"  >> [M0-ERROR] frame={int(last_idx)} error={type(e).__name__}: {e}")
        # Stage 04v (spec C2): the rotation-only state's summary (MapTracker has no summary_line).
        if m1_tracker is not None and hasattr(m1_tracker, "summary_line"):
            print(m1_tracker.summary_line())

        # --- Stage 3 instrumentation: step lengths + map metrics ---
        def _write_step_lengths(est_frames, out_path):
            if out_path is None:
                return
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("frame_idx,step\n")
                for i in range(1, len(est_frames)):
                    c0 = est_frames[i - 1].world_T_cam[:3, 3]
                    c1 = est_frames[i].world_T_cam[:3, 3]
                    step = float(np.linalg.norm(c1 - c0))
                    f.write(f"{i},{step:.8f}\n")
            print(f"  >> Wrote step lengths to {path}")

        def _write_map_metrics(out_path):
            if out_path is None:
                return
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            obs_stats = compute_map_observation_stats(world_map)
            dup_pairs, n_pts = count_duplicate_neighbours(world_map, radius=0.05)
            try:
                cur_fid = int(getattr(frames_estimated[-1], "id", last_idx))
            except Exception:
                cur_fid = last_idx
            kf_stats = sample_keyframe_link_counts(
                loop_closer.keyframes, world_map,
                min_age_frames=200, current_fid=cur_fid,
            )
            dup_density = (dup_pairs / n_pts) if n_pts > 0 else 0.0
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Stage 3 map metrics\n")
                f.write(f"n_points={obs_stats['n_points']}\n")
                f.write(f"median_obs_per_point={obs_stats['median_obs']:.6f}\n")
                f.write(f"mean_obs_per_point={obs_stats['mean_obs']:.6f}\n")
                f.write(f"frac_obs_ge3={obs_stats['frac_obs_ge3']:.6f}\n")
                f.write(f"duplicate_pairs_radius_0.05={dup_pairs}\n")
                f.write(f"duplicate_density_pairs_per_point={dup_density:.6f}\n")
                f.write(f"fusion_merges_total={stats.get('fusion_merges', 0)}\n")
                f.write(f"fusion_runs={stats.get('fusion_runs', 0)}\n")
                # Format-stable line: the KF-protected prune skip counter was retired in 04-1a.
                f.write("prune_skipped_kf_protected=0\n")
                f.write(f"lm_keyframes={stats.get('lm_keyframes', 0)}\n")
                f.write(f"lm_created={stats.get('lm_created', 0)}\n")
                f.write(f"lm_obs_inserted={stats.get('lm_obs_inserted', 0)}\n")
                f.write(f"lm_culled={stats.get('lm_culled', 0)}\n")
                f.write(f"old_keyframes_age_ge200={kf_stats['n_old_keyframes']}\n")
                f.write(f"old_kf_median_links={kf_stats['median_links']:.6f}\n")
                f.write(f"old_kf_mean_links={kf_stats['mean_links']:.6f}\n")
                f.write(f"old_kf_min_links={kf_stats['min_links']}\n")
                f.write(f"old_kf_frac_links_ge15={kf_stats['frac_links_ge15']:.6f}\n")
                f.write(f"old_kf_frac_links_ge10={kf_stats['frac_links_ge10']:.6f}\n")
                reasons = stats.get("pnp_fail_reasons", {}) or {}
                f.write(f"pnp_too_few_well_established={reasons.get('too_few_well_established', 0)}\n")
            print(f"  >> Wrote map metrics to {path}")
            print("--- Map metrics (Stage 3) ---")
            print(f"  Points: {obs_stats['n_points']}")
            print(
                f"  Obs/point: median={obs_stats['median_obs']:.3f} "
                f"mean={obs_stats['mean_obs']:.3f} "
                f"frac>=3={obs_stats['frac_obs_ge3']:.3%}"
            )
            print(
                f"  Duplicate pairs (r=0.05): {dup_pairs} "
                f"(density={dup_density:.4f} pairs/point)"
            )
            print(
                f"  Fusion merges: {stats.get('fusion_merges', 0)} "
                f"over {stats.get('fusion_runs', 0)} runs"
            )
            print(
                f"  Old KF links (age>=200): n={kf_stats['n_old_keyframes']} "
                f"median={kf_stats['median_links']:.1f} "
                f"min={kf_stats['min_links']} "
                f"frac>=15={kf_stats['frac_links_ge15']:.3%} "
                f"frac>=10={kf_stats['frac_links_ge10']:.3%}"
            )
            print(
                f"  Local mapping: keyframes={stats.get('lm_keyframes', 0)} "
                f"created={stats.get('lm_created', 0)} "
                f"obs_inserted={stats.get('lm_obs_inserted', 0)} "
                f"culled={stats.get('lm_culled', 0)}"
            )

        # Auto paths next to estimate if not set explicitly.
        auto_step = step_lengths_path
        auto_map = map_metrics_path
        if save_estimated_path:
            est_p = Path(save_estimated_path)
            if auto_step is None:
                auto_step = est_p.with_name("step_lengths.csv")
            if auto_map is None:
                auto_map = est_p.with_name("map_metrics.txt")

        _write_step_lengths(frames_estimated, auto_step)
        _write_map_metrics(auto_map)

        if save_estimated_path:
            frame_index_path = _write_estimated_trajectory_tum(frames_estimated, timestamps, save_estimated_path)
            print(f"  >> Saved estimated trajectory ({len(frames_estimated)} poses) to {save_estimated_path}")
            print(f"  >> Saved timestamp->frame index map to {frame_index_path} (for evo_ape: look up time on x-axis to get image frame)")
            # Stage 03a (spec A3): the ONLINE stream, one row per row of estimated.txt, from the
            # per-frame snapshots taken at the end of _run_one_frame (frames without one fall
            # back to the final pose and are counted; must be 0 on the reference run).
            online_path = est_p.with_name(est_p.stem + "_online" + est_p.suffix)
            n_missing_online = sum(
                1 for f in frames_estimated if getattr(f, "online_world_T_cam", None) is None
            )
            _write_estimated_trajectory_tum(
                frames_estimated, timestamps, online_path,
                pose_of=lambda f: (
                    f.online_world_T_cam
                    if getattr(f, "online_world_T_cam", None) is not None else f.pose
                ),
            )
            print(
                f"  >> Saved online trajectory ({len(frames_estimated)} poses, "
                f"{n_missing_online} without online snapshot) to {online_path}"
            )
            # evo_ape plot x-axis = relative GT time (0 to ~T). Write mapping so user can infer frame from plot.
            if ground_truth_path is not None and timestamps is not None:
                evo_frame_path = _write_evo_frame_index_file(
                    frames_estimated, timestamps, ground_truth_path, save_estimated_path
                )
                if evo_frame_path is not None:
                    print(f"  >> Saved evo_ape x-axis -> frame index to {evo_frame_path} (run: python scripts/lookup_evo_frame.py <that file> <x-value>)")

            # Stage 03d (spec A2): export the final map (after every correction) next
            # to the trajectory. Never aborts the run; a failure here changes nothing
            # upstream (estimated.txt / estimated_online.txt are already written).
            try:
                write_map_points(
                    est_p.parent / "map_points.txt", world_map, frames_estimated,
                    [idx for idx, _ in loop_closer.keyframes],
                )
            except Exception as e:
                print(f"  >> Map export failed: {e}")

        print("")
        print("--- Pipeline summary ---")
        print(f"  Homography (rotation-only) frames: {stats['homography']}")
        print(f"  Essential matrix frames:          {stats['essential']}")
        vo_attempts = stats.get("vo_attempts", 0)
        vo_fail = stats.get("vo_fail", 0)
        vo_ratio = (vo_fail / vo_attempts) if vo_attempts > 0 else 0.0
        print(f"  VO: fail count={vo_fail}, attempts={vo_attempts}, fail ratio={vo_ratio:.2%}")
        pnp_total = stats["pnp_success"] + stats["pnp_fail"]
        pnp_ratio = (stats["pnp_success"] / pnp_total) if pnp_total > 0 else 0.0
        print(f"  PnP localization: success={stats['pnp_success']}, fail={stats['pnp_fail']}, success/overall={pnp_ratio:.2%}")
        reasons = stats.get("pnp_fail_reasons", {})
        if reasons and any(reasons.get(k, 0) > 0 for k in ("too_few_correspondences", "too_few_well_established", "ransac_failed", "inliers_below_min", "ratio_below_threshold", "multi_all_failed", "safety_gate", "unknown_solver")):
            print("  PnP fail reasons (count):")
            for key in ("too_few_correspondences", "too_few_well_established", "ransac_failed", "inliers_below_min", "ratio_below_threshold", "multi_all_failed", "safety_gate", "unknown_solver"):
                n = reasons.get(key, 0)
                if n > 0:
                    label = key.replace("_", " ")
                    cond = ""
                    if key == "too_few_correspondences" and "pnp_min_correspondences" in stats:
                        cond = f" min_correspondences={stats['pnp_min_correspondences']}"
                    elif key == "too_few_well_established" and "pnp_min_observations" in stats:
                        cond = f" min_observations={stats['pnp_min_observations']}"
                    elif key == "inliers_below_min" and "pnp_min_correspondences" in stats:
                        cond = f" min_correspondences={stats['pnp_min_correspondences']}"
                    elif key == "ratio_below_threshold" and "pnp_min_inlier_ratio" in stats:
                        cond = f" min_inlier_ratio={stats['pnp_min_inlier_ratio']}"
                    elif key == "safety_gate" and "pnp_safety_gate_max_step" in stats and "pnp_safety_gate_max_correction" in stats:
                        cond = f" max_step={stats['pnp_safety_gate_max_step']:.2f}, max_correction={stats['pnp_safety_gate_max_correction']:.2f}"
                    print(f"    {label} ({cond.strip()}): {n}" if cond else f"    {label}: {n}")
        for line in ba_summary_lines(stats):
            print(line)
        if oracle is not None:
            print(oracle.summary_line())  # Stage 03c (spec C3)
        print(f"  Loop closure detections:         {stats.get('loop_closures', 0)}")
        print(f"  Relocalizations (after VO fail): {stats.get('relocalizations', 0)}")
        if stats.get("fusion_runs", 0) > 0 or stats.get("fusion_merges", 0) > 0:
            print(
                f"  Point fusion: merges={stats.get('fusion_merges', 0)} "
                f"runs={stats.get('fusion_runs', 0)}"
            )
        # Stage 04-1a (spec A7(b)): the local mapper's cumulative counters.
        print(
            f"  Local mapping: keyframes={stats.get('lm_keyframes', 0)} "
            f"created={stats.get('lm_created', 0)} "
            f"obs_inserted={stats.get('lm_obs_inserted', 0)} "
            f"culled={stats.get('lm_culled', 0)}"
        )
        if stats.get("relocalization_failures", 0) > 0:
            print(f"  Relocalization failures:         {stats['relocalization_failures']}")
        attempts = stats.get("relocalization_attempts", [])
        if attempts:
            print("  Relocalization attempts (frame -> success/fail, matched keyframe or reason):")
            for entry in attempts:
                fk, ok, kf_idx, n_inl = entry[0], entry[1], entry[2], entry[3]
                fail_reason = entry[4] if len(entry) > 4 else "no keyframe match"
                if ok:
                    print(f"    frame {fk}: success (matched keyframe {kf_idx}, {n_inl} inliers)")
                else:
                    print(f"    frame {fk}: failed ({fail_reason})")
        pose_holds = stats.get("pose_holds", 0)
        if pose_holds > 0:
            print(f"  Pose holds (VO+reloc fail, non-identity): {pose_holds}")
            for fk, policy in stats.get("pose_hold_events", []):
                print(f"    frame {fk}: {policy}")
        scale_reasons = stats.get("scale_fallback_reasons") or {}
        if scale_reasons:
            print("  Scale estimator outcomes (count by reason):")
            for reason, n in sorted(scale_reasons.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {reason}: {n}")
        if keyframe_strategy in ("hybrid", "ratio"):
            print(loop_closer.get_keyframe_reason_summary())
        wins = stats.get("pnp_solver_wins", {})
        if any(wins.get(n, 0) > 0 for n in ("iterative", "ap3p", "epnp", "sqpnp")):
            print("  PnP solver wins (count per method):")
            for name in ("iterative", "ap3p", "epnp", "sqpnp"):
                if wins.get(name, 0) > 0:
                    print(f"    {name}: {wins[name]}")
        _print_timing_summary(stats, run_time_bounds)

        if show_realtime:
            finish_visualizer(world_map=world_map)
    finally:
        if log_file_path is not None and _stdout_orig is not None:
            sys.stdout = _stdout_orig
            if log_file is not None:
                log_file.close()
            print(f"Pipeline log written to {log_file_path}")
            print("--- Summary ---")
            print(f"  Homography: {stats['homography']}  Essential: {stats['essential']}")
            vo_attempts = stats.get("vo_attempts", 0)
            vo_fail = stats.get("vo_fail", 0)
            vo_ratio = (vo_fail / vo_attempts) if vo_attempts > 0 else 0.0
            print(f"  VO fail: {vo_fail} / {vo_attempts} attempts, ratio: {vo_ratio:.2%}")
            pnp_total = stats["pnp_success"] + stats["pnp_fail"]
            pnp_ratio = (stats["pnp_success"] / pnp_total) if pnp_total > 0 else 0.0
            print(f"  PnP success: {stats['pnp_success']}  PnP fail: {stats['pnp_fail']}  success/overall: {pnp_ratio:.2%}")
            for line in ba_summary_lines(stats):
                print(line)
            if oracle is not None:
                print(oracle.summary_line())  # Stage 03c (spec C3)
            reasons = stats.get("pnp_fail_reasons", {})
            if reasons and any(reasons.get(k, 0) > 0 for k in ("too_few_correspondences", "too_few_well_established", "ransac_failed", "inliers_below_min", "ratio_below_threshold", "multi_all_failed", "safety_gate", "unknown_solver")):
                print("  PnP fail reasons (count):")
                for key in ("too_few_correspondences", "too_few_well_established", "ransac_failed", "inliers_below_min", "ratio_below_threshold", "multi_all_failed", "safety_gate", "unknown_solver"):
                    n = reasons.get(key, 0)
                    if n > 0:
                        label = key.replace("_", " ")
                        cond = ""
                        if key == "too_few_correspondences" and "pnp_min_correspondences" in stats:
                            cond = f" min_correspondences={stats['pnp_min_correspondences']}"
                        elif key == "too_few_well_established" and "pnp_min_observations" in stats:
                            cond = f" min_observations={stats['pnp_min_observations']}"
                        elif key == "inliers_below_min" and "pnp_min_correspondences" in stats:
                            cond = f" min_correspondences={stats['pnp_min_correspondences']}"
                        elif key == "ratio_below_threshold" and "pnp_min_inlier_ratio" in stats:
                            cond = f" min_inlier_ratio={stats['pnp_min_inlier_ratio']}"
                        elif key == "safety_gate" and "pnp_safety_gate_max_step" in stats and "pnp_safety_gate_max_correction" in stats:
                            cond = f" max_step={stats['pnp_safety_gate_max_step']:.2f}, max_correction={stats['pnp_safety_gate_max_correction']:.2f}"
                        print(f"    {label} ({cond.strip()}): {n}" if cond else f"    {label}: {n}")
            print(f"  Loop closures: {stats.get('loop_closures', 0)}")
            print(f"  Relocalizations: {stats.get('relocalizations', 0)}")
            if stats.get("relocalization_failures", 0) > 0:
                print(f"  Relocalization failures: {stats['relocalization_failures']}")
            if stats.get("pose_holds", 0) > 0:
                print(f"  Pose holds (non-identity): {stats['pose_holds']}")
            scale_reasons = stats.get("scale_fallback_reasons") or {}
            if scale_reasons:
                n_ok = sum(v for k, v in scale_reasons.items() if str(k).startswith("ok"))
                n_fb = sum(v for k, v in scale_reasons.items() if not str(k).startswith("ok"))
                print(f"  Scale outcomes: ok={n_ok}, fallback={n_fb}")
            attempts = stats.get("relocalization_attempts", [])
            if attempts:
                print("  Relocalization attempts (frame -> success/fail, matched keyframe or reason):")
                for entry in attempts:
                    fk, ok, kf_idx, n_inl = entry[0], entry[1], entry[2], entry[3]
                    fail_reason = entry[4] if len(entry) > 4 else "no keyframe match"
                    if ok:
                        print(f"    frame {fk}: success (matched keyframe {kf_idx}, {n_inl} inliers)")
                    else:
                        print(f"    frame {fk}: failed ({fail_reason})")
            if keyframe_strategy in ("hybrid", "ratio"):
                print(loop_closer.get_keyframe_reason_summary())
            wins = stats.get("pnp_solver_wins", {})
            if any(wins.get(n, 0) > 0 for n in ("iterative", "ap3p", "epnp", "sqpnp")):
                print("  PnP solver wins:", wins)
            _print_timing_summary(stats, run_time_bounds)
