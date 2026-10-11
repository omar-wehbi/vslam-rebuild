"""
Stage 03a instrumentation helpers shared by pipeline.py and frame.py (spec B and C).

Kept in a tiny module of its own so frame.py can print [CORRECTION] lines without importing
pipeline.py (which pulls in the viewer); pipeline.py re-exports these names at module level.
Nothing here changes a pose: these are pure functions of poses already set.
"""
import numpy as np

POSE_ONLINE_SOURCES = (
    "init", "vo_essential", "vo_homography", "pnp", "reloc",
    "hold_prev", "constant_velocity", "hold_prev_fallback",
)
BA_OUTCOMES = ("none", "skipped", "accepted", "rejected", "error")
CORRECTION_KINDS = ("pnp_distributed", "ba", "loop_closure")


def pose_delta_stats(T_a, T_b):
    """
    (rot_deg, trans) between two 4x4 poses of the same kind (both world_T_cam or both cam_T_world):
    rotation angle of R_a^T R_b in degrees via the trace (clipped to [-1, 1]) and the Euclidean
    distance between the translation columns.
    """
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    c = (float(np.trace(R_rel)) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    trans = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
    return rot_deg, trans


def format_pose_online_line(k, source, state, rot_deg, step, pnp, ba, lc):
    """The per-frame [POSE-ONLINE] provenance line (spec B)."""
    return (
        f"  >> [POSE-ONLINE] frame={int(k)} source={source} state={state} "
        f"rel_rot_deg={float(rot_deg):.6f} step={float(step):.6f} "
        f"pnp={1 if pnp else 0} ba={ba} lc={1 if lc else 0}"
    )


def format_correction_line(k, kind, n, first, last, max_rot_deg, max_trans):
    """The [CORRECTION] event line (spec C), printed after a rewrite of already-emitted frames."""
    return (
        f"  >> [CORRECTION] frame={int(k)} kind={kind} frames_touched={int(n)} "
        f"first={int(first)} last={int(last)} "
        f"max_rot_deg={float(max_rot_deg):.6f} max_trans={float(max_trans):.6f}"
    )


def correction_stats(before, frames_after, exclude_id=None, tol=1e-12):
    """
    Compare poses before and after an operation (spec C).

    before: dict Frame.id -> 4x4 pose snapshot; frames_after: iterable of (Frame, pose after the
    operation), the pose being the same kind of matrix as the snapshot (both world_T_cam or both
    cam_T_world). Frames whose pose changed (any element by more than tol), excluding exclude_id
    (the current frame), are counted.
    Returns (n, first_id, last_id, max_rot_deg, max_trans) with first=last=-1 and zeros if none.
    """
    n = 0
    first = last = -1
    max_rot = max_trans = 0.0
    for f, after in frames_after:
        fid = f.id
        if fid == exclude_id or fid not in before:
            continue
        old = before[fid]
        if not np.any(np.abs(np.asarray(after, dtype=np.float64) - old) > tol):
            continue
        n += 1
        first = fid if first < 0 else min(first, fid)
        last = max(last, fid)
        rot, trans = pose_delta_stats(old, after)
        max_rot = max(max_rot, rot)
        max_trans = max(max_trans, trans)
    return n, first, last, max_rot, max_trans


def print_correction(k, kind, before, frames_after, exclude_id=None):
    """Compute correction_stats and print the [CORRECTION] line; never raises."""
    try:
        n, first, last, max_rot, max_trans = correction_stats(before, frames_after, exclude_id=exclude_id)
    except Exception:
        n, first, last, max_rot, max_trans = 0, -1, -1, 0.0, 0.0
    print(format_correction_line(k, kind, n, first, last, max_rot, max_trans))
