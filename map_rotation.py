"""
Stage 04v (milestone 04-3b; rulings B1-B5 and B7 of workshop/45, B6 and T2 of workshop/47, human
2026-10-02): the rotation-only state inside a map, behind --map_rot_state on.

When the map tracker's pose fails at frame k after a tracked frame k-1 while the 04r rotation-only
fit succeeds (design.md slot "No-parallax states"; papers/pirchheim-hybrid-rotation-slam.md
#the-3-dof-state-and-the-transitions), a stint starts: frame k-1 becomes the panorama keyframe with
its 6-DoF pose (Pirchheim's localized panorama), each frame's orientation comes from the fit against
the segment's current panorama keyframe, and the position is bridged by the tracker's constant body
twist (map_tracker.predict_pose iterated on its own predictions, ruling B6). A frame of a stint is
labelled bridged for 30 frames after the anchor, then rotation (translation unsupported, ruling
T2), and lost when its fit is unsupported (conventions.md#observability-under-pure-rotation).

The state ends only through the tracker's existing paths (ruling B2): normal tracking from the
bridged prediction, then the 04m EPnP relocalization, both accepted at MapTrackerParams.min_inliers
(30). No new acceptance test, no mixed cost (B7), no platform prior.

The state creates no point, observation, keyframe or seed: links are made only by an accepted
tracking or relocalization, exactly as in MapTracker.

Stage 04x (rulings R52-1 to R52-3, human 2026-10-03; design.md 04-3 row, amendment R52): under
--map_rot_state r52 (self.variant == "r52", set by main.run_pipeline; the constructor default "on" keeps
the 04v behaviour) the state is entered only when the tracked history of the last 2.0 s shows rotation
without parallax (papers/pirchheim-hybrid-rotation-slam.md#detecting-rotation-only-motion) and the
fit succeeds; in the state the position is held at the anchor and measured frames are labelled
rotation (no bridged label, no horizon); a panorama keyframe switch also needs r > 0.2
(#keyframe-selection-6-dof-localized-panorama-panorama); lost frames outside the state carry the
tracker's prediction with its translation from the windowed body velocity (R50-2, R50-3).

Convention (conventions.md#pose-convention): cam_T_world maps world to camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import map_tracker
import rotation_only

# Stage 04-r04 (ruling R65-4): the evaluation's recording-gap convention (jumps.py GAP_EXCLUDE_S): an rgb
# interval over 150 ms is a recording pause, and the frame after it is a post-gap frame.
POST_GAP_S = 0.150


@dataclass
class RotStateParams:
    """Constants of the rotation-only state inside a map. No CLI flag sets them.

    horizon: ruling T2 (workshop/47 addendum), the label horizon in frames derived from the
    workshop/47 pre-test: a measured frame of a stint is "bridged" while its age (frames since the
    anchor, the last tracked frame) is at most horizon, "rotation" after.
    """
    horizon: int = 30


@dataclass
class FallbackParams:
    """Constants of the 04x fallback structure (--map_rot_state r52). No CLI flag sets them; they come
    from the paper and the rulings, not from this data.

    hist_s: R52-2, the history window in seconds (the tracked poses of the last 2.0 s before the anchor).
    alpha_max_deg: Pirchheim 4.2, the parallax bound (a history pose is kept when alpha < 5 deg).
    r_min: Pirchheim 4.2 and 4.4, the rotation ratio beta / gamma (detection needs a kept pose with
    r > 0.2; a panorama keyframe switch needs r > 0.2).
    window_s: R50-2, the body-velocity fit window in seconds.
    min_frames: R50-2, fewer frames in the window give zero velocity (the centre is held).
    gap_s: R50-3, the fit window crosses no rgb interval over 50 ms.
    """
    hist_s: float = 2.0
    alpha_max_deg: float = 5.0
    r_min: float = 0.2
    window_s: float = 1.0
    min_frames: int = 10
    gap_s: float = 0.050


def fov_h_deg(K, width):
    """Spec C3: the horizontal field of view in degrees, 2 atan(width / (2 fx))."""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * float(K[0][0]))))


def history_frames(e, is_tracked, timestamps, hist_s):
    """Spec C3 (R52-2): the ascending frames j < e with is_tracked(j) and t_e - t_j <= hist_s (1e-9).
    Scans down from e - 1 and stops at the first frame beyond hist_s; an untracked frame inside the
    window is skipped. The anchor e is not in the list."""
    out = []
    t_e = float(timestamps[e])
    j = int(e) - 1
    while j >= 0:
        if t_e - float(timestamps[j]) > float(hist_s) + 1e-9:
            break
        if is_tracked(j):
            out.append(j)
        j -= 1
    out.reverse()
    return out


def detect_rotation(W_e, W_hist, f, gamma_deg, params):
    """Spec C3 (Pirchheim 4.2): rotation without parallax on the history. W_e and each W_j are
    world_T_cam. Per W_j: alpha_j = 2 atan(|c_j - c_e| / (2 f)), beta_j the angle between the optical
    axes, r_j = beta_j / gamma. Returns (detected, n_kept, r_max, alpha_deg): kept poses have
    alpha_j < alpha_max_deg; r_max is the largest r_j over them and alpha_deg that pose's alpha_j
    (both -1.0 when none is kept); detected iff some kept pose has r_j > r_min."""
    try:
        f = float(f)
    except (TypeError, ValueError):
        return False, 0, -1.0, -1.0
    if not math.isfinite(f) or f <= 0.0:
        return False, 0, -1.0, -1.0
    W_e = np.asarray(W_e, dtype=np.float64)
    c_e = W_e[:3, 3]
    z_e = W_e[:3, 2]
    n_kept = 0
    r_max = -1.0
    alpha_at = -1.0
    detected = False
    for W in W_hist:
        W = np.asarray(W, dtype=np.float64)
        d = float(np.linalg.norm(W[:3, 3] - c_e))
        alpha = math.degrees(2.0 * math.atan(d / (2.0 * f)))
        beta = math.degrees(float(np.arccos(np.clip(float(np.dot(W[:3, 2], z_e)), -1.0, 1.0))))
        r = beta / float(gamma_deg)
        if alpha < float(params.alpha_max_deg):
            n_kept += 1
            if n_kept == 1 or r > r_max:
                r_max = r
                alpha_at = alpha
            if r > float(params.r_min):
                detected = True
    return bool(detected), int(n_kept), float(r_max), float(alpha_at)


def bridge_window(e, is_tracked, timestamps, params):
    """Spec C3 (R50-2, R50-3): the ascending frames [j0, ..., e] of the contiguous tracked run ending
    at e; [] when e is not tracked. Frame j = first - 1 joins while j >= 0, is_tracked(j),
    t_e - t_j <= window_s (1e-9) and t_(j+1) - t_j <= gap_s; the first failure ends the window."""
    if not is_tracked(e):
        return []
    out = [int(e)]
    t_e = float(timestamps[e])
    j = int(e) - 1
    while (j >= 0 and is_tracked(j) and t_e - float(timestamps[j]) <= float(params.window_s) + 1e-9
           and float(timestamps[j + 1]) - float(timestamps[j]) <= float(params.gap_s)):
        out.append(j)
        j -= 1
    out.reverse()
    return out


def fit_body_velocity(world_T_cams, times, min_frames):
    """Spec C3 (R50-2, the workshop/50 pre-test's fit without its F test): the body velocity s in the
    camera frame (map units per second) of c_j = c_0 + A_j s, A_0 = 0, A_i = A_(i-1) + R_i (t_i - t_(i-1)),
    R_i and c_i the rotation and centre of world_T_cams[i], by least squares over (c_0, s). Returns
    (s, fitted); (zeros, False) with fewer than min_frames poses."""
    n = len(world_T_cams)
    if n < int(min_frames):
        return np.zeros(3), False
    A = np.zeros((3, 3), dtype=np.float64)
    X = np.zeros((3 * n, 6), dtype=np.float64)
    y = np.zeros(3 * n, dtype=np.float64)
    for i in range(n):
        W = np.asarray(world_T_cams[i], dtype=np.float64)
        if i > 0:
            A = A + W[:3, :3] * (float(times[i]) - float(times[i - 1]))
        X[3 * i:3 * i + 3, :3] = np.eye(3)
        X[3 * i:3 * i + 3, 3:] = A
        y[3 * i:3 * i + 3] = W[:3, 3]
    sol = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.array(sol[3:6], dtype=np.float64), True


class PanoramaSegment(rotation_only.RotationOnlyTracker):
    """Spec C3: the 04r rotation-only segment with Pirchheim 4.4's switch rule: the coverage test
    (< 0.75, the parent's, unchanged) calls _new_keyframe, and a switch also needs r > r_min, r the
    angle between the optical axes of the frame and the keyframe over the horizontal field of view.
    The anchor keyframe is always taken."""

    def __init__(self, r_min, params=None):
        super().__init__(params)
        self.r_min = float(r_min)
        self._frames = None
        self.n_holds = 0

    def on_frame(self, k, frames):
        self._frames = frames
        return super().on_frame(k, frames)

    def _new_keyframe(self, k):
        if self.kf is None or self._frames is None:
            return super()._new_keyframe(k)
        cur = self._frames[k]
        z_k = np.asarray(cur.world_T_cam, dtype=np.float64)[:3, 2]
        z_kf = np.asarray(self._frames[self.kf].world_T_cam, dtype=np.float64)[:3, 2]
        beta = math.degrees(float(np.arccos(np.clip(float(np.dot(z_k, z_kf)), -1.0, 1.0))))
        r = beta / fov_h_deg(cur.K, cur.img_shape[1])
        if r > self.r_min:
            return super()._new_keyframe(k)
        self.n_holds += 1
        print(f"  >> [MAP-ROT-KF-HOLD] frame={int(k)} kf={int(self.kf)} r={r:.6f}")
        return None


def _fmt9(v):
    return ",".join(f"{float(x):.9f}" for x in np.asarray(v, dtype=np.float64).reshape(-1))


class MapRotationTracker(map_tracker.MapTracker):
    """The map tracker with the rotation-only state of stage 04v (spec D)."""

    def __init__(self, m1map, mode, params=None):
        super().__init__(m1map, mode, params)
        self.rot_params = RotStateParams()
        # the stint (spec D2): anchor e, its rotation-only segment, and whether the entry twist is unknown
        self._anchor = None
        self._seg = None
        self._entry_prev2_none = False
        self._world_map = None
        # counters of the summary line (spec D6)
        self.n_stints = 0
        self.n_refused = 0
        self.n_exits_track = 0
        self.n_exits_reloc = 0
        self.n_bridged = 0
        self.n_rotation = 0
        self.n_lost_in_state = 0
        self.n_tracked = 0
        self.n_lost = 0
        # stage 04x (spec C2): the variant ("on" = 04v; main.run_pipeline sets "r52"), the per-frame
        # timestamps, the constants, the anchor's linked points (C4), the lost-run velocity (C6)
        self.variant = "on"
        self.timestamps = None
        self.fb_params = FallbackParams()
        self._inl_k = None
        self._inl_X = None
        self._lost_vel = None
        # counters of the [MAP-ROT-R52-SUMMARY] line (spec C8)
        self.n_attempts = 0
        self.n_detected = 0
        self.n_refused_detect = 0
        self.n_refused_fit = 0
        self.n_entered = 0
        self.n_lost_runs = 0
        self.n_lost_runs_fitted = 0
        self._segments = []
        # stage 04-r04 (spec B2, R65-4): the post-gap label rule ("carry" = unchanged; main.run_pipeline sets
        # it from --map_rot_gap) and its counter
        self.gap_rule = "carry"
        self.n_gap_relabelled = 0

    # ---------------------------------------------------------------- setup / summary
    def setup_line(self):
        lines = (map_tracker.MapTracker.setup_line(self) + "\n"
                 + f"  >> [MAP-ROT-SETUP] horizon={int(self.rot_params.horizon)}")
        if self.variant == "r52":
            p = self.fb_params
            lines += ("\n" + f"  >> [MAP-ROT-R52-SETUP] hist_s={float(p.hist_s):.6f} "
                      f"alpha_max_deg={float(p.alpha_max_deg):.6f} r_min={float(p.r_min):.6f} "
                      f"window_s={float(p.window_s):.6f} min_frames={int(p.min_frames)} gap_s={float(p.gap_s):.6f}")
        if getattr(self, "gap_rule", "carry") == "lost":
            lines += "\n" + f"  >> [MAP-ROT-GAP-SETUP] rule=lost post_gap_s={float(POST_GAP_S):.6f}"
        return lines

    def summary_line(self):
        n_points = -1 if self._world_map is None else len(self._world_map.points)
        line = (f"  >> [MAP-ROT-SUMMARY] stints={int(self.n_stints)} refused={int(self.n_refused)} "
                f"exits_track={int(self.n_exits_track)} exits_reloc={int(self.n_exits_reloc)} "
                f"bridged={int(self.n_bridged)} rotation={int(self.n_rotation)} "
                f"lost_in_state={int(self.n_lost_in_state)} tracked={int(self.n_tracked)} "
                f"lost={int(self.n_lost)} map_points={int(n_points)}")
        if self.variant == "r52":
            holds = sum(int(seg.n_holds) for seg in self._segments)
            line += ("\n" + f"  >> [MAP-ROT-R52-SUMMARY] attempts={int(self.n_attempts)} "
                     f"detected={int(self.n_detected)} refused_detect={int(self.n_refused_detect)} "
                     f"refused_fit={int(self.n_refused_fit)} entered={int(self.n_entered)} kf_holds={holds} "
                     f"lost_runs={int(self.n_lost_runs)} lost_runs_fitted={int(self.n_lost_runs_fitted)}")
        if getattr(self, "gap_rule", "carry") == "lost":
            line += "\n" + f"  >> [MAP-ROT-GAP-SUMMARY] relabelled={int(self.n_gap_relabelled)}"
        return line

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _is_tracked(frames, j):
        """Spec D: frame j is tracked iff its tracking_state is OK (frame 0 is OK by construction)."""
        return j == 0 or getattr(frames[j], "tracking_state", None) == "OK"

    def _label(self, age, rot):
        """Spec D4 (ruling T2)."""
        if rot != "measured":
            return "lost"
        return "bridged" if age <= int(self.rot_params.horizon) else "rotation"

    def _prev2(self, k, frames):
        """Spec D3: T_(k-2), or None at the stint's first frame when frame e-1 is not tracked."""
        if k - 1 == self._anchor and self._entry_prev2_none:
            return None
        return np.array(frames[k - 2].cam_T_world, dtype=np.float64, copy=True)

    def _bridged_pose(self, k, frames, r):
        """Spec D3 (ruling B6): T_b = predict_pose(T_(k-1), T_prev2, R_step), R_step = R_fit R_(k-1)^T
        when the fit is measured (R_fit as seg.on_frame has just set it), None otherwise."""
        T1 = np.array(frames[k - 1].cam_T_world, dtype=np.float64, copy=True)
        T2 = self._prev2(k, frames)
        R_step = None
        if r == "measured":
            R_fit = np.asarray(frames[k].cam_T_world, dtype=np.float64)[:3, :3]
            R_step = R_fit @ T1[:3, :3].T
        return map_tracker.predict_pose(T1, T2, R_step)

    @staticmethod
    def _twist(k, frames):
        """The translation block of T_k inv(T_(k-1)) for frame k's final pose (spec D5)."""
        Tk = np.asarray(frames[k].cam_T_world, dtype=np.float64)
        T1 = np.asarray(frames[k - 1].cam_T_world, dtype=np.float64)
        return (Tk @ np.linalg.inv(T1))[:3, 3]

    def _rot_line(self, k, frames, event, label, anchor, kf, rot, exit_, track_inliers):
        return (f"  >> [MAP-ROT] frame={int(k)} event={event} label={label} anchor={int(anchor)} "
                f"age={int(k - anchor)} kf={int(kf)} rot={rot} exit={exit_} "
                f"track_inliers={int(track_inliers)} twist={_fmt9(self._twist(k, frames))}")

    def _count(self, label, in_stint):
        if label == "bridged":
            self.n_bridged += 1
        elif label == "rotation":
            self.n_rotation += 1
        elif label == "tracked":
            self.n_tracked += 1
        else:
            self.n_lost += 1
            if in_stint:
                self.n_lost_in_state += 1
        return label

    def _post_gap(self, k):
        """Spec B3 (R65-4): True iff frame k follows a recording gap, the rgb interval ts[k] - ts[k-1]
        strictly over POST_GAP_S (as jumps.classify_pairs)."""
        ts = self.timestamps
        if ts is None or not (1 <= int(k) < len(ts)):
            return False
        return float(ts[k]) - float(ts[k - 1]) > POST_GAP_S

    def _gap_relabel(self, k, frames, event):
        """Spec B4 (R65-4): under gap_rule "lost", a post-gap frame of the rotation-only state is labelled
        lost (pose_valid False, state LOST) and one [MAP-ROT-GAP] line is printed; returns True. The
        frame's pose, the segment and the anchor are not touched. Otherwise returns False and does nothing."""
        if getattr(self, "gap_rule", "carry") != "lost" or not self._post_gap(k):
            return False
        fr = frames[k]
        fr.pose_valid = False
        fr.tracking_state = "LOST"
        self.n_gap_relabelled += 1
        ts = self.timestamps
        anchor = -1 if self._anchor is None else int(self._anchor)
        kf = -1 if (self._seg is None or self._seg.kf is None) else int(self._seg.kf)
        print(f"  >> [MAP-ROT-GAP] frame={int(k)} event={event} "
              f"gap_s={float(ts[k]) - float(ts[k - 1]):.6f} anchor={anchor} kf={kf}")
        return True

    def _end_stint(self):
        self._anchor = None
        self._seg = None
        self._entry_prev2_none = False

    # ---------------------------------------------------------------- per frame (spec D2)
    def on_frame(self, k, frames, world_map):
        """Track frame k >= 1; returns "tracked", "lost", "bridged" or "rotation"."""
        self._world_map = world_map
        if self.variant == "r52":
            # stage 04x (spec C5 to C7); under "on" the 04v code below runs untouched
            if self._anchor is None:
                return self._outside_r52(k, frames, world_map)
            return self._inside_r52(k, frames, world_map)
        if self._anchor is None:
            return self._outside(k, frames, world_map)
        return self._inside(k, frames, world_map)

    def _outside(self, k, frames, world_map):
        # (1) the verified 04l/04m behaviour and lines, unchanged
        s = map_tracker.MapTracker.on_frame(self, k, frames, world_map)
        if s == "tracked":
            return self._count("tracked", False)
        if s != "lost" or not self._is_tracked(frames, k - 1):
            return self._count("lost", False)                    # (3) no rotation fit is run
        # (2) the entry test (B2): the last tracked frame k-1 is the panorama keyframe
        fr = frames[k]
        T_lost = np.array(fr.cam_T_world, dtype=np.float64, copy=True)
        W_lost = np.array(fr.world_T_cam, dtype=np.float64, copy=True)
        e = k - 1
        try:
            seg = rotation_only.RotationOnlyTracker()
            seg.world_map = world_map
            seg._new_keyframe(e)
            r = seg.on_frame(k, frames)
            if r == "measured":
                self._anchor = e
                self._seg = seg
                self._entry_prev2_none = (e - 1 < 0) or not self._is_tracked(frames, e - 1)
                T_b = self._bridged_pose(k, frames, r)
                if not fr.set_camera_pose(T_b):
                    raise RuntimeError("set_camera_pose rejected the bridged pose")
                fr.pose_valid = True
                fr.tracking_state = "ROT"
                label = self._label(k - e, r)
                self.n_stints += 1
                print(self._rot_line(k, frames, "enter", label, e, seg.kf, r, "none", -1))
                return self._count(label, True)
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")
            self._end_stint()
        # refused: no stint; frame k bitwise as the parent left it
        fr.cam_T_world = T_lost
        fr.world_T_cam = W_lost
        fr.pose_valid = False
        fr.tracking_state = "LOST"
        self.n_refused += 1
        print(self._rot_line(k, frames, "refused", "lost", e, e, "unsupported", "none", -1))
        return self._count("lost", False)

    def _inside(self, k, frames, world_map):
        e = self._anchor
        seg = self._seg
        fr = frames[k]
        r = "unsupported"
        T_b = None
        n_track = -1
        track_printed = False
        try:
            # (4) the rotation fit against the segment's current panorama keyframe (04r unchanged)
            r = seg.on_frame(k, frames)
            # (5) the bridged pose
            T_b = self._bridged_pose(k, frames, r)
            # (6) normal tracking from the bridged prediction
            if self.mode == "m1a":
                res = self._track_m1a(k, frames, T_b)
            else:
                res = self._track_m1b(k, frames, T_b)
            T_fin, rows, kps, mask, counts = res
            n_track = int(np.sum(mask)) if len(mask) else 0
            if T_fin is not None and n_track >= int(self.params.min_inliers):
                rmse = self._rmse(fr, T_fin, rows, kps, mask)
                fr.set_camera_pose(T_fin)
                fr.pose_valid = True
                fr.tracking_state = "OK"
                for row, kp, keep in zip(rows, kps, mask):
                    if keep:
                        world_map.link(fr, int(kp), self._rows_pt[int(row)])
                track_printed = True
                print(self._track_line(k, "tracked", "cv", counts["step_a"], counts["step_a_inliers"],
                                       counts["step_a_wide"], counts["step_b"], len(rows), n_track, rmse))
                self.n_exits_track += 1
                print(self._rot_line(k, frames, "exit", "tracked", e, seg.kf, r, "track", n_track))
                self._end_stint()
                return self._count("tracked", True)
            # (7) relocalization: the parent's 04m path, with frame k-1 seen as LOST for the call
            prev = frames[k - 1]
            prev_state = getattr(prev, "tracking_state", None)
            prev.tracking_state = "LOST"
            track_printed = True                             # the parent prints its own [MAP-TRACK] line
            try:
                s = map_tracker.MapTracker.on_frame(self, k, frames, world_map)
            finally:
                prev.tracking_state = prev_state
            if s == "tracked":
                self.n_exits_reloc += 1
                print(self._rot_line(k, frames, "exit", "tracked", e, seg.kf, r, "reloc", n_track))
                self._end_stint()
                return self._count("tracked", True)
            # (8) stay: the bridged pose, labelled; a failed fit does not end the stint (P4)
            if not fr.set_camera_pose(T_b):
                raise RuntimeError("set_camera_pose rejected the bridged pose")
            if r == "measured":
                fr.pose_valid = True
                fr.tracking_state = "ROT"
            else:
                fr.pose_valid = False
                fr.tracking_state = "LOST"
            label = self._label(k - e, r)
            print(self._rot_line(k, frames, "stay", label, e, seg.kf, r, "none", n_track))
            return self._count(label, True)
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")
            if T_b is None:
                T1 = np.array(frames[k - 1].cam_T_world, dtype=np.float64, copy=True)
                T_b = map_tracker.predict_pose(T1, self._prev2(k, frames), None)
            fr.set_camera_pose(T_b)
            fr.pose_valid = False
            fr.tracking_state = "LOST"
            if not track_printed:
                # one [MAP-TRACK] line per frame, as the parent's own error path prints
                print(self._track_line(k, "lost", "cv", -1 if self.mode == "m1b" else 0,
                                       -1 if self.mode == "m1b" else 0, 0, -1 if self.mode == "m1b" else 0,
                                       0, 0, float("nan")))
            kf = seg.kf if seg is not None and seg.kf is not None else e
            print(self._rot_line(k, frames, "stay", "lost", e, kf, "unsupported", "none", n_track))
            return self._count("lost", True)

    # ---------------------------------------------------------------- stage 04x: --map_rot_state r52
    def _tracked_fn(self, frames):
        return lambda j: self._is_tracked(frames, j)

    def _record_inliers(self, k, frames, world_map):
        """Spec C4: when frame k is about to be returned tracked, keep the map points its accepted pose
        linked (keypoint_to_point values present in world_map.point_by_id, insertion order)."""
        rows = []
        try:
            pbi = getattr(world_map, "point_by_id", None)
            pbi = {} if pbi is None else pbi
            for pid in frames[k].keypoint_to_point.values():
                p = pbi.get(pid)
                if p is not None:
                    rows.append([p.x, p.y, p.z])
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")
            rows = []
        self._inl_k = int(k)
        self._inl_X = np.array(rows, dtype=np.float64).reshape(-1, 3)

    def _anchor_depth(self, e, frames):
        """Spec C4: f(e), the mean depth in the anchor camera of the points its accepted pose linked;
        nan when the anchor was not recorded or linked no point."""
        if self._inl_k != e or self._inl_X is None or len(self._inl_X) == 0:
            return float("nan")
        T = np.asarray(frames[e].cam_T_world, dtype=np.float64)
        z = self._inl_X @ T[2, :3] + T[2, 3]
        return float(np.mean(z))

    @staticmethod
    def _restore(fr, T, W):
        """Bitwise restore by direct assignment (set_camera_pose is not bitwise idempotent)."""
        fr.cam_T_world = T
        fr.world_T_cam = W

    @staticmethod
    def _entry_line(k, n_hist, f, gamma, kept, r_max, alpha, det, fit, result):
        return (f"  >> [MAP-ROT-ENTRY] frame={int(k)} anchor={int(k - 1)} hist={int(n_hist)} f={float(f):.6f} "
                f"gamma_deg={float(gamma):.6f} kept={int(kept)} r_max={float(r_max):.6f} "
                f"alpha_deg={float(alpha):.6f} detected={int(bool(det))} fit={fit} result={result}")

    def _outside_r52(self, k, frames, world_map):
        # (1) the verified 04l/04m behaviour and lines, unchanged
        s = map_tracker.MapTracker.on_frame(self, k, frames, world_map)
        if s == "tracked":
            self._record_inliers(k, frames, world_map)
            self._lost_vel = None
            return self._count("tracked", False)
        result = None
        if s == "lost" and self._is_tracked(frames, k - 1):
            result = self._entry_r52(k, frames, world_map)          # (2)
            if result == "enter":
                # stage 04-r04 (spec B5 (a)): a post-gap entry frame relabelled by _gap_relabel counts as lost
                return self._count("lost" if getattr(frames[k], "tracking_state", None) == "LOST" else "rotation",
                                   True)
        # (3) and every refused attempt: the frame is lost and carries the lost prediction (C6)
        self._lost_prediction(k, frames)
        if result == "refused_fit":
            self.n_refused += 1
            print(self._rot_line(k, frames, "refused", "lost", k - 1, k - 1, "unsupported", "none", -1))
        return self._count("lost", False)

    def _entry_r52(self, k, frames, world_map):
        """Spec C5 (2), R52-2: the entry attempt at the first lost frame after the tracked anchor e = k-1.
        Returns enter, refused_detect, refused_fit or error."""
        fr = frames[k]
        T_lost = np.array(fr.cam_T_world, dtype=np.float64, copy=True)
        W_lost = np.array(fr.world_T_cam, dtype=np.float64, copy=True)
        e = k - 1
        n_hist, f, gamma = 0, float("nan"), float("nan")
        det, kept, r_max, alpha = False, 0, -1.0, -1.0
        fit = "none"
        enter_line = None
        try:
            hist = history_frames(e, self._tracked_fn(frames), self.timestamps, self.fb_params.hist_s)
            n_hist = len(hist)
            f = self._anchor_depth(e, frames)
            gamma = fov_h_deg(frames[e].K, frames[e].img_shape[1])
            det, kept, r_max, alpha = detect_rotation(frames[e].world_T_cam, [frames[j].world_T_cam for j in hist],
                                                      f, gamma, self.fb_params)
            if not det:
                result = "refused_detect"                         # no fit, no segment
            else:
                seg = PanoramaSegment(self.fb_params.r_min)
                seg.world_map = world_map
                self._segments.append(seg)
                seg._new_keyframe(e)
                r = seg.on_frame(k, frames)
                fit = r
                if r == "measured":
                    # frame k keeps the fit's pose: its rotation about the panorama keyframe's centre,
                    # which is the anchor's
                    self._anchor = e
                    self._seg = seg
                    fr.pose_valid = True
                    fr.tracking_state = "ROT"
                    # stage 04-r04 (spec B5 (a), R65-4): the post-gap entry frame is labelled lost
                    label = "lost" if self._gap_relabel(k, frames, "enter") else "rotation"
                    enter_line = self._rot_line(k, frames, "enter", label, e, seg.kf, r, "none", -1)
                    result = "enter"
                else:
                    self._restore(fr, T_lost, W_lost)
                    fr.pose_valid = False
                    fr.tracking_state = "LOST"
                    result = "refused_fit"
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")
            self._end_stint()
            self._restore(fr, T_lost, W_lost)
            fr.pose_valid = False
            fr.tracking_state = "LOST"
            result = "error"
            enter_line = None
        self.n_attempts += 1
        self.n_detected += int(bool(det))
        if result == "refused_detect":
            self.n_refused_detect += 1
        elif result == "refused_fit":
            self.n_refused_fit += 1
        elif result == "enter":
            self.n_entered += 1
        print(self._entry_line(k, n_hist, f, gamma, kept, r_max, alpha, det, fit, result))
        if result == "enter":
            self.n_stints += 1
            print(enter_line)
        return result

    def _lost_prediction(self, k, frames):
        """Spec C6 (R52-3): frame k, lost outside a stint, keeps the tracker's rotation and gets the
        centre c_k = c_(k-1) + R_k^T s dt, s the body velocity fitted once per lost run over the R50-3
        window ending at the last tracked frame (zeros when it could not be fitted). No horizon."""
        try:
            ts = self.timestamps
            if self._is_tracked(frames, k - 1):
                w = bridge_window(k - 1, self._tracked_fn(frames), ts, self.fb_params)
                s, fitted = fit_body_velocity([frames[j].world_T_cam for j in w], [ts[j] for j in w],
                                              self.fb_params.min_frames)
                self._lost_vel = s
                self.n_lost_runs += 1
                self.n_lost_runs_fitted += int(bool(fitted))
                print(f"  >> [MAP-LOST-VEL] frame={int(k)} anchor={int(k - 1)} window_first={int(w[0]) if w else -1} "
                      f"window_n={len(w)} fitted={int(bool(fitted))} vel={_fmt9(s)}")
            s = np.zeros(3) if self._lost_vel is None else np.asarray(self._lost_vel, dtype=np.float64)
            fr = frames[k]
            R_k = np.array(np.asarray(fr.cam_T_world, dtype=np.float64)[:3, :3], dtype=np.float64, copy=True)
            c_prev = np.asarray(frames[k - 1].world_T_cam, dtype=np.float64)[:3, 3]
            dt = float(ts[k]) - float(ts[k - 1])
            c_k = c_prev + (R_k.T @ s) * dt
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_k
            T[:3, 3] = -R_k @ c_k
            if not fr.set_camera_pose(T):
                raise RuntimeError("set_camera_pose rejected the lost prediction")
            fr.pose_valid = False
            fr.tracking_state = "LOST"
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")

    def _inside_r52(self, k, frames, world_map):
        """Spec C7: a stint frame under r52; the position is held at the anchor's centre (every
        panorama keyframe of a stint has the anchor's centre, so the fit's pose has it)."""
        e = self._anchor
        seg = self._seg
        fr = frames[k]
        r = "unsupported"
        T_h = W_h = None
        n_track = -1
        track_printed = False
        try:
            # (4) the fit against the segment's current panorama keyframe, with the C3 switch rule
            r = seg.on_frame(k, frames)
            T_h = np.array(fr.cam_T_world, dtype=np.float64, copy=True)
            W_h = np.array(fr.world_T_cam, dtype=np.float64, copy=True)
            # (5) normal tracking from T_h (04v D2 (6) with T_h in place of T_b)
            T_seed = np.array(T_h, dtype=np.float64, copy=True)
            if self.mode == "m1a":
                res = self._track_m1a(k, frames, T_seed)
            else:
                res = self._track_m1b(k, frames, T_seed)
            T_fin, rows, kps, mask, counts = res
            n_track = int(np.sum(mask)) if len(mask) else 0
            # stage 04-m1 (spec C3 (b)): the count's acceptance, then the residual test (off: always True);
            # a rejected exit goes on to (6) as a count failure does
            if (T_fin is not None and n_track >= int(self.params.min_inliers)
                    and self._resid_accept(k, "exit", fr, T_fin, rows, kps, mask)):
                rmse = self._rmse(fr, T_fin, rows, kps, mask)
                fr.set_camera_pose(T_fin)
                fr.pose_valid = True
                fr.tracking_state = "OK"
                for row, kp, keep in zip(rows, kps, mask):
                    if keep:
                        world_map.link(fr, int(kp), self._rows_pt[int(row)])
                track_printed = True
                print(self._track_line(k, "tracked", "cv", counts["step_a"], counts["step_a_inliers"],
                                       counts["step_a_wide"], counts["step_b"], len(rows), n_track, rmse))
                self.n_exits_track += 1
                print(self._rot_line(k, frames, "exit", "tracked", e, seg.kf, r, "track", n_track))
                self._end_stint()
                self._record_inliers(k, frames, world_map)
                self._lost_vel = None
                return self._count("tracked", True)
            # (6) relocalization (04v D2 (7)): the parent's 04m path, frame k-1 seen as LOST for the call
            prev = frames[k - 1]
            prev_state = getattr(prev, "tracking_state", None)
            prev.tracking_state = "LOST"
            track_printed = True                             # the parent prints its own [MAP-TRACK] line
            try:
                s = map_tracker.MapTracker.on_frame(self, k, frames, world_map)
            finally:
                prev.tracking_state = prev_state
            if s == "tracked":
                self.n_exits_reloc += 1
                print(self._rot_line(k, frames, "exit", "tracked", e, seg.kf, r, "reloc", n_track))
                self._end_stint()
                self._record_inliers(k, frames, world_map)
                self._lost_vel = None
                return self._count("tracked", True)
            # (7) stay: T_h and W_h back bitwise (the relocalization attempt may have moved the frame)
            self._restore(fr, T_h, W_h)
            if r == "measured":
                if self._gap_relabel(k, frames, "stay"):
                    # stage 04-r04 (spec B5 (b), R65-4): the post-gap stay frame is labelled lost
                    label = "lost"
                else:
                    fr.pose_valid = True
                    fr.tracking_state = "ROT"
                    label = "rotation"
            else:
                fr.pose_valid = False
                fr.tracking_state = "LOST"
                label = "lost"
            print(self._rot_line(k, frames, "stay", label, e, seg.kf, r, "none", n_track))
            return self._count(label, True)
        except Exception as ex:
            msg = f"{type(ex).__name__}: {ex}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [MAP-ROT-ERROR] frame={int(k)} error={msg}")
            if T_h is not None:
                self._restore(fr, T_h, W_h)
            else:
                R1 = np.asarray(frames[k - 1].cam_T_world, dtype=np.float64)[:3, :3]
                c_e = np.asarray(frames[e].world_T_cam, dtype=np.float64)[:3, 3]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R1
                T[:3, 3] = -R1 @ c_e
                fr.set_camera_pose(T)
            fr.pose_valid = False
            fr.tracking_state = "LOST"
            if not track_printed:
                # one [MAP-TRACK] line per frame, as the parent's own error path prints
                print(self._track_line(k, "lost", "cv", -1 if self.mode == "m1b" else 0,
                                       -1 if self.mode == "m1b" else 0, 0, -1 if self.mode == "m1b" else 0,
                                       0, 0, float("nan")))
            kf = seg.kf if seg is not None and seg.kf is not None else e
            print(self._rot_line(k, frames, "stay", "lost", e, kf, "unsupported", "none", n_track))
            return self._count("lost", True)
