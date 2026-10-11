"""
Stage 04u (milestone 04-3a, initializer theme, third contract; rulings P1, P2, R37-1, R37-2, R40-1,
R41-1, R41-2, R43-1): the verified two-view initializer (initializer.py, 04s, with the refit switch
of 04t) wired into the verified pre-map rotation-only mode (rotation_only.py, 04r), behind
--tracking rotation --init twoview.

The driver has three states:
- ROTATION: a rotation-only segment orients every frame exactly as 04r does; the newest frame is
  then tried against the last `window` entries of the segment's candidate list (its panorama
  keyframes and its renewal frames, R43-1) with ORB-SLAM's selection and refusal;
- SURVIVAL: an accepted map has been handed to the verified map tracker (map_tracker.py, 04l, with
  the relocalizing state of 04m); a lost frame within `survival_frames` tracked frames is a false
  start: the map's points are removed and a new rotation segment starts at the previous frame;
- MAP: the map survived; the tracker poses every later frame. There is no further initialization
  attempt, rotation segment or reset (the mapper arrives with 04-4).

Convention (conventions.md#pose-convention): cam_T_world maps world to camera. The initializer's
x_b = R x_a + t (camera a the earlier frame), so cam_T_world(b) = [R | t] @ cam_T_world(a) and a
point X_a of camera a is X_w = R_a^T (X_a - t_a) in the world, [R_a | t_a] = cam_T_world(a).

This module reads no dataset file, image or truth: its only inputs are the frames' keypoints,
descriptors, intrinsics, image shape and poses. It draws no random number; the initializer and
the tracker are seeded as verified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import initializer
import local_mapping
import map_tracker
import rotation_only


@dataclass
class OpeningParams:
    """Constants of the opening driver. No CLI flag sets them (status note, O-2).

    window: the panorama keyframes tried per frame. On the 4 x 3 coverage grid a new panorama
    keyframe is taken after about 22.7 deg of turn, so two keyframes offer partners from 0 to about
    45 deg back, which holds the 22-25 deg best-partner separation measured on the opening. Under
    straight driving the coverage rule does not fire; the candidate list then renews on a matches
    refusal (R43-1), so overlap is read by what the initializer matches.
    survival_frames: the survival interval, in tracked frames after the accepting frame. At the
    opening's highest turn rate (23 deg/s) 10 frames at 30 Hz are 7.67 deg, so 0.81 of a map's
    41 deg band stays in view and a correct map of 51 points or more keeps 30 inliers even at the
    p10 retention of 04l; long enough to catch the collapse ORB-SLAM3's first map showed at once.
    """
    window: int = 2
    survival_frames: int = 10


_REFIT = ("off", "shadow", "live")


def _f6(x):
    return "%.6f" % float(x)


def _f9(x):
    return "%.9f" % float(x)


def format_pose_line(a, b, arm, result):
    """The [INIT-POSE] line of an accepted result (spec D7): R row-major and t, %.9f, comma-separated."""
    R = np.asarray(result.R, dtype=np.float64).reshape(-1)
    t = np.asarray(result.t, dtype=np.float64).reshape(-1)
    return (f"  >> [INIT-POSE] a={int(a)} b={int(b)} arm={arm} R={','.join(_f9(x) for x in R)} "
            f"t={','.join(_f9(x) for x in t)}")


def _error_line(a, b, arm, e):
    msg = f"{type(e).__name__}: {e}".replace("\r", " ").replace("\n", " ")
    return f"  >> [INIT-ERROR] a={int(a)} b={int(b)} arm={arm} error={msg}"


class OpeningTracker:
    """The opening driver (spec D). The pipeline sets world_map and use_homography before start()."""

    def __init__(self, params=None, refit="off"):
        if refit not in _REFIT:
            raise ValueError(f"unknown refit setting {refit!r} (expected one of {', '.join(_REFIT)})")
        self.params = OpeningParams() if params is None else params
        self.refit = refit
        self.world_map = None
        self.use_homography = False
        self.state = "ROTATION"
        # the current rotation segment, its keyframe list and its candidate list (R43-1)
        self.seg = None
        self.seg_kfs = []
        self.seg_cands = []
        self.segments = []          # every rotation segment, for the summed rot_ counts
        # the map handed to the tracker: number, anchor (m, a, b), tracker, tracked counter
        self.tracker = None
        self.anchor = None
        self.counter = 0
        # stage 04-m0 (spec C1): builds the hand-off's tracker from its M1Map when set (main.run_pipeline
        # under --live_map on); None builds map_tracker.MapTracker(m1map, "m1a", None) as 04u
        self.tracker_factory = None
        # counters of the summary line
        self.n_frames = 0
        self.n_attempts = 0
        self.n_accepted_live = 0
        self.n_accepted_shadow = 0
        self.n_errors = 0
        self.n_maps = 0
        self.n_resets = 0
        self.n_renewals = 0
        self.survived = 0
        self.first_accept_b = -1
        self.survived_b = -1
        self.n_tracked = 0
        self.n_lost = 0

    def setup_line(self):
        p = self.params
        return (f"  >> [INIT-SETUP] window={int(p.window)} survival_frames={int(p.survival_frames)} "
                f"refit={self.refit}")

    # ---------------------------------------------------------------- rotation segments
    def _new_segment(self):
        seg = rotation_only.RotationOnlyTracker()
        seg.world_map = self.world_map
        self.seg = seg
        self.segments.append(seg)
        return seg

    def start(self, frames):
        """Spec D2: the first rotation segment; frame 0 is the gauge and the first panorama keyframe."""
        self.n_frames = 1
        seg = self._new_segment()
        print(seg.setup_line())
        seg.start(frames)
        self.seg_kfs = [0]
        self.seg_cands = [0]

    def _orient(self, k, frames):
        """Spec D3 (1): the segment orients frame k (04r unchanged); a new panorama keyframe joins
        the segment's keyframe and candidate lists."""
        before = self.seg.kf
        r = self.seg.on_frame(k, frames)
        if self.seg.kf != before:
            self.seg_kfs.append(self.seg.kf)
            self.seg_cands.append(self.seg.kf)
        return r

    # ---------------------------------------------------------------- per frame
    def on_frame(self, k, frames):
        """Spec D3-D5: returns "measured", "unsupported", "init", "tracked" or "lost"."""
        self.n_frames += 1
        if self.state == "ROTATION":
            r = self._on_rotation(k, frames)
        elif self.state == "SURVIVAL":
            r = self._on_survival(k, frames)
        else:
            r = self.tracker.on_frame(k, frames, self.world_map)
        if r == "tracked":
            self.n_tracked += 1
        elif r == "lost":
            self.n_lost += 1
        return r

    def _attempt(self, a, k, frames, refit):
        """One initialize_frames call; (result, None) or (None, exception)."""
        try:
            return initializer.initialize_frames(frames[a], frames[k], refit=refit), None
        except Exception as e:
            return None, e

    def _on_rotation(self, k, frames):
        r = self._orient(k, frames)
        cands = [a for a in self.seg_cands if a < k][-int(self.params.window):]
        accepted = []               # (a, result) of the live arm
        live_reasons = []           # (status, reason) of each live pair line; None for an error
        for a in cands:
            res, err = self._attempt(a, k, frames, self.refit == "live")
            self.n_attempts += 1
            if err is not None:
                self.n_errors += 1
                live_reasons.append(None)
                print(_error_line(a, k, "live", err))
            else:
                live_reasons.append((res.status, res.reason))
                print(initializer.format_init_line(a, k, res))
                if res.status == "accepted":
                    self.n_accepted_live += 1
                    print(format_pose_line(a, k, "live", res))
                    accepted.append((a, res))
            if self.refit == "shadow":
                sh, err = self._attempt(a, k, frames, True)
                if err is not None:
                    self.n_errors += 1
                    print(_error_line(a, k, "shadow", err))
                else:
                    print(initializer.format_init_line(a, k, sh).replace("[INIT-PAIR]", "[INIT-PAIR-SHADOW]", 1))
                    if sh.status == "accepted":
                        self.n_accepted_shadow += 1
                        print(format_pose_line(a, k, "shadow", sh))
        if not accepted:
            # spec D3 (3a), R43-1: the reference renews on a matches refusal of every live pair
            if (cands and all(x == ("refused", "matches") for x in live_reasons)
                    and len(frames[k].KeyPoints) > initializer.InitParams().min_matches):
                self.seg_cands.append(k)
                self.n_renewals += 1
                print(f"  >> [INIT-CAND] frame={int(k)} reason=matches")
            return r
        a, res = accepted[0]
        for a2, res2 in accepted[1:]:
            if len(res2.points) > len(res.points):
                a, res = a2, res2
        self._hand_off(a, k, res, frames)
        return "init"

    def _hand_off(self, a, k, res, frames):
        """Spec D3 (5): frame k's pose from the chosen pair, the map's points in the world, and the
        hand-off to the map tracker with its relocalizing state."""
        self.n_maps += 1
        m = self.n_maps
        if self.first_accept_b < 0:
            self.first_accept_b = int(k)
        fa, fb = frames[a], frames[k]
        T_a = np.asarray(fa.cam_T_world, dtype=np.float64)
        Rt = np.eye(4, dtype=np.float64)
        Rt[:3, :3] = np.asarray(res.R, dtype=np.float64)
        Rt[:3, 3] = np.asarray(res.t, dtype=np.float64).reshape(3)
        if not fb.set_camera_pose(Rt @ T_a):
            raise RuntimeError("set_camera_pose rejected the initialized pose")
        fb.pose_valid = True
        fb.tracking_state = "OK"
        n = len(res.points)
        print(f"  >> [INIT-ACCEPT] map={m} a={int(a)} b={int(k)} published={n} model={res.model}")
        sig_a = np.asarray(local_mapping._keypoint_sigmas(fa), dtype=np.float64)
        sig_b = np.asarray(local_mapping._keypoint_sigmas(fb), dtype=np.float64)
        R_a, t_a = T_a[:3, :3], T_a[:3, 3]
        recs = []
        for i in range(n):
            ka, kb = int(res.kp_a[i]), int(res.kp_b[i])
            ua, va = fa.KeyPoints[ka].pt
            ub, vb = fb.KeyPoints[kb].pt
            X_a = np.asarray(res.points[i], dtype=np.float64)
            print(f"  >> [INIT-POINT] map={m} i={i} kp_a={ka} kp_b={kb} ua={_f6(ua)} va={_f6(va)} "
                  f"ub={_f6(ub)} vb={_f6(vb)} x={_f9(X_a[0])} y={_f9(X_a[1])} z={_f9(X_a[2])} "
                  f"par_deg={_f9(res.parallax_deg[i])} sig_a={_f6(sig_a[ka])} sig_b={_f6(sig_b[kb])}")
            X_w = R_a.T @ (X_a - t_a)
            recs.append((i, int(k), kb, float(ub), float(vb), float(X_w[0]), float(X_w[1]), float(X_w[2])))
        m1map = map_tracker.M1Map(header={}, points=recs, corr={})
        if self.tracker_factory is not None:
            tracker = self.tracker_factory(m1map)
        else:
            tracker = map_tracker.MapTracker(m1map, "m1a", None)
        tracker.use_homography = self.use_homography
        tracker.reloc = "pnp"
        tracker.reloc_params = map_tracker.MapRelocParams()
        tracker.reloc_solver = "epnp"
        tracker.reloc_shadow = False
        print(tracker.setup_line())
        tracker.release(k, frames, self.world_map)
        self.tracker = tracker
        self.anchor = (m, int(a), int(k))
        self.counter = 0
        self.state = "SURVIVAL"

    def _on_survival(self, k, frames):
        """Spec D4: the survival check over params.survival_frames tracked frames."""
        s = self.tracker.on_frame(k, frames, self.world_map)
        m, a, b = self.anchor
        if s == "tracked":
            self.counter += 1
            if self.counter >= int(self.params.survival_frames):
                print(f"  >> [INIT-SURVIVED] map={m} a={a} b={b} frame={int(k)}")
                self.survived = 1
                self.survived_b = b
                self.state = "MAP"
            return "tracked"
        # a false start: remove the map, restart the rotation mode at frame k - 1
        removed = 0
        for pt in list(self.tracker.points_by_pid.values()):
            self.world_map.remove_point(pt.id, frames)
            removed += 1
        print(f"  >> [INIT-RESET] map={m} a={a} b={b} frame={int(k)} tracked={int(self.counter)} "
              f"removed={removed}")
        self.n_resets += 1
        self.tracker = None
        self.anchor = None
        self.counter = 0
        seg = self._new_segment()
        seg._new_keyframe(k - 1)
        self.seg_kfs = [k - 1]
        self.seg_cands = [k - 1]
        self.state = "ROTATION"
        return self._orient(k, frames)

    def summary_line(self):
        n_points = -1 if self.world_map is None else len(self.world_map.points)
        rot_measured = sum(s.n_measured for s in self.segments)
        rot_unsupported = sum(s.n_unsupported for s in self.segments)
        rot_keyframes = sum(s.n_kf for s in self.segments)
        return (f"  >> [INIT-SUMMARY] frames={int(self.n_frames)} rot_segments={len(self.segments)} "
                f"rot_measured={int(rot_measured)} rot_unsupported={int(rot_unsupported)} "
                f"rot_keyframes={int(rot_keyframes)} attempts={int(self.n_attempts)} "
                f"accepted_live={int(self.n_accepted_live)} accepted_shadow={int(self.n_accepted_shadow)} "
                f"errors={int(self.n_errors)} maps={int(self.n_maps)} resets={int(self.n_resets)} "
                f"renewals={int(self.n_renewals)} survived={int(self.survived)} "
                f"first_accept_b={int(self.first_accept_b)} survived_b={int(self.survived_b)} "
                f"tracked={int(self.n_tracked)} lost={int(self.n_lost)} map_points={int(n_points)}")
