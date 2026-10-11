"""
Oracle ladder, pose rungs O1 to O4 (stage 03c; design.md#oracle-ladder-rung-definitions and
the per-rung protocol).

This is the ONLY product module that reads the TUM ground-truth file for estimation. It is
constructed by main.py only when ``--oracle`` is not ``none``; with the flag off no Oracle
exists and nothing in this module runs. It never reads images of any kind and it imports
nothing from the pipeline or the viewer.

Frame assumption (TUM RGB-D file formats): each ground-truth row is
``timestamp tx ty tz qx qy qz qw`` = world_T_cam of the colour camera's optical centre (the
motion-capture-to-camera calibration is already applied in the published file). The
verifier's o3nd and rotation-agreement checks are the empirical test of this assumption.

Ground truth at frame times (A3; the same rule as the harness): for frame k with timestamp
t_k take the row nearest in time; within 0.02 s the frame has truth ("exact"); otherwise, if
t_k lies between two consecutive rows whose gap is at most 0.5 s, interpolate linearly in
position and slerp in orientation ("interp"); otherwise the frame has NO truth.

Conventions (theory/conventions.md#pose-convention, tracking.MotionEstimate):
  * the estimator's motion for the pair (a, b), a earlier, b later, is x_b = R x_a + t;
    Frame.update_pose composes prev.world_T_cam @ geometry.rt_to_t_inv(R, t);
  * gt_relative(a, b) returns (R, t, m_gt) in exactly that convention with t in METRES:
        Rel = G_a^-1 G_b  (both world_T_cam),  R = Rel[:3, :3].T,  t = -R Rel[:3, 3],
    so that rt_to_t_inv(R, t) == Rel and the chain reproduces the truth (test D1);
  * one scale factor s (map units per metre) is fixed from ground truth alone before any
    frame is processed (A4) and never refitted;
  * O4 uses one gauge mapping (A5) defined once from the anchor frame and never re-aligned.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

# Stationary classification of a ground-truth step (design.md per-rung protocol row "S1, O2";
# the same thresholds as the harness).
STATIONARY_M = 0.001        # below this the translation direction is undefined
NEAR_STATIONARY_M = 0.003   # below this it is unstable (and too small to fix a scale)
GT_TOL_S = 0.02             # nearest ground-truth row accepted as "at" the frame time
GT_INTERP_MAX_GAP_S = 0.5   # interpolate across mocap gaps up to this; beyond it: no truth

RUNGS = ("o1", "o2", "o3", "o4")
STATIONARY_MODES = ("estimator", "zero")
ACTIONS = ("inject", "hold", "skip")
REASONS = ("ok", "bridge", "no_truth", "vo_fail", "homography")
CLASSES = ("translating", "near_stationary", "stationary", "undefined")


# --------------------------------------------------------------------------- #
# small rotation helpers (numpy only)
# --------------------------------------------------------------------------- #
def quat_to_R(q):
    """(qx, qy, qz, qw) -> 3x3 rotation; the quaternion is normalised first so the matrix is
    orthogonal to machine precision (a 6-decimal file quaternion is not exactly unit)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("zero or non-finite quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_slerp(q0, q1, a):
    q0 = np.asarray(q0, dtype=np.float64); q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d); s = math.sin(th)
    return (math.sin((1 - a) * th) / s) * q0 + (math.sin(a * th) / s) * q1


def rotation_angle_deg(R):
    """Angle of a rotation matrix in degrees, atan2 form (accurate for small angles)."""
    R = np.asarray(R, dtype=np.float64)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = 0.5 * float(np.linalg.norm(ax))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    return float(math.degrees(math.atan2(s, c)))


def angle_between_deg(u, v):
    """Angle between two 3-vectors in degrees; nan if either is (numerically) zero."""
    u = np.asarray(u, dtype=np.float64).reshape(3); v = np.asarray(v, dtype=np.float64).reshape(3)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu <= 1e-12 or nv <= 1e-12:
        return float("nan")
    u = u / nu; v = v / nv
    return float(math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v)))))


def classify_step(m):
    """Class of a ground-truth step from its length in metres (nan -> undefined)."""
    if m is None or not np.isfinite(m):
        return "undefined"
    if m < STATIONARY_M:
        return "stationary"
    if m < NEAR_STATIONARY_M:
        return "near_stationary"
    return "translating"


def _pose_mat(p3, q4):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(q4)
    T[:3, 3] = np.asarray(p3, dtype=np.float64)
    return T


def _fmt(v):
    """%.6f with nan printed as 'nan' (spec C1)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return f"{v:.6f}"


# --------------------------------------------------------------------------- #
# ground truth file
# --------------------------------------------------------------------------- #
def load_tum_file(path):
    """Rows of a TUM trajectory file as (times (N,), poses (N, 7) = tx ty tz qx qy qz qw),
    sorted by time; '#' comment lines and blank lines are skipped."""
    times, rows = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            vals = [float(x) for x in parts[:8]]
            times.append(vals[0]); rows.append(vals[1:8])
    if not rows:
        raise ValueError(f"no pose rows in {path}")
    times = np.asarray(times, dtype=np.float64); rows = np.asarray(rows, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    return times[order], rows[order]


def truth_at_times(gt_times, gt_rows, timestamps, tol=GT_TOL_S, max_gap=GT_INTERP_MAX_GAP_S):
    """Per frame: (world_T_cam 4x4 or None, status in {'exact', 'interp', 'none'}) (spec A3)."""
    out = []
    n = len(gt_times)
    for t in timestamps:
        t = float(t)
        j = int(np.searchsorted(gt_times, t))
        cands = [c for c in (j - 1, j) if 0 <= c < n]
        c = min(cands, key=lambda c: abs(gt_times[c] - t))
        if abs(gt_times[c] - t) <= tol:
            out.append((_pose_mat(gt_rows[c, :3], gt_rows[c, 3:]), "exact")); continue
        if 0 < j < n and (gt_times[j] - gt_times[j - 1]) <= max_gap:
            a = (t - gt_times[j - 1]) / (gt_times[j] - gt_times[j - 1])
            p = (1 - a) * gt_rows[j - 1, :3] + a * gt_rows[j, :3]
            q = _quat_slerp(gt_rows[j - 1, 3:], gt_rows[j, 3:], a)
            out.append((_pose_mat(p, q), "interp")); continue
        out.append((None, "none"))
    return out


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #
class Oracle:
    """Ground-truth injector for the pose rungs (spec A2). Index k = trajectory index =
    Frame.id - 1; timestamps has one float per frame."""

    def __init__(self, rung, gt_path, timestamps, stationary="estimator"):
        rung = str(rung).lower()
        if rung not in RUNGS:
            raise ValueError(f"unknown oracle rung {rung!r}; expected one of {RUNGS}")
        if stationary not in STATIONARY_MODES:
            raise ValueError(f"unknown --oracle_stationary {stationary!r}; expected one of {STATIONARY_MODES}")
        if timestamps is None or len(timestamps) == 0:
            raise ValueError("oracle needs one timestamp per frame (none given)")
        self.rung = rung
        self.stationary = stationary
        self.gt_path = str(gt_path)
        self.timestamps = [float(t) for t in timestamps]
        self.n_frames = len(self.timestamps)

        gt_times, gt_rows = load_tum_file(self.gt_path)
        truth = truth_at_times(gt_times, gt_rows, self.timestamps)
        self._G = [T for T, _ in truth]           # world_T_cam per frame, None without truth
        self._status = [st for _, st in truth]
        self.frames_with_truth = sum(1 for T in self._G if T is not None)
        self.frames_interp = sum(1 for st in self._status if st == "interp")
        if self.frames_with_truth == 0:
            raise ValueError("no frame has ground truth at its timestamp")

        # A5: the anchor is the first frame with truth; its estimator pose E is the identity
        # when anchor_k == 0, else captured once via set_anchor_pose when that frame finishes.
        self.anchor_k = next(k for k, T in enumerate(self._G) if T is not None)
        self._anchor_pose = np.eye(4, dtype=np.float64) if self.anchor_k == 0 else None

        # A4: one scale factor from the first consecutive pair with truth on both frames and a
        # baseline of at least NEAR_STATIONARY_M; fixed here, before any frame is processed.
        self.scale_pair = None
        self.scale_s = float("nan")
        self.scale_gt_step_m = float("nan")
        for k in range(1, self.n_frames):
            if self._G[k - 1] is None or self._G[k] is None:
                continue
            _, _, m = self.gt_relative(k - 1, k)
            if m >= NEAR_STATIONARY_M:
                self.scale_pair = (k - 1, k)
                self.scale_s = 1.0 / m
                self.scale_gt_step_m = m
                break
        if self.scale_pair is None:
            raise ValueError(
                f"no consecutive frame pair with truth on both frames and a baseline of at least "
                f"{NEAR_STATIONARY_M} m; cannot fix the scale factor s")

        self.counts = {"inject": 0, "hold": 0, "skip": 0, "bridge": 0, "accepted": 0}

    # ---- truth access -------------------------------------------------------------------
    def has_truth(self, k):
        return 0 <= int(k) < self.n_frames and self._G[int(k)] is not None

    def status(self, k):
        return self._status[int(k)] if 0 <= int(k) < self.n_frames else "none"

    def gt_world_T_cam(self, k):
        """Ground-truth world_T_cam of frame k (a copy) or None."""
        T = self._G[int(k)] if 0 <= int(k) < self.n_frames else None
        return None if T is None else T.copy()

    def gt_relative(self, a, b):
        """Ground-truth motion of the pair (a, b) in the MotionEstimate convention
        x_b = R x_a + t (a earlier, b later), t in METRES as a (3, 1) column; m_gt = ||t||."""
        Ga, Gb = self._G[int(a)], self._G[int(b)]
        if Ga is None or Gb is None:
            raise ValueError(f"gt_relative({a}, {b}): a frame without truth")
        Rel = np.linalg.inv(Ga) @ Gb
        R = np.ascontiguousarray(Rel[:3, :3].T, dtype=np.float64)
        t = (-R @ Rel[:3, 3]).reshape(3, 1).astype(np.float64)
        return R, t, float(np.linalg.norm(t))

    def _last_truth_before(self, k):
        for a in range(int(k) - 1, -1, -1):
            if self._G[a] is not None:
                return a
        return None

    # ---- injection point 1: O1 / O2 / O3 --------------------------------------------------
    def _blank_record(self, k, action, reason, cls):
        nan = float("nan")
        return {
            "frame": int(k), "rung": self.rung, "action": action, "reason": reason, "class": cls,
            "gt_step_m": nan, "inj_rot_deg": nan, "inj_step": nan, "est_rot_deg": nan,
            "est_step": nan, "rot_err_deg": nan, "dir_err_deg": nan, "accepted": "-",
        }

    def override_motion(self, k, motion):
        """Rungs o1/o2/o3 (spec B1). Returns (motion_out, record): a NEW MotionEstimate (the
        input is never mutated) carrying the injected R and t, and the [ORACLE] record."""
        if self.rung not in ("o1", "o2", "o3"):
            raise ValueError(f"override_motion is not defined for rung {self.rung}")
        k = int(k)
        nan = float("nan")

        # VO failure: the pipeline's own failure path (hold / relocalization) acts.
        if motion is None:
            rec = self._blank_record(k, "skip", "vo_fail", "undefined")
            self._fill_gt_step(rec, k)
            self.counts["skip"] += 1
            return None, rec

        R_est = np.asarray(motion.R, dtype=np.float64).reshape(3, 3)
        u_est = np.asarray(motion.t, dtype=np.float64).reshape(3, 1)
        est_rot_deg = rotation_angle_deg(R_est)
        est_step = float(np.linalg.norm(u_est))

        if getattr(motion, "model", None) == "HOMOGRAPHY":
            rec = self._blank_record(k, "skip", "homography", "undefined")
            self._fill_gt_step(rec, k)
            rec.update(est_rot_deg=est_rot_deg, est_step=est_step, inj_rot_deg=est_rot_deg, inj_step=est_step)
            self.counts["skip"] += 1
            return motion, rec

        # Reference frame for the injected motion: k-1, or the last frame with truth (bridge).
        reason = None
        if self.has_truth(k):
            if self.has_truth(k - 1):
                a, reason = k - 1, "ok"
            else:
                a = self._last_truth_before(k)
                reason = "bridge" if a is not None else None
        if reason is None:
            # No truth on frame k (or nothing earlier to measure from): keep the previous
            # pose (R = I, t = 0 -> [POSE-ONLINE] step 0; ||t|| = 0 skips triangulation).
            rec = self._blank_record(k, "hold", "no_truth", "undefined")
            rec.update(est_rot_deg=est_rot_deg, est_step=est_step, inj_rot_deg=0.0, inj_step=0.0)
            out = dataclasses.replace(motion, R=np.eye(3, dtype=np.float64), t=np.zeros((3, 1), dtype=np.float64))
            self.counts["hold"] += 1
            return out, rec

        R_gt, t_gt, m_gt = self.gt_relative(a, k)
        cls = classify_step(m_gt)
        u_gt = (t_gt / m_gt) if m_gt > 0.0 else np.zeros((3, 1), dtype=np.float64)
        u_est_unit = (u_est / est_step) if est_step > 1e-12 else u_est.copy()

        if self.rung == "o1":
            R = R_est.copy()
            t = u_est_unit * (self.scale_s * m_gt)
        elif self.rung == "o2":
            R = R_gt
            if cls in ("translating", "near_stationary"):
                t = u_gt.copy()
            elif self.stationary == "zero":
                t = np.zeros((3, 1), dtype=np.float64)
            else:
                t = u_est_unit.copy()
        else:  # o3
            R = R_gt
            t = self.scale_s * t_gt

        R = np.ascontiguousarray(R, dtype=np.float64).reshape(3, 3)
        t = np.ascontiguousarray(t, dtype=np.float64).reshape(3, 1)
        out = dataclasses.replace(motion, R=R, t=t)

        rec = self._blank_record(k, "inject", reason, cls)
        rec.update(
            gt_step_m=m_gt,
            inj_rot_deg=rotation_angle_deg(R), inj_step=float(np.linalg.norm(t)),
            est_rot_deg=est_rot_deg, est_step=est_step,
            rot_err_deg=rotation_angle_deg(R_est.T @ R_gt),
            dir_err_deg=angle_between_deg(u_est, u_gt) if cls in ("translating", "near_stationary") else nan,
        )
        self.counts["inject"] += 1
        if reason == "bridge":
            self.counts["bridge"] += 1
        return out, rec

    def _fill_gt_step(self, rec, k):
        """On a skip, still report the pair's ground-truth step and class when both frames
        have truth (informational; the estimator's own numbers stay nan)."""
        if self.has_truth(k) and self.has_truth(k - 1):
            _, _, m = self.gt_relative(k - 1, k)
            rec["gt_step_m"] = m
            rec["class"] = classify_step(m)

    # ---- injection point 2: O4 --------------------------------------------------------------
    def set_anchor_pose(self, E):
        """Capture the estimator's world_T_cam of frame anchor_k once (A5; anchor_k > 0 only)."""
        self._anchor_pose = np.array(E, dtype=np.float64, copy=True).reshape(4, 4)

    def anchor_pose(self):
        return None if self._anchor_pose is None else self._anchor_pose.copy()

    def est_world_T_cam(self, k):
        """Ground truth of frame k mapped into the estimator frame by the single gauge mapping
        E @ scale_s(G(anchor)^-1 G(k)); None without truth."""
        if not self.has_truth(k):
            return None
        if self._anchor_pose is None:
            raise RuntimeError("oracle anchor pose not set yet (frame anchor_k has not finished)")
        rel = np.linalg.inv(self._G[self.anchor_k]) @ self._G[int(k)]
        rel = rel.copy()
        rel[:3, 3] *= self.scale_s
        return self._anchor_pose @ rel

    def absolute_pose(self, k):
        """Rung o4 (spec B5): cam_T_world of frame k in the estimator frame, or None."""
        if self.rung != "o4":
            raise ValueError(f"absolute_pose is not defined for rung {self.rung}")
        T = self.est_world_T_cam(k)
        return None if T is None else np.linalg.inv(T)

    def o4_record(self, k, action, reason, prev_world_T_cam=None, chained_world_T_cam=None,
                  injected_world_T_cam=None, accepted=0):
        """The [ORACLE] record of one PnP frame under o4 (spec C1, o4 form): inj_* are the
        injected pose's rotation angle and translation norm relative to frame k-1's pose,
        est_* the chained pose's; accepted is the gate's decision (0 when skipped)."""
        k = int(k)
        rec = self._blank_record(k, action, reason, "undefined")
        rec["accepted"] = 1 if accepted else 0
        if action == "inject" and self.has_truth(k) and self.has_truth(k - 1):
            _, _, m = self.gt_relative(k - 1, k)
            rec["gt_step_m"] = m
            rec["class"] = classify_step(m)
        if prev_world_T_cam is not None:
            P = np.asarray(prev_world_T_cam, dtype=np.float64)
            if injected_world_T_cam is not None:
                I_ = np.asarray(injected_world_T_cam, dtype=np.float64)
                rec["inj_rot_deg"] = rotation_angle_deg(P[:3, :3].T @ I_[:3, :3])
                rec["inj_step"] = float(np.linalg.norm(I_[:3, 3] - P[:3, 3]))
            if chained_world_T_cam is not None:
                C = np.asarray(chained_world_T_cam, dtype=np.float64)
                rec["est_rot_deg"] = rotation_angle_deg(P[:3, :3].T @ C[:3, :3])
                rec["est_step"] = float(np.linalg.norm(C[:3, 3] - P[:3, 3]))
        if action in self.counts:
            self.counts[action] += 1
        if accepted:
            self.counts["accepted"] += 1
        return rec

    # ---- log lines --------------------------------------------------------------------------
    def format_line(self, record):
        """Spec C1."""
        r = record
        return (
            f"  >> [ORACLE] frame={int(r['frame'])} rung={r['rung']} action={r['action']} "
            f"reason={r['reason']} class={r['class']} gt_step_m={_fmt(r['gt_step_m'])} "
            f"inj_rot_deg={_fmt(r['inj_rot_deg'])} inj_step={_fmt(r['inj_step'])} "
            f"est_rot_deg={_fmt(r['est_rot_deg'])} est_step={_fmt(r['est_step'])} "
            f"rot_err_deg={_fmt(r['rot_err_deg'])} dir_err_deg={_fmt(r['dir_err_deg'])} "
            f"accepted={r['accepted']}"
        )

    def setup_line(self):
        """Spec C2."""
        a, b = self.scale_pair
        return (
            f"  >> [ORACLE-SETUP] rung={self.rung} stationary={self.stationary} "
            f"frames={self.n_frames} frames_with_truth={self.frames_with_truth} "
            f"frames_interp={self.frames_interp} scale_pair={a},{b} "
            f"gt_step_m={_fmt(self.scale_gt_step_m)} s={_fmt(self.scale_s)} anchor={self.anchor_k}"
        )

    def summary_line(self):
        """Spec C3."""
        c = self.counts
        return (
            f"  Oracle: rung={self.rung} inject={c['inject']} hold={c['hold']} "
            f"skip={c['skip']} bridge={c['bridge']} accepted={c['accepted']}"
        )
