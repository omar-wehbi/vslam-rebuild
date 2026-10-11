"""
Stage 04r (milestone 04-3a, first contract; P1-P7 ruled 2026-09-26, theory/workshop/35): the
pre-map rotation-only mode (--tracking rotation).

From frame 0, each frame's orientation is solved by a rotation-only fit against the current
panorama keyframe (design.md slot "No-parallax states"; papers/pirchheim-hybrid-rotation-slam.md
#the-3-dof-state-and-the-transitions): the product's ratio-test matches, bearings from the
undistorted keypoints, 2-point RANSAC with a per-match gate from the keypoint scales, then the
SO(3) least-squares fit (Wahba's problem, SVD) on the consensus set. This is the "simple fit (A)"
of theory/workshop/35-rotation-pretest.py (fit_A, wahba), reproduced exactly (P4's default).

A new panorama keyframe is taken when the keyframe's view covers less than min_coverage of the
current image, measured on Pirchheim's grid of cell centres (view_coverage;
papers/pirchheim-hybrid-rotation-slam.md#keyframe-selection-6-dof-localized-panorama-panorama).

Translation is held at the panorama keyframe's centre (the origin before any map) and is
unsupported (conventions.md#observability-under-pure-rotation). No map point, observation, Map
keyframe or depth-filter seed is created. A frame without an accepted solve is flagged
unsupported (pose_valid False, tracking_state LOST) and carries the constant-angular-velocity
prediction, never the identity unless the prediction is the identity.

Convention (conventions.md#pose-convention): a holds bearings in the panorama keyframe's camera,
b in the current frame's, b ~ R a, so under pure rotation R is the rotation block of
cam_T_world(current) @ world_T_cam(keyframe).

This module reads no dataset file, image or truth: its only inputs are the frames' keypoints,
descriptors, intrinsics, image shape and poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from keypoints_match import match_frames
from local_mapping import _keypoint_sigmas


@dataclass
class RotOnlyParams:
    """Constants of the pre-map rotation-only mode. No CLI flag sets them.

    hypotheses, seed, gate_sigmas, min_inliers: the workshop/35 pre-test's fit (A), pre-registered
    there (512 two-point hypotheses from np.random.default_rng(43), a gate of 3 sigma of the
    combined keypoint scale over f, at least 5 inliers).
    grid_cols, grid_rows, min_coverage: Pirchheim's coverage constants (4 x 3 cell centres, a new
    panorama keyframe below 0.75 coverage; papers/pirchheim-hybrid-rotation-slam.md
    #keyframe-selection-6-dof-localized-panorama-panorama), recorded as starting points.
    """
    hypotheses: int = 512
    seed: int = 43
    gate_sigmas: float = 3.0
    min_inliers: int = 5
    grid_cols: int = 4
    grid_rows: int = 3
    min_coverage: float = 0.75


def wahba(a, b):
    """The rotation R minimizing sum |b_i - R a_i|^2 over (n, 3) unit bearings a, b."""
    U, _S, Vt = np.linalg.svd(b.T @ a)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def solve_rotation(a, b, sigma_a, sigma_b, f, params):
    """2-point RANSAC plus the SO(3) least-squares fit on the consensus set (spec C).

    Returns (R or None, mask, reason): mask a boolean (n,) array (the best consensus set; all
    False when RANSAC was not reached), reason one of "none", "matches", "inliers".
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    n = len(a)
    if n < params.min_inliers:
        return None, np.zeros(n, dtype=bool), "matches"
    sa = np.asarray(sigma_a, dtype=np.float64)
    sb = np.asarray(sigma_b, dtype=np.float64)
    gate = params.gate_sigmas * np.sqrt(sa ** 2 + sb ** 2) / f
    rng = np.random.default_rng(params.seed)
    best = np.zeros(n, dtype=bool)
    for _ in range(params.hypotheses):
        i = rng.choice(n, 2, replace=False)
        if min(np.linalg.norm(np.cross(a[i[0]], a[i[1]])), np.linalg.norm(np.cross(b[i[0]], b[i[1]]))) < 1e-3:
            continue
        R_h = wahba(a[i], b[i])
        m = np.linalg.norm(b - a @ R_h.T, axis=1) < gate
        if m.sum() > best.sum():
            best = m
    if best.sum() < params.min_inliers:
        return None, best, "inliers"
    return wahba(a[best], b[best]), best, "none"


def view_coverage(K, R, width, height, params):
    """Share in [0, 1] of the current image's grid_cols x grid_rows cell centres whose bearings,
    mapped into the keyframe's camera (q = R.T d), lie in front of it and project inside
    [0, width) x [0, height) (spec D). R is R_(k,P) as in solve_rotation."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Ki = np.linalg.inv(K)
    count = 0
    for j in range(params.grid_rows):
        for i in range(params.grid_cols):
            x = (i + 0.5) * width / params.grid_cols
            y = (j + 0.5) * height / params.grid_rows
            q = R.T @ (Ki @ np.array([x, y, 1.0]))
            if q[2] > 0:
                p = K @ q / q[2]
                if 0 <= p[0] < width and 0 <= p[1] < height:
                    count += 1
    return count / (params.grid_cols * params.grid_rows)


def rotation_angle_deg(R):
    """Rotation angle of R in degrees (atan2 form, accurate near 0 and near 180)."""
    R = np.asarray(R, dtype=np.float64)
    s = 0.5 * math.sqrt((R[2, 1] - R[1, 2]) ** 2 + (R[0, 2] - R[2, 0]) ** 2 + (R[1, 0] - R[0, 1]) ** 2)
    c = 0.5 * (float(np.trace(R)) - 1.0)
    return math.degrees(math.atan2(s, c))


def _bearings(frame, idx):
    """Unit bearings inv(K) [u, v, 1] of the frame's keypoints idx (pixels as extracted, i.e.
    undistorted under --undistort), K = the frame's own intrinsics."""
    kps = frame.KeyPoints
    uv = np.array([kps[i].pt for i in idx], dtype=np.float64).reshape(-1, 2)
    h = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(np.asarray(frame.K, dtype=np.float64)).T
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def _fmt(x):
    return f"{float(x):.6f}"


class RotationOnlyTracker:
    """The pre-map rotation-only mode (spec E). The pipeline sets world_map before start()."""

    def __init__(self, params=None):
        self.params = RotOnlyParams() if params is None else params
        self.world_map = None
        self.kf = None          # frame index of the current panorama keyframe
        self.n_kf = 0
        self.n_frames = 0       # frames processed, frame 0 included
        self.n_measured = 0
        self.n_unsupported = 0

    def setup_line(self):
        p = self.params
        return (f"  >> [ROT-SETUP] hypotheses={int(p.hypotheses)} seed={int(p.seed)} "
                f"gate_sigmas={_fmt(p.gate_sigmas)} min_inliers={int(p.min_inliers)} "
                f"grid={int(p.grid_cols)}x{int(p.grid_rows)} min_coverage={_fmt(p.min_coverage)}")

    def _new_keyframe(self, k):
        prev = -1 if self.kf is None else self.kf
        self.kf = k
        self.n_kf += 1
        print(f"  >> [ROT-KF] frame={int(k)} prev={int(prev)} n={int(self.n_kf)}")

    def start(self, frames):
        """Frame 0 keeps its identity pose (the gauge: world = camera 0) and is the first
        panorama keyframe."""
        self.n_frames = 1
        self._new_keyframe(0)

    def _predicted_rotation(self, k, frames):
        """Constant angular velocity from the two previous output poses (cam_T_world blocks)."""
        R1 = np.asarray(frames[k - 1].cam_T_world, dtype=np.float64)[:3, :3]
        if k < 2:
            return R1.copy()
        R2 = np.asarray(frames[k - 2].cam_T_world, dtype=np.float64)[:3, :3]
        V = R1 @ R2.T
        return V @ R1

    def on_frame(self, k, frames):
        """Solve frame k (k >= 1) against the current panorama keyframe; returns "measured" or
        "unsupported"."""
        p = self.params
        P = self.kf
        n_matches = 0
        n_inliers = 0
        reason = "error"
        R = None
        coverage = float("nan")
        try:
            kf_frame, cur = frames[P], frames[k]
            matches = match_frames(kf_frame, cur, cur.method)
            n_matches = len(matches)
            ia = [m.queryIdx for m in matches]
            ib = [m.trainIdx for m in matches]
            a = _bearings(kf_frame, ia)
            b = _bearings(cur, ib)
            sa = np.asarray(_keypoint_sigmas(kf_frame), dtype=np.float64)[np.asarray(ia, dtype=np.int64)]
            sb = np.asarray(_keypoint_sigmas(cur), dtype=np.float64)[np.asarray(ib, dtype=np.int64)]
            f = float(np.asarray(cur.K, dtype=np.float64)[0, 0])
            R, mask, reason = solve_rotation(a, b, sa, sb, f, p)
            n_inliers = 0 if reason == "matches" else int(np.sum(mask))
            if R is not None:
                T_P = np.asarray(kf_frame.cam_T_world, dtype=np.float64)
                c_P = np.asarray(kf_frame.world_T_cam, dtype=np.float64)[:3, 3]
                R_k = R @ T_P[:3, :3]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_k
                T[:3, 3] = -R_k @ c_P
                h, w = cur.img_shape[:2]
                coverage = float(view_coverage(cur.K, R, w, h, p))
                if not cur.set_camera_pose(T):
                    raise RuntimeError("set_camera_pose rejected the solved pose")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".replace("\r", " ").replace("\n", " ")
            print(f"  >> [ROT-ONLY-ERROR] frame={int(k)} error={msg}")
            R = None
            reason = "error"
            n_inliers = 0
            coverage = float("nan")

        self.n_frames += 1
        fr = frames[k]
        if R is not None:
            fr.pose_valid = True
            fr.tracking_state = "ROT"
            self.n_measured += 1
            print(f"  >> [ROT-ONLY] frame={int(k)} state=measured kf={int(P)} matches={int(n_matches)} "
                  f"inliers={int(n_inliers)} rel_rot_deg={_fmt(rotation_angle_deg(R))} "
                  f"coverage={_fmt(coverage)} reason=none")
            if coverage < p.min_coverage:
                self._new_keyframe(k)
            return "measured"

        c_P = np.asarray(frames[P].world_T_cam, dtype=np.float64)[:3, 3]
        R_pred = self._predicted_rotation(k, frames)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_pred
        T[:3, 3] = -R_pred @ c_P
        fr.set_camera_pose(T)
        fr.pose_valid = False
        fr.tracking_state = "LOST"
        self.n_unsupported += 1
        print(f"  >> [ROT-ONLY] frame={int(k)} state=unsupported kf={int(P)} matches={int(n_matches)} "
              f"inliers={int(n_inliers)} rel_rot_deg=nan coverage=nan reason={reason}")
        return "unsupported"

    def summary_line(self):
        n_points = -1 if self.world_map is None else len(self.world_map.points)
        return (f"  >> [ROT-SUMMARY] frames={int(self.n_frames)} measured={int(self.n_measured)} "
                f"unsupported={int(self.n_unsupported)} keyframes={int(self.n_kf)} "
                f"map_points={int(n_points)}")
