"""
Stage 04d (foundation repair; conventions.md#camera-model-and-distortion): the camera
model is K plus radial/tangential distortion coefficients (k1, k2, p1, p2, k3), not K
alone. Keypoint coordinates are undistorted ONCE at extraction (Frame.__init__) with
cv2.undistortPoints(..., K, D, P=K), so every downstream step (two-view RANSAC, PnP,
triangulation, BA, local map tracking, export, log lines) stays pinhole in pixel units
and is untouched by this module. Images themselves are never undistorted on the
product path; depth-image lookups are the harness's business (system/verify).

Provenance: TUM RGB-D freiburg2 intrinsics (k1 k2 p1 p2 k3), recorded 2026-09-19,
conventions.md#camera-model-and-distortion. OpenCV's order: (k1, k2, p1, p2, k3).
"""
import numpy as np
import cv2

DISTORTION_FR2 = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172], dtype=np.float64)


def undistort_points(xy, K, D):
    """xy: (N, 2) raw pixels -> (N, 2) undistorted pixels with P = K
    (cv2.undistortPoints, the same call the harness mirrors). N == 0 returns (0, 2)."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = xy.reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    und = cv2.undistortPoints(pts, K, D, P=K)
    return und.reshape(-1, 2)
