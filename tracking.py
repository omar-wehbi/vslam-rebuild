from dataclasses import dataclass

import cv2
import numpy as np

from keypoints_match import match_frames


@dataclass
class MotionEstimate:
    R: np.ndarray
    t: np.ndarray
    inlier_matches: list
    all_matches: list            # all descriptor matches (before RANSAC filtering)
    model: str
    inliers_E: int
    inliers_H: int
    epipolar_err_before: float   # mean epipolar error on raw descriptor matches
    epipolar_err_after: float    # mean epipolar error on inlier matches only


def compute_epipolar_error(pts1, pts2, E, K):
    """
    Compute the mean symmetric epipolar error (Sampson distance).

    Given the Essential matrix E and the camera intrinsic K, the
    Fundamental matrix is  F = K^{-T} E K^{-1}.  For a pair of
    corresponding points (x1, x2) in pixel coordinates the epipolar
    constraint says  x2^T F x1 = 0.  The *symmetric epipolar distance*
    normalises this by the gradient magnitude so the result is in pixels:

        d = |x2^T F x1| / sqrt( (Fx1)_0^2 + (Fx1)_1^2 + (F^Tx2)_0^2 + (F^Tx2)_1^2 )

    basically we're computing how far the points are from the epipolar line.

    Parameters
    ----------
    pts1, pts2 : np.ndarray, shape (N, 2)
        Matching pixel coordinates in image 1 and image 2.
    E : np.ndarray, shape (3, 3)
        Essential matrix.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix (shared by both views).

    Returns
    -------
    float
        Mean symmetric epipolar distance in pixels.
    """
    if len(pts1) == 0:
        return 0.0

    # Ensure E is 3x3 (OpenCV may return stacked matrices).
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 3 or E.shape[1] != 3:
        E = E[:3, :3].copy()

    # Fundamental matrix: F = K^{-T} E K^{-1}
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    # Convert to homogeneous coordinates: (N, 3)
    ones = np.ones((len(pts1), 1), dtype=np.float64)
    x1_h = np.hstack([pts1, ones])
    x2_h = np.hstack([pts2, ones])

    # Epipolar lines: Fx1 in image 2, F^T x2 in image 1
    Fx1 = (F @ x1_h.T).T            # (N, 3)
    Ftx2 = (F.T @ x2_h.T).T         # (N, 3)

    # Numerator: |x2^T F x1| per pair
    numerator = np.abs(np.sum(x2_h * Fx1, axis=1))

    # Denominator: Sampson distance normalization (symmetric)
    denom = np.sqrt(
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        + 1e-12
    )

    return float(np.mean(numerator / denom))


def rotation_angle_deg(R):
    """Rotation angle of R in degrees from the trace, clipped to [-1, 1] before arccos."""
    c = (float(np.trace(np.asarray(R, dtype=np.float64)[:3, :3])) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _essential_hypotheses(E):
    """
    The four decomposition hypotheses of E in the fixed order of Stage 03a, from
    cv2.decomposeEssentialMat(E) -> R1, R2, t (||t|| = 1):
        0: (R1, +t)   1: (R1, -t)   2: (R2, +t)   3: (R2, -t)
    Returns a list of (R (3,3) float64, t (3,) float64).
    """
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 3 or E.shape[1] != 3:
        E = E[:3, :3].copy()
    R1, R2, t = cv2.decomposeEssentialMat(E)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    R2 = np.asarray(R2, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def _normalized_coordinates(pts_prev, pts_curr, K):
    """Pixel correspondences -> normalized image coordinates (K^-1 applied), each 2 x N."""
    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    p1 = np.asarray(pts_prev, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts_curr, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((p1.shape[0], 1), dtype=np.float64)
    x1 = (K_inv @ np.hstack([p1, ones]).T)[:2]
    x2 = (K_inv @ np.hstack([p2, ones]).T)[:2]
    return x1, x2


def _cheirality_votes(hypotheses, x1_norm, x2_norm, near_thresh):
    """
    One triangulation-and-vote pass shared by select_essential_hypothesis (the selection,
    Stage 03b) and twoview_hypothesis_votes (the 03a diagnostic), so the two cannot disagree.

    For each hypothesis (R, t) triangulate with P1 = [I | 0], P2 = [R | t] (cv2.triangulatePoints
    on normalized coordinates), dehomogenize, and count
        votes_all[i]  : points with FINITE depth > 0 in camera 1 AND finite depth > 0 in camera 2,
                        where camera-2 depth is the third component of R X + t
        votes_near[i] : the subset of votes_all[i] with both depths < near_thresh (units ||t|| = 1),
                        mirroring the far-point cutoff of OpenCV's cheirality test
                        (distanceThresh = 50 in the overload estimate_motion used to call).
    Non-finite depths are guarded with np.errstate and never counted. Deterministic.
    """
    P1 = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    x1 = np.ascontiguousarray(x1_norm, dtype=np.float64)
    x2 = np.ascontiguousarray(x2_norm, dtype=np.float64)
    votes_all, votes_near = [], []
    for R, tt in hypotheses:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tt = np.asarray(tt, dtype=np.float64).reshape(3, 1)
        P2 = np.hstack([R, tt])
        Xh = cv2.triangulatePoints(P1, P2, x1, x2)  # 4 x N homogeneous
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            X = Xh[:3] / Xh[3]
            d1 = X[2]
            d2 = (R @ X + tt)[2]
            ok = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0) & (d2 > 0)
            near = ok & (d1 < near_thresh) & (d2 < near_thresh)
        votes_all.append(int(np.count_nonzero(ok)))
        votes_near.append(int(np.count_nonzero(near)))
    return votes_all, votes_near


def select_essential_hypothesis(E, pts_prev, pts_curr, K):
    """
    Stage 03b: choose among the four decomposition hypotheses of E by the full cheirality
    vote, with NO far-point cutoff (the reference method's disambiguation, orb-slam.md
    #initialization-and-model-selection, minus its refusal rules).

    Parameters
    ----------
    E : (3, 3) essential matrix (already reduced from findEssentialMat's output).
    pts_prev, pts_curr : (N, 2) pixel correspondences, N >= 5; the essential-matrix inliers.
        Camera 1 = prev_frame, camera 2 = curr_frame.
    K : (3, 3) intrinsics shared by both views.

    Returns
    -------
    R : (3, 3) float64 rotation; t : (3, 1) float64 unit translation, with x2 = R x1 + t
        (the cv2.recoverPose convention; t = +/- u3 of the SVD of E).
    votes_all, votes_near : lists of 4 ints (see _cheirality_votes); votes_near is recorded
        only and plays no part in the selection.
    chosen : int, the index of the largest votes_all; ties go to the LOWEST index
        (np.argmax). No ratio test, no minimum vote count: if every count is zero the
        result is hypothesis 0, (R1, +t).
    """
    hypotheses = _essential_hypotheses(E)
    x1, x2 = _normalized_coordinates(pts_prev, pts_curr, K)
    votes_all, votes_near = _cheirality_votes(hypotheses, x1, x2, 50.0)
    chosen = int(np.argmax(votes_all))
    R, t = hypotheses[chosen]
    R = np.array(R, dtype=np.float64).reshape(3, 3)
    t = np.array(t, dtype=np.float64).reshape(3, 1)
    return R, t, votes_all, votes_near, chosen


def twoview_hypothesis_votes(E, pts_prev, pts_curr, K, R_chosen, t_chosen, near_thresh=50.0):
    """
    Stage 03a diagnostic (spec D): score the four essential-matrix decomposition hypotheses.

    Hypotheses, in the fixed order of cv2.decomposeEssentialMat(E) -> R1, R2, t:
        0: (R1, +t)   1: (R1, -t)   2: (R2, +t)   3: (R2, -t)
    Each is scored by triangulating the SAME correspondences that were used for the selection,
    in normalized coordinates (K^-1 applied), with P1 = [I | 0] and P2 = [R | t]:
        votes_all  : points with depth > 0 in camera 1 AND depth > 0 in camera 2
        votes_near : subset of votes_all with both depths < near_thresh (units: ||t|| = 1),
                     mirroring OpenCV's cheirality test with its far-point cutoff
                     (distanceThresh = 50 in the overload estimate_motion used to call).
    chosen is the hypothesis index whose (R, t) equals (R_chosen, t_chosen) within 1e-6
    (Frobenius norm on R, vector norm on t), or None. rot_deg is the angle of R_chosen.
    Only deterministic OpenCV functions are used; nothing here draws random numbers.
    The vote pass is _cheirality_votes, shared with select_essential_hypothesis (Stage 03b).
    """
    hypotheses = _essential_hypotheses(E)
    x1, x2 = _normalized_coordinates(pts_prev, pts_curr, K)
    votes_all, votes_near = _cheirality_votes(hypotheses, x1, x2, near_thresh)

    chosen = None
    Rc = np.asarray(R_chosen, dtype=np.float64).reshape(3, 3)
    tc = np.asarray(t_chosen, dtype=np.float64).reshape(3)
    for i, (R, tt) in enumerate(hypotheses):
        if np.linalg.norm(R - Rc) < 1e-6 and np.linalg.norm(tt - tc) < 1e-6:
            chosen = i
            break
    return dict(votes_all=votes_all, votes_near=votes_near, chosen=chosen, rot_deg=rotation_angle_deg(Rc))


def format_twoview_line(frame_id, n_inliers, rot_deg, t, votes):
    """
    The [TWOVIEW] log line (03a spec D; 03b spec B). `votes` is a dict with the keys of
    twoview_hypothesis_votes (votes_all, votes_near, chosen).

    `frame_id` is printed verbatim as frame=<k>. Since Stage 03b the caller passes the
    TRAJECTORY INDEX (curr_frame.id - 1, the k that [POSE-ONLINE] and [CORRECTION] print),
    not Frame.id; 03a printed Frame.id, which put the per-pair CSV of jumps.py off by one row.
    """
    tv = np.asarray(t, dtype=np.float64).reshape(-1)
    va = "/".join(str(int(v)) for v in votes["votes_all"])
    vn = "/".join(str(int(v)) for v in votes["votes_near"])
    chosen = votes.get("chosen")
    chosen_s = "none" if chosen is None else str(int(chosen))
    return (
        f"  >> [TWOVIEW] frame={int(frame_id)} inliers={int(n_inliers)} rot_deg={float(rot_deg):.4f} "
        f"t_cam={tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f} votes_all={va} votes_near={vn} chosen={chosen_s}"
    )


def _print_twoview_diagnostic(E, pts_prev_inline, pts_curr_inline, K, R, t, frame_id):
    """
    03a helper: re-score the hypotheses for an externally chosen (R, t) and print the [TWOVIEW]
    line; never raises, never touches R, t or the returned estimate. Since Stage 03b
    estimate_motion no longer uses it (the line there is fed from the selection's own vote
    pass, see _print_twoview_from_selection); it is kept for tests and offline diagnostics.
    """
    try:
        votes = twoview_hypothesis_votes(E, pts_prev_inline, pts_curr_inline, K, R, t)
    except Exception:
        try:
            rot_deg = rotation_angle_deg(R)
        except Exception:
            rot_deg = float("nan")
        votes = dict(votes_all=[0, 0, 0, 0], votes_near=[0, 0, 0, 0], chosen=None, rot_deg=rot_deg)
    try:
        print(format_twoview_line(frame_id, len(pts_prev_inline), votes["rot_deg"], t, votes))
    except Exception:
        pass


def _print_twoview_from_selection(frame_index, n_inliers, R, t, votes_all, votes_near, chosen):
    """
    Stage 03b (spec B): print the [TWOVIEW] line from the result of select_essential_hypothesis
    (one triangulation pass per frame; no re-scoring). frame_index is the trajectory index.
    Never raises; a print failure must not change the returned estimate.
    """
    try:
        votes = dict(votes_all=votes_all, votes_near=votes_near, chosen=int(chosen))
        print(format_twoview_line(frame_index, n_inliers, rotation_angle_deg(R), t, votes))
    except Exception:
        pass


def estimate_motion(
    prev_frame,
    curr_frame,
    ratio_threshold=1.0,
    min_h_inlier_margin=10,
    min_h_inliers=50,
    ransac_threshold=1.5,
    use_homography=True,
):
    """
    use_homography: If True, compute homography and choose between H and E based on
    inlier thresholds. If False, always use essential matrix only.
    ransac_threshold: use 1.5--2.0 for small inter-frame motion / shaky sequences.
    """
    matches = match_frames(prev_frame, curr_frame, curr_frame.method)
    if len(matches) < 5:
        return None
    ''' get the keypoints of the previous and current frame.'''
    pts_prev = np.array(
        [prev_frame.KeyPoints[m.queryIdx].pt for m in matches], dtype=np.float64
    )
    pts_curr = np.array(
        [curr_frame.KeyPoints[m.trainIdx].pt for m in matches], dtype=np.float64
    )
    ''' find the essential matrix using the RANSAC algorithm.'''
    E, mask_E = cv2.findEssentialMat(
        pts_prev,
        pts_curr,
        cameraMatrix=curr_frame.K,
        method=cv2.RANSAC,
        threshold=ransac_threshold,
    )
    if E is None or mask_E is None:
        return None

    # OpenCV can return multiple essential matrices stacked (e.g. shape (18, 3)); use first 3x3.
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 3 or E.shape[1] != 3:
        E = E[:3, :3].copy()

    # --- Epipolar error BEFORE filtering (all descriptor matches) ---
    # Uses the Essential matrix estimated by RANSAC, evaluated on every
    # match including outliers.  A high value here is expected because
    # many raw matches are wrong.
    epipolar_err_before = compute_epipolar_error(pts_prev, pts_curr, E, curr_frame.K)

    H = None
    mask_H = None
    ''' if use_hemography is enabled, find the homography matrix using the RANSAC algorithm.'''
    if use_homography and len(matches) >= 4:
        H, mask_H = cv2.findHomography(
            pts_prev, pts_curr, cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

    inliers_E = int(np.sum(mask_E)) # count the number of inliers in the essential matrix.
    inliers_H = int(np.sum(mask_H)) if mask_H is not None else 0 # count the number of inliers in the homography matrix.
    h_ratio = (inliers_H / inliers_E) if inliers_E > 0 else 0.0 # ratio of inliers in the homography matrix to the essential matrix.
    
    ''' 5 conditions to use homography. it usually fails all the time because E is at least just as 
    good except in very rare cases. kept for illustration purposes'''
    take_homography_path = (
        use_homography
        and H is not None
        and inliers_E > 0
        and inliers_H >= min_h_inliers
        and h_ratio >= ratio_threshold
        and (inliers_H - inliers_E) >= min_h_inlier_margin
    )
    ''' if the homography path is taken, compute the epipolar error on the inlier set.'''
    if take_homography_path and H is not None:
        K_inv = np.linalg.inv(curr_frame.K)
        R = K_inv @ H @ curr_frame.K
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        t = np.zeros((3, 1), dtype=np.float64)

        # For homography frames, compute epipolar error on the inlier set
        inline_H = mask_H.ravel().astype(bool) if mask_H is not None else np.zeros(len(matches), dtype=bool)
        epipolar_err_after = compute_epipolar_error(
            pts_prev[inline_H], pts_curr[inline_H], E, curr_frame.K
        )
        #inlier_matches is empty because in pure rotation t=0 so no triangulation.
        return MotionEstimate(
            R, t, [], list(matches), "HOMOGRAPHY", inliers_E, inliers_H,
            epipolar_err_before, epipolar_err_after
        )

    inline = mask_E.ravel().astype(bool) # get the inliers in the essential matrix.
    if np.count_nonzero(inline) < 5: # if there are less than 5 inliers, return None.   (for safety)  
        return None
    pts_prev_inline = pts_prev[inline] # get the inliers in the previous frame.
    pts_curr_inline = pts_curr[inline] # get the inliers in the current frame.

    # --- Epipolar error AFTER filtering (Essential Matrix inliers only) ---
    # Should be much lower than before, since outliers were removed by RANSAC.
    epipolar_err_after = compute_epipolar_error(
        pts_prev_inline, pts_curr_inline, E, curr_frame.K
    )

    # Stage 03b (spec A): select the decomposition hypothesis of E by the full cheirality vote on
    # the same inliers, without a far-point cutoff. x2 = R x1 + t, camera 1 = prev_frame,
    # camera 2 = curr_frame, t a (3,1) unit vector (the convention the old OpenCV call used).
    # If the selector raises anyway (spec A4), print no [TWOVIEW] line and take the existing
    # VO-failure path.
    try:
        R, t, votes_all, votes_near, chosen = select_essential_hypothesis(
            E, pts_prev_inline, pts_curr_inline, curr_frame.K
        )
    except Exception:
        return None
    # [TWOVIEW] (spec B): fed from the same vote pass; frame= is the trajectory index
    # (curr_frame.id - 1), aligned with [POSE-ONLINE] and the jumps.py per-pair CSV.
    _print_twoview_from_selection(
        curr_frame.id - 1, len(pts_prev_inline), R, t, votes_all, votes_near, chosen
    )
    inlier_matches = [m for m, keep in zip(matches, inline) if keep]
    return MotionEstimate(
        R, t, inlier_matches, list(matches), "ESSENTIAL_MATRIX", inliers_E, inliers_H,
        epipolar_err_before, epipolar_err_after
    )
