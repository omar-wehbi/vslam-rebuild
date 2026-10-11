"""
PnP-based localization module.

Every N frames we re-localize the camera against the existing 3D map
instead of relying solely on accumulated relative motion.  This corrects
drift that builds up from chaining frame-to-frame Essential-Matrix poses.

Algorithm
---------
1.  Collect 2D-3D correspondences: for each keypoint in the current frame
    that is linked to a map point, check that the point is *well-established*
    (observed in >= 3 frames).  Points triangulated from only the current
    frame-pair are excluded because they were created using the very pose
    we are trying to correct (circular dependency).
2.  Run cv2.solvePnPRansac to robustly estimate the camera pose (R, t)
    that minimizes reprojection error.
3.  If enough inliers support the solution, return the recovered pose;
    otherwise fall back to the regular relative-motion estimate.
"""

import time
import cv2
import numpy as np

# OpenCV PnP method flags for RANSAC. Used when solver is a single name or in "multi" mode.
PNP_SOLVERS = {
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "ap3p": cv2.SOLVEPNP_AP3P,
    "epnp": cv2.SOLVEPNP_EPNP,
    "sqpnp": cv2.SOLVEPNP_SQPNP,
}


def _reprojection_error_rvec_tvec(pts_3d, pts_2d, K, rvec, tvec):
    """Mean reprojection error in pixels for the given pose (rvec, tvec)."""
    projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)
    diff = np.linalg.norm(pts_2d - projected, axis=1)
    return float(np.mean(diff))


def pnp_localize(
    frame,
    world_map,
    min_correspondences=20,
    min_inlier_ratio=0.55,
    solver="iterative",
    refinement=True,
    use_extrinsic_guess=True,
    multi_inlier_gate=0.8,
    reprojection_error=3.0,
    stats=None,
    outlier_hit_accounting=False,
):
    """
    Estimate the absolute camera pose of *frame* from known 3D map points.

    Parameters
    ----------
    frame : Frame
        The current frame whose pose we want to refine.  Its
        ``keypoint_to_point`` dict must already contain some entries
        (populated by triangulation in earlier frames).
    world_map : Map
        The global map holding all triangulated 3D points.
    min_correspondences : int
        Minimum number of 2D-3D pairs needed to attempt PnP.
    min_inlier_ratio : float
        Minimum fraction of correspondences that must be RANSAC inliers
        to accept the pose (default 0.55). Stricter (e.g. 0.65) rejects
        more bad poses but can reject good ones when matches are noisy.
    solver : str
        One of: "iterative" (default, single run with SOLVEPNP_ITERATIVE; timing
        logged; refinement is skipped as redundant), "multi" (run all PNP_SOLVERS,
        pick best by reprojection error), or a single solver name: "ap3p", "epnp", "sqpnp".
    refinement : bool
        If True, refine the chosen pose with solvePnPRefineLM on inliers.
        Skipped when solver is "iterative" (already iterative LM).
    use_extrinsic_guess : bool
        If True and the inner method supports it (iterative), use the
        current frame pose as initial guess. If False, do not pass a guess
        (for comparison / debugging).
    multi_inlier_gate : float
        When solver == "multi", only consider candidates with inlier count
        >= multi_inlier_gate * max_inlier_count (default 0.8). Among those,
        pick the one with lowest mean reprojection error. If none pass the
        gate, all candidates are considered.
    reprojection_error : float
        Maximum reprojection error in pixels for a point to be considered an
        inlier in RANSAC (default 3.0). Larger values accept noisier matches.
    stats : dict | None
        If provided and solver == "multi", each (solver_name, elapsed_sec)
        is appended to stats.setdefault("pnp_multi_solver_times", []).

    Returns
    -------
    success : bool
        True if PnP produced a valid pose with enough inliers.
    cam_T_world : np.ndarray | None
        4x4 camera-from-world transformation, or None on failure.
    n_inliers : int
        Number of RANSAC inliers that supported the solution.
    solver_used : str | None
        When solver == "multi", the name of the solver that won (e.g. "epnp").
        Otherwise None (single-solver mode does not report a name for logging).
    """

    # --- Step 1: gather 2D-3D correspondences from the map ---
    pts_2d = []
    pts_3d = []
    corr_point_ids = []
    
    'iterate over the keypoint_to_point dictionary of the current frame and get the 2D and 3D points.'
    for kp_idx, point_id in frame.keypoint_to_point.items(): 
        pt = world_map.get_point(point_id)
        if pt is None:# if you cannot find the keypoint of the current frame in the world map, skip the rest of the loop.
            continue
        if len(pt.frames) < 3: # if the point is not well-established ( has less than 3 observations), skip the rest of the loop.
            continue
        pts_2d.append(frame.KeyPoints[kp_idx].pt) #append the 2D keypoint location to the pts_2d list.
        pts_3d.append(pt.point) #append the 3D point location to the pts_3d list.
        corr_point_ids.append(point_id)

    n_corr = len(pts_2d) #number of correspondences.
    n_links_total = len(frame.keypoint_to_point)
    if n_corr < min_correspondences:
        if stats is not None:
            stats.setdefault("pnp_fail_reasons", {})
            stats["pnp_fail_reasons"]["too_few_correspondences"] = stats["pnp_fail_reasons"].get("too_few_correspondences", 0) + 1
            stats["pnp_min_correspondences"] = min_correspondences
            # Count when the 3-observation filter caused the skip (enough links but too few well-established)
            if n_links_total >= min_correspondences:
                stats["pnp_fail_reasons"]["too_few_well_established"] = stats["pnp_fail_reasons"].get("too_few_well_established", 0) + 1
                stats["pnp_min_observations"] = 3
        print(
            f"  >> PnP: {n_links_total} map links but only {n_corr} well-established "
            f"(need ≥{min_correspondences}) — skipping PnP"
        )
        return False, None, 0, None
    'convert the pts_2d and pts_3d lists to numpy arrays.'
    pts_2d = np.array(pts_2d, dtype=np.float64)
    pts_3d = np.array(pts_3d, dtype=np.float64)

    init_rvec, _ = cv2.Rodrigues(frame.cam_T_world[:3, :3]) #convert the rotation matrix to a rotation vector.
    init_tvec = frame.cam_T_world[:3, 3].reshape(3, 1).astype(np.float64) #convert the translation vector to a 3x1 numpy array.
    use_guess = use_extrinsic_guess #if use_extrinsic_guess is True, use the current frame pose as initial guess.

    'define a function to run the pnp localization with a single solver.'
    def run_one(name, flag):
        use_guess_here = use_guess and (name == "iterative") #if use_extrinsic_guess is True and the solver is iterative, use the current frame pose as initial guess.
        return cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=frame.K,
            distCoeffs=None,
            iterationsCount=500, #number of iterations ransac will run.
            reprojectionError=reprojection_error, #maximum reprojection error in pixels for a point to be considered an inlier in RANSAC.
            confidence=0.999,
            useExtrinsicGuess=use_guess_here,
            rvec=init_rvec if use_guess_here else None, #if use_extrinsic_guess is True and the solver is iterative, use the current frame pose as initial guess.
            tvec=init_tvec if use_guess_here else None, #if use_extrinsic_guess is True and the solver is iterative, use the current frame pose as initial guess.
            flags=flag,
        )

    success = False
    rvec = tvec = inliers = None
    solver_used = None

    if solver in PNP_SOLVERS:
        # Single named solver.
        ''' success is a boolean value that indicates if the pnp localization was successful.
        rvec is a 3x1 numpy array that contains the rotation vector.
        tvec is a 3x1 numpy array that contains the translation vector.
        inliers is a numpy array that contains the indices of the inliers.'''
        success, rvec, tvec, inliers = run_one(solver, PNP_SOLVERS[solver])
        solver_used = solver
    elif solver == "multi":
        
        candidates = [] #list to store the candidates.
        multi_solver_times = [] #list to store the time it took to run each solver.
        '''iterate over the PNP_SOLVERS dictionary and run each solver.  store the time it took to run each solver in the multi_solver_times list.
        if the solver is successful and the inliers is not None and the number of inliers is greater than or equal to the minimum number of correspondences,
        store the number of inliers, the reprojection error, the name of the solver, the rotation vector, the translation vector, and the inliers in the candidates list.'''
        for name, flag in PNP_SOLVERS.items():
            t0 = time.perf_counter()
            ok, r, t, inl = run_one(name, flag)
            elapsed = time.perf_counter() - t0
            multi_solver_times.append((name, elapsed))
            if ok and inl is not None and len(inl) >= min_correspondences:
                n_in = len(inl)
                ratio = n_in / n_corr
                if ratio >= min_inlier_ratio: #if the inlier ratio is greater than or equal to the minimum inlier ratio, store the number of inliers, the reprojection error, the name of the solver, the rotation vector, the translation vector, and the inliers in the candidates list.
                    err = _reprojection_error_rvec_tvec(
                        pts_3d[inl.ravel()], pts_2d[inl.ravel()], frame.K, r, t
                    )
                    candidates.append((n_in, err, name, r, t, inl))
        if not candidates: #if all solvers failed or below threshold, return False, None, 0, None.
            if stats is not None:
                stats.setdefault("pnp_fail_reasons", {})
                stats["pnp_fail_reasons"]["multi_all_failed"] = stats["pnp_fail_reasons"].get("multi_all_failed", 0) + 1
            print(f"  >> PnP: multi — all solvers failed or below threshold ({n_corr} correspondences)")
            return False, None, 0, None
        ''' inliner gate: only consider candidates with inliers >= multi_inlier_gate * max_inlier_count. max_inlier_count is the maximum number of inliers in all candidates.'''
        max_inlier_count = max(c[0] for c in candidates)
        gate_threshold = multi_inlier_gate * max_inlier_count
        gate_candidates = [c for c in candidates if c[0] >= gate_threshold]
        if not gate_candidates:
            gate_candidates = candidates
        # Best = lowest mean reprojection error; tiebreaker: prefer iterative.
        def key(c):
            _, err, name, _, _, _ = c
            return (err, 0 if name == "iterative" else 1)
        best = min(gate_candidates, key=key) #best solver is the one with the lowest mean reprojection error.
        _, _, solver_used, rvec, tvec, inliers = best
        success = True
        if stats is not None:
            stats.setdefault("pnp_multi_solver_times", []).extend(multi_solver_times)
        print(f"  >> PnP: multi chose {solver_used} ({len(inliers)} inliers)")
    else:
        if stats is not None:
            stats.setdefault("pnp_fail_reasons", {})
            stats["pnp_fail_reasons"]["unknown_solver"] = stats["pnp_fail_reasons"].get("unknown_solver", 0) + 1
        print(f"  >> PnP: unknown solver '{solver}'")
        return False, None, 0, None

    if not success or inliers is None:
        if stats is not None:
            stats.setdefault("pnp_fail_reasons", {})
            stats["pnp_fail_reasons"]["ransac_failed"] = stats["pnp_fail_reasons"].get("ransac_failed", 0) + 1
        print(f"  >> PnP: RANSAC failed ({n_corr} correspondences)")
        return False, None, 0, solver_used

    n_inliers = len(inliers)
    inlier_ratio = n_inliers / n_corr

    # Optional bookkeeping for pruning policy E: increment outlier_hits for correspondences
    # that were rejected by RANSAC (non-inliers). Conservative: only on success.
    if outlier_hit_accounting and corr_point_ids:
        try:
            inl_idx = set(int(i) for i in inliers.ravel().tolist())
            for j, pid in enumerate(corr_point_ids):
                if j in inl_idx:
                    continue
                pt = world_map.get_point(pid)
                if pt is None:
                    continue
                try:
                    pt.outlier_hits = int(getattr(pt, "outlier_hits", 0)) + 1
                except Exception:
                    pt.outlier_hits = 1
        except Exception:
            pass

    ''' if the number of inliers is less than the minimum number of correspondences or the inlier ratio is less than the minimum inlier ratio, return False, None, n_inliers, solver_used.'''
    if n_inliers < min_correspondences or inlier_ratio < min_inlier_ratio:
        reasons = []
        if stats is not None:
            stats.setdefault("pnp_fail_reasons", {})
        if n_inliers < min_correspondences:
            reasons.append(f"inliers {n_inliers} < {min_correspondences}")
            if stats is not None:
                stats["pnp_fail_reasons"]["inliers_below_min"] = stats["pnp_fail_reasons"].get("inliers_below_min", 0) + 1
                stats["pnp_min_correspondences"] = min_correspondences
        if inlier_ratio < min_inlier_ratio:
            reasons.append(f"ratio {inlier_ratio:.2f} < {min_inlier_ratio}")
            if stats is not None:
                stats["pnp_fail_reasons"]["ratio_below_threshold"] = stats["pnp_fail_reasons"].get("ratio_below_threshold", 0) + 1
                stats["pnp_min_inlier_ratio"] = min_inlier_ratio
        print(
            f"  >> PnP: {n_corr} correspondences, {n_inliers} inliers "
            f"(ratio {inlier_ratio:.2f}) — rejecting ({'; '.join(reasons)})"
        )
        return False, None, n_inliers, solver_used
    ''' if refinement is True and the solver is not iterative, run solvePnPRefineLM on the inliers.'''
    if refinement and solver != "iterative":
        # Skip RefineLM for iterative (already uses LM); apply for ap3p/epnp/sqpnp and multi winner.
        try:
            idx = inliers.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=pts_3d[idx],
                imagePoints=pts_2d[idx],
                cameraMatrix=frame.K,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )
        except AttributeError:
            pass

    R, _ = cv2.Rodrigues(rvec)
    cam_T_world = np.eye(4, dtype=np.float64)
    cam_T_world[:3, :3] = R
    cam_T_world[:3, 3] = tvec.flatten()

    return True, cam_T_world, n_inliers, solver_used #return True, the camera-from-world transformation matrix, the number of inliers, and the name of the solver used.
