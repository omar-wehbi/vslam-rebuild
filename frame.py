import time
import cv2
import numpy as np

import camera
from geometry import rt_to_t_inv
from localization import pnp_localize
from mapping import associate_known_points, local_map_track
from pose_log import print_correction
from tracking import estimate_motion

# Default camera intrinsics (edit here when using --default_k).
# Used when main.py is run with --default_k instead of inferring from image size.
DEFAULT_K = np.array([
    [520.9,   0.0, 325.1],
    [  0.0, 521.0, 249.7],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

FRAME_ID = 1


def _estimate_translation_scale(prev_frame, curr_frame, R, t_unit, inlier_matches, world_map,
                                prev_step=None,
                                _MIN_POINTS=5, _MIN_OBS=3,
                                _SCALE_REL_LO=0.5, _SCALE_REL_HI=2.0,
                                _SCALE_ABS_LO=0.2, _SCALE_ABS_HI=5.0):
    """
    Estimate the scale of the unit translation returned by recoverPose
    using 3D map points already triangulated in previous frames.

    For each well-established map point visible in prev_frame that also
    has a 2D match in curr_frame, the constraint
        R @ X_prev + s * t = lambda * ray
    (the transformed point must lie on the backprojected ray) yields one
    scale estimate.  Outliers are rejected with an IQR filter, then the
    median is taken and clamped.

    Stage 2 hardening: once a previous accepted step length ``prev_step``
    exists, all early-exit / failure paths return that value (never bare
    1.0).  Successful estimates are clamped to
    [prev_step * _SCALE_REL_LO, prev_step * _SCALE_REL_HI].

    Returns
    -------
    (scale, reason) : (float, str)
        scale is the factor applied to unit ``t``; reason documents accept
        or fallback path for logging.
    """
    def _fallback(reason):
        if prev_step is not None and float(prev_step) > 0.0:
            return float(prev_step), reason
        return 1.0, f"{reason}_init_unit"

    if world_map is None or len(inlier_matches) == 0:
        return _fallback("no_map_or_matches")

    K_inv = np.linalg.inv(curr_frame.K)
    t_flat = np.asarray(t_unit, dtype=np.float64).flatten()

    # --- Collect candidate 3D points and their depths in prev camera ---
    candidates = []
    for m in inlier_matches:
        pt_id = prev_frame.keypoint_to_point.get(m.queryIdx)
        if pt_id is None:
            continue
        pt = world_map.get_point(pt_id)
        if pt is None or len(pt.frames) < _MIN_OBS:
            continue

        P_hom = np.array([pt.point[0], pt.point[1], pt.point[2], 1.0])
        X = (prev_frame.cam_T_world @ P_hom)[:3]
        if X[2] <= 0.1:
            continue
        candidates.append((X, m))

    if len(candidates) < _MIN_POINTS:
        return _fallback("insufficient_candidates")

    # --- Depth-range filter: reject very far / very close outlier points ---
    depths = np.array([c[0][2] for c in candidates])
    med_depth = float(np.median(depths))
    if med_depth <= 0:
        return _fallback("bad_median_depth")
    depth_lo = 0.1 * med_depth
    depth_hi = 10.0 * med_depth
    candidates = [c for c in candidates if depth_lo <= c[0][2] <= depth_hi]

    if len(candidates) < _MIN_POINTS:
        return _fallback("insufficient_after_depth_filter")

    # --- Per-point scale from cross-product constraint ---
    scales = []
    for X, m in candidates:
        px = np.array([
            curr_frame.KeyPoints[m.trainIdx].pt[0],
            curr_frame.KeyPoints[m.trainIdx].pt[1],
            1.0,
        ])
        ray = K_inv @ px

        RX = R @ X
        cross_RX_ray = np.cross(RX, ray)
        cross_t_ray = np.cross(t_flat, ray)

        best = int(np.argmax(np.abs(cross_t_ray)))
        if abs(cross_t_ray[best]) < 1e-8:
            continue

        s = -cross_RX_ray[best] / cross_t_ray[best]
        if s > 0:
            scales.append(s)

    if len(scales) < _MIN_POINTS:
        return _fallback("insufficient_positive_scales")

    scales = np.array(scales)

    # --- IQR outlier rejection ---
    q1, q3 = np.percentile(scales, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        scales = scales[(scales >= lo) & (scales <= hi)]

    if len(scales) < 3:
        return _fallback("insufficient_after_iqr")

    scale = float(np.median(scales))

    # --- Accept / reject relative to prev_step (Stage 2 soft rejection) ---
    # Clamping *to* the relative bound allowed unbounded doubling when estimates
    # were systematically high (ok_clamped_rel_prev cascade). Soft-reject instead.
    if prev_step is not None and float(prev_step) > 0.0:
        lo = float(prev_step) * _SCALE_REL_LO
        hi = float(prev_step) * _SCALE_REL_HI
        if scale < lo or scale > hi:
            return float(prev_step), "rejected_rel_prev"
        # Outer absolute safety belt (map/BA can still invent wild scales).
        if scale < _SCALE_ABS_LO or scale > _SCALE_ABS_HI:
            return float(prev_step), "rejected_abs_band"
        return scale, "ok"
    # First frames (no history): allow absolute init band once.
    if scale < _SCALE_ABS_LO or scale > _SCALE_ABS_HI:
        return _fallback("rejected_abs_band")
    return scale, "ok_init"

class Frame:
    def __init__(self, img, K, method, match_type, detector_config=None, distortion=None):
        cfg = detector_config or {}

        # Ensure grayscale input for feature extraction stability.
        if img is None:
            raise RuntimeError("Frame received empty image")
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        self.img_shape = tuple(img_gray.shape[:2])

        if method == "SIFT":
            sift_cfg = (cfg.get("sift") or {})
            detector = cv2.SIFT_create(
                nfeatures=int(sift_cfg.get("nfeatures", 0)),
                nOctaveLayers=int(sift_cfg.get("nOctaveLayers", 3)),
                contrastThreshold=float(sift_cfg.get("contrastThreshold", 0.04)),
                edgeThreshold=float(sift_cfg.get("edgeThreshold", 10.0)),
                sigma=float(sift_cfg.get("sigma", 1.6)),
            )
        elif method == "ORB":
            orb_cfg = (cfg.get("orb") or {})
            detector = cv2.ORB_create(
                nfeatures=int(orb_cfg.get("nfeatures", 500)),
                scaleFactor=float(orb_cfg.get("scaleFactor", 1.2)),
                nlevels=int(orb_cfg.get("nlevels", 8)),
            )
        else:
            raise RuntimeError("pick a valid method")

        self.KeyPoints, self.Descriptors = detector.detectAndCompute(img_gray, None)

        # Stage 04d (spec B2(a)): the raw pixel is always recoverable, with the switch
        # on or off, without inverting the distortion polynomial.
        if self.KeyPoints:
            self.keypoint_xy_raw = np.array([kp.pt for kp in self.KeyPoints], dtype=np.float64)
        else:
            self.keypoint_xy_raw = np.zeros((0, 2), dtype=np.float64)

        # Stage 04d (spec B2(b)/B4): undistort keypoint coordinates once at extraction
        # so every downstream consumer of KeyPoints[i].pt stays pinhole, unchanged.
        if distortion is not None:
            if K is None:
                raise ValueError("Frame: K is required when distortion is not None")
            if len(self.KeyPoints) > 0:
                und = camera.undistort_points(self.keypoint_xy_raw, K, distortion)
                for kp, (u, v) in zip(self.KeyPoints, und):
                    kp.pt = (float(u), float(v))

        global FRAME_ID
        self.id = FRAME_ID
        FRAME_ID += 1

        self.img = img_gray
        self.world_T_cam = np.eye(4, dtype=np.float64)
        self.cam_T_world = np.eye(4, dtype=np.float64)
        self.K = K
        self.method = method
        self.match_type = match_type
        self.detector_config = cfg
        self.distortion = distortion
        # keypoint_idx -> point_id in global map
        self.keypoint_to_point = {}
        # Stage 2: tracking integrity — LOST frames keep a held/CV pose but are not
        # treated as valid for BA / triangulation / keyframe promotion.
        self.pose_valid = True
        self.tracking_state = "OK"

        # Optional debug print (keypoint count and size stats).
        if cfg.get("debug_features", False):
            every = int(cfg.get("debug_features_every", 30)) if cfg.get("debug_features_every", 30) else 30
            if every <= 0:
                every = 30
            if (self.id == 1) or (self.id % every == 0):
                sizes = np.array([kp.size for kp in (self.KeyPoints or [])], dtype=np.float64)
                if sizes.size > 0:
                    msg = (
                        f"  >> [FEATURES] frame {self.id} ({self.method}) "
                        f"kp={len(self.KeyPoints)} size[min/med/max]="
                        f"{float(np.min(sizes)):.1f}/{float(np.median(sizes)):.1f}/{float(np.max(sizes)):.1f}"
                    )
                else:
                    msg = f"  >> [FEATURES] frame {self.id} ({self.method}) kp=0"
                print(msg)

    def release_image(self):
        """Release pixel buffer to reduce memory; keeps `img_shape`."""
        self.img = None

    @property
    def pose(self):
        return self.world_T_cam

    @pose.setter
    def pose(self, world_T_cam):
        # Keep the two pose representations consistent.
        self.set_world_pose(world_T_cam)

    @staticmethod
    def _sanitize_pose(T):
        """
        Make pose numerically safe:
        - ensure finite values
        - project rotation to nearest proper SO(3) via SVD
        - keep translation as-is
        """
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got {T.shape}")
        if not np.all(np.isfinite(T)):
            raise ValueError("Pose contains NaN/inf")

        R = T[:3, :3]
        t = T[:3, 3]

        # Project to nearest rotation.
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1.0
            Rn = U @ Vt

        Tn = np.eye(4, dtype=np.float64)
        Tn[:3, :3] = Rn
        Tn[:3, 3] = t
        return Tn

    def set_world_pose(self, world_T_cam):
        try:
            T = Frame._sanitize_pose(world_T_cam)
        except Exception as e:
            # Keep last valid pose instead of crashing the whole pipeline.
            print(f"  >> [POSE] Warning: rejected invalid world_T_cam for frame {getattr(self, 'id', '?')}: {e}")
            return False
        self.world_T_cam = np.array(T, dtype=np.float64, copy=True)
        self.cam_T_world = np.linalg.inv(self.world_T_cam)
        return True

    def set_camera_pose(self, cam_T_world):
        try:
            T = Frame._sanitize_pose(cam_T_world)
        except Exception as e:
            print(f"  >> [POSE] Warning: rejected invalid cam_T_world for frame {getattr(self, 'id', '?')}: {e}")
            return False
        self.cam_T_world = np.array(T, dtype=np.float64, copy=True)
        self.world_T_cam = np.linalg.inv(self.cam_T_world)
        return True

    def update_pose(self, prev_frame, world_map=None, use_pnp=False, use_homography=True,
                    t_norm_log_file=None, recent_frames=None, stats=None,
                    pnp_min_inlier_ratio=0.55,
                    pnp_solver="iterative", pnp_refinement=True, pnp_use_initial_guess=True,
                    pnp_multi_inlier_gate=0.8,
                    pnp_reprojection_error=3.0,
                    pnp_adaptive_gate=False, pnp_adaptive_gate_factor=0.2,
                    use_relative_scale=True,
                    outlier_hit_accounting=False,
                    pnp_correction_window=None,
                    local_map_search_radius=25.0,
                    local_map_max_desc_dist=None,
                    local_map_max_depth_ratio=5.0,
                    oracle=None):
        """
        Estimate this frame's pose relative to prev_frame, optionally
        refining with PnP localization against existing map points.

        Parameters
        ----------
        prev_frame : Frame
            The previous frame (already has a valid pose).
        world_map : Map | None
            The global 3D point map.
        use_pnp : bool
            If True, attempt PnP re-localization after computing the
            initial relative pose.  PnP uses 2D-3D correspondences
            between the current frame's keypoints and known map points
            to recover an *absolute* pose, correcting accumulated drift.
        use_homography : bool
            If True, choose between homography and essential matrix based on
            inlier thresholds. If False, always use essential matrix only.
        t_norm_log_file : file | None
            If set, write one line: frame_id and np.linalg.norm(t) from recoverPose
            (0.0 for homography) for scale/debugging.
        recent_frames : list[Frame] | None
            If provided, map points from these frames are projected into
            the current image and matched by descriptor, enriching
            keypoint_to_point before PnP.
        stats : dict | None
            If provided, updated with keys 'homography', 'essential',
            'pnp_success', 'pnp_fail', and optionally 'pnp_solver_wins'
            (dict of solver name -> count when PnP was used).
        pnp_min_inlier_ratio : float
            Minimum inlier ratio for PnP pose acceptance (default 0.55).
        pnp_solver : str
            One of "iterative" (default), "multi", "ap3p", "epnp", "sqpnp".
        pnp_refinement : bool
            If True, refine PnP pose with solvePnPRefineLM on inliers.
        pnp_use_initial_guess : bool
            If True, use current chained pose as initial guess for iterative solver.
        pnp_multi_inlier_gate : float
            When pnp_solver == "multi", only consider solvers with inlier count
            >= this fraction of the max (default 0.8); then pick by reprojection error.
        pnp_reprojection_error : float
            PnP RANSAC max reprojection error in pixels for inliers (default 3.0).
        pnp_adaptive_gate : bool
            If True, widen the step/correction safety gate proportionally to
            consecutive PnP non-accepts (see pnp_adaptive_gate_factor). Helps avoid
            rejecting valid PnP poses when chained VO has accumulated drift.
        pnp_adaptive_gate_factor : float
            When pnp_adaptive_gate is True: gate scale = 1 + factor * consecutive_failures
            (default 0.2; e.g. 5 consecutive failures -> 2x wider gate).
        use_relative_scale : bool
            If True (default), estimate translation scale from map points for Essential
            path so chained VO keeps metric scale. If False, use unit translation (scale=1)
            to test e.g. PnP behavior without scale recovery.
        oracle : oracle.Oracle | None
            Stage 03c oracle ladder injector (off by default: None). Rungs o1/o2/o3 replace
            the two-view motion right after estimate_motion (injection point 1); rung o4
            replaces the PnP pose at PnP frames (injection point 2). None = no-op everywhere.
        """
        if stats is not None:
            stats["vo_attempts"] = stats.get("vo_attempts", 0) + 1
        motion = estimate_motion(prev_frame, self, use_homography=use_homography)  # motion holds dataclass MotionEstimate which contains the motion estimate and the inlier matches and epipolar error.
        # --- Stage 03c injection point 1 (spec B1): O1/O2/O3 replace the two-view motion
        # here, before the vo-fail branch and the [SCALE-LOG] line; everything after runs
        # unchanged on the returned motion. k = self.id - 1 is the trajectory index.
        if oracle is not None and oracle.rung in ("o1", "o2", "o3"):
            motion, oracle_rec = oracle.override_motion(self.id - 1, motion)
            print(oracle.format_line(oracle_rec))
        if motion is None:
            if stats is not None:
                stats["vo_fail"] = stats.get("vo_fail", 0) + 1
            _nan = float("nan")
            print(
                f"  >> [SCALE-LOG] frame={self.id} model=NONE "
                f"norm_t={_nan:.6f} scale={_nan:.6f} step={_nan:.6f} reason=vo_fail"
            )
            print(f"  >> VO failed for frame {self.id}")
            return

        t_scaled = np.asarray(motion.t, dtype=np.float64).flatten() # get the translation from the motion estimate.
        t_unit_for_log = t_scaled

        # --- Relative scale recovery (optional) ---
        # recoverPose returns a unit translation (||t|| = 1).  When use_relative_scale
        # is True we re-scale t using existing 3D map points so chained VO keeps metric scale.
        scale = 1.0
        scale_reason = "unit_disabled"
        scale_estimator_ran = False
        if (
            use_relative_scale
            and world_map is not None
            and motion.model == "ESSENTIAL_MATRIX"
            and np.linalg.norm(t_scaled) > 1e-8
        ):
            prev_step = None
            if stats is not None:
                prev_step = stats.get("last_vo_step")
            scale, scale_reason = _estimate_translation_scale(
                prev_frame, self, motion.R, t_scaled,
                motion.inlier_matches, world_map,
                prev_step=prev_step,
            )
            scale_estimator_ran = True
            t_scaled = t_scaled * scale
            if stats is not None:
                stats.setdefault("scale_fallback_reasons", {})
                stats["scale_fallback_reasons"][scale_reason] = (
                    stats["scale_fallback_reasons"].get(scale_reason, 0) + 1
                )
                if not scale_reason.startswith("ok"):
                    print(
                        f"  >> [SCALE] frame {self.id}: fallback scale={scale:.6f} "
                        f"reason={scale_reason}"
                    )

        if t_norm_log_file is not None:
            norm_t = float(np.linalg.norm(motion.t))
            t_norm_log_file.write(
                f"{self.id} {norm_t:.6f} scale={scale:.6f} "
                f"||t_scaled||={float(np.linalg.norm(t_scaled)):.6f} reason={scale_reason}\n"
            )
            t_norm_log_file.flush()

        # --- Stage 02 change A: one [SCALE-LOG] line per update_pose call ---
        # (frame.py, Frame.update_pose) reporting the estimator's raw outcome,
        # printed before PnP / BA / loop closure can touch the pose.
        if motion.model == "HOMOGRAPHY":
            log_model = "HOMOGRAPHY"
            log_norm_t = float(np.linalg.norm(motion.t))
            log_scale = 1.0
            log_reason = "homography"
        else:
            log_model = "ESSENTIAL_MATRIX"
            log_norm_t = float(np.linalg.norm(motion.t))
            if use_relative_scale and not scale_estimator_ran:
                # The real gate above did not fire (e.g. no map yet, or a
                # near-zero unit translation); still report what the
                # estimator would say, without applying it to the pose.
                log_prev_step = stats.get("last_vo_step") if stats is not None else None
                log_scale, log_reason = _estimate_translation_scale(
                    prev_frame, self, motion.R, t_unit_for_log,
                    motion.inlier_matches, world_map,
                    prev_step=log_prev_step,
                )
            else:
                log_scale, log_reason = scale, scale_reason
        print(
            f"  >> [SCALE-LOG] frame={self.id} model={log_model} "
            f"norm_t={log_norm_t:.6f} scale={log_scale:.6f} "
            f"step={float(np.linalg.norm(t_scaled)):.6f} reason={log_reason}"
        )

        # --- Standard relative-motion pose (chained from previous frame) ---
        pose_prev_to_curr = rt_to_t_inv(motion.R, t_scaled)
        ok_pose = self.set_world_pose(prev_frame.world_T_cam @ pose_prev_to_curr)
        if not ok_pose:
            if stats is not None:
                stats["vo_fail"] = stats.get("vo_fail", 0) + 1
            print(f"  >> VO failed for frame {self.id} (invalid chained pose)")
            return

        self.pose_valid = True
        self.tracking_state = "OK"

        # Track accepted VO step for scale fallback / PnP median gate (Stage 2).
        vo_step_len = float(np.linalg.norm(
            self.world_T_cam[:3, 3] - prev_frame.world_T_cam[:3, 3]
        ))
        if stats is not None and vo_step_len > 1e-12:
            stats["last_vo_step"] = vo_step_len
            recent = stats.setdefault("recent_vo_steps", [])
            recent.append(vo_step_len)
            if len(recent) > 30:
                del recent[:-30]

        if world_map is not None and len(motion.inlier_matches) > 0:
            # Propagate existing 3D point tracks to this frame (per-frame link transfer).
            # This enables PnP to use only previously-known landmarks.  Since stage 04-1a
            # the unmatched remainder is not triangulated here: landmarks are created only
            # at keyframe insertion by local_mapping.process_new_keyframe (pipeline.py).
            associate_known_points(prev_frame, self, motion.inlier_matches, world_map)

        # --- Local map tracking ---
        # Project map points visible in recent frames (not just k-1) into
        # the current image and match by descriptor in a search window.
        # This recovers 2D-3D correspondences that frame-to-frame matching
        # missed, giving PnP a much larger and more stable inlier set.
        if world_map is not None and recent_frames:
            n_local = local_map_track(
                self, world_map, recent_frames,
                search_radius=float(local_map_search_radius),
                max_desc_dist=local_map_max_desc_dist,
                max_depth_ratio=local_map_max_depth_ratio,
            )
            if n_local > 0:
                print(
                    f"  >> Local map: +{n_local} matches from recent frames "
                    f"({len(self.keypoint_to_point)} total map links)"
                )

        # --- PnP re-localization---
        # When use_pnp is True we attempt to localize this frame against
        # existing 3D map points.  If successful the absolute PnP pose
        # replaces the chained relative-motion pose, reducing drift.
        pnp_accepted = False
        pnp_attempt = use_pnp and world_map is not None
        # --- Stage 03c injection point 2 (spec B5): under rung o4 the PnP solver is not
        # called; the ground-truth absolute pose (mapped by the single gauge of A5) takes its
        # place and EVERYTHING from the safety gate on runs as built. Without truth on this
        # frame there is no PnP attempt at all (no gate line, no counter change).
        oracle_o4 = pnp_attempt and oracle is not None and oracle.rung == "o4"
        if oracle_o4 and not oracle.has_truth(self.id - 1):
            print(oracle.format_line(oracle.o4_record(self.id - 1, "skip", "no_truth")))
            pnp_attempt = False
        if pnp_attempt:
            if oracle_o4:
                ok, cam_T_world, n_inliers, solver_used = True, oracle.absolute_pose(self.id - 1), 0, "oracle"
                oracle_prev_world_T_cam = prev_frame.world_T_cam.copy()
                oracle_chained_world_T_cam = self.world_T_cam.copy()
            else:
                t_pnp_start = time.perf_counter() # start timing the pnp localization.
                ok, cam_T_world, n_inliers, solver_used = pnp_localize(
                    self, world_map, min_inlier_ratio=pnp_min_inlier_ratio,
                    solver=pnp_solver, refinement=pnp_refinement,
                    use_extrinsic_guess=pnp_use_initial_guess,
                    multi_inlier_gate=pnp_multi_inlier_gate,
                    reprojection_error=pnp_reprojection_error,
                    stats=stats,
                    outlier_hit_accounting=bool(outlier_hit_accounting),
                )
                t_pnp_elapsed = time.perf_counter() - t_pnp_start
                method_label = solver_used if solver_used is not None else "iterative"
                if stats is not None:
                    stats.setdefault("pnp_times", []).append((t_pnp_elapsed, solver_used))
                print(f"  >> [TIMING] PnP frame {self.id} (method={method_label}): {t_pnp_elapsed:.3f}s")
            if ok:
                # --- Safety gate (Stage 2 policy A): bound PnP step/correction by
                # median recent VO step, not by max(3×vo_step, 2.0) which allowed
                # ~2.5-unit snaps whenever unit-scale VO made vo_step≈1.
                pnp_world_T_cam = np.linalg.inv(cam_T_world)
                pnp_step = np.linalg.norm(
                    pnp_world_T_cam[:3, 3] - prev_frame.world_T_cam[:3, 3]
                )
                vo_step = np.linalg.norm(
                    self.world_T_cam[:3, 3] - prev_frame.world_T_cam[:3, 3]
                )
                correction_jump = np.linalg.norm(
                    pnp_world_T_cam[:3, 3] - self.world_T_cam[:3, 3]
                )

                recent_steps = []
                if stats is not None:
                    recent_steps = list(stats.get("recent_vo_steps") or [])
                if recent_steps:
                    ref_step = float(np.median(recent_steps))
                else:
                    ref_step = float(max(vo_step, 1e-6))
                ref_step = max(ref_step, 1e-6)

                max_step = 3.0 * ref_step
                max_correction = 2.5 * ref_step

                # Adaptive gate: after a run of PnP failures we kept VO, so drift grew.
                # Widen the gate so a legitimate large correction is not rejected.
                if pnp_adaptive_gate and stats is not None:
                    consecutive = stats.get("pnp_consecutive_failures", 0)
                    gate_scale = 1.0 + pnp_adaptive_gate_factor * consecutive
                    gate_scale = min(gate_scale, 3.0)  # cap so one bad accept cannot blow up
                    max_step *= gate_scale
                    max_correction *= gate_scale

                print(
                    f"  >> [PnP-GATE] frame {self.id}: step={pnp_step:.4f} "
                    f"correction_jump={correction_jump:.4f} "
                    f"max_step={max_step:.4f} max_correction={max_correction:.4f} "
                    f"ref_step(median)={ref_step:.4f} vo_step={vo_step:.4f}"
                )

                if pnp_step <= max_step and correction_jump <= max_correction:
                    # Stage 2 policy B: distribute SE(3) delta over the PnP interval
                    # window (same idea as loop linear corrector) instead of a single
                    # frame overwrite when a window of prior frames is available.
                    applied = False
                    if (
                        pnp_correction_window is not None
                        and len(pnp_correction_window) >= 2
                        and pnp_correction_window[-1] is self
                    ):
                        snapshots = {
                            id(f): f.world_T_cam.copy() for f in pnp_correction_window
                        }
                        old_world_T_cam = snapshots[id(self)]
                        delta = pnp_world_T_cam @ np.linalg.inv(old_world_T_cam)
                        delta_R = delta[:3, :3]
                        delta_t = delta[:3, 3]
                        delta_rvec, _ = cv2.Rodrigues(delta_R)
                        delta_rvec = delta_rvec.ravel()
                        n_span = len(pnp_correction_window) - 1
                        anchor = pnp_correction_window[0]
                        ok_dist = True
                        for i, f in enumerate(pnp_correction_window):
                            if i == 0:
                                continue
                            alpha = i / float(n_span)
                            interp_rvec = alpha * delta_rvec
                            interp_R, _ = cv2.Rodrigues(interp_rvec.reshape(3, 1))
                            interp_delta = np.eye(4, dtype=np.float64)
                            interp_delta[:3, :3] = interp_R
                            interp_delta[:3, 3] = alpha * delta_t
                            if not f.set_world_pose(interp_delta @ snapshots[id(f)]):
                                ok_dist = False
                                break
                        if ok_dist:
                            applied = True
                            pnp_accepted = True
                            print(
                                f"  >> PnP localization succeeded for frame {self.id} "
                                f"({n_inliers} inliers; distributed over "
                                f"{len(pnp_correction_window)} frames, "
                                f"anchor={getattr(anchor, 'id', '?')})"
                            )
                            # Stage 03a (spec C1): which earlier frames the distributed
                            # correction rewrote. frame= is the trajectory index k used by
                            # _run_one_frame (Frame.id starts at 1, so k = Frame.id - 1),
                            # matching the ba / loop_closure lines; first=/last= stay Frame.id.
                            print_correction(
                                self.id - 1, "pnp_distributed",
                                {f.id: snapshots[id(f)] for f in pnp_correction_window},
                                [(f, f.world_T_cam) for f in pnp_correction_window],
                                exclude_id=self.id,
                            )
                        else:
                            for f in pnp_correction_window:
                                f.set_world_pose(snapshots[id(f)])
                    if not applied:
                        if self.set_camera_pose(cam_T_world):
                            pnp_accepted = True
                            print(
                                f"  >> PnP localization succeeded for frame {self.id} "
                                f"({n_inliers} inliers)"
                            )
                        else:
                            if stats is not None:
                                stats.setdefault("pnp_fail_reasons", {})
                                stats["pnp_fail_reasons"]["invalid_pose"] = (
                                    stats["pnp_fail_reasons"].get("invalid_pose", 0) + 1
                                )
                            print(f"  >> PnP localization rejected for frame {self.id} (invalid pose)")
                            pnp_accepted = False
                else:
                    if stats is not None:
                        stats.setdefault("pnp_fail_reasons", {})
                        stats["pnp_fail_reasons"]["safety_gate"] = (
                            stats["pnp_fail_reasons"].get("safety_gate", 0) + 1
                        )
                        stats["pnp_safety_gate_max_step"] = max_step
                        stats["pnp_safety_gate_max_correction"] = max_correction
                    print(
                        f"  >> PnP localization rejected for frame {self.id} "
                        f"(step={pnp_step:.2f}, correction={correction_jump:.2f})"
                    )
            else:
                print(
                    f"  >> PnP localization failed for frame {self.id}, "
                    f"keeping relative-motion pose"
                )
            if stats is not None:
                if pnp_accepted:
                    stats["pnp_success"] += 1
                    stats["pnp_consecutive_failures"] = 0  # reset when we accept PnP
                else:
                    stats["pnp_fail"] += 1
                    # Count this as a consecutive non-accept only when we actually attempted PnP
                    stats["pnp_consecutive_failures"] = stats.get("pnp_consecutive_failures", 0) + 1
                if solver_used is not None and "pnp_solver_wins" in stats:
                    stats["pnp_solver_wins"][solver_used] = (
                        stats["pnp_solver_wins"].get(solver_used, 0) + 1
                    )
            if oracle_o4:
                # Stage 03c (spec B5/C1): the o4 record, printed AFTER the gate decision.
                print(oracle.format_line(oracle.o4_record(
                    self.id - 1, "inject", "ok",
                    prev_world_T_cam=oracle_prev_world_T_cam,
                    chained_world_T_cam=oracle_chained_world_T_cam,
                    injected_world_T_cam=np.linalg.inv(cam_T_world),
                    accepted=int(pnp_accepted),
                )))

        # --- Epipolar error report (requirement 2 of תרגיל 3) ---
        # Shows how well the matched points satisfy the epipolar constraint
        # x2^T F x1 ≈ 0, both before RANSAC filtering and after.
        print(
            f"  Epipolar error: before={motion.epipolar_err_before:.4f} px, "
            f"after={motion.epipolar_err_after:.4f} px  "
            f"(matches: {len(motion.all_matches)} -> {len(motion.inlier_matches)} inliers)"
        )

        if motion.model == "HOMOGRAPHY":
            ratio = motion.inliers_H / motion.inliers_E if motion.inliers_E > 0 else 0.0
            print(
                f"Frame {self.id}: Using HOMOGRAPHY "
                f"(pure rotation detected, ratio={ratio:.2f})"
            )
            if stats is not None:
                stats["homography"] += 1
        else:
            print(f"Frame {self.id}: Using ESSENTIAL MATRIX")
            if stats is not None:
                stats["essential"] += 1

        # Store motion data so the visualizer can draw match lines.
        self._last_motion = motion
        print("pose of image ", self.id, " updated\n")


'''# Backward compatibility with the previous class name.
FRAME = Frame


def load_frames(*args, **kwargs):
    from dataset import load_frames as _load_frames

    return _load_frames(*args, **kwargs)


def show_trajectory(*args, **kwargs):
    from pipeline import show_trajectory as _show_trajectory

    return _show_trajectory(*args, **kwargs)
'''
