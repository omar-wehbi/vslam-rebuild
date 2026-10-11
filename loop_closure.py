"""
Loop Closure detection and trajectory correction.

The idea: as the camera moves it may revisit a location it saw earlier.
When this happens we can detect the overlap by matching the current frame's
descriptors against stored *keyframes* and, if enough matches are found,
use PnP to compute the geometric relationship.  The discrepancy between
the PnP pose and the current estimated pose reveals the accumulated drift,
which we then distribute linearly across the intermediate frames (a simple
pose-graph relaxation).

Algorithm
---------
1.  **Keyframe selection** — every ``keyframe_interval`` frames we store a
    reference to the frame (with its descriptors and map-point links).
2.  **Loop detection** — for each new frame we match its descriptors
    against all keyframes that are at least ``min_gap``(default=50) frames away.
    If the number of good keypoint matches exceeds ``match_threshold``(default=50) we have a
    candidate. (Lowe's ratio test.)
3.  **Geometric verification** — we verify the candidate using PnP:
    gather 2D (current keypoints) ↔ 3D (keyframe map points) pairs and
    run ``cv2.solvePnPRansac``.  If enough inliers confirm the match the
    loop is accepted.
4.  **Trajectory correction** — the correction (difference between PnP
    pose and current pose) is spread linearly across all frames between
    the loop keyframe and the current frame.
"""

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from keypoints_match import match_frames
from optimization import se3_adjoint, se3_exp, se3_inv, se3_log, se3_right_jacobian_inv

# Optional BoW backends for loop detection (strategy "bow"). Tried in order:
# 1) loopclosuretoolbox (https://github.com/HyeonJaeGil/loop-closure-toolbox), 2) pyDBoW3.
# If none available or vocabulary load fails, fall back to "spaced" strategy.
try:
    import loopclosuretoolbox.dbow as _toolbox_dbow  # type: ignore[import-untyped]
    _TOOLBOX_AVAILABLE = True
except ImportError:
    _toolbox_dbow = None
    _TOOLBOX_AVAILABLE = False

try:
    import pyDBoW3 as _pyDBoW3  # type: ignore[import-untyped]
    _PYDBOW3_AVAILABLE = True
except ImportError:
    _pyDBoW3 = None
    _PYDBOW3_AVAILABLE = False


def verify_loop_candidate(pts_3d, pts_2d, K):
    """
    Geometric verification of a loop-closure candidate via unseeded PnP.

    Runs cv2.solvePnPRansac WITHOUT an initial-pose guess (useExtrinsicGuess=False,
    SOLVEPNP_EPNP hypotheses — OpenCV's RANSAC draws hypotheses with EPnP regardless
    of the flags passed here; this only controls the final-inlier refit inside RANSAC).
    On success with at least 10 inliers, refines the pose on the inlier set with
    solvePnPRefineLM, starting from the RANSAC result (never from any external pose
    guess). See theory/conventions.md#loop-closure-relocalization-seed: the seed is a
    hypothesis about the failure mode, not an established fact, so "unseeded" is the
    defensible default and no benefit is assumed here.

    Parameters
    ----------
    pts_3d : np.ndarray (N, 3)
    pts_2d : np.ndarray (N, 2)
    K : np.ndarray (3, 3)
        Camera intrinsics of the frame the 2D points come from.

    Returns
    -------
    ok : bool
        True if PnP succeeded with >= 10 inliers.
    cam_T_world : np.ndarray (4, 4) | None
        World-to-camera pose, or None on failure.
    n_inliers : int
        Number of RANSAC inliers (0 on failure).
    """
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts_3d,
        imagePoints=pts_2d,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=500,
        reprojectionError=4.0,
        confidence=0.999,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not ok or inliers is None or len(inliers) < 10:
        return False, None, 0

    idx = inliers.ravel()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=pts_3d[idx],
            imagePoints=pts_2d[idx],
            cameraMatrix=K,
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

    return True, cam_T_world, len(inliers)


def _tracking_ratio(last_keyframe, current_frame):
    """
    Fraction of last keyframe's features that still match in the current frame.

    Returns
    -------
    float
        len(matches) / len(last_keyframe.KeyPoints), or 1.0 if last keyframe has no keypoints.
    """
    n_kp = len(last_keyframe.KeyPoints)
    if n_kp == 0:
        return 1.0
    matches = match_frames(last_keyframe, current_frame, current_frame.method)
    return len(matches) / n_kp


class LoopCloser:
    """Maintains keyframes and checks for loop closures."""

    # Keyframe strategy: 'fixed' (interval only), 'ratio' (tracking ratio), 'hybrid' (min/max + ratio).
    STRATEGY_FIXED = "fixed"
    STRATEGY_RATIO = "ratio"
    STRATEGY_HYBRID = "hybrid"

    def __init__(
        self,
        keyframe_interval=20,
        min_gap=30,
        match_threshold=25,
        keyframe_strategy="fixed",
        keyframe_min_spacing=5,
        keyframe_max_spacing=50,
        keyframe_ratio_threshold=0.80,
        loop_detection_strategy="spaced",
        vocabulary_path=None,
        num_spaced_candidates=5,
        relocalization_match_threshold=25,
        relocalization_min_2d3d_pairs=10,
        relocalization_min_pnp_inliers=10,
        relocalization_ratio_threshold=0.85,
        relocalization_sanity_check_enabled=False,
        relocalization_max_distance_from_keyframe=2.0,
    ):
        """
        Parameters
        ----------
        keyframe_interval : int
            For strategy 'fixed': store a keyframe every this many frames.
            For 'ratio'/'hybrid': used as fallback when there is no previous keyframe.
        min_gap : int
            Minimum frame-index distance between the current frame and a
            candidate keyframe — prevents matching against very recent
            neighbours which would trivially match.
        match_threshold : int
            Minimum number of good descriptor matches to consider a
            keyframe a loop-closure candidate.
        keyframe_strategy : str
            'fixed': keyframe every keyframe_interval frames.
            'ratio': keyframe when tracking ratio (matched features / last_kf features) < keyframe_ratio_threshold, with min/max spacing.
            'hybrid': same as ratio with default min 5, max 50, ratio 0.80.
        keyframe_min_spacing : int
            For 'ratio' and 'hybrid': minimum frames since last keyframe before considering a new one (avoids burst).
        keyframe_max_spacing : int
            For 'ratio' and 'hybrid': force a keyframe after this many frames since last keyframe.
        keyframe_ratio_threshold : float
            For 'ratio' and 'hybrid': create keyframe when tracking_ratio < this value (e.g. 0.80 = less than 80% tracked).
        loop_detection_strategy : str
            'spaced': pick up to num_spaced_candidates evenly-spaced keyframes (no vocabulary).
            'bow': use Bag-of-Words (loopclosuretoolbox or pyDBoW3) to query candidates; requires vocabulary_path and ORB features.
        vocabulary_path : str | None
            Path to ORBvoc.txt (or DBoW3-compatible vocabulary). Required when loop_detection_strategy is 'bow'.
        num_spaced_candidates : int
            When loop_detection_strategy is 'spaced', max number of evenly-spaced keyframe candidates to try (default 5).
        relocalization_match_threshold : int
            Minimum descriptor matches for relocalization (default 25).
        relocalization_min_2d3d_pairs : int
            Minimum 2D-3D pairs for relocalization PnP (default 10).
        relocalization_min_pnp_inliers : int
            Minimum PnP RANSAC inliers to accept relocalization (default 10).
        relocalization_ratio_threshold : float
            Lowe ratio threshold for descriptor matching in relocalization (default 0.85; more permissive than 0.75).
        relocalization_sanity_check_enabled : bool
            If True, reject relocalization when the PnP pose is farther than relocalization_max_distance_from_keyframe
            from the matched keyframe (default False). When False, the distance is logged but not used to reject.
        relocalization_max_distance_from_keyframe : float
            When sanity check is enabled: reject if PnP pose camera position is farther than this (world units) from
            the matched keyframe (default 2.0). Ignored when relocalization_sanity_check_enabled is False.
        """
        self.keyframe_interval = keyframe_interval
        self.min_gap = min_gap
        self.match_threshold = match_threshold
        self.keyframe_strategy = keyframe_strategy
        self.keyframe_min_spacing = keyframe_min_spacing
        self.keyframe_max_spacing = keyframe_max_spacing
        self.keyframe_ratio_threshold = keyframe_ratio_threshold
        self.loop_detection_strategy = loop_detection_strategy
        self.num_spaced_candidates = max(1, int(num_spaced_candidates))
        self.relocalization_match_threshold = max(1, int(relocalization_match_threshold))
        self.relocalization_min_2d3d_pairs = max(6, int(relocalization_min_2d3d_pairs))
        self.relocalization_min_pnp_inliers = max(6, int(relocalization_min_pnp_inliers))
        self.relocalization_ratio_threshold = float(relocalization_ratio_threshold)
        self.relocalization_sanity_check_enabled = bool(relocalization_sanity_check_enabled)
        self.relocalization_max_distance_from_keyframe = float(relocalization_max_distance_from_keyframe)
        # List of (frame_index_in_sequence, Frame) tuples.
        self.keyframes = []
        # For hybrid/ratio: count keyframes by reason ('interval', 'ratio', 'max_spacing') for end-of-run summary.
        self.keyframe_reason_counts = {"interval": 0, "ratio": 0, "max_spacing": 0}

        # Bag-of-Words database for strategy "bow". None => use spaced candidates.
        # Try loopclosuretoolbox first, then pyDBoW3.
        self.bow_db = None
        if loop_detection_strategy == "bow" and vocabulary_path:
            import warnings
            def _fmt_err(name, e):
                s = str(e).strip()
                return "%s: %s" % (name, s if s else "%s" % (type(e).__name__,))
            _bow_errors = []
            if _TOOLBOX_AVAILABLE:
                try:
                    voc = _toolbox_dbow.Vocabulary(vocabulary_path)
                    self.bow_db = _toolbox_dbow.Database(voc, False, 0)
                except Exception as e:
                    self.bow_db = None
                    _bow_errors.append(_fmt_err("loopclosuretoolbox", e))
            else:
                _bow_errors.append("loopclosuretoolbox: not available (import failed at startup)")
            if self.bow_db is None and _PYDBOW3_AVAILABLE and getattr(_pyDBoW3, "Vocabulary", None) is not None and getattr(_pyDBoW3, "Database", None) is not None:
                try:
                    voc = _pyDBoW3.Vocabulary()
                    voc.load(vocabulary_path)
                    db = _pyDBoW3.Database()
                    db.setVocabulary(voc)
                    self.bow_db = db
                except Exception as e:
                    self.bow_db = None
                    _bow_errors.append(_fmt_err("pyDBoW3", e))
            elif self.bow_db is None and _PYDBOW3_AVAILABLE:
                _bow_errors.append("pyDBoW3: skipped (Vocabulary/Database not found)")
            if self.bow_db is None:
                msg = (
                    "Loop detection: could not load vocabulary from %r "
                    "(tried loopclosuretoolbox and pyDBoW3); falling back to spaced strategy."
                ) % (vocabulary_path,)
                if _bow_errors:
                    msg += " Errors: %s" % ("; ".join(_bow_errors),)
                else:
                    msg += " (loopclosuretoolbox available: %s)" % (_TOOLBOX_AVAILABLE,)
                warnings.warn(msg, UserWarning)

    def _should_create_keyframe(self, frame_idx, frame):
        """
        Return (should_add, reason) for the current strategy.

        reason is one of 'interval', 'ratio', 'max_spacing' when should_add is True;
        otherwise None. Used for hybrid/ratio to report counts at end of run.
        """
        if self.keyframe_strategy == self.STRATEGY_FIXED:
            ok = frame_idx % self.keyframe_interval == 0
            return (ok, "interval" if ok else None)

        # Ratio or hybrid: need last keyframe for tracking ratio.
        if not self.keyframes:
            # First keyframe (frame 0) is always added by caller; for any other frame with no keyframes yet, use interval.
            ok = frame_idx % self.keyframe_interval == 0
            return (ok, "interval" if ok else None)

        last_kf_idx, last_kf = self.keyframes[-1]
        frames_since_last = frame_idx - last_kf_idx

        if frames_since_last < self.keyframe_min_spacing:
            return (False, None)
        if frames_since_last >= self.keyframe_max_spacing:
            return (True, "max_spacing")
        ''' ratio check is used to detect when many of the last keyframe's keypoints no longer match,
         which indicates a significant change in the scene. at which point the frame is added as a keyframe'''
        tracking_ratio = _tracking_ratio(last_kf, frame)
        if tracking_ratio < self.keyframe_ratio_threshold:
            return (True, "ratio")
        return (False, None)

    def maybe_add_keyframe(self, frame_idx, frame):
        """Store frame as a keyframe if selection criteria (fixed interval, ratio, or hybrid) are met."""
        should_add, reason = self._should_create_keyframe(frame_idx, frame)
        if should_add:
            self.keyframes.append((frame_idx, frame))
            if reason is not None:
                self.keyframe_reason_counts[reason] = self.keyframe_reason_counts.get(reason, 0) + 1
            # BoW: add this keyframe's descriptors so entry order matches keyframes list.
            if self.bow_db is not None and getattr(frame, "method", None) == "ORB" and frame.Descriptors is not None and len(frame.Descriptors) > 0:
                desc = np.ascontiguousarray(frame.Descriptors, dtype=np.uint8)
                self.bow_db.add(desc)
            return True
        return False

    def get_keyframe_reason_summary(self):
        """Return a string summarizing keyframe counts by reason (for hybrid/ratio strategy)."""
        c = self.keyframe_reason_counts
        return (
            f"  Keyframes by reason: interval={c.get('interval', 0)}, "
            f"ratio={c.get('ratio', 0)}, max_spacing={c.get('max_spacing', 0)}"
        )

    def detect(self, curr_idx, curr_frame, world_map):
        """
        Check whether curr_frame closes a loop with any stored keyframe.

        Returns
        -------
        loop_kf_idx : int | None
            Sequence index of the matched keyframe, or None.
        loop_kf : Frame | None
            The matched keyframe Frame object.
        cam_T_world : np.ndarray | None
            PnP-derived absolute pose for curr_frame (if loop confirmed).
        n_inliers : int
            Number of PnP inliers (0 if no loop).
        """
        # Build candidate keyframes: BoW (query by appearance) or spaced (evenly-spaced).
        candidates = []
        if self.bow_db is not None and getattr(curr_frame, "method", None) == "ORB" and curr_frame.Descriptors is not None and len(curr_frame.Descriptors) > 0:
            # BoW path: query database, map entry_id to keyframe, filter by min_gap.
            try:
                desc = np.ascontiguousarray(curr_frame.Descriptors, dtype=np.uint8)
                N = min(10, max(1, len(self.keyframes)))
                try:
                    results = self.bow_db.query(desc, N)
                except TypeError:
                    results = self.bow_db.query(desc)[:N]
                for entry_id, score in results:
                    if entry_id < 0 or entry_id >= len(self.keyframes):
                        continue
                    kf_idx, kf = self.keyframes[entry_id]
                    if curr_idx - kf_idx >= self.min_gap:
                        candidates.append((kf_idx, kf))
                # Prefer higher score: take up to num_spaced_candidates (same as spaced strategy).
                candidates = candidates[: self.num_spaced_candidates]
            except Exception:
                candidates = []
        if not candidates:
            # Spaced path (or BoW fallback): up to num_spaced_candidates evenly-spaced keyframes.
            spaced = [
                (idx, kf) for idx, kf in self.keyframes
                if curr_idx - idx >= self.min_gap
            ]
            n = self.num_spaced_candidates
            if len(spaced) > n:
                step = max(1, len(spaced) // n)
                candidates = spaced[::step][:n]
            else:
                candidates = spaced

        for kf_idx, kf in candidates:

            # --- Descriptor matching ---
            matches = match_frames(kf, curr_frame, curr_frame.method)
            if len(matches) < self.match_threshold:
                continue

            # --- Geometric verification via PnP ---
            # Gather 2D (curr) ↔ 3D (keyframe map points) correspondences.
            pts_2d = []
            pts_3d = []
            for m in matches:
                pid = kf.keypoint_to_point.get(m.queryIdx)
                if pid is None:
                    continue
                pt = world_map.get_point(pid)
                if pt is None:
                    continue
                pts_2d.append(curr_frame.KeyPoints[m.trainIdx].pt)
                pts_3d.append(pt.point)

            if len(pts_2d) < 10:
                continue

            pts_2d = np.array(pts_2d, dtype=np.float64)
            pts_3d = np.array(pts_3d, dtype=np.float64)

            ok, cam_T_world, n_inliers = verify_loop_candidate(pts_3d, pts_2d, curr_frame.K)
            if not ok:
                continue

            return kf_idx, kf, cam_T_world, n_inliers

        return None, None, None, 0

    def relocalize(self, curr_idx, curr_frame, world_map):
        """
        Try to relocalize the current frame against keyframes when VO failed (e.g. cut in sequence).

        Uses the same candidate selection as loop closure (BoW then evenly-spaced) but without
        min_gap so we can match any keyframe. PnP is run without initial guess since the frame
        pose is meaningless when lost.

        Returns
        -------
        loop_kf_idx : int | None
            Sequence index of the matched keyframe, or None.
        cam_T_world : np.ndarray | None
            PnP-derived absolute pose for curr_frame (if relocalization succeeded).
        n_inliers : int
            Number of PnP inliers (0 if failed).
        fail_reason : str | None
            If failed, human-readable reason: no_keyframes, no_candidates, insufficient_matches,
            insufficient_2d3d_pairs, or pnp_failed. None on success.
        """
        if not self.keyframes:
            return None, None, 0, "no_keyframes"

        # Build candidates: BoW (no min_gap) or evenly-spaced (no min_gap).
        candidates = []
        if self.bow_db is not None and getattr(curr_frame, "method", None) == "ORB" and curr_frame.Descriptors is not None and len(curr_frame.Descriptors) > 0:
            try:
                desc = np.ascontiguousarray(curr_frame.Descriptors, dtype=np.uint8)
                N = min(10, max(1, len(self.keyframes)))
                try:
                    results = self.bow_db.query(desc, N)
                except TypeError:
                    results = self.bow_db.query(desc)[:N]
                for entry_id, score in results:
                    if entry_id < 0 or entry_id >= len(self.keyframes):
                        continue
                    kf_idx, kf = self.keyframes[entry_id]
                    candidates.append((kf_idx, kf))
                candidates = candidates[: self.num_spaced_candidates]
            except Exception:
                candidates = []
        if not candidates:
            spaced = list(self.keyframes)
            n = self.num_spaced_candidates
            if len(spaced) > n:
                step = max(1, len(spaced) // n)
                candidates = spaced[::step][:n]
            else:
                candidates = spaced

        min_2d3d = self.relocalization_min_2d3d_pairs
        min_inliers = self.relocalization_min_pnp_inliers
        fail_reason = "no_candidates"  # overwritten if we try any candidate
        for kf_idx, kf in candidates:
            matches = match_frames(
                kf, curr_frame, curr_frame.method,
                ratio_threshold=self.relocalization_ratio_threshold,
            )
            if len(matches) < self.relocalization_match_threshold:
                print(
                    f"  >> [reloc] keyframe {kf_idx}: {len(matches)} matches "
                    f"(need {self.relocalization_match_threshold})"
                )
                fail_reason = "insufficient_matches"
                continue

            pts_2d = []
            pts_3d = []
            for m in matches:
                pid = kf.keypoint_to_point.get(m.queryIdx)
                if pid is None:
                    continue
                pt = world_map.get_point(pid)
                if pt is None:
                    continue
                pts_2d.append(curr_frame.KeyPoints[m.trainIdx].pt)
                pts_3d.append(pt.point)

            n_2d3d = len(pts_2d)
            if n_2d3d < min_2d3d:
                print(
                    f"  >> [reloc] keyframe {kf_idx}: {len(matches)} matches, "
                    f"{n_2d3d} 2D-3D pairs (need {min_2d3d})"
                )
                fail_reason = "insufficient_2d3d_pairs"
                continue

            pts_2d = np.array(pts_2d, dtype=np.float64)
            pts_3d = np.array(pts_3d, dtype=np.float64)

            # Use matched keyframe pose as initial guess (current frame pose is meaningless when lost).
            init_rvec, _ = cv2.Rodrigues(kf.cam_T_world[:3, :3])
            init_tvec = kf.cam_T_world[:3, 3].reshape(3, 1).astype(np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pts_3d,
                imagePoints=pts_2d,
                cameraMatrix=curr_frame.K,
                distCoeffs=None,
                iterationsCount=500,
                reprojectionError=4.0,
                confidence=0.999,
                useExtrinsicGuess=True,
                rvec=init_rvec,
                tvec=init_tvec,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            n_inl = 0 if inliers is None else len(inliers)
            if not ok or inliers is None or n_inl < min_inliers:
                print(
                    f"  >> [reloc] keyframe {kf_idx}: {len(matches)} matches, "
                    f"{n_2d3d} 2D-3D pairs, {n_inl} inliers (need {min_inliers})"
                )
                fail_reason = "pnp_failed"
                continue

            R, _ = cv2.Rodrigues(rvec)
            cam_T_world = np.eye(4, dtype=np.float64)
            cam_T_world[:3, :3] = R
            cam_T_world[:3, 3] = tvec.flatten()

            # Optional sanity check: reject if relocalized pose is too far from the matched keyframe.
            reloc_cam_in_world = np.linalg.inv(cam_T_world)[:3, 3]
            kf_cam_in_world = kf.world_T_cam[:3, 3]
            dist_from_kf = float(np.linalg.norm(reloc_cam_in_world - kf_cam_in_world))
            if self.relocalization_sanity_check_enabled:
                max_dist = self.relocalization_max_distance_from_keyframe
                if dist_from_kf > max_dist:
                    print(
                        f"  >> [reloc] keyframe {kf_idx}: PnP pose {dist_from_kf:.2f} from keyframe "
                        f"(max {max_dist:.2f}) -> rejected (wrong solution)"
                    )
                    fail_reason = "too_far_from_keyframe"
                    continue

            print(
                f"  >> [reloc] keyframe {kf_idx}: {len(matches)} matches, "
                f"{n_2d3d} 2D-3D pairs, {n_inl} inliers, {dist_from_kf:.3f} from kf -> accepted"
            )
            return kf_idx, cam_T_world, n_inl, None

        return None, None, 0, fail_reason


# --------------------------------------------------------------------------- #
# Pose graph on the right-perturbation SE(3) chart (stage 01a, spec B).
#
# Nodes are world_T_cam poses.  Node 0 = frames[loop_kf_idx] is fixed; free node j
# (0-based) is frames[loop_kf_idx + 1 + j].  Chart: T_j <- T_j * se3_exp(delta_j),
# delta_j = (omega_j, nu_j) (theory/conventions.md#perturbation-model-for-pose-graphs).
# This is the 6-DoF SE(3) sub-chart of the ruled Sim(3) chart; stage 05 appends the
# scale coordinate (the node state is a plain vector with a `dof` attribute for that).
# --------------------------------------------------------------------------- #

class PoseGraphProblem:
    """
    Pose-graph problem: odometry chain plus one loop-closure unary factor.

    Residual vector (rows), with Z_i the odometry measurement and T_loop the loop pose:
      rows 6i..6i+5   = se3_log( Z_i^-1 T_i^-1 T_{i+1} )          i = 0..n_nodes-1
      rows 6n..6n+5   = w_loop * se3_log( T_loop^-1 T_N ),  N = n_nodes
    where T_0 is the fixed node and T_1..T_N are the free nodes.
    """

    dof = 6

    def __init__(self, T, Z, T_loop, w_loop, node_ids=None):
        # Private constructor: arrays already copied by build_pose_graph_problem / retract.
        self.T = T                      # list of n_nodes+1 world_T_cam (T[0] fixed)
        self.Z = Z                      # list of n_nodes odometry measurements
        self.T_loop = T_loop            # world_T_cam measurement for the last node
        self.w_loop = float(w_loop)
        self.node_ids = list(node_ids) if node_ids is not None else None
        self.n_nodes = len(T) - 1
        self.x_dim = self.dof * self.n_nodes
        self.n_res = self.dof * (self.n_nodes + 1)

    def _block(self, k):
        """Row/column slice of block k (residual block k, or free node k), width self.dof."""
        d = self.dof
        return slice(d * k, d * k + d)

    def residuals(self):
        n = self.n_nodes
        r = np.zeros(self.n_res, dtype=np.float64)
        for i in range(n):
            E = se3_inv(self.Z[i]) @ se3_inv(self.T[i]) @ self.T[i + 1]
            r[self._block(i)] = se3_log(E)
        r[self._block(n)] = self.w_loop * se3_log(se3_inv(self.T_loop) @ self.T[n])
        return r

    def jacobian(self):
        """
        Sparse (n_res, x_dim) derivative of residuals() w.r.t. the right perturbation
        T_j <- T_j se3_exp(delta_j) at delta = 0.  Columns 6j..6j+5 belong to free node j
        (= T_{j+1}).  With r_i = se3_log(E_i):
          d r_i / d delta_{i+1} =  Jr^-1(r_i)
          d r_i / d delta_i     = -Jr^-1(r_i) Ad(T_{i+1}^-1 T_i)      (i >= 1: T_0 is fixed)
          d r_loop / d delta_N  =  w_loop * Jr^-1(r_loop / w_loop)
        Derivation: perturbing T_{i+1} on the right gives E_i se3_exp(delta), whose log is
        r_i + Jr^-1(r_i) delta.  Perturbing T_i gives Z^-1 se3_exp(-delta) T_i^-1 T_{i+1}
        = E_i se3_exp(-Ad(T_{i+1}^-1 T_i) delta).
        """
        n = self.n_nodes
        r = self.residuals()
        J = scipy.sparse.lil_matrix((self.n_res, self.x_dim), dtype=np.float64)
        for i in range(n):
            rows = self._block(i)
            ri = r[rows]
            Jr_inv = se3_right_jacobian_inv(ri)
            J[rows, self._block(i)] = Jr_inv                                    # node i+1 = free i
            if i >= 1:
                A = se3_inv(self.T[i + 1]) @ self.T[i]
                J[rows, self._block(i - 1)] = -Jr_inv @ se3_adjoint(A)          # node i = free i-1
        r_loop = r[self._block(n)] / self.w_loop
        J[self._block(n), self._block(n - 1)] = self.w_loop * se3_right_jacobian_inv(r_loop)
        return J.tocsr()

    def retract(self, delta):
        """NEW problem with T_j <- T_j se3_exp(delta[dof*j : dof*j+dof]) for every free node j."""
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        if delta.shape[0] != self.x_dim:
            raise ValueError(f"delta must have length {self.x_dim}, got {delta.shape[0]}")
        T = [self.T[0].copy()]
        for j in range(self.n_nodes):
            T.append(self.T[j + 1] @ se3_exp(delta[self._block(j)]))
        return PoseGraphProblem(T, [z.copy() for z in self.Z], self.T_loop.copy(),
                                self.w_loop, self.node_ids)

    def poses(self):
        """world_T_cam (4,4) of the free nodes, in order."""
        return [t.copy() for t in self.T[1:]]

    def cost(self, r):
        return 0.5 * float(np.dot(r, r))


def build_pose_graph_problem(frames, loop_kf_idx, curr_idx, corrected_cam_T_world, w_loop=None):
    """
    Build the pose graph over frames[loop_kf_idx .. curr_idx].  Uses ONLY frames[i].world_T_cam
    and frames[i].id; every matrix is copied (no live references).  Odometry measurements
    Z_i = T_i^-1 T_{i+1} come from the poses at build time; the loop measurement is
    inv(corrected_cam_T_world).  w_loop defaults to sqrt(n_nodes).
    """
    n_nodes = int(curr_idx) - int(loop_kf_idx)
    if n_nodes < 1:
        raise ValueError(f"pose graph needs curr_idx > loop_kf_idx, got {loop_kf_idx}, {curr_idx}")
    T = [np.array(frames[i].world_T_cam, dtype=np.float64, copy=True)
         for i in range(loop_kf_idx, curr_idx + 1)]
    Z = [se3_inv(T[i]) @ T[i + 1] for i in range(n_nodes)]
    T_loop = np.linalg.inv(np.array(corrected_cam_T_world, dtype=np.float64, copy=True))
    if w_loop is None:
        w_loop = float(np.sqrt(n_nodes))
    node_ids = [frames[i].id for i in range(loop_kf_idx, curr_idx + 1)]
    return PoseGraphProblem(T, Z, T_loop, w_loop, node_ids)


def _solve_damped_normal_equations(A, g, lam):
    """Solve (A + lam diag(A)) delta = -g; sparse direct solve with dense fallback."""
    A = A.tocsc()
    D = scipy.sparse.diags(A.diagonal())
    M = (A + lam * D).tocsc()
    try:
        delta = scipy.sparse.linalg.spsolve(M, -g)
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        if np.all(np.isfinite(delta)):
            return delta
    except Exception:
        pass
    Md = M.toarray()
    try:
        delta = np.linalg.solve(Md, -g)
        if np.all(np.isfinite(delta)):
            return delta
    except np.linalg.LinAlgError:
        pass
    return np.linalg.lstsq(Md, -g, rcond=None)[0]


def solve_pose_graph(problem, *, max_iterations=20, lambda_init=1e-3, lambda_up=10.0,
                     lambda_down=0.1, cost_rel_tol=1e-8, step_tol=1e-10):
    """
    Levenberg-Marquardt on the chart, re-linearized at every accepted state.  Cost is
    0.5 ||r||^2 (no robust loss on the graph; declared).  Returns (solved_problem, info).
    """
    max_retries = 5
    r = problem.residuals()
    cost = problem.cost(r)
    cost_initial = cost
    lam = float(lambda_init)
    iterations = 0
    converged = False
    stop_reason = "max_iterations"
    for _ in range(int(max_iterations)):
        if cost <= 0.0:
            converged = True
            stop_reason = "zero_cost"
            break
        iterations += 1
        J = problem.jacobian()
        A = (J.T @ J).tocsc()
        g = np.asarray(J.T @ r).reshape(-1)
        accepted = False
        delta = np.zeros(problem.x_dim)
        for _retry in range(max_retries + 1):
            delta = _solve_damped_normal_equations(A, g, lam)
            candidate = problem.retract(delta)
            r_c = candidate.residuals()
            cost_c = candidate.cost(r_c)
            if np.isfinite(cost_c) and cost_c < cost:
                accepted = True
                break
            lam *= lambda_up
        if not accepted:
            stop_reason = "rejected_steps"
            break
        rel_decrease = (cost - cost_c) / cost if cost > 0.0 else 0.0
        problem, r, cost = candidate, r_c, cost_c
        lam *= lambda_down
        if float(np.linalg.norm(delta)) < step_tol:
            converged = True
            stop_reason = "step_tol"
            break
        if rel_decrease < cost_rel_tol:
            converged = True
            stop_reason = "cost_rel_tol"
            break
    info = {
        "status": "completed",
        "iterations": iterations,
        "cost_initial": float(cost_initial),
        "cost_final": float(cost),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "n_nodes": problem.n_nodes,
    }
    return problem, info


def correct_trajectory_pose_graph(
    frames, loop_kf_idx, curr_idx, corrected_cam_T_world, world_map=None, w_loop=None
):
    """
    Correct the trajectory between loop keyframe and current frame by pose-graph optimization.

    Builds a pose graph over frames [loop_kf_idx, curr_idx]: nodes are world_T_cam poses,
    odometry edges between consecutive frames (measurement = current relative pose from VO),
    and one loop-closure unary factor at curr_idx (measurement = PnP pose). The first node
    is fixed; the rest are optimized on the right-perturbation SE(3) chart.  Map points
    observed in the span are then corrected using the optimized poses (same alpha-based
    scheme as the linear correction; stage 05 replaces it).

    Parameters
    ----------
    frames : list[Frame]
        Full list of frames (by sequence index, 0-based).
    loop_kf_idx : int
        Sequence index of the loop keyframe.
    curr_idx : int
        Sequence index of the current frame.
    corrected_cam_T_world : np.ndarray (4,4)
        PnP-verified cam_T_world for frames[curr_idx].
    world_map : Map | None
        If provided, apply consistent correction to map points in the span.
    w_loop : float | None
        Weight for the loop-closure residual so it balances the odometry chain.
        If None, use sqrt(n_span) as default.

    Returns
    -------
    int
        Number of map points corrected.
    """
    n_span = curr_idx - loop_kf_idx
    if n_span <= 0:
        return 0

    problem = build_pose_graph_problem(frames, loop_kf_idx, curr_idx, corrected_cam_T_world,
                                       w_loop=w_loop)
    # Old world_T_cam of every free node, stored BEFORE write-back (for the map correction).
    T_old = problem.poses()
    solved, info = solve_pose_graph(problem)
    T_new = solved.poses()

    # Write back optimized poses through the setter.
    for j in range(n_span):
        frames[loop_kf_idx + 1 + j].set_world_pose(T_new[j])

    if world_map is None:
        return 0

    # Map correction: per point, pick the frame at the mean observation fraction alpha and
    # apply that frame's correction delta_k = T_new_k @ inv(T_old_k).
    frame_id_to_seq = {f.id: i for i, f in enumerate(frames)}
    corrected_points = 0
    for pt in world_map.points:
        obs_seq = [
            frame_id_to_seq[fid]
            for fid in pt.frames
            if frame_id_to_seq.get(fid) is not None and loop_kf_idx < frame_id_to_seq[fid] <= curr_idx
        ]
        if not obs_seq:
            continue
        alpha = sum((idx - loop_kf_idx) / n_span for idx in obs_seq) / len(obs_seq)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        k = loop_kf_idx + int(round(alpha * n_span))
        k = min(max(k, loop_kf_idx + 1), curr_idx)
        j = k - loop_kf_idx - 1
        delta_k = T_new[j] @ np.linalg.inv(T_old[j])
        R_d = delta_k[:3, :3]
        t_d = delta_k[:3, 3]
        new_xyz = R_d @ pt.point + t_d
        pt.point = new_xyz
        pt.x = float(new_xyz[0])
        pt.y = float(new_xyz[1])
        pt.z = float(new_xyz[2])
        corrected_points += 1

    return corrected_points


def correct_trajectory(frames, loop_kf_idx, curr_idx, corrected_cam_T_world, world_map=None):
    """
    Distribute the loop-closure correction linearly across intermediate frames.

    The correction is the difference (in SE(3)) between the PnP-derived
    pose and the current estimated pose of ``frames[curr_idx]``.  We
    interpolate this correction so that ``frames[loop_kf_idx]`` keeps its
    original pose and ``frames[curr_idx]`` gets the full correction.

    Parameters
    ----------
    frames : list[Frame]
        Full list of frames (by sequence index, 0-based).
    loop_kf_idx : int
        Sequence index of the loop keyframe.
    curr_idx : int
        Sequence index of the current frame.
    corrected_cam_T_world : np.ndarray (4,4)
        The PnP-verified cam_T_world for frames[curr_idx].
    world_map : Map | None
        If provided, also applies a consistent correction to map points.

    Returns
    -------
    int
        Number of map points corrected.
    """
    curr_frame = frames[curr_idx]

    # Correction in world coordinates: delta that maps old world_T_cam
    # to the corrected one.
    old_world_T_cam = curr_frame.world_T_cam.copy()
    corrected_world_T_cam = np.linalg.inv(corrected_cam_T_world)
    # delta @ old = corrected  =>  delta = corrected @ old^{-1}
    delta = corrected_world_T_cam @ np.linalg.inv(old_world_T_cam)

    # Extract axis-angle for interpolation.
    delta_R = delta[:3, :3]
    delta_t = delta[:3, 3]
    delta_rvec, _ = cv2.Rodrigues(delta_R)
    delta_rvec = delta_rvec.ravel()

    n_span = curr_idx - loop_kf_idx
    if n_span <= 0:
        return 0

    # Linearly interpolate the correction from 0 (at loop_kf_idx) to 1
    # (at curr_idx).
    for i in range(loop_kf_idx + 1, curr_idx + 1):
        alpha = (i - loop_kf_idx) / n_span

        # Interpolated rotation (axis-angle scales linearly for small angles).
        interp_rvec = alpha * delta_rvec
        interp_R, _ = cv2.Rodrigues(interp_rvec.reshape(3, 1))
        interp_t = alpha * delta_t

        # Build interpolated correction matrix.
        interp_delta = np.eye(4, dtype=np.float64)
        interp_delta[:3, :3] = interp_R
        interp_delta[:3, 3] = interp_t

        # Apply: corrected = interp_delta @ original
        frames[i].set_world_pose(interp_delta @ frames[i].world_T_cam)

    if world_map is None:
        return 0

    # Approximate map correction: apply the same interpolated drift to each
    # point based on where it is observed in the corrected frame span.
    frame_id_to_seq = {f.id: idx for idx, f in enumerate(frames)}
    corrected_points = 0
    for pt in world_map.points:
        obs_seq = []
        for fid in pt.frames:
            seq_idx = frame_id_to_seq.get(fid)
            if seq_idx is None:
                continue
            if loop_kf_idx < seq_idx <= curr_idx:
                obs_seq.append(seq_idx)

        if not obs_seq:
            continue

        alpha = sum((idx - loop_kf_idx) / n_span for idx in obs_seq) / len(obs_seq)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        interp_rvec = alpha * delta_rvec
        interp_R, _ = cv2.Rodrigues(interp_rvec.reshape(3, 1))
        interp_t = alpha * delta_t

        new_xyz = interp_R @ pt.point + interp_t
        pt.point = new_xyz
        pt.x = float(new_xyz[0])
        pt.y = float(new_xyz[1])
        pt.z = float(new_xyz[2])
        corrected_points += 1

    return corrected_points
