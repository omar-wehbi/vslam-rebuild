class Map:
    """
    Landmark container plus the ONE mutation API for the two association
    tables (Stage 01b, conventions.md#landmark-and-observation-consistency):

        Frame.keypoint_to_point[keypoint_index] = point_id
        Point.observations[frame_id]            = keypoint_index   (+ Point.frames)

    These must be mutual inverses, and no frame may link two keypoints to one
    landmark.  Every write to either table in the product code goes through
    ``link`` / ``unlink`` / ``remove_point`` / ``add_point`` so the invariant
    holds by construction; ``check_invariants`` is the executor-side copy of
    the verifier's counters.
    """

    def __init__(self, points=None, frames=None, max_points=6000):
        self.points = []
        # Hold the caller's list by reference: the pipeline appends frames to
        # it while streaming, and frame_by_id must see them (ids are stable).
        if frames is None:
            self.frames = []
        elif isinstance(frames, list):
            self.frames = frames
        else:
            self.frames = list(frames)
        self.point_by_id = {}
        self.max_points = int(max_points)
        self._frame_index = {}
        self._frame_index_len = -1

        if points is not None:
            for p in points:
                self.add_point(p)

    # ------------------------------------------------------------------ frames
    def _rebuild_frame_index(self):
        self._frame_index = {}
        for f in self.frames:
            fid = getattr(f, "id", None)
            if fid is not None:
                self._frame_index[fid] = f
        self._frame_index_len = len(self.frames)

    def frame_by_id(self, fid):
        """Frame with id ``fid`` from the live frame list, or None."""
        if len(self.frames) != self._frame_index_len:
            self._rebuild_frame_index()
        return self._frame_index.get(fid)

    def resolve_frame(self, fid, frames=None):
        """frame_by_id, then the ``frames`` argument, then a scan of self.frames."""
        f = self.frame_by_id(fid)
        if f is not None:
            return f
        if frames:
            for f in frames:
                if getattr(f, "id", None) == fid:
                    return f
        for f in self.frames:
            if getattr(f, "id", None) == fid:
                return f
        return None

    # ------------------------------------------------------------------- links
    def link(self, frame, kp_idx, point):
        """
        Link keypoint ``kp_idx`` of ``frame`` to ``point`` on both sides.

        Idempotent: True if both tables already say (frame.id, kp_idx) <-> point.id.
        FIRST LINK WINS: if kp_idx is already linked to a different point, or the
        point already has a different keypoint in this frame, nothing is changed
        and False is returned.  Also refused (False) when ``point`` is not in
        this map, so a frame link can never point at a missing landmark.
        """
        kp_idx = int(kp_idx)
        fid = frame.id
        if self.point_by_id.get(point.id) is not point:
            return False
        cur_pid = frame.keypoint_to_point.get(kp_idx)
        cur_kp = point.observations.get(fid)
        if cur_pid == point.id and cur_kp == kp_idx:
            if fid not in point.frames:
                point.frames.append(fid)
            return True
        if cur_pid is not None and cur_pid != point.id:
            return False
        if cur_kp is not None and cur_kp != kp_idx:
            return False
        frame.keypoint_to_point[kp_idx] = point.id
        point.add_observation(fid, kp_idx)
        return True

    @staticmethod
    def _drop_observation(point, fid, kp_idx):
        """Remove observation (fid -> kp_idx) from a point's two tables."""
        if point.observations.get(fid) != kp_idx:
            return False
        del point.observations[fid]
        point.frames = [x for x in point.frames if x != fid]
        return True

    def unlink(self, frame, kp_idx):
        """
        Remove the link of keypoint ``kp_idx`` of ``frame`` and the matching
        observation on the point.  False if there was no link.
        """
        kp_idx = int(kp_idx)
        pid = frame.keypoint_to_point.pop(kp_idx, None)
        if pid is None:
            return False
        pt = self.point_by_id.get(pid)
        if pt is not None:
            self._drop_observation(pt, frame.id, kp_idx)
        return True

    # ------------------------------------------------------------------ points
    def add_point(self, point, frames=None):
        """
        Add a point.  Observations the point arrives with (constructor
        argument) are mirrored into the frame tables through the same rule as
        ``link``; an observation whose keypoint is already linked elsewhere is
        removed from the point so (i)-(iv) hold on insertion.  An observation
        whose frame is not known to the map (nor in ``frames``) is kept as is:
        it cannot be mirrored, but it also cannot contradict any frame table.
        Eviction on ``max_points`` goes through ``remove_point``.
        """
        if point.id in self.point_by_id:
            return

        self.points.append(point)
        self.point_by_id[point.id] = point

        for fid, kp in list(point.observations.items()):
            f = self.resolve_frame(fid, frames)
            if f is None:
                continue
            if not self.link(f, kp, point):
                self._drop_observation(point, fid, int(kp))
        # (iv): frames is exactly the set of observed frame ids.
        obs = point.observations
        kept = [fid for fid in point.frames if fid in obs]
        seen = set(kept)
        for fid in obs:
            if fid not in seen:
                kept.append(fid)
                seen.add(fid)
        point.frames = kept

        while len(self.points) > self.max_points:
            self.remove_point(self.points[0].id, frames=frames)

    def remove_point(self, point_id, frames=None):
        """
        Unlink every observation of the point (frames resolved through
        frame_by_id, then ``frames``, then self.frames) and drop it from the
        map.  Uses the observation table; no scan of all frame link tables.
        """
        pt = self.point_by_id.pop(point_id, None)
        if pt is not None:
            for fid, kp in list(pt.observations.items()):
                f = self.resolve_frame(fid, frames)
                if f is not None and f.keypoint_to_point.get(kp) == point_id:
                    del f.keypoint_to_point[kp]
            pt.observations.clear()
            pt.frames = []
        if self.points and getattr(self.points[0], "id", None) == point_id:
            self.points.pop(0)
        else:
            self.points = [p for p in self.points if getattr(p, "id", None) != point_id]

    def get_point(self, point_id):
        return self.point_by_id.get(point_id)

    # -------------------------------------------------------------- invariants
    def check_invariants(self, frames=None):
        """
        Same five counters, same names, as the verifier's
        system/verify/checks/map_invariants.py::check_map_consistency.
        """
        if frames is None:
            frames = self.frames
        by_id = {}
        for f in frames:
            fid = getattr(f, "id", None)
            if fid is not None:
                by_id[fid] = f
        c = {
            "inv_link_to_missing_point": 0,
            "inv_link_without_observation": 0,
            "inv_observation_without_link": 0,
            "inv_frame_double_link": 0,
            "inv_frames_obs_mismatch": 0,
        }
        for f in frames:
            seen = {}
            for kp, pid in (getattr(f, "keypoint_to_point", {}) or {}).items():
                if pid in seen:
                    c["inv_frame_double_link"] += 1
                seen[pid] = kp
                pt = self.get_point(pid)
                if pt is None:
                    c["inv_link_to_missing_point"] += 1
                    continue
                if (getattr(pt, "observations", {}) or {}).get(f.id) != kp:
                    c["inv_link_without_observation"] += 1
        for pt in self.points:
            obs = getattr(pt, "observations", {}) or {}
            frs = getattr(pt, "frames", []) or []
            if len(obs) != len(frs):
                c["inv_frames_obs_mismatch"] += 1
            for fid, kp in obs.items():
                f = by_id.get(fid)
                if f is None:
                    continue
                if (f.keypoint_to_point or {}).get(kp) != pt.id:
                    c["inv_observation_without_link"] += 1
        return c
