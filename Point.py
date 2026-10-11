import numpy as np

ID = 1


class Point:
    def __init__(self, point, frames=None, observations=None, descriptor=None):
        global ID
        self.id = ID
        ID += 1

        self.point = point
        self.x = float(point[0])
        self.y = float(point[1])
        self.z = float(point[2])

        self.frames = list(frames) if frames is not None else []
        # observations: {frame_id: keypoint_idx}
        self.observations = dict(observations) if observations is not None else {}
        # Representative descriptor for this map point (e.g. from frame where
        # it was triangulated). Used by local map tracking for matching;
        # avoids using a stale descriptor from an old observing frame.
        self.descriptor = np.array(descriptor, copy=True) if descriptor is not None else None

        # --- Pruning metadata (Frame.id is the time base) ---
        # created_fid: Frame.id when this point was created/triangulated.
        # last_seen_fid: most recent Frame.id that observed this point.
        # outlier_hits: count of times this point was part of a correspondence set but rejected as an outlier.
        # These fields are optional; if not explicitly set by the creator, they fall back to the earliest/latest
        # known observation frame id when available.
        try:
            fids = list(self.frames) if self.frames is not None else []
            self.created_fid = int(min(fids)) if fids else -1
            self.last_seen_fid = int(max(fids)) if fids else -1
        except Exception:
            self.created_fid = -1
            self.last_seen_fid = -1
        self.outlier_hits = 0

    ''' add an observation to the 3D map point.'''
    def add_observation(self, frame_id, keypoint_idx):
        if frame_id not in self.frames:
            self.frames.append(frame_id)
        self.observations[frame_id] = int(keypoint_idx)
        try:
            fid = int(frame_id)
            if self.created_fid < 0:
                self.created_fid = fid
            self.last_seen_fid = max(int(self.last_seen_fid), fid) if self.last_seen_fid >= 0 else fid
        except Exception:
            pass
