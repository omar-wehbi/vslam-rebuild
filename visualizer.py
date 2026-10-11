import os
import time

import cv2
import numpy as np

# The 3D viewer needs PyOpenGL and Pangolin's Python bindings (built from source). A headless run
# (--no_display) and the test suite need neither, so the import failure is deferred to the moment a
# window is actually requested (VOVisualizer.__init__) instead of breaking `import pipeline`.
try:
    import OpenGL.GL as gl
    import pypangolin as pangolin
    PANGOLIN_AVAILABLE = True
    PANGOLIN_IMPORT_ERROR = None
except ImportError as _e:
    gl = None
    pangolin = None
    PANGOLIN_AVAILABLE = False
    PANGOLIN_IMPORT_ERROR = _e


def _pix_attach(px, total=1024):
    """Pixel-based pangolin Attach with fallback to fractional."""
    try:
        return pangolin.Attach.Pix(int(px))
    except (AttributeError, TypeError):
        return pangolin.Attach(float(px) / total)


def load_ground_truth(path):
    """Load ground-truth trajectory from a TUM-format file.

    Format per line: timestamp tx ty tz qx qy qz qw
    Returns list of 4x4 world_T_cam matrices, or None on failure.
    """
    from scipy.spatial.transform import Rotation

    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
            except (ValueError, IndexError):
                continue
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses if poses else None


def build_overlay_image(frame, prev_frame=None, tracking_state="TRACKING OK", keypoints_only=False):
    """
    Build the keypoint/tracking overlay image (BGR) from frame data.
    Used by the tracker thread to produce a display snapshot without touching GL.
    If keypoints_only is True, always show only the current frame with its keypoints
    (green = linked to map, red = not); no match view between two frames.
    """
    if keypoints_only:
        prev_frame = None
        motion = None
    else:
        motion = getattr(frame, '_last_motion', None)

    if prev_frame is not None and motion is not None:
        img_prev = prev_frame.img
        img_curr = frame.img
        if len(img_prev.shape) == 2:
            img_prev = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR)
        if len(img_curr.shape) == 2:
            img_curr = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)

        img_before = cv2.drawMatches(
            img_prev, prev_frame.KeyPoints,
            img_curr, frame.KeyPoints,
            motion.all_matches[:80],
            None,
            matchColor=(0, 0, 255),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            img_before,
            f"BEFORE filtering: {len(motion.all_matches)} matches",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

        img_after = cv2.drawMatches(
            img_prev, prev_frame.KeyPoints,
            img_curr, frame.KeyPoints,
            motion.inlier_matches[:80],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            img_after,
            f"AFTER filtering: {len(motion.inlier_matches)} inliers",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        img = np.vstack([img_before, img_after])
    else:
        if len(frame.img.shape) == 2:
            img = cv2.cvtColor(frame.img, cv2.COLOR_GRAY2BGR)
        else:
            img = frame.img.copy()

        tracked_kp = set(frame.keypoint_to_point.keys())
        for idx, kp in enumerate(frame.KeyPoints):
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            if idx in tracked_kp:
                cv2.circle(img, pt, 3, (0, 255, 0), -1)
            else:
                cv2.circle(img, pt, 3, (0, 0, 255), -1)

    banner_colors = {
        "TRACKING OK": (0, 160, 0),
        "LOST": (0, 0, 200),
        "INITIALIZING": (0, 180, 220),
    }
    bg = banner_colors.get(tracking_state, (100, 100, 100))
    banner_h = 32
    cv2.rectangle(img, (0, 0), (img.shape[1], banner_h), bg, -1)
    cv2.putText(img, tracking_state, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    n_kp = len(frame.KeyPoints) if frame.KeyPoints else 0
    pos = frame.pose[:3, 3]
    h = img.shape[0]
    cv2.putText(img, f"Frame: {frame.id} | KP: {n_kp}",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    cv2.putText(img, f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 1)

    max_w = 640
    if img.shape[1] > max_w:
        scale_f = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=scale_f, fy=scale_f)

    return img


class VOVisualizer:

    PANEL_WIDTH = 200

    def __init__(self, camera_spacing=None, show_grid=True,
                 show_traj=True, ground_truth_path=None, stop_requested_check=None):
        if not PANGOLIN_AVAILABLE:
            raise RuntimeError(
                "The 3D viewer needs PyOpenGL and pypangolin (Pangolin built from source with its Python "
                "bindings; see requirements.txt). Run with --no_display for a headless run. "
                f"Import error: {PANGOLIN_IMPORT_ERROR}"
            )
        self.trajectory = []
        self.is_initialized = False
        self.camera_spacing = camera_spacing
        self.show_grid = show_grid
        self.show_traj = show_traj
        self._stop_requested_check = stop_requested_check  # callable, returns True if pipeline should stop

        self.keyframe_indices = set()
        self.frame_id_to_traj_idx = {}
        self.browse_index = 0
        self._quit_requested = False
        self._last_map_points_display = []

        # Covisibility graph cache
        self._covis_edges = {}
        self._covis_spanning_tree = set()
        self._covis_cache_frame = -1

        ''' load the ground truth trajectory if it is given'''
        self.gt_trajectory = None
        if ground_truth_path:
            self.gt_trajectory = load_ground_truth(ground_truth_path)
            if self.gt_trajectory:
                print(f"  >> Loaded ground truth: {len(self.gt_trajectory)} poses "
                      f"from {ground_truth_path}")
            else:
                print(f"  >> Warning: no poses loaded from {ground_truth_path}")
    ''' initialize the visualizer'''
    def init(self, win_w=1680, win_h=768):
        self.win_w = win_w
        self.win_h = win_h
        pw = self.PANEL_WIDTH
        # Left half: panel + 3D trajectory. Right half: keypoints image.
        self._traj_view_left = float(pw) / win_w
        self._traj_view_right = 0.5
        self._keypoints_view_left = 0.5
        self._keypoints_view_right = 1.0
        traj_view_w = int((self._traj_view_right - self._traj_view_left) * win_w)
        traj_view_h = win_h

        pangolin.CreateWindowAndBind('Visual SLAM Viewer', win_w, win_h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # ---- UI Panel (left strip, applies to trajectory view) ----
        self.panel = pangolin.CreatePanel("ui")
        self.panel.SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(0.0), _pix_attach(pw, win_w),
        )

        self.ui = pangolin.Var("ui")
        self.ui.Show_KeyFrames = True
        self.ui.Show_Points = True
        self.ui.Show_Graph = False
        self.ui.Show_Trajectory = self.show_traj
        self.ui.Show_Grid = self.show_grid
        self.ui.Follow_Camera = False
        self.ui.Show_GT = bool(self.gt_trajectory)
        self.ui.Reset_View = False

        self.ui.Tracking = "INITIALIZING"
        self.ui.Map_Points = "0"
        self.ui.Keyframes = "0"

        # ---- 3D trajectory display (left half: panel + viewport) ----
        self._initial_model_view = pangolin.ModelViewLookAt(
            -15, 10, -15, 0, 0, 0, 0, 1, 0
        )
        eye0 = np.array([-15.0, 10.0, -15.0])
        target0 = np.array([0.0, 0.0, 0.0])
        up0 = np.array([0.0, 1.0, 0.0])
        self._initial_view_gl = self._look_at_matrix(eye0, target0, up0).T.astype(
            np.float32
        )
        self._apply_reset_view = False
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(traj_view_w, traj_view_h, 420, 420,
                                      traj_view_w // 2, traj_view_h // 2, 0.2, 1000),
            self._initial_model_view,
        )
        self.handler = pangolin.Handler3D(self.scam)

        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(self._traj_view_left), pangolin.Attach(self._traj_view_right),
            -float(traj_view_w) / traj_view_h,
        )
        self.d_cam.SetHandler(self.handler)

        # ---- Keypoints window (right half: current frame keypoints only) ----
        self.d_img = pangolin.CreateDisplay()
        self.d_img.SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(self._keypoints_view_left), pangolin.Attach(self._keypoints_view_right),
        )
        self.tex = None

        # ---- Keyboard ----
        pangolin.RegisterKeyPressCallback(ord('q'), self._on_quit)

        self.is_initialized = True
        print("Visualizer initialized: two panels side by side.")
        print("Left: map/trajectory (panel toggles). Right: current frame keypoints.")
        print("Controls: Mouse drag=rotate, Scroll=zoom, Right-click drag=pan. Press 'q' to quit.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_quit(self):
        self._quit_requested = True

    # ------------------------------------------------------------------
    # Per-frame entry point
    # ------------------------------------------------------------------

    def update(self, frame, world_map=None, prev_frame=None,
               tracking_state="TRACKING OK", keyframe_indices=None):
        if not self.is_initialized:
            raise RuntimeError("Visualizer not initialized. Call init() first.")

        traj_idx = len(self.trajectory)
        self.trajectory.append(frame.pose.copy())
        self.frame_id_to_traj_idx[frame.id] = traj_idx

        if keyframe_indices is not None:
            self.keyframe_indices = set(keyframe_indices)

        # Update panel stats
        self.ui.Tracking = tracking_state
        if world_map is not None:
            self.ui.Map_Points = str(len(world_map.points))
        self.ui.Keyframes = str(len(self.keyframe_indices))

        # Render 3D scene (does NOT call FinishFrame)
        self._render_pangolin(frame.pose, world_map)

        # Render image overlay into sub-display
        self._render_keypoints(frame, prev_frame, tracking_state)

        pangolin.FinishFrame()
        time.sleep(1.0 / 30.0)  # ~30 Hz for legacy update() path
        return not self._quit_requested and not pangolin.ShouldQuit()

    def update_from_state(self, state):
        """
        Render from a snapshot (trajectory, points, overlay image).
        Used when the viewer runs in a separate thread and pulls the latest state.
        Does not sleep; caller controls frame rate.
        state: object with .trajectory, .map_points_display, .keyframe_indices,
               .tracking_state, .overlay_image (BGR numpy array)
        """
        if not self.is_initialized:
            raise RuntimeError("Visualizer not initialized. Call init() first.")

        self.trajectory = list(state.trajectory)
        self.keyframe_indices = set(state.keyframe_indices) if state.keyframe_indices is not None else set()
        self.frame_id_to_traj_idx = {i: i for i in range(len(self.trajectory))}
        self._last_map_points_display = getattr(state, 'map_points_display', None) or []

        self.ui.Tracking = state.tracking_state
        self.ui.Map_Points = str(len(self._last_map_points_display))
        self.ui.Keyframes = str(len(self.keyframe_indices))

        current_pose = self.trajectory[-1] if self.trajectory else np.eye(4, dtype=np.float64)
        self._render_pangolin(current_pose, world_map=None, points_display=self._last_map_points_display)
        if state.overlay_image is not None:
            self._upload_image(state.overlay_image)

        pangolin.FinishFrame()
        return not self._quit_requested and not pangolin.ShouldQuit()

    # ------------------------------------------------------------------
    # 3D rendering
    # ------------------------------------------------------------------

    def _render_pangolin(self, current_pose, world_map=None,
                         points_display=None, skip_camera_index=None):
        gl.glClearColor(0.05, 0.05, 0.15, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Reset View button: restore initial overview (trajectory from above)
        if getattr(self.ui, 'Reset_View', False):
            self.scam.SetModelViewMatrix(self._initial_model_view)
            self.ui.Reset_View = False
            self._apply_reset_view = True

        # Follow-camera mode: update render state so the 3D view trails the current pose
        follow = getattr(self.ui, 'Follow_Camera', False)
        if follow and len(self.trajectory) > 0:
            pos = current_pose[:3, 3]
            forward = current_pose[:3, 2]
            up = np.array([0.0, 1.0, 0.0])
            eye = pos - 8.0 * forward + 4.0 * up
            self.scam.SetModelViewMatrix(
                pangolin.ModelViewLookAt(
                    float(eye[0]), float(eye[1]), float(eye[2]),
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    float(up[0]), float(up[1]), float(up[2]),
                )
            )

        self.d_cam.Activate(self.scam)

        # Apply follow-camera or reset view in GL so it overrides Handler3D
        if follow and len(self.trajectory) > 0:
            pos = current_pose[:3, 3]
            forward = current_pose[:3, 2]
            up = np.array([0.0, 1.0, 0.0])
            eye = pos - 8.0 * forward + 4.0 * up
            view = self._look_at_matrix(eye, pos, up)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(view.T.astype(np.float32))
        elif getattr(self, '_apply_reset_view', False):
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(self._initial_view_gl)
            self._apply_reset_view = False

        if self.ui.Show_Grid:
            self._draw_grid()

        self._draw_axes(scale=5.0, line_width=4)

        if self.ui.Show_Trajectory and len(self.trajectory) > 1:
            self._draw_trajectory_line()

        if self.ui.Show_GT and self.gt_trajectory:
            self._draw_gt_trajectory()

        if self.ui.Show_Points:
            if points_display is not None and len(points_display) > 0:
                self._draw_point_cloud_from_display(points_display)
            elif world_map is not None and len(world_map.points) > 0:
                self._draw_point_cloud(world_map.points)

        if self.ui.Show_KeyFrames and len(self.trajectory) > 0:
            self._draw_all_cameras(skip_index=skip_camera_index)

        if self.ui.Show_Graph and world_map is not None:
            self._draw_covisibility_graph(world_map)

        # Current camera (always drawn, highlighted green)
        self._draw_camera_frustum(current_pose, (0.0, 1.0, 0.0), scale=1.5)
        self._draw_camera_axes(current_pose, scale=2.0)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _look_at_matrix(eye, target, up):
        """Build 4x4 OpenGL view matrix (column-major): camera at eye looking at target."""
        f = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
        f = f / (np.linalg.norm(f) + 1e-12)
        u = np.array(up, dtype=np.float64)
        u = u / (np.linalg.norm(u) + 1e-12)
        s = np.cross(f, u)
        sn = np.linalg.norm(s)
        if sn < 1e-12:
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = s / sn
        u = np.cross(s, f)
        R = np.eye(4, dtype=np.float64)
        R[:3, 0] = s
        R[:3, 1] = u
        R[:3, 2] = -f
        R[:3, 3] = -R[:3, :3] @ np.array(eye, dtype=np.float64)
        return R

    def _draw_grid(self, size=20, step=2):
        gl.glLineWidth(1)
        gl.glColor3f(0.3, 0.3, 0.3)
        gl.glBegin(gl.GL_LINES)
        for i in range(-size, size + 1, step):
            gl.glVertex3f(-size, 0, i)
            gl.glVertex3f(size, 0, i)
            gl.glVertex3f(i, 0, -size)
            gl.glVertex3f(i, 0, size)
        gl.glEnd()

    def _draw_axes(self, scale=5.0, line_width=3):
        gl.glLineWidth(line_width)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1, 0, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(scale, 0, 0)
        gl.glColor3f(0, 1, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, scale, 0)
        gl.glColor3f(0, 0, 1)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, scale)
        gl.glEnd()

    def _draw_trajectory_line(self):
        gl.glLineWidth(3)
        gl.glColor3f(1.0, 1.0, 0.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for pose in self.trajectory:
            pos = pose[:3, 3]
            gl.glVertex3f(pos[0], pos[1], pos[2])
        gl.glEnd()

    def _draw_gt_trajectory(self):
        gl.glLineWidth(2)
        gl.glColor3f(1.0, 0.0, 1.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for pose in self.gt_trajectory:
            pos = pose[:3, 3]
            gl.glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        gl.glEnd()

    # ------------------------------------------------------------------
    # Camera frustum drawing
    # ------------------------------------------------------------------

    def _draw_all_cameras(self, skip_index=None):
        n = len(self.trajectory)
        if self.camera_spacing is None:
            # Keyframe-only mode (default)
            for i in self.keyframe_indices:
                if i >= n or i == n - 1:
                    continue
                if skip_index is not None and i == skip_index:
                    continue
                self._draw_camera_frustum(
                    self.trajectory[i], (0.2, 0.4, 1.0), scale=0.8)
                self._draw_camera_axes(
                    self.trajectory[i], scale=1.0, alpha=0.6)
        else:
            # Every-N-frames mode (legacy, activated by --camera_spacing N)
            for i in range(0, n, self.camera_spacing):
                if i == n - 1:
                    continue
                if skip_index is not None and i == skip_index:
                    continue
                t = i / max(n - 1, 1)
                r = 0.5 + 0.5 * t
                g = 0.5
                b = 0.5 - 0.5 * t
                self._draw_camera_frustum(
                    self.trajectory[i], (r, g, b), scale=0.8)
                self._draw_camera_axes(
                    self.trajectory[i], scale=1.0, alpha=0.6)

    def _draw_camera_axes(self, pose, scale=1.5, alpha=1.0):
        gl.glPushMatrix()
        gl.glMultMatrixf(pose.T.flatten())
        gl.glLineWidth(2)
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(1, 0, 0, alpha)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(scale, 0, 0)
        gl.glColor4f(0, 1, 0, alpha)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, scale, 0)
        gl.glColor4f(0, 0, 1, alpha)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, scale)
        gl.glEnd()
        gl.glPopMatrix()

    def _draw_camera_frustum(self, pose, color, scale=1):
        w, h, z = 1.0 * scale, 0.75 * scale, 0.8 * scale
        gl.glPushMatrix()
        gl.glMultMatrixf(pose.T.flatten())
        gl.glLineWidth(2)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)
        for dx, dy in [(w, h), (w, -h), (-w, -h), (-w, h)]:
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(dx, dy, z)
        corners = [(w, h, z), (w, -h, z), (-w, -h, z), (-w, h, z)]
        for i in range(4):
            gl.glVertex3f(*corners[i])
            gl.glVertex3f(*corners[(i + 1) % 4])
        gl.glEnd()
        gl.glPopMatrix()

    # ------------------------------------------------------------------
    # Point cloud (batched)
    # ------------------------------------------------------------------

    def _draw_point_cloud(self, points, max_points_to_draw=150000):
        if len(points) > max_points_to_draw:
            step = max(1, len(points) // max_points_to_draw)
            points_to_draw = points[::step]
        else:
            points_to_draw = points

        if not points_to_draw:
            return

        obs_counts = np.array([len(p.frames) for p in points_to_draw],
                              dtype=np.float32)
        c_min, c_max = obs_counts.min(), obs_counts.max()
        rng = c_max - c_min if c_max > c_min else 1.0
        t = (obs_counts - c_min) / rng
        colors = self._compute_colors_batch(t)

        # Immediate-mode drawing (avoids PyOpenGL vertex-array context lookup under Pangolin).
        gl.glPointSize(2)
        gl.glBegin(gl.GL_POINTS)
        for i, p in enumerate(points_to_draw):
            gl.glColor3f(colors[i, 0], colors[i, 1], colors[i, 2])
            gl.glVertex3f(p.x, p.y, p.z)
        gl.glEnd()

    @staticmethod
    def _compute_colors_batch(t_values):
        """Vectorized blue->cyan->green->yellow->red gradient."""
        t = np.clip(t_values, 0.0, 1.0)
        colors = np.zeros((len(t), 3), dtype=np.float32)

        m1 = t < 0.25
        s = t[m1] / 0.25
        colors[m1, 1] = s
        colors[m1, 2] = 1.0

        m2 = (t >= 0.25) & (t < 0.5)
        s = (t[m2] - 0.25) / 0.25
        colors[m2, 1] = 1.0
        colors[m2, 2] = 1.0 - s

        m3 = (t >= 0.5) & (t < 0.75)
        s = (t[m3] - 0.5) / 0.25
        colors[m3, 0] = s
        colors[m3, 1] = 1.0

        m4 = t >= 0.75
        s = (t[m4] - 0.75) / 0.25
        colors[m4, 0] = 1.0
        colors[m4, 1] = 1.0 - s

        return np.ascontiguousarray(colors)

    # ------------------------------------------------------------------
    # Covisibility graph
    # ------------------------------------------------------------------

    def _build_covisibility_graph(self, world_map):
        """Incrementally cached covisibility edge computation."""
        frame_count = len(self.trajectory)
        if (self._covis_cache_frame >= 0
                and frame_count - self._covis_cache_frame < 10):
            return

        edges = {}
        for pt in world_map.points:
            traj_indices = []
            for fid in pt.frames:
                tidx = self.frame_id_to_traj_idx.get(fid)
                if tidx is not None:
                    traj_indices.append(tidx)
            for a in range(len(traj_indices)):
                for b in range(a + 1, len(traj_indices)):
                    key = (min(traj_indices[a], traj_indices[b]),
                           max(traj_indices[a], traj_indices[b]))
                    edges[key] = edges.get(key, 0) + 1

        self._covis_edges = edges
        self._covis_spanning_tree = self._compute_spanning_tree(edges)
        self._covis_cache_frame = frame_count

    def _compute_spanning_tree(self, edges):
        """Maximum spanning tree over keyframe pairs (Kruskal)."""
        kf = self.keyframe_indices
        kf_edges = [(w, i, j) for (i, j), w in edges.items()
                    if i in kf and j in kf]
        kf_edges.sort(reverse=True)

        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                # Use .get so parent[x] not yet in parent (e.g. root) does not KeyError
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b
                return True
            return False

        tree = set()
        for _, i, j in kf_edges:
            if union(i, j):
                tree.add((min(i, j), max(i, j)))
        return tree

    def _draw_covisibility_graph(self, world_map):
        self._build_covisibility_graph(world_map)
        min_shared = 15
        n = len(self.trajectory)

        # Regular covisibility edges (faded gray)
        gl.glLineWidth(1)
        gl.glBegin(gl.GL_LINES)
        for (i, j), count in self._covis_edges.items():
            if count < min_shared:
                continue
            if i >= n or j >= n:
                continue
            if (i, j) in self._covis_spanning_tree:
                continue
            gl.glColor4f(0.5, 0.5, 0.5, 0.3)
            pi = self.trajectory[i][:3, 3]
            pj = self.trajectory[j][:3, 3]
            gl.glVertex3f(float(pi[0]), float(pi[1]), float(pi[2]))
            gl.glVertex3f(float(pj[0]), float(pj[1]), float(pj[2]))
        gl.glEnd()

        # Spanning tree edges (bright cyan)
        gl.glLineWidth(2)
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0.0, 1.0, 1.0, 0.8)
        for (i, j) in self._covis_spanning_tree:
            if i >= n or j >= n:
                continue
            pi = self.trajectory[i][:3, 3]
            pj = self.trajectory[j][:3, 3]
            gl.glVertex3f(float(pi[0]), float(pi[1]), float(pi[2]))
            gl.glVertex3f(float(pj[0]), float(pj[1]), float(pj[2]))
        gl.glEnd()

    # ------------------------------------------------------------------
    # Image overlay (replaces separate OpenCV window)
    # ------------------------------------------------------------------

    def _render_keypoints(self, frame, prev_frame=None,
                          tracking_state="TRACKING OK"):
        # Right panel: current frame keypoints only (no match view)
        img = build_overlay_image(frame, prev_frame, tracking_state, keypoints_only=True)
        self._upload_image(img)

    def _draw_point_cloud_from_display(self, points_display, max_points_to_draw=150000):
        """Draw point cloud from list of (x, y, z, obs_count) tuples."""
        if not points_display:
            return
        if len(points_display) > max_points_to_draw:
            step = max(1, len(points_display) // max_points_to_draw)
            points_to_draw = points_display[::step]
        else:
            points_to_draw = points_display
        obs_counts = np.array([p[3] for p in points_to_draw], dtype=np.float32)
        c_min, c_max = obs_counts.min(), obs_counts.max()
        rng = c_max - c_min if c_max > c_min else 1.0
        t = (obs_counts - c_min) / rng
        colors = self._compute_colors_batch(t)
        gl.glPointSize(2)
        gl.glBegin(gl.GL_POINTS)
        for i, p in enumerate(points_to_draw):
            gl.glColor3f(colors[i, 0], colors[i, 1], colors[i, 2])
            gl.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        gl.glEnd()

    def _upload_image(self, img_bgr):
        """Upload an OpenCV BGR image to the Pangolin texture overlay."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        ih, iw = img_rgb.shape[:2]

        if self.tex is None or self.tex.width != iw or self.tex.height != ih:
            self.tex = pangolin.GlTexture(
                iw, ih, gl.GL_RGB, True, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        self.tex.Upload(img_rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.d_img.Activate()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glColor3f(1.0, 1.0, 1.0)
        self.tex.RenderToViewportFlipY()
        gl.glEnable(gl.GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # Post-run interactive window
    # ------------------------------------------------------------------

    def keep_window_open(self, world_map=None):
        n_traj = len(self.trajectory)
        if n_traj == 0:
            return

        spacing = self.camera_spacing if self.camera_spacing else 1
        self.browse_index = n_traj - 1

        def _browse_prev():
            self.browse_index = max(0, self.browse_index - spacing)

        def _browse_next():
            self.browse_index = min(n_traj - 1, self.browse_index + spacing)

        pangolin.RegisterKeyPressCallback(ord('a'), _browse_prev)
        pangolin.RegisterKeyPressCallback(ord('d'), _browse_next)

        print(f"\nTrajectory complete! Total frames: {n_traj}")
        print("3D window: rotate, zoom, pan. A/D: step frames. Close window or press 'q' to exit.")

        # Clear the image overlay (no keypoint/nav panel after run)
        blank = np.zeros((120, 400, 3), dtype=np.uint8)
        blank[:] = (30, 30, 35)
        self._upload_image(blank)

        points_display = None if world_map is not None else getattr(self, '_last_map_points_display', None)
        def _stop():
            if self._stop_requested_check is not None and self._stop_requested_check():
                return True
            return False
        while not pangolin.ShouldQuit() and not self._quit_requested and not _stop():
            pose = self.trajectory[self.browse_index]
            self._render_pangolin(pose, world_map,
                                  points_display=points_display,
                                  skip_camera_index=self.browse_index)

            pangolin.FinishFrame()
            time.sleep(0.05)

        # Exit process immediately to avoid GIL/tstate crash during interpreter shutdown
        os._exit(0)

    def finish(self):
        print(f"Visualization finished. Total frames: {len(self.trajectory)}")


# ======================================================================
# Module-level API (thin wrappers around a global VOVisualizer instance)
# ======================================================================

_visualizer = None


def init_visualizer(camera_spacing=None, show_grid=True,
                    show_traj=True, ground_truth_path=None, stop_requested_check=None):
    global _visualizer
    _visualizer = VOVisualizer(
        camera_spacing=camera_spacing,
        show_grid=show_grid,
        show_traj=show_traj,
        ground_truth_path=ground_truth_path,
        stop_requested_check=stop_requested_check,
    )
    _visualizer.init()


def visualize_frame(frame, world_map=None, prev_frame=None,
                    tracking_state="TRACKING OK", keyframe_indices=None):
    global _visualizer
    if _visualizer is None:
        init_visualizer()
    return _visualizer.update(
        frame, world_map, prev_frame=prev_frame,
        tracking_state=tracking_state,
        keyframe_indices=keyframe_indices,
    )


def is_quit_requested():
    """Return True if the user requested quit (e.g. pressed 'q' in the viewer)."""
    global _visualizer
    return _visualizer is not None and _visualizer._quit_requested


def update_visualizer_from_state(state):
    """Render from a state snapshot (for viewer thread). Returns False if user quit."""
    global _visualizer
    if _visualizer is None:
        return True
    return _visualizer.update_from_state(state)


def finish_visualizer(keep_open=True, world_map=None):
    global _visualizer
    if _visualizer is not None:
        if keep_open:
            _visualizer.keep_window_open(world_map)
        _visualizer.finish()
        _visualizer = None
