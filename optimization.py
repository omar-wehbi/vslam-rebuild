"""
Bundle adjustment on a local-perturbation chart (stage 01a).

The optimizer state is a tangent increment about the current estimate, applied by the
exponential map and re-linearized every iteration.  Chart (ruled 2026-09-17,
theory/conventions.md#rotation-parameterization):

    LEFT perturbation on the rotation of cam_T_world with additive translation,
        R <- exp([dtheta]x) R,   t <- t + dt,
    additive points,
        p <- p + dp.

The derivative (``BAProblem.jacobian``) and the update (``BAProblem.retract``) are on the
same side by construction; the finite-difference probes perturb through ``retract`` and
compare against ``jacobian``.

Back-end disposition (theory/design.md#foundation-repairs-that-precede-any-method-change,
item 6):
  * gauge: frames_window[0] fixed plus fixed outer observers (every processed frame
    outside the window that observes a selected point); scale is anchored only when
    enough selected points are seen by two or more distinct fixed cameras, otherwise
    frames_window[1] is fixed as well, otherwise the solve is recorded as skipped
    (theory/conventions.md#gauge-freedom-in-monocular-bundle-adjustment);
  * robust loss: Huber on the 2-vector reprojection residual, delta = 2.4477 px
    (sqrt(5.991), the ORB-SLAM monocular value), unit weight per observation;
  * update acceptance and rollback live in pipeline.py (both poses and points restored);
  * accounting: ``bundle_adjust`` returns an info dict whose ``status`` is "completed" only
    when a solve actually ran; the "BA completed" line is printed exactly once per
    completed solve and nowhere else.

The Lie-group helpers at the top are shared with loop_closure.py (import them from here).
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Lie-group helpers: SO(3) and SE(3), (omega, nu) ordering for SE(3) tangents.
# Pure numpy; no cv2.Rodrigues in any derivative path.
# --------------------------------------------------------------------------- #

_SMALL_ANGLE = 1e-6


def skew(v):
    """3x3 skew-symmetric matrix [v]x such that [v]x @ w = v x w."""
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]], dtype=np.float64)


def so3_exp(omega):
    """Exponential map so(3) -> SO(3): rotation matrix for axis-angle vector omega."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    W = skew(omega)
    if theta < _SMALL_ANGLE:
        # Second-order series; exact to O(theta^3).
        return np.eye(3) + W + 0.5 * (W @ W)
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3) + (s / theta) * W + ((1.0 - c) / (theta * theta)) * (W @ W)


def so3_log(R):
    """Logarithm SO(3) -> so(3): axis-angle vector with angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    if theta < _SMALL_ANGLE:
        return 0.5 * vee
    if theta > np.pi - 1e-4:
        # Near pi: sin(theta) ~ 0, recover the axis from the symmetric part
        # R = I + (1 - cos) [k]x^2 (+ sin [k]x) => (R + R^T)/2 - I = (1 - cos) ([k]x)^2
        # and (k k^T) = I + ((R + R^T)/2 - I) / (1 - cos).
        S = 0.5 * (R + R.T)
        M = np.eye(3) + (S - np.eye(3)) / max(1.0 - cos_theta, 1e-12)  # = k k^T
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        # Fix the sign using the antisymmetric part where it is informative.
        if np.dot(k, vee) < 0.0:
            k = -k
        return theta * k
    return (theta / (2.0 * np.sin(theta))) * vee


def so3_left_jacobian(omega):
    """Left Jacobian J_l of SO(3): so3_exp(omega + d) ~= so3_exp(J_l d) so3_exp(omega)."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    W = skew(omega)
    if theta < 1e-4:
        return np.eye(3) + 0.5 * W + (W @ W) / 6.0
    s, c = np.sin(theta), np.cos(theta)
    t2, t3 = theta * theta, theta * theta * theta
    return np.eye(3) + ((1.0 - c) / t2) * W + ((theta - s) / t3) * (W @ W)


def so3_left_jacobian_inv(omega):
    """Inverse of the left Jacobian of SO(3)."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    W = skew(omega)
    if theta < 1e-4:
        return np.eye(3) - 0.5 * W + (W @ W) / 12.0
    s, c = np.sin(theta), np.cos(theta)
    coef = 1.0 / (theta * theta) - (1.0 + c) / (2.0 * theta * s)
    return np.eye(3) - 0.5 * W + coef * (W @ W)


def so3_right_jacobian(omega):
    """Right Jacobian J_r(omega) = J_l(-omega)."""
    return so3_left_jacobian(-np.asarray(omega, dtype=np.float64))


def so3_right_jacobian_inv(omega):
    """Inverse right Jacobian of SO(3): J_r^-1(omega) = J_l^-1(-omega)."""
    return so3_left_jacobian_inv(-np.asarray(omega, dtype=np.float64))


def se3_exp(xi):
    """Exponential map se(3) -> SE(3) for xi = (omega, nu): R = exp(omega), t = J_l(omega) nu."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, nu = xi[:3], xi[3:]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = so3_left_jacobian(omega) @ nu
    return T


def se3_log(T):
    """Logarithm SE(3) -> se(3): (omega, nu) with omega = log(R), nu = J_l(omega)^-1 t."""
    T = np.asarray(T, dtype=np.float64)
    omega = so3_log(T[:3, :3])
    nu = so3_left_jacobian_inv(omega) @ T[:3, 3]
    return np.concatenate([omega, nu])


def se3_inv(T):
    """Inverse of a 4x4 rigid transform (uses R^T; cheaper and better conditioned than inv)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def se3_adjoint(T):
    """SE(3) adjoint Ad(T) with (omega, nu) ordering: T Exp(xi) T^-1 = Exp(Ad(T) xi)."""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


def _se3_Q(rho, phi):
    """
    Barfoot's Q(rho, phi) block (State Estimation for Robotics, eq. 7.86), the lower-left
    3x3 block of the SE(3) left Jacobian with (phi = rotation, rho = translation) tangent.
    """
    theta = float(np.linalg.norm(phi))
    Px, Rx = skew(phi), skew(rho)
    if theta < 1e-2:
        # Taylor series of the three closed-form coefficients of the large-angle branch
        # (cancellation-free for small theta):
        #   a  = (theta - sin theta) / theta^3               = 1/6 - theta^2/120 + ...
        #   b  = (1 - theta^2/2 - cos theta) / theta^4       = -1/24 + theta^2/720 + ...
        #   c3 = (theta - sin theta - theta^3/6) / theta^5   = -1/120 + theta^2/5040 + ...
        t2 = theta * theta
        a = 1.0 / 6.0 - t2 / 120.0
        b = -1.0 / 24.0 + t2 / 720.0
        c3 = -1.0 / 120.0 + t2 / 5040.0
    else:
        s, c = np.sin(theta), np.cos(theta)
        a = (theta - s) / theta ** 3
        b = (1.0 - 0.5 * theta * theta - c) / theta ** 4
        c3 = (theta - s - theta ** 3 / 6.0) / theta ** 5
    d = 0.5 * (b - 3.0 * c3)
    return (0.5 * Rx
            + a * (Px @ Rx + Rx @ Px + Px @ Rx @ Px)
            - b * (Px @ Px @ Rx + Rx @ Px @ Px - 3.0 * Px @ Rx @ Px)
            - d * (Px @ Rx @ Px @ Px + Px @ Px @ Rx @ Px))


def se3_left_jacobian(xi):
    """Left Jacobian of SE(3) at xi = (omega, nu): se3_exp(xi + d) ~= se3_exp(J_l d) se3_exp(xi)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, nu = xi[:3], xi[3:]
    Jw = so3_left_jacobian(omega)
    J = np.zeros((6, 6), dtype=np.float64)
    J[:3, :3] = Jw
    J[3:, 3:] = Jw
    J[3:, :3] = _se3_Q(nu, omega)
    return J


def se3_left_jacobian_inv(xi):
    """Inverse of the SE(3) left Jacobian (closed-form block inverse)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, nu = xi[:3], xi[3:]
    Jw_inv = so3_left_jacobian_inv(omega)
    Q = _se3_Q(nu, omega)
    J = np.zeros((6, 6), dtype=np.float64)
    J[:3, :3] = Jw_inv
    J[3:, 3:] = Jw_inv
    J[3:, :3] = -Jw_inv @ Q @ Jw_inv
    return J


def se3_right_jacobian(xi):
    """Right Jacobian of SE(3): se3_exp(xi + d) ~= se3_exp(xi) se3_exp(J_r d); J_r(xi) = J_l(-xi)."""
    return se3_left_jacobian(-np.asarray(xi, dtype=np.float64))


def se3_right_jacobian_inv(xi):
    """Inverse right Jacobian of SE(3): J_r^-1(xi) = J_l^-1(-xi)."""
    return se3_left_jacobian_inv(-np.asarray(xi, dtype=np.float64))


# --------------------------------------------------------------------------- #
# The BA problem object (spec A2)
# --------------------------------------------------------------------------- #

BEHIND_CAMERA_Z = 1e-6
BEHIND_CAMERA_RESIDUAL = 100.0
HUBER_DELTA_PX_DEFAULT = 2.4477  # sqrt(5.991), ORB-SLAM monocular
# Stage 04y: the local-BA outlier test of ORB-SLAM section VI.D, chi-square with 2 degrees of
# freedom at 95 percent on (e / sigma)^2.  The paper gives no number; 5.991 is the ORB-SLAM
# implementation's value, recorded here as an implementation constant (never tuned).
BA_OUTLIER_CHI2 = 5.991

# Stage 04z (ruling R57-2, conventions.md#local-ba-participation): a local-BA window camera is a free
# variable only when it observes at least this many points in the BA problem (O3+ form of "supported").
# The tracker's acceptance count (R56-4), carried to BA by analogy with ORB-SLAM section V.E (the paper's
# 50 tracked points for a keyframe); not tuned against a run.
BA_MIN_FREE_CAMERA_POINTS = 30


class BAProblem:
    """
    Bundle-adjustment problem on the left-perturbation chart.

    State: cam_T_world (n_cams, 4, 4), points (n_points, 3), fixed cameras contribute
    residual rows but no columns.  Observation k owns residual rows 2k (u) and 2k+1 (v).
    Column layout of the tangent increment delta (also the Jacobian columns):
      for the c-th free camera: 6c..6c+2 = dtheta (R <- exp([dtheta]x) R),
                                6c+3..6c+5 = dt (t <- t + dt);
      for point p: 6*n_free + 3p .. +2 = dp (p <- p + dp).
    """

    def __init__(self, K, cam_T_world, fixed_mask, points, cam_idx, pt_idx, uv,
                 huber_delta_px=HUBER_DELTA_PX_DEFAULT):
        # Private constructor: arrays are taken as-is (already copied by from_arrays/retract).
        self.K = K
        self.cam_T_world = cam_T_world
        self.fixed_mask = fixed_mask
        self._points = points
        self.cam_idx = cam_idx
        self.pt_idx = pt_idx
        self.uv = uv
        self.huber_delta_px = float(huber_delta_px)

        self.n_cams = int(cam_T_world.shape[0])
        self.free_cam_indices = [int(c) for c in np.flatnonzero(~fixed_mask)]
        self.n_free = len(self.free_cam_indices)
        self.n_fixed = self.n_cams - self.n_free
        self.n_points = int(points.shape[0])
        self.n_obs = int(cam_idx.shape[0])
        self.x_dim = 6 * self.n_free + 3 * self.n_points
        # camera index -> free column block index (or -1 when fixed)
        self._free_col = np.full(self.n_cams, -1, dtype=np.int64)
        self._free_col[self.free_cam_indices] = np.arange(self.n_free)

    @classmethod
    def from_arrays(cls, K, cam_T_world, fixed_mask, points, observations,
                    huber_delta_px=HUBER_DELTA_PX_DEFAULT):
        """
        Build a problem from plain arrays.  All inputs are copied; the problem never holds
        live references to Frame or Point objects.

        K: (3,3).  cam_T_world: (n_cams,4,4).  fixed_mask: (n_cams,) bool, True = held fixed.
        points: (n_pts,3).  observations: sequence of (cam_idx, pt_idx, u, v); order preserved.
        """
        K = np.array(K, dtype=np.float64, copy=True).reshape(3, 3)
        cam_T_world = np.array(cam_T_world, dtype=np.float64, copy=True)
        if cam_T_world.ndim != 3 or cam_T_world.shape[1:] != (4, 4):
            raise ValueError(f"cam_T_world must be (n_cams,4,4), got {cam_T_world.shape}")
        n_cams = cam_T_world.shape[0]
        fixed_mask = np.array(fixed_mask, dtype=bool, copy=True).reshape(n_cams)
        points = np.array(points, dtype=np.float64, copy=True).reshape(-1, 3)
        obs = list(observations)
        n_obs = len(obs)
        cam_idx = np.empty(n_obs, dtype=np.int64)
        pt_idx = np.empty(n_obs, dtype=np.int64)
        uv = np.empty((n_obs, 2), dtype=np.float64)
        for k, (c, p, u, v) in enumerate(obs):
            cam_idx[k] = int(c)
            pt_idx[k] = int(p)
            uv[k, 0] = float(u)
            uv[k, 1] = float(v)
        if n_obs and (cam_idx.min() < 0 or cam_idx.max() >= n_cams):
            raise ValueError("observation camera index out of range")
        if n_obs and (pt_idx.min() < 0 or pt_idx.max() >= points.shape[0]):
            raise ValueError("observation point index out of range")
        return cls(K, cam_T_world, fixed_mask, points, cam_idx, pt_idx, uv, huber_delta_px)

    # -- geometry ----------------------------------------------------------------

    def _camera_coords(self):
        """x_cam (n_obs,3) = R p + t per observation, and the per-observation R (n_obs,3,3)."""
        R = self.cam_T_world[self.cam_idx, :3, :3]          # (n_obs,3,3)
        t = self.cam_T_world[self.cam_idx, :3, 3]           # (n_obs,3)
        P = self._points[self.pt_idx]                       # (n_obs,3)
        Rp = np.einsum("kij,kj->ki", R, P)
        return Rp + t, Rp, R

    def _project(self, x_cam):
        """Perspective projection through K; returns uv_proj (n_obs,2) and depth z (n_obs,)."""
        h = x_cam @ self.K.T                                # (n_obs,3)
        z = h[:, 2]
        safe_z = np.where(np.abs(z) > 0.0, z, 1.0)
        return h[:, :2] / safe_z[:, None], z

    def residuals(self):
        """RAW (unweighted) reprojection residuals (2*n_obs,), [u_proj-u_obs, v_proj-v_obs]."""
        x_cam, _, _ = self._camera_coords()
        uv_proj, _ = self._project(x_cam)
        behind = x_cam[:, 2] <= BEHIND_CAMERA_Z
        r = uv_proj - self.uv
        r[behind, :] = BEHIND_CAMERA_RESIDUAL
        return r.reshape(-1)

    def behind_mask(self):
        """(n_obs,) bool: observations whose point is behind (or at) its camera."""
        x_cam, _, _ = self._camera_coords()
        return x_cam[:, 2] <= BEHIND_CAMERA_Z

    def jacobian(self):
        """
        Dense (2*n_obs, x_dim) derivative of residuals() with respect to the tangent
        increment delta at delta = 0 (see class docstring for the column layout).
        Per observation, with x_cam = R p + t and P = d(u,v)/d x_cam (2x3):
          d r / d dtheta = P @ (-[R p]x)   (left perturbation)
          d r / d dt     = P
          d r / d dp     = P @ R
        Behind-camera observations get zero rows.
        """
        x_cam, Rp, R = self._camera_coords()
        uv_proj, z = self._project(x_cam)
        behind = x_cam[:, 2] <= BEHIND_CAMERA_Z
        safe_z = np.where(behind, 1.0, z)
        # P_k = (K[:2,:] - uv_k K[2,:]) / z_k  (general form; K[2,:] = [0,0,1] for a pinhole)
        Pm = (self.K[None, :2, :] - uv_proj[:, :, None] * self.K[None, 2, :][:, None, :]) / safe_z[:, None, None]
        Pm[behind] = 0.0

        J = np.zeros((2 * self.n_obs, self.x_dim), dtype=np.float64)
        rows = (2 * np.arange(self.n_obs))[:, None] + np.arange(2)[None, :]   # (n_obs,2)
        # Camera blocks (free cameras only).
        free_col = self._free_col[self.cam_idx]             # (n_obs,)
        has_col = np.flatnonzero(free_col >= 0)
        if has_col.size:
            Rp_skew = np.zeros((self.n_obs, 3, 3), dtype=np.float64)
            Rp_skew[:, 0, 1] = -Rp[:, 2]
            Rp_skew[:, 0, 2] = Rp[:, 1]
            Rp_skew[:, 1, 0] = Rp[:, 2]
            Rp_skew[:, 1, 2] = -Rp[:, 0]
            Rp_skew[:, 2, 0] = -Rp[:, 1]
            Rp_skew[:, 2, 1] = Rp[:, 0]
            d_rot = -np.einsum("kij,kjl->kil", Pm, Rp_skew)  # (n_obs,2,3)
            c0 = (6 * free_col[has_col])[:, None, None]      # (m,1,1)
            rr = rows[has_col][:, :, None]                    # (m,2,1)
            cc = np.arange(3)[None, None, :]                  # (1,1,3)
            J[rr, c0 + cc] = d_rot[has_col]
            J[rr, c0 + 3 + cc] = Pm[has_col]
        # Point blocks.
        d_pt = np.einsum("kij,kjl->kil", Pm, R)             # (n_obs,2,3)
        p0 = (6 * self.n_free + 3 * self.pt_idx)[:, None, None]
        J[rows[:, :, None], p0 + np.arange(3)[None, None, :]] = d_pt
        return J

    def retract(self, delta):
        """NEW problem with the base state moved by delta (same layout as jacobian columns)."""
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        if delta.shape[0] != self.x_dim:
            raise ValueError(f"delta must have length {self.x_dim}, got {delta.shape[0]}")
        cam = self.cam_T_world.copy()
        for i, c in enumerate(self.free_cam_indices):
            d = delta[6 * i:6 * i + 6]
            cam[c, :3, :3] = so3_exp(d[:3]) @ cam[c, :3, :3]
            cam[c, :3, 3] = cam[c, :3, 3] + d[3:6]
        pts = self._points + delta[6 * self.n_free:].reshape(-1, 3)
        return BAProblem(self.K, cam, self.fixed_mask.copy(), pts,
                         self.cam_idx, self.pt_idx, self.uv, self.huber_delta_px)

    def poses(self):
        """(n_cams,4,4) cam_T_world of ALL cameras (fixed ones unchanged)."""
        return self.cam_T_world.copy()

    def points(self):
        """(n_points,3) point coordinates."""
        return self._points.copy()

    # -- robust loss -------------------------------------------------------------

    def _obs_norms(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 2)
        return np.linalg.norm(r, axis=1)

    def robust_weights(self, r):
        """(n_obs,) Huber weights on e_k = ||r[2k:2k+2]||: 1 if e <= delta else delta / e."""
        e = self._obs_norms(r)
        d = self.huber_delta_px
        w = np.ones_like(e)
        big = e > d
        w[big] = d / e[big]
        return w

    def cost(self, r):
        """sum_k rho(e_k) with Huber rho(e) = e^2/2 if e <= delta else delta (e - delta/2)."""
        e = self._obs_norms(r)
        d = self.huber_delta_px
        rho = np.where(e <= d, 0.5 * e * e, d * (e - 0.5 * d))
        return float(np.sum(rho))

    def mean_reprojection_error(self):
        """Mean of e_k over ALL observations (fixed cameras included), excluding behind-camera ones."""
        x_cam, _, _ = self._camera_coords()
        uv_proj, _ = self._project(x_cam)
        ok = x_cam[:, 2] > BEHIND_CAMERA_Z
        if not np.any(ok):
            return 0.0
        e = np.linalg.norm(uv_proj[ok] - self.uv[ok], axis=1)
        return float(np.mean(e))


# --------------------------------------------------------------------------- #
# Solver (spec A3): re-linearize every iteration; never differentiate at nonzero delta.
# --------------------------------------------------------------------------- #

def finite_difference_jacobian(problem, eps=1e-6):
    """Central finite-difference Jacobian of problem.residuals() through problem.retract()."""
    r0 = np.asarray(problem.residuals(), dtype=np.float64)
    J = np.empty((r0.shape[0], problem.x_dim), dtype=np.float64)
    for j in range(problem.x_dim):
        e = np.zeros(problem.x_dim)
        e[j] = eps
        rp = np.asarray(problem.retract(e).residuals(), dtype=np.float64)
        rm = np.asarray(problem.retract(-e).residuals(), dtype=np.float64)
        J[:, j] = (rp - rm) / (2.0 * eps)
    return J


def _solve_normal_equations(A, g):
    """Dense solve of A delta = -g with a least-squares fallback."""
    try:
        delta = np.linalg.solve(A, -g)
        if np.all(np.isfinite(delta)):
            return delta
    except np.linalg.LinAlgError:
        pass
    delta = np.linalg.lstsq(A, -g, rcond=None)[0]
    return delta


def _weighted_normal_equations(J, w_obs, r):
    """A = J^T W J and g = J^T W r with W = diag(w_obs repeated per residual row)."""
    w_rows = np.repeat(w_obs, 2)
    JW = J * w_rows[:, None]
    A = J.T @ JW
    g = JW.T @ r
    return A, g


def solve_ba(problem, *, method="lm", max_iterations=10, lambda_init=1e-3,
             lambda_up=10.0, lambda_down=0.1, cost_rel_tol=1e-6, step_tol=1e-8,
             use_analytical_jacobian=True):
    """
    Robust (Huber-weighted) least squares on the chart.  Returns (solved_problem, info).

    method "lm": Levenberg-Marquardt with diagonal damping, accept iff the Huber cost
    decreases, at most 5 retries per linearization.  "newton": the same loop started with
    lambda = 0 (Gauss-Newton), damping only after a rejected step.  "gradient_descent":
    steepest descent delta = -alpha J^T W r, alpha initialised each iteration at the Cauchy
    step (the minimiser of the quadratic model along -g) and halved on rejection
    (backtracking on the cost, up to 20 halvings).  All three share residuals() /
    jacobian() / retract() and re-linearize at the accepted state.
    """
    if method not in ("lm", "newton", "gradient_descent"):
        raise ValueError(f"Unknown ba method: {method}")
    max_retries = 20 if method == "gradient_descent" else 5

    def jac(p):
        return p.jacobian() if use_analytical_jacobian else finite_difference_jacobian(p)

    r = np.asarray(problem.residuals(), dtype=np.float64)
    cost = problem.cost(r)
    cost_initial = cost
    lam = 0.0 if method == "newton" else float(lambda_init)
    alpha = 1.0  # gradient-descent step length (reset to the Cauchy step each iteration)
    iterations = 0
    converged = False
    stop_reason = "max_iterations"
    n_rejected_total = 0

    for _ in range(int(max_iterations)):
        if cost <= 0.0:
            converged = True
            stop_reason = "zero_cost"
            break
        iterations += 1
        J = jac(problem)
        w = problem.robust_weights(r)
        A, g = _weighted_normal_equations(J, w, r)
        accepted = False
        delta = np.zeros(problem.x_dim)
        if method == "gradient_descent":
            gAg = float(g @ (A @ g))
            gg = float(g @ g)
            alpha = gg / gAg if gAg > 0.0 else 1.0
        for _retry in range(max_retries + 1):
            if method == "gradient_descent":
                delta = -alpha * g
            else:
                D = np.diag(np.diag(A))
                delta = _solve_normal_equations(A + lam * D, g)
            candidate = problem.retract(delta)
            r_c = np.asarray(candidate.residuals(), dtype=np.float64)
            cost_c = candidate.cost(r_c)
            if np.isfinite(cost_c) and cost_c < cost:
                accepted = True
                break
            # rejected: increase damping (or shrink the descent step) and retry the same linearization
            n_rejected_total += 1
            if method == "gradient_descent":
                alpha *= 0.5
            else:
                lam = lam * lambda_up if lam > 0.0 else float(lambda_init)
        if not accepted:
            stop_reason = "rejected_steps"
            break
        rel_decrease = (cost - cost_c) / cost if cost > 0.0 else 0.0
        problem, r, cost = candidate, r_c, cost_c
        if method != "gradient_descent":
            lam = lam * lambda_down
        step_norm = float(np.linalg.norm(delta))
        if step_norm < step_tol:
            converged = True
            stop_reason = "step_tol"
            break
        if rel_decrease < cost_rel_tol:
            converged = True
            stop_reason = "cost_rel_tol"
            break

    behind = problem.behind_mask()
    fixed_obs = problem.fixed_mask[problem.cam_idx]
    info = {
        "status": "completed",
        "method": method,
        "iterations": iterations,
        "cost_initial": float(cost_initial),
        "cost_final": float(cost),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "n_rejected_steps": int(n_rejected_total),
        "n_free": problem.n_free,
        "n_fixed": problem.n_fixed,
        "n_points": problem.n_points,
        "n_obs": problem.n_obs,
        "n_obs_fixed": int(np.count_nonzero(fixed_obs)),
        "n_obs_behind": int(np.count_nonzero(behind)),
        "gauge": None,
    }
    return problem, info


# --------------------------------------------------------------------------- #
# Window builder and gauge (spec A4): the pipeline adapter.
# --------------------------------------------------------------------------- #

def _skipped(reason, **extra):
    d = {"status": "skipped", "reason": reason}
    d.update(extra)
    return None, d


def build_ba_problem(frames_window, world_map, K, *, all_frames=None, max_points=250,
                     min_point_observations=3, point_sanity_check=True,
                     max_fixed_observers=0, huber_delta_px=HUBER_DELTA_PX_DEFAULT,
                     min_anchor_points=10, min_camera_points=0, camera_supported=None):
    """
    Build a BAProblem for a window of frames.  Returns (problem, meta) or (None, meta) with
    meta["status"] == "skipped" and meta["reason"] in
    {window_too_small, too_few_points, too_few_observations, gauge_unanchored, underdetermined}
    (stage 04z adds no_supported_camera, only with min_camera_points > 0).

    Point selection: points linked from any window frame with >= min_point_observations
    observations (finite coordinates when point_sanity_check), top max_points by observation
    count, ties by point id ascending.  Observations come from frame.keypoint_to_point.

    Anchoring: frames_window[0] fixed; every frame of all_frames outside the window with
    pose_valid True that links a selected point is a FIXED outer observer (its observations
    of selected points are residual rows).  Scale is anchored when >= min_anchor_points
    selected points are observed by two or more distinct fixed cameras (gauge
    "fixed_observers"); otherwise frames_window[1] is fixed too (gauge "two_fixed"), and if
    the window has < 3 frames or the two fixed window frames have baseline < 1e-9 the build
    is skipped with reason "gauge_unanchored".

    Stage 04z (R57-2): with min_camera_points m > 0, every window camera other than index 0 with
    fewer than m residual rows (selected points it observes) is HELD fixed before the gauge step and
    counts as a fixed camera for the anchor count, exactly as an outer observer; when no camera is
    left free (after the hold, or after the gauge step) the build is skipped with reason
    "no_supported_camera".  m == 0 is the stage-01a build.  meta carries held_frame_ids,
    n_held_build, free_min_rows (-1 when no camera is free) and min_camera_points.

    Stage 04-m0 (the M0 form of R57-2, conventions.md#local-ba-participation): with a callable
    camera_supported (frame -> bool) the hold above tests ``not camera_supported(frame)`` in place of
    the row count, for the same cameras (index 1 to the window's end); the rule is then on whatever
    min_camera_points is, with the same "no_supported_camera" skips.  None is the 04z build.
    """
    frames_window = list(frames_window)
    m_cam = int(min_camera_points)
    rule_on = m_cam > 0 or camera_supported is not None
    if len(frames_window) < 2:
        return _skipped_and_print("window_too_small")

    window_ids = {f.id for f in frames_window}

    # --- 1. Point selection --------------------------------------------------------
    candidates = {}
    for f in frames_window:
        for _kp_idx, pid in f.keypoint_to_point.items():
            if pid in candidates:
                continue
            pt = world_map.get_point(pid)
            if pt is None:
                continue
            if len(getattr(pt, "frames", [])) < int(min_point_observations):
                continue
            x = getattr(pt, "point", None)
            if point_sanity_check:
                if x is None or not np.all(np.isfinite(np.asarray(x, dtype=np.float64))):
                    continue
            elif x is None:
                continue
            candidates[pid] = len(getattr(pt, "observations", {}) or {})
    point_ids = sorted(candidates.keys(), key=lambda pid: (-candidates[pid], pid))
    if len(point_ids) > int(max_points):
        point_ids = point_ids[:int(max_points)]
    if len(point_ids) < 5:
        return _skipped_and_print("too_few_points", n_points=len(point_ids))
    pid_to_local = {pid: i for i, pid in enumerate(point_ids)}

    # --- 2. Observations from the window frames -----------------------------------
    obs_keys = []   # stage 04y: per residual row (camera index, frame id, keypoint index, point id)

    def frame_observations(f, cam_local):
        out = []
        for kp_idx, pid in f.keypoint_to_point.items():
            pi = pid_to_local.get(pid)
            if pi is None:
                continue
            u, v = f.KeyPoints[kp_idx].pt
            out.append((cam_local, pi, float(u), float(v)))
            obs_keys.append((cam_local, f.id, int(kp_idx), pid))
        return out

    observations = []
    for ci, f in enumerate(frames_window):
        observations.extend(frame_observations(f, ci))

    # --- 3. Fixed outer observers ---------------------------------------------------
    outer = []
    if all_frames is not None:
        selected = set(point_ids)
        seen = set()
        for f in all_frames:
            if f.id in window_ids or f.id in seen:
                continue
            seen.add(f.id)
            if not getattr(f, "pose_valid", True):
                continue
            k2p = getattr(f, "keypoint_to_point", None)
            if not k2p:
                continue
            if any(pid in selected for pid in k2p.values()):
                outer.append(f)
        outer.sort(key=lambda f: f.id)
        if int(max_fixed_observers) > 0 and len(outer) > int(max_fixed_observers):
            outer = outer[-int(max_fixed_observers):]
    n_window = len(frames_window)
    for oi, f in enumerate(outer):
        observations.extend(frame_observations(f, n_window + oi))

    if len(observations) < 10:
        return _skipped_and_print("too_few_observations", n_obs=len(observations),
                                  n_points=len(point_ids))

    cams = frames_window + outer
    obs_arr = np.array([(c, p) for c, p, _, _ in observations], dtype=np.int64).reshape(-1, 2)
    # Stage 04z: residual rows per camera (one row per keypoint linked to a selected point).
    rows_per_cam = np.bincount(obs_arr[:, 0], minlength=len(cams))

    # --- 3b. Stage 04z participation rule (R57-2): thin window cameras are held -----------
    held = np.zeros(len(cams), dtype=bool)
    if rule_on:
        for c in range(1, n_window):
            if camera_supported is not None:
                if not camera_supported(frames_window[c]):
                    held[c] = True
            elif int(rows_per_cam[c]) < m_cam:
                held[c] = True
        if bool(np.all(held[1:n_window])):
            return _skipped_and_print("no_supported_camera", n_held_build=int(np.count_nonzero(held)))

    # --- 4. Gauge ---------------------------------------------------------------------
    fixed_mask = np.zeros(len(cams), dtype=bool)
    fixed_mask[0] = True
    fixed_mask[n_window:] = True
    fixed_mask |= held
    fixed_cams_of_point = {}
    for c, p in obs_arr:
        if fixed_mask[c]:
            fixed_cams_of_point.setdefault(int(p), set()).add(int(c))
    n_anchor = sum(1 for s in fixed_cams_of_point.values() if len(s) >= 2)
    if n_anchor >= int(min_anchor_points):
        gauge = "fixed_observers"
    else:
        gauge = "two_fixed"
        if n_window < 3:
            return _skipped_and_print("gauge_unanchored", n_anchor_points=n_anchor,
                                      n_fixed_observers=len(outer))
        c0 = np.linalg.inv(frames_window[0].cam_T_world)[:3, 3]
        c1 = np.linalg.inv(frames_window[1].cam_T_world)[:3, 3]
        if float(np.linalg.norm(c1 - c0)) < 1e-9:
            return _skipped_and_print("gauge_unanchored", n_anchor_points=n_anchor,
                                      n_fixed_observers=len(outer))
        fixed_mask[1] = True

    n_free = int(np.count_nonzero(~fixed_mask))
    if rule_on and n_free == 0:
        return _skipped_and_print("no_supported_camera", n_held_build=int(np.count_nonzero(held)))
    x_dim = 6 * n_free + 3 * len(point_ids)
    if 2 * len(observations) < x_dim:
        return _skipped_and_print("underdetermined", n_obs=len(observations), x_dim=x_dim)

    cam_T_world = np.array([f.cam_T_world for f in cams], dtype=np.float64)
    points = np.array([np.asarray(world_map.get_point(pid).point, dtype=np.float64).reshape(3)
                       for pid in point_ids], dtype=np.float64)
    problem = BAProblem.from_arrays(K, cam_T_world, fixed_mask, points, observations,
                                    huber_delta_px=huber_delta_px)
    meta = {
        "status": "built",
        "gauge": gauge,
        "n_anchor_points": int(n_anchor),
        "n_fixed_observers": len(outer),
        "fixed_frame_ids": [cams[c].id for c in range(len(cams)) if fixed_mask[c]],
        "free_frame_ids": [cams[c].id for c in range(len(cams)) if not fixed_mask[c]],
        "point_ids": list(point_ids),
        "held_frame_ids": [cams[c].id for c in range(n_window) if held[c]],
        "n_held_build": int(np.count_nonzero(held)),
        "free_min_rows": (int(np.min(rows_per_cam[~fixed_mask])) if n_free > 0 else -1),
        "min_camera_points": m_cam,
        "_cams": cams,  # internal: Frame objects in camera order, for write-back
        "_obs_keys": obs_keys,  # internal (04y): per residual row (cam idx, frame id, kp idx, point id)
    }
    return problem, meta


def _skipped_and_print(reason, **extra):
    print(f"  >> BA skipped: {reason}")
    return _skipped(reason, **extra)


# --------------------------------------------------------------------------- #
# Pipeline entry point (spec A5)
# --------------------------------------------------------------------------- #

def bundle_adjust(frames_window, world_map, K, max_points=250, method='lm',
                  use_analytical_jacobian=True,
                  min_point_observations=3,
                  point_sanity_check=True, *, all_frames=None, max_fixed_observers=0,
                  huber_delta_px=HUBER_DELTA_PX_DEFAULT, min_anchor_points=10,
                  max_iterations=10, reject_outliers=False, sigma_fn=None,
                  min_camera_points=0, camera_supported=None):
    """
    Run local bundle adjustment on a window of frames and write the result back.

    Returns the info dict.  status "skipped" (with reason) writes nothing.  status
    "completed" writes the free cameras' poses through Frame.set_camera_pose and the
    selected points' coordinates, and prints exactly one "BA completed" line.  The info
    dict carries reproj_before_px / reproj_after_px (mean reprojection error over the
    problem's observations, fixed observers included), fixed_frame_ids, free_frame_ids,
    point_ids and gauge.

    Stage 04y: reject_outliers=True runs the two-phase solve of ORB-SLAM section VI.D
    (see _bundle_adjust_reject_outliers); sigma_fn (frame -> per-keypoint sigma array) is
    then required.  reject_outliers=False is the stage-01a solve, unchanged.

    Stage 04z: min_camera_points (default 0) is handed to build_ba_problem (the participation
    rule of R57-2) and to the phase-2 test of the outlier step.  info carries the R57-3 fields
    p2_min_rows, p2_under10, n_held_p2 and p2_ran (-1, -1, 0, 0 without the outlier step).

    Stage 04-m0: camera_supported (default None, frame -> bool) is handed to build_ba_problem (the
    M0 form of the participation rule); the phase-2 test is unchanged.  None adds no info key.
    """
    if reject_outliers and (sigma_fn is None or not callable(sigma_fn)):
        raise ValueError("bundle_adjust: reject_outliers=True needs a callable sigma_fn (frame -> sigmas)")
    problem, meta = build_ba_problem(
        frames_window, world_map, K, all_frames=all_frames, max_points=max_points,
        min_point_observations=min_point_observations,
        point_sanity_check=point_sanity_check, max_fixed_observers=max_fixed_observers,
        huber_delta_px=huber_delta_px, min_anchor_points=min_anchor_points,
        min_camera_points=min_camera_points, camera_supported=camera_supported,
    )
    if problem is None:
        return meta

    cams = meta.pop("_cams")
    obs_keys = meta.pop("_obs_keys")
    if reject_outliers:
        return _bundle_adjust_reject_outliers(
            frames_window, world_map, problem, meta, cams, obs_keys, sigma_fn,
            method=method, max_iterations=max_iterations,
            use_analytical_jacobian=use_analytical_jacobian,
            min_camera_points=min_camera_points)
    reproj_before = problem.mean_reprojection_error()
    solved, info = solve_ba(problem, method=method, max_iterations=max_iterations,
                            use_analytical_jacobian=use_analytical_jacobian)
    reproj_after = solved.mean_reprojection_error()

    # --- Write back: free cameras through the setter, selected points' coordinates ---
    poses = solved.poses()
    for c in solved.free_cam_indices:
        cams[c].set_camera_pose(poses[c])
    pts = solved.points()
    for j, pid in enumerate(meta["point_ids"]):
        pt = world_map.get_point(pid)
        if pt is None:
            continue
        new_xyz = np.array(pts[j], dtype=np.float64, copy=True)
        pt.point = new_xyz
        pt.x = float(new_xyz[0])
        pt.y = float(new_xyz[1])
        pt.z = float(new_xyz[2])

    info.update({k: v for k, v in meta.items() if k != "status"})
    info["reproj_before_px"] = float(reproj_before)
    info["reproj_after_px"] = float(reproj_after)
    info["n_frames_window"] = len(frames_window)
    # Stage 04z (R57-3): no outlier step, no phase 2.
    info["p2_min_rows"] = -1
    info["p2_under10"] = -1
    info["n_held_p2"] = 0
    info["p2_ran"] = 0
    print(
        f"  >> BA completed ({method}): {len(frames_window)} frames, {solved.n_points} points, "
        f"{solved.n_obs} observations, cost {info['cost_final']:.2f}"
    )
    return info


# --------------------------------------------------------------------------- #
# Stage 04y: local-BA outlier rejection (ORB-SLAM section VI.D), behind --ba_outlier_reject
# --------------------------------------------------------------------------- #

def _row_sigmas(cams, obs_keys, sigma_fn):
    """(n_obs,) sigma_k = sigma_fn(observing frame)[keypoint index], one call per camera."""
    cache = {}
    sig = np.empty(len(obs_keys), dtype=np.float64)
    for k, (c, _fid, kp, _pid) in enumerate(obs_keys):
        s = cache.get(c)
        if s is None:
            s = np.asarray(sigma_fn(cams[c]), dtype=np.float64).reshape(-1)
            cache[c] = s
        sig[k] = float(s[int(kp)])
    return sig


def classify_ba_outliers(problem, sigmas, chi2=BA_OUTLIER_CHI2):
    """
    (n_obs,) bool: OUTLIER iff the point is behind its camera (BAProblem.behind_mask) or
    (e_k / sigma_k)^2 > chi2, with e_k the Euclidean norm of the row's raw 2-vector
    residual in pixels.
    """
    behind = problem.behind_mask()
    e = np.linalg.norm(np.asarray(problem.residuals(), dtype=np.float64).reshape(-1, 2), axis=1)
    sig = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    return behind | ((e / sig) ** 2 > float(chi2))


def _mean_reprojection_error_rows(problem, rows):
    """BAProblem.mean_reprojection_error restricted to the rows where ``rows`` is True."""
    x_cam, _, _ = problem._camera_coords()
    uv_proj, _ = problem._project(x_cam)
    ok = (x_cam[:, 2] > BEHIND_CAMERA_Z) & np.asarray(rows, dtype=bool)
    if not np.any(ok):
        return 0.0
    e = np.linalg.norm(uv_proj[ok] - problem.uv[ok], axis=1)
    return float(np.mean(e))


def _bundle_adjust_reject_outliers(frames_window, world_map, problem, meta, cams, obs_keys, sigma_fn,
                                   *, method, max_iterations, use_analytical_jacobian,
                                   min_camera_points=0):
    """
    Two-phase solve (spec 04y A3).  Phase 1: n1 = (N + 1) // 2 iterations on every row.
    Mid classification at the phase-1 state.  Phase 2: n2 = N - n1 iterations on the rows
    not marked, from the phase-1 state (a point left with no row keeps its phase-1
    coordinates exactly; skipped when fewer than 10 rows remain or n2 == 0).  Final
    classification of every original row at the final state.  BA weights are unchanged
    (unit weight, Huber in both phases).  Writes back as bundle_adjust; never edits a link.

    Stage 04z (R57-2, R57-3): after the mid classification r2_c = the rows of free camera c not
    marked.  With min_camera_points m > 0 every free camera with r2_c < m is held at its phase-1
    pose for phase 2 (the phase-2 fixed mask is the problem's OR these cameras); phase 2 is skipped
    when no camera is left free.  A phase-2-held camera is still written back (its phase-1 pose).
    """
    N = int(max_iterations)
    n1 = (N + 1) // 2
    n2 = N - n1
    sig = _row_sigmas(cams, obs_keys, sigma_fn)

    # (b) phase 1
    p1, info1 = solve_ba(problem, method=method, max_iterations=n1,
                         use_analytical_jacobian=use_analytical_jacobian)
    # (c) mid classification
    out_mid = classify_ba_outliers(p1, sig)
    n_mid = int(np.count_nonzero(out_mid))

    # (c') stage 04z: inlier rows per free camera after the mid classification (R57-3), and the
    # phase-2 hold of R57-2
    m_cam = int(min_camera_points)
    r2 = {c: int(np.count_nonzero((problem.cam_idx == c) & ~out_mid)) for c in problem.free_cam_indices}
    p2_min_rows = min(r2.values()) if r2 else -1
    p2_under10 = sum(1 for v in r2.values() if v < 10)
    fixed_p2 = problem.fixed_mask.copy()
    n_held_p2 = 0
    if m_cam > 0:
        for c, v in r2.items():
            if v < m_cam:
                fixed_p2[c] = True
                n_held_p2 += 1
    p2_ran = 0

    # (d) phase 2 on the rows not marked
    keep = ~out_mid
    final = p1
    last_info = info1
    iterations = int(info1["iterations"])
    n_rejected = int(info1["n_rejected_steps"])
    if n2 > 0 and int(np.count_nonzero(keep)) >= 10 and not bool(np.all(fixed_p2)):
        rows = np.flatnonzero(keep)
        used_pts = np.unique(problem.pt_idx[rows])
        local_of = np.full(problem.n_points, -1, dtype=np.int64)
        local_of[used_pts] = np.arange(used_pts.size)
        pts1 = p1.points()
        sub_obs = [(int(problem.cam_idx[r]), int(local_of[problem.pt_idx[r]]),
                    float(problem.uv[r, 0]), float(problem.uv[r, 1])) for r in rows.tolist()]
        sub = BAProblem.from_arrays(problem.K, p1.poses(), fixed_p2, pts1[used_pts], sub_obs,
                                    huber_delta_px=problem.huber_delta_px)
        p2, info2 = solve_ba(sub, method=method, max_iterations=n2,
                             use_analytical_jacobian=use_analytical_jacobian)
        p2_ran = 1
        pts_final = pts1.copy()
        pts_final[used_pts] = p2.points()
        final = BAProblem(problem.K, p2.poses(), problem.fixed_mask.copy(), pts_final,
                          problem.cam_idx, problem.pt_idx, problem.uv, problem.huber_delta_px)
        last_info = info2
        iterations += int(info2["iterations"])
        n_rejected += int(info2["n_rejected_steps"])

    # (e) final classification over the ORIGINAL rows
    out_end = classify_ba_outliers(final, sig)
    n_end = int(np.count_nonzero(out_end))
    outlier_obs = [(obs_keys[k][1], int(obs_keys[k][2]), obs_keys[k][3])
                   for k in np.flatnonzero(out_end).tolist()]

    # (f) both reprojection figures over the final inliers (the pipeline's gates compare them)
    inl_end = ~out_end
    reproj_before = _mean_reprojection_error_rows(problem, inl_end)
    reproj_after = _mean_reprojection_error_rows(final, inl_end)

    # (h) write back: free cameras through the setter, selected points' coordinates
    poses = final.poses()
    for c in final.free_cam_indices:
        cams[c].set_camera_pose(poses[c])
    pts = final.points()
    for j, pid in enumerate(meta["point_ids"]):
        pt = world_map.get_point(pid)
        if pt is None:
            continue
        new_xyz = np.array(pts[j], dtype=np.float64, copy=True)
        pt.point = new_xyz
        pt.x = float(new_xyz[0])
        pt.y = float(new_xyz[1])
        pt.z = float(new_xyz[2])

    # (g) info: the original problem's counts, the phases' solver figures
    fixed_obs = problem.fixed_mask[problem.cam_idx]
    info = {
        "status": "completed",
        "method": method,
        "iterations": iterations,
        "cost_initial": float(info1["cost_initial"]),
        "cost_final": float(last_info["cost_final"]),
        "converged": bool(last_info["converged"]),
        "stop_reason": last_info["stop_reason"],
        "n_rejected_steps": n_rejected,
        "n_free": problem.n_free,
        "n_fixed": problem.n_fixed,
        "n_points": problem.n_points,
        "n_obs": problem.n_obs,
        "n_obs_fixed": int(np.count_nonzero(fixed_obs)),
        "n_obs_behind": int(np.count_nonzero(final.behind_mask())),
        "gauge": None,
    }
    info.update({k: v for k, v in meta.items() if k != "status"})
    info["reproj_before_px"] = float(reproj_before)
    info["reproj_after_px"] = float(reproj_after)
    info["n_frames_window"] = len(frames_window)
    info["outlier_obs"] = outlier_obs
    info["n_outliers_mid"] = n_mid
    info["n_outliers_end"] = n_end
    info["p2_min_rows"] = int(p2_min_rows)
    info["p2_under10"] = int(p2_under10)
    info["n_held_p2"] = int(n_held_p2)
    info["p2_ran"] = int(p2_ran)
    print(
        f"  >> BA completed ({method}): {len(frames_window)} frames, {problem.n_points} points, "
        f"{problem.n_obs} observations, cost {info['cost_final']:.2f}"
    )
    return info
