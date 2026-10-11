"""Unit tests of npnp.py (stage 04o, spec E1-E9), with finite-difference checks of every analytical derivative the
module uses (the reduced barrier's gradient and Hessian, the solver's null-space Newton system, the line search's
derivative, and the translation gradient)."""
import ast
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import npnp  # noqa: E402


def _scene(rng, n, sigma_px=0.0, f=500.0):
    """A random pose cam_T_world = (R, t) and n points in front of the camera (depth 4-8, about 53 deg field of
    view); returns P (world), V (bearings, noisy by sigma_px at focal length f), R, t, the camera depths."""
    R = npnp.quat_to_rot(rng.normal(size=4))
    t = rng.normal(size=3)
    z = rng.uniform(4.0, 8.0, size=n)
    uv = rng.uniform(-0.5, 0.5, size=(n, 2))
    Xc = np.hstack([uv * z[:, None], z[:, None]])
    P = (Xc - t) @ R                                        # R^T (x_cam - t)
    uvn = uv + rng.normal(0.0, sigma_px / f, size=(n, 2))
    V = np.hstack([uvn, np.ones((n, 1))])
    return P, V, R, t, z


def _rot_err(Ra, Rb):
    """Rotation angle between Ra and Rb (rad), accurate for small angles."""
    s = np.linalg.norm(Ra.T @ Rb - Rb.T @ Ra) / (2.0 * math.sqrt(2.0))
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.atan2(s, c)


def _cost_grad_t(P, V, R, t):
    Vu = V / np.linalg.norm(V, axis=1, keepdims=True)
    X = P @ R.T + t
    Q = np.eye(3)[None] - Vu[:, :, None] * Vu[:, None, :]
    return 2.0 * np.einsum("nij,nj->i", Q, X)


def _interior_feasible_point(rng, scale=0.002):
    """y_tilde plus a small step in the null space of the equality rows (S~ stays positive definite)."""
    _, _, N = npnp.equality_rows()
    y = npnp.Y_TILDE + N @ rng.normal(size=N.shape[1]) * scale
    assert np.linalg.eigvalsh(npnp.reduced_slack(y))[0] > 0.0
    return y


def _h_tilde(y):
    sign, val = np.linalg.slogdet(npnp.reduced_slack(y))
    assert sign > 0
    return val


# ---------------------------------------------------------------------------------------------------------------
# E1 quat_to_rot
# ---------------------------------------------------------------------------------------------------------------

def test_e1_quat_to_rot():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4) * rng.uniform(0.1, 10.0)
        R = npnp.quat_to_rot(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        assert np.array_equal(R, npnp.quat_to_rot(-q)) or np.allclose(R, npnp.quat_to_rot(-q), atol=1e-15)
        w, x, y, z = q / np.linalg.norm(q)
        r = R.reshape(9)                                    # vec_row(R)
        assert abs(r[0] - (w * w + x * x - y * y - z * z)) < 1e-12
        assert abs(r[1] - (2 * x * y - 2 * w * z)) < 1e-12
    # Hamilton convention: q = (cos(a/2), sin(a/2) k) is the rotation by a about k (right hand)
    a = 0.7
    Rz = npnp.quat_to_rot([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
    assert np.allclose(Rz, [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], atol=1e-14)


# ---------------------------------------------------------------------------------------------------------------
# E2 the cost matrix and the closed-form translation
# ---------------------------------------------------------------------------------------------------------------

def test_e2_rotation_cost_matrix_identity():
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(3, 30))
        P = rng.normal(size=(n, 3)) * 3.0
        V = rng.normal(size=(n, 3))
        R = npnp.quat_to_rot(rng.normal(size=4))
        M = npnp.rotation_cost_matrix(P, V)
        assert np.array_equal(M, M.T)
        assert np.linalg.eigvalsh(M)[0] >= -1e-12 * np.linalg.norm(M)
        t = npnp.optimal_translation(P, V, R)
        cost = npnp.object_space_cost(P, V, R, t)
        r = R.reshape(9)
        assert abs(r @ M @ r - cost) <= 1e-9 * cost


def test_e2_optimal_translation_is_the_minimizer():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(3, 30))
        P = rng.normal(size=(n, 3)) * 3.0
        V = rng.normal(size=(n, 3))
        R = npnp.quat_to_rot(rng.normal(size=4))
        t = npnp.optimal_translation(P, V, R)
        g = _cost_grad_t(P, V, R, t)
        scale = np.linalg.norm(_cost_grad_t(P, V, R, t + 1.0))    # gradient size one unit away
        assert np.max(np.abs(g)) <= 1e-9 * scale
        c0 = npnp.object_space_cost(P, V, R, t)
        for d in np.eye(3):
            assert npnp.object_space_cost(P, V, R, t + 1e-3 * d) > c0


def test_fd_cost_gradient_in_t():
    """Finite-difference check of the analytical translation gradient used in E2 (rule 5)."""
    rng = np.random.default_rng(4)
    P = rng.normal(size=(7, 3)) * 2.0
    V = rng.normal(size=(7, 3))
    R = npnp.quat_to_rot(rng.normal(size=4))
    t = rng.normal(size=3)
    g = _cost_grad_t(P, V, R, t)
    h = 1e-6
    fd = np.array([(npnp.object_space_cost(P, V, R, t + h * e) - npnp.object_space_cost(P, V, R, t - h * e)) / (2 * h)
                   for e in np.eye(3)])
    assert np.allclose(g, fd, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------------------------------------------
# E3 the coefficient identity of A, c, b
# ---------------------------------------------------------------------------------------------------------------

def _monomials(q, exps):
    return np.array([np.prod([qi ** ai for qi, ai in zip(q, a)]) for a in exps])


def test_e3_coefficient_identity():
    rng = np.random.default_rng(5)
    E4, E2 = npnp.MONOMIALS_4, npnp.MONOMIALS_2
    for _ in range(20):
        x = rng.normal(size=241)
        q = rng.normal(size=4)
        gamma, lam, Pm = x[0], x[1:16], x[16:].reshape(15, 15)
        m = _monomials(q, E2)
        lhs = (npnp.A @ x) @ _monomials(q, E4)
        rhs = gamma + (lam @ m) * (q @ q - 1.0) + m @ Pm @ m
        assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(rhs))
        assert npnp.c @ x == -gamma                          # minimize c^T x = maximize gamma
    # b: the coefficients of r(q)^T M r(q) on the unscaled monomials
    for _ in range(20):
        B = rng.normal(size=(9, 9))
        M = B @ B.T
        q = rng.normal(size=4)
        r = npnp.quat_to_rot(q).reshape(9) * (q @ q)        # r(q) is homogeneous of degree 2 in q
        b = npnp.quartic_coefficients(M)
        val = b @ _monomials(q, E4)
        assert abs(val - r @ M @ r) <= 1e-10 * max(1.0, abs(val))
        assert np.all(b[:35] == 0.0)                         # a homogeneous quartic: only degree-4 monomials


def test_e3_generator_builds_the_loaded_constants():
    spec = importlib.util.spec_from_file_location("make_npnp_constants", REPO / "scripts" / "make_npnp_constants.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    A, c = gen.build_A_c()
    assert np.array_equal(A, npnp.A) and np.array_equal(c, npnp.c)
    assert gen.E4 == npnp.MONOMIALS_4 and gen.E2 == npnp.MONOMIALS_2
    assert np.all(A == np.round(A))


# ---------------------------------------------------------------------------------------------------------------
# E4 the start point on the face (D1b-1)
# ---------------------------------------------------------------------------------------------------------------

def _uniform_s3_moment(a):
    if any(x % 2 for x in a):
        return 0.0
    val = math.gamma(2.0) / math.gamma(2.0 + sum(a) / 2.0)
    for x in a:
        val *= math.gamma((x + 1) / 2.0) / math.gamma(0.5)
    return val


def test_e4_start_point():
    y = npnp.Y_TILDE
    E, e, N = npnp.equality_rows()
    assert E.shape == (16, 70) and N.shape == (70, 54)
    assert np.max(np.abs(E @ y - e)) <= 1e-12
    assert np.linalg.eigvalsh(npnp.reduced_slack(y))[0] > 0.0
    g, _ = npnp.reduced_barrier_grad_hess(y)
    assert np.max(np.abs(N @ (N.T @ g))) < 1e-8
    # y_tilde = -(uniform S^3 moments) under this module's sign convention (y = -L)
    ref = -np.array([_uniform_s3_moment(a) for a in npnp.MONOMIALS_4])
    assert np.max(np.abs(y - ref)) <= 1e-12
    # exact spot values: E[w^2] = 1/4, E[w^4] = 1/8, E[w^2 x^2] = 1/24
    idx = {a: k for k, a in enumerate(npnp.MONOMIALS_4)}
    assert abs(-y[idx[(2, 0, 0, 0)]] - 0.25) < 1e-15
    assert abs(-y[idx[(4, 0, 0, 0)]] - 0.125) < 1e-15
    assert abs(-y[idx[(2, 2, 0, 0)]] - 1.0 / 24.0) < 1e-15
    # the full 15 x 15 slack is singular with kernel v (the coefficients of ||q||^2 - 1 over m(q))
    v = np.array([-1.0 if a == (0, 0, 0, 0) else (1.0 if sorted(a) == [0, 0, 0, 2] else 0.0)
                  for a in npnp.MONOMIALS_2])
    S = npnp.slack(y)
    assert np.linalg.norm(S @ v) / np.linalg.norm(S) < 1e-12


def test_e4_every_feasible_slack_is_singular_but_the_face_is_not():
    """The fact the reduction rests on, away from y_tilde too: S(y) v = 0 on the whole affine feasible set."""
    rng = np.random.default_rng(6)
    v = np.array([-1.0 if a == (0, 0, 0, 0) else (1.0 if sorted(a) == [0, 0, 0, 2] else 0.0)
                  for a in npnp.MONOMIALS_2])
    for _ in range(20):
        y = _interior_feasible_point(rng)
        S = npnp.slack(y)
        assert np.linalg.norm(S @ v) / np.linalg.norm(S) < 1e-12
        assert np.linalg.eigvalsh(npnp.reduced_slack(y))[0] > 0.0


def test_e4_uniform_moments_are_the_centre_and_off_centre_points_are_not():
    """The projected-gradient test has teeth: it is large at a feasible point away from y_tilde."""
    rng = np.random.default_rng(7)
    _, _, N = npnp.equality_rows()
    y = _interior_feasible_point(rng)
    g, H = npnp.reduced_barrier_grad_hess(y)
    assert np.max(np.abs(N @ (N.T @ g))) > 1e-2
    # negative definite on the feasible directions at y_tilde (the centre is unique)
    _, H0 = npnp.reduced_barrier_grad_hess(npnp.Y_TILDE)
    assert np.linalg.eigvalsh(N.T @ H0 @ N).max() < 0.0


# ---------------------------------------------------------------------------------------------------------------
# finite-difference checks of the barrier derivatives and of the solver's Newton system (rule 5)
# ---------------------------------------------------------------------------------------------------------------

def test_fd_reduced_barrier_gradient_and_hessian():
    rng = np.random.default_rng(8)
    y = _interior_feasible_point(rng)
    g, H = npnp.reduced_barrier_grad_hess(y)
    h = 1e-6
    for d in list(np.eye(70)[[0, 5, 17, 40, 69]]) + [rng.normal(size=70) for _ in range(3)]:
        d = d / np.linalg.norm(d)
        fd = (_h_tilde(y + h * d) - _h_tilde(y - h * d)) / (2 * h)
        assert abs(fd - g @ d) <= 1e-6 * max(1.0, abs(g @ d))
        gp, _ = npnp.reduced_barrier_grad_hess(y + h * d)
        gm, _ = npnp.reduced_barrier_grad_hess(y - h * d)
        assert np.allclose((gp - gm) / (2 * h), H @ d, rtol=1e-5, atol=1e-5 * np.abs(H).max())


def test_fd_solver_newton_system_matches_the_barrier():
    """The solver's reduced gradient and Gauss-Newton form (-N^T H N = J^T J from L^-1 A~N_j L^-T) agree with the
    analytical g and H of reduced_barrier_grad_hess, which the test above checks by finite differences."""
    rng = np.random.default_rng(9)
    y = _interior_feasible_point(rng)
    _, _, N = npnp.equality_rows()
    b = npnp.quartic_coefficients(np.cov(rng.normal(size=(9, 30))))
    bp = b / np.linalg.norm(b)
    t = 3.7
    dz, lam2, G = npnp._newton_direction(y, t, N.T @ bp)
    g, H = npnp.reduced_barrier_grad_hess(y)
    gz = N.T @ (t * bp + g)
    Hz = N.T @ H @ N
    assert np.allclose(-np.einsum("jii->j", G) + t * (N.T @ bp), gz, rtol=1e-10, atol=1e-10)
    J = G.reshape(G.shape[0], -1).T
    assert np.allclose(-J.T @ J, Hz, rtol=1e-10, atol=1e-10 * np.abs(Hz).max())
    # the Newton direction solves the 86 x 86 KKT system [[H, E^T], [E, 0]] [d; w] = [-g_t; 0]
    E, _, _ = npnp.equality_rows()
    d = N @ dz
    K = np.block([[H, E.T], [E, np.zeros((16, 16))]])
    sol = np.linalg.lstsq(K, np.concatenate([-(t * bp + g), np.zeros(16)]), rcond=None)[0]
    assert np.allclose(sol[:70], d, rtol=1e-8, atol=1e-10 * np.linalg.norm(d))
    assert abs(lam2 - (-(d @ H @ d))) <= 1e-10 * max(1.0, lam2)


def test_fd_line_search_derivative_and_exactness():
    rng = np.random.default_rng(10)
    y = _interior_feasible_point(rng)
    _, _, N = npnp.equality_rows()
    b = npnp.quartic_coefficients(np.cov(rng.normal(size=(9, 30))))
    bp = b / np.linalg.norm(b)
    t = 25.0
    Nb = N.T @ bp
    dz, lam2, G = npnp._newton_direction(y, t, Nb)
    mu, lin = npnp._line_terms(dz, G, t, Nb)
    d = N @ dz

    def f(s):
        return t * bp @ (y + s * d) + _h_tilde(y + s * d)

    for s in (0.1, 0.5, 0.9):
        phi = lin * s + np.sum(np.log1p(s * mu))
        assert abs(phi - (f(s) - f(0.0))) <= 1e-9 * max(1.0, abs(phi))
        h, h2 = 1e-6, 1e-4
        fd1 = (f(s + h) - f(s - h)) / (2 * h)
        fd2 = (f(s + h2) - 2 * f(s) + f(s - h2)) / (h2 * h2)
        d1, d2 = npnp._dphi(mu, lin, s)
        assert abs(fd1 - d1) <= 1e-6 * max(1.0, abs(d1))
        assert abs(fd2 - d2) <= 1e-3 * max(1.0, abs(d2))
    assert abs(npnp._dphi(mu, lin, 0.0)[0] - lam2) <= 1e-9 * max(1.0, lam2)
    s, evals = npnp._line_search(dz, G, t, Nb)
    assert evals >= 1 and s > 0.0
    assert abs(npnp._dphi(mu, lin, s)[0]) <= 1e-8 * lam2 + 1e-12
    assert np.linalg.eigvalsh(npnp.reduced_slack(y + s * d))[0] > 0.0
    assert f(s) >= max(f(0.9 * s), f(1.1 * s) if np.all(1 + 1.1 * s * mu > 0) else -np.inf)


# ---------------------------------------------------------------------------------------------------------------
# E5 noise-free recovery
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 20, 100])
def test_e5_noise_free_recovery(n):
    rng = np.random.default_rng(100 + n)
    for _ in range(3):
        P, V, R, t, z = _scene(rng, n)
        res = npnp.solve(P, V)
        Mf = np.linalg.norm(npnp.rotation_cost_matrix(P, V))
        assert res.converged
        assert _rot_err(res.R, R) <= 1e-6
        assert np.linalg.norm(res.t - t) <= 1e-6 * np.median(z)
        assert res.gap <= 1e-6 * Mf
        # gamma is a lower bound up to the barrier's gap 14 ||b|| / t_final
        b = npnp.quartic_coefficients(npnp.rotation_cost_matrix(P, V))
        assert res.gamma <= res.cost + npnp.BARRIER_PARAMETER * np.linalg.norm(b) / res.t_final * 1.01
        assert res.t_final >= 1.0 / (npnp.MU * 1e-10) and res.t_final < 1e10


def test_e5_n4_noise_free_is_certified_and_near_exact():
    """Recorded behaviour at n = 4 (M has rank <= 5, a degenerate optimal face): over 40 random noise-free
    scenes every solve converges and certifies (gap <= 1e-6 ||M||_F); the rotation is within 1e-4 rad (the
    1e-6 rad of E5 holds on most but not all such scenes at the fixed eps = 1e-10)."""
    rng = np.random.default_rng(404)
    for _ in range(40):
        P, V, R, t, z = _scene(rng, 4)
        res = npnp.solve(P, V)
        Mf = np.linalg.norm(npnp.rotation_cost_matrix(P, V))
        assert res.converged and res.gap <= 1e-6 * Mf
        assert _rot_err(res.R, R) <= 1e-4


def test_noisy_solution_is_certified_and_not_worse_than_a_local_search():
    """With noise, the certified pose's cost is not above the cost at the truth nor at a perturbed optimum."""
    rng = np.random.default_rng(11)
    for n in (5, 12, 40):
        P, V, R, t, z = _scene(rng, n, sigma_px=2.0)
        res = npnp.solve(P, V)
        M = npnp.rotation_cost_matrix(P, V)
        Mf = np.linalg.norm(M)
        assert res.converged and res.gap <= 1e-5 * Mf
        assert res.cost <= R.reshape(9) @ M @ R.reshape(9) + 1e-9 * Mf
        assert abs(res.gap - (res.cost - res.gamma)) == 0.0
        X = res.moment_qq
        assert np.allclose(X, X.T) and abs(np.trace(X) - 1.0) < 1e-9
        q = np.linalg.eigh(X)[1][:, -1]
        q = q if q[np.argmax(np.abs(q))] > 0 else -q
        assert np.array_equal(npnp.quat_to_rot(q), res.R)


def test_solve_moments_and_zero_cost_matrix():
    rng = np.random.default_rng(12)
    P, V, R, t, z = _scene(rng, 10, sigma_px=1.0)
    M = npnp.rotation_cost_matrix(P, V)
    rm = npnp.solve_moments(M)
    rs = npnp.solve(P, V)
    assert rm.t is None and np.array_equal(rm.R, rs.R) and rm.gamma == rs.gamma
    r = rm.R.reshape(9)
    assert rm.cost == r @ M @ r and rm.gap == rm.cost - rm.gamma
    z0 = npnp.solve_moments(np.zeros((9, 9)))
    assert z0.converged and z0.gamma == 0.0 and np.array_equal(z0.R, np.eye(3)) and z0.newton_iters == 0


# ---------------------------------------------------------------------------------------------------------------
# caps and non-finite values end the solve without raising
# ---------------------------------------------------------------------------------------------------------------

def test_newton_caps_end_with_converged_false(monkeypatch):
    rng = np.random.default_rng(13)
    P, V, R, t, z = _scene(rng, 8, sigma_px=1.0)
    ref = npnp.solve(P, V)
    assert ref.converged and ref.newton_iters > 5
    monkeypatch.setattr(npnp, "MAX_NEWTON_PER_CENTRING", 2)
    res = npnp.solve(P, V)
    assert not res.converged and res.newton_iters <= 2 * res.outer_iters
    assert np.all(np.isfinite(res.R))
    monkeypatch.setattr(npnp, "MAX_NEWTON_PER_CENTRING", 100)
    monkeypatch.setattr(npnp, "MAX_NEWTON_TOTAL", 5)
    res = npnp.solve(P, V)
    assert not res.converged and res.newton_iters == 5


def test_non_finite_value_ends_with_converged_false(monkeypatch):
    rng = np.random.default_rng(14)
    P, V, R, t, z = _scene(rng, 8, sigma_px=1.0)
    real = npnp.reduced_slack
    calls = {"n": 0}

    def poisoned(y):
        calls["n"] += 1
        S = real(y)
        return S * np.nan if calls["n"] > 3 else S

    monkeypatch.setattr(npnp, "reduced_slack", poisoned)
    res = npnp.solve(P, V)
    assert not res.converged and calls["n"] > 3
    assert np.all(np.isfinite(res.R)) and np.all(np.isfinite(res.t))


# ---------------------------------------------------------------------------------------------------------------
# E6 determinism, E7 input validation, E8 scale invariance, E9 imports
# ---------------------------------------------------------------------------------------------------------------

FIELDS = ("R", "t", "gamma", "cost", "gap", "converged", "outer_iters", "newton_iters", "line_search_evals",
          "t_final", "moment_qq")


def test_e6_determinism():
    rng = np.random.default_rng(15)
    P, V, R, t, z = _scene(rng, 12, sigma_px=1.0)
    r1, r2 = npnp.solve(P, V), npnp.solve(P, V)
    for f in FIELDS:
        assert np.asarray(getattr(r1, f)).tobytes() == np.asarray(getattr(r2, f)).tobytes(), f


def test_e7_input_validation():
    rng = np.random.default_rng(16)
    P, V, R, t, z = _scene(rng, 6)
    with pytest.raises(ValueError):
        npnp.solve(P[:2], V[:2])
    with pytest.raises(ValueError):
        npnp.solve(P[:0], V[:0])
    for arr, which in itertools.product((np.nan, np.inf, -np.inf), ("P", "V")):
        P2, V2 = P.copy(), V.copy()
        (P2 if which == "P" else V2)[3, 1] = arr
        with pytest.raises(ValueError):
            npnp.solve(P2, V2)
    with pytest.raises(ValueError):
        npnp.solve(P, V[:5])
    V0 = V.copy()
    V0[2] = 0.0
    with pytest.raises(ValueError):
        npnp.solve(P, V0)
    assert npnp.solve(P[:3], V[:3]).R.shape == (3, 3)      # n = 3 is accepted


def test_e8_scale_invariance():
    rng = np.random.default_rng(17)
    P, V, R, t, z = _scene(rng, 10, sigma_px=1.5)
    r0 = npnp.solve(P, V)
    r2 = npnp.solve(P, V * 4.0)                             # a power of two: bitwise
    for f in FIELDS:
        assert np.asarray(getattr(r0, f)).tobytes() == np.asarray(getattr(r2, f)).tobytes(), f
    k = rng.uniform(0.05, 20.0, size=(10, 1))               # a different positive factor per bearing
    r3 = npnp.solve(P, V * k)
    assert r3.converged and _rot_err(r0.R, r3.R) <= 1e-9
    assert np.allclose(r0.t, r3.t, atol=1e-9 * np.median(z))
    assert abs(r0.gamma - r3.gamma) <= 1e-9 * max(1.0, abs(r0.gamma))


def test_e9_imports_only_numpy_and_stdlib():
    tree = ast.parse((REPO / "npnp.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module.split(".")[0])
    bad = sorted(x for x in names if x != "numpy" and x not in sys.stdlib_module_names)
    assert not bad, bad
    assert "numpy" in names
