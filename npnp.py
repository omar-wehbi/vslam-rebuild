"""Newton-PnP (NPnP): a pure-NumPy reimplementation of Algorithm 1 of Jubran et al. 2022 (stage 04o, part A of the
Newton-PnP solver study, theory/design.md#newton-pnp-solver-study-d1, D1a and D1b). A study module: nothing in the
SLAM product imports it.

THE PROBLEM (paper section II.A). World points P (n x 3) and bearing vectors V (n x 3, camera frame, any nonzero
length; normalized here). The pose (R, t) is cam_T_world, x_cam = R p + t (conventions.md#pose-convention). The cost
is the object-space cost sum_i ||(I - v_i v_i^T)(R p_i + t)||^2 (paper eq. 1), the squared distance of each
transformed point to its bearing LINE. For fixed R the optimal t is closed form, and the cost becomes
r^T M r with r = vec_row(R) (paper eq. 2); with R the Hamilton rotation of a unit quaternion q = (w, x, y, z), the
problem is min over ||q|| = 1 of the quartic p(q) = r(q)^T M r(q).

MONOMIAL ORDER (shared with scripts/make_npnp_constants.py and checked at import). Monomials q^a of q = (w, x, y, z),
a = (a_w, a_x, a_y, a_z), sorted by total degree and then lexicographically descending: m(q) = (1, w, x, y, z, w^2,
wx, wy, wz, x^2, xy, xz, y^2, yz, z^2) (15, degree <= 2), and the 70 monomials of degree <= 4 begin with m(q).

THE RELAXATION (paper eq. 4-6). Primal x = (gamma, lambda_1..15, vec_row(Pm)) in R^241 with
    r(q)^T M r(q) = gamma + lambda(q) (||q||^2 - 1) + m(q)^T Pm m(q),   Pm PSD,
read coefficient by coefficient over the 70 monomials: A x = b, b = the coefficients of p(q) (unscaled monomials,
b_a = coefficient of q^a), c = (-1, 0, ..., 0), minimize c^T x. Dual: maximize b^T y subject to (c - A^T y)_{0:16} = 0
and S(y) = mat((c - A^T y)_{16:241}) PSD. SIGN CONVENTION: y = -L, with L the pseudo-moment functional (L(q^a) =
-y_a): the gamma row reads y_0 = -1, i.e. L(1) = 1; the 15 lambda rows read L(m_j(q)(||q||^2 - 1)) = 0; and S(y) =
-sum_k y_k A_k = [L(m_i m_j)] is the 15 x 15 moment matrix (A_k = mat(A[k, 16:]), the 0/1 matrix of column k of the
slack map; c contributes nothing to S). So dS/dy_k = -A_k, and the relaxation's value at y is L(p) = -b^T y.

THE FACE (D1b-1, ruled 2026-09-24; theory/studies/newton-pnp-dual-interior.md sections 2 and 6). With v the
coefficient vector of ||q||^2 - 1 over m(q) (-1 on the constant, +1 on w^2, x^2, y^2, z^2), the 15 lambda
equalities are exactly S(y) v = 0. So EVERY dual feasible S(y) is singular, log det S(y) = -inf on the whole feasible
set, and the paper's barrier (eq. 7) and analytic centre (eq. 8) do not exist as written. This module therefore puts
the barrier on the 14-dimensional face: W (15 x 14) is the identity with the constant-monomial column deleted,
S~(y) = W^T S(y) W (S with its constant row and column deleted), A~_k = W^T A_k W, and
    h~(y) = log det S~(y).
The feasible set is unchanged (every feasible S = U Z U^T with U a basis of v's orthogonal complement), any complement
of v gives barriers differing by a constant (identical Newton iterates), and the barrier parameter is 14, so the
duality gap on the central path is 14 / t (in b' units). The full 15 x 15 S is never inverted or factored. The
analytic centre of h~ over the dual feasible set is y_tilde = -(the moments of the uniform distribution on S^3),
by symmetry and uniqueness; it is loaded from npnp_constants.npz (closed form, checked by the generator).

ALGORITHM 1 (the paper's, with the barrier on the face; the only departure). b' = b / ||b||; y = y_tilde;
t = t0 = 1 / (140 Delta^2), Delta = 1 (D1b-1's reading); while t < 1/eps: centre f_t(y) = t b'^T y + h~(y) under the
16 equalities by Newton steps with equality constraints and an exact line search, until the Newton decrement squared
falls below 2 eps; then t = 50 t. The Newton direction solves the 86 x 86 KKT system [[H, E^T], [E, 0]] [d; w] =
[-g; 0] (E = the 16 equality rows). It is solved by the null-space method, which gives the same d: d = N dz with N an
orthonormal basis of null(E) (70 x 54, constant) and (N^T (-H) N) dz = N^T g. Since -H = J^T J with the columns of J
the vectors vec(L^-1 A~_k L^-T) (S~ = L L^T, Cholesky), the reduced system is solved through a QR factorization of
J N, whose condition number is the square root of the reduced Hessian's; this is the numerically stable form (the
Hessian's conditioning grows like t^2). The exact line search maximizes the concave f_t(y + s d) over s inside the
domain where S~ stays positive definite; with mu_i the eigenvalues of L^-1 dS~ L^-T (the generalized eigenvalues of
the pencil (dS~, S~)), f_t(y + s d) - f_t(y) = s t b'^T d + sum_i log(1 + s mu_i), and its derivative is solved for
zero to a relative tolerance of 1e-12 in s (safeguarded Newton inside a bracket). Caps: 100 Newton steps per
centring, 2000 in total; a cap or a non-finite value ends the solve with converged = False (never raises).
Extraction: X = the 4 x 4 degree-2 moment block L(q q^T) = S(y)[1:5, 1:5]; q_hat = its leading eigenvector
(numpy.linalg.eigh, sign so its largest-magnitude component is positive); R = quat_to_rot(q_hat); t by the closed form.

Deterministic: no randomness, threads or clocks. Imports numpy and the standard library only.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

EPS_DEFAULT = 1e-10
DELTA = 1.0                       # the paper's precision constant, read as 1 for normalized b' (D1b-1)
T0 = 1.0 / (140.0 * DELTA ** 2)   # initial barrier weight (Algorithm 1)
MU = 50.0                         # barrier weight multiplier (Algorithm 1)
MAX_NEWTON_PER_CENTRING = 100
MAX_NEWTON_TOTAL = 2000
LINE_SEARCH_RTOL = 1e-12
BARRIER_PARAMETER = 14            # dimension of the face (D1b-1): the duality gap on the central path is 14 / t


# ---------------------------------------------------------------------------------------------------------------
# monomials and constants
# ---------------------------------------------------------------------------------------------------------------

def _exponents(d):
    """Exponent tuples of the monomials of q in R^4 of degree <= d: by degree, then lexicographically descending."""
    out = [a for a in itertools.product(range(d + 1), repeat=4) if sum(a) <= d]
    return sorted(out, key=lambda a: (sum(a), tuple(-x for x in a)))


MONOMIALS_2 = _exponents(2)      # m(q), 15
MONOMIALS_4 = _exponents(4)      # the 70 rows / dual coordinates
_IDX4 = {a: k for k, a in enumerate(MONOMIALS_4)}

_CONST_PATH = Path(__file__).resolve().parent / "npnp_constants.npz"
with np.load(_CONST_PATH, allow_pickle=False) as _z:
    A = np.array(_z["A"], dtype=np.float64)
    c = np.array(_z["c"], dtype=np.float64)
    Y_TILDE = np.array(_z["y_tilde"], dtype=np.float64)


def _check_constants():
    """Import-time guard: the loaded A, c, y_tilde follow this module's monomial order and sign convention."""
    if A.shape != (70, 241) or c.shape != (241,) or Y_TILDE.shape != (70,):
        raise RuntimeError("npnp_constants.npz has unexpected shapes")
    if len(MONOMIALS_2) != 15 or MONOMIALS_4[:15] != MONOMIALS_2:
        raise RuntimeError("monomial order broken")
    for i, a in enumerate(MONOMIALS_2):
        for j, bb in enumerate(MONOMIALS_2):
            col = np.zeros(70)
            col[_IDX4[tuple(x + y for x, y in zip(a, bb))]] = 1.0
            if not np.array_equal(A[:, 16 + 15 * i + j], col):
                raise RuntimeError("npnp_constants.npz does not match npnp.py's monomial order (Gram columns)")
    if c[0] != -1.0 or np.any(c[1:] != 0.0) or Y_TILDE[0] != -1.0:
        raise RuntimeError("npnp_constants.npz does not match npnp.py's sign convention")


_check_constants()

# Equality rows (c - A^T y)_{0:16} = 0  <=>  E y = e.
_E = A[:, :16].T.copy()
_e = c[:16].copy()
# Slack map: S(y) = C_S - sum_k y_k A_k (15 x 15); reduced to the face by deleting the constant row and column.
_AK = A[:, 16:].reshape(70, 15, 15)
_CS = c[16:].reshape(15, 15)
_AKR = _AK[:, 1:, 1:].copy()
_CSR = _CS[1:, 1:].copy()
# Orthonormal basis of the null space of the equality rows (70 x 54) and the reduced slack derivatives along it:
# dS~/dz_j = -_ANR[j].
_N = np.linalg.svd(_E)[2][16:].T.copy()
_ANR = np.einsum("kj,kab->jab", _N, _AKR)

# r(q) = vec_row(R(q)) as quadratic forms: _RQ[k] = [(coefficient, exponent), ...] (spec B1, the paper's r(q), q1 = w)
_W2, _X2, _Y2, _Z2 = (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)
_WX, _WY, _WZ = (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)
_XY, _XZ, _YZ = (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1)
_RQ = [
    [(1, _W2), (1, _X2), (-1, _Y2), (-1, _Z2)], [(2, _XY), (-2, _WZ)], [(2, _XZ), (2, _WY)],
    [(2, _XY), (2, _WZ)], [(1, _W2), (-1, _X2), (1, _Y2), (-1, _Z2)], [(2, _YZ), (-2, _WX)],
    [(2, _XZ), (-2, _WY)], [(2, _YZ), (2, _WX)], [(1, _W2), (-1, _X2), (-1, _Y2), (1, _Z2)],
]
# _TQ[k, l, a] = coefficient of q^a in r_k(q) r_l(q), so that b = sum_kl M_kl _TQ[k, l].
_TQ = np.zeros((9, 9, 70))
for _k in range(9):
    for _l in range(9):
        for _ca, _ea in _RQ[_k]:
            for _cb, _eb in _RQ[_l]:
                _TQ[_k, _l, _IDX4[tuple(x + y for x, y in zip(_ea, _eb))]] += _ca * _cb


# ---------------------------------------------------------------------------------------------------------------
# the problem (spec B)
# ---------------------------------------------------------------------------------------------------------------

def quat_to_rot(q):
    """Hamilton rotation matrix of q = (w, x, y, z) (scalar first; normalized here)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    nq = float(np.linalg.norm(q))
    if not (np.isfinite(nq) and nq > 0.0):
        raise ValueError("quat_to_rot needs a finite nonzero quaternion")
    w, x, y, z = q / nq
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def _validate(P, V):
    P = np.asarray(P, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or V.shape != P.shape:
        raise ValueError(f"P and V must both be n x 3 (got {P.shape} and {V.shape})")
    if P.shape[0] < 3:
        raise ValueError(f"need n >= 3 correspondences (got {P.shape[0]})")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(V))):
        raise ValueError("P and V must be finite")
    nv = np.linalg.norm(V, axis=1)
    if np.any(nv == 0.0):
        raise ValueError("bearing vectors must be nonzero")
    return P, V / nv[:, None]


def _projectors(Vu):
    return np.eye(3)[None, :, :] - Vu[:, :, None] * Vu[:, None, :]


def _solve_sum_q(Q, rhs):
    try:
        return np.linalg.solve(Q.sum(axis=0), rhs)
    except np.linalg.LinAlgError as err:
        raise ValueError("degenerate bearings: sum_i (I - v_i v_i^T) is singular") from err


def object_space_cost(P, V, R, t):
    """sum_i ||(I - v_i v_i^T)(R p_i + t)||^2 with v_i the unit bearings (paper eq. 1)."""
    P, Vu = _validate(P, V)
    X = P @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).reshape(1, 3)
    D = X - Vu * np.sum(Vu * X, axis=1, keepdims=True)
    return float(np.sum(D * D))


def optimal_translation(P, V, R):
    """The minimizer over t of object_space_cost for fixed R: -(sum_i Q_i)^-1 sum_i Q_i R p_i."""
    P, Vu = _validate(P, V)
    Q = _projectors(Vu)
    X = P @ np.asarray(R, dtype=np.float64).T
    return -_solve_sum_q(Q, np.einsum("nij,nj->i", Q, X))


def rotation_cost_matrix(P, V):
    """M (9 x 9, symmetric PSD) with object_space_cost(P, V, R, optimal_translation(P, V, R)) = r^T M r,
    r = vec_row(R) (paper eq. 2): with R p_i = K_i r, K_i = kron(I3, p_i^T), and t = T r,
    T = -(sum Q_i)^-1 sum Q_i K_i, M = sum_i (K_i + T)^T Q_i (K_i + T)."""
    P, Vu = _validate(P, V)
    n = P.shape[0]
    Q = _projectors(Vu)
    K = np.zeros((n, 3, 9))
    for k in range(3):
        K[:, k, 3 * k:3 * k + 3] = P
    T = -_solve_sum_q(Q, np.einsum("nij,njk->ik", Q, K))
    D = K + T[None, :, :]
    M = np.einsum("nji,njk,nkl->il", D, Q, D)
    return 0.5 * (M + M.T)


def quartic_coefficients(M):
    """b in R^70: b_a = the coefficient of q^a in r(q)^T M r(q) (unscaled monomials, this module's order)."""
    M = np.asarray(M, dtype=np.float64)
    return np.einsum("kl,kla->a", M, _TQ)


# ---------------------------------------------------------------------------------------------------------------
# the solver (spec C2, C3)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NPnPResult:
    """R (3x3), t (3,) or None (solve_moments), gamma (the relaxation's value L(p) at the final iterate, original
    units), cost (object-space cost of (R, t); r^T M r for solve_moments), gap = cost - gamma, converged, iteration
    counters, t_final (the barrier weight of the last completed centring, b' units; on a failed solve, the weight
    of the interrupted one), moment_qq (the 4 x 4 block L(q q^T) at the final iterate)."""
    R: np.ndarray
    t: Optional[np.ndarray]
    gamma: float
    cost: float
    gap: float
    converged: bool
    outer_iters: int
    newton_iters: int
    line_search_evals: int
    t_final: float
    moment_qq: np.ndarray


def slack(y):
    """S(y) = mat((c - A^T y)_{16:241}), the 15 x 15 moment matrix (singular at every feasible y)."""
    return _CS - np.einsum("k,kab->ab", np.asarray(y, dtype=np.float64), _AK)


def reduced_slack(y):
    """S~(y) = W^T S(y) W, the 14 x 14 slack on the face (constant row and column deleted)."""
    return _CSR - np.einsum("k,kab->ab", np.asarray(y, dtype=np.float64), _AKR)


def reduced_barrier_grad_hess(y):
    """Gradient and Hessian (70 and 70 x 70) of h~(y) = log det S~(y): g_k = -tr(S~^-1 A~_k),
    H_kl = -tr(S~^-1 A~_k S~^-1 A~_l). For tests and the generator's check; the solver works in the null space."""
    Si = np.linalg.inv(reduced_slack(y))
    B = np.einsum("ij,kjl->kil", Si, _AKR)
    g = -np.einsum("kii->k", B)
    H = -np.einsum("kij,lji->kl", B, B)
    return g, H


def equality_rows():
    """(E, e): the 16 equality rows (c - A^T y)_{0:16} = 0 written as E y = e, and the null-space basis N of E."""
    return _E.copy(), _e.copy(), _N.copy()


class _NonFinite(Exception):
    pass


def _newton_direction(y, t, Nb):
    """Reduced Newton direction dz (y moves by N dz) and the decrement squared for f_t = t b'^T y + h~(y)."""
    L = np.linalg.cholesky(reduced_slack(y))
    Li = np.linalg.inv(L)
    G = Li[None, :, :] @ _ANR @ Li.T[None, :, :]          # (54, 14, 14): L^-1 A~N_j L^-T
    gz = t * Nb - np.einsum("jii->j", G)                   # N^T g
    J = G.reshape(G.shape[0], -1).T                        # 196 x 54, -N^T H N = J^T J
    Rf = np.linalg.qr(J, mode="r")
    w = np.linalg.solve(Rf.T, gz)
    dz = np.linalg.solve(Rf, w)
    lam2 = float(w @ w)
    if not (np.all(np.isfinite(dz)) and np.isfinite(lam2)):
        raise _NonFinite
    return dz, lam2, G


def _line_terms(dz, G, t, Nb):
    """phi(s) = f_t(y + s N dz) - f_t(y) = lin s + sum_i log(1 + s mu_i): mu = the eigenvalues of L^-1 dS~ L^-T
    (the generalized eigenvalues of the pencil (dS~, S~), dS~ = -sum_j dz_j A~N_j), lin = t b'^T N dz."""
    Dm = -np.einsum("j,jab->ab", dz, G)
    mu = np.linalg.eigvalsh(0.5 * (Dm + Dm.T))
    return mu, float(t * (Nb @ dz))


def _dphi(mu, lin, s):
    """phi'(s) and phi''(s) of _line_terms' phi."""
    r = mu / (1.0 + s * mu)
    return lin + float(np.sum(r)), -float(np.sum(r * r))


def _line_search(dz, G, t, Nb):
    """Exact line search: the zero of phi'(s) inside the domain 1 + s mu_i > 0; returns (s, evaluations)."""
    mu, lin = _line_terms(dz, G, t, Nb)
    if not (np.all(np.isfinite(mu)) and np.isfinite(lin)):
        raise _NonFinite
    neg = mu[mu < 0.0]
    s_max = float(1.0 / np.max(-neg)) if neg.size else math.inf
    evals = 0

    def dphi(s):
        nonlocal evals
        evals += 1
        return _dphi(mu, lin, s)

    lo, hi = 0.0, s_max
    if not math.isfinite(hi):                               # bracket the root when the domain is unbounded
        hi = 1.0
        while dphi(hi)[0] > 0.0:
            lo, hi = hi, 2.0 * hi
            if hi > 1e300:
                raise _NonFinite
    s = 1.0 if lo < 1.0 < hi else 0.5 * (lo + hi)          # the full Newton step first
    for _ in range(200):
        f, fp = dphi(s)
        if f == 0.0:
            return s, evals
        if f > 0.0:
            lo = s
        else:
            hi = s
        s_new = s - f / fp if fp < 0.0 else 0.5 * (lo + hi)
        if not (lo < s_new < hi):
            s_new = 0.5 * (lo + hi)
        if abs(s_new - s) <= LINE_SEARCH_RTOL * s_new or (hi - lo) <= LINE_SEARCH_RTOL * hi:
            s = s_new
            break
        s = s_new
    if not (np.isfinite(s) and s > 0.0 and np.all(1.0 + s * mu > 0.0)):
        raise _NonFinite
    return s, evals


def _run_algorithm1(b, eps):
    """Algorithm 1 on the face from y_tilde; returns (y, converged, outer, newton, line_search_evals, t_final)."""
    bp = b / np.linalg.norm(b)
    Nb = _N.T @ bp
    y = Y_TILDE.copy()
    t = T0
    outer = newton = evals = 0
    t_final = 0.0
    try:
        while t < 1.0 / eps:
            outer += 1
            steps = 0
            while True:
                dz, lam2, G = _newton_direction(y, t, Nb)
                if lam2 < 2.0 * eps:
                    break
                if steps >= MAX_NEWTON_PER_CENTRING or newton >= MAX_NEWTON_TOTAL:
                    return y, False, outer, newton, evals, t
                s, ev = _line_search(dz, G, t, Nb)
                evals += ev
                y_new = y + _N @ (s * dz)
                if not np.all(np.isfinite(y_new)):
                    raise _NonFinite
                y = y_new
                steps += 1
                newton += 1
            t_final = t
            t *= MU
    except (_NonFinite, np.linalg.LinAlgError, FloatingPointError):
        return y, False, outer, newton, evals, t
    return y, True, outer, newton, evals, t_final


def _extract(y):
    X = slack(y)[1:5, 1:5]
    X = 0.5 * (X + X.T)
    _, U = np.linalg.eigh(X)
    q = U[:, -1].copy()
    if q[int(np.argmax(np.abs(q)))] < 0.0:
        q = -q
    return quat_to_rot(q), X


def solve_moments(M, eps=EPS_DEFAULT):
    """Algorithm 1 on a given 9 x 9 cost matrix M: the result without translation (t None, cost = r^T M r)."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (9, 9) or not np.all(np.isfinite(M)):
        raise ValueError("M must be a finite 9 x 9 matrix")
    M = 0.5 * (M + M.T)
    b = quartic_coefficients(M)
    if not np.any(b != 0.0):
        R = np.eye(3)
        X = slack(Y_TILDE)[1:5, 1:5]
        return NPnPResult(R=R, t=None, gamma=0.0, cost=0.0, gap=0.0, converged=True, outer_iters=0,
                          newton_iters=0, line_search_evals=0, t_final=0.0, moment_qq=X)
    y, conv, outer, newton, evals, t_final = _run_algorithm1(b, eps)
    R, X = _extract(y)
    gamma = float(-(b @ y))                                 # L(p), original units (y = -L)
    r = R.reshape(9)
    cost = float(r @ M @ r)
    return NPnPResult(R=R, t=None, gamma=gamma, cost=cost, gap=cost - gamma, converged=bool(conv),
                      outer_iters=outer, newton_iters=newton, line_search_evals=evals, t_final=float(t_final),
                      moment_qq=X)


def solve(P, V, eps=EPS_DEFAULT):
    """Newton-PnP on world points P and bearings V (n x 3 each, n >= 3): the pose cam_T_world as NPnPResult."""
    _validate(P, V)
    M = rotation_cost_matrix(P, V)
    res = solve_moments(M, eps)
    t = optimal_translation(P, V, res.R)
    cost = object_space_cost(P, V, res.R, t)
    return NPnPResult(R=res.R, t=t, gamma=res.gamma, cost=cost, gap=cost - res.gamma, converged=res.converged,
                      outer_iters=res.outer_iters, newton_iters=res.newton_iters,
                      line_search_evals=res.line_search_evals, t_final=res.t_final, moment_qq=res.moment_qq)
