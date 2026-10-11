#!/usr/bin/env python3
"""Generate npnp_constants.npz, the input-independent constants of npnp.py (stage 04o, spec C1).

What it writes (arrays of npnp_constants.npz):
  A           (70, 241) float64, integer entries: the coefficient-matching matrix of the primal (paper eq. 5).
              Column 0 is gamma, columns 1..15 the coefficients of lambda(q) over m(q), columns 16..240 the entries
              Pm[i, j] (row-major, 16 + 15 i + j) of the 15 x 15 Gram matrix. Row a is the monomial q^a of degree
              <= 4. The primal is  A x = b, minimize c^T x = -gamma, x_{16:241} = vec(Pm) PSD, which is the identity
                  r(q)^T M r(q) = gamma + lambda(q) (||q||^2 - 1) + m(q)^T Pm m(q)
              read coefficient by coefficient (b = the coefficients of the quartic r(q)^T M r(q)).
  c           (241,) float64: (-1, 0, ..., 0).
  y_tilde     (70,) float64: the start point of Algorithm 1, the analytic centre of the reduced barrier (npnp.py,
              module docstring): y_tilde = -E[q^a] for q uniform on the unit 3-sphere, in closed form. The minus sign
              is the dual convention of c = (-1, 0, ..., 0): (c - A^T y)_0 = 0 reads y_0 = -1, so y = -L with L the
              pseudo-moment functional.
  provenance  a string array: this script's path and a paragraph describing the construction.

Monomial order (shared with npnp.py, which checks it at import): the exponent tuples a = (a_w, a_x, a_y, a_z) of
q = (w, x, y, z), sorted by total degree and, within a degree, lexicographically DESCENDING. So m(q) = (1, w, x, y,
z, w^2, wx, wy, wz, x^2, xy, xz, y^2, yz, z^2) and the first 15 of the 70 degree-<=4 monomials are m(q).

Before writing, the script checks y_tilde (spec C1): the 16 equalities hold to 1e-12, the smallest eigenvalue of
the reduced slack S~(y_tilde) = W^T S(y_tilde) W is positive, the gradient of h~ = log det S~ projected onto the null
space of the equality rows is below 1e-8 in max-abs, and ||S(y_tilde) v|| / ||S(y_tilde)||_F is below 1e-12 (v the
coefficients of ||q||^2 - 1 over m(q)). It prints the four numbers and exits 1 if one fails; nothing is written then.

Usage: .venv/bin/python scripts/make_npnp_constants.py [--out PATH]   (default: npnp_constants.npz next to npnp.py)
Imports numpy and the standard library only; it does not import npnp (which loads the file this script writes).
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def exponents(d):
    """Exponent tuples of the monomials of q in R^4 of degree <= d: by degree, then lexicographically descending."""
    out = [a for a in itertools.product(range(d + 1), repeat=4) if sum(a) <= d]
    return sorted(out, key=lambda a: (sum(a), tuple(-x for x in a)))


E2 = exponents(2)   # the 15 monomials of m(q)
E4 = exponents(4)   # the 70 monomials indexing the equality rows and the dual variable y
IDX4 = {a: k for k, a in enumerate(E4)}
SQUARES = [tuple(2 if k == i else 0 for k in range(4)) for i in range(4)]


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def build_A_c():
    """A (70 x 241) and c (241) by exact coefficient matching (integer arithmetic, stored as float64)."""
    A = np.zeros((70, 241), dtype=np.int64)
    A[IDX4[(0, 0, 0, 0)], 0] = 1                        # gamma * 1
    for j, a in enumerate(E2):                          # lambda_j * m_j(q) * (||q||^2 - 1)
        for s in SQUARES:
            A[IDX4[add(a, s)], 1 + j] += 1
        A[IDX4[a], 1 + j] -= 1
    for i, a in enumerate(E2):                          # Pm[i, j] * m_i(q) m_j(q)
        for j, b in enumerate(E2):
            A[IDX4[add(a, b)], 16 + 15 * i + j] += 1
    c = np.zeros(241, dtype=np.int64)
    c[0] = -1
    return A.astype(np.float64), c.astype(np.float64)


def uniform_sphere_moment(a):
    """E[q^a] for q uniform on the unit 3-sphere: Gamma(2)/Gamma(2+|a|/2) prod_i Gamma((a_i+1)/2)/Gamma(1/2)."""
    if any(x % 2 for x in a):
        return 0.0
    val = math.gamma(2.0) / math.gamma(2.0 + sum(a) / 2.0)
    for x in a:
        val *= math.gamma((x + 1) / 2.0) / math.gamma(0.5)
    return val


def check_start(A, c, y):
    """The four numbers of spec C1 at y: equality residual, min eig of S~, projected gradient of h~, ||S v||/||S||."""
    Aeq, ceq = A[:, :16].T, c[:16]
    Ak = A[:, 16:].reshape(70, 15, 15)
    S = (c[16:] - A[:, 16:].T @ y).reshape(15, 15)
    Sr = S[1:, 1:]                                      # W^T S W, W = I with the constant column deleted
    Akr = Ak[:, 1:, 1:]
    eq_res = float(np.max(np.abs(Aeq @ y - ceq)))
    min_eig = float(np.linalg.eigvalsh(Sr)[0])
    Sri = np.linalg.inv(Sr)
    grad = -np.einsum("ij,kji->k", Sri, Akr)            # d/dy_k log det S~ = -tr(S~^-1 A~_k)
    N = np.linalg.svd(Aeq)[2][16:].T                    # orthonormal basis of the null space of the equality rows
    proj = float(np.max(np.abs(N @ (N.T @ grad))))
    v = np.zeros(15)
    v[0] = -1.0
    for k, a in enumerate(E2):
        if a in SQUARES:
            v[k] = 1.0
    kern = float(np.linalg.norm(S @ v) / np.linalg.norm(S))
    return eq_res, min_eig, proj, kern


PROVENANCE = (
    "Generated by scripts/make_npnp_constants.py (stage 04o, spec C1). A (70 x 241) and c (241) are the constant data "
    "of the first-level SOS relaxation of Newton-PnP (Jubran et al. 2022, eq. 4-6): x = (gamma, 15 coefficients of "
    "lambda(q) over m(q), the 225 row-major entries of the 15 x 15 Gram matrix Pm), rows = the 70 monomials of q of "
    "degree <= 4 (by degree, then lexicographically descending exponents of q = (w, x, y, z)), built by exact integer "
    "coefficient matching of r(q)^T M r(q) = gamma + lambda(q)(||q||^2 - 1) + m(q)^T Pm m(q); c = (-1, 0, ..., 0). "
    "y_tilde = -E[q^a] for q uniform on the unit 3-sphere, from the closed-form Gamma-function moments (the sign is "
    "the dual convention of c); it is the analytic centre of log det of the slack reduced to the 14-dimensional face "
    "(constant monomial row and column deleted; design.md D1b-1) and was checked, not searched for: 16 equalities to "
    "1e-12, reduced slack positive definite, feasible-direction gradient below 1e-8, ||S v|| / ||S||_F below 1e-12."
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "npnp_constants.npz"), help="output .npz path")
    args = ap.parse_args(argv)
    assert len(E2) == 15 and len(E4) == 70 and E4[:15] == E2 and E2[0] == (0, 0, 0, 0)
    A, c = build_A_c()
    y_tilde = -np.array([uniform_sphere_moment(a) for a in E4], dtype=np.float64)
    eq_res, min_eig, proj, kern = check_start(A, c, y_tilde)
    ok = {"equality residual (<= 1e-12)": eq_res <= 1e-12,
          "min eigenvalue of S~ (> 0)": min_eig > 0.0,
          "projected gradient of h~, max-abs (< 1e-8)": proj < 1e-8,
          "||S v|| / ||S||_F (< 1e-12)": kern < 1e-12}
    for (name, good), val in zip(ok.items(), (eq_res, min_eig, proj, kern)):
        print(f"{name}: {val:.6e} {'ok' if good else 'FAILED'}")
    if not all(ok.values()):
        print("y_tilde failed its check; nothing written")
        return 1
    prov = np.array(["scripts/make_npnp_constants.py", PROVENANCE])
    with open(args.out, "wb") as f:
        np.savez(f, A=A, c=c, y_tilde=y_tilde, provenance=prov)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
