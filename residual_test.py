"""
Stage 04-m1 (milestone 04-4; rulings R63-1, R63-7, workshop/63): registered risk 1's residual-consistency
test of a pose the 30-inlier count accepts.

The final inliers of an accepted pose have squared sigma-normalized residual norms e_i = ||r_i / sigma_i||^2.
Under the noise model each e_i is a chi-square draw with 2 degrees of freedom, and an inlier is one with
e_i <= cutoff (MapTrackerParams.chi2 = 5.991), so the inliers' e_i are chi-square(2) draws truncated to
[0, cutoff]. The test compares the mean of the n values with the one-sided LEVEL point of the mean of n such
draws (R63-7: one-sided, the level of the 5.991 inlier cutoff; the exact distribution, never the normal
approximation).

The critical point is computed by convolution: the truncated density exp(-x/2) / (2 (1 - exp(-cutoff/2))) is
replaced by its exact cell masses on a uniform grid of step h; the lattice is shifted so that the discrete
draw has the exact mean of the truncated draw (the midpoint rule's bias, about h^2 / 24 per draw, would not
shrink with n); the n-fold convolution is the n-th power of the masses' discrete Fourier transform, on a
length that holds the whole support (no wrap-around); the quantile is read from the cumulative masses with
each lattice mass spread over its cell (linear interpolation of the cumulative masses). The grid step grows
with n so that the transform length stays near 2^20: the remaining error is the per-draw variance excess
h^2 / 12 and the interpolation, both far below 1e-3 on the mean for 1 <= n <= 5000.

Pure numpy and the standard library; no product module, no random numbers. Deterministic.
"""
from __future__ import annotations

import functools
import math

import numpy as np

LEVEL = 0.95                    # R63-7: one-sided, the level of the 5.991 inlier cutoff

_TARGET_LEN = 1 << 20           # transform length the grid step is chosen for
_H_MIN = 1e-4                   # finest grid step (small n)


def _truncated_mean(cutoff):
    """Mean of chi-square(2) truncated to [0, cutoff]: 2 - cutoff exp(-cutoff/2) / (1 - exp(-cutoff/2))."""
    z = -math.expm1(-cutoff / 2.0)
    return 2.0 - cutoff * math.exp(-cutoff / 2.0) / z


@functools.lru_cache(maxsize=None)
def _critical_mean_cached(n, cutoff, level):
    h = max(_H_MIN, cutoff * n / _TARGET_LEN)
    m = int(math.ceil(cutoff / h - 1e-12))
    edges = np.minimum(np.arange(m + 1, dtype=np.float64) * h, cutoff)
    # exact cell masses of the truncated density: (exp(-a/2) - exp(-b/2)) / (1 - exp(-cutoff/2))
    z = -math.expm1(-cutoff / 2.0)
    p = (np.exp(-edges[:-1] / 2.0) - np.exp(-edges[1:] / 2.0)) / z
    p = p / p.sum()
    # the lattice value of cell j is j h + delta, delta chosen so the discrete mean is exact
    delta = _truncated_mean(cutoff) - float(np.dot(p, np.arange(m, dtype=np.float64))) * h
    if n == 1:
        q = p
    else:
        length = n * (m - 1) + 1
        nfft = 1 << int(math.ceil(math.log2(length)))
        q = np.fft.irfft(np.fft.rfft(p, nfft) ** n, nfft)[:length]
        q = np.clip(q, 0.0, None)
        q = q / q.sum()
    cum = np.cumsum(q)
    # the cumulative mass at the right edge of lattice cell J (value J h + n delta, spread over +-h/2)
    j = int(np.searchsorted(cum, level, side="left"))
    j = min(j, len(cum) - 1)
    c_hi = float(cum[j])
    c_lo = float(cum[j - 1]) if j > 0 else 0.0
    frac = 0.0 if c_hi <= c_lo else (level - c_lo) / (c_hi - c_lo)
    s = (j - 0.5 + frac) * h + n * delta
    return float(s / n)


def critical_mean(n, cutoff, level=LEVEL):
    """The level-quantile of the mean of n independent chi-square(2) draws truncated to [0, cutoff].

    n is an int >= 1 (ValueError otherwise); cutoff > 0. Exact up to numerical error (absolute error well
    below 1e-3 for 1 <= n <= 5000); cached per (n, cutoff, level)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError(f"critical_mean: n must be an int >= 1, got {n!r}")
    cutoff = float(cutoff)
    if not (cutoff > 0.0) or not math.isfinite(cutoff):
        raise ValueError(f"critical_mean: cutoff must be a finite value > 0, got {cutoff!r}")
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"critical_mean: level must be in (0, 1), got {level!r}")
    return _critical_mean_cached(int(n), cutoff, level)


def evaluate(e2, cutoff, level=LEVEL):
    """The test on the final inliers' squared normalized residual norms e2 (1-D). Returns dict(n, stat, crit,
    max_e2, passed) with Python ints and floats: stat = mean(e2), crit = critical_mean(n, cutoff, level),
    passed = stat <= crit. With no inlier, stat, crit and max_e2 are nan and passed is False."""
    e2 = np.asarray(e2, dtype=np.float64).reshape(-1)
    n = int(len(e2))
    if n == 0:
        nan = float("nan")
        return dict(n=0, stat=nan, crit=nan, max_e2=nan, passed=False)
    stat = float(np.mean(e2))
    crit = float(critical_mean(n, cutoff, level))
    return dict(n=n, stat=stat, crit=crit, max_e2=float(np.max(e2)), passed=bool(stat <= crit))
