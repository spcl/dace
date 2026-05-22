# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize refusal contracts on loop-carried recurrences.

Distilled from the production kernels (faithful Fortran shapes re-expressed
via the Python frontend):

* **Tridiagonal / Thomas solve** -- ``mo_solve_nonhydro.f90:2385-2416``
  (forward elimination ``z_q(jk) = f(z_q(jk-1))`` then back substitution
  ``w(jk) = f(w(jk+1))``). The vertical ``jk`` axis is loop-carried; the
  horizontal ``jc`` axis is fully parallel. **Refusal contract**:
  canonicalize must keep the ``jk`` sweeps SEQUENTIAL (a ``LoopRegion``,
  never a Map) -- "parallelizing" a Thomas sweep is a correctness bug --
  while it MAY map the independent ``jc`` axis.

* **Vertical flux prefix-scan / sedimentation** --
  ``cloudsc.F90:2705-2709`` (``flux(jk+1) = f(flux(jk))``, falling
  precipitation accumulating downward). Loop-carried over ``jk``, parallel
  over ``jl``. Same refusal contract on the carried axis.

The primary contract is numerical: canonicalize must be value-preserving
on these recurrences (a wrongly-parallelized sweep would diverge).
Structural assertions additionally pin that the carried axis stays a
``LoopRegion``.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')  # horizontal (parallel) extent, e.g. columns / jl
K = dace.symbol('K')  # vertical (carried) extent, e.g. levels / jk


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


# ----------------------------------------------------------------------
# Tridiagonal Thomas solve (forward elimination + back substitution)
# ----------------------------------------------------------------------


@dace.program
def thomas_solve(a: dace.float64[N, K], b: dace.float64[N, K], c: dace.float64[N, K], d: dace.float64[N, K],
                 x: dace.float64[N, K]):
    """Per-column tridiagonal solve. ``jc`` (= ``i``, rows/columns) is
    parallel; ``jk`` (vertical) is loop-carried in both sweeps. Mirrors the
    ICON implicit vertical solver shape. The outer ``i`` is written as a
    plain ``range`` so canonicalize must DISCOVER its parallelism while
    keeping the carried ``k`` sweeps sequential (writing it as ``dace.map``
    instead asserts a parallelism the in-place carried array does not have
    at the frontend level)."""
    for i in range(N):
        # forward elimination: cp[k], dp[k] depend on k-1
        cp = np.zeros(K, dace.float64)
        dp = np.zeros(K, dace.float64)
        cp[0] = c[i, 0] / b[i, 0]
        dp[0] = d[i, 0] / b[i, 0]
        for k in range(1, K):
            m = b[i, k] - a[i, k] * cp[k - 1]
            cp[k] = c[i, k] / m
            dp[k] = (d[i, k] - a[i, k] * dp[k - 1]) / m
        # back substitution: x[k] depends on x[k+1]
        x[i, K - 1] = dp[K - 1]
        for k in range(K - 2, -1, -1):
            x[i, k] = dp[k] - cp[k] * x[i, k + 1]


def _thomas_oracle(a, b, c, d):
    n, k = a.shape
    x = np.zeros((n, k))
    for i in range(n):
        cp = np.zeros(k)
        dp = np.zeros(k)
        cp[0] = c[i, 0] / b[i, 0]
        dp[0] = d[i, 0] / b[i, 0]
        for kk in range(1, k):
            m = b[i, kk] - a[i, kk] * cp[kk - 1]
            cp[kk] = c[i, kk] / m
            dp[kk] = (d[i, kk] - a[i, kk] * dp[kk - 1]) / m
        x[i, k - 1] = dp[k - 1]
        for kk in range(k - 2, -1, -1):
            x[i, kk] = dp[kk] - cp[kk] * x[i, kk + 1]
    return x


def _diagonally_dominant_tridiag(n, k, rng):
    a = rng.uniform(-1.0, -0.3, (n, k))  # sub-diagonal
    c = rng.uniform(-1.0, -0.3, (n, k))  # super-diagonal
    b = np.abs(a) + np.abs(c) + rng.uniform(1.0, 2.0, (n, k))  # dominant diagonal
    d = rng.uniform(-1.0, 1.0, (n, k))
    return a, b, c, d


def test_thomas_solve_value_preserving():
    n, k = 5, 6
    rng = np.random.default_rng(20)
    a, b, c, d = _diagonally_dominant_tridiag(n, k, rng)
    exp = _thomas_oracle(a, b, c, d)
    sdfg = thomas_solve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    x = np.zeros((n, k))
    sdfg(a=a.copy(), b=b.copy(), c=c.copy(), d=d.copy(), x=x, N=n, K=k)
    assert np.allclose(x, exp), 'Thomas solve mis-canonicalized (carried vertical axis likely parallelized)'


def test_thomas_solve_keeps_vertical_axis_sequential():
    """The two ``jk`` sweeps are loop-carried; they must remain
    ``LoopRegion`` s after canonicalize (never become Maps). At least one
    surviving loop is mandatory."""
    sdfg = thomas_solve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nloops(sdfg) >= 1, 'the loop-carried vertical sweeps must stay sequential LoopRegions'


# ----------------------------------------------------------------------
# Vertical flux prefix-scan / sedimentation
# ----------------------------------------------------------------------


@dace.program
def vertical_flux_prefix_scan(fall: dace.float64[N, K], flux: dace.float64[N, K]):
    """``flux[i, k] = flux[i, k-1] * decay + fall[i, k]`` -- falling
    precipitation accumulating downward over levels ``k`` (loop-carried),
    independent across columns ``i``. cloudsc ZPFPLSX sedimentation shape.
    Outer ``i`` is a plain ``range`` (canonicalize discovers its
    parallelism); the carried ``k`` accumulation must stay sequential."""
    for i in range(N):
        flux[i, 0] = fall[i, 0]
        for k in range(1, K):
            flux[i, k] = flux[i, k - 1] * 0.9 + fall[i, k]


def _prefix_scan_oracle(fall):
    n, k = fall.shape
    flux = np.zeros((n, k))
    for i in range(n):
        flux[i, 0] = fall[i, 0]
        for kk in range(1, k):
            flux[i, kk] = flux[i, kk - 1] * 0.9 + fall[i, kk]
    return flux


def test_vertical_flux_prefix_scan_value_preserving():
    n, k = 7, 8
    rng = np.random.default_rng(21)
    fall = rng.uniform(0.0, 1.0, (n, k))
    exp = _prefix_scan_oracle(fall)
    sdfg = vertical_flux_prefix_scan.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    flux = np.zeros((n, k))
    sdfg(fall=fall.copy(), flux=flux, N=n, K=k)
    assert np.allclose(flux, exp), 'prefix-scan mis-canonicalized (carried level axis likely parallelized)'


def test_vertical_flux_prefix_scan_keeps_level_axis_sequential():
    """The downward ``k`` accumulation is loop-carried and must stay a
    ``LoopRegion`` after canonicalize."""
    sdfg = vertical_flux_prefix_scan.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert _nloops(sdfg) >= 1, 'the loop-carried level accumulation must stay a sequential LoopRegion'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
