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


# ----------------------------------------------------------------------
# Loop-carried scratch SLOT across a fully unrolled loop
# ----------------------------------------------------------------------

SLOT_N, SLOT_T = 8, 4


@dace.program
def carried_scratch_slot(A: dace.float64[SLOT_N], hist: dace.float64[SLOT_T], out: dace.float64[SLOT_T, SLOT_N]):
    """One scratch element ``arr[1]`` rewritten and re-read every iteration.

    Both trip counts are constant, so canonicalize fully unrolls the nest and every
    iteration becomes its own state holding a write of ``arr[1]`` and the read that
    consumes it. Nothing links iteration ``i`` to iteration ``i+1`` by dataflow -- the
    ordering is purely the state sequence -- so a state fusion that does not carry that
    ordering over silently makes ``out[t, i]`` read a later iteration's value.
    ``hist`` keeps the slot live across the outer iteration.
    """
    arr = np.zeros((4, ), dtype=np.float64)
    arr[1] = 5.0
    for t in range(SLOT_T):
        hist[t] = arr[1] * 2.0
        for i in range(SLOT_N):
            arr[1] = A[i] + t
            out[t, i] = arr[1] * 3.0


def _carried_scratch_slot_oracle(A):
    hist = np.zeros(SLOT_T)
    out = np.zeros((SLOT_T, SLOT_N))
    slot = 5.0
    for t in range(SLOT_T):
        hist[t] = slot * 2.0
        for i in range(SLOT_N):
            slot = A[i] + t
            out[t, i] = slot * 3.0
    return hist, out


def _ambiguous_write_nodes(sdfg):
    """AccessNodes with more than one incoming DATA write AND at least one read.

    Such a node has no defined value for its readers: which of the writes they observe
    is decided by whatever order codegen happens to emit. Empty (happens-before) memlets
    are excluded -- they carry no data.
    """
    hits = []
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            writes = [e for e in state.in_edges(node) if not e.data.is_empty()]
            reads = [e for e in state.out_edges(node) if not e.data.is_empty()]
            if len(writes) > 1 and reads:
                hits.append((state.label, node.data, len(writes), len(reads)))
    return hits


def test_carried_scratch_slot_value_preserving():
    """Regression: ``StateFusion._check_paths`` used to treat "SOME first-state write of
    the candidate is ordered before the second state" as "ALL of them are", so an
    unrolled iteration's write to the scratch slot was fused in unordered next to the
    following iteration's write. ``out[t, N-2]`` then received ``out[t, N-1]``'s value.
    The SDFG stayed valid -- only the numbers were wrong."""
    A = np.arange(1.0, SLOT_N + 1.0)
    exp_hist, exp_out = _carried_scratch_slot_oracle(A)
    sdfg = carried_scratch_slot.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    sdfg.validate()
    assert not _ambiguous_write_nodes(sdfg), \
        f'fusion merged unordered writes onto one AccessNode: {_ambiguous_write_nodes(sdfg)}'
    hist = np.zeros(SLOT_T)
    out = np.zeros((SLOT_T, SLOT_N))
    sdfg(A=A, hist=hist, out=out)
    assert np.allclose(hist, exp_hist)
    assert np.allclose(out, exp_out), 'unrolled iterations of the carried scratch slot were mis-fused'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
