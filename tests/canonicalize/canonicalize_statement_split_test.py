# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-frontend (numpy-style) value-preservation tests for the statement /
loop-nesting scenarios the redesign targets.

The hard invariant for EVERY case is that canonicalization preserves the value:
the canonicalized SDFG must compute exactly what the un-canonicalized (direct
frontend) SDFG does. Parallelization (maps) is best-effort on top; where the
current pipeline already achieves it we also assert it, otherwise we only require
correctness (and note the aspiration for SplitStatements / PerfectLoopNesting).

Scenarios (grouped):
* Independent / dependent-but-splittable statements (A=B+C; V=2*A, etc.).
* Anti-dependence by value -- a later statement reads the ORIGINAL (pre-write)
  value of an array the loop also writes (s1244 / s1213 shapes; a recurrence
  whose neighbour read sees the un-updated element).
* Genuinely loop-carried statements that must stay sequential (prefix sum).
* Perfect vs. imperfect 2-D nests.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _canon_vs_raw(prog, inputs, **symbols):
    """Compile the un-canonicalized SDFG (ground truth) and the canonicalized SDFG
    on identical inputs; assert they agree. Returns the canonicalized SDFG so
    callers can additionally probe structure (maps / loops)."""
    raw = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in inputs.items()}
    raw.compile()(**ref, **symbols)

    cand = prog.to_sdfg(simplify=True)
    canonicalize(cand, validate=True, peel_limit=4, break_anti_dependence=True)
    got = {k: v.copy() for k, v in inputs.items()}
    cand.compile()(**got, **symbols)

    for k in inputs:
        assert np.allclose(ref[k], got[k], equal_nan=True), f"{prog.name}: '{k}' diverged after canonicalize"
    return cand


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).random(shape)


# ===========================================================================
# Independent / dependent-but-splittable statements
# ===========================================================================
@dace.program
def _dependent_same_index(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], V: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        V[i] = 2.0 * A[i]


def test_dependent_same_index_splittable():
    """V depends on A at the SAME index (RAW per-iteration) -> two independent
    parallel statements after splitting; always value-preserving."""
    n = 48
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=1), 'C': _rand(n, seed=2), 'V': np.zeros(n)}
    cand = _canon_vs_raw(_dependent_same_index, ins, N=n)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1, "elementwise dependent statements should parallelize"


@dace.program
def _two_independent(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N],
                     E: dace.float64[N], F: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        D[i] = E[i] * F[i]


def test_two_independent_statements():
    n = 40
    ins = {k: (_rand(n, seed=ord(k)) if k not in 'AD' else np.zeros(n)) for k in 'ABCDEF'}
    cand = _canon_vs_raw(_two_independent, ins, N=n)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1


@dace.program
def _chain_three(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], V: dace.float64[N], W: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        V[i] = 2.0 * A[i]
        W[i] = V[i] + 1.0


def test_chain_three_statements():
    n = 33
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=3), 'C': _rand(n, seed=4), 'V': np.zeros(n), 'W': np.zeros(n)}
    _canon_vs_raw(_chain_three, ins, N=n)


@dace.program
def _pass_through_write_read(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        C[i] = A[i] + A[i]  # direct read of the just-written A[i] (same index)


def test_pass_through_write_then_direct_read():
    n = 36
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=5), 'C': np.zeros(n)}
    _canon_vs_raw(_pass_through_write_read, ins, N=n)


# ===========================================================================
# Anti-dependence by value: a later statement reads the ORIGINAL value of an
# array the loop also writes.
# ===========================================================================
@dace.program
def _forward_read_antidep(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N]):
    for i in range(N - 1):
        A[i] = B[i] + 1.0
        D[i] = A[i] + A[i + 1]  # A[i+1] is the ORIGINAL (not-yet-written) value


def test_forward_read_antidependence_s1244():
    """s1244 shape: D reads A[i+1] which the fused loop has NOT written yet.
    Splitting D into its own loop needs a snapshot of the original A; the hard
    requirement here is that the value is preserved however it is lowered."""
    n = 50
    ins = {'A': _rand(n, seed=6), 'B': _rand(n, seed=7), 'D': np.zeros(n)}
    _canon_vs_raw(_forward_read_antidep, ins, N=n)


@dace.program
def _cross_antidep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in range(1, N - 1):
        A[i] = B[i - 1] + C[i]
        B[i] = A[i + 1] * D[i]  # reads original A[i+1]; overwrites B (WAR vs the A read)


def test_cross_antidependence_s1213():
    n = 44
    ins = {'A': _rand(n, seed=8), 'B': _rand(n, seed=9), 'C': _rand(n, seed=10), 'D': _rand(n, seed=11)}
    _canon_vs_raw(_cross_antidep, ins, N=n)


K = dace.symbol('K')


@dace.program
def _sym_forward_antidep(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N]):
    for i in range(N - K):
        A[i] = B[i] + 1.0
        D[i] = A[i] + A[i + K]  # A[i+K] is the ORIGINAL value; K a positive symbol


def test_symbolic_forward_offset_antidependence():
    """Same shape as s1244 but the read-ahead offset is the runtime symbol ``K``
    (``a[i + K]``). Under the nonnegative-symbol assumption this is a forward-read
    anti-dependence: SplitStatements snapshots ``A`` (guarding ``K >= 0``) and the
    value must be preserved for the concrete positive ``K``."""
    n, k = 50, 3
    ins = {'A': _rand(n, seed=21), 'B': _rand(n, seed=22), 'D': np.zeros(n)}
    cand = _canon_vs_raw(_sym_forward_antidep, ins, N=n, K=k)
    assert _nmaps(cand) >= 1, "the symbolic-offset anti-dependence should still parallelize a cone"


@dace.program
def _recurrence_then_read_original(A: dace.float64[N], Bout: dace.float64[N]):
    for i in range(1, N - 1):
        A[i] = A[i - 1] * 0.5      # sequential recurrence on A
        Bout[i] = A[i + 1]         # reads the ORIGINAL A[i+1] (pre-loop value)


def test_recurrence_plus_read_of_original_value():
    """User's 'second statement only relies on the pre-loop value': the B
    statement reads A[i+1] before the recurrence overwrites it. B is separable
    (snapshot of original A) while the A recurrence stays sequential."""
    n = 40
    ins = {'A': _rand(n, seed=12), 'Bout': np.zeros(n)}
    _canon_vs_raw(_recurrence_then_read_original, ins, N=n)


# ===========================================================================
# Genuinely loop-carried: must stay a correct sequential loop (or a scan).
# ===========================================================================
@dace.program
def _prefix_sum(A: dace.float64[N], B: dace.float64[N]):
    for i in range(1, N):
        A[i] = A[i - 1] + B[i]


def test_prefix_sum_stays_correct():
    """A[i] = A[i-1] + B[i] is a genuine carried recurrence. Correctness is the
    invariant: a naive parallel map over the carried axis would give WRONG values,
    so the value-preservation check (in ``_canon_vs_raw``) is what proves it was
    not mis-parallelized. The pipeline lowers it to a Scan -- the desired
    semantic-op recognition -- which is value-exact."""
    n = 32
    ins = {'A': _rand(n, seed=13), 'B': _rand(n, seed=14)}
    _canon_vs_raw(_prefix_sum, ins, N=n)


# ===========================================================================
# Perfect vs. imperfect 2-D nests.
# ===========================================================================
@dace.program
def _perfect_2d(A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            A[i, j] = B[i, j] + C[i, j]


def test_perfect_2d_nest_parallelizes():
    n, m = 12, 10
    ins = {'A': np.zeros((n, m)), 'B': _rand(n, m, seed=15), 'C': _rand(n, m, seed=16)}
    cand = _canon_vs_raw(_perfect_2d, ins, N=n, M=m)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1


@dace.program
def _imperfect_2d_siblings(A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M], D: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            A[i, j] = B[i, j] + C[i, j]
        for j in range(M):
            D[i, j] = A[i, j] * 2.0


def test_imperfect_2d_sibling_inner_loops():
    """Two sibling inner loops under one outer loop -- an imperfect nest that
    needs fission to perfect-nest. Correctness is the invariant regardless."""
    n, m = 10, 8
    ins = {'A': np.zeros((n, m)), 'B': _rand(n, m, seed=17), 'C': _rand(n, m, seed=18), 'D': np.zeros((n, m))}
    _canon_vs_raw(_imperfect_2d_siblings, ins, N=n, M=m)


@dace.program
def _inner_carried_2d(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in range(N):
        for j in range(1, M):
            A[i, j] = A[i, j - 1] + B[i, j]


def test_inner_carried_outer_parallel_2d():
    """Inner axis carries a recurrence, outer axis is data-parallel: correctness
    always; ideally the outer i becomes a map with j a sequential/scan inner."""
    n, m = 12, 9
    ins = {'A': _rand(n, m, seed=19), 'B': _rand(n, m, seed=20)}
    _canon_vs_raw(_inner_carried_2d, ins, N=n, M=m)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
