# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loops that ARE parallel but that affine / polyhedral analysis cannot prove.

Each kernel here is genuinely parallelizable, and each is out of reach of the dependence
analysis canonicalize ships today -- non-linear subscripts, array indirection, or a fact that
lives in a programmer assertion rather than in the index expressions. They are the target set
for an SMT-backed dependence oracle.

Every case is asserted twice:

* **numerics** -- unconditionally. Canonicalize is value-preserving, so whatever it decides,
  the kernel must still compute the reference. These assertions guard against the far worse
  outcome than "left sequential": wrongly parallelizing one of these shapes.
* **parallelism** -- ``xfail(strict=True)``. Today the analysis refuses, which is the SOUND
  answer without a solver. When the oracle lands these XPASS and fail the suite, forcing the
  author to flip them rather than letting the new capability land unnoticed.

The negative case must stay refused forever; it is marked as a normal (non-xfail) assertion
because no solver should ever certify it.

Sources: cases 1-4 are the SMT-required shapes from the parallelization paper (guarded
polynomial indirection, asserted unique scatter, hybrid sparse partitioning, and the CLOUDSC
triangular nest where the solver found 6 reducible loops DaCe's heuristics missed).

A solver proving a loop parallel only UNDER an assumption (case 2's uniqueness precondition is
the archetype) must not silently assume it: the assumption is a CONTRACT and belongs in the
machinery that already exists for this --
:func:`~dace.transformation.passes.canonicalize.tracked_assumptions.record_assumption` plus the
runtime trap emitted by ``AssumeSymbolConstraints``, the same way ``ScatterToGuardedMaps``
(``assume_no_conflicts``) and ``ParallelizeUnderConstraint`` (``assume_constraint``) already
turn an unprovable precondition into a checked one.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
K = dace.symbol('K')


def cpu_canon(sdfg: dace.SDFG) -> dace.SDFG:
    canonicalize(sdfg,
                 target='cpu',
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True,
                 validate_all=False)
    return sdfg


def residual_loops(sdfg: dace.SDFG) -> int:
    """Sequential LoopRegions, counted across nested SDFGs (Maps live there too)."""
    return sum(1 for sd in sdfg.all_sdfgs_recursive() for cfr in sd.all_control_flow_regions()
               if isinstance(cfr, LoopRegion) and cfr.loop_variable)


# --------------------------------------------------------------------------- #
# Case 1 -- guarded polynomial indirection.                                    #
# --------------------------------------------------------------------------- #
# ``A[i] = A[IDX[i] * IDX[i-1]] + 1`` under the guard ``IDX[i]*IDX[i-1] > i``. Affine analysis
# sees a non-linear product feeding an indirect read and must assume a loop-carried RAW for
# every iteration. The solver asks whether the write at k can be the read at k+1: a real
# dependence needs ``P == k`` while the guard forces ``P > k+1``, i.e. ``k > k+1`` -- UNSAT, so
# the sequential chain provably breaks.


@dace.program
def guarded_poly_indirection(A: dace.float64[N], IDX: dace.int64[N]):
    for i in range(1, N):
        if IDX[i] * IDX[i - 1] > i:
            A[i] = A[IDX[i] * IDX[i - 1]] + 1.0
        else:
            A[i] = 0.0


def reference_guarded_poly(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = a.copy()
    for i in range(1, a.shape[0]):
        p = int(idx[i]) * int(idx[i - 1])
        out[i] = out[p] + 1.0 if p > i else 0.0
    return out


def test_guarded_poly_indirection_is_value_preserving():
    n = 16
    rng = np.random.default_rng(1)
    a = rng.random(n)
    # Keep every guarded product in range so the reference and the SDFG read the same slots.
    idx = np.ones(n, dtype=np.int64) * 3
    expected = reference_guarded_poly(a, idx)

    sdfg = cpu_canon(guarded_poly_indirection.to_sdfg(simplify=True))
    got = a.copy()
    sdfg(A=got, IDX=idx.copy(), N=n)
    assert np.allclose(got, expected), 'canonicalize must preserve the guarded indirection'


@pytest.mark.xfail(strict=True, reason='needs an SMT oracle: guard makes the RAW chain UNSAT')
def test_guarded_poly_indirection_parallelizes():
    sdfg = cpu_canon(guarded_poly_indirection.to_sdfg(simplify=True))
    assert residual_loops(sdfg) == 0


# --------------------------------------------------------------------------- #
# Case 2 -- asserted unique scatter.                                           #
# --------------------------------------------------------------------------- #
# ``A[IDX[k]] += B[k]``. Whether this is parallel depends on a fact no compiler can read off
# the code: that IDX holds no duplicates. Under the programmer assertion
# ``forall i<j: IDX[i] != IDX[j]`` the solver (LIA + theory of arrays) certifies no two
# iterations collide. Without it the WAW must be assumed. This is the case that MUST come out
# as a contract plus a runtime check, never a silent assumption.


@dace.program
def unique_scatter(A: dace.float64[N], B: dace.float64[N], IDX: dace.int64[N]):
    for k in range(N):
        A[IDX[k]] = A[IDX[k]] + B[k]


def test_unique_scatter_is_value_preserving():
    n = 12
    rng = np.random.default_rng(2)
    a, b = rng.random(n), rng.random(n)
    idx = rng.permutation(n).astype(np.int64)  # the uniqueness precondition holds
    expected = a.copy()
    for k in range(n):
        expected[idx[k]] += b[k]

    sdfg = cpu_canon(unique_scatter.to_sdfg(simplify=True))
    got = a.copy()
    sdfg(A=got, B=b.copy(), IDX=idx.copy(), N=n)
    assert np.allclose(got, expected), 'canonicalize must preserve the scatter'


def test_unique_scatter_already_parallelizes_under_a_runtime_contract():
    """MEASURED: this one needs no solver. ``ScatterToGuardedMaps`` already lifts it to a Map
    guarded by a runtime duplicate-index check, which is precisely the "prove it under a
    precondition, then check the precondition at runtime" contract an SMT oracle would feed.
    Recorded here so the SMT work does not re-solve a solved case."""
    sdfg = cpu_canon(unique_scatter.to_sdfg(simplify=True))
    assert residual_loops(sdfg) == 0, 'the guarded-scatter path should already parallelize this'


def test_unique_scatter_guard_rejects_duplicate_indices():
    """The other half of the contract: feed indices that VIOLATE uniqueness and the emitted
    guard must stop the program rather than race. The trap aborts the process, so this runs in
    a child -- an uncaught SIGILL/abort here is the guard doing its job."""
    import subprocess
    import sys
    src = ('import numpy as np\n'
           'from tests.canonicalize import smt_required_parallel_test as T\n'
           'sdfg = T.cpu_canon(T.unique_scatter.to_sdfg(simplify=True))\n'
           'idx = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)\n'
           'print("BUILT", flush=True)\n'
           'sdfg(A=np.zeros(8), B=np.arange(1.0, 9.0), IDX=idx, N=8)\n'
           'print("NO_TRAP")\n')
    proc = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True, timeout=900)
    ctx = f'(rc={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr[-400:]!r})'
    # Non-vacuity: without BUILT a child that crashed for any other reason prints nothing and the
    # NO_TRAP check passes, so the test could never fail.
    assert 'BUILT' in proc.stdout, f'child died before the guarded call {ctx}'
    assert 'NO_TRAP' not in proc.stdout, f'duplicate indices must trip the guard, not run silently {ctx}'


# --------------------------------------------------------------------------- #
# Case 3 -- hybrid sparse kernel, conditional recurrence.                      #
# --------------------------------------------------------------------------- #
# A sparse row product feeding a CONDITIONAL recurrence: ``y[i] = sum[i]`` for ``i < K`` (no
# carried dependence) and ``y[i] = sum[i] + y[i-1]`` beyond. Seeing ``y[i-1]`` anywhere in the
# body, affine analysis serializes the whole iteration space. The solver refutes FULL
# sequentiality, which lets the loop be PARTITIONED at ``i == K``: ``[0,K)`` parallel,
# ``[K,N)`` a genuine recurrence.


@dace.program
def hybrid_sparse(y: dace.float64[N], val: dace.float64[N], x: dace.float64[N], col: dace.int64[N],
                  row_ptr: dace.int64[N + 1]):
    for i in range(N):
        acc = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            acc = acc + val[k] * x[col[k]]
        if i < K:
            y[i] = acc
        else:
            y[i] = acc + y[i - 1]


@pytest.mark.xfail(strict=True,
                   reason='BUG: canonicalize raises InvalidSDFGEdgeError "Memlet subset negative '
                   'out-of-bounds (y[-1...])" -- the y[i-1] read is hoisted without respecting the '
                   'i >= K guard that makes it in-range')
def test_hybrid_sparse_is_value_preserving():
    n, ksplit = 10, 4
    rng = np.random.default_rng(3)
    row_ptr = np.arange(n + 1, dtype=np.int64)  # one nonzero per row
    col = rng.integers(0, n, size=n).astype(np.int64)
    val, x = rng.random(n), rng.random(n)

    expected = np.zeros(n)
    for i in range(n):
        acc = sum(val[k] * x[col[k]] for k in range(row_ptr[i], row_ptr[i + 1]))
        expected[i] = acc if i < ksplit else acc + expected[i - 1]

    sdfg = cpu_canon(hybrid_sparse.to_sdfg(simplify=True))
    got = np.zeros(n)
    sdfg(y=got, val=val.copy(), x=x.copy(), col=col.copy(), row_ptr=row_ptr.copy(), N=n, K=ksplit)
    assert np.allclose(got, expected), 'the conditional recurrence must be preserved'


@pytest.mark.xfail(strict=True,
                   reason='blocked by the same InvalidSDFGEdgeError as above; then needs an SMT oracle '
                   'to refute FULL sequentiality and split at K')
def test_hybrid_sparse_partitions_at_k():
    """The ``[0,K)`` half is parallel; only ``[K,N)`` need remain a loop."""
    sdfg = cpu_canon(hybrid_sparse.to_sdfg(simplify=True))
    assert residual_loops(sdfg) == 1


# --------------------------------------------------------------------------- #
# Case 4 -- CLOUDSC triangular nest.                                           #
# --------------------------------------------------------------------------- #
# ``j`` ranges over ``[i+1, M)`` and every ``(i, j)`` reads ``(i,i)`` and ``(j,i)`` to update
# ``(j,i)``. Writes land strictly in the lower triangle, so the diagonal stays read-only and
# the nest is reducible -- but only once ``i < j`` is modelled formally. This is the shape
# where the solver found 6 reducible loops that the shipped heuristics missed.

M = dace.symbol('M')


@dace.program
def triangular_update(A: dace.float64[M, M]):
    for i in range(M):
        for j in range(i + 1, M):
            A[j, i] = A[j, i] - A[i, i] * A[j, i]


def test_triangular_update_is_value_preserving():
    m = 8
    rng = np.random.default_rng(4)
    a = rng.random((m, m))
    expected = a.copy()
    for i in range(m):
        for j in range(i + 1, m):
            expected[j, i] = expected[j, i] - expected[i, i] * expected[j, i]

    sdfg = cpu_canon(triangular_update.to_sdfg(simplify=True))
    got = a.copy()
    sdfg(A=got, M=m)
    assert np.allclose(got, expected), 'the triangular update must be preserved'


def test_triangular_update_already_parallelizes():
    """MEASURED: DaCe already reduces this nest (0 residual loops, 2 Maps). The paper reports
    the solver finding 6 such loops that the *published* heuristics missed; this shape is not
    one of the gaps in the current tree. Pinned so a future change cannot silently lose it."""
    sdfg = cpu_canon(triangular_update.to_sdfg(simplify=True))
    assert residual_loops(sdfg) == 0


# --------------------------------------------------------------------------- #
# Case 5 -- non-linear injective subscript.                                    #
# --------------------------------------------------------------------------- #
# ``A[i*i]`` is injective on ``[0, N)``, so the writes never collide, but the subscript is
# quadratic and outside affine reach. The solver discharges
# ``i1*i1 == i2*i2 AND 0 <= i1 < i2 < N`` as UNSAT.


@dace.program
def quadratic_scatter(A: dace.float64[N * N], B: dace.float64[N]):
    for i in range(N):
        A[i * i] = B[i]


def test_quadratic_scatter_is_value_preserving():
    n = 8
    rng = np.random.default_rng(5)
    b = rng.random(n)
    expected = np.zeros(n * n)
    for i in range(n):
        expected[i * i] = b[i]

    sdfg = cpu_canon(quadratic_scatter.to_sdfg(simplify=True))
    got = np.zeros(n * n)
    sdfg(A=got, B=b.copy(), N=n)
    assert np.allclose(got, expected), 'the quadratic scatter must be preserved'


@pytest.mark.xfail(strict=True, reason='needs an SMT oracle: i1^2 == i2^2 with i1 < i2 is UNSAT on [0,N)')
def test_quadratic_scatter_parallelizes():
    sdfg = cpu_canon(quadratic_scatter.to_sdfg(simplify=True))
    assert residual_loops(sdfg) == 0


# --------------------------------------------------------------------------- #
# Negative -- must stay sequential forever.                                    #
# --------------------------------------------------------------------------- #
# ``A[min(i, N-1-i)]`` is NOT injective: ``i`` and ``N-1-i`` collide. A solver asked the same
# question returns SAT, so this must never be parallelized. Deliberately NOT xfail -- the day
# this starts passing in parallel form, something is unsound.


@dace.program
def colliding_scatter(A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[min(i, N - 1 - i)] = B[i]


@pytest.mark.xfail(strict=True,
                   reason='BUG: canonicalize raises ValueError "invalid literal for int() with base 10: '
                   '\'A_slice_0\'" on the min() subscript, so the refusal cannot even be observed')
def test_colliding_scatter_stays_sequential():
    sdfg = cpu_canon(colliding_scatter.to_sdfg(simplify=True))
    assert residual_loops(sdfg) >= 1, 'a colliding scatter must never be parallelized'


@pytest.mark.xfail(strict=True, reason='same min()-subscript crash in canonicalize')
def test_colliding_scatter_is_value_preserving():
    n = 9
    rng = np.random.default_rng(6)
    b = rng.random(n)
    expected = np.zeros(n)
    for i in range(n):
        expected[min(i, n - 1 - i)] = b[i]

    sdfg = cpu_canon(colliding_scatter.to_sdfg(simplify=True))
    got = np.zeros(n)
    sdfg(A=got, B=b.copy(), N=n)
    assert np.allclose(got, expected), 'last write must win, in iteration order'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
