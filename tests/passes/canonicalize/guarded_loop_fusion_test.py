# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for guarded fusion -- ``plan_guarded_fusion`` / ``commit_guarded_fusion``.

``FuseConsecutiveLoops`` joins two adjacent-range loops only when their bodies are IDENTICAL,
because then the merge is free: widen one bound, drop the twin. Guarded fusion is the same
adjacent-range rewrite for loops whose bodies DIFFER -- the merged loop picks a body per
iteration from a ``ConditionalBlock`` keyed on the iterator.

It is a plan/commit pair rather than a pass because the rewrite is not unconditionally
desirable: fusing a recurrence with a DOALL sibling makes the whole range a recurrence and loses
the parallel half. Only a caller that has already proved it gains something may commit it, and
today that caller is ``WavefrontSkew``, which needs polybench ``lu``'s two sibling ``j`` loops
seen as the single 2-D space they are before a diagonal exists to find.

The tests below pin both halves: what the planner accepts and refuses (structure), and that a
committed fusion computes the same values (execution).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import (commit_guarded_fusion, plan_guarded_fusion)

N = dace.symbol('N')


@dace.program
def lu_factorization(A: dace.float64[N, N]):
    """polybench ``lu``: the ``j < i`` column update, then the ``j >= i`` row update. Two sibling
    loops over complementary halves of one ``j`` range -- the shape guarded fusion exists for."""
    for i in range(0, N):
        for j in range(0, i):
            for k in range(0, j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            for k in range(0, i):
                A[i, j] -= A[i, k] * A[k, j]


@dace.program
def disjoint_but_not_adjacent(A: dace.float64[N], B: dace.float64[N]):
    """The two ``j`` loops leave a hole at ``j == N // 2``, so their union is NOT one range and
    the concatenation argument does not hold."""
    for i in range(0, N):
        for j in range(0, N // 2):
            B[j] += A[j] * 2.0
        for j in range(N // 2 + 1, N):
            B[j] += A[j] * 3.0


@dace.program
def computation_beside_the_chain(A: dace.float64[N], B: dace.float64[N]):
    """Adjacent ranges, but a store sits between the two loops -- fusing would sweep it into a
    branch it does not belong to, and it would run once per ``j`` instead of once per ``i``."""
    for i in range(0, N):
        for j in range(0, N // 2):
            B[j] += A[j] * 2.0
        B[0] = A[0]
        for j in range(N // 2, N):
            B[j] += A[j] * 3.0


def outer_loop(sdfg: dace.SDFG, var: str) -> LoopRegion:
    """The one ``LoopRegion`` whose iterator is ``var``."""
    found = [
        c for sd in sdfg.all_sdfgs_recursive() for c in sd.all_control_flow_regions()
        if isinstance(c, LoopRegion) and c.loop_variable == var
    ]
    assert len(found) == 1, f'expected exactly one {var} loop, got {len(found)}'
    return found[0]


def lu_reference(a: np.ndarray) -> np.ndarray:
    """``lu_factorization`` in plain NumPy, evaluated sequentially."""
    e = a.copy()
    m = e.shape[0]
    for i in range(m):
        for j in range(i):
            for k in range(j):
                e[i, j] -= e[i, k] * e[k, j]
            e[i, j] /= e[j, j]
        for j in range(i, m):
            for k in range(i):
                e[i, j] -= e[i, k] * e[k, j]
    return e


def diagonally_dominant(m: int) -> np.ndarray:
    """A matrix whose pivots ``A[j, j]`` stay far from zero, so the division in ``lu`` is exact
    enough to compare bit-for-bit against the same operation order."""
    rng = np.random.default_rng(4)
    return rng.random((m, m)) + np.eye(m) * 20 * m


def test_plan_reads_lu_siblings_as_one_range_with_their_ranges_as_guards():
    """The planner's whole contribution is the pairing of each sibling with the constraints that
    hold inside it. For lu those are ``0 <= j <= i - 1`` and ``i <= j <= N - 1`` -- and it is the
    FIRST of those that makes lu analysable: it is what tells the dependence engine that the
    ``A[j, j]`` read happens only where ``j < i``, hence at a strictly earlier iteration."""
    sdfg = lu_factorization.to_sdfg(simplify=True)
    plan = plan_guarded_fusion(outer_loop(sdfg, 'i'))

    assert plan is not None
    assert plan.var == 'j'
    assert len(plan.loops) == 2
    assert symbolic.simplify(plan.lo) == 0
    assert symbolic.simplify(plan.hi - (symbolic.symbol('N') - 1)) == 0

    j, i, n = symbolic.symbol('j'), symbolic.symbol('i'), symbolic.symbol('N')
    assert [symbolic.simplify(g) for g in plan.guards[0]] == [j, symbolic.simplify(i - j - 1)]
    assert [symbolic.simplify(g) for g in plan.guards[1]] == [symbolic.simplify(j - i), symbolic.simplify(n - j - 1)]


@pytest.mark.parametrize('program, why', [
    (disjoint_but_not_adjacent, 'a hole between the two ranges'),
    (computation_beside_the_chain, 'a store beside the chain'),
])
def test_planner_refuses_what_would_not_concatenate(program, why):
    """Both refusals are load-bearing, and for different reasons: a hole means the fused sweep
    would run iterations neither sibling ever ran, and a store beside the chain would move from
    once-per-``i`` to once-per-``j``. Neither is caught downstream -- the skew only ever asks
    whether a schedule is legal, never whether the nest it was handed is the original one."""
    sdfg = program.to_sdfg(simplify=True)
    assert plan_guarded_fusion(outer_loop(sdfg, 'i')) is None, f'must refuse {why}'


def test_planner_does_not_mutate_on_refusal():
    """A refusing planner must leave the graph bit-identical: ``_try_skew`` calls it speculatively
    on every two-level nest it meets, and most of those never fuse."""
    sdfg = computation_beside_the_chain.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert plan_guarded_fusion(outer_loop(sdfg, 'i')) is None
    assert sdfg.to_json() == before


def test_commit_leaves_one_loop_over_the_union_with_a_total_partition():
    """After commit the two siblings are gone, replaced by one loop over ``[0, N)`` whose body is
    a ``ConditionalBlock``. The last branch must be the ``else``: the sub-ranges partition the
    union, so a trailing condition would leave the final iteration able to fall through and
    silently skip the row update."""
    sdfg = lu_factorization.to_sdfg(simplify=True)
    outer = outer_loop(sdfg, 'i')
    merged = commit_guarded_fusion(plan_guarded_fusion(outer), outer)

    assert [b for b in outer.nodes() if isinstance(b, LoopRegion)] == [merged]
    assert merged.loop_variable == 'j'
    assert symbolic.pystr_to_symbolic(merged.init_statement.as_string.split('=', 1)[1]) == 0

    selectors = [b for b in merged.nodes() if isinstance(b, ConditionalBlock)]
    assert len(selectors) == 1
    conditions = [c for c, _ in selectors[0].branches]
    assert len(conditions) == 2
    assert conditions[0] is not None
    assert conditions[-1] is None, 'the final sub-range must be the else, or an iteration can fall through'

    sdfg.validate()


def test_commit_preserves_lu_values():
    """The structural assertions above cannot see a body spliced into the wrong branch, or an
    iterator renamed onto a symbol that already meant something. Running the fused graph against
    a sequential reference can."""
    sdfg = lu_factorization.to_sdfg(simplify=True)
    outer = outer_loop(sdfg, 'i')
    commit_guarded_fusion(plan_guarded_fusion(outer), outer)

    m = 8
    a = diagonally_dominant(m)
    expected = lu_reference(a)
    got = a.copy()
    sdfg(A=got, N=m)

    assert np.allclose(got, expected), 'guarded fusion must not change what lu computes'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
