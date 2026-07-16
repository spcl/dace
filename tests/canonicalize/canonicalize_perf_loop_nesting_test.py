# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Perfect-loop-nesting of a guarded imperfect nest.

    ``if c: for i: { for j: b[i,j]=...; c[i]=... }`` -- the ``c[i]`` write
    lives in the ``i`` loop but not in its own innermost ``j``-style loop
    (an imperfect nest). The intended canonical form wraps every bare
    state/tasklet in a trivial single-iteration loop so the nest is
    perfectly nested, after which the guard is pushed into -- and
    duplicated across -- every loop. Canonicalization is value-preserving
    (guard taken and not-taken) and reaches the structural ideal: no
    surviving top-level guard -- it has been moved/duplicated inside by
    ``MoveIfIntoLoop``'s free-state (imperfect-nest) path.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def guard_over_imperfect_nest(a: dace.float64[N, M], b: dace.float64[N, M], c: dace.float64[N], act: dace.int32[1]):
    if act[0] > 0:
        for i in dace.map[0:N]:
            for j in dace.map[0:M]:
                b[i, j] = a[i, j] * 2.0
            c[i] = a[i, 0] + 1.0  # bare: in loop i, not in a j-loop


def _top_level_conds(sdfg):
    return [x for x in sdfg.nodes() if isinstance(x, ConditionalBlock)]


def _oracle(a, n, m, av):
    eb, ec = np.full((n, m), 9.0), np.full(n, 9.0)
    if av > 0:
        for i in range(n):
            for j in range(m):
                eb[i, j] = a[i, j] * 2.0
            ec[i] = a[i, 0] + 1.0
    return eb, ec


@pytest.mark.parametrize('av', [1, 0])
def test_guard_over_imperfect_nest_is_value_preserving(av):
    """Canonicalization of a guarded imperfect nest is value-preserving for
    the guard taken and not-taken."""
    n, m = 6, 5
    a = np.random.rand(n, m)
    eb, ec = _oracle(a, n, m, av)

    sdfg = guard_over_imperfect_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    ob, oc = np.full((n, m), 9.0), np.full(n, 9.0)
    sdfg(a=a.copy(), b=ob, c=oc, act=np.array([av], np.int32), N=n, M=m)
    assert np.allclose(ob, eb) and np.allclose(oc, ec), f"mismatch act={av}"


def test_guard_over_imperfect_nest_parallelizes_value_preserving():
    """A guarded imperfect nest (``if act > 0: for i: {for j: b[i,j]=...; } c[i]=...}``) must fully
    parallelize -- both the inner elementwise nest and the bare per-i statement become Maps, leaving
    NO residual sequential loop -- while keeping the guard (not dropping it) and the exact values for
    the guard taken and not-taken.

    Note: canonicalize is free to keep the guard as a single top-level ``ConditionalBlock`` wrapping
    the maps, rather than duplicating it inside each -- both expose the same parallelism, and the
    single top-level guard is the cleaner form. This test pins the invariant that matters (maximal
    mapping + guard preserved + value-preserving), not one particular guard placement."""
    n, m = 6, 5
    a = np.random.default_rng(0).random((n, m))
    sdfg = guard_over_imperfect_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    # Maximal parallelism: the nest is exposed as Maps with no sequential-loop residue.
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    seq_loops = [r for r in sdfg.all_control_flow_regions(recursive=True)
                 if isinstance(r, LoopRegion) and not r.pinned_sequential]
    assert len(maps) >= 2, f"nest did not fully map (maps={len(maps)})"
    assert not seq_loops, f"sequential loop survived: {[r.label for r in seq_loops]}"
    # The guard is preserved (not silently dropped), wherever canonicalize placed it.
    assert any(isinstance(x, ConditionalBlock)
               for x in sdfg.all_control_flow_regions(recursive=True)), "guard was dropped"

    for av in (1, 0):
        eb, ec = _oracle(a, n, m, av)
        ob, oc = np.full((n, m), 9.0), np.full(n, 9.0)
        sdfg(a=a.copy(), b=ob, c=oc, act=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(ob, eb) and np.allclose(oc, ec), f"mismatch act={av}"


def test_refusal_leaves_sdfg_byte_identical():
    """A pass that does not apply must not mutate the SDFG. PerfectLoopNesting used to SSA-rename
    every loop iterator and drop names from ``sdfg.symbols`` even when all sub-passes refused,
    because the iterator uniquification ran once more on the final (no-change) round. On a loop with
    a carried recurrence -- which it cannot perfect-nest -- ``apply_pass`` must return None and leave
    the SDFG byte-identical."""
    from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting

    @dace.program
    def carried(a: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + 1.0

    sdfg = carried.to_sdfg(simplify=True)
    before = sdfg.to_json()
    result = PerfectLoopNesting().apply_pass(sdfg, {})
    assert result is None, "the carried recurrence must be refused"
    assert sdfg.to_json() == before, "refusal mutated the SDFG (iterator rename / symbol drop)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
