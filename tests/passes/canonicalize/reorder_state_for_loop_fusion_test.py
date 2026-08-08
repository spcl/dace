# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ReorderStateForLoopFusion`` must only sink a between-loops state past the second loop when the reorder is
dependence-legal (no RAW/WAR/WAW against the second loop, no side effect) AND it actually unlocks
``FuseLoops`` on the resulting adjacency. Either condition failing alone must leave the SDFG untouched.
"""
import copy
from typing import Tuple
from unittest import mock

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import BreakBlock, LoopRegion, SDFGState
from dace.transformation.passes.analysis.analysis import AccessSets
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion

N = 8


def _loop(label: str, end: int) -> LoopRegion:
    return LoopRegion(label,
                      condition_expr=f"i < {end}",
                      loop_var="i",
                      initialize_expr="i = 1",
                      update_expr="i = i + 1")


def _build(conflict: str = 'none',
           loop2_end: int = N,
           state_side_effects: bool = False) -> Tuple[dace.SDFG, LoopRegion, SDFGState, LoopRegion]:
    """``loop1{T[i]=T[i-1]+P[i]}`` ; ``state{...}`` ; ``loop2{U[i]=U[i-1]+Q[i]}``.

    The two loop bodies touch completely disjoint arrays (``P``/``T`` vs ``Q``/``U``), so once
    directly adjacent they are trivially fusable (no cross-body dependence at all). ``conflict``
    controls what the middle state touches, to probe RAW / WAR / WAW against ``loop2`` in isolation;
    ``'none'`` makes the state touch only ``C``/``M``, which neither loop goes near.
    """
    sdfg = dace.SDFG(f"reorder_state_for_loop_fusion_{conflict}_{loop2_end}_{state_side_effects}")
    for name in ("P", "Q", "T", "U"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_array("M", [1], dace.float64, transient=True)
    sdfg.add_symbol("i", dace.int64)

    first = _loop("loop1", N)
    body1 = first.add_state("body1", is_start_block=True)
    acc1 = body1.add_tasklet("acc1", {"prev": None, "p": None}, {"o"}, "o = prev + p")
    body1.add_edge(body1.add_access("T"), None, acc1, "prev", dace.Memlet("T[i - 1]"))
    body1.add_edge(body1.add_access("P"), None, acc1, "p", dace.Memlet("P[i]"))
    body1.add_edge(acc1, "o", body1.add_access("T"), None, dace.Memlet("T[i]"))

    state = SDFGState("state")
    if conflict == 'raw':  # state writes Q, which loop2 reads -> illegal
        touch = state.add_tasklet("touch", {}, {"o"}, "o = 5.0", side_effects=state_side_effects or None)
        state.add_edge(touch, "o", state.add_access("Q"), None, dace.Memlet("Q[0]"))
    elif conflict == 'war':  # state reads U, which loop2 writes -> illegal
        touch = state.add_tasklet("touch", {"v"}, {"o"}, "o = v", side_effects=state_side_effects or None)
        state.add_edge(state.add_access("U"), None, touch, "v", dace.Memlet("U[0]"))
        state.add_edge(touch, "o", state.add_access("M"), None, dace.Memlet("M[0]"))
    elif conflict == 'waw':  # state writes U, which loop2 also writes -> illegal
        touch = state.add_tasklet("touch", {}, {"o"}, "o = 9.0", side_effects=state_side_effects or None)
        state.add_edge(touch, "o", state.add_access("U"), None, dace.Memlet("U[0]"))
    else:
        touch = state.add_tasklet("touch", {"v"}, {"o"}, "o = v + 1.0", side_effects=state_side_effects or None)
        state.add_edge(state.add_access("C"), None, touch, "v", dace.Memlet("C[0]"))
        state.add_edge(touch, "o", state.add_access("M"), None, dace.Memlet("M[0]"))

    second = _loop("loop2", loop2_end)
    body2 = second.add_state("body2", is_start_block=True)
    acc2 = body2.add_tasklet("acc2", {"prev": None, "q": None}, {"o"}, "o = prev + q")
    body2.add_edge(body2.add_access("U"), None, acc2, "prev", dace.Memlet("U[i - 1]"))
    body2.add_edge(body2.add_access("Q"), None, acc2, "q", dace.Memlet("Q[i]"))
    body2.add_edge(acc2, "o", body2.add_access("U"), None, dace.Memlet("U[i]"))

    sdfg.add_node(first, is_start_block=True)
    sdfg.add_node(state)
    sdfg.add_node(second)
    sdfg.add_edge(first, state, InterstateEdge())
    sdfg.add_edge(state, second, InterstateEdge())
    sdfg.validate()
    return sdfg, first, state, second


def _run(sdfg: dace.SDFG) -> dict:
    args = {
        "P": np.arange(1, N + 1, dtype=np.float64),
        "Q": np.arange(1, N + 1, dtype=np.float64),
        "T": np.zeros(N, dtype=np.float64),
        "U": np.zeros(N, dtype=np.float64),
        "C": np.array([2.0], dtype=np.float64),
    }
    sdfg(**args)
    return args


def _nloops(sdfg: dace.SDFG) -> int:
    return sum(1 for b in sdfg.nodes() if isinstance(b, LoopRegion))


def test_legal_reorder_that_unlocks_fusion_is_sunk_and_correct():
    """Disjoint state, same-range disjoint-array loops: the reorder is legal and, once first/second
    are adjacent, trivially fusable (no cross-body dependence at all)."""
    sdfg, first, state, second = _build(conflict='none')
    oracle = _run(copy.deepcopy(sdfg))
    assert state in sdfg.nodes(), "state starts as a sibling block of first/second"

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) == 1
    assert state in sdfg.nodes(), "state must still be a SIBLING block, not nested into second's body"
    assert state not in second.nodes(), "state must never be absorbed into second's own node set"
    assert len(sdfg.edges_between(first, state)) == 0
    assert len(sdfg.edges_between(state, second)) == 0
    assert len(sdfg.edges_between(first, second)) == 1
    assert len(sdfg.edges_between(second, state)) == 1
    sdfg.validate()

    got = _run(copy.deepcopy(sdfg))
    assert np.array_equal(got["T"], oracle["T"]), f"sinking changed T: {got['T']} != {oracle['T']}"
    assert np.array_equal(got["U"], oracle["U"]), f"sinking changed U: {got['U']} != {oracle['U']}"

    # Adjacency was the only thing blocking fusion.
    assert LoopFusion().apply_pass(sdfg, {}) == 1
    assert _nloops(sdfg) == 1
    fused = _run(sdfg)
    assert np.array_equal(fused["T"], oracle["T"]), "fusing after the sink changed T"
    assert np.array_equal(fused["U"], oracle["U"]), "fusing after the sink changed U"


def test_a_second_loop_with_side_effects_is_not_sunk_past():
    """A side-effecting node in loop2 is invisible to the name-level RAW/WAR/WAW checks -- by
    definition its effects are not the ones its memlets describe -- so it must refuse the reorder from
    ITS side too, not just when the side effect sits in the state being moved."""
    sdfg, first, state, second = _build(conflict='none')
    body2 = next(b for b in second.nodes() if isinstance(b, SDFGState))
    acc2 = next(n for n in body2.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet))
    assert ReorderStateForLoopFusion.has_side_effecting_code(second, sdfg) is False, "fixture must start effect-free"
    acc2.side_effects = True
    assert ReorderStateForLoopFusion.has_side_effecting_code(second, sdfg) is True

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


@pytest.mark.parametrize("entry", ["state", "second"])
def test_a_candidate_holding_the_entry_block_is_not_sunk(entry: str):
    """The reorder rewires the edges around both ``state`` and ``second``; if either is where execution
    begins, that rewiring moves the entry itself and the program starts somewhere else.

    Built as a cycle ``first -> state -> second -> first`` because that is the only shape in which the
    entry can be anything other than ``first``: with a plain chain ``first`` is the unique source node
    and ``start_block`` reports it no matter what was declared.
    """
    sdfg = dace.SDFG(f"reorder_state_for_loop_fusion_entry_{entry}")
    for name in ("P", "Q", "T", "U"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_symbol("i", dace.int64)
    first, second = _loop("loop1", N), _loop("loop2", N)
    for loop, (dst, src) in ((first, ("T", "P")), (second, ("U", "Q"))):
        body = loop.add_state(f"body_{dst}", is_start_block=True)
        t = body.add_tasklet(f"acc_{dst}", {"prev": None, "x": None}, {"o"}, "o = prev + x")
        body.add_edge(body.add_access(dst), None, t, "prev", dace.Memlet(f"{dst}[i - 1]"))
        body.add_edge(body.add_access(src), None, t, "x", dace.Memlet(f"{src}[i]"))
        body.add_edge(t, "o", body.add_access(dst), None, dace.Memlet(f"{dst}[i]"))
    state = SDFGState("state")

    sdfg.add_node(first)
    sdfg.add_node(state, is_start_block=(entry == "state"))
    sdfg.add_node(second, is_start_block=(entry == "second"))
    sdfg.add_edge(first, state, InterstateEdge())
    sdfg.add_edge(state, second, InterstateEdge())
    sdfg.add_edge(second, first, InterstateEdge())
    assert sdfg.start_block is (state if entry == "state" else second)

    assert ReorderStateForLoopFusion.candidates(sdfg) == []
    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


@pytest.mark.parametrize("conflict", ["raw", "war", "waw"])
def test_a_conflicting_state_is_not_sunk(conflict: str):
    """RAW (state writes what loop2 reads), WAR (state reads what loop2 writes), and WAW (both write
    the same container) must each refuse the reorder outright."""
    sdfg, first, state, second = _build(conflict=conflict)

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_a_state_with_side_effects_is_not_sunk():
    """Even with disjoint data, a side-effecting state must not be reordered relative to loop2."""
    sdfg, first, state, second = _build(conflict='none', state_side_effects=True)

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_legal_reorder_but_loops_not_fusable_is_not_sunk():
    """The state/loop2 conflict checks all pass (disjoint data), but loop2's range differs from
    loop1's, so ``FuseLoops`` would refuse the resulting adjacency -- condition 2 must gate the sink."""
    sdfg, first, state, second = _build(conflict='none', loop2_end=N - 2)

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_a_view_touched_by_the_state_is_not_sunk():
    """Item 4 (aliasing): the state writes through a View onto an array ("V") neither loop otherwise
    reaches. A name-only check would call this obviously disjoint from loop2's ``Q``/``U``; this pass
    refuses anyway, because it never attempts to resolve what a View/Reference actually aliases -- two
    different names can be the same memory, and undecided is not legal.
    """
    sdfg, first, state, second = _build(conflict='none')
    sdfg.add_array("V", [N], dace.float64)
    sdfg.add_view("Valias", [N], dace.float64)
    touch = state.add_tasklet("touch_view", {}, {"o"}, "o = 5.0")
    valias = state.add_access("Valias")
    vreal = state.add_access("V")
    state.add_edge(touch, "o", valias, None, dace.Memlet("Valias[0]"))
    state.add_nedge(valias, vreal, dace.Memlet(f"V[0:{N}]"))
    sdfg.validate()

    # Isolate the predicate itself: it must flag the touch as aliasing-risky on its own terms, not
    # merely coincide with some other guard also refusing this fixture.
    reads_s, writes_s = AccessSets().apply_pass(sdfg, {})[state]
    assert ReorderStateForLoopFusion.touches_aliasing_data(sdfg, reads_s | writes_s) is True

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_a_loop2_with_an_early_exit_block_is_not_sunk():
    """Item 7 (early exit): loop2's body contains a BreakBlock. Were the state moved to run only after
    loop2, any execution taking the escape path would never reach the state at all -- refused on the
    block's mere presence, without trying to prove whether it is reachable.
    """
    sdfg, first, state, second = _build(conflict='none')
    brk = BreakBlock("brk")
    second.add_node(brk)
    body2 = next(b for b in second.nodes() if isinstance(b, SDFGState))
    second.add_edge(body2, brk, InterstateEdge(condition="i == 100"))
    sdfg.validate()

    # Isolate the predicate: FuseLoops' own body-shape recognizer also happens to reject a body
    # containing a bare BreakBlock, so an end-to-end None here would not by itself prove THIS guard
    # is the one doing the work -- assert the guard directly fires on its own terms.
    assert ReorderStateForLoopFusion.escapes(second) is True

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_an_edge_assignment_the_second_loop_consumes_is_not_sunk():
    """Item 8 (symbols) / item 9 (conditions): the state->loop2 edge assigns symbol ``K``, which
    loop2's own condition reads. Dropping that edge (exactly what the reorder would do) leaves loop2's
    bound stale/undefined -- refused by the same "no assignments, no condition on either edge" guard
    that also rules out a non-trivial condition on either side of the state.
    """
    sdfg, first, state, second = _build(conflict='none')
    sdfg.add_symbol("K", dace.int64)
    old_edge = sdfg.edges_between(state, second)[0]
    sdfg.remove_edge(old_edge)
    sdfg.add_edge(state, second, InterstateEdge(assignments={"K": "5"}))
    second.loop_condition = dace.properties.CodeBlock("i < K")
    sdfg.validate()

    # Isolate the guard: candidates() must reject the match itself (not merely fail some later,
    # unrelated check) -- the identical fixture without the assignment yields exactly one candidate.
    assert ReorderStateForLoopFusion.candidates(sdfg) == []

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1
    assert len(sdfg.edges_between(first, second)) == 0


def test_another_path_into_second_is_not_sunk():
    """Item 10 (topology): loop1 also reaches loop2 directly through an ``extra`` state, bypassing
    ``state`` entirely -- so ``state`` is no longer the ONLY path from loop1 to loop2, and loop2 has
    two predecessors. Reordering would run ``state`` after loop2 regardless of which path was taken,
    including the one that never used to run it at all -- refused.
    """
    sdfg, first, state, second = _build(conflict='none')
    extra = sdfg.add_state("extra")
    sdfg.add_edge(first, extra, InterstateEdge(condition="i == 999"))
    sdfg.add_edge(extra, second, InterstateEdge())
    sdfg.validate()

    # Isolate the guard: candidates() must reject the match itself, because second now has two
    # predecessors -- the identical fixture without ``extra`` yields exactly one candidate.
    assert ReorderStateForLoopFusion.candidates(sdfg) == []
    assert len(sdfg.in_edges(second)) == 2

    assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) is None
    assert len(sdfg.edges_between(first, state)) == 1
    assert len(sdfg.edges_between(state, second)) == 1


def test_non_vacuous_neutered_pass_would_fail_the_positive_assertions():
    """Prove the positive test is not vacuously true: with ``ReorderStateForLoopFusion`` neutered (stubbed to always
    return ``None``, exactly a `return None` no-op), the positive test's own load-bearing assertions --
    the sink firing, and ``LoopFusion`` then succeeding -- both fail."""
    sdfg, first, state, second = _build(conflict='none')

    with mock.patch.object(ReorderStateForLoopFusion, 'apply_pass', return_value=None):
        with pytest.raises(AssertionError, match=r'assert None == 1'):
            assert ReorderStateForLoopFusion().apply_pass(sdfg, {}) == 1

    # The neutered pass touched nothing, so loop1/loop2 are still not adjacent -> fusion must refuse too.
    with pytest.raises(AssertionError, match=r'assert None == 1'):
        assert LoopFusion().apply_pass(sdfg, {}) == 1
    assert _nloops(sdfg) == 2, "loops must not have fused without the sink"


if __name__ == "__main__":
    pytest.main([__file__])
