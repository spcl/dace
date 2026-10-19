# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :mod:`~dace.transformation.passes.parallelization_prep` range recovery."""

import pytest

import dace
from dace import symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling
from tests.sdfg.cfg_list_checks import assert_cfg_list_as_after_a_reset, loop_over_nested_sdfg, record_tree_resets


@pytest.mark.parametrize("multi_state", [False, True])
@pytest.mark.parametrize("reuse_loop", [False, True])
def test_a_split_keeps_the_cfg_list_in_place(monkeypatch, multi_state, reuse_loop):
    """Every split rebuilt the CFG list of the whole tree after wiring its segments; the graph operations keep it."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    resets = record_tree_resets(monkeypatch, lambda root: root is sdfg)
    assert BestEffortLoopPeeling()._split_loop_at(sdfg, loop, symbolic.pystr_to_symbolic("2"), reuse_loop=reuse_loop)
    assert resets == []
    assert [r.label for r in sdfg.nodes() if isinstance(r, LoopRegion)] == ["outer_p0", "outer_p1", "outer_p2"]
    assert_cfg_list_as_after_a_reset(sdfg)
    sdfg.validate()


@pytest.mark.parametrize("multi_state", [False, True])
def test_an_isolated_loop_arrives_with_its_cfg_list_in_place(monkeypatch, multi_state):
    """The probe copy of a loop is re-homed by ``add_node``; no reset of its tree is needed before validation."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    resets = record_tree_resets(monkeypatch, lambda root: root.name.endswith("_peelprobe"))
    mini, mini_loop = BestEffortLoopPeeling()._isolate_loop(loop, sdfg)
    assert mini is not None and mini_loop.parent_graph is mini
    assert resets == []
    assert [type(r).__name__ for r in mini.cfg_list] == [type(r).__name__ for r in sdfg.cfg_list]
    assert_cfg_list_as_after_a_reset(mini)


@pytest.mark.parametrize("multi_state", [False, True])
def test_a_specialized_loop_keeps_the_cfg_list_in_place(monkeypatch, multi_state):
    """Swapping a loop for ``if cond: split else: loop`` needs no reset, before or after the callback splits."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    peeling = BestEffortLoopPeeling()
    seen = []

    def parallelize(par_loop, par_region, owner):
        assert_cfg_list_as_after_a_reset(owner)
        seen.append(peeling._split_loop_at(owner, par_loop, symbolic.pystr_to_symbolic("2")))

    resets = record_tree_resets(monkeypatch, lambda root: root is sdfg)
    conditional = specialize_loop_under_condition(loop, "N > 4", parallelize, sdfg)
    assert resets == [] and [len(segments) for segments in seen] == [3]
    assert isinstance(conditional, ConditionalBlock) and conditional.parent_graph is sdfg
    loops = [r.label for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert loops == ["outer_p0", "outer_p1", "outer_p2", "outer"], loops
    assert_cfg_list_as_after_a_reset(sdfg)
    sdfg.validate()


N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def guarded_recurrence(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in range(1, N):
        if i == 1:
            b[i, 0] = 0.0
        for j in range(M):
            a[i, j] = a[i - 1, j] + b[i, j]


def loop_labels(sdfg: dace.SDFG) -> list[str]:
    return sorted(r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def test_a_split_that_only_copies_mappable_loops_is_not_taken():
    """Splitting the carried ``i`` loop at its guard ``i == 1`` copies the mappable ``j`` loop into each segment
    and maps nothing new: the copies are one loop, so the count does not rise and the nest is not duplicated."""
    sdfg = guarded_recurrence.to_sdfg(simplify=True)
    before = loop_labels(sdfg)
    assert BestEffortLoopPeeling().apply_pass(sdfg, {}) is None
    assert loop_labels(sdfg) == before


if __name__ == "__main__":
    import _pytest.monkeypatch

    for multi_state in (False, True):
        for reuse_loop in (False, True):
            with _pytest.monkeypatch.MonkeyPatch.context() as mp:
                test_a_split_keeps_the_cfg_list_in_place(mp, multi_state, reuse_loop)
        with _pytest.monkeypatch.MonkeyPatch.context() as mp:
            test_an_isolated_loop_arrives_with_its_cfg_list_in_place(mp, multi_state)
        with _pytest.monkeypatch.MonkeyPatch.context() as mp:
            test_a_specialized_loop_keeps_the_cfg_list_in_place(mp, multi_state)
    test_a_split_that_only_copies_mappable_loops_is_not_taken()
