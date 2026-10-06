# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :mod:`~dace.transformation.passes.parallelization_prep` range recovery."""
import pytest

from dace import symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling
from tests.sdfg.cfg_list_checks import assert_cfg_list_as_after_a_reset, loop_over_nested_sdfg, record_tree_resets


@pytest.mark.parametrize('multi_state', [False, True])
@pytest.mark.parametrize('reuse_loop', [False, True])
def test_a_split_keeps_the_cfg_list_in_place(monkeypatch, multi_state, reuse_loop):
    """Every split rebuilt the CFG list of the whole tree after wiring its segments; the graph operations keep it."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    resets = record_tree_resets(monkeypatch, lambda root: root is sdfg)
    assert BestEffortLoopPeeling()._split_loop_at(sdfg, loop, symbolic.pystr_to_symbolic('2'), reuse_loop=reuse_loop)
    assert resets == []
    assert [r.label for r in sdfg.nodes() if isinstance(r, LoopRegion)] == ['outer_p0', 'outer_p1', 'outer_p2']
    assert_cfg_list_as_after_a_reset(sdfg)
    sdfg.validate()


@pytest.mark.parametrize('multi_state', [False, True])
def test_an_isolated_loop_arrives_with_its_cfg_list_in_place(monkeypatch, multi_state):
    """The probe copy of a loop is re-homed by ``add_node``; no reset of its tree is needed before validation."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    resets = record_tree_resets(monkeypatch, lambda root: root.name.endswith('_peelprobe'))
    mini, mini_loop = BestEffortLoopPeeling()._isolate_loop(loop, sdfg)
    assert mini is not None and mini_loop.parent_graph is mini
    assert resets == []
    assert [type(r).__name__ for r in mini.cfg_list] == [type(r).__name__ for r in sdfg.cfg_list]
    assert_cfg_list_as_after_a_reset(mini)


@pytest.mark.parametrize('multi_state', [False, True])
def test_a_specialized_loop_keeps_the_cfg_list_in_place(monkeypatch, multi_state):
    """Swapping a loop for ``if cond: split else: loop`` needs no reset, before or after the callback splits."""
    sdfg = loop_over_nested_sdfg(multi_state)
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    peeling = BestEffortLoopPeeling()
    seen = []

    def parallelize(par_loop, par_region, owner):
        assert_cfg_list_as_after_a_reset(owner)
        seen.append(peeling._split_loop_at(owner, par_loop, symbolic.pystr_to_symbolic('2')))

    resets = record_tree_resets(monkeypatch, lambda root: root is sdfg)
    conditional = specialize_loop_under_condition(loop, 'N > 4', parallelize, sdfg)
    assert resets == [] and seen == [True]
    assert isinstance(conditional, ConditionalBlock) and conditional.parent_graph is sdfg
    loops = [r.label for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
    assert loops == ['outer_p0', 'outer_p1', 'outer_p2', 'outer'], loops
    assert_cfg_list_as_after_a_reset(sdfg)
    sdfg.validate()
