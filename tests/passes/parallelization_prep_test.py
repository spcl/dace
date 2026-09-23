# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :mod:`~dace.transformation.passes.parallelization_prep` range recovery."""
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling
from tests.sdfg.cfg_list_checks import assert_cfg_list_as_after_a_reset, loop_over_nested_sdfg, record_tree_resets


def test_a_loop_range_keys_its_iterator_at_the_width_the_loop_infers():
    """``for i = 0; i <= N32``: the literal origin makes the loop type ``i`` as int64.

    ``i`` is registered in no symbol table, so a lookup there leaves the parsed default width and the
    range key is a second ``i`` that a body subset's ``i`` never cancels against.
    """
    sdfg = dace.SDFG('own_range')
    sdfg.add_symbol('N32', dace.int32)
    loop = LoopRegion('L', 'i <= N32', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state('body', is_start_block=True)

    (key, ) = BestEffortLoopPeeling()._loop_own_ranges(loop)

    assert 'i' not in sdfg.symbols
    assert key.dtype == dace.int64
    assert symbolic.simplify(key - symbolic.symbol('i', dace.int64)) == 0


def test_a_nested_loop_range_keys_its_iterator_at_the_width_its_enclosing_iterator_gives_it():
    """``for j = 0 ..: for i = j; i <= N32; i += S32``: only the int64 enclosing ``j`` widens ``i``."""
    sdfg = dace.SDFG('nested_range')
    for name in ('N32', 'S32'):
        sdfg.add_symbol(name, dace.int32)
    outer = LoopRegion('outer', 'j < 4', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion('inner', 'i <= N32', 'i', 'i = j', 'i = i + S32')
    outer.add_node(inner, is_start_block=True)
    inner.add_state('body', is_start_block=True)

    (key, ) = BestEffortLoopPeeling()._loop_own_ranges(inner)

    assert 'i' not in sdfg.symbols and 'j' not in sdfg.symbols
    assert key.dtype == dace.int64


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
