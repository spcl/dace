# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :mod:`~dace.transformation.passes.parallelization_prep` range recovery."""
import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization_prep import BestEffortLoopPeeling


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
