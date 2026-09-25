# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.hoist_loop_range_calls.HoistLoopRangeCalls`.

The pass binds a call-bearing map STEP to a fresh symbol so the emitted increment is a plain
name. Minting that symbol is where dtype identity bites: on this branch ``dtype`` is part of a
symbol's ``_hashable_content``, so a name DECLARED at one width and USED at another is two
symbols. Two of them inside one ``subsets.Range`` stop folding -- ``Min(s, s)`` never collapses
and ``s - s`` never cancels -- and the injectivity tests that decide whether a write can be
tiled then read a non-affine expression. The declaration and the instance placed in the range
must therefore be the same symbol, and the width must be READ from the step's operands, never
assumed.

The map ranges below are built as explicit ``subsets.Range`` objects rather than through the
``'0:N'`` sugar on purpose: ``MapEntry.new_symbols`` types a parameter from ``result_type_of``
over the range's begin and END, and ``Range`` stores the end as ``N - 1``, whose integer literal
infers as int64. Any sugared range therefore reports int64 no matter how its bounds are
declared, which is exactly why a guessed int64 looks right almost everywhere.
"""
import sympy

import dace
from dace import subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.passes.canonicalize.hoist_loop_range_calls import (HoistLoopRangeCalls, RANGE_SYMBOL_PREFIX,
                                                                            contains_call)


def chunked_map_sdfg(extent_dtypes: dict[str, dace.typeclass],
                     step_text: str) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """A one-map SDFG whose stride is ``step_text`` -- the chunked form ``int_ceil(extent, threads)``.

    :param extent_dtypes: Declared dtype per bound symbol; the pass must read the step's width
                          from these and from nowhere else.
    :param step_text: The call-bearing stride, as the analysis passes spell it.
    """
    sdfg = dace.SDFG('chunked')
    for name, dtype in extent_dtypes.items():
        sdfg.add_symbol(name, dtype)
    sdfg.add_array('A', (128, ), dace.float64)
    state = sdfg.add_state()
    begin = symbolic.pystr_to_symbolic('M')
    end = symbolic.pystr_to_symbolic('N')
    step = symbolic.pystr_to_symbolic(step_text)
    entry, exit_node = state.add_map('chunk', {'c': subsets.Range([(begin, end, step)])})
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    write = state.add_write('A')
    exit_node.add_in_connector('IN_A')
    exit_node.add_out_connector('OUT_A')
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'o', exit_node, 'IN_A', dace.Memlet('A[c]'))
    state.add_edge(exit_node, 'OUT_A', write, None, dace.Memlet('A[0:128]'))
    return sdfg, state, entry


def minted_name(sdfg: dace.SDFG) -> str:
    minted = [name for name in sdfg.symbols if name.startswith(RANGE_SYMBOL_PREFIX)]
    assert len(minted) == 1
    return minted[0]


def step_instance(entry: nodes.MapEntry, name: str) -> symbolic.symbol:
    """The symbol object the pass actually placed in the map's range."""
    instances = [sym for _, _, step in entry.map.range for sym in step.free_symbols if sym.name == name]
    assert len(instances) == 1
    return instances[0]


def bound_value(sdfg: dace.SDFG, name: str) -> str:
    """The interstate assignment the pass wrote for ``name``."""
    values = [edge.data.assignments[name] for edge in sdfg.all_interstate_edges() if name in edge.data.assignments]
    assert len(values) == 1
    return values[0]


def test_chunked_stride_over_int32_extents_is_bound_at_the_extents_width():
    sdfg, _, entry = chunked_map_sdfg({'M': dace.int32, 'N': dace.int32}, 'int_ceil(N, 4)')
    assert contains_call(entry.map.range[0][2])  # RED baseline: the call is in the increment

    assert HoistLoopRangeCalls().apply_pass(sdfg, {}) == 1

    name = minted_name(sdfg)
    used = step_instance(entry, name)
    assert sdfg.symbols[name] == dace.int32
    assert used.dtype == dace.int32
    # The property that breaks when the two halves disagree: one name at two widths is two
    # symbols, so neither of these folds.
    declared = symbolic.symbol(name, sdfg.symbols[name])
    assert symbolic.simplify(used - declared) == 0
    assert sympy.Min(used, declared) == used
    # The call left the increment and became the bound value.
    assert not contains_call(entry.map.range[0][2])
    assert 'int_ceil' in bound_value(sdfg, name)


def test_chunked_stride_over_int64_extents_is_bound_at_the_extents_width():
    sdfg, _, entry = chunked_map_sdfg({'M': dace.int64, 'N': dace.int64}, 'int_ceil(N, 4)')

    assert HoistLoopRangeCalls().apply_pass(sdfg, {}) == 1

    name = minted_name(sdfg)
    used = step_instance(entry, name)
    assert sdfg.symbols[name] == dace.int64
    assert used.dtype == dace.int64
    declared = symbolic.symbol(name, sdfg.symbols[name])
    assert symbolic.simplify(used - declared) == 0
    assert sympy.Min(used, declared) == used


def test_a_stride_over_mixed_width_extents_takes_the_wider_one():
    """``int_ceil(N, M)`` with a 32-bit and a 64-bit extent is a 64-bit expression."""
    sdfg, _, entry = chunked_map_sdfg({'M': dace.int64, 'N': dace.int32}, 'int_ceil(N, M)')

    assert HoistLoopRangeCalls().apply_pass(sdfg, {}) == 1

    name = minted_name(sdfg)
    assert sdfg.symbols[name] == dace.int64
    assert step_instance(entry, name).dtype == dace.int64


def test_a_stride_without_a_call_is_left_alone():
    """Only a call must leave the increment; a plain expression is what OpenMP already accepts."""
    sdfg, _, entry = chunked_map_sdfg({'M': dace.int32, 'N': dace.int32}, 'N - M')
    before = str(entry.map.range)

    assert HoistLoopRangeCalls().apply_pass(sdfg, {}) is None

    assert str(entry.map.range) == before
    assert not [name for name in sdfg.symbols if name.startswith(RANGE_SYMBOL_PREFIX)]


def test_a_stride_naming_an_enclosing_map_parameter_stays_in_the_range():
    """An interstate assignment runs at STATE scope, so a stride reading an enclosing map's
    parameter cannot be hoisted out of the scope that defines it."""
    sdfg, state, outer = chunked_map_sdfg({'M': dace.int32, 'N': dace.int32}, 'N - M')
    inner_entry, inner_exit = state.add_map('inner', {'d': subsets.Range([(0, 7, 1)])})
    inner_entry.map.range = subsets.Range([(symbolic.pystr_to_symbolic('0'), symbolic.pystr_to_symbolic('7'),
                                            symbolic.pystr_to_symbolic('int_ceil(c, 2)'))])
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_edge(outer, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'o', inner_exit, None, dace.Memlet())

    assert HoistLoopRangeCalls().apply_pass(sdfg, {}) is None

    assert contains_call(inner_entry.map.range[0][2])
    assert not [name for name in sdfg.symbols if name.startswith(RANGE_SYMBOL_PREFIX)]


if __name__ == '__main__':
    test_chunked_stride_over_int32_extents_is_bound_at_the_extents_width()
    test_chunked_stride_over_int64_extents_is_bound_at_the_extents_width()
    test_a_stride_over_mixed_width_extents_takes_the_wider_one()
    test_a_stride_without_a_call_is_left_alone()
    test_a_stride_naming_an_enclosing_map_parameter_stays_in_the_range()
