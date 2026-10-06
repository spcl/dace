# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.sdfg import infer_types

N = dace.symbol('N', dace.int64)

OFFSET = 2**33


def _map_over_nested_sdfg(mapped_value: str = 'i', start: int = 0) -> dace.SDFG:
    """
    A map over ``start:start + N`` whose body is a nested SDFG writing its undeclared symbol ``n``, which is mapped to
    ``mapped_value``, to ``A[i - start]``.
    """
    inner = dace.SDFG('inner')
    inner.add_array('a', [1], dace.int64)
    inner_state = inner.add_state()
    tasklet = inner_state.add_tasklet('write_n', {}, {'o'}, 'o = n')
    inner_state.add_edge(tasklet, 'o', inner_state.add_write('a'), None, dace.Memlet('a[0]'))

    sdfg = dace.SDFG('nested_sdfg_symbol_widening')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.int64)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map('m', {'i': f'{start}:{start} + N'})
    nsdfg = state.add_nested_sdfg(inner, {}, {'a'}, symbol_mapping={'n': mapped_value})
    state.add_nedge(map_entry, nsdfg, dace.Memlet())
    state.add_memlet_path(nsdfg, map_exit, state.add_write('A'), src_conn='a', memlet=dace.Memlet(f'A[i - {start}]'))
    return sdfg


def nested_symbol_program(name: str,
                          outer_type: dace.typeclass,
                          inner_type: dace.typeclass,
                          body: str,
                          extent: str = '8') -> dace.SDFG:
    """ A nested SDFG declaring ``n`` as ``inner_type`` and writing ``body`` to ``a[k]`` for ``k`` in ``0:extent``;
        ``n`` is mapped to the outer symbol ``s`` of ``outer_type``. """
    inner = dace.SDFG(f'{name}_inner')
    inner.add_symbol('n', inner_type)
    inner.add_array('a', [8], dace.float64)
    inner_state = inner.add_state()
    map_entry, map_exit = inner_state.add_map('m', {'k': f'0:{extent}'})
    tasklet = inner_state.add_tasklet('w', {}, {'o'}, body)
    inner_state.add_nedge(map_entry, tasklet, dace.Memlet())
    inner_state.add_memlet_path(tasklet, map_exit, inner_state.add_write('a'), src_conn='o', memlet=dace.Memlet('a[k]'))

    sdfg = dace.SDFG(name)
    sdfg.add_symbol('s', outer_type)
    sdfg.add_array('A', [8], dace.float64)
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'a'}, symbol_mapping={'n': 's'})
    state.add_edge(nsdfg, 'a', state.add_write('A'), None, dace.Memlet('A[0:8]'))
    return sdfg


def inferred_declaration(sdfg: dace.SDFG) -> dace.typeclass:
    infer_types.infer_connector_types(sdfg)
    return next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)).symbols['n']


def test_symbol_mapped_to_a_wider_map_parameter_is_widened():
    sdfg = _map_over_nested_sdfg()
    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    # Declared before the node was placed in the map, so the map parameter was not visible yet
    assert inner.symbols['n'] == dace.int32
    infer_types.infer_connector_types(sdfg)
    assert inner.symbols['n'] == dace.int64


def test_value_beyond_32_bits_reaches_the_nested_sdfg():
    sdfg = _map_over_nested_sdfg(start=OFFSET)
    A = np.zeros(4, dtype=np.int64)
    sdfg(A=A, N=4)
    assert np.array_equal(A, np.arange(4, dtype=np.int64) + OFFSET)


def test_declared_wider_symbol_is_kept():
    sdfg = _map_over_nested_sdfg()
    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    inner.symbols['n'] = dace.int64
    sdfg.symbols['N'] = dace.int32
    infer_types.infer_connector_types(sdfg)
    assert inner.symbols['n'] == dace.int64


def test_explicitly_narrow_declaration_is_widened_within_its_kind():
    sdfg = nested_symbol_program('explicit_int8_widening', dace.int64, dace.int8, 'o = n')
    assert inferred_declaration(sdfg) == dace.int64
    A = np.zeros(8)
    sdfg(A=A, s=300)
    assert np.array_equal(A, np.full(8, 300.0)), A


def test_integer_symbol_mapped_to_a_float_stays_integer():
    """A ``double`` map bound does not compile; the value is truncated into the integer declaration instead."""
    sdfg = nested_symbol_program('int_symbol_float_value', dace.float64, dace.int32, 'o = k', extent='n')
    assert inferred_declaration(sdfg) == dace.int32
    A = np.zeros(8)
    sdfg(A=A, s=5.0)
    assert np.array_equal(A, np.where(np.arange(8) < 5, np.arange(8), 0.0)), A


def test_unsigned_value_keeps_a_signed_declaration_signed():
    """``n - 10`` with n = 5 is -5, not 2**64 - 5."""
    sdfg = nested_symbol_program('signed_symbol_unsigned_value', dace.uint64, dace.int32, 'o = n - 10')
    assert inferred_declaration(sdfg) == dace.int32
    A = np.zeros(8)
    sdfg(A=A, s=5)
    assert np.array_equal(A, np.full(8, -5.0)), A


if __name__ == '__main__':
    test_symbol_mapped_to_a_wider_map_parameter_is_widened()
    test_value_beyond_32_bits_reaches_the_nested_sdfg()
    test_declared_wider_symbol_is_kept()
    test_explicitly_narrow_declaration_is_widened_within_its_kind()
    test_integer_symbol_mapped_to_a_float_stays_integer()
    test_unsigned_value_keeps_a_signed_declaration_signed()
