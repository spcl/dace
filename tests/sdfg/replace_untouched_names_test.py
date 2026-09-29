# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that symbol replacement leaves everything it does not rename as it was. """

import dace
from dace import subsets

N = dace.symbol('N', dtype=dace.int64, nonnegative=True)
M = dace.symbol('M', dtype=dace.int64)
T = dace.symbol('T', dtype=dace.int64)


def tiled_map_sdfg() -> tuple[dace.SDFG, dace.nodes.MapEntry]:
    sdfg = dace.SDFG('tiled_map')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('T', dace.int64)
    sdfg.add_array('A', [N], dace.float64)
    state = sdfg.add_state('s0')
    map_entry, map_exit = state.add_map('m', dict(k='0:N'))
    map_entry.map.range = subsets.Range([(0, N - 1, 1, T)])
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[k]'))
    state.add_memlet_path(tasklet, map_exit, state.add_write('A'), src_conn='b', memlet=dace.Memlet('A[k]'))
    return sdfg, map_entry


def nested_mapping_sdfg() -> tuple[dace.SDFG, dace.nodes.NestedSDFG]:
    inner = dace.SDFG('inner')
    inner.add_symbol('P', dace.int64)
    inner.add_symbol('Q', dace.int64)
    inner.add_state('s')
    sdfg = dace.SDFG('outer')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    nsdfg = sdfg.add_state('s').add_nested_sdfg(inner, {}, {}, {'P': N + M, 'Q': N})
    return sdfg, nsdfg


def test_map_keeps_its_tile_sizes_when_an_unrelated_symbol_is_replaced():
    sdfg, map_entry = tiled_map_sdfg()

    sdfg.replace_dict({'M': 'K'})

    assert map_entry.map.range.tile_sizes == [T]


def test_map_range_replacement_renames_bounds_and_tile_sizes_together():
    sdfg, map_entry = tiled_map_sdfg()

    sdfg.replace_dict({'N': 'L', 'T': 'U'})

    assert map_entry.map.range.ranges == [(0, dace.symbol('L') - 1, 1)]
    assert map_entry.map.range.tile_sizes == [dace.symbol('U')]


def test_mapped_symbols_keep_dtype_and_assumptions_when_another_symbol_is_replaced():
    sdfg, nsdfg = nested_mapping_sdfg()
    untouched = nsdfg.symbol_mapping['Q']

    sdfg.replace_dict({'M': 'K'})

    assert nsdfg.symbol_mapping['Q'] is untouched
    touched = {s.name: s for s in nsdfg.symbol_mapping['P'].free_symbols}
    assert list(sorted(touched)) == ['K', 'N']
    assert touched['N'].dtype == dace.int64
    assert touched['N'].is_nonnegative


def test_interstate_edge_without_a_replaced_name_is_left_untouched():
    edge = dace.InterstateEdge('i < N', assignments={'i': 'i+1'})
    parsed_condition = edge.condition.code

    edge.replace_dict({'M': 'K'})

    assert edge.condition.code is parsed_condition
    assert edge.assignments == {'i': 'i+1'}


def test_interstate_edge_with_a_replaced_name_is_rewritten():
    edge = dace.InterstateEdge('i < N', assignments={'i': 'i+1'})
    edge.condition_sympy()

    edge.replace_dict({'N': 'L', 'i': 'j'}, replace_keys=False)

    assert edge.condition.as_string == '(j < L)'
    assert edge.condition_sympy() == dace.symbolic.pystr_to_symbolic('j < L')
    assert edge.assignments == {'i': '(j + 1)'}


if __name__ == '__main__':
    test_map_keeps_its_tile_sizes_when_an_unrelated_symbol_is_replaced()
    test_map_range_replacement_renames_bounds_and_tile_sizes_together()
    test_mapped_symbols_keep_dtype_and_assumptions_when_another_symbol_is_replaced()
    test_interstate_edge_without_a_replaced_name_is_left_untouched()
    test_interstate_edge_with_a_replaced_name_is_rewritten()
