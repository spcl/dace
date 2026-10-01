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


if __name__ == '__main__':
    test_symbol_mapped_to_a_wider_map_parameter_is_widened()
    test_value_beyond_32_bits_reaches_the_nested_sdfg()
    test_declared_wider_symbol_is_kept()
