# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MapEntry.used_symbols_within_scope`` reports the symbols the map's own ranges name."""
import dace


def _map_over(range_str: str) -> dace.SDFG:
    """One map with the given range writing a constant, so only the range names symbols."""
    sdfg = dace.SDFG('map_range_symbols')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_symbol('M', dace.int32)
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state()

    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    entry, exit_node = state.add_map('m', {'i': range_str})
    exit_node.add_in_connector('IN_A')
    exit_node.add_out_connector('OUT_A')
    state.add_edge(entry, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, 'o', exit_node, 'IN_A', dace.Memlet('A[0]'))
    state.add_edge(exit_node, 'OUT_A', state.add_write('A'), None, dace.Memlet('A[0]'))
    return sdfg


def test_map_range_symbol_is_used():
    sdfg = _map_over('0:N')
    entry = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.MapEntry))
    assert 'N' in entry.used_symbols_within_scope(sdfg.states()[0], all_symbols=True)


def test_map_range_bounds_and_step_symbols_are_used():
    sdfg = _map_over('M:N:M')
    entry = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.MapEntry))
    assert {'M', 'N'} <= entry.used_symbols_within_scope(sdfg.states()[0], all_symbols=True)


def test_map_parameter_is_not_reported():
    """The parameter is defined by the map, not used by it."""
    sdfg = _map_over('0:N')
    entry = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.MapEntry))
    assert 'i' not in entry.used_symbols_within_scope(sdfg.states()[0], all_symbols=True)


if __name__ == '__main__':
    test_map_range_symbol_is_used()
    test_map_range_bounds_and_step_symbols_are_used()
    test_map_parameter_is_not_reported()
