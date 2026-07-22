# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``new_symbol_names`` must name exactly what ``new_symbols`` defines, without inferring types."""
import builtins

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes

N = dace.symbol('N')
M = dace.symbol('M')


def assert_names_match(sdfg: dace.SDFG) -> int:
    """Every scope entry must agree with the keys ``new_symbols`` would produce."""
    entries = 0
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.nodes():
                if not isinstance(node, nodes.EntryNode):
                    continue
                entries += 1
                assert node.new_symbol_names(state) == set(node.new_symbols(state.sdfg, state, {}).keys())
    return entries


@dace.program
def single_map(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0


@dace.program
def nested_maps(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in dace.map[0:N]:
        for j in dace.map[0:M]:
            B[i, j] = A[i, j] + 1.0


@dace.program
def with_reduction(A: dace.float64[N, M], B: dace.float64[N]):
    B[:] = np.sum(A, axis=1)


@pytest.mark.parametrize('program', [single_map, nested_maps, with_reduction])
def test_new_symbol_names_matches_new_symbols(program):
    sdfg = program.to_sdfg(simplify=False)
    assert_names_match(sdfg)
    sdfg.simplify(validate=False)
    # A reduction only becomes a map once its library node expands, as it does before code generation.
    sdfg.expand_library_nodes()
    assert assert_names_match(sdfg) > 0


def test_new_symbol_names_dynamic_map_range():
    """A dynamic scope input is a defined symbol, so it must be named too."""
    sdfg = dace.SDFG('dynrange')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_scalar('lim', dace.int32, transient=False)
    state = sdfg.add_state()

    entry, exit_node = state.add_map('m', dict(i='0:N'))
    entry.add_in_connector('dyn')
    state.add_edge(state.add_access('lim'), None, entry, 'dyn', dace.Memlet('lim[0]'))

    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(state.add_access('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_access('B'), src_conn='b', memlet=dace.Memlet('B[i]'))

    names = entry.new_symbol_names(state)
    assert names == set(entry.new_symbols(sdfg, state, {}).keys())
    assert 'i' in names and 'dyn' in names


def test_new_symbol_names_consume():
    sdfg = dace.SDFG('consume')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_stream('S', dace.float64, transient=True)
    state = sdfg.add_state()

    entry, exit_node = state.add_consume('c', ('p', '4'))
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(state.add_access('S'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('S[0]'))
    state.add_memlet_path(tasklet, exit_node, state.add_access('A'), src_conn='b', memlet=dace.Memlet('A[p]'))

    names = entry.new_symbol_names(state)
    assert names == set(entry.new_symbols(sdfg, state, {}).keys())
    assert 'p' in names


def test_node_base_defines_nothing():
    assert nodes.Tasklet('t').new_symbol_names(None) == set()


def test_python_bool_normalizes_like_the_other_scalars():
    """``dace.bool`` shadows the builtin, which once left this branch unreachable."""
    assert dtypes.typeclass(builtins.bool).type is np.bool_
    assert dtypes.typeclass(builtins.bool) == dtypes.typeclass(np.bool_)
    # Equal typeclasses must hash alike, or both can occupy one dictionary.
    assert hash(dtypes.typeclass(builtins.bool)) == hash(dtypes.typeclass(np.bool_))


def test_scalar_widths_follow_the_configuration():
    for setting, expected in (('python', (np.int64, np.float64)), ('c', (np.int32, np.float32))):
        with dace.config.set_temporary('compiler', 'default_data_types', value=setting):
            assert dtypes.typeclass(int).type is expected[0]
            assert dtypes.typeclass(float).type is expected[1]


def test_unknown_configuration_still_builds_non_python_types():
    """Only Python's scalars consult the configuration; nothing else may start failing on it."""
    with dace.config.set_temporary('compiler', 'default_data_types', value='nonsense'):
        with pytest.raises(NameError):
            dtypes.typeclass(int)
        assert dtypes.typeclass(np.float32).type is np.float32


if __name__ == '__main__':
    test_new_symbol_names_matches_new_symbols(single_map)
    test_new_symbol_names_matches_new_symbols(nested_maps)
    test_new_symbol_names_matches_new_symbols(with_reduction)
    test_new_symbol_names_dynamic_map_range()
    test_new_symbol_names_consume()
    test_node_base_defines_nothing()
    test_python_bool_normalizes_like_the_other_scalars()
    test_scalar_widths_follow_the_configuration()
    test_unknown_configuration_still_builds_non_python_types()
