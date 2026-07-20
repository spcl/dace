# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``symbol_scopes`` must agree with ``symbols_defined_at`` node for node.

The pass exists only to avoid recomputing an SDFG-invariant table per node, so equivalence is the
whole specification: same keys, same types, and the same ORDER (both return an OrderedDict, and
callers such as ``mpi.py`` feed the result straight back into ``new_symbols``).
"""

import dace
import pytest
from dace import dtypes
from dace.codegen.symbol_scopes import defined_at, symbol_scopes

N = dace.symbol('N')
M = dace.symbol('M')


def assert_matches(sdfg: dace.SDFG):
    """Every node of every state must get the identical table from both routes."""
    scopes = symbol_scopes(sdfg)
    checked = 0
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.nodes():
                expected = state.symbols_defined_at(node)
                actual = defined_at(scopes, state, node)
                assert list(actual.keys()) == list(expected.keys()), \
                    f'{state.label}/{node}: key order differs\n  {list(actual.keys())}\n  {list(expected.keys())}'
                assert actual == expected, f'{state.label}/{node}: values differ'
                checked += 1
    assert checked > 0, 'test built no nodes to compare'
    return checked


def test_nested_maps():

    @dace.program
    def nested_maps(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in dace.map[0:M]:
                A[i, j] = A[i, j] * 2.0

    assert_matches(nested_maps.to_sdfg(simplify=False))


def test_loop_region_iterator_is_visible():
    """A node inside a LoopRegion must see the loop variable, from both routes."""
    sdfg = dace.SDFG('loopreg')
    sdfg.add_array('A', [N], dace.float64)
    loop = dace.sdfg.state.LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('w', {}, {'o'}, 'o = i')
    body.add_edge(tasklet, 'o', body.add_access('A'), None, dace.Memlet('A[i]'))

    # Whether the iterator itself lands in the table is symbols_defined_at's business; the pass
    # only has to agree with it.
    assert_matches(sdfg)


def test_dynamic_map_range_connector():
    """A dynamic map range binds its connector as a symbol -- MapEntry.new_symbols must be folded in."""
    sdfg = dace.SDFG('dynrange')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('lim', [1], dace.int32)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_ = state.add_map('m', {'i': '0:bound'})
    entry.add_in_connector('bound')
    state.add_edge(state.add_access('lim'), None, entry, 'bound', dace.Memlet('lim[0]'))
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_, state.add_access('A'), src_conn='o', memlet=dace.Memlet('A[i]'))

    scopes = symbol_scopes(sdfg)
    inner = defined_at(scopes, state, tasklet)
    assert 'i' in inner and 'bound' in inner, f'dynamic range connector missing: {sorted(inner)}'
    # The entry itself sees its OUTER scope, not its own parameters
    assert 'i' not in defined_at(scopes, state, entry)
    assert_matches(sdfg)


def test_nested_sdfg():

    @dace.program
    def inner(A: dace.float64[N]):
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0

    @dace.program
    def outer(A: dace.float64[N]):
        inner(A)

    assert_matches(outer.to_sdfg(simplify=False))


@pytest.mark.parametrize('simplify', [False, True])
def test_realistic_program(simplify):

    @dace.program
    def gemm(A: dace.float64[N, M], B: dace.float64[M, N], C: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            acc = dace.float64(0)
            for k in range(M):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    assert assert_matches(gemm.to_sdfg(simplify=simplify)) > 0


def test_scalar_free_symbols_reach_every_scope():
    """An array extent symbol is SDFG-global and must appear at the innermost scope too."""
    sdfg = dace.SDFG('extents')
    sdfg.add_array('A', [N * M], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_ = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_, state.add_access('A'), src_conn='o', memlet=dace.Memlet('A[i]'))

    scopes = symbol_scopes(sdfg)
    inner = defined_at(scopes, state, tasklet)
    assert 'N' in inner and 'M' in inner
    assert isinstance(inner['N'], dtypes.typeclass)
    assert_matches(sdfg)


if __name__ == '__main__':
    test_nested_maps()
    test_loop_region_iterator_is_visible()
    test_dynamic_map_range_connector()
    test_nested_sdfg()
    test_realistic_program(False)
    test_realistic_program(True)
    test_scalar_free_symbols_reach_every_scope()
