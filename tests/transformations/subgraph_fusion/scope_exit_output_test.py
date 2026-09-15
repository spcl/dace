# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Maps nested in an outer map whose output leaves straight through the outer map's exit, with no access node
in between, are refused: fusion reroutes every output through an access node. InlineSDFG produces this shape
when it inlines a nested SDFG that writes its output connector inside a map (vadv in the offload corpus). """

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion

N = 8


def build_write_through_outer_exit_sdfg() -> dace.SDFG:
    """ ``B[k, i] = 2 * A[k, i]`` over an outer map on ``k``, via two inner maps; the second writes ``B`` through
    the outer map's exit. """
    sdfg = dace.SDFG('subgraph_fusion_scope_exit_output')
    sdfg.add_array('A', [2, N], dace.float64)
    sdfg.add_array('B', [2, N], dace.float64)
    sdfg.add_array('tmp', [N], dace.float64, transient=True)
    state = sdfg.add_state()

    outer_entry, outer_exit = state.add_map('outer', dict(k='0:2'))
    first_entry, first_exit = state.add_map('first', dict(i=f'0:{N}'))
    second_entry, second_exit = state.add_map('second', dict(i=f'0:{N}'))
    first = state.add_tasklet('first', {'a'}, {'o'}, 'o = a')
    second = state.add_tasklet('second', {'x'}, {'o'}, 'o = 2 * x')
    tmp = state.add_access('tmp')

    state.add_memlet_path(state.add_read('A'),
                          outer_entry,
                          first_entry,
                          first,
                          dst_conn='a',
                          memlet=dace.Memlet('A[k, i]'))
    state.add_memlet_path(first, first_exit, tmp, src_conn='o', memlet=dace.Memlet('tmp[i]'))
    state.add_memlet_path(tmp, second_entry, second, dst_conn='x', memlet=dace.Memlet('tmp[i]'))
    state.add_memlet_path(second,
                          second_exit,
                          outer_exit,
                          state.add_write('B'),
                          src_conn='o',
                          memlet=dace.Memlet('B[k, i]'))
    sdfg.validate()
    return sdfg


def test_maps_writing_through_the_outer_map_exit_are_not_fused():
    sdfg = build_write_through_outer_exit_sdfg()
    state = sdfg.states()[0]
    outer_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.label == 'outer')
    body = state.scope_subgraph(outer_entry, include_entry=False, include_exit=False)
    subgraph = SubgraphView(state, body.nodes())

    sut = SubgraphFusion()
    sut.setup_match(subgraph)

    assert not sut.can_be_applied(sdfg, subgraph)
    A = np.random.default_rng(3).random((2, N))
    B = np.zeros((2, N))
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


if __name__ == '__main__':
    test_maps_writing_through_the_outer_map_exit_are_not_fused()
