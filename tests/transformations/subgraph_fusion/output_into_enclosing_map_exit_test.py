# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" SubgraphFusion refuses maps whose output flows straight into an enclosing map's exit instead of raising. """
import dace
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion

N = 8


def nested_maps_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('output_into_enclosing_map_exit')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('C', [N, N], dace.float64)
    sdfg.add_transient('tmp', [N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i=f'0:{N}'))
    first_entry, first_exit = state.add_map('first', dict(j=f'0:{N}'))
    second_entry, second_exit = state.add_map('second', dict(j=f'0:{N}'))
    copy = state.add_tasklet('copy', {'a'}, {'b'}, 'b = a + 1')
    scale = state.add_tasklet('scale', {'a'}, {'b'}, 'b = a * 2')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_read('A'),
                          outer_entry,
                          first_entry,
                          copy,
                          dst_conn='a',
                          memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(copy, first_exit, tmp, src_conn='b', memlet=dace.Memlet('tmp[j]'))
    state.add_memlet_path(tmp, second_entry, scale, dst_conn='a', memlet=dace.Memlet('tmp[j]'))
    state.add_memlet_path(scale,
                          second_exit,
                          outer_exit,
                          state.add_write('C'),
                          src_conn='b',
                          memlet=dace.Memlet('C[i, j]'))
    return sdfg


def test_a_map_output_into_an_enclosing_exit_is_not_fused():
    sdfg = nested_maps_sdfg()
    sdfg.validate()
    state = sdfg.node(0)
    inner = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label in ('first', 'second')]
    assert len(inner) == 2
    subgraph = SubgraphView(state, [n for e in inner for n in state.scope_subgraph(e).nodes()])
    fusion = SubgraphFusion()
    fusion.disjoint_subsets = True
    assert fusion.can_be_applied(state, subgraph) is False


if __name__ == '__main__':
    test_a_map_output_into_an_enclosing_exit_is_not_fused()
