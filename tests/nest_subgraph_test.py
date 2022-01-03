# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.nodes import MapEntry, Tasklet
from dace.sdfg.graph import NodeNotFoundError, SubgraphView
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.dataflow import tiling
import pytest

N = dace.symbol('N')


def create_sdfg():
    sdfg = dace.SDFG('badscope_test')
    sdfg.add_array('A', [2], dace.float32)
    sdfg.add_array('B', [2], dace.float32)
    state = sdfg.add_state()
    t, me, mx = state.add_mapped_tasklet('map',
                                         dict(i='0:2'),
                                         dict(a=dace.Memlet.simple('A', 'i')),
                                         'b = a * 2',
                                         dict(b=dace.Memlet.simple('B', 'i')),
                                         external_edges=True)
    return sdfg, state, t, me, mx


def create_sdfg_4():
    sdfg = dace.SDFG('sdfg_4_test')
    sdfg.add_array('A', [4], dace.float32)
    sdfg.add_array('B', [4], dace.float32)
    state = sdfg.add_state()
    t, me, mx = state.add_mapped_tasklet('map',
                                         dict(i='0:4'),
                                         dict(a=dace.Memlet.simple('A', 'i')),
                                         'b = a * 2',
                                         dict(b=dace.Memlet.simple('B', 'i')),
                                         external_edges=True)
    return sdfg, state, t, me, mx


def create_tiled_sdfg():
    sdfg = dace.SDFG('badscope_tile_test')
    sdfg.add_array('A', [4], dace.float32)
    sdfg.add_array('B', [4], dace.float32)
    state = sdfg.add_state()
    ome, omx = state.add_map('outer_map', dict(i='0:2'))
    ime, imx = state.add_map('inner_map', dict(j='0:2'))
    t = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a * 2')
    A = state.add_read('A')
    B = state.add_write('B')
    state.add_memlet_path(A,
                          ome,
                          ime,
                          t,
                          dst_conn='a',
                          memlet=dace.Memlet.simple('A', 'i*2 + j'))
    state.add_memlet_path(t,
                          imx,
                          omx,
                          B,
                          src_conn='b',
                          memlet=dace.Memlet.simple('B', 'i*2 + j'))
    return sdfg, state


def test_simple_program():
    @dace.program
    def multiply(a: dace.float32[N]):
        a *= 2
        a *= 3

    sdfg = multiply.to_sdfg(coarsen=True)
    for state in sdfg.nodes():
        if any(isinstance(node, Tasklet) for node in state.nodes()):
            break
    else:
        raise KeyError('State with tasklet not found')

    tasklet_nodes = [n for n in state.nodes() if isinstance(n, Tasklet)]
    with pytest.raises(ValueError):
        nest_state_subgraph(sdfg, state, SubgraphView(state, tasklet_nodes))

    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet_nodes[0]]))
    sdfg.validate()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet_nodes[1]]))
    sdfg.validate()


def test_simple_sdfg():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [t]))
    sdfg.validate()


def test_index_propagation_in_tiled_sdfg():
    sdfg, state, t, me, mx = create_sdfg_4()
    tiling.MapTiling.apply_to(sdfg=sdfg,
                              options={'tile_sizes': (2, )},
                              map_entry=me)
    nested_me = state.in_edges(t)[0].src
    nested_mx = state.out_edges(t)[0].dst
    nest_state_subgraph(sdfg, state,
                        SubgraphView(state, [nested_me, t, nested_mx]))
    sdfg.validate()
    sdfg.compile()


def test_simple_sdfg_map():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, [me, t, mx]))
    sdfg.validate()


def test_simple_sdfg_program():
    sdfg, state, t, me, mx = create_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
    sdfg.validate()


def test_badscope():
    with pytest.raises(ValueError):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, me]))

    with pytest.raises(ValueError):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, mx]))

    with pytest.raises(NodeNotFoundError):
        sdfg, state, t, me, mx = create_sdfg()
        b_node = state.sink_nodes()[0]
        sdfg, state, t, me, mx = create_sdfg()
        # Notice that b_node comes from another graph
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t, b_node]))


def test_tiled_program():
    # Tasklet only
    sdfg, state = create_tiled_sdfg()
    tasklet = next(n for n in state.nodes() if isinstance(n, Tasklet))
    nest_state_subgraph(sdfg, state, SubgraphView(state, [tasklet]))
    sdfg.validate()

    # Inner map scope
    sdfg, state = create_tiled_sdfg()
    tasklet = next(n for n in state.nodes() if isinstance(n, Tasklet))
    entry = state.entry_node(tasklet)
    nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
    sdfg.validate()

    # Outer map scope
    sdfg, state = create_tiled_sdfg()
    sdc = state.scope_children()
    entry = next(n for n in sdc[None] if isinstance(n, MapEntry))
    nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
    sdfg.validate()

    # Entire state
    sdfg, state = create_tiled_sdfg()
    nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
    sdfg.validate()


if __name__ == '__main__':
    test_simple_program()
    test_simple_sdfg()
    test_index_propagation_in_tiled_sdfg()
    test_simple_sdfg_map()
    test_simple_sdfg_program()
    test_badscope()
    test_tiled_program()