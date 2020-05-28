import unittest

import dace
from dace.sdfg.nodes import MapEntry, Tasklet
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import nest_state_subgraph

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


class NestStateSubgraph(unittest.TestCase):
    def test_simple_program(self):
        @dace.program
        def multiply(a: dace.float32[N]):
            a *= 2

        sdfg = multiply.to_sdfg(strict=True)
        for state in sdfg.nodes():
            if any(isinstance(node, Tasklet) for node in state.nodes()):
                break
        else:
            raise KeyError('State with tasklet not found')

        tasklet_nodes = [n for n in state.nodes() if isinstance(n, Tasklet)]
        with self.assertRaises(ValueError):
            nest_state_subgraph(sdfg, state,
                                SubgraphView(state, tasklet_nodes))

        nest_state_subgraph(sdfg, state, SubgraphView(state,
                                                      [tasklet_nodes[0]]))
        sdfg.validate()
        nest_state_subgraph(sdfg, state, SubgraphView(state,
                                                      [tasklet_nodes[1]]))
        sdfg.validate()

    def test_simple_sdfg(self):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [t]))
        sdfg.validate()

    def test_simple_sdfg_map(self):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, [me, t, mx]))
        sdfg.validate()

    def test_simple_sdfg_program(self):
        sdfg, state, t, me, mx = create_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
        sdfg.validate()

    def test_badscope(self):
        with self.assertRaises(ValueError):
            sdfg, state, t, me, mx = create_sdfg()
            nest_state_subgraph(sdfg, state, SubgraphView(state, [t, me]))

        with self.assertRaises(ValueError):
            sdfg, state, t, me, mx = create_sdfg()
            nest_state_subgraph(sdfg, state, SubgraphView(state, [t, mx]))

        with self.assertRaises(KeyError):
            sdfg, state, t, me, mx = create_sdfg()
            b_node = state.sink_nodes()[0]
            sdfg, state, t, me, mx = create_sdfg()
            # Notice that b_node comes from another graph
            nest_state_subgraph(sdfg, state, SubgraphView(state, [t, b_node]))

    def test_tiled_program(self):
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
        sdc = state.scope_dict(True)
        entry = next(n for n in sdc[None] if isinstance(n, MapEntry))
        nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
        sdfg.validate()

        # Entire state
        sdfg, state = create_tiled_sdfg()
        nest_state_subgraph(sdfg, state, SubgraphView(state, state.nodes()))
        sdfg.validate()


if __name__ == '__main__':
    unittest.main()
