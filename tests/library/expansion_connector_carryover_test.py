# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An expansion into a ``CodeNode`` keeps the wired connectors a pass added to the library node."""
import dace
from dace import library, nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandAddOnePure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return parent_state.add_tasklet(node.label, {'_inp'}, {'_out'}, '_out = _inp + 1')


@library.node
class AddOne(nodes.LibraryNode):
    implementations = {'pure': ExpandAddOnePure}
    default_implementation = 'pure'

    def __init__(self, name):
        super().__init__(name, inputs={'_inp'}, outputs={'_out'})


def test_added_connectors_survive_expansion():
    sdfg = dace.SDFG('expansion_connector_carryover')
    for name in 'ABSC':
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state()
    lib = AddOne('addone')
    state.add_edge(state.add_read('A'), None, lib, '_inp', dace.Memlet('A[0]'))
    state.add_edge(lib, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    lib.add_in_connector('_side', dtype=dace.float64)
    state.add_edge(state.add_read('S'), None, lib, '_side', dace.Memlet('S[0]'))
    lib.add_out_connector('_extra', dtype=dace.float64)
    state.add_edge(lib, '_extra', state.add_write('C'), None, dace.Memlet('C[0]'))
    lib.add_in_connector('_unwired', dtype=dace.float64)

    sdfg.expand_library_nodes()
    sdfg.validate()

    tasklet, = (n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert set(tasklet.in_connectors) == {'_inp', '_side'}
    assert set(tasklet.out_connectors) == {'_out', '_extra'}
    assert {e.dst_conn for e in state.in_edges(tasklet)} == {'_inp', '_side'}
    assert {e.src_conn for e in state.out_edges(tasklet)} == {'_out', '_extra'}


if __name__ == '__main__':
    test_added_connectors_survive_expansion()
