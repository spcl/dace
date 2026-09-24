# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An expansion into a ``CodeNode`` keeps connectors a pass added to the library node."""
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


def _sdfg_with_extra_connector() -> dace.SDFG:
    """One ``AddOne`` node carrying an extra ``_side`` input wired from a scalar."""
    sdfg = dace.SDFG('expansion_connector_carryover')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('S', [1], dace.float64)
    state = sdfg.add_state()

    lib = AddOne('addone')
    state.add_node(lib)
    state.add_edge(state.add_read('A'), None, lib, '_inp', dace.Memlet('A[0]'))
    state.add_edge(lib, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))

    lib.add_in_connector('_side', dtype=dace.float64)
    state.add_edge(state.add_read('S'), None, lib, '_side', dace.Memlet('S[0]'))
    return sdfg


def test_extra_connector_survives_expansion():
    sdfg = _sdfg_with_extra_connector()
    sdfg.expand_library_nodes()
    sdfg.validate()

    state = sdfg.states()[0]
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1
    assert '_side' in tasklets[0].in_connectors
    assert any(e.dst_conn == '_side' for e in state.in_edges(tasklets[0]))


if __name__ == '__main__':
    test_extra_connector_survives_expansion()
