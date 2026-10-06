# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InlineSDFG`` names an inlined transient apart from the connectors of the parent's code nodes, which validation
forbids data to share (an expanded copy's tasklet reads ``_cpy_in``, and so is named every copy nest's local view)."""
import numpy as np

import dace
from dace.libraries.standard.nodes import CopyLibraryNode
from dace.transformation.interstate import InlineSDFG

#: The connector names of an expanded copy's tasklet, which every copy nest also gives its local views.
CPY_IN, CPY_OUT = CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME


def parent_with_a_cpy_in_connector_and_a_nest() -> dace.SDFG:
    """A tasklet with the copy connectors, beside a nest computing through its own transient named like the input
    connector."""
    nest = dace.SDFG('nest')
    nest.add_array('x', [4], dace.float64)
    nest.add_array('y', [4], dace.float64)
    nest.add_transient(CPY_IN, [4], dace.float64)
    ns = nest.add_state('s')
    ns.add_mapped_tasklet('double',
                          dict(i='0:4'), {'a': dace.Memlet('x[i]')},
                          'b = 2 * a', {'b': dace.Memlet(f'{CPY_IN}[i]')},
                          external_edges=True)
    tmp = next(n for n in ns.data_nodes() if n.data == CPY_IN)
    ns.add_mapped_tasklet('copy_back',
                          dict(i='0:4'), {'a': dace.Memlet(f'{CPY_IN}[i]')},
                          'b = a', {'b': dace.Memlet('y[i]')},
                          external_edges=True,
                          input_nodes={CPY_IN: tmp})

    sdfg = dace.SDFG('inline_beside_a_cpy_in_connector')
    for name in ('A', 'B', 'C', 'D'):
        sdfg.add_array(name, [4], dace.float64)
    state = sdfg.add_state('s')
    state.add_mapped_tasklet('copy_C_to_D',
                             dict(i='0:4'), {CPY_IN: dace.Memlet('C[i]')},
                             f'{CPY_OUT} = {CPY_IN}', {CPY_OUT: dace.Memlet('D[i]')},
                             external_edges=True)
    node = state.add_nested_sdfg(nest, {'x'}, {'y'})
    state.add_edge(state.add_read('A'), None, node, 'x', dace.Memlet('A[0:4]'))
    state.add_edge(node, 'y', state.add_write('B'), None, dace.Memlet('B[0:4]'))
    sdfg.validate()
    return sdfg


def test_an_inlined_transient_is_named_apart_from_a_connector():
    sdfg = parent_with_a_cpy_in_connector_and_a_nest()

    assert sdfg.apply_transformations_repeated(InlineSDFG) == 1
    sdfg.validate()
    assert CPY_IN not in sdfg.arrays

    a, c = np.arange(4.0), np.arange(4.0) + 10
    b, d = np.zeros(4), np.zeros(4)
    sdfg(A=a, B=b, C=c, D=d)
    assert np.allclose(b, 2 * a) and np.allclose(d, c)


if __name__ == '__main__':
    test_an_inlined_transient_is_named_apart_from_a_connector()
