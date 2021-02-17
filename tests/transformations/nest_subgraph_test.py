# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.helpers import nest_state_subgraph
import numpy as np


@dace.program
def nest_subgraph(A: dace.float64[1], B: dace.float64[1]):
    for i in dace.map[0:1]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            b = a


def test_nest_oneelementmap():
    A, B = np.random.rand(1), np.random.rand(1)
    sdfg: dace.SDFG = nest_subgraph.to_sdfg()
    state: dace.SDFGState
    # Nest outer region
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node)
            nest_state_subgraph(sdfg, state, subgraph)
    # Nest inner scope
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            subgraph = state.scope_subgraph(node,
                                            include_entry=False,
                                            include_exit=False)
            nest_state_subgraph(state.parent, state, subgraph)

    sdfg(A=A, B=B)
    assert np.allclose(A, B)


if __name__ == '__main__':
    test_nest_oneelementmap()