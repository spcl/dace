import copy
import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.dataflow import MapFission
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest
import sys

from dace.transformation.subgraph import MultiExpansion, SubgraphFusion

from typing import Union, List
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph.helpers import *

N = dace.symbol('N')
N.set(1000)


@dace.program
def TEST(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N],
         D: dace.float64[N]):

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << A[2 * i]
            in2 << A[2 * i + 1]
            out >> C[2 * i]

            out = in1 + in2

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << B[2 * i]
            in2 << B[2 * i + 1]
            out >> C[2 * i + 1]

            out = in1 + in2

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << C[2 * i:2 * i + 2]
            out1 >> D[2 * i:2 * i + 2]

            out1[0] = in1[0] * in1[0]
            out1[1] = in1[1] * in1[1]


def test_quantitatively(sdfg):
    graph = sdfg.nodes()[0]
    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(N.get()).astype(np.float64)
    C1 = np.random.rand(N.get()).astype(np.float64)
    C2 = np.random.rand(N.get()).astype(np.float64)
    D1 = np.random.rand(N.get()).astype(np.float64)
    D2 = np.random.rand(N.get()).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, D=D1, N=N)

    subgraph = SubgraphView(graph, [node for node in graph.nodes()])
    expansion = MultiExpansion()
    fusion = SubgraphFusion()
    assert expansion.match(sdfg, subgraph) == True
    expansion.apply(sdfg, subgraph)
    assert fusion.match(sdfg, subgraph) == True
    fusion.apply(sdfg, subgraph)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, D=D2, N=N)

    assert np.allclose(C1, C2)
    assert np.allclose(D1, D2)


if __name__ == '__main__':
    sdfg = TEST.to_sdfg()
    from dace.transformation.interstate.state_fusion import StateFusion
    sdfg.apply_transformations_repeated(StateFusion)
    # merge the C array
    C1 = None
    C2 = None
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data == 'C':
            if not C1:
                C1 = node
            elif not C2:
                C2 = node
                break
    print(C1, C2)
    dace.sdfg.utils.change_edge_dest(sdfg.nodes()[0], C2, C1)
    dace.sdfg.utils.change_edge_src(sdfg.nodes()[0], C2, C1)
    sdfg.nodes()[0].remove_node(C2)
    sdfg.validate()
    test_quantitatively(sdfg)
