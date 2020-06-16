import copy
import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFission
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest
import sys

from dace.transformation.heterogeneous.pipeline import fusion


N = dace.symbol('N')
N.set(1000)
@dace.program
def TEST(A: dace.float64[N], B: dace.float64[N], C:dace.float64[N], D:dace.float64[N]):

    for i in dace.map[0:N//2]:
        with dace.tasklet:
            in1 << A[2*i]
            in2 << A[2*i+1]
            out >> C[2*i]

            out = in1 + in2

    for i in dace.map[0:N//2]:
        with dace.tasklet:
            in1 << B[2*i]
            in2 << B[2*i+1]
            out >> C[2*i+1]

            out = in1 + in2

    for i in dace.map[0:N//2]:
        with dace.tasklet:
            in1 << C[2*i:2*i+2]
            out1 >> D[2*i:2*i+2]

            out1[0] = in1[0]*in1[0]
            out1[1] = in1[1]*in1[1]

def test_quantitatively(sdfg):
    runner = dace.measure.Runner()
    runner.go(sdfg, sdfg.nodes()[0], None,
              N, pipeline = [dace.transformation.heterogeneous.pipeline.fusion],
              output = ['C','D'])

if __name__ == '__main__':
    sdfg = TEST.to_sdfg()
    sdfg.apply_transformations(dace.transformation.interstate.state_fusion.StateFusion)
    # merge the C array
    C1 = None; C2 = None
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
    sys.exit(0)

    dace.transformation.heterogeneous.pipeline.fusion(sdfg, sdfg.nodes()[0])
    sdfg.view()
