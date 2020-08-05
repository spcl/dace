import dace
from dace.transformation.heterogeneous import MultiExpansion
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView


from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion


import sys

N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)

A = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
B = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
C = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
OUT1 = np.ndarray((N.get(), M.get(), O.get()), np.float64)


@dace.program
def TEST(A: dace.float64[N,M,O], B: dace.float64[N,M,O], C: dace.float64[N,M,O]):
    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i,j,0]
            in2 << B[i,j,0]
            out >> C[i,j,0]

            out = in1 + in2

    for i,j in dace.map[0:N, 0:M]:
        for z in range(1,O):
            with dace.tasklet:
                in1 << A[i,j,z]
                in2 << B[i,j,z]
                in3 << C[i,j,0]
                out >> C[i,j,z]

                out = 2*in1 + 2*in2 + in3

def test_qualitatively(sdfg, graph):
    fusion(sdfg, graph)
    sdfg.view()
    sdfg.validate()
    print("PASS")

def test_quantitatively(sdfg, graph):
    A = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
    B = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
    C1 = np.zeros([N.get(), M.get(), O.get()], dtype = np.float64)
    C2 = np.zeros([N.get(), M.get(), O.get()], dtype = np.float64)

    sdfg.view()
    sdfg.validate()
    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C1,N=N,M=M,O=O)

    fusion(sdfg, graph)
    sdfg.view()
    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C2,N=N,M=M,O=O)

    assert np.allclose(C1,C2)


if __name__ == '__main__':
    sdfg = TEST.to_sdfg()
    graph = sdfg.nodes()[0]
    test_quantitatively(sdfg, graph)
