import dace
from dace.transformation.subgraph import MultiExpansion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.subgraph import ReduceExpansion
from dace.transformation.subgraph.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView


from dace.transformation.subgraph.pipeline import expand_reduce, expand_maps, fusion


import sys

N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)

A = np.random.rand(N.get()).astype(np.float64)
B = np.random.rand(M.get()).astype(np.float64)
C = np.random.rand(O.get()).astype(np.float64)
OUT1 = np.ndarray((N.get(), M.get()), np.float64)
OUT2 = np.ndarray((1), np.float64)
OUT3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)



@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O], \
         OUT1: dace.float64[N,M], OUT2: dace.float64[1], OUT3: dace.float64[N,M,O]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp2 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp3 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp4 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp5 = np.ndarray([N,M,O], dtype = dace.float64)


    t1 = np.ndarray([N,M], dtype = dace.float64)
    t2 = np.ndarray([N,M], dtype = dace.float64)
    t3 = np.ndarray([N,M], dtype = dace.float64)

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        tp = np.ndarray([1], dtype = dace.float64)
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tp

            out = in1 + in2 + in3

        with dace.tasklet:
            in1 << tp
            out >> tmp1[i,j,k]

            out = in1 + 42

    dace.reduce(lambda a,b: a+b, tmp1, t1, axis = 2, identity = 0)

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            out >> t2[i,j]
            out = in1 + in2 + 42

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << t2[i,j]
            in2 << A[i]
            out >> OUT1[i,j]

            out = in1*in1*in2 + in2

    for i,j,k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << t1[i,j]
            in2 << t2[i,j]
            in3 << C[k]
            out >> tmp3[i,j,k]

            out = in1 + in2 + in3

    for i,j,k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << tmp3[i,j,k]
            in2 << tmp1[i,j,k]
            out >> OUT3[i,j,k]

            out = in1 + in2

    @dace.tasklet
    def fun():
        in1 << tmp3[0,0,0]
        out >> OUT2

        out = in1 * 42

def test_qualitatively(sdfg, graph):
    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)
    fusion(sdfg, graph)
    sdfg.validate()
    print("PASS")

def test_quantitatively(sdfg, graph):


    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(M.get()).astype(np.float64)
    C = np.random.rand(O.get()).astype(np.float64)
    OUT1_base = np.ndarray((N.get(), M.get()), np.float64)
    OUT2_base = np.ndarray((1), np.float64)
    OUT3_base = np.ndarray((N.get(), M.get(), O.get()), np.float64)
    OUT1 = np.ndarray((N.get(), M.get()), np.float64)
    OUT2 = np.ndarray((1), np.float64)
    OUT3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, OUT1 = OUT1_base, OUT2 = OUT2_base, OUT3 = OUT3_base, N=N, M=M, O=O)

    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)
    #sgf = SubgraphFusion()
    #matcher = sgf.match(sdfg, SubgraphView(graph, [node for node in graph.nodes()]))
    #assert matcher == True
    fusion(sdfg, graph)
    sdfg.validate()
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, OUT1 = OUT1, OUT2 = OUT2, OUT3 = OUT3, N=N, M=M, O=O)

    assert np.allclose(OUT1, OUT1_base)
    assert np.allclose(OUT2, OUT2_base)
    assert np.allclose(OUT3, OUT3_base)
    print('PASS')


if __name__ == "__main__":

    sdfg = TEST.to_sdfg()
    sdfg.apply_strict_transformations()
    test_quantitatively(sdfg, sdfg.nodes()[0])
