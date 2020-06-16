import dace
from dace.transformation.heterogeneous import MultiExpansion
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np

from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion


import sys

N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]
N.set(50)
M.set(60)
O.set(70)
P.set(80)
Q.set(90)
R.set(100)


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
    sdfg.view()
    sdfg.validate()
    print("PASS")

def test_quantitatively(sdfg, graph):
    runner = dace.measure.Runner(view_all = True)
    runner.go(sdfg, graph, None,
              M, N, O,
              output = ["OUT1", "OUT2", "OUT3"],
              performance_spec = dace.perf.specs.PERF_CPU_CRAPBOOK
              )



if __name__ == "__main__":

    sdfg = TEST.to_sdfg()
    #sdfg.apply_strict_transformations()
    #sdfg.apply_gpu_transformations()

    test_qualitatively(sdfg, sdfg.nodes()[0])
    #test_quantitatively(sdfg, sdfg.nodes()[0])
