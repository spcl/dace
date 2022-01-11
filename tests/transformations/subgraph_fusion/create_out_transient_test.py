# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.subgraph import SubgraphFusion
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView
from dace.transformation.interstate import StateFusion
from typing import List, Union
import sys
from util import fusion

N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)


@dace.program
def program1(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, j]
            out1 >> A[i, j]
            out1 = in1 + 1.0

    with dace.tasklet:
        in1 << A[:]
        out1 >> B[:]
        out1 = in1

    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, j]
            out >> A[i, j]
            out = in1 + 2.0

    with dace.tasklet:
        in1 << A[:]
        out1 >> C[:]
        out1 = in1


@dace.program
def program2(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    tmp = np.ndarray(shape=(M, N), dtype=np.float64)
    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, j]
            out1 >> tmp[i, j]
            out1 = in1 + 1.0

    with dace.tasklet:
        in1 << tmp[:]
        out1 >> B[:]
        out1 = in1

    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << tmp[i, j]
            out >> A[i, j]
            out = in1 + 2.0

    with dace.tasklet:
        in1 << A[:]
        out1 >> C[:]
        out1 = in1


def _test_quantitatively(sdfg, graph):
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B1 = np.zeros(shape=[M.get(), N.get()], dtype=np.float64)
    C1 = np.zeros(shape=[M.get(), N.get()], dtype=np.float64)
    B2 = np.zeros(shape=[M.get(), N.get()], dtype=np.float64)
    C2 = np.zeros(shape=[M.get(), N.get()], dtype=np.float64)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B1, C=C1, N=N, M=M)
    del csdfg
    fusion(sdfg, graph)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B2, C=C2, N=N, M=M)
    del csdfg
    assert np.allclose(B1, B2)
    assert np.allclose(C1, C2)


def test_out_transient1():
    # non-transient
    sdfg = program1.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]
    _test_quantitatively(sdfg, graph)


def test_out_transient2():
    # transient
    sdfg = program2.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]
    _test_quantitatively(sdfg, graph)


if __name__ == "__main__":
    test_out_transient1()
    test_out_transient2()
