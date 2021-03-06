# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
from dace.transformation.subgraph import ReduceExpansion
from dace.sdfg.graph import SubgraphView
import dace.sdfg.nodes as nodes
import numpy as np
import dace.libraries.standard as stdlib

from typing import Union, List
from util import expand_reduce

M = dace.symbol('M')
N = dace.symbol('N')
N.set(20)
M.set(30)


@dace.program
def program(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[N]):

    tmp = np.ndarray(shape=[M, N], dtype=np.float64)
    tmp[:] = 2 * A[:] + B[:]
    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=0)


@dace.program
def program2(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[N]):

    tmp = np.ndarray(shape=[M, N], dtype=np.float64)
    C[:] = dace.reduce(lambda a, b: max(a, b), B, axis=0)
    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << C[j]
            in2 << A[i, j]
            out1 >> tmp[i, j]
            out1 = in1 * in2
    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=0)


def test_p1():
    sdfg = program.to_sdfg()
    sdfg.apply_strict_transformations()
    state = sdfg.nodes()[0]
    for node in state.nodes():
        if isinstance(node, dace.libraries.standard.nodes.Reduce):
            reduce_node = node

    assert ReduceExpansion.can_be_applied(state, \
                                          {ReduceExpansion._reduce: state.nodes().index(reduce_node)}, \
                                          0, \
                                          sdfg) == True

    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype=np.float64)
    C2 = np.zeros([N.get()], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M)
    del csdfg

    expand_reduce(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M)
    del csdfg

    assert np.allclose(C1, C2)
    print(np.linalg.norm(C1))
    print(np.linalg.norm(C2))
    print("PASS")


def test_p2():
    sdfg = program2.to_sdfg()
    sdfg.apply_strict_transformations()
    state = sdfg.nodes()[0]
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype=np.float64)
    C2 = np.zeros([N.get()], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M)
    del csdfg

    expand_reduce(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M)

    assert np.allclose(C1, C2)
    print(np.linalg.norm(C1))
    print(np.linalg.norm(C2))
    print("PASS")


if __name__ == "__main__":
    test_p1()
    test_p2()
