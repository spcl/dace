# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
from dace.transformation.subgraph import ReduceExpansion
from dace.sdfg.graph import SubgraphView
import dace.sdfg.nodes as nodes
import numpy as np
import dace.libraries.standard as stdlib

from typing import Union, List
from util import expand_reduce, expand_maps, fusion

import pytest

M = dace.symbol('M')
N = dace.symbol('N')
N.set(20)
M.set(30)


@dace.program
def reduction_test_1(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[N]):

    tmp = np.ndarray(shape=[M, N], dtype=np.float64)
    tmp[:] = 2 * A[:] + B[:]
    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=0)


@dace.program
def reduction_test_2(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[N]):

    tmp = np.ndarray(shape=[M, N], dtype=np.float64)
    C[:] = dace.reduce(lambda a, b: max(a, b), B, axis=0)
    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << C[j]
            in2 << A[i, j]
            out1 >> tmp[i, j]
            out1 = in1 * in2
    C[:] = dace.reduce(lambda a, b: a + b, tmp, axis=0)


settings = [[False, False], [True, False], [False, True]]


@pytest.mark.parametrize(["in_transient", "out_transient"], settings)
def test_p1(in_transient, out_transient):
    sdfg = reduction_test_1.to_sdfg()
    sdfg.coarsen_dataflow()
    state = sdfg.nodes()[0]
    for node in state.nodes():
        if isinstance(node, dace.libraries.standard.nodes.Reduce):
            reduce_node = node

    rexp = ReduceExpansion(sdfg, sdfg.sdfg_id, 0, {ReduceExpansion.reduce: state.node_id(reduce_node)}, 0)
    assert rexp.can_be_applied(state, 0, sdfg) == True

    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype=np.float64)
    C2 = np.zeros([N.get()], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M)
    del csdfg

    expand_reduce(sdfg, state, create_in_transient=in_transient, create_out_transient=out_transient)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M)
    del csdfg

    assert np.linalg.norm(C1) > 0.01
    assert np.allclose(C1, C2)


settings = [[False, False], [True, False], [False, True]]


@pytest.mark.parametrize(["in_transient", "out_transient"], settings)
def test_p2(in_transient, out_transient):
    sdfg = reduction_test_2.to_sdfg()
    sdfg.coarsen_dataflow()
    state = sdfg.nodes()[0]
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype=np.float64)
    C2 = np.zeros([N.get()], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M)
    del csdfg

    expand_reduce(sdfg, state, create_in_transient=in_transient, create_out_transient=out_transient)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M)

    assert np.linalg.norm(C1) > 0.01
    assert np.allclose(C1, C2)


if __name__ == "__main__":
    test_p1(in_transient=False, out_transient=False)
    test_p2(in_transient=False, out_transient=False)

    test_p1(in_transient=True, out_transient=False)
    test_p2(in_transient=True, out_transient=False)

    test_p1(in_transient=True, out_transient=True)
    test_p2(in_transient=True, out_transient=True)
