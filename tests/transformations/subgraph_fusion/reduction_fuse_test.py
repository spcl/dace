# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List, Union

import numpy as np
import pytest
from util import expand_maps, expand_reduce, fusion

import dace
import dace.libraries.standard as stdlib
import dace.sdfg.nodes as nodes
import dace.transformation.subgraph.helpers as helpers
from dace.sdfg.graph import SubgraphView
from dace.transformation.dataflow import ReduceExpansion

M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def reduction_test_3(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[N]):

    tmp = dace.reduce(lambda a, b: max(a, b), A, identity=-9999999, axis=0)
    tmp2 = dace.reduce(lambda a, b: a + b, B, identity=0, axis=0)
    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << tmp[i]
            in2 << tmp2[i]
            out1 >> C[i]

            out1 = in1 + in2


settings = [[False, False], [True, False], [False, True]]


@pytest.mark.parametrize(["in_transient", "out_transient"], settings)
def test_p3(in_transient, out_transient):
    sdfg = reduction_test_3.to_sdfg()
    sdfg.simplify()
    state = sdfg.nodes()[0]
    A = np.random.rand(30, 20).astype(np.float64)
    B = np.random.rand(30, 20).astype(np.float64)
    C1 = np.zeros([20], dtype=np.float64)
    C2 = np.zeros([20], dtype=np.float64)
    C3 = np.zeros([20], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=20, M=30)
    del csdfg

    expand_reduce(sdfg, state, create_in_transient=in_transient, create_out_transient=out_transient)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=20, M=30)
    del csdfg

    expand_maps(sdfg, state)
    fusion(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C3, N=20, M=30)
    del csdfg

    assert np.linalg.norm(C1) > 0.01
    assert np.allclose(C1, C2)
    assert np.allclose(C1, C3)


if __name__ == "__main__":
    test_p3(False, False)
    test_p3(True, False)
    test_p3(False, True)
