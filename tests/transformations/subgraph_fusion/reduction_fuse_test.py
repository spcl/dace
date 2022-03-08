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
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype=np.float64)
    C2 = np.zeros([N.get()], dtype=np.float64)
    C3 = np.zeros([N.get()], dtype=np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M)
    del csdfg

    expand_reduce(sdfg, state, create_in_transient=in_transient, create_out_transient=out_transient)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M)
    del csdfg

    expand_maps(sdfg, state)
    fusion(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C3, N=N, M=M)
    del csdfg

    assert np.linalg.norm(C1) > 0.01
    assert np.allclose(C1, C2)
    assert np.allclose(C1, C3)


if __name__ == "__main__":
    test_p3()
    test_p3(in_transient=True)
    test_p3(out_transient=True)
