# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.graph import SubgraphView
import dace.transformation.subgraph.helpers as helpers
from typing import List, Union
import dace.sdfg.nodes as nodes
import numpy as np
from util import fusion

N = dace.symbol('N')


@dace.program
def sequential2(A: dace.float64[N], C: dace.float64[N]):
    B = np.ndarray(shape=[N], dtype=np.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> B[i]
            out1 = in1 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << B[i]
            out1 >> C[i]
            out1 = in1 + 1


def test_sequential():
    N = 1000

    sdfg = sequential2.to_sdfg()
    state = sdfg.nodes()[0]

    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C1 = np.random.rand(N).astype(np.float64)
    C2 = np.random.rand(N).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N)
    del csdfg

    fusion(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N)

    assert np.allclose(C1, C2)


if __name__ == "__main__":
    test_sequential()
