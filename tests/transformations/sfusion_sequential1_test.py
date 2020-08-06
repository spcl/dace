
import dace
from dace.transformation.subgraph.pipeline import expand_maps, expand_reduce, fusion
from dace.transformation.subgraph.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np



N = dace.symbol('N')


@dace.program
def TEST(A: dace.float64[N], B:dace.float64[N],
          C: dace.float64[N]):

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


if __name__ == "__main__":
    N.set(1000)

    sdfg = TEST.to_sdfg()
    state = sdfg.nodes()[0]

    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(N.get()).astype(np.float64)
    C1 = np.random.rand(N.get()).astype(np.float64)
    C2 = np.random.rand(N.get()).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C1,N=N)

    fusion(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C2,N=N)

    assert np.allclose(C1, C2)
