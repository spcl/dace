
import dace
from dace.transformation.heterogeneous import MultiExpansion
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous.helpers import *
from dace.measure import Runner
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

    runner = Runner()
    runner.go(sdfg, state, None, N,
              output = ['B','C'])
