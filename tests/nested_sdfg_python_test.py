# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

N = dp.symbol('N')


@dp.program
def sdfg_with_children(A: dp.float32[N, N], B: dp.float32[N, N]):

    @dp.mapscope
    def elements(i: _[0:N], j: _[0:N]):

        @dp.tasklet
        def init():
            inp << A[i, j]
            out >> B[i, j]
            out = inp

        for k in range(4):

            @dp.tasklet
            def do():
                inp << A[i, j]
                oin << B[i, j]
                out >> B[i, j]
                out = oin * inp


def test():
    print('Nested SDFG test (Python syntax)')
    # Externals (parameters, symbols)
    N = 64

    input = np.random.rand(N, N).astype(dp.float32.type)
    output = np.zeros((N, N), dp.float32.type)

    sdfg = sdfg_with_children.to_sdfg()
    sdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N * N)
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
