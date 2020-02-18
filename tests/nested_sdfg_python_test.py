#!/usr/bin/env python
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


if __name__ == '__main__':
    print('Nested SDFG test (Python syntax)')
    # Externals (parameters, symbols)
    N.set(64)

    input = np.random.rand(N.get(), N.get()).astype(dp.float32.type)
    output = np.zeros((N.get(), N.get()), dp.float32.type)

    sdfg = sdfg_with_children.to_sdfg()
    sdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N.get() * N.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
