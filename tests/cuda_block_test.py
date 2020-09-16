# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def cudahello(V, Vout):
    # Transient variable
    @dace.mapscope(_[0:N:32])
    def multiplication(i):
        @dace.map(_[0:32])
        def mult_block(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2

        @dace.map(_[0:32])
        def mult_block_2(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2


if __name__ == "__main__":
    N.set(128)

    print('Vector double CUDA (block) %d' % (N.get()))

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)
    V[:] = np.random.rand(N.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    cudahello(V, Vout)

    diff = np.linalg.norm(2 * V - Vout) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
