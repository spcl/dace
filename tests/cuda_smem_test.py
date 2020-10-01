# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def cudahello(A, Vout):
    @dace.mapscope(_[0:ceiling(N / 32)])
    def multiplication(i):
        @dace.map(_[i * 32:min(N, (i + 1) * 32)])
        def mult_block(bi):
            in_V << A[bi]
            out >> Vout[bi]
            out = in_V * 2.0


if __name__ == "__main__":
    N.set(144)

    print('Vector double CUDA (shared memory) %d' % (N.get()))

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)
    V[:] = np.random.rand(N.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    cudahello(V, Vout)

    diff = np.linalg.norm(2 * V - Vout) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
