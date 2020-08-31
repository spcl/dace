# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

N = dace.symbol('N')


@dace.program(dace.float32[N], dace.float32[N])
def cudahello(V, Vout):
    # Transient variable
    @dace.map(_[0:N])
    def multiplication(i):
        in_V << V[i]
        out >> Vout[i]
        out = in_V * 2.0


if __name__ == "__main__":
    N.set(52)

    print('Vector double CUDA %d' % (N.get()))

    V = dace.ndarray([N], dace.float32)
    Vout = dace.ndarray([N], dace.float32)
    V[:] = np.random.rand(N.get()).astype(dace.float32.type)
    Vout[:] = dace.float32(0)

    cudahello(V, Vout)

    diff = np.linalg.norm(2 * V - Vout) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
