# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

H = dace.symbol('H')
W = dace.symbol('W')


@dace.program(dace.float64[H, W], dace.float64[H, W])
def cudahello(V, Vout):
    @dace.mapscope(_[0:H:8, 0:W:32])
    def multiplication(i, j):
        @dace.map(_[0:8, 0:32])
        def mult_block(bi, bj):
            in_V << V[i + bi, j + bj]
            out >> Vout[i + bi, j + bj]
            out = in_V * 2.0


if __name__ == "__main__":
    W.set(128)
    H.set(64)

    print('Vector double CUDA (shared memory 2D) %dx%d' % (W.get(), H.get()))

    V = dace.ndarray([H, W], dace.float64)
    Vout = dace.ndarray([H, W], dace.float64)
    V[:] = np.random.rand(H.get(), W.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    cudahello(V, Vout)

    diff = np.linalg.norm(2 * V - Vout) / (H.get() * W.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
