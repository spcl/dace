# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program(dace.float32[H, W], dace.float32[W, H])
def transpose(A, B):
    @dace.map(_[0:W, 0:H])
    def compute(i, j):
        a << A[j, i]
        b >> B[i, j]

        b = a


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=64)
    parser.add_argument("H", type=int, nargs="?", default=64)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])

    print('Transpose %dx%d' % (W.get(), H.get()))

    A = np.random.rand(H.get(), W.get()).astype(np.float32)
    B = np.zeros([W.get(), H.get()], dtype=np.float32)

    transpose(A, B)

    if dace.Config.get_bool('profiling'):
        dace.timethis('transpose', 'numpy', (H.get() * W.get()), np.transpose, A)

    diff = np.linalg.norm(np.transpose(A) - B) / (H.get() * W.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
