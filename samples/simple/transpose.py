#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program(dace.float32[H, W], dace.float32[H, W])
def transpose(A, B):
    @dace.map(_[0:H, 0:W])
    def compute(i, j):
        a << A[j, i]
        b >> B[i, j]

        b = a


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=64)
    parser.add_argument("H", type=int, nargs="?", default=64)
    args = vars(parser.parse_args())

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)

    W.set(args["W"])
    H.set(args["H"])

    print('Transpose %dx%d' % (W.get(), H.get()))

    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)

    transpose(A, B)

    if dace.Config.get_bool('profiling'):
        dace.timethis('transpose', 'numpy', dace.eval(H * W), np.transpose, A)

    diff = np.linalg.norm(np.transpose(A) - B) / float(dace.eval(H * W))
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
