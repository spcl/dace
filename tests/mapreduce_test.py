# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program(dace.float32[H, W], dace.float32[H, W], dace.float32[1])
def mapreduce_test(A, B, sum):
    tmp = dace.define_local([H, W], dace.float32)

    @dace.map(_[0:H, 0:W])
    def compute_tile(i, j):
        a << A[i, j]
        b >> B[i, j]
        t >> tmp[i, j]

        b = a * 5
        t = a * 5

    sum[:] = dace.reduce(lambda a, b: a + b, tmp, identity=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=128)
    parser.add_argument("H", type=int, nargs="?", default=128)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])

    print('Map-Reduce Test %dx%d' % (W.get(), H.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    res = dace.ndarray([1], dtype=dace.float32)
    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)
    res[:] = dace.float32(0)

    mapreduce_test(A, B, res)

    diff = np.linalg.norm(5 * A - B) / np.linalg.norm(5 * A)
    diff_res = np.linalg.norm(np.sum(B) - res[0]) / np.linalg.norm(np.sum(B))
    # diff_res = abs((np.sum(B) - res[0])).view(type=np.ndarray)
    print("Difference:", diff, diff_res)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 and diff_res <= 1 else 1)
