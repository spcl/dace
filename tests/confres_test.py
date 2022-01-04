# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program(dace.float32[W, H], dace.float32[H, W, H], dace.float32[3], dace.float32[1])
def confres_test(A, B, red1, red2):
    @dace.map(_[0:H - 1, 0:W - 1])
    def compute(i, j):
        a << A[j, i]
        b >> B[i, j, 0]
        c >> A[j + 1, i + 1]
        r1 >> red1(1, lambda x, y: x * y)[1]
        r2 >> red2(1, lambda x, y: x + y)[:]

        b = a
        c = 5 * b
        r1 = 1
        r2 = 2

    dace.reduce(lambda a, b: a + b, A, red2)
    red1[0:1] = dace.reduce(lambda a, b: a + b, B[2:H - 2, 5])
    red1[1:] = dace.reduce(lambda a, b: a + b, B[3:H - 3, 5:7, :], axis=(2, 0))
    red1[0:1] = dace.reduce(lambda a, b: a - b, B[2:H - 2, 5, :])


def test():
    W.set(20)
    H.set(20)

    print('Conflict Resolution Test %dx%d' % (W.get(), H.get()))

    A = dace.ndarray([W, H], dtype=dace.float32)
    B = dace.ndarray([H, W, H], dtype=dace.float32)
    red1 = dace.ndarray([3], dtype=dace.float32)
    red2 = dace.ndarray([1], dtype=dace.float32)

    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = np.random.rand(H.get(), W.get(), H.get()).astype(dace.float32.type)
    red1[:] = dace.float32(0)
    red2[:] = dace.float32(0)

    confres_test.compile(A, B, red1, red2)


if __name__ == "__main__":
    test()
