# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

N = dace.symbol('N')


@dace.program
def dot(A, B, out):
    @dace.map
    def product(i: _[0:N]):
        a << A[i]
        b << B[i]
        o >> out(1, lambda x, y: x + y)
        o = a * b


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=64)
    args = vars(parser.parse_args())

    N.set(args["N"])

    print('Dot product %d' % (N.get()))

    A = dace.ndarray([N], dtype=dace.float32)
    out_AA = dace.scalar(dace.float64)
    A[:] = np.random.rand(N.get()).astype(dace.float32.type)
    out_AA[0] = dace.float64(0)

    dot(A, A, out_AA, N=N)

    diff_aa = np.linalg.norm(np.dot(A, A) - out_AA) / float(N.get())
    print("Difference:", diff_aa)
    exit(0 if (diff_aa <= 1e-5) else 1)
