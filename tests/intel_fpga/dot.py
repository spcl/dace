# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Dot product with WCR
# Used as simple test for WCR over scalar

#!/usr/bin/env python

from __future__ import print_function

import argparse
import dace
import numpy as np

N = dace.symbol("N")


@dace.program
def dot(A: dace.float32[N], B: dace.float32[N], out: dace.float32[1]):
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
    A = dace.ndarray([N], dtype=dace.float32)
    B = dace.ndarray([N], dtype=dace.float32)
    out_AB = dace.scalar(dace.float32)

    print('Dot product %d' % (N.get()))

    A[:] = np.random.rand(N.get()).astype(dace.float32.type)
    B[:] = np.random.rand(N.get()).astype(dace.float32.type)
    out_AB[0] = dace.float32(0)
    dot(A, B, out_AB)

    diff_ab = np.linalg.norm(np.dot(A, B) - out_AB) / float(N.get())
    print("Difference (A*B):", diff_ab)
    exit(0 if (diff_ab <= 1e-5) else 1)
