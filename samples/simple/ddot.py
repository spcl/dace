# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import numpy as np

N = dace.symbol("N")


@dace.program
def dot(A: dace.float32[N], B: dace.float32[N], out: dace.float64[1]):
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

    # Create two numpy ndarrays of size 1
    out_AB = dace.scalar(dace.float64)
    out_AA = dace.scalar(dace.float64)

    N.set(args["N"])

    print('Dot product %d' % (N.get()))

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)
    out_AB[0] = np.float64(0)
    out_AA[0] = np.float64(0)

    cdot = dot.compile(A, B, out_AB)
    cdot(A=A, B=B, out=out_AB, N=N)

    # To allow reloading the SDFG code file with the same name
    del cdot

    cdot_self = dot.compile(A, A, out_AA)
    cdot_self(A=A, B=A, out=out_AA, N=N)

    diff_ab = np.linalg.norm(np.dot(A, B) - out_AB) / float(N.get())
    diff_aa = np.linalg.norm(np.dot(A, A) - out_AA) / float(N.get())
    print("Difference (A*B):", diff_ab)
    print("Difference (A*A):", diff_aa)
    exit(0 if (diff_ab <= 1e-5 and diff_aa <= 1e-5) else 1)
