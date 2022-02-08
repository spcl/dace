# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')
BINS = 256  # dace.symbol('BINS')


@dace.program(dace.uint8[H, W], dace.uint32[BINS])
def histogram(A, hist):
    # Declarative version
    tmp = dace.define_local([BINS, H, W], dace.uint32)

    @dace.map(_[0:BINS, 0:H, 0:W])
    def zero_tmp(b, i, j):
        t >> tmp[b, i, j]
        t = 0

    @dace.map(_[0:H, 0:W])
    def compute_declarative(i, j):
        a << A[i, j]
        out >> tmp(1)[:, i, j]
        out[a] = 1

    dace.reduce(lambda a, b: a + b, tmp, hist, axis=(1, 2), identity=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=32)
    parser.add_argument("H", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])

    print('Histogram (dec) %dx%d' % (W.get(), H.get()))

    A = np.random.randint(0, BINS, (H.get(), W.get())).astype(np.uint8)
    hist = np.zeros([BINS], dtype=np.uint32)

    histogram(A, hist)

    if dace.Config.get_bool('profiling'):
        dace.timethis('histogram', 'numpy', (H.get() * W.get()), np.histogram, A, BINS)

    diff = np.linalg.norm(np.histogram(A, bins=BINS, range=(0, BINS))[0] - hist)
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
