#!/usr/bin/env python
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
    tmp = dace.define_local([H, W, BINS], dace.uint32)

    @dace.map(_[0:H, 0:W, 0:BINS])
    def zero_tmp(i, j, b):
        t >> tmp[i, j, b]
        t = 0

    @dace.map(_[0:H, 0:W])
    def compute_declarative(i, j):
        a << A[i, j]
        out >> tmp(1)[i, j, :]
        out[a] = 1

    dace.reduce(lambda a, b: a + b, tmp, hist, axis=(0, 1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=32)
    parser.add_argument("H", type=int, nargs="?", default=32)
    args = vars(parser.parse_args())

    A = dace.ndarray([H, W], dtype=dace.uint8)
    hist = dace.ndarray([BINS], dtype=dace.uint32)

    W.set(args["W"])
    H.set(args["H"])

    print('Histogram (dec) %dx%d' % (W.get(), H.get()))

    A[:] = np.random.randint(0, 256,
                             (H.get(), W.get())).astype(dace.uint8.type)
    hist[:] = dace.uint32(0)

    histogram(A, hist)

    if dace.Config.get_bool('profiling'):
        dace.timethis('histogram', 'numpy', dace.eval(H * W), np.histogram, A,
                      BINS)

    diff = np.linalg.norm(np.histogram(A, bins=BINS)[0] - hist)
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
