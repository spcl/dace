#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import math
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')

TW = dace.symbol('TW')
TH = dace.symbol('TH')


@dace.program(dace.float32[H, W], dace.float32[H, W], dace.int32, dace.int32)
def transpose_tiled(A, B, TW, TH):
    @dace.mapscope(_[0:H:TH, 0:W:TW])
    def compute(tile_i, tile_j):
        @dace.map(_[0:TH, 0:TW])
        def compute_tile(i, j):
            a << A[tile_j + j, tile_i + i]
            b >> B[tile_i + i, tile_j + j]

            b = a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=128)
    parser.add_argument("H", type=int, nargs="?", default=128)
    parser.add_argument("TH", type=int, nargs="?", default=16)
    parser.add_argument("TW", type=int, nargs="?", default=16)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])
    TW.set(args["TW"])
    TH.set(args["TH"])

    print('Transpose (Tiled) %dx%d (tile size: %dx%d)' % (W.get(), H.get(),
                                                          TW.get(), TH.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)

    transpose_tiled(A, B, TW, TH)

    diff = np.linalg.norm(np.transpose(A) - B) / (H.get() * W.get())
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
