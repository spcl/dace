# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def test():
    W.set(128)
    H.set(128)
    TW.set(16)
    TH.set(16)

    print('Transpose (Tiled) %dx%d (tile size: %dx%d)' %
          (W.get(), H.get(), TW.get(), TH.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    A[:] = np.random.rand(H.get(), W.get()).astype(dace.float32.type)
    B[:] = dace.float32(0)

    transpose_tiled(A, B, TW, TH)

    diff = np.linalg.norm(np.transpose(A) - B) / (H.get() * W.get())
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
