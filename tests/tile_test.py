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


@dace.program
def transpose_tiled(A: dace.float32[H, W], B: dace.float32[H, W]):
    for tile_i, tile_j in dace.map[0:H:TH, 0:W:TW]:
        for i, j in dace.map[0:TH, 0:TW]:
            with dace.tasklet:
                a << A[tile_j + j, tile_i + i]
                b >> B[tile_i + i, tile_j + j]

                b = a


def test():
    W = 128
    H = 128
    TW = 16
    TH = 16

    print('Transpose (Tiled) %dx%d (tile size: %dx%d)' % (W, H, TW, TH))

    A = dace.ndarray([H, W], dtype=dace.float32)
    B = dace.ndarray([H, W], dtype=dace.float32)
    A[:] = np.random.rand(H, W).astype(dace.float32.type)
    B[:] = dace.float32(0)

    transpose_tiled(A, B, TW=TW, TH=TH)

    diff = np.linalg.norm(np.transpose(A) - B) / (H * W)
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
