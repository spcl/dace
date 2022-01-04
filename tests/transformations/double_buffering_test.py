# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" An example and test for the DoubleBuffering transformation. """
import dace
import numpy as np

from dace.transformation.pattern_matching import match_patterns
from dace.transformation.dataflow import DoubleBuffering, InLocalStorage


@dace.program
def mm_double_buffered(A: dace.float32[256, 256], B: dace.float32[256, 256], C: dace.float32[256, 256]):
    # Write to C in 128x128 output tiles
    for tile_i, tile_j in dace.map[0:256:128, 0:256:128]:
        # Load inputs in increments of 8 (128x8 tiles)
        for tile_k in dace.map[0:256:8]:
            # Compute outer products on input tiles
            for k, i, j in dace.map[0:8, 0:128, 0:128]:
                with dace.tasklet:
                    a << A[tile_i + i, tile_k + k]
                    b << B[tile_k + k, tile_j + j]
                    c >> C(1, lambda x, y: x + y)[tile_i + i, tile_j + j]
                    c = a * b


def test_double_buffering():
    A = np.random.rand(256, 256).astype(np.float32)
    B = np.random.rand(256, 256).astype(np.float32)
    expected_C = A @ B
    C = np.zeros((256, 256), dtype=np.float32)

    sdfg = mm_double_buffered.to_sdfg()
    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (before):', diff)

    # Apply local storage transformation on inner map (last two transformations)
    sdfg.coarsen_dataflow()
    for i in range(2):
        for match in reversed(list(match_patterns(sdfg, InLocalStorage, states=[sdfg.node(0)]))):
            match.apply(sdfg.node(0), sdfg)
            break
        else:
            raise ValueError('Local storage transformation not applied')

    applied = sdfg.apply_transformations(DoubleBuffering)
    if applied != 1:
        raise ValueError('Double-buffering transformation not applied')
    C = np.zeros((256, 256), dtype=np.float32)
    sdfg(A=A, B=B, C=C)

    diff2 = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (after):', diff2)

    assert (diff <= 1e-5 and diff2 <= 1e-5)


if __name__ == '__main__':
    test_double_buffering()
