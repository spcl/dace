""" An example and test for the DoubleBuffering transformation. """
import dace
import numpy as np

from dace.transformation.dataflow import DoubleBuffering


@dace.program
def mm_double_buffered(A: dace.float32[256, 256], B: dace.float32[256, 256],
                       C: dace.float32[256, 256]):
    # Write to C in 128x128 output tiles
    for tile_i, tile_j in dace.map[0:256:128, 0:256:128]:
        # Load inputs in increments of 8 (128x8 tiles)
        for tile_k in dace.map[0:256:8]:
            sA = np.ndarray([128, 8], dtype=A.dtype)
            sB = np.ndarray([8, 128], dtype=B.dtype)
            A[tile_i:tile_i + 128, tile_k:tile_k + 8] >> sA[0:128, 0:8]
            B[tile_k:tile_k + 8, tile_j:tile_j + 128] >> sB[0:8, 0:128]

            # Compute outer products on input tiles
            for k, i, j in dace.map[0:8, 0:128, 0:128]:
                with dace.tasklet:
                    a << sA[i, k]
                    b << sB[k, j]
                    c >> C(1, lambda x, y: x + y)[tile_i + i, tile_j + j]
                    c = a * b


if __name__ == '__main__':
    A = np.random.rand(256, 256).astype(np.float32)
    B = np.random.rand(256, 256).astype(np.float32)
    expected_C = A @ B
    C = np.zeros((256, 256), dtype=np.float32)

    sdfg = mm_double_buffered.to_sdfg()
    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (before):', diff)

    sdfg.apply_transformations(DoubleBuffering)
    C = np.zeros((256, 256), dtype=np.float32)
    sdfg(A=A, B=B, C=C)

    diff2 = np.linalg.norm(expected_C - C) / (256 * 256)
    print('Difference (after):', diff2)

    exit(1 if (diff > 1e-5 or diff2 > 1e-5) else 0)
