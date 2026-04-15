# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test():
    """Strided range copy tasklet test"""
    sr = dace.SDFG('stiledcopy')
    sr.add_array('A', [2, 16, 4], dace.float32)
    sr.add_array('B', [4], dace.float32)
    sr.add_array('C', [2, 16, 4], dace.float32)

    sr.add_array('D', [128, 128], dace.float32)
    sr.add_array('E', [8, 8], dace.float32)
    sr.add_array('F', [128, 128], dace.float32)

    s0 = sr.add_state('s0')
    A = s0.add_access('A')
    B = s0.add_access('B')
    C = s0.add_access('C')

    D = s0.add_access('D')
    E = s0.add_access('E')
    F = s0.add_access('F')

    # Reading A at [1, 0:8:8:2, 3]
    s0.add_nedge(A, B, dace.Memlet.simple(A, '1, 0:10:8:2, 3'))
    s0.add_nedge(B, C, dace.Memlet.simple(C, '1, 0:10:8:2, 3'))

    # Emulate a blocked tiled matrix multiplication pattern
    s0.add_nedge(D, E, dace.Memlet.simple(D, '8:76:64:4,4:72:64:4'))
    s0.add_nedge(E, F, dace.Memlet.simple(F, '8:76:64:4,4:72:64:4'))

    A = np.random.rand(2, 16, 4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    C = np.random.rand(2, 16, 4).astype(np.float32)
    D = np.random.rand(128, 128).astype(np.float32)
    E = np.random.rand(8, 8).astype(np.float32)
    F = np.random.rand(128, 128).astype(np.float32)

    sr(A=A, B=B, C=C, D=D, E=E, F=F)

    diffs = [
        B[0:2] - A[1, 0:2, 3],
        B[2:4] - A[1, 8:10, 3],
        B[0:2] - C[1, 0:2, 3],
        B[2:4] - C[1, 8:10, 3],
        E[0:4, 0:4] - D[8:12, 4:8],
        E[0:4, 4:8] - D[8:12, 68:72],
        E[4:8, 0:4] - D[72:76, 4:8],
        E[4:8, 4:8] - D[72:76, 68:72],
        E[0:4, 0:4] - F[8:12, 4:8],
        E[0:4, 4:8] - F[8:12, 68:72],
        E[4:8, 0:4] - F[72:76, 4:8],
        E[4:8, 4:8] - F[72:76, 68:72],
    ]
    diff_array = [np.linalg.norm(d) for d in diffs]
    assert np.average(np.array(diff_array)) <= 1e-5


if __name__ == "__main__":
    test()
