# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import MapExpansion
import numpy as np


@dace.program
def expansion(A: dace.float32[20, 30, 5], rng: dace.int32[2]):
    @dace.map
    def mymap(i: _[0:20], j: _[rng[0]:rng[1]], k: _[0:5]):
        a << A[i, j, k]
        b >> A[i, j, k]
        b = a * 2


def test():
    A = np.random.rand(20, 30, 5).astype(np.float32)
    b = np.array([5, 10], dtype=np.int32)
    expected = A.copy()
    expected[:, 5:10, :] *= 2

    sdfg = expansion.to_sdfg()
    sdfg(A=A, rng=b)
    diff = np.linalg.norm(A - expected)
    print('Difference (before transformation):', diff)

    sdfg.apply_transformations(MapExpansion)

    sdfg(A=A, rng=b)
    expected[:, 5:10, :] *= 2
    diff2 = np.linalg.norm(A - expected)
    print('Difference:', diff2)
    assert (diff <= 1e-5) and (diff2 <= 1e-5)


if __name__ == "__main__":
    test()
