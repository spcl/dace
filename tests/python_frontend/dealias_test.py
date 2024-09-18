# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
''' Tests dealising of the SDFG produced by the Python frontend. '''

import dace
import numpy as np

size = 32


@dace.program()
def mmm_dace(X: dace.float32[size, size], Y: dace.float32[size, size], Z: dace.float32[size, size], S: dace.float32[1]):
    T: dace.float32[size, size, size] = np.zeros((size, size, size), dtype=dace.float32)

    for i in dace.map[0:size]:
        for j in dace.map[0:size]:
            for k in dace.map[0:size]:
                T[i, j, k] = X[i, k] * Y[k, j]
            Z[i, j] = np.sum(T[i, j])

    @dace.map(_[0:size, 0:size])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << Z[i, j]
        s = z


def test_simplify_mmm():
    # Input initialization
    X = np.random.rand(size, size).astype(np.float32)
    Y = np.random.rand(size, size).astype(np.float32)
    Z = np.zeros((size, size), dtype=np.float32)
    S = np.zeros((1, ), dtype=np.float32)

    sdfg = mmm_dace.to_sdfg(simplify=False)
    sdfg(X=X, Y=Y, Z=Z, S=S)

    # Numerically validate the results of the non-simplified SDFG
    assert np.allclose(Z, np.matmul(X, Y))
    assert np.allclose(S, sum(sum(np.matmul(X, Y))))

    # Simplify the SDFG
    # NOTE: If the SDFG has not been dealised properly, simplification will violate semantics.
    sdfg.simplify()

    # Input reinitialization just in case
    X = np.random.rand(size, size).astype(np.float32)
    Y = np.random.rand(size, size).astype(np.float32)
    Z = np.zeros((size, size), dtype=np.float32)
    S = np.zeros((1, ), dtype=np.float32)

    sdfg(X=X, Y=Y, Z=Z, S=S)

    # Numerically validate the results of the simplified SDFG
    assert np.allclose(Z, np.matmul(X, Y))
    assert np.allclose(S, sum(sum(np.matmul(X, Y))))


if __name__ == "__main__":
    test_simplify_mmm()
