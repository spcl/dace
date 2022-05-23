# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = 100


def test_numpy_where():
    @dace.program
    def numpy_where(A: dace.float64[N]):
        return np.where(A > 0.5, A, 0.0)

    for _ in range(10):
        A = np.random.randn(N)
        assert (np.allclose(numpy_where(A), np.where(A > 0.5, A, 0.0)))


if __name__ == "__main__":
    test_numpy_where()
