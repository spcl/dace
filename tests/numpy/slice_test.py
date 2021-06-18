# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def slicetest(A: dace.float64[N, N - 1], B: dace.float64[N - 1, N],
              C: dace.float64[N - 1, N - 1]):
    tmp = A[1:N] * B[:, 0:N - 1]
    for i, j in dace.map[0:4, 0:4]:
        with dace.tasklet:
            t << tmp[i, j]
            c >> C[i, j]
            c = t


def test():
    A = np.random.rand(5, 4)
    B = np.random.rand(4, 5)
    C = np.random.rand(4, 4)
    N.set(5)

    slicetest(A, B, C)
    diff = np.linalg.norm(C - (A[1:N.get()] * B[:, 0:N.get() - 1]))
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test()
