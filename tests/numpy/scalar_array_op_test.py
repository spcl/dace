# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def saoptest(A: dace.float64[5, 5], alpha: dace.float64, B: dace.float64[5, 5]):
    tmp = alpha * A * 5
    for i, j in dace.map[0:5, 0:5]:
        with dace.tasklet:
            t << tmp[i, j]
            c >> B[i, j]
            c = t


def test():
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)

    saoptest(A, 10, B)
    diff = np.linalg.norm(B - (50 * A))
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test()
