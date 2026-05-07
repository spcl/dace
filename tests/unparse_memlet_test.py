# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def floor_div(Input: dace.float64[N], Output: dace.float64[N]):

    @dace.map(_[0:N])
    def div(i):
        inp << Input[i // 2]
        out >> Output[i]
        out = inp


def test():
    N = 25
    A = np.random.rand(N)
    B = np.zeros([N], dtype=np.float64)

    floor_div(A, B)

    if N % 2 == 0:
        expected = 2.0 * np.sum(A[0:N // 2])
    else:
        expected = 2.0 * np.sum(A[0:N // 2]) + A[N // 2]
    actual = np.sum(B)
    diff = abs(actual - expected)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
