# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def floor_div(Input, Output):
    @dace.map(_[0:N])
    def div(i):
        inp << Input[i // 2]
        out >> Output[i]
        out = inp


def test():
    N.set(25)
    A = np.random.rand(N.get())
    B = np.zeros([N.get()], dtype=np.float64)

    floor_div(A, B)

    if N.get() % 2 == 0:
        expected = 2.0 * np.sum(A[0:N.get() // 2])
    else:
        expected = 2.0 * np.sum(A[0:N.get() // 2]) + A[N.get() // 2]
    actual = np.sum(B)
    diff = abs(actual - expected)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
