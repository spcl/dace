# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def plus_1(X_in: dace.float32[N], num: dace.int32[1], X_out: dace.float32[N]):
    @dace.map
    def p1(i: _[0:num[0]]):
        x_in << X_in[i]
        x_out >> X_out[i]

        x_out = x_in + 1


def test():
    X = np.random.rand(10).astype(np.float32)
    Y = np.zeros(10).astype(np.float32)
    num = np.zeros(1).astype(np.int32)
    num[0] = 7

    plus_1(X_in=X, num=num, X_out=Y, N=10)

    diff = np.linalg.norm((X[0:num[0]] + 1) - Y[0:num[0]])
    if any(abs(y - 0.0) > 1e-5 for y in Y[num[0]:]) or diff > 1e-5:
        print('Y =', Y)
        raise AssertionError


if __name__ == "__main__":
    test()
