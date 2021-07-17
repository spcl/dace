# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')


@dace.program
def my_assign(X_in: dace.float32[N], X_out: dace.float32[N]):
    X_out[:] = X_in[:]


@dace.program
def my_func(a: dace.float32[M, K], b: dace.float32[M, K]):

    for j in dace.map[0:M]:
        my_assign(b[j], a[j])


def test():
    A = np.random.rand(3, 4).astype(np.float32)
    B = np.random.rand(3, 4).astype(np.float32)

    my_func(a=A, b=B, M=B.shape[0], N=B.shape[1], K=B.shape[1])


if __name__ == '__main__':
    test()
