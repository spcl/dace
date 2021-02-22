# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

M, N, K = (dace.symbol(name) for name in ['M', 'N', 'K'])


@dace.program
def ttest(A: dace.float32[M, N, K], B: dace.float32[M, N, K]):
    s = np.ndarray(shape=(K, N, M), dtype=np.int32)
    t = np.ndarray(A.shape, A.dtype)

    for i in dace.map[0:M]:
        for j in dace.map[0:N]:
            for k in dace.map[0:K]:
                s[k, j, i] = t[i, j, k]
                t[i, j, k] = 1.0
                s[k, j, i] = t[i, j, k]

    t += 5 * A
    B -= t


def test():
    M.set(13)
    N.set(8)
    K.set(25)
    A = np.random.rand(M.get(), N.get(), K.get()).astype(np.float32)
    B = np.random.rand(M.get(), N.get(), K.get()).astype(np.float32)

    realB = B - 5 * A - 1.0
    ttest(A, B)

    diff = np.linalg.norm(B - realB) / (M.get() * K.get() * N.get())
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test()
