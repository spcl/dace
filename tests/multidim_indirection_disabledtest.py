# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np

W = dp.symbol('W')


@dp.program
def indirection(A: dp.float32[W, W, W], x: dp.int32[W], y: dp.int32[W],
                B: dp.float32[W, W, W]):
    @dp.map(_[0:W, 0:W, 0:W])
    def ind(i, j, k):
        # evaluates to A[i,x[j]+1,y[k]/2]
        inp << A[i, x[j]:x[j + 1],
                 y[k] / 2]  # yapf: disable
        out >> B[i, j, k]
        out = inp


if __name__ == '__main__':
    W.set(5)

    A = dp.ndarray([W, W, W], dtype=dp.float32)
    B = dp.ndarray([W, W, W], dtype=dp.float32)
    x = dp.ndarray([W], dtype=dp.int32)
    y = dp.ndarray([W], dtype=dp.int32)

    A[:] = np.mgrid[0:W.get(), 0:W.get(), 0:W.get()][0].astype(dp.float32.type)
    B[:] = dp.float32(0)

    x[:] = np.random.randint(0, W.get(), W.get())
    x -= 1

    y[:] = np.random.randint(0, W.get(), W.get())

    indirection(A, x, y, B)

    print(x.view(type=np.ndarray))
    print(y.view(type=np.ndarray))
    print(B.view(type=np.ndarray))
