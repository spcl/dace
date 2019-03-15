#!/usr/bin/env python
import dace as dp
import numpy as np

W = dp.symbol('W')


@dp.program
def indirection(A, x, B):
    @dp.map(_[0:W])
    def ind(i):
        bla << A[x[i]]
        out >> B[i]
        out = bla


if __name__ == '__main__':
    W.set(5)

    A = dp.ndarray([W * W])
    B = dp.ndarray([W])
    x = dp.ndarray([W], dtype=dp.uint32)

    A[:] = np.arange(10, 10 + W.get() * W.get())
    B[:] = dp.float32(0)
    x[:] = np.random.randint(0, W.get() * W.get(), W.get())

    indirection(A, x, B)

    print(x.view(type=np.ndarray))
    print(B.view(type=np.ndarray))
