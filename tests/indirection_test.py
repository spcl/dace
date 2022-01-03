# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def test():
    W.set(5)

    A = dp.ndarray([W * W])
    B = dp.ndarray([W])
    x = dp.ndarray([W], dtype=dp.uint32)

    A[:] = np.arange(10, 10 + W.get() * W.get())
    B[:] = dp.float32(0)
    x[:] = np.random.randint(0, W.get() * W.get(), W.get())

    indirection(A, x, B, W=W)

    print(x)
    print(B)
    B_ref = np.array([A[x[i]] for i in range(W.get())], dtype=B.dtype)
    diff = np.linalg.norm(B - B_ref)
    assert diff == 0


if __name__ == "__main__":
    test()
