# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np

W = dp.symbol('W')


@dp.program
def prog(A, B, integer):
    @dp.map(_[0:W])
    def compute(i):
        a << A[i]
        b >> B[i]
        b = a * integer


def test():
    W.set(3)

    A = dp.ndarray([W])
    B = dp.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = dp.float32(0.0)

    prog(A, B, 5, W=W)

    diff = np.linalg.norm(5 * A - B) / W.get()
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
