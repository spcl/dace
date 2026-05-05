# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def dot(A, B, out):

    @dace.map
    def product(i: _[0:N]):
        a << A[i]
        b << B[i]
        o >> out(1, lambda x, y: x + y)
        o = a * b


def test_dot():
    n = 64
    A = dace.ndarray([n], dtype=dace.float32)
    out_AA = dace.scalar(dace.float64)
    A[:] = np.random.rand(n).astype(dace.float32.type)
    out_AA[0] = dace.float64(0)

    dot(A, A, out_AA, N=n)

    diff_aa = np.linalg.norm(np.dot(A, A) - out_AA) / float(n)
    print("Difference:", diff_aa)
    assert diff_aa <= 1e-5


if __name__ == "__main__":
    test_dot()
