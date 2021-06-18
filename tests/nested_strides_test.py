# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def nested_strides_test(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    for i, j in dace.map[0:3, 0:4]:
        with dace.tasklet:
            a << A[:, i, j]
            b >> B[i, j]
            b = a[0] + a[1]


def test():
    A = np.random.rand(2, 3, 4).astype(np.float32)
    B = np.random.rand(3, 4).astype(np.float32)
    expected = A[0, :, :] + A[1, :, :]

    sdfg = nested_strides_test.to_sdfg()
    sdfg(A=A, B=B)

    diff = np.linalg.norm(expected - B)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
