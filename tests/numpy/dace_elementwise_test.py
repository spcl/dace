# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_output():

    @dace.program
    def elementwise(A: dace.float64[5, 3], B: dace.float64[5, 3]):
        dace.elementwise(lambda x: log(x), A, B)

    A = np.random.rand(5, 3)
    B = np.zeros((5, 3))
    elementwise(A=A.copy(), B=B)

    diff = np.linalg.norm(np.log(A) - B)
    print('Difference:', diff)
    assert diff < 1e-5


def test_output_none():

    @dace.program
    def elementise_none(A: dace.float64[5, 3], B: dace.float64[5, 3]):
        B[:] = dace.elementwise(lambda x: log(x), A)

    A = np.random.rand(5, 3)
    B = np.zeros((5, 3))
    elementise_none(A=A.copy(), B=B)

    diff = np.linalg.norm(np.log(A) - B)
    print('Difference:', diff)
    assert diff < 1e-5


def test_cast():

    @dace.program
    def elementwise_cast(A: dace.float32[5, 3], B: dace.float64[5, 3]):
        dace.elementwise(lambda x: x, A, B)

    A = np.random.rand(5, 3).astype(np.float32)
    B = np.zeros((5, 3)).astype(np.float64)
    elementwise_cast(A=A.copy(), B=B)

    diff = np.linalg.norm(A - B)
    print('Difference:', diff)
    assert diff < 1e-5


if __name__ == '__main__':
    test_output()
    test_output_none()
    test_cast()
