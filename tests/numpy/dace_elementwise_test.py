# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_output():
    @dace.program
    def prog(A: dace.float64[5, 3], B: dace.float64[5, 3]):
        dace.elementwise(lambda x: log(x), A, B)

    A = np.random.rand(5, 3)
    B = np.zeros((5, 3))
    prog(A=A.copy(), B=B)

    diff = np.linalg.norm(np.log(A) - B)
    print('Difference:', diff)
    assert diff < 1e-5


def test_output_none():
    @dace.program
    def prog(A: dace.float64[5, 3], B: dace.float64[5, 3]):
        B[:] = dace.elementwise(lambda x: log(x), A)

    A = np.random.rand(5, 3)
    B = np.zeros((5, 3))
    prog(A=A.copy(), B=B)

    diff = np.linalg.norm(np.log(A) - B)
    print('Difference:', diff)
    assert diff < 1e-5


if __name__ == '__main__':
    test_output()
    test_output_none()
