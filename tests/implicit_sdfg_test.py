# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def computation(a, b):
    b[:] = a * 5


def test_implicit():
    A = np.random.rand(20, 30)
    B = np.random.rand(20, 30)
    computation(A, B)
    assert np.allclose(B, A * 5)


if __name__ == '__main__':
    test_implicit()
