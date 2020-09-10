# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace
def testprogram(A: dace.float64[20]):
    return A + A


def test_short_decorator():
    A = np.random.rand(20)
    assert np.allclose(testprogram(A), A + A)


if __name__ == '__main__':
    test_short_decorator()
