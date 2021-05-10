# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace
def short_decorator(A: dace.float64[20]):
    return A + A


def test_short_decorator():
    A = np.random.rand(20)
    assert np.allclose(short_decorator(A), A + A)


if __name__ == '__main__':
    test_short_decorator()
