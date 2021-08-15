# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
from sympy.core.numbers import comp
import dace
from common import compare_numpy_output


@compare_numpy_output()
def test_flip_1d(A: dace.int32[10]):
    return np.flip(A)


if __name__ == '__main__':
    test_flip_1d()
