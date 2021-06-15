# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


@compare_numpy_output(non_zero=True, positive=True)
def test_uadd(A: dace.int64[5, 5]):
    return +A


@compare_numpy_output(non_zero=True, positive=True)
def test_usub(A: dace.int64[5, 5]):
    return -A


@compare_numpy_output(non_zero=True, positive=True)
def test_invert(A: dace.int64[5, 5]):
    return ~A


@dace.program
def nottest(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    B[:] = not A


def test_not():
    A = np.random.randint(0, 2, size=(5, 5)).astype(np.bool_)
    B = np.zeros((5, 5), dtype=np.int64).astype(np.bool_)
    regression = np.logical_not(A)
    nottest(A, B)
    assert np.alltrue(B == regression)


if __name__ == '__main__':
    test_uadd()
    test_usub()
    test_not()
    test_invert()
