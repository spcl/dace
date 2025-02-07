# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def noelse(A: dace.float32[1]):
    if A[0] > 0:

        @dace.tasklet
        def mytask():
            o >> A[0]
            o = 5


@dace.program
def ifchain(A: dace.float32[1]):
    if A[0] > 0:

        @dace.tasklet
        def mytask():
            o >> A[0]
            o = 0

    if A[0] > 1:

        @dace.tasklet
        def mytask():
            o >> A[0]
            o = 1

    if A[0] > 0:

        @dace.tasklet
        def mytask():
            o >> A[0]
            o = -5
    else:

        @dace.tasklet
        def mytask():
            o >> A[0]
            o = 9


def test_if_without_else():
    A = np.ndarray([1], np.float32)
    A[0] = 1
    noelse(A)
    if A[0] != 5:
        raise AssertionError("ERROR in test: %f != 5" % A[0])


def test_if_chain():
    A = np.ndarray([1], np.float32)
    A[0] = 5
    ifchain(A)
    if A[0] != 9:
        raise AssertionError("ERROR in test: %f != 9" % A[0])


if __name__ == "__main__":
    test_if_without_else()
    test_if_chain()
