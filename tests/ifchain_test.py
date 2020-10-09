# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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


if __name__ == '__main__':
    print('If without else test')
    A = np.ndarray([1], np.float32)
    A[0] = 1
    noelse(A)
    if A[0] != 5:
        print("ERROR in test: %f != 5" % A[0])
        exit(1)
    print('If chain test')
    ifchain(A)
    if A[0] != 9:
        print("ERROR in test: %f != 9" % A[0])
        exit(1)
    print("Success!")
    exit(0)
