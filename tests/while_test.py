# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace


@dace.program
def whiletest(A: dace.int32[1]):
    while A[0] > 0:
        with dace.tasklet:
            a << A[0]
            b >> A[0]
            b = a - 1


if __name__ == '__main__':
    A = dace.ndarray([1], dace.int32)
    A[0] = 5

    whiletest(A)

    if A[0] != 0:
        print('FAIL')
        exit(1)
    print('PASSED')
