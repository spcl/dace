# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def nested_loop_test(A: dace.int32[1]):
    for i in range(11):
        for j in range(5):
            with dace.tasklet:
                in_a << A[0]
                out_a >> A[0]
                out_a = in_a + 1


if __name__ == '__main__':
    A = np.zeros(1).astype(np.int32)
    nested_loop_test(A)

    if A[0] != 11 * 5:
        print('ERROR: %d != %d' % (A[0], 11 * 5))
        exit(1)

    print('OK')
    exit(0)
