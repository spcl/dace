# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def nested_cflow_test(A: dace.int32[1]):
    if A[0] == 0:
        for i in range(11):
            for j in range(5):
                for k in range(6):
                    if j < 3:
                        with dace.tasklet:
                            in_a << A[0]
                            out_a >> A[0]
                            out_a = in_a + 1
                    else:
                        with dace.tasklet:
                            in_a << A[0]
                            out_a >> A[0]
                            out_a = in_a + 2


if __name__ == '__main__':
    A = np.zeros(1).astype(np.int32)
    nested_cflow_test(A)

    expected_result = 11 * 6 * 3 + 11 * 6 * 2 * 2

    if A[0] != expected_result:
        print('ERROR: %d != %d' % (A[0], expected_result))
        exit(1)

    print('OK')
    exit(0)
