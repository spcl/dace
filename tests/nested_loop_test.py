# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def test():
    A = np.zeros(1).astype(np.int32)
    nested_loop_test(A)

    assert A[0] == 11 * 5


if __name__ == "__main__":
    test()
