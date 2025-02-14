# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import sys
import os

# Ensure files from the same directory can be imported (for pytest)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import external_module

W = external_module.W
H = external_module.H


@dace.program
def extmodtest(A: dace.float32[W, H], result: dace.float32[1]):
    tmp = np.ndarray([H, W], dace.float32)

    external_module.transpose(A, tmp)

    with dace.tasklet:
        a << tmp[1, 2]
        b >> result[0]

        b = a


def test():
    W = 12
    H = 12
    A = np.random.rand(W, H).astype(np.float32)
    res = np.zeros([1], np.float32)

    extmodtest(A, res)

    assert res[0] == A[2, 1]


if __name__ == "__main__":
    test()
