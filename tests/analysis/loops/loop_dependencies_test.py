# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from math import sqrt
import dace
import numpy as np


def test_simple_loop_carried_dependency():
    I = dace.symbol('I')
    J = dace.symbol('J')
    K = dace.symbol('K')

    @dace.program(use_experimental_cfg_blocks=True)
    def testprog(A: dace.float64[I, J, K], B: dace.float64[I, J, K]):
        for k in range(1, K):
            for i in range(I):
                for j in range(J):
                    A[i, j, k] = B[i, j, k] + A[i, j, k - 1]
                    B[i, j, k] = A[i, j, k] + 1

    sdfg = testprog.to_sdfg()

    for state in sdfg.states():
        print()
    print(sdfg)


if __name__ == '__main__':
    test_simple_loop_carried_dependency()
