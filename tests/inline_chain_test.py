# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program(dace.float64[20, 20])
def inline_chain_test(A):
    tmp = dace.define_local([20, 20], dtype=dace.float64)
    tmp[:] = 0

    for i in dace.map[0:20]:
        for j in dace.map[0:20]:
            with dace.tasklet:
                inA << A[i, j]
                out >> tmp[i, j]
                out = inA + 1

        for j in dace.map[0:20]:
            with dace.tasklet:
                intmp << tmp[i, j]
                out >> A[i, j]
                out = intmp + 1


if __name__ == '__main__':
    A = np.random.rand(20, 20).astype(np.float64)
    refA = np.ndarray([20, 20], dtype=np.float64)
    refA[:] = A

    inline_chain_test(A)

    diff = np.linalg.norm(A - (refA + 2)) / 400
    print('Difference:', diff)
    exit(1 if diff > 1e-5 else 0)
