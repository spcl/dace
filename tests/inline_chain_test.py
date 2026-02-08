# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def inline_chain_test(A: dace.float64[20, 20]):
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


def test():
    A = np.random.rand(20, 20).astype(np.float64)
    refA = np.ndarray([20, 20], dtype=np.float64)
    refA[:] = A

    inline_chain_test(A)

    diff = np.linalg.norm(A - (refA + 2)) / 400
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
