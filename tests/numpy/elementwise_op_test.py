# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def optest(A: dace.float64[5, 5], B: dace.float64[5, 5],
           C: dace.float64[5, 5]):
    tmp = (-A) * B
    for i, j in dace.map[0:5, 0:5]:
        with dace.tasklet:
            t << tmp[i, j]
            c >> C[i, j]
            c = t


if __name__ == '__main__':
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)
    C = np.random.rand(5, 5)

    optest(A, B, C)
    diff = np.linalg.norm(C - ((-A) * B))
    print('Difference:', diff)
    if diff > 1e-5:
        exit(1)
