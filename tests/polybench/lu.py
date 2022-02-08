# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import numpy as np
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype)]


def init_array(A):
    n = N.get()

    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            # Python does modulo, while C does remainder ...
            A[i, j] = datatype(-(j % n)) / n + 1
        for j in range(i + 1, n, 1):
            A[i, j] = datatype(0)
        A[i, i] = datatype(1)

    A[:] = np.dot(A, np.transpose(A))


@dace.program(datatype[N, N])
def lu(A):
    for i in range(0, N, 1):
        for j in range(0, i, 1):

            @dace.map
            def k_loop1(k: _[0:j]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> A(1, lambda x, y: x + y)[i, j]
                out = -i_in * j_in

            @dace.tasklet
            def div():
                ij_in << A[i, j]
                jj_in << A[j, j]
                out >> A[i, j]
                out = ij_in / jj_in

        for j in range(i, N, 1):

            @dace.map
            def k_loop2(k: _[0:i]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> A(1, lambda x, y: x + y)[i, j]
                out = -i_in * j_in


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, lu)
