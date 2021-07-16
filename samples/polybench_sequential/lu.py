# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    for i in range(N):
        for j in range(i):
            for k in range(j):
                with dace.tasklet:
                    Aij << A[i, j]
                    Aik << A[i, k]
                    Akj << A[k, j]
                    A_out >> A[i, j]
                    A_out = Aij - Aik * Akj
                # A[i, j] = A[i, j] - A[i, k] * A[k, j]
            with dace.tasklet:
                Aij << A[i, j]
                Ajj << A[j, j]
                A_out >> A[i, j]
                A_out = Aij / Ajj
            # A[i, j] = A[i, j] / A[j, j]
        for j in range(i, N, 1):
            for k in range(i):
                with dace.tasklet:
                    Aij << A[i, j]
                    Aik << A[i, k]
                    Akj << A[k, j]
                    A_out >> A[i, j]
                    A_out = Aij - Aik * Akj
                # A[i, j] = A[i, j] - A[i, k] * A[k, j]


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, lu)
