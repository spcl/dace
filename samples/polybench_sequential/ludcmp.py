# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench
import numpy as np

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype), ([N], datatype), ([N], datatype), ([N], datatype)]


def init_array(A, b, x, y):
    n = N.get()

    x[:] = datatype(0)
    y[:] = datatype(0)

    for i in range(0, n, 1):
        b[i] = datatype(i + 1) / datatype(n) / 2.0 + 4

    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            # Python does modulo, while C does remainder ...
            A[i, j] = datatype(-(j % n)) / n + 1
        for j in range(i + 1, n, 1):
            A[i, j] = datatype(0)
        A[i, i] = datatype(1)

    A[:] = np.dot(A, np.transpose(A))


@dace.program(datatype[N, N], datatype[N], datatype[N], datatype[N])
def ludcmp(A, b, x, y):
    w = dace.define_local([1], datatype)
    for i in range(N):
        for j in range(i):
            w = A[i, j]
            for k in range(j):
                with dace.tasklet:
                    i_in << A[i, k]
                    j_in << A[k, j]
                    w_in << w
                    w_out >> w
                    w_out = w_in - i_in * j_in
            with dace.tasklet:
                a_in << A[j, j]
                w_in << w
                a_out >> A[i, j]
                a_out = w_in / a_in
        for j in range(i, N):
            w = A[i, j]
            for k in range(i):
                with dace.tasklet:
                    i_in << A[i, k]
                    j_in << A[k, j]
                    w_in << w
                    w_out >> w
                    w_out = w_in - i_in * j_in
            A[i, j] = w

    for i in range(N):
        w = b[i]
        for j in range(i):
            with dace.tasklet:
                a_in << A[i, j]
                y_in << y[j]
                w_in << w
                w_out >> w
                w_out = w_in - a_in * y_in
        y[i] = w

    for i in range(N - 1, -1, -1):
        w = y[i]
        for j in range(i+1, N):
            with dace.tasklet:
                a_in << A[i, j]
                x_in << x[j]
                w_in << w
                w_out >> w
                w_out = w_in - a_in * x_in
        with dace.tasklet:
            a_in << A[i, i]
            w_in << w
            x_out >> x[i]
            x_out = w_in / a_in

if __name__ == '__main__':
    polybench.main(sizes, args, [(2, 'x')], init_array, ludcmp)
