# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
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

    for i in range(0, N, 1):
        for j in range(0, i, 1):

            @dace.tasklet
            def init_w1():
                in_A << A[i, j]
                out >> w
                out = in_A

            @dace.map
            def k_loop1(k: _[0:j]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> w(1, lambda x, y: x + y)
                out = -i_in * j_in

            @dace.tasklet
            def div1():
                jj_in << A[j, j]
                in_w << w
                out >> A[i, j]
                out = in_w / jj_in

        for j in range(i, N, 1):

            @dace.tasklet
            def set_w2():
                in_A << A[i, j]
                out >> w
                out = in_A

            @dace.map
            def k_loop2(k: _[0:i]):
                i_in << A[i, k]
                j_in << A[k, j]
                out >> w(1, lambda x, y: x + y)
                out = -i_in * j_in

            @dace.tasklet
            def set_a2():
                in_w << w
                out >> A[i, j]
                out = in_w

    for i in range(0, N, 1):

        @dace.tasklet
        def init_w3():
            in_b << b[i]
            out >> w
            out = in_b

        @dace.map
        def set_w3(j: _[0:i]):
            in_A << A[i, j]
            in_y << y[j]
            out >> w(1, lambda x, y: x + y)
            out = -in_A * in_y

        @dace.tasklet
        def set_y3():
            in_w << w
            out >> y[i]
            out = in_w

    for i in range(N - 1, -1, -1):

        @dace.tasklet
        def init_w4():
            in_y << y[i]
            out >> w
            out = in_y

        @dace.map
        def set_w4(j: _[i + 1:N]):
            in_A << A[i, j]
            in_x << x[j]
            out >> w(1, lambda x, y: x + y)
            out = -in_A * in_x

        @dace.tasklet
        def set_x4():
            in_w << w
            in_A << A[i, i]
            out >> x[i]
            out = in_w / in_A


if __name__ == '__main__':
    polybench.main(sizes, args, [(2, 'x')], init_array, ludcmp)
