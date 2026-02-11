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

args = [([N, N], datatype), ([N], datatype), ([N], datatype)]


def init_array(L, x, b, n):
    x[:] = datatype(-999)
    for i in range(0, n, 1):
        b[i] = datatype(i)
    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            L[i, j] = 2 * datatype(i + n - j + 1) / n
        for j in range(i + 1, n, 1):
            L[i, j] = datatype(0)


@dace.program
def trisolv(L: datatype[N, N], x: datatype[N], b: datatype[N]):
    for i in range(0, N, 1):

        @dace.tasklet
        def init_x():
            in_b << b[i]
            out >> x[i]
            out = in_b

        @dace.map
        def set_x(j: _[0:i]):
            in_L << L[i, j]
            in_x << x[j]
            out >> x(1, lambda x, y: x + y)[i]
            out = -in_L * in_x

        @dace.tasklet
        def div():
            in_x << x[i]
            in_L << L[i, i]
            out >> x[i]
            out = in_x / in_L


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'x')], init_array, trisolv)
