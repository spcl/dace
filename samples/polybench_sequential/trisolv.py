# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{N: 40}, {N: 120}, {N: 400}, {N: 2000}, {N: 4000}]

args = [([N, N], datatype), ([N], datatype), ([N], datatype)]


def init_array(L, x, b):
    n = N.get()

    x[:] = datatype(-999)
    for i in range(0, n, 1):
        b[i] = datatype(i)
    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            L[i, j] = 2 * datatype(i + n - j + 1) / n
        for j in range(i + 1, n, 1):
            L[i, j] = datatype(0)


@dace.program(datatype[N, N], datatype[N], datatype[N])
def trisolv(L, x, b):
    for i in range(N):
        with dace.tasklet:
            x_out >> x[i]
            b_in << b[i]
            x_out = b_in
        # x[i] = b[i]
        for j in range(i):
            with dace.tasklet:
                L_in << L[i][j]
                xi_in << x[i]
                xj_in << x[j]
                xi_out >> x[i]
                xi_out = xi_in - L_in * xj_in
            # x[i] = x[i] - L[i][j] * x[j]
        with dace.tasklet:
            L_in << L[i][i]
            x_in << x[i]
            x_out >> x[i]
            x_out = x_in / L_in
        # x[i] = x[i] / L[i][i]


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'x')], init_array, trisolv)
