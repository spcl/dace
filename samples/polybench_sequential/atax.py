# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')
M = dace.symbol('M')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 38,
    N: 42,
}, {
    M: 116,
    N: 124,
}, {
    M: 390,
    N: 410,
}, {
    M: 1900,
    N: 2100,
}, {
    M: 1800,
    N: 2200,
}]

args = [([M, N], datatype), ([N], datatype), ([N], datatype)]


def init_array(A, x, y):
    n = N.get()
    m = M.get()
    fn = datatype(n)

    for i in range(n):
        x[i] = 1 + (i / fn)
    for i in range(m):
        for j in range(n):
            A[i, j] = datatype((i + j) % n) / (5 * m)


@dace.program(datatype[M, N], datatype[N], datatype[N])
def atax(A, x, y):
    tmp = dace.define_local([M], dtype=datatype)

    for i in range(N):
        y[i] = 0.0
    for i in range(M):
        tmp[i] = 0.0
        for j in range(N):
            with dace.tasklet:
                a_in << A[i, j]
                x_in << x[j]
                tmp_in << tmp[i]
                tmp_out >> tmp[i]
                tmp_out = tmp_in + (a_in * x_in)
            # tmp[i] = tmp[i] + A[i, j] * x[j]
        for j in range(N):
            with dace.tasklet:
                a_in << A[i, j]
                tmp_in << tmp[i]
                y_in << y[j]
                y_out >> y[j]
                y_out = y_in + (a_in * tmp_in)
            # y[j] = y[j] + A[i, j] * tmp[i]


if __name__ == '__main__':
    polybench.main(sizes, args, [(2, 'y')], init_array, atax)
