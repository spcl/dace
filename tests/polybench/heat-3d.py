# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

N = dace.symbol('N')
tsteps = dace.symbol('tsteps')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{tsteps: 20, N: 10}, {tsteps: 40, N: 20}, {tsteps: 100, N: 40}, {tsteps: 500, N: 120}, {tsteps: 1000, N: 200}]
args = [
    ([N, N, N], datatype),
    ([N, N, N], datatype)  #, N, tsteps
]


@dace.program(datatype[N, N, N], datatype[N, N, N])  #, dace.int32, dace.int32)
def heat3d(A, B):  #, N, tsteps):
    for t in range(tsteps):

        @dace.map
        def a(i: _[1:N - 1], j: _[1:N - 1], k: _[1:N - 1]):
            a11 << A[i + 1, j, k]
            a12 << A[i - 1, j, k]
            a21 << A[i, j + 1, k]
            a22 << A[i, j - 1, k]
            a31 << A[i, j, k + 1]
            a32 << A[i, j, k - 1]
            a << A[i, j, k]
            b >> B[i, j, k]

            b = 0.125 * (a11 - datatype(2.0) * a + a12) +\
                0.125 * (a21 - datatype(2.0) * a + a22) +\
                0.125 * (a31 - datatype(2.0) * a + a32) +\
                a

        @dace.map
        def a(i: _[1:N - 1], j: _[1:N - 1], k: _[1:N - 1]):
            a11 << B[i + 1, j, k]
            a12 << B[i - 1, j, k]
            a21 << B[i, j + 1, k]
            a22 << B[i, j - 1, k]
            a31 << B[i, j, k + 1]
            a32 << B[i, j, k - 1]
            a << B[i, j, k]
            b >> A[i, j, k]

            b = 0.125 * (a11 - datatype(2.0) * a + a12) +\
                0.125 * (a21 - datatype(2.0) * a + a22) +\
                0.125 * (a31 - datatype(2.0) * a + a32) +\
                a


def init_array(A, B):  #, N, tsteps):
    n = N.get()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                A[i, j, k] = datatype((i + j + (n - k)) * 10) / n
                B[i, j, k] = datatype((i + j + (n - k)) * 10) / n


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, heat3d)
