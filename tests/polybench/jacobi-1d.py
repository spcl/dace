# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench
from absl import app, flags

N = dace.symbol('N')
tsteps = dace.symbol('tsteps')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    tsteps: 20,
    N: 30
}, {
    tsteps: 40,
    N: 120
}, {
    tsteps: 100,
    N: 400
}, {
    tsteps: 500,
    N: 2000
}, {
    tsteps: 1000,
    N: 4000
}]
args = [([N], datatype), ([N], datatype)]  #, N, tsteps]


@dace.program
def jacobi1d(A: datatype[N], B: datatype[N]):  #, N, tsteps):
    for t in range(tsteps):

        @dace.map
        def a(i: _[1:N - 1]):
            a1 << A[i - 1]
            a2 << A[i]
            a3 << A[i + 1]
            b >> B[i]
            b = 0.33333 * (a1 + a2 + a3)

        @dace.map
        def b(i: _[1:N - 1]):
            a1 << B[i - 1]
            a2 << B[i]
            a3 << B[i + 1]
            b >> A[i]
            b = 0.33333 * (a1 + a2 + a3)


def init_array(A, B, n, tsteps):
    for i in range(n):
        A[i] = datatype(i + 2) / n
        B[i] = datatype(i + 3) / n


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, jacobi1d)
