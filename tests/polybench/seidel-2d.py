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
    N: 40
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
args = [([N, N], datatype)]


@dace.program
def seidel2d(A: datatype[N, N], tsteps: dace.int32):
    for t in range(tsteps):
        for i in range(1, N - 1):
            for j in range(1, N - 1):

                @dace.tasklet
                def a():
                    a1 << A[i - 1, j - 1]
                    a2 << A[i - 1, j]
                    a3 << A[i - 1, j + 1]
                    a4 << A[i, j - 1]
                    a5 << A[i, j]
                    a6 << A[i, j + 1]
                    a7 << A[i + 1, j - 1]
                    a8 << A[i + 1, j]
                    a9 << A[i + 1, j + 1]
                    out >> A[i, j]

                    out = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9) / datatype(9.0)


def init_array(A, tsteps, n):
    for i in range(n):
        for j in range(n):
            A[i, j] = datatype(i * (j + 2) + 2) / n


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, seidel2d)
