# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
try:
    import polybench
except ImportError:
    polybench = None

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
    N: 90
}, {
    tsteps: 100,
    N: 250
}, {
    tsteps: 500,
    N: 1300
}, {
    tsteps: 1000,
    N: 2800
}]
args = [
    ([N, N], datatype),
    ([N, N], datatype)
]


@dace.program(datatype[N, N], datatype[N, N])
def jacobi2d(A, B):
    for t in range(tsteps):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                with dace.tasklet:
                    a1 << A[i, j]
                    a2 << A[i, j - 1]
                    a3 << A[i, j + 1]
                    a4 << A[i + 1, j]
                    a5 << A[i - 1, j]
                    b >> B[i, j]
                    b = 0.2 * (a1 + a2 + a3 + a4 + a5)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                with dace.tasklet:
                    a1 << B[i, j]
                    a2 << B[i, j - 1]
                    a3 << B[i, j + 1]
                    a4 << B[i + 1, j]
                    a5 << B[i - 1, j]
                    b >> A[i, j]
                    b = 0.2 * (a1 + a2 + a3 + a4 + a5)



def init_array(A, B):  #, N, tsteps):
    n = N.get()
    for i in range(n):
        for j in range(n):
            A[i, j] = datatype(i * (j + 2) + 2) / n
            B[i, j] = datatype(i * (j + 3) + 3) / n


if __name__ == '__main__':
    polybench.main(sizes, args, [(0, 'A')], init_array, jacobi2d)
