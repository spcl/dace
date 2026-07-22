# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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
    # npbench formulation: slice-vectorized neighbor sum over each row, then a sequential
    # in-row Gauss-Seidel scan (``A[i, j] += A[i, j-1]``) that is inherently serial.
    for t in range(0, tsteps - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] +
                           A[i + 1, 1:-1] + A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def init_array(A, tsteps, n):
    for i in range(n):
        for j in range(n):
            A[i, j] = datatype(i * (j + 2) + 2) / n


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'A')], init_array, seidel2d)
