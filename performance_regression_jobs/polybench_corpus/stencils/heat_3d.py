# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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


@dace.program
def heat3d(A: datatype[N, N, N], B: datatype[N, N, N]):  #, N, tsteps):
    # npbench formulation: slice-vectorized 7-point heat stencil sweeps.
    for t in range(1, tsteps):
        B[1:-1, 1:-1, 1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) +
                               0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) +
                               0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) +
                               A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1, 1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) +
                               0.125 * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) +
                               0.125 * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) +
                               B[1:-1, 1:-1, 1:-1])


def init_array(A, B, n, tsteps):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                A[i, j, k] = datatype((i + j + (n - k)) * 10) / n
                B[i, j, k] = datatype((i + j + (n - k)) * 10) / n


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'A')], init_array, heat3d)
