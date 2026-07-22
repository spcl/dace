# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

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
    # npbench formulation: slice-vectorized Jacobi sweeps (each ``[1:-1]`` assignment is a
    # single vectorizable map).
    for t in range(1, tsteps):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def init_array(A, B, n, tsteps):
    for i in range(n):
        A[i] = datatype(i + 2) / n
        B[i] = datatype(i + 3) / n


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(0, 'A')], init_array, jacobi1d)
