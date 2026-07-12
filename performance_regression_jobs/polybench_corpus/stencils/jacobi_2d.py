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
    ([N, N], datatype)  #, N, tsteps
]


@dace.program
def jacobi2d(A: datatype[N, N], B: datatype[N, N]):  #, N, tsteps):
    # npbench formulation: slice-vectorized 5-point Jacobi sweeps.
    for t in range(1, tsteps):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


def init_array(A, B, n, tsteps):
    for i in range(n):
        for j in range(n):
            A[i, j] = datatype(i * (j + 2) + 2) / n
            B[i, j] = datatype(i * (j + 3) + 3) / n


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    if polybench:
        polybench.main(sizes, args, [(0, 'A')], init_array, jacobi2d)
    else:
        init_array(*args, **{str(k).lower(): v for k, v in sizes[2].items()})
        jacobi2d(*args)
