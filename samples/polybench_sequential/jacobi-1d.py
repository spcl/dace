# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import polybench

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


@dace.program(datatype[N], datatype[N])
def jacobi1d(A, B):
    for t in range(tsteps):
        for i in range(1, N-1):
            with dace.tasklet:
                a1 << A[i - 1]
                a2 << A[i]
                a3 << A[i + 1]
                b >> B[i]
                b = 0.33333 * (a1 + a2 + a3)
        for i in range(1, N-1):
            with dace.tasklet:
                b1 << B[i - 1]
                b2 << B[i]
                b3 << B[i + 1]
                a >> A[i]
                a = 0.33333 * (b1 + b2 + b3)



def init_array(A, B):
    n = N.get()
    for i in range(n):
        A[i] = datatype(i + 2) / n
        B[i] = datatype(i + 3) / n


if __name__ == '__main__':
    # pluto: has no parallelism and strict control flow bug in DaCe (if stmt)
    polybench.main(sizes, args, [(0, 'A')], init_array, jacobi1d)
