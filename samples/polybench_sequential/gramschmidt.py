# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    M: 20,
    N: 30
}, {
    M: 60,
    N: 180
}, {
    M: 200,
    N: 240
}, {
    M: 1000,
    N: 1200
}, {
    M: 2000,
    N: 2600
}]

args = [([M, N], datatype), ([N, N], datatype), ([M, N], datatype)]


def init_array(A, R, Q):
    m = M.get()
    n = N.get()

    for i in range(0, m, 1):
        for j in range(0, n, 1):
            A[i, j] = ((datatype((i * j) % m) / m) * 100) + 10
            Q[i, j] = datatype(0)
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            R[i, j] = datatype(0)


@dace.program(datatype[M, N], datatype[N, N], datatype[M, N])
def gramschmidt(A, R, Q):
    nrm = dace.define_local([1], datatype)
    for k in range(N):
        with dace.tasklet:
            out_nrm >> nrm
            out_nrm = 0.0
        for i in range(M):
            with dace.tasklet:
                in_A << A[i, k]
                in_nrm << nrm
                out_nrm >> nrm
                out_nrm = in_nrm + (in_A * in_A)
            # nrm += A[i, k] * A[i, k]
        with dace.tasklet:
            in_nrm << nrm
            r_out >> R[k, k]
            r_out = math.sqrt(in_nrm)
        for i in range(M):
            with dace.tasklet:
                in_A << A[i, k]
                in_R << R[k, k]
                out_Q >> Q[i, k]
                out_Q = in_A / in_R
            # Q[i, k] = A[i, k] / R[k, k]

        for j in range(k + 1, N):
            with dace.tasklet:
                R_out >> R[k, j]
                R_out = 0.0
            for i in range(M):
                with dace.tasklet:
                    A_in << A[i, j]
                    Q_in << Q[i, k]
                    R_in << R[k, j]
                    R_out >> R[k, j]
                    R_out = R_in + (Q_in * A_in)
                # R[k, j] = R[k, j] + (Q[i, k] * A[i, j])

            for i in range(M):
                with dace.tasklet:
                    in_R << R[k, j]
                    in_Q << Q[i, k]
                    in_A << A[i, j]
                    out_A >> A[i, j]
                    out_A = in_A - (in_R * in_Q)
                # A[i, j] -= Q[i, k] * R[k, j


if __name__ == '__main__':
    polybench.main(sizes, args, [(1, 'R'), (2, 'Q')], init_array, gramschmidt)
