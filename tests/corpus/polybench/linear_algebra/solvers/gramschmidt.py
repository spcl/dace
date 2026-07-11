# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes.
# The small preset is TALL (M >= N): a Gram-Schmidt QR of a wide matrix (M < N, the original
# PolyBench/C ``{M:20, N:30}``) is intrinsically rank-deficient -- the last N-M columns
# orthogonalize to a ~0 residual, so ``R[k,k]`` collapses and dividing by it is catastrophically
# ill-conditioned (a harmless FMA-contraction ULP then amplifies to ~100% relative error). A tall
# matrix has full column rank, so with the diagonal-dominance ``init_array`` below the QR is
# well-conditioned and canonicalization is value-preserving with FMA on. Upstream npbench uses the
# same tall convention (its ``S`` preset is M=70, N=60).
sizes = [{M: 30, N: 20}, {M: 60, N: 180}, {M: 200, N: 240}, {M: 1000, N: 1200}, {M: 2000, N: 2600}]

args = [([M, N], datatype), ([N, N], datatype), ([M, N], datatype)]


def init_array(A, R, Q, m, n):
    # The original polybench formula ``A[i,j] = ((i*j) % m)/m * 100 + 10`` makes A
    # rank-deficient (repeated columns from the modular pattern), so the QR is ill-posed
    # and a harmless FMA-contraction ULP difference amplifies to ~100% relative error.
    # Adding a diagonal-dominance term makes A full-column-rank / well-conditioned (cond
    # ~1.7 at the small preset) so the QR is stable and canonicalization stays value-preserving.
    for i in range(0, m, 1):
        for j in range(0, n, 1):
            A[i, j] = ((datatype((i * j) % m) / m) * 100) + 10
            if i == j:
                A[i, j] += datatype(n) * 100
            Q[i, j] = datatype(0)
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            R[i, j] = datatype(0)


@dace.program
def gramschmidt(A: datatype[M, N], R: datatype[N, N], Q: datatype[M, N]):

    nrm = dace.define_local([1], datatype)

    for k in range(0, N, 1):

        @dace.tasklet
        def set_nrm():
            out_nrm >> nrm
            out_nrm = datatype(0)

        @dace.map
        def set_sum(i: _[0:M]):
            in_A << A[i, k]
            out_nrm >> nrm(1, lambda x, y: x + y)
            out_nrm = in_A * in_A

        @dace.tasklet
        def set_rkk():
            in_nrm << nrm
            out_R >> R[k, k]
            out_R = math.sqrt(in_nrm)

        @dace.map
        def set_q(i: _[0:M]):
            in_A << A[i, k]
            in_R << R[k, k]
            out_Q >> Q[i, k]
            out_Q = in_A / in_R

        @dace.mapscope
        def set_rna(j: _[k + 1:N]):
            # for j in range(k+1, N, 1):

            @dace.tasklet
            def init_r():
                out_R >> R[k, j]
                out_R = datatype(0)

            @dace.map
            def set_r(i: _[0:M]):
                in_A << A[i, j]
                in_Q << Q[i, k]
                out_R >> R(1, lambda x, y: x + y)[k, j]
                out_R = in_A * in_Q

            @dace.map
            def set_a(i: _[0:M]):
                in_R << R[k, j]
                in_Q << Q[i, k]
                out_A >> A(1, lambda x, y: x + y)[i, j]
                out_A = -in_R * in_Q


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(1, 'R'), (2, 'Q')], init_array, gramschmidt)
