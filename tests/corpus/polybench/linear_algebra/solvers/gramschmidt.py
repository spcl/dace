# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes, all SQUARE. A Gram-Schmidt QR of a wide matrix (M < N, the original PolyBench/C
# ``{M:20, N:30}``) is rank-deficient by construction: rank is at most M, so the last N-M columns
# orthogonalize to a ~0 residual, ``R[k,k]`` collapses to rounding noise and ``Q[:,k] = A[:,k]/R[k,k]``
# divides by it. Two implementations then disagree by ~100% past column M whatever they do, and a
# harmless FMA-contraction ULP is enough to trigger it.
#
# The diagonal-dominance term in ``init_array`` cannot repair that -- only min(M, N) diagonal entries
# exist, so the columns past M get no boost and the rank does not change. MEASURED at {M:200, N:240}:
# raising the boost 100x takes cond from 1.587 to 1.005 while the rank stays 200 of 240. Square is
# what actually fixes it: full column rank at every preset, cond ~1.7 throughout (30, 120, 220, 1100
# all measured), so the QR is stable and canonicalization stays value-preserving with FMA on.
sizes = [{M: 30, N: 30}, {M: 120, N: 120}, {M: 220, N: 220}, {M: 1100, N: 1100}, {M: 2300, N: 2300}]

#: ported from the npbench bench_info paper row -- M > N here (tall), so full column rank holds
#: even though sizes above is square; see the rank-deficiency note above for why that matters.
paper_sizes = {M: 240, N: 200}

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
    # npbench formulation: modified Gram-Schmidt QR. The column inner products
    # ``np.dot(A[:, k], A[:, k])`` and ``np.dot(Q[:, k], A[:, j])`` lower to Dot library
    # nodes instead of scalar ``i`` reduction maps.
    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]


if __name__ == '__main__':
    import polybench  # noqa: E402  (CLI only; corpus loads module without it)
    polybench.main(sizes, args, [(1, 'R'), (2, 'Q')], init_array, gramschmidt)
