# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import scipy as sp
import scipy.sparse as sparse

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz_A = dace.symbol('nnz_A')
nnz_B = dace.symbol('nnz_B')
'''
C = A @ B
[M, K] = [M, N] @ [N, K]
C[i, k] = A[i, j] * B[j, k]
'''


@dace.program
def spmspm_csr_csr(A2_pos: dace.int32[M + 1], A2_crd: dace.int32[nnz_A], A_val: dace.float64[nnz_A],
                   B2_pos: dace.int32[N + 1], B2_crd: dace.int32[nnz_B], B_val: dace.float64[nnz_B],
                   C: dace.float64[M, K]):
    for i in dace.map[0:M]:
        for pj in dace.map[A2_pos[i]:A2_pos[i + 1]]:
            for pk in dace.map[B2_pos[A2_crd[pj]]:B2_pos[A2_crd[pj] + 1]]:
                C[i, B2_crd[pk]] += A_val[pj] * B_val[pk]


def test_spmspm_csr_csr():
    csr_A = sparse.random(200, 100, density=0.5, format='csr')
    csr_B = sparse.random(100, 150, density=0.5, format='csr')
    ref_dense_C = (csr_A @ csr_B).todense()
    dace_dense_C = np.zeros_like(ref_dense_C)
    spmspm_csr_csr(csr_A.indptr, np.copy(csr_A.indices), np.copy(csr_A.data), csr_B.indptr, np.copy(csr_B.indices),
                   np.copy(csr_B.data), dace_dense_C)
    assert np.allclose(ref_dense_C, dace_dense_C)


if __name__ == '__main__':
    test_spmspm_csr_csr()
