# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from scipy import sparse
from typing import Union


class CSRMatrix(dace.data.Structure):

    indptr: dace.data.Array
    indices: dace.data.Array
    data: dace.data.Array
    rows: Union[int, dace.symbolic.SymbolicType]
    cols: Union[int, dace.symbolic.SymbolicType]
    nnz: Union[int, dace.symbolic.SymbolicType]

    def __init__(self,
                 rows: Union[int, dace.symbolic.SymbolicType],
                 cols: Union[int, dace.symbolic.SymbolicType],
                 nnz: Union[int, dace.symbolic.SymbolicType],
                 dtype: dace.typeclass,
                 itype: dace.typeclass = dace.int32):

        self.indptr = itype[rows + 1]
        self.indices = itype[nnz]
        self.data = dtype[nnz]
        self.rows = rows
        self.cols = cols
        self.nnz = nnz

        super().__init__() 


def test_csrmm():

    M, N, K, nnz = (dace.symbol(s) for s in ('M', 'N', 'K', 'nnz'))
    CSR = CSRMatrix(M, K, nnz, dace.float32)

    @dace.program
    def csrmm(A: CSR, B: dace.float32[K, N]) -> dace.float32[M, N]:
        C = np.zeros((M, N), dtype=dace.float32)
        for i in range(M):
            for j in range(N):
                for k in range(A.indptr[i], A.indptr[i + 1]):
                    k_idx = A.indices[k]
                    C[i, j] += A.data[k] * B[k_idx, j]
        return C
    
    sdfg = csrmm.to_sdfg()
    func = sdfg.compile()
    
    rng = np.random.default_rng(42)
    A = sparse.random(200, 100, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    B = rng.random((100, 50), dtype=np.float32)

    inpA = CSR.dtype.as_ctypes()(
        indptr=A.indptr.__array_interface__['data'][0],
        indices=A.indices.__array_interface__['data'][0],
        data=A.data.__array_interface__['data'][0],
        rows=A.shape[0],
        cols=A.shape[1],
        M=A.shape[0],
        K=A.shape[1],
        nnz=A.nnz
    )

    val = func(A=inpA, B=B, M=200, K=100, N=50, nnz=A.nnz)
    ref = A @ B

    assert np.allclose(val, ref)


def test_batched_csrmm():

    M, N, K, L, nnz = (dace.symbol(s) for s in ('M', 'N', 'K', 'L', 'nnz'))
    CSR = CSRMatrix(M, K, nnz, dace.float32)

    @dace.program
    def batched_csrmm(A: CSR[L], B: dace.float32[L, K, N]) -> dace.float32[L, M, N]:
        C = np.zeros((L, M, N), dtype=dace.float32)
        for l in range(L):
            for i in range(M):
                for j in range(N):
                    for k in range(A[l].indptr[i], A[l].indptr[i + 1]):
                        k_idx = A[l].indices[k]
                        C[l, i, j] += A[l].data[k] * B[l, k_idx, j]
        return C
    
    sdfg = batched_csrmm.to_sdfg()
    func = sdfg.compile()
    
    rng = np.random.default_rng(42)
    B = rng.random((3, 100, 50), dtype=np.float32)

    inpA = np.empty((3,), dtype=np.dtype(CSR.dtype.as_ctypes()))
    ref = np.empty((3, 200, 50), dtype=np.float32)
    for l in range(3):
        A = sparse.random(200, 100, density=0.1, format='csr', dtype=np.float32, random_state=rng)
        inpA[l] = CSR.dtype.as_ctypes()(
            indptr=A.indptr.__array_interface__['data'][0],
            indices=A.indices.__array_interface__['data'][0],
            data=A.data.__array_interface__['data'][0],
            rows=A.shape[0],
            cols=A.shape[1],
            M=A.shape[0],
            K=A.shape[1],
            nnz=A.nnz
        )
        ref[l] = A @ B[l]

    val = func(A=inpA, B=B, M=200, K=100, N=50, nnz=A.nnz)

    assert np.allclose(val, ref)


if __name__ == '__main__':
    test_csrmm()
    test_batched_csrmm()
