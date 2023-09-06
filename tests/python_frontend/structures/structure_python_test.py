# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from scipy import sparse


def test_read_structure():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                              name='CSRMatrix')

    @dace.program
    def csr_to_dense_python(A: CSR, B: dace.float32[M, N]):
        for i in dace.map[0:M]:
            for idx in dace.map[A.indptr[i]:A.indptr[i + 1]]:
                B[i, A.indices[idx]] = A.data[idx]
    
    rng = np.random.default_rng(42)
    A = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    B = np.zeros((20, 20), dtype=np.float32)

    inpA = CSR.dtype._typeclass.as_ctypes()(indptr=A.indptr.__array_interface__['data'][0],
                                            indices=A.indices.__array_interface__['data'][0],
                                            data=A.data.__array_interface__['data'][0])

    # TODO: The following doesn't work because we need to create a Structure data descriptor from the ctypes class.
    # csr_to_dense_python(inpA, B)
    func = csr_to_dense_python.compile()
    func(A=inpA, B=B, M=A.shape[0], N=A.shape[1], nnz=A.nnz)
    ref = A.toarray()

    assert np.allclose(B, ref)


def test_write_structure():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                              name='CSRMatrix')
    
    @dace.program
    def dense_to_csr_python(A: dace.float32[M, N], B: CSR):
        idx = 0
        for i in range(M):
            B.indptr[i] = idx
            for j in range(N):
                if A[i, j] != 0:
                    B.data[idx] = A[i, j]
                    B.indices[idx] = j
                    idx += 1
        B.indptr[M] = idx
    
    rng = np.random.default_rng(42)
    tmp = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    A = tmp.toarray()
    B = tmp.tocsr(copy=True)
    B.indptr[:] = -1
    B.indices[:] = -1
    B.data[:] = -1

    outB = CSR.dtype._typeclass.as_ctypes()(indptr=B.indptr.__array_interface__['data'][0],
                                            indices=B.indices.__array_interface__['data'][0],
                                            data=B.data.__array_interface__['data'][0])

    func = dense_to_csr_python.compile()
    func(A=A, B=outB, M=tmp.shape[0], N=tmp.shape[1], nnz=tmp.nnz)


def test_rgf():

    class BTD:

        def __init__(self, diag, upper, lower):
            self.diag = diag
            self.upper = upper
            self.lower = lower

    n, nblocks = dace.symbol('n'), dace.symbol('nblocks')
    BlockTriDiagonal = dace.data.Structure(
        dict(diag=dace.complex128[nblocks, n, n],
             upper=dace.complex128[nblocks, n, n],
             lower=dace.complex128[nblocks, n, n]),
        name='BlockTriDiagonalMatrix')
    
    @dace.program
    def rgf_leftToRight(A: BlockTriDiagonal, B: BlockTriDiagonal, n_: dace.int32, nblocks_: dace.int32):

        # Storage for the incomplete forward substitution
        tmp = np.zeros_like(A.diag)
        identity = np.zeros_like(tmp[0])

        # 1. Initialisation of tmp
        tmp[0] = np.linalg.inv(A.diag[0])
        for i in dace.map[0:identity.shape[0]]:
            identity[i, i] = 1

        # 2. Forward substitution
        # From left to right
        for i in range(1, nblocks_):
            tmp[i] = np.linalg.inv(A.diag[i] - A.lower[i-1] @ tmp[i-1] @ A.upper[i-1])

        # 3. Initialisation of last element of B
        B.diag[-1] = tmp[-1]

        # 4. Backward substitution
        # From right to left

        for i in range(nblocks_-2, -1, -1): 
            B.diag[i]  =  tmp[i] @ (identity + A.upper[i] @ B.diag[i+1] @ A.lower[i] @ tmp[i])
            B.upper[i] = -tmp[i] @ A.upper[i] @ B.diag[i+1]
            B.lower[i] =  np.transpose(B.upper[i])
    
    rng = np.random.default_rng(42)

    A_diag = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20))
    A_upper = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20))
    A_lower = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20)) 
    inpBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diag=A_diag.__array_interface__['data'][0],
                                                           upper=A_upper.__array_interface__['data'][0],
                                                           lower=A_lower.__array_interface__['data'][0])
    
    B_diag = np.zeros((10, 20, 20), dtype=np.complex128)
    B_upper = np.zeros((10, 20, 20), dtype=np.complex128)
    B_lower = np.zeros((10, 20, 20), dtype=np.complex128)
    outBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diag=B_diag.__array_interface__['data'][0],
                                                           upper=B_upper.__array_interface__['data'][0],
                                                           lower=B_lower.__array_interface__['data'][0])
    
    func = rgf_leftToRight.compile()
    func(A=inpBTD, B=outBTD, n_=A_diag.shape[1], nblocks_=A_diag.shape[0], n=A_diag.shape[1], nblocks=A_diag.shape[0])

    A = BTD(A_diag, A_upper, A_lower)
    B = BTD(np.zeros((10, 20, 20), dtype=np.complex128),
            np.zeros((10, 20, 20), dtype=np.complex128),
            np.zeros((10, 20, 20), dtype=np.complex128))
    
    rgf_leftToRight.f(A, B, A_diag.shape[1], A_diag.shape[0])

    assert np.allclose(B.diag, B_diag)
    assert np.allclose(B.upper, B_upper)
    assert np.allclose(B.lower, B_lower)


if __name__ == '__main__':
    test_read_structure()
    test_write_structure()
    test_rgf()
