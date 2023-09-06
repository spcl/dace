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


if __name__ == '__main__':
    test_read_structure()
