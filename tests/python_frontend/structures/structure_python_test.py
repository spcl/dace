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
            # B.diag[i] = np.linalg.inv(A.diag[i] - A.lower[i-1] @ B.diag[i-1] @ A.upper[i-1])
            # B.diag[i] = np.linalg.inv(A.diag[i])
            # tmp[i] = np.linalg.inv(A.diag[i])

        # 3. Initialisation of last element of B
        B.diag[-1] = tmp[-1]

        # 4. Backward substitution
        # From right to left

        for i in range(nblocks_-2, -1, -1): 
            B.diag[i]  =  tmp[i] @ (identity + A.upper[i] @ B.diag[i+1] @ A.lower[i] @ tmp[i])
            B.upper[i] = -tmp[i] @ A.upper[i] @ B.diag[i+1]
            B.lower[i] =  np.transpose(B.upper[i])
            # B.diag[i] = tmp[i]
    
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
    
    sdfg = rgf_leftToRight.to_sdfg()
    from dace.transformation.auto.auto_optimize import auto_optimize
    auto_optimize(sdfg, dace.DeviceType.GPU)
    sdfg.simplify()
    func = sdfg.compile()
    # func = rgf_leftToRight.compile()
    func(A=inpBTD, B=outBTD, n_=A_diag.shape[1], nblocks_=A_diag.shape[0], n=A_diag.shape[1], nblocks=A_diag.shape[0])

    A = BTD(A_diag, A_upper, A_lower)
    B = BTD(np.zeros((10, 20, 20), dtype=np.complex128),
            np.zeros((10, 20, 20), dtype=np.complex128),
            np.zeros((10, 20, 20), dtype=np.complex128))
    
    rgf_leftToRight.f(A, B, A_diag.shape[1], A_diag.shape[0])

    assert np.allclose(B.diag, B_diag)
    assert np.allclose(B.upper, B_upper)
    assert np.allclose(B.lower, B_lower)


def test_rgf2():

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
        tmp = dace.define_local((nblocks, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        A_diag_ = dace.define_local((2, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        A_lower_ = dace.define_local((2, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        A_upper_ = dace.define_local((2, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        B_diag_ = dace.define_local((2, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        B_upper_ = dace.define_local((2, n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        identity = dace.define_local((n, n), dtype=dace.complex128, storage=dace.StorageType.GPU_Global)
        for i in dace.map[0:identity.shape[0]]:
            identity[i, i] = 1

        # 1. Initialisation of tmp
        A_diag_[0] = A.diag[0]
        tmp[0] = np.linalg.inv(A_diag_[0])

        # 2. Forward substitution
        # From left to right
        for i in range(1, nblocks_):
            A_diag_[i % 2] = A.diag[i]
            A_lower_[i % 2] = A.lower[i-1]
            A_upper_[i % 2] = A.upper[i-1]
            tmp[i] = np.linalg.inv(A_diag_[i % 2] - A_lower_[i % 2] @ tmp[i-1] @ A_upper_[i % 2])

        # 3. Initialisation of last element of B
        B_diag_[(nblocks - 1) % 2] = tmp[-1]
        B.diag[-1] = tmp[-1]

        # 4. Backward substitution
        # From right to left

        for i in range(nblocks_-2, -1, -1):
            A_lower_[i % 2] = A.lower[i]
            A_upper_[i % 2] = A.upper[i]
            B_upper_[i % 2] = B.upper[i]
            B_diag_[i % 2]  =  tmp[i] @ (identity + A_upper_[i % 2] @ B_diag_[(i+1) % 2] @ A_lower_[i % 2] @ tmp[i])
            B.diag[i] = B_diag_[i % 2]
            B.upper[i] = -tmp[i] @ A_upper_[i % 2] @ B_diag_[(i+1) % 2]
            B.lower[i] =  np.transpose(B_upper_[i % 2])
    
    sdfg = rgf_leftToRight.to_sdfg()
    from dace.transformation.auto.auto_optimize import auto_optimize, set_fast_implementations, make_transients_persistent
    set_fast_implementations(sdfg, dace.DeviceType.GPU)
    # NOTE: We need to `infer_types` in case a LibraryNode expands to other LibraryNodes (e.g., np.linalg.solve)
    from dace.sdfg import infer_types
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()
    sdfg.expand_library_nodes()
    for sd in sdfg.all_sdfgs_recursive():
        for _, desc in sd.arrays.items():
            if desc.storage == dace.StorageType.GPU_Shared:
                desc.storage = dace.StorageType.GPU_Global
    from dace.transformation.interstate import InlineSDFG
    sdfg.apply_transformations_repeated([InlineSDFG])
    make_transients_persistent(sdfg, dace.DeviceType.GPU)
    sdfg.view()      
    func = sdfg.compile()
    # func = rgf_leftToRight.compile()
    
    rng = np.random.default_rng(42)

    num_blocks = 10
    block_size = 512

    A_diag = rng.random((num_blocks, block_size, block_size)) + 1j * rng.random((num_blocks, block_size, block_size))
    A_upper = rng.random((num_blocks, block_size, block_size)) + 1j * rng.random((num_blocks, block_size, block_size))
    A_lower = rng.random((num_blocks, block_size, block_size)) + 1j * rng.random((num_blocks, block_size, block_size)) 
    inpBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diag=A_diag.__array_interface__['data'][0],
                                                           upper=A_upper.__array_interface__['data'][0],
                                                           lower=A_lower.__array_interface__['data'][0])
    
    B_diag = np.zeros((10, 20, 20), dtype=np.complex128)
    B_upper = np.zeros((10, 20, 20), dtype=np.complex128)
    B_lower = np.zeros((10, 20, 20), dtype=np.complex128)
    outBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diag=B_diag.__array_interface__['data'][0],
                                                           upper=B_upper.__array_interface__['data'][0],
                                                           lower=B_lower.__array_interface__['data'][0])
    
    func(A=inpBTD, B=outBTD, n_=A_diag.shape[1], nblocks_=A_diag.shape[0], n=A_diag.shape[1], nblocks=A_diag.shape[0])

    print(B_diag)

    # A = BTD(A_diag, A_upper, A_lower)
    # B = BTD(np.zeros((10, 20, 20), dtype=np.complex128),
    #         np.zeros((10, 20, 20), dtype=np.complex128),
    #         np.zeros((10, 20, 20), dtype=np.complex128))
    
    # rgf_leftToRight.f(A, B, A_diag.shape[1], A_diag.shape[0])

    # assert np.allclose(B.diag, B_diag)
    # assert np.allclose(B.upper, B_upper)
    # assert np.allclose(B.lower, B_lower)


if __name__ == '__main__':
    # test_read_structure()
    # test_write_structure()
    # test_rgf()
    test_rgf2()
