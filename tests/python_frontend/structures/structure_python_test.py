# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dataclasses import dataclass
import numpy as np
import pytest

from dace.transformation.auto.auto_optimize import auto_optimize
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


def test_write_structure_scalar():

    N = dace.symbol('N')
    SumStruct = dace.data.Structure(dict(sum=dace.data.Scalar(dace.float64)), name='SumStruct')

    @dace.program
    def struct_member_based_sum(A: dace.float64[N], B: SumStruct, C: dace.float64[N]):
        tmp = 0.0
        for i in range(N):
            tmp += A[i]
        B.sum = tmp
        for i in range(N):
            C[i] = A[i] + B.sum

    N = 40
    A = np.random.rand(N)
    C = np.random.rand(N)
    C_val = np.zeros((N, ))
    sum = 0
    for i in range(N):
        sum += A[i]
    for i in range(N):
        C_val[i] = A[i] + sum

    outB = SumStruct.dtype._typeclass.as_ctypes()(sum=0)

    func = struct_member_based_sum.compile()
    func(A=A, B=outB, C=C, N=N)

    # C is used for numerical validation because the Python frontend does not allow directly writing to scalars as an
    # output (B.sum). Using them as intermediate values is possible though.
    assert np.allclose(C, C_val)


def test_local_structure():

    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                              name='CSRMatrix')

    @dace.program
    def dense_to_csr_local_python(A: dace.float32[M, N], B: CSR):
        tmp = dace.define_local_structure(CSR)
        idx = 0
        for i in range(M):
            tmp.indptr[i] = idx
            for j in range(N):
                if A[i, j] != 0:
                    tmp.data[idx] = A[i, j]
                    tmp.indices[idx] = j
                    idx += 1
        tmp.indptr[M] = idx
        B.indptr[:] = tmp.indptr[:]
        B.indices[:] = tmp.indices[:]
        B.data[:] = tmp.data[:]

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

    func = dense_to_csr_local_python.compile()
    func(A=A, B=outB, M=tmp.shape[0], N=tmp.shape[1], nnz=tmp.nnz)


def test_rgf():
    # NOTE: "diag" is a sympy function
    class BTD:

        def __init__(self, diag, upper, lower):
            self.diagonal = diag
            self.upper = upper
            self.lower = lower

    n, nblocks = dace.symbol('n'), dace.symbol('nblocks')
    BlockTriDiagonal = dace.data.Structure(dict(diagonal=dace.complex128[nblocks, n, n],
                                                upper=dace.complex128[nblocks, n, n],
                                                lower=dace.complex128[nblocks, n, n]),
                                           name='BlockTriDiagonalMatrix')

    @dace.program
    def rgf_leftToRight(A: BlockTriDiagonal, B: BlockTriDiagonal, n_: dace.int32, nblocks_: dace.int32):

        # Storage for the incomplete forward substitution
        tmp = np.zeros_like(A.diagonal)
        identity = np.zeros_like(tmp[0])

        # 1. Initialisation of tmp
        tmp[0] = np.linalg.inv(A.diagonal[0])
        for i in dace.map[0:identity.shape[0]]:
            identity[i, i] = 1

        # 2. Forward substitution
        # From left to right
        for i in range(1, nblocks_):
            tmp[i] = np.linalg.inv(A.diagonal[i] - A.lower[i - 1] @ tmp[i - 1] @ A.upper[i - 1])
        # 3. Initialisation of last element of B
        B.diagonal[-1] = tmp[-1]

        # 4. Backward substitution
        # From right to left

        for i in range(nblocks_ - 2, -1, -1):
            B.diagonal[i] = tmp[i] @ (identity + A.upper[i] @ B.diagonal[i + 1] @ A.lower[i] @ tmp[i])
            B.upper[i] = -tmp[i] @ A.upper[i] @ B.diagonal[i + 1]
            B.lower[i] = np.transpose(B.upper[i])

    rng = np.random.default_rng(42)

    A_diag = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20))
    A_upper = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20))
    A_lower = rng.random((10, 20, 20)) + 1j * rng.random((10, 20, 20))
    inpBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diagonal=A_diag.__array_interface__['data'][0],
                                                           upper=A_upper.__array_interface__['data'][0],
                                                           lower=A_lower.__array_interface__['data'][0])

    B_diag = np.zeros((10, 20, 20), dtype=np.complex128)
    B_upper = np.zeros((10, 20, 20), dtype=np.complex128)
    B_lower = np.zeros((10, 20, 20), dtype=np.complex128)
    outBTD = BlockTriDiagonal.dtype._typeclass.as_ctypes()(diagonal=B_diag.__array_interface__['data'][0],
                                                           upper=B_upper.__array_interface__['data'][0],
                                                           lower=B_lower.__array_interface__['data'][0])

    func = rgf_leftToRight.compile()
    func(A=inpBTD, B=outBTD, n_=A_diag.shape[1], nblocks_=A_diag.shape[0], n=A_diag.shape[1], nblocks=A_diag.shape[0])

    A = BTD(A_diag, A_upper, A_lower)
    B = BTD(np.zeros((10, 20, 20), dtype=np.complex128), np.zeros((10, 20, 20), dtype=np.complex128),
            np.zeros((10, 20, 20), dtype=np.complex128))

    rgf_leftToRight.f(A, B, A_diag.shape[1], A_diag.shape[0])

    assert np.allclose(B.diagonal, B_diag)
    assert np.allclose(B.upper, B_upper)
    assert np.allclose(B.lower, B_lower)


@pytest.mark.skip('Compiler error (const conversion)')
@pytest.mark.gpu
def test_read_structure_gpu():

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
    ref = A.toarray()

    inpA = CSR.dtype._typeclass.as_ctypes()(indptr=A.indptr.__array_interface__['data'][0],
                                            indices=A.indices.__array_interface__['data'][0],
                                            data=A.data.__array_interface__['data'][0])

    # TODO: The following doesn't work because we need to create a Structure data descriptor from the ctypes class.
    # csr_to_dense_python(inpA, B)
    naive = csr_to_dense_python.to_sdfg(simplify=False)
    naive.apply_gpu_transformations()
    B = np.zeros((20, 20), dtype=np.float32)
    naive(inpA, B, M=A.shape[0], N=A.shape[1], nnz=A.nnz)
    assert np.allclose(B, ref)

    simple = csr_to_dense_python.to_sdfg(simplify=True)
    simple.apply_gpu_transformations()
    B = np.zeros((20, 20), dtype=np.float32)
    simple(inpA, B, M=A.shape[0], N=A.shape[1], nnz=A.nnz)
    assert np.allclose(B, ref)

    auto = auto_optimize(simple)
    B = np.zeros((20, 20), dtype=np.float32)
    auto(inpA, B, M=A.shape[0], N=A.shape[1], nnz=A.nnz)
    assert np.allclose(B, ref)


def test_struct_interface():
    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    CSR = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                              name='CSRMatrix')

    @dace.program
    def dense_to_csr_local_python(A: dace.float32[M, N], B: CSR):
        tmp = dace.define_local_structure(CSR)
        idx = 0
        for i in range(M):
            tmp.indptr[i] = idx
            for j in range(N):
                if A[i, j] != 0:
                    tmp.data[idx] = A[i, j]
                    tmp.indices[idx] = j
                    idx += 1
        tmp.indptr[M] = idx
        B.indptr[:] = tmp.indptr[:]
        B.indices[:] = tmp.indices[:]
        B.data[:] = tmp.data[:]

    rng = np.random.default_rng(42)
    tmp = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
    A = tmp.toarray()
    B = tmp.tocsr(copy=True)
    ref = tmp.tocsr(copy=True)
    B.indptr[:] = -1
    B.indices[:] = -1
    B.data[:] = -1

    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        dense_to_csr_local_python(A, CSR.make_argument_from_object(B), M=tmp.shape[0], N=tmp.shape[1], nnz=tmp.nnz)

    assert np.allclose(B.indptr, ref.indptr)
    assert np.allclose(B.indices, ref.indices)
    assert np.allclose(B.data, ref.data)


def test_struct_recursive():

    @dataclass
    class Inner:
        a: dace.float32[20]
        b: dace.int32

    @dataclass
    class Outer:
        x: Inner
        y: dace.float64[10, 10]

    Struct = dace.data.Structure(
        {
            'x': dace.data.Structure({
                'a': dace.float32[20],
                'b': dace.int32
            }, name='InnerStruct'),
            'y': dace.float64[10, 10]
        },
        name='OuterStruct')

    @dace.program
    def struct_recursive(A: Struct, B: Struct):
        B.x.a[:] = A.x.a[:]
        B.x.b = A.x.b  # This assignment will not propagate back to the calling code (due to struct passing by value)
        B.y[:] = A.y[:]

    A = Outer(x=Inner(a=np.random.rand(20).astype(np.float32), b=42), y=np.random.rand(10, 10).astype(np.float64))
    B = Outer(x=Inner(a=np.zeros(20, dtype=np.float32), b=0), y=np.zeros((10, 10), dtype=np.float64))

    struct_recursive(Struct.make_argument_from_object(A), Struct.make_argument_from_object(B))

    assert np.allclose(B.x.a, A.x.a)
    assert not np.allclose(B.x.b, A.x.b)
    assert np.allclose(B.y, A.y)


def test_struct_recursive_from_dataclass():

    @dataclass
    class Inner:
        a: dace.float32[20]
        b: dace.int32

    @dataclass
    class Outer:
        x: Inner
        y: dace.float64[10, 10]

    Struct = dace.data.Structure.from_dataclass(Outer)

    @dace.program
    def struct_recursive(A: Struct, B: Struct):
        B.x.a[:] = A.x.a[:]
        B.x.b = A.x.b  # This assignment will not propagate back to the calling code (due to struct passing by value)
        B.y[:] = A.y[:]

    A = Outer(x=Inner(a=np.random.rand(20).astype(np.float32), b=42), y=np.random.rand(10, 10).astype(np.float64))
    B = Outer(x=Inner(a=np.zeros(20, dtype=np.float32), b=0), y=np.zeros((10, 10), dtype=np.float64))

    struct_recursive(Struct.make_argument_from_object(A), Struct.make_argument_from_object(B))

    assert np.allclose(B.x.a, A.x.a)
    assert not np.allclose(B.x.b, A.x.b)
    assert np.allclose(B.y, A.y)


if __name__ == '__main__':
    test_read_structure()
    test_write_structure()
    test_write_structure_scalar()
    test_local_structure()
    test_rgf()
    # test_read_structure_gpu()
    test_struct_interface()
    test_struct_recursive()
    test_struct_recursive_from_dataclass()
