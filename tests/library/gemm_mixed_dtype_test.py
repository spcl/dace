# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A vendor GEMM of a real and a complex matrix computes in the complex type, as NumPy does.

A BLAS routine takes one element type for all its matrices, and the expansions named it after A and
cast every pointer to it. A real A times a complex B called ``dgemm`` on the complex buffers: the
OpenBLAS build rejects the pointer, and the rocBLAS/cuBLAS C-style casts compile and return wrong
numbers. npbench cegterg's non-local projector ``deeq @ ps`` is that product; it failed to compile on
the CPU canonicalize column and returned an eigenvalue error of 100% on the GPU one.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import common
from dace.libraries.blas.nodes.gemm import Gemm

M, K, N = 5, 7, 3
#: The operands are read from inside bigger arrays at an offset, so the cast must follow the subset.
PAD = 2


def mixed_gemm(implementation: str, a_dtype: dace.typeclass, b_dtype: dace.typeclass,
               storage: dtypes.StorageType) -> dace.SDFG:
    """``C = A[PAD:, PAD:] @ B[PAD:, :]`` with one real and one complex operand."""
    sdfg = dace.SDFG(f'mixed_gemm_{implementation}_{a_dtype.to_string()}_{b_dtype.to_string()}')
    sdfg.add_array('A', [M + PAD, K + PAD], a_dtype, storage=storage)
    sdfg.add_array('B', [K + PAD, N], b_dtype, storage=storage)
    sdfg.add_array('C', [M, N], dace.complex128, storage=storage)
    state = sdfg.add_state()
    node = Gemm('gemm')
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[{PAD}:{M + PAD}, {PAD}:{K + PAD}]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet(f'B[{PAD}:{K + PAD}, 0:{N}]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[0:{M}, 0:{N}]'))
    return sdfg


def operands(a_dtype: dace.typeclass, b_dtype: dace.typeclass):
    """Random operands of the given types and the NumPy product of their subsets."""
    rng = np.random.default_rng(42)

    def draw(shape, dtype):
        values = rng.standard_normal(shape)
        return values + 1j * rng.standard_normal(shape) if dtype.is_complex() else values

    a = draw((M + PAD, K + PAD), a_dtype)
    b = draw((K + PAD, N), b_dtype)
    return a, b, a[PAD:, PAD:] @ b[PAD:]


MIXED = [(dace.float64, dace.complex128), (dace.complex128, dace.float64)]


@pytest.mark.parametrize('a_dtype, b_dtype', MIXED, ids=['real_a', 'real_b'])
def test_a_mixed_openblas_gemm_matches_numpy(a_dtype, b_dtype):
    sdfg = mixed_gemm('OpenBLAS', a_dtype, b_dtype, dtypes.StorageType.Default)
    a, b, ref = operands(a_dtype, b_dtype)
    c = np.zeros((M, N), np.complex128)
    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('a_dtype, b_dtype', MIXED, ids=['real_a', 'real_b'])
def test_the_vendor_call_is_complex_and_the_real_operand_is_cast(a_dtype, b_dtype):
    """One ``zgemm``, fed a complex copy of the real operand and never the real buffer itself."""
    sdfg = mixed_gemm('OpenBLAS', a_dtype, b_dtype, dtypes.StorageType.Default)
    sdfg.expand_library_nodes()
    code = '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert 'cblas_zgemm' in code and 'cblas_dgemm' not in code, code
    real = 'A' if a_dtype == dace.float64 else 'B'
    readers = {
        e.dst
        for e in sdfg.start_state.out_edges(next(n for n in sdfg.start_state.data_nodes() if n.data == real))
    }
    assert all(isinstance(r, dace.nodes.MapEntry) for r in readers), readers


def test_a_loop_expansion_reads_the_operands_in_their_own_types():
    """``pure`` needs no copy -- and inside a kernel it could not allocate one."""
    sdfg = mixed_gemm('pure', dace.float64, dace.complex128, dtypes.StorageType.Default)
    arrays = set(sdfg.arrays)
    sdfg.expand_library_nodes()
    assert set(sdfg.arrays) == arrays
    a, b, ref = operands(dace.float64, dace.complex128)
    c = np.zeros((M, N), np.complex128)
    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize('a_dtype, b_dtype', MIXED, ids=['real_a', 'real_b'])
def test_a_mixed_device_gemm_matches_numpy(a_dtype, b_dtype):
    import cupy
    implementation = 'rocBLAS' if common.get_gpu_backend() == 'hip' else 'cuBLAS'
    sdfg = mixed_gemm(implementation, a_dtype, b_dtype, dtypes.StorageType.GPU_Global)
    a, b, ref = operands(a_dtype, b_dtype)
    c = cupy.zeros((M, N), np.complex128)
    sdfg(A=cupy.asarray(a), B=cupy.asarray(b), C=c)
    np.testing.assert_allclose(cupy.asnumpy(c), ref, rtol=1e-12, atol=1e-12)
