# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ROCm vendor call casts its operands to the rocBLAS complex type, not a CUDA or HIP one.

The GPU expansions take their vendor C type from ``cublas_type_metadata``, which names the CUDA
types. ``float`` and ``double`` are spelled the same in every dialect, so the real paths were fine
and the complex ones were not: the solvers emitted ``(cuDoubleComplex*)``, a type ROCm does not
declare at all, and GEMM emitted ``(hipDoubleComplex*)``, which compiles as a type but does not
match the call. Measured against ROCm 6.3: ``rocblas_zgeam`` and ``rocblas_zgemm`` both reject a
``hipDoubleComplex*`` operand and accept ``rocblas_double_complex*``, because rocBLAS declares its
complex parameters as ``rocblas_complex_num<T>`` in C++.

quatrex_rgf carries all three on the canon GPU column: an ``Inv`` (getrf + getrs), a transpose
(``geam``) and complex GEMMs.
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.gemv import Gemv
from dace.libraries.lapack import Getrf, Getrs
from dace.libraries.linalg.nodes.transpose import Transpose

N = 8
#: What rocSOLVER calls the two complex types, against what CUDA calls them.
ROCBLAS_SPELLING = {
    dace.complex64: ('rocblas_float_complex', 'cuComplex'),
    dace.complex128: ('rocblas_double_complex', 'cuDoubleComplex')
}


def getrf_code(dtype: dace.typeclass) -> str:
    """The tasklet code the rocSOLVER LU factorization expands to."""
    sdfg = dace.SDFG(f'getrf_{dtype.to_string()}')
    sdfg.add_array('A', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('pivots', [N], dace.int32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('info', [1], dace.int32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Getrf('getrf')
    node.implementation = 'rocSOLVER'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_xin', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(node, '_xout', state.add_write('A'), None, dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(node, '_ipiv', state.add_write('pivots'), None, dace.Memlet(f'pivots[0:{N}]'))
    state.add_edge(node, '_res', state.add_write('info'), None, dace.Memlet('info[0]'))
    return node.expand(state) and _tasklet_code(sdfg)


def getrs_code(dtype: dace.typeclass) -> str:
    """The tasklet code the rocSOLVER triangular solve expands to."""
    sdfg = dace.SDFG(f'getrs_{dtype.to_string()}')
    sdfg.add_array('A', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('rhs', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('pivots', [N], dace.int32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('info', [1], dace.int32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Getrs('getrs')
    node.implementation = 'rocSOLVER'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('rhs'), None, node, '_rhs_in', dace.Memlet(f'rhs[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('pivots'), None, node, '_ipiv', dace.Memlet(f'pivots[0:{N}]'))
    state.add_edge(node, '_rhs_out', state.add_write('rhs'), None, dace.Memlet(f'rhs[0:{N}, 0:{N}]'))
    state.add_edge(node, '_res', state.add_write('info'), None, dace.Memlet('info[0]'))
    return node.expand(state) and _tasklet_code(sdfg)


def _tasklet_code(sdfg: dace.SDFG) -> str:
    """Every tasklet body in the expanded graph, concatenated."""
    return '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


@pytest.mark.parametrize('build', [getrf_code, getrs_code], ids=['getrf', 'getrs'])
@pytest.mark.parametrize('dtype', list(ROCBLAS_SPELLING), ids=lambda d: d.to_string())
def test_a_rocsolver_call_casts_to_the_rocblas_complex_type(build, dtype):
    rocblas_name, cuda_name = ROCBLAS_SPELLING[dtype]
    code = build(dtype)
    assert cuda_name not in code, code
    assert rocblas_name in code, code


@pytest.mark.parametrize('build', [getrf_code, getrs_code], ids=['getrf', 'getrs'])
def test_a_real_rocsolver_call_keeps_its_plain_c_type(build):
    """float64 is spelled the same in both dialects, so the cast is unchanged there."""
    code = build(dace.float64)
    assert '(double*)' in code, code


def gemm_code(dtype: dace.typeclass) -> str:
    """The tasklet code the rocBLAS GEMM expands to."""
    sdfg = dace.SDFG(f'gemm_{dtype.to_string()}')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Gemm('gemm')
    node.implementation = 'rocBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet(f'B[0:{N}, 0:{N}]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[0:{N}, 0:{N}]'))
    return node.expand(state) and _tasklet_code(sdfg)


def transpose_code(dtype: dace.typeclass) -> str:
    """The tasklet code the rocBLAS transpose (``geam``) expands to."""
    sdfg = dace.SDFG(f'transpose_{dtype.to_string()}')
    sdfg.add_array('A', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Transpose('transpose', dtype=dtype)
    node.implementation = 'rocBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_inp', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet(f'B[0:{N}, 0:{N}]'))
    return node.expand(state) and _tasklet_code(sdfg)


@pytest.mark.parametrize('build', [gemm_code, transpose_code], ids=['gemm', 'transpose'])
@pytest.mark.parametrize('dtype', list(ROCBLAS_SPELLING), ids=lambda d: d.to_string())
def test_a_rocblas_call_casts_to_the_rocblas_complex_type(build, dtype):
    """Neither the CUDA name (undeclared on ROCm) nor the hip vector type (declared, wrong type)."""
    rocblas_name, cuda_name = ROCBLAS_SPELLING[dtype]
    code = build(dtype)
    hip_name = cuda_name.replace('cu', 'hip')
    assert cuda_name not in code, code
    assert hip_name not in code, code
    assert rocblas_name in code, code


def gemv_code(dtype: dace.typeclass, alpha=1) -> str:
    """The tasklet code the rocBLAS matrix-vector product expands to.

    ``alpha=1`` takes the handle's device constants; any other value is built on the host and
    handed over under host pointer mode, which is the branch npbench ``vexx_k`` hit.
    """
    sdfg = dace.SDFG(f'gemv_{dtype.to_string()}_{alpha}')
    sdfg.add_array('A', [N, N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('x', [N], dtype, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('y', [N], dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Gemv('gemv', alpha=alpha)
    node.implementation = 'rocBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_A', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('x'), None, node, '_x', dace.Memlet(f'x[0:{N}]'))
    state.add_edge(node, '_y', state.add_write('y'), None, dace.Memlet(f'y[0:{N}]'))
    return node.expand(state) and _tasklet_code(sdfg)


@pytest.mark.parametrize('alpha', [1, 2], ids=['device-constant', 'host-coefficient'])
@pytest.mark.parametrize('dtype', list(ROCBLAS_SPELLING), ids=lambda d: d.to_string())
def test_a_rocblas_gemv_casts_coefficients_and_operands_to_the_rocblas_type(dtype, alpha):
    """``(cuDoubleComplex *)&alpha`` does not compile on ROCm, and neither does handing
    ``rocblas_zgemv`` the ``dace::complex128`` connector pointers without a cast."""
    rocblas_name, cuda_name = ROCBLAS_SPELLING[dtype]
    code = gemv_code(dtype, alpha)
    assert cuda_name not in code, code
    for operand in ('_A', '_x', '_y'):
        assert f'({rocblas_name} *){operand}' in code, code
