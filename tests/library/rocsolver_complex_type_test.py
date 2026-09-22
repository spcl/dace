# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A rocSOLVER call casts its operands to the rocBLAS complex type, not the CUDA one.

The GPU solver expansions take their vendor C type from ``cublas_type_metadata``, which names the
CUDA types. ``float`` and ``double`` are spelled the same in both dialects, so the real paths were
fine and the complex ones emitted ``(cuDoubleComplex*)`` into a ROCm build, where that type does
not exist: quatrex_rgf's complex128 ``Inv`` reached hipcc and failed with ``'cuDoubleComplex' was
not declared in this scope`` on the canon GPU column.
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.lapack import Getrf, Getrs

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
