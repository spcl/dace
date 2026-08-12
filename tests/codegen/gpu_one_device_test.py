# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the one-process-one-GPU model: init selects the device, everything reads it back."""
import numpy as np

import dace
import pytest

from dace.library import gpu_device_setup_code
from dace.transformation.interstate import GPUTransformSDFG

# Every GPU library environment that creates a handle, and the accessor each one emits.
_ENVIRONMENTS = [
    ('dace.libraries.blas.environments.cublas', 'cuBLAS', 'cublas_handle'),
    ('dace.libraries.blas.environments.rocblas', 'rocBLAS', 'rocblas_handle'),
    ('dace.libraries.lapack.environments.cusolverdn', 'cuSolverDn', 'cusolverDn_handle'),
    ('dace.libraries.linalg.environments.cutensor', 'cuTensor', 'cutensor_handle'),
    ('dace.libraries.sparse.environments.cusparse', 'cuSPARSE', 'cusparse_handle'),
]


def _environment(module_name: str, cls_name: str):
    import importlib
    return getattr(importlib.import_module(module_name), cls_name)


def _gpu_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.arrays['A'].optional = False
    state = sdfg.add_state()
    state.add_mapped_tasklet('inc',
                             dict(i='0:8'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp + 1.0',
                             dict(out=dace.Memlet('A[i]')),
                             external_edges=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    return sdfg


def test_init_records_the_device_in_the_context():
    """The ordinal is stored, so later code has something to read instead of re-deriving it."""
    cu = next(o.clean_code for o in _gpu_sdfg('one_device_probe').generate_code() if o.language == 'cu')

    assert '__state->gpu_context->device = __dace_device;' in cu
    assert cu.index('__state->gpu_context = new') < cu.index('__state->gpu_context->device = __dace_device;'), \
        'the context has to exist before a field of it is assigned'


def test_device_defaults_to_the_one_the_process_is_on():
    """With no configuration, init takes the process's current device."""
    with dace.config.set_temporary('compiler', 'cuda', 'device', value=-1):
        cu = next(o.clean_code for o in _gpu_sdfg('device_default_probe').generate_code() if o.language == 'cu')

    assert 'int __dace_device = -1;' in cu
    assert 'cudaGetDevice(&__dace_device)' in cu, 'a -1 setting has to fall back to the current device'
    assert 'cudaSetDevice(__dace_device)' in cu


def test_configured_device_is_baked_into_init():
    """An explicit compiler.cuda.device is what init selects, without consulting the process."""
    with dace.config.set_temporary('compiler', 'cuda', 'device', value=2):
        cu = next(o.clean_code for o in _gpu_sdfg('device_pinned_probe').generate_code() if o.language == 'cu')

    assert 'int __dace_device = 2;' in cu
    assert '__dace_device >= count' in cu, 'a configured ordinal still has to be range-checked'
    assert cu.index('int __dace_device = 2;') < cu.index('cudaSetDevice(__dace_device)')


def test_the_device_is_selected_exactly_once():
    """Settable at init, immutable after: a second cudaSetDevice would undo the whole model."""
    with dace.config.set_temporary('compiler', 'cuda', 'device', value=1):
        cu = next(o.clean_code for o in _gpu_sdfg('device_once_probe').generate_code() if o.language == 'cu')

    assert cu.count('cudaSetDevice(') == 1, 'the device must be selected in exactly one place'


@pytest.mark.parametrize('module_name,cls_name,accessor', _ENVIRONMENTS)
def test_handle_setup_uses_the_recorded_device(module_name, cls_name, accessor):
    """Every environment takes its ordinal from the context. All five, since all five drifted."""
    env = _environment(module_name, cls_name)
    node = dace.sdfg.nodes.LibraryNode('probe')

    code = env.handle_setup_code(node)

    assert 'const int __dace_cuda_device = __state->gpu_context->device;' in code
    assert accessor in code, 'the handle accessor should still be emitted'
    assert 'location' not in code


@pytest.mark.parametrize('module_name,cls_name,accessor', _ENVIRONMENTS)
def test_per_node_gpu_placement_is_rejected(module_name, cls_name, accessor):
    """``location['gpu']`` fails loudly; silently dropping it would give the same wrong answer."""
    env = _environment(module_name, cls_name)
    node = dace.sdfg.nodes.LibraryNode('probe')
    node.location['gpu'] = 1

    with pytest.raises(ValueError, match='one GPU per process'):
        env.handle_setup_code(node)


def test_rejection_names_the_node_and_the_way_out():
    """A refusal is only useful if it says which node and what to do instead."""
    node = dace.sdfg.nodes.LibraryNode('my_gemm')
    node.location['gpu'] = 3

    with pytest.raises(ValueError) as excinfo:
        gpu_device_setup_code(node)

    message = str(excinfo.value)
    assert 'my_gemm' in message, 'the offending node has to be identifiable'
    assert '3' in message, 'the rejected placement has to be echoed back'
    assert 'CUDA_VISIBLE_DEVICES' in message, 'the message has to point at the supported way'


def test_gpu_placement_survives_an_empty_location():
    """The common case - no location at all - must not be mistaken for a placement."""
    node = dace.sdfg.nodes.LibraryNode('probe')
    assert node.location == {}
    assert '__state->gpu_context->device' in gpu_device_setup_code(node)


@pytest.mark.gpu
def test_a_configured_device_is_the_one_actually_used():
    """The setting reaches the hardware: the thread is still on it when the program returns."""
    import ctypes
    from ctypes.util import find_library

    cudart = None
    for name in (find_library('cudart'), 'libcudart.so', 'libcudart.so.13', 'libcudart.so.12'):
        if name:
            try:
                cudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
    if cudart is None:
        pytest.skip('libcudart is not loadable from this process')

    count = ctypes.c_int(0)
    cudart.cudaGetDeviceCount(ctypes.byref(count))
    if count.value < 2:
        pytest.skip(f'need two visible GPUs to tell a configured device apart from the default, saw {count.value}')

    a = np.random.rand(8)
    with dace.config.set_temporary('compiler', 'cuda', 'device', value=1):
        sdfg = _gpu_sdfg('device_runs_where_configured')
        sdfg(A=a)

    current = ctypes.c_int(-1)
    cudart.cudaGetDevice(ctypes.byref(current))
    assert current.value == 1, f'the program should have left this thread on the configured device, got {current.value}'


@pytest.mark.gpu
def test_a_gpu_library_node_still_runs():
    """End to end: a cuBLAS gemm still compiles and is correct through the rewritten path."""
    a = np.random.rand(64, 64)
    b = np.random.rand(64, 64)

    @dace.program
    def _matmul(x: dace.float64[64, 64], y: dace.float64[64, 64], z: dace.float64[64, 64]):
        z[:] = x @ y

    sdfg = _matmul.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.LibraryNode) and 'cuBLAS' in node.implementations:
            node.implementation = 'cuBLAS'

    z = np.zeros((64, 64))
    sdfg(x=a, y=b, z=z)
    assert np.allclose(z, a @ b)


if __name__ == '__main__':
    test_init_records_the_device_in_the_context()
    for _module, _cls, _accessor in _ENVIRONMENTS:
        test_handle_setup_uses_the_recorded_device(_module, _cls, _accessor)
        test_per_node_gpu_placement_is_rejected(_module, _cls, _accessor)
    test_rejection_names_the_node_and_the_way_out()
    test_gpu_placement_survives_an_empty_location()
    test_a_configured_device_is_the_one_actually_used()
    test_a_gpu_library_node_still_runs()
