# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Behavioral tests for __dace_init_cuda: the inherited-error drain (a foreign
pending error must not fail the next program) and one-GPU-per-process."""
import ctypes
import importlib
from ctypes.util import find_library

import numpy as np

import dace
import pytest

from dace.transformation.interstate import GPUTransformSDFG


def _gpu_sdfg(name: str = 'drain_probe') -> dace.SDFG:
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


def test_gpu_init_template_substitution_does_not_break_comments():
    """A ``{placeholder}`` in a generated C++ comment is substituted there too, so anything
    expanding to statements breaks the comment. Checked against the template source: reproducing
    it needs a non-empty ``{initcode}``. ``{backend}`` is exempt, expanding to one identifier."""
    import re
    from pathlib import Path

    import dace.codegen.targets.cuda as cuda_target

    source = Path(cuda_target.__file__).read_text().split('\n')
    placeholder = re.compile(r'\{([a-z_][a-z_0-9]*)\}')
    offenders = []
    for lineno, line in enumerate(source, 1):
        if not line.strip().startswith('//'):
            continue
        for match in placeholder.finditer(line):
            if match.group(1) != 'backend':
                offenders.append(f'{cuda_target.__file__}:{lineno}: {{{match.group(1)}}} in {line.strip()!r}')

    assert not offenders, ('a format placeholder inside a generated C++ comment is substituted there, '
                           'so anything expanding to statements breaks the comment:\n  ' + '\n  '.join(offenders))


def _load_cudart():
    """The runtime as a ctypes handle: generated modules link it dynamically, so it is the same
    instance and the same per-thread error slot."""
    for name in (find_library('cudart'), 'libcudart.so', 'libcudart.so.13', 'libcudart.so.12'):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_reduction_axes = (0, )


@dace.program
def _summed(a, b):
    b[:] = np.sum(a, axis=_reduction_axes)


@pytest.mark.gpu
def test_foreign_error_is_not_charged_to_the_next_program():
    """An error left pending by another GPU user must not fail the next SDFG.

    A CUB reduction is the victim: its size query reads the device through ``cudaGetDevice``, gets
    the pending error back, and reports ``cudaErrorInvalidDevice`` - so ``invalid argument (1)``
    surfaces as ``invalid device ordinal (101)``. Poisoned via ctypes; cupy clears the slot.
    """
    cudart = _load_cudart()
    if cudart is None:
        pytest.skip('libcudart is not loadable from this process')

    a = np.random.rand(4096).astype(np.float64)
    b = np.zeros(1, dtype=np.float64)
    sdfg = _summed.to_sdfg(a, b)
    sdfg.apply_gpu_transformations()
    import dace.libraries.standard as std
    reduce_node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
    reduce_node.implementation = 'CUDA (device)'
    csdfg = sdfg.compile()

    cudart.cudaFree(ctypes.c_void_p(0))  # initialize the runtime before handing it a bad call
    rc = cudart.cudaMemcpy(ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_size_t(1), ctypes.c_int(2))
    if rc == 0 or cudart.cudaPeekAtLastError() == 0:
        pytest.skip('this runtime build left nothing pending, so there is nothing to inherit')

    csdfg(a=a, b=b)  # must not raise: the pending error is not this SDFG's
    assert np.allclose(b, np.sum(a, axis=_reduction_axes))


# Every GPU library environment that creates a handle, and the accessor each emits.
_ENVIRONMENTS = [
    ('dace.libraries.blas.environments.cublas', 'cuBLAS', 'cublas_handle'),
    ('dace.libraries.blas.environments.rocblas', 'rocBLAS', 'rocblas_handle'),
    ('dace.libraries.lapack.environments.cusolverdn', 'cuSolverDn', 'cusolverDn_handle'),
    ('dace.libraries.linalg.environments.cutensor', 'cuTensor', 'cutensor_handle'),
    ('dace.libraries.sparse.environments.cusparse', 'cuSPARSE', 'cusparse_handle'),
]


def test_the_device_ordinal_is_not_configurable():
    """A configuration entry for it is the thing that was wrong, not its default."""
    assert 'device' not in dace.Config.get('compiler', 'cuda')


@pytest.mark.parametrize('module_name,cls_name,accessor', _ENVIRONMENTS)
def test_handle_setup_takes_no_device(module_name, cls_name, accessor):
    """All five, since all five carried their own copy of the location parsing. The handle lives on
    the one device init selected, so there is no ordinal left to pass it."""
    env = getattr(importlib.import_module(module_name), cls_name)
    node = dace.sdfg.nodes.LibraryNode('probe')

    code = env.handle_setup_code(node)
    assert f'{accessor}.Get()' in code
    assert '__dace_cuda_device' not in code

    node.location['gpu'] = 3
    with pytest.raises(ValueError, match='one GPU per process') as excinfo:
        env.handle_setup_code(node)
    assert 'probe' in str(excinfo.value) and '3' in str(excinfo.value)


@pytest.mark.gpu
def test_the_program_runs_on_device_zero_whatever_the_caller_was_on():
    """``__dace_init_cuda`` SELECTS device 0 rather than inheriting the caller's, so the thread is
    on it when the program returns. Started from a different device, which is the only way to tell
    selecting apart from inheriting."""
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
        pytest.skip(f'need two visible GPUs to tell selecting apart from inheriting, saw {count.value}')

    cudart.cudaSetDevice(1)
    a = np.random.rand(8)
    _gpu_sdfg('device_runs_on_zero')(A=a)

    current = ctypes.c_int(-1)
    cudart.cudaGetDevice(ctypes.byref(current))
    assert current.value == 0


if __name__ == '__main__':
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith('test_') and not hasattr(fn, 'pytestmark'):
            fn()
