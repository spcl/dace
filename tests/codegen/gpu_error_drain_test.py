# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the inherited-GPU-error drain (``__dace_gpu_drain_error``).

All but the last assert on generated source only, so they need neither a GPU nor a CUDA
toolchain. The last one is marked ``gpu``: it reproduces the failure end to end.
"""
import ctypes
from ctypes.util import find_library

import numpy as np

import dace
import pytest

from dace.transformation.interstate import GPUTransformSDFG


def _gpu_sdfg(name: str = 'drain_probe') -> dace.SDFG:
    """A minimal SDFG that generates GPU code."""
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


def _sources(sdfg: dace.SDFG):
    """The generated CUDA file and the frame (host) file, as strings."""
    objs = sdfg.generate_code()
    cu = next(o.clean_code for o in objs if o.language == 'cu')
    frame = next(o.clean_code for o in objs if o.language == 'cpp' and o.name == sdfg.name)
    return cu, frame


def test_gpu_drain_emitted_in_init_and_per_call():
    """Defined in the .cu, called from init, and again per call - init runs only once, but a
    foreign error can land between any two calls."""
    cu, frame = _sources(_gpu_sdfg())

    assert 'void __dace_gpu_drain_error(' in cu  # defined in the CUDA file
    assert '__dace_gpu_drain_error(__state);' in cu  # and called by the initializer

    # The host file cannot include the CUDA headers, so it declares the function and links
    # against the .cu definition - the same arrangement as __dace_init_cuda.
    assert 'DACE_EXPORTED void __dace_gpu_drain_error(' in frame
    decl = frame.index('DACE_EXPORTED void __dace_gpu_drain_error(')
    call = frame.index('__dace_gpu_drain_error(__state);')
    assert decl < call, 'the declaration must precede the call'
    assert call < frame.index('_internal(__state'), 'the drain must run before the program body'


def test_gpu_drain_absent_without_gpu_code():
    """A CPU-only SDFG must not reference the drain: the symbol only exists when the CUDA
    target emitted it, so an unconditional call would be a link error."""
    sdfg = dace.SDFG('drain_cpu_only_probe')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.arrays['A'].optional = False
    state = sdfg.add_state()
    state.add_mapped_tasklet('inc',
                             dict(i='0:8'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp + 1.0',
                             dict(out=dace.Memlet('A[i]')),
                             external_edges=True)
    frame = next(o.clean_code for o in sdfg.generate_code() if o.language == 'cpp' and o.name == 'drain_cpu_only_probe')
    assert '__dace_gpu_drain_error' not in frame


def test_gpu_init_template_substitution_does_not_break_comments():
    """Regression: a ``{placeholder}`` inside a C++ comment in the init template is substituted
    there too, so anything expanding to statements breaks the comment.

    Checked against the template source, not generated output: reproducing it needs a non-empty
    ``{initcode}``, so a test on a simpler SDFG would pass while the bug is present.
    ``{backend}`` is exempt - it expands to one identifier.
    """
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


def test_gpu_mempool_setup_is_checked_and_follows_context_creation():
    """The pool calls are wrapped and run after the context exists - DACE_GPU_CHECK records
    into it, so a checked call placed earlier could not record."""
    sdfg = dace.SDFG('drain_pool_probe')
    sdfg.add_array('A', [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['A'].optional = False
    sdfg.add_transient('t', [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['t'].pool = True
    state = sdfg.add_state()
    state.add_nedge(state.add_read('A'), state.add_write('t'), dace.Memlet('A[0:16]'))
    state.add_nedge(state.add_read('t'), state.add_write('A'), dace.Memlet('t[0:16]'))

    cu, _ = _sources(sdfg)
    if 'MemPool_t' not in cu:
        pytest.skip('this SDFG did not request pooled allocation')

    assert 'DACE_GPU_CHECK(cudaDeviceGetDefaultMemPool' in cu
    assert 'DACE_GPU_CHECK(cudaMemPoolSetAttribute' in cu
    assert cu.index('__state->gpu_context = new') < cu.index('MemPool_t')

    # The pool has to be the one belonging to the device this state runs on. A literal 0 is a
    # valid ordinal whenever any device exists, so this never fails loudly: it silently applies
    # the release threshold to a pool the allocations below never touch.
    assert 'DACE_GPU_CHECK(cudaDeviceGetDefaultMemPool(&mempool, __dace_device))' in cu


def test_gpu_init_selects_the_current_device_explicitly():
    """Init reads the device, range-checks it, and selects it - which also forces the context to
    exist here, where a placement failure is still attributable."""
    cu, _ = _sources(_gpu_sdfg('drain_device_probe'))

    assert 'cudaGetDevice(&__dace_device)' in cu
    assert 'cudaSetDevice(__dace_device)' in cu
    assert '__dace_device >= count' in cu, 'the ordinal must be range-checked against the device count'
    assert cu.index('cudaSetDevice(__dace_device)') < cu.index('__dace_gpu_drain_error(__state);'), \
        'the device must be bound before the slot is claimed, so the drain covers the bound thread'


def _load_cudart():
    """The CUDA runtime as a ctypes handle, or None. Generated modules link it dynamically, so
    this is the same library instance and therefore the same per-thread error slot."""
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

    The victim is a CUB reduction: its size query reads the current device through
    ``cudaGetDevice``, gets the pending error back, and reports ``cudaErrorInvalidDevice`` - so
    ``invalid argument (1)`` surfaces as ``invalid device ordinal (101)``. Poisoned through
    ctypes, not cupy, which clears the slot when it raises.
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


if __name__ == '__main__':
    test_gpu_drain_emitted_in_init_and_per_call()
    test_gpu_drain_absent_without_gpu_code()
    test_gpu_init_template_substitution_does_not_break_comments()
    test_gpu_mempool_setup_is_checked_and_follows_context_creation()
    test_gpu_init_selects_the_current_device_explicitly()
    test_foreign_error_is_not_charged_to_the_next_program()
