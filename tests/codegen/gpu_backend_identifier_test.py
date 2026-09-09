# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU code generator spells most runtime calls by prefixing the backend onto one shared name.
That works because CUDA and HIP agree on the rest of the identifier -- but they do not always agree,
and a fabricated name is only caught when someone compiles that path on that backend. These tests
generate code for both backends on any machine, GPU or not, so a divergence fails in CPU CI.

The backend is forced through the environment rather than :func:`dace.config.set_temporary`, because
``DACE_*`` variables outrank the configuration and the GPU CI images set this one.
"""
import contextlib
import os
from typing import Optional

import pytest

import dace
from dace.codegen import common

N = dace.symbol('N')


@contextlib.contextmanager
def forced_environment(**values: Optional[str]):
    """Sets environment variables for the duration of the block, restoring what was there before.

    Clears the HIP platform cache around the block: it is read once per process, which is right in
    a real run and wrong for a test that has to look at both platforms.
    """
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is not None:
            os.environ[name] = value
    if 'HIP_PLATFORM' in values:
        common.get_hip_platform.cache_clear()
    try:
        yield
    finally:
        for name, was in previous.items():
            if was is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = was
        if 'HIP_PLATFORM' in values:
            common.get_hip_platform.cache_clear()


@dace.program
def persistent_axpy(x: dace.float32[N], y: dace.float32[N]):
    for _ in dace.map[0:1]:
        for i in dace.map[0:N]:
            y[i] = y[i] + x[i]


@dace.program
def pinned_roundtrip(a: dace.float64[N]):
    staging = dace.define_local([N], dace.float64, storage=dace.StorageType.CPU_Pinned)
    staging[:] = a
    a[:] = staging


def generated_code(program, backend: str) -> str:
    """Every code object ``program`` produces under ``backend``, joined.

    Every object, not the GPU one alone: the GPU object's target type is '' under CUDA and 'hip'
    under HIP, so selecting it by type silently selects nothing on one of the two backends.
    """
    with forced_environment(DACE_compiler_cuda_backend=backend):
        return '\n'.join(code.clean_code for code in program.generate_code())


def persistent_map_sdfg() -> dace.SDFG:
    sdfg = persistent_axpy.to_sdfg()
    sdfg.apply_gpu_transformations()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.map.schedule = (dace.ScheduleType.GPU_Persistent
                                     if state.entry_node(node) is None else dace.ScheduleType.GPU_Device)
    return sdfg


@pytest.mark.parametrize('backend,attribute', [('cuda', 'cudaDevAttrMultiProcessorCount'),
                                               ('hip', 'hipDeviceAttributeMultiprocessorCount')])
def test_persistent_map_queries_sm_count_by_the_backend_s_own_name(backend, attribute):
    """The SM-count enumerator is the one name the two backends spell differently."""
    code = generated_code(persistent_map_sdfg(), backend)
    assert 'DeviceGetAttribute' in code, 'persistent map did not emit the SM-count query'
    assert attribute in code
    # hipDevAttrMultiProcessorCount is what prefixing the backend produces, and HIP declares no such
    # identifier -- every persistent-map kernel failed to compile under HIP because of it.
    assert 'DevAttrMultiProcessorCount' not in code or backend == 'cuda'


@pytest.mark.parametrize('backend,alloc,free', [('cuda', 'cudaMallocHost', 'cudaFreeHost'),
                                                ('hip', 'hipHostMalloc', 'hipHostFree')])
def test_pinned_host_memory_uses_the_backend_s_own_allocator(backend, alloc, free):
    """Pinned host memory is another name the two do not share."""
    code = generated_code(pinned_roundtrip.to_sdfg(simplify=True), backend)
    assert alloc in code, 'no pinned host allocation was emitted'
    assert free in code
    # HIP does carry hipMallocHost/hipFreeHost, but they are deprecated aliases declared to take a
    # plain void** where CUDA's take a template parameter, so the emitted double** did not convert.
    assert 'MallocHost' not in code or backend == 'cuda'
    assert 'FreeHost' not in code or backend == 'cuda'


@pytest.mark.parametrize('backend,platform,available', [('cuda', None, True), ('hip', 'nvidia', True),
                                                        ('hip', 'amd', False)])
def test_cutensor_availability_follows_the_platform_not_the_backend(backend, platform, available):
    """HIP's NVIDIA platform links cuTENSOR like any CUDA build; only a real AMD build cannot."""
    from dace.libraries.linalg import environments
    with forced_environment(DACE_compiler_cuda_backend=backend, HIP_PLATFORM=platform):
        assert environments.cuTensor.is_available() is available


if __name__ == '__main__':
    test_persistent_map_queries_sm_count_by_the_backend_s_own_name('cuda', 'cudaDevAttrMultiProcessorCount')
    test_persistent_map_queries_sm_count_by_the_backend_s_own_name('hip', 'hipDeviceAttributeMultiprocessorCount')
    test_pinned_host_memory_uses_the_backend_s_own_allocator('cuda', 'cudaMallocHost', 'cudaFreeHost')
    test_pinned_host_memory_uses_the_backend_s_own_allocator('hip', 'hipHostMalloc', 'hipHostFree')
    test_cutensor_availability_follows_the_platform_not_the_backend('cuda', None, True)
    test_cutensor_availability_follows_the_platform_not_the_backend('hip', 'nvidia', True)
    test_cutensor_availability_follows_the_platform_not_the_backend('hip', 'amd', False)
