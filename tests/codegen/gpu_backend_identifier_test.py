# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU code generator spells most runtime calls by prefixing the backend onto one shared name.
That works because CUDA and HIP agree on the rest of the identifier -- but they do not always agree,
and a fabricated name is only caught when someone compiles that path on that backend. These tests
generate code for both backends on any machine, GPU or not, so a divergence fails in CPU CI.

The backend is forced through the environment rather than :func:`dace.config.set_temporary`, because
``DACE_*`` variables outrank the configuration and the GPU CI images set this one.
"""
import os

import pytest

import dace

N = dace.symbol('N')


@dace.program
def persistent_axpy(x: dace.float32[N], y: dace.float32[N]):
    for _ in dace.map[0:1]:
        for i in dace.map[0:N]:
            y[i] = y[i] + x[i]


def generated_gpu_code(backend: str) -> str:
    """All GPU code DaCe emits for a persistent map under ``backend``."""
    sdfg = persistent_axpy.to_sdfg()
    sdfg.apply_gpu_transformations()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.map.schedule = (dace.ScheduleType.GPU_Persistent
                                     if state.entry_node(node) is None else dace.ScheduleType.GPU_Device)

    previous = os.environ.get('DACE_compiler_cuda_backend')
    os.environ['DACE_compiler_cuda_backend'] = backend
    try:
        codes = sdfg.generate_code()
    finally:
        if previous is None:
            del os.environ['DACE_compiler_cuda_backend']
        else:
            os.environ['DACE_compiler_cuda_backend'] = previous
    # Every object, not the GPU one alone: the GPU object's target type is '' under CUDA and 'hip'
    # under HIP, so selecting it by type silently selects nothing on one of the two backends.
    return '\n'.join(code.clean_code for code in codes)


@pytest.mark.parametrize('backend,attribute', [('cuda', 'cudaDevAttrMultiProcessorCount'),
                                               ('hip', 'hipDeviceAttributeMultiprocessorCount')])
def test_persistent_map_queries_sm_count_by_the_backend_s_own_name(backend, attribute):
    """The SM-count enumerator is the one name the two backends spell differently."""
    code = generated_gpu_code(backend)
    assert 'DeviceGetAttribute' in code, 'persistent map did not emit the SM-count query'
    assert attribute in code
    # hipDevAttrMultiProcessorCount is what prefixing the backend produces, and HIP declares no such
    # identifier -- every persistent-map kernel failed to compile under HIP because of it.
    assert 'DevAttrMultiProcessorCount' not in code or backend == 'cuda'


if __name__ == '__main__':
    test_persistent_map_queries_sm_count_by_the_backend_s_own_name('cuda', 'cudaDevAttrMultiProcessorCount')
    test_persistent_map_queries_sm_count_by_the_backend_s_own_name('hip', 'hipDeviceAttributeMultiprocessorCount')
