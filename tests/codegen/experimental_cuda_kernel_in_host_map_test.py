# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU_Device map nested inside a host-scheduled map (the CLOUDSC shape: an outer block loop
orchestrates on the host, the compute map inside it is the kernel).

The stream edge then reaches the kernel map entry *through* the enclosing MapEntry, so the
dynamic-map-input source is that MapEntry and not the ``gpu_streams`` AccessNode.
"""
import dace
from dace import dtypes

N = dace.symbol('N', dace.int64)
K = dace.symbol('K', dace.int64)


def build_kernel_in_host_map_sdfg() -> dace.SDFG:
    """Outer ``blocks`` map stays Sequential (host); the inner ``kernel`` map is GPU_Device."""
    gpu = dtypes.StorageType.GPU_Global
    sdfg = dace.SDFG('kernel_in_host_map')
    sdfg.add_array('a', (K, N), dace.float64, storage=gpu)
    sdfg.add_array('b', (K, N), dace.float64, storage=gpu)

    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('blocks', {'ibl': '0:N'}, schedule=dtypes.ScheduleType.Sequential)
    kernel_entry, kernel_exit = state.add_map('kernel', {'jk': '0:K'}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')

    state.add_memlet_path(state.add_read('a'),
                          outer_entry,
                          kernel_entry,
                          tasklet,
                          dst_conn='inp',
                          memlet=dace.Memlet('a[jk, ibl]'))
    state.add_memlet_path(tasklet,
                          kernel_exit,
                          outer_exit,
                          state.add_write('b'),
                          src_conn='out',
                          memlet=dace.Memlet('b[jk, ibl]'))
    return sdfg


def generate_experimental_cuda(sdfg: dace.SDFG):
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        return sdfg.generate_code()


def test_kernel_nested_in_host_map_generates_code():
    """Regression: resolving the stream descriptor via ``edge.src.desc`` raised
    ``AttributeError: 'MapEntry' object has no attribute 'desc'`` for this shape."""
    sdfg = build_kernel_in_host_map_sdfg()
    sdfg.validate()
    code_objects = generate_experimental_cuda(sdfg)

    kernels = sum(obj.clean_code.count('__global__ void') for obj in code_objects)
    assert kernels == 1, f'expected exactly one kernel, generated {kernels}'


def test_nested_kernel_launches_on_the_real_stream():
    """The wrapper must receive the stream that reached it through the enclosing map, not the
    ``nullptr`` default-stream fallback used when no stream edge is wired at all."""
    sdfg = build_kernel_in_host_map_sdfg()
    launches = [
        line.strip() for obj in generate_experimental_cuda(sdfg) for line in obj.clean_code.splitlines()
        if '__dace_runkernel_' in line and line.strip().endswith(');') and 'DACE_EXPORTED' not in line
    ]
    assert launches, 'no kernel wrapper call was generated'
    assert all('__dace_current_stream' in line for line in launches), launches
    assert not any('nullptr' in line for line in launches), launches


if __name__ == '__main__':
    test_kernel_nested_in_host_map_generates_code()
    test_nested_kernel_launches_on_the_real_stream()
