# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The default GPU stream model: every kernel launches on stream 0 and the program synchronises
once, rather than spreading kernels over several streams and syncing each one.

This is the schedule CLOUDSC needs. Its whole computation runs on the device (bar some scalar
init), so consecutive kernels are dependent and there is nothing for extra streams to overlap --
one stream plus a single terminal sync is both correct and the cheapest schedule. The legacy
backend instead hands out four streams and emits eight syncs for the same graph.

The kernels here are nested inside a host ``nblocks`` map, the CLOUDSC shape, which also exercises
the stream lookup fixed in ``KernelSpec.__init__``.

Codegen-only: no GPU and no nvcc required.
"""
import re

import dace
import pytest
from dace import dtypes

KLEV = dace.symbol('klev')
NBLOCKS = dace.symbol('nblocks')


def add_block_kernel(sdfg: dace.SDFG, label: str, src: str, dst: str, code: str) -> None:
    """One CLOUDSC-shaped state: a Sequential host ``nblocks`` map wrapping a GPU_Device map."""
    state = sdfg.add_state(label)
    block_entry, block_exit = state.add_map(f'{label}_blocks', {'ibl': '0:nblocks'},
                                            schedule=dtypes.ScheduleType.Sequential)
    kernel_entry, kernel_exit = state.add_map(f'{label}_kernel', {'jk': '0:klev'},
                                              schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet(f'{label}_compute', {'inp'}, {'out'}, code)
    state.add_memlet_path(state.add_read(src),
                          block_entry,
                          kernel_entry,
                          tasklet,
                          dst_conn='inp',
                          memlet=dace.Memlet(f'{src}[jk, ibl]'))
    state.add_memlet_path(tasklet,
                          kernel_exit,
                          block_exit,
                          state.add_write(dst),
                          src_conn='out',
                          memlet=dace.Memlet(f'{dst}[jk, ibl]'))


def build_multi_kernel_sdfg() -> dace.SDFG:
    """Three dependent kernels -- enough to tell one sync per kernel apart from one at the end."""
    sdfg = dace.SDFG('gpu_stream_model')
    for name in ('a', 'b', 'c', 'd'):
        sdfg.add_array(name, [KLEV, NBLOCKS], dace.float64, storage=dtypes.StorageType.GPU_Global)

    add_block_kernel(sdfg, 'k0', 'a', 'b', 'out = inp * 2.0')
    add_block_kernel(sdfg, 'k1', 'b', 'c', 'out = inp + 1.0')
    add_block_kernel(sdfg, 'k2', 'c', 'd', 'out = inp * 0.5')

    states = list(sdfg.states())
    for first, second in zip(states, states[1:]):
        sdfg.add_edge(first, second, dace.InterstateEdge())
    return sdfg


@pytest.fixture(scope='module')
def generated_source() -> str:
    sdfg = build_multi_kernel_sdfg()
    sdfg.validate()
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        objects = sdfg.generate_code()
    assert any(obj.language == 'cu' for obj in objects), 'no CUDA code object was generated'
    return '\n'.join(obj.clean_code for obj in objects)


def test_every_kernel_is_generated(generated_source: str):
    assert generated_source.count('__global__ void') == 3


def test_all_kernels_use_stream_zero(generated_source: str):
    """No kernel may be handed a stream other than 0."""
    streams = set(re.findall(r'streams\[(\d+)\]', generated_source))
    assert streams == {'0'}, sorted(streams)


def test_synchronises_once(generated_source: str):
    """One stream sync for the whole program, not one per kernel."""
    syncs = re.findall(r'(?:cuda|hip)StreamSynchronize', generated_source)
    assert len(syncs) == 1, f'expected a single sync, found {len(syncs)}'
