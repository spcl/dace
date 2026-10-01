# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The offloaded SDFG computes what the original computes, with device memory emulated on the host.

Device containers become separate host buffers, filled with NaN where the pass only declares them, and kernels run
as sequential loops. A copy that is missing, stale or run on the wrong path then changes the result, which needs no
GPU to observe.
"""
import copy

import numpy as np
import pytest

import dace
from dace import dtypes, Memlet
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator
from shared_graphs import host_write_then_kernel_arm

N = dace.symbol('N')
LENGTH = 8
CONDITIONS = [0, 1, 3, 10]


def emulate_device(sdfg: dace.SDFG) -> dace.SDFG:
    """Run ``sdfg``'s device code on the host; the twins the pass declared start out as NaN."""
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            if desc.storage == dtypes.StorageType.GPU_Global:
                desc.storage = dtypes.StorageType.CPU_Heap
        for node, _ in nested.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule in dtypes.GPU_SCHEDULES:
                node.map.schedule = dtypes.ScheduleType.Sequential
    twins = [
        name for name, desc in sdfg.arrays.items() if name.endswith(('_gpu', '_host')) and desc.transient
        and desc.dtype == dace.float64 and not isinstance(desc, dace.data.View)
    ]
    if twins:
        poison = sdfg.add_state_before(sdfg.start_block, 'poison', is_start_block=True)
        for name in twins:
            ranges = {f'i{axis}': f'0:{extent}' for axis, extent in enumerate(sdfg.arrays[name].shape)}
            poison.add_mapped_tasklet(f'poison_{name}',
                                      ranges, {},
                                      'o = std::nan("");', {'o': Memlet(f'{name}[{", ".join(ranges)}]')},
                                      external_edges=True,
                                      language=dace.Language.CPP)
    return sdfg


def offloaded_on_the_host(sdfg: dace.SDFG) -> dace.SDFG:
    off = copy.deepcopy(sdfg)
    off.name = f'{sdfg.name}_offloaded'
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(off, {})
    emulate_device(off).validate()
    return off


@dace.program
def host_read_on_an_edge(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for i in dace.map[0:N]:
        A[i] = A[i] + 1.0
    x = A[0]
    if x > c:
        for i in dace.map[0:N]:
            B[i] = A[i] * x


@dace.program
def host_write_in_a_loop_arm(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for k in range(4):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1.0
        if k < c:
            A[0] = A[0] + 1.0


@dace.program
def break_after_host_read(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for k in range(5):
        B[0] = A[0]
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0
        if k > c:
            break
    B[1] = A[1]


@dace.program
def return_after_a_kernel(A: dace.float64[N], B: dace.float64[N], c: dace.int64):
    for i in dace.map[0:N]:
        A[i] = A[i] + 1.0
    if c > 0:
        return
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0


PROGRAMS = {
    'host_write_then_kernel_arm': host_write_then_kernel_arm,
    'host_read_on_an_edge': lambda: host_read_on_an_edge.to_sdfg(simplify=True),
    'host_write_in_a_loop_arm': lambda: host_write_in_a_loop_arm.to_sdfg(simplify=True),
    'break_after_host_read': lambda: break_after_host_read.to_sdfg(simplify=True),
    'return_after_a_kernel': lambda: return_after_a_kernel.to_sdfg(simplify=True),
}


@pytest.fixture(scope='module', params=sorted(PROGRAMS))
def compiled(request, tmp_path_factory):
    """The original and the offloaded program, built in a folder of their own: parallel workers share names."""
    original = PROGRAMS[request.param]()
    with dace.config.set_temporary('default_build_folder', value=str(tmp_path_factory.mktemp(request.param))):
        return original.compile(), offloaded_on_the_host(original).compile()


@pytest.mark.parametrize('c', CONDITIONS)
def test_the_offloaded_program_computes_what_the_original_computes(compiled, c):
    reference, offloaded = compiled
    rng = np.random.default_rng(7)
    inputs = {name: rng.random(LENGTH) + 1.0 for name in 'ABC'}
    want = {name: value.copy() for name, value in inputs.items()}
    got = {name: value.copy() for name, value in inputs.items()}
    names = {name for name in want if name in reference.sdfg.arglist()}

    reference(**{name: want[name] for name in names}, c=c, N=LENGTH)
    offloaded(**{name: got[name] for name in names}, c=c, N=LENGTH)

    for name in names:
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
