# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host levels below the top: a map kept on the host launches the kernels in it, and a nested SDFG at a host
level places its own data."""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = dace.symbol('N')


def row_sums_with_a_device_reduce() -> dace.SDFG:
    """A map over rows around a reduce whose chosen expansion is a host-issued cub call."""
    sdfg = dace.SDFG('row_sums_with_a_device_reduce')
    sdfg.add_array('A', [8, 64], dace.float64)
    sdfg.add_array('out', [8], dace.float64)
    state = sdfg.add_state('rows')
    entry, exit_node = state.add_map('rows', {'i': '0:8'})
    reduce = state.add_reduce('lambda a, b: a + b', None, 0)
    reduce.implementation = 'CUDA (device)'
    state.add_memlet_path(state.add_read('A'), entry, reduce, memlet=dace.Memlet('A[i, 0:64]'))
    state.add_memlet_path(reduce, exit_node, state.add_write('out'), memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def test_a_map_around_a_host_issued_library_call_launches_it_from_the_host():
    """Inside a kernel the cub device reduce did not compile; the map now stays host code around the call."""
    sdfg = row_sums_with_a_device_reduce()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    schedules = {
        node.label: node.schedule
        for node, _ in sdfg.all_nodes_recursive() if isinstance(node, (dace.nodes.MapEntry, dace.nodes.LibraryNode))
    }
    assert schedules == {'rows': dtypes.ScheduleType.Sequential, 'Reduce': dtypes.ScheduleType.GPU_Device}
    state = next(state for state in sdfg.states() if state.label == 'rows')
    reduce = next(node for node in state.nodes() if isinstance(node, dace.nodes.LibraryNode))
    operands = [edge.data.data for edge in state.memlet_path(state.in_edges(reduce)[0])]
    assert all(sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global for name in operands), operands


def test_a_map_around_an_in_kernel_reduce_is_the_kernel():
    """A cub block reduce expands to device code, so the map around it is the kernel."""
    sdfg = row_sums_with_a_device_reduce()
    next(node for node, _ in sdfg.all_nodes_recursive()
         if isinstance(node, dace.nodes.LibraryNode)).implementation = 'CUDA (block)'
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    rows = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry))
    assert rows.map.schedule == dtypes.ScheduleType.GPU_Device


@pytest.mark.gpu
def test_a_host_launched_device_reduce_computes_what_numpy_computes():
    sdfg = row_sums_with_a_device_reduce()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    A = np.random.default_rng(0).random((8, 64))
    out = np.zeros(8)
    sdfg(A=A, out=out)
    np.testing.assert_allclose(out, A.sum(axis=1))


@dace.program
def double(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = x[i] * 2.0


@dace.program
def calls_double(x: dace.float64[N], y: dace.float64[N]):
    double(x, y)


def test_a_nested_sdfg_at_the_top_level_is_a_host_level_of_its_own():
    """Its map is the kernel, and the arrays it binds reach it in device memory."""
    sdfg = calls_double.to_sdfg(simplify=False)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    nested = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG))
    kernels = [node for node, _ in nested.sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert kernels and all(entry.map.schedule == dtypes.ScheduleType.GPU_Device for entry in kernels)
    assert all(desc.storage == dtypes.StorageType.GPU_Global for name, desc in nested.sdfg.arrays.items()
               if not desc.transient), nested.sdfg.arrays


@pytest.mark.gpu
def test_a_nested_host_level_computes_what_numpy_computes():
    sdfg = calls_double.to_sdfg(simplify=False)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    x = np.arange(16, dtype=np.float64)
    y = np.zeros(16)
    sdfg(x=x, y=y, N=16)
    np.testing.assert_allclose(y, x * 2.0)


def body_reading_a_device_element_on_the_host() -> dace.SDFG:
    """``A`` handed over in device memory as the connector ``s`` of a nested SDFG whose host loop adds ``s[0]`` to
    every ``out[i]`` (No-View nested SDFGs pass the whole array, read at its element inside)."""
    body = dace.SDFG('add_on_the_host')
    body.add_array('s', [8], dace.float64)
    body.add_array('out', [8], dace.float64)
    loop = dace.sdfg.state.LoopRegion('over_out', 'i < 8', 'i', 'i = 0', 'i = i + 1')
    body.add_node(loop, is_start_block=True)
    step = loop.add_state('add', is_start_block=True)
    add = step.add_tasklet('add', {'o_in': None, 's_in': None}, {'o': None}, 'o = o_in + s_in')
    step.add_edge(step.add_read('out'), None, add, 'o_in', dace.Memlet('out[i]'))
    step.add_edge(step.add_read('s'), None, add, 's_in', dace.Memlet('s[0]'))
    step.add_edge(add, 'o', step.add_write('out'), None, dace.Memlet('out[i]'))

    sdfg = dace.SDFG('body_reading_a_device_element_on_the_host')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('out', [8], dace.float64)
    state = sdfg.add_state('call')
    call = state.add_nested_sdfg(body, {'s': None, 'out': None}, {'out': None})
    state.add_edge(state.add_read('A'), None, call, 's', dace.Memlet('A[0:8]'))
    state.add_edge(state.add_read('out'), None, call, 'out', dace.Memlet('out[0:8]'))
    state.add_edge(call, 'out', state.add_write('out'), None, dace.Memlet('out[0:8]'))
    sdfg.validate()
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


def test_a_device_element_a_body_reads_on_the_host_is_staged_on_the_host():
    """npbench azimint_hist: a scalar connector bound to device memory and read by host code in the body."""
    sdfg = body_reading_a_device_element_on_the_host()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    state = next(state for state in sdfg.states() if state.label == 'call')
    call = next(node for node in state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    bound = [edge.data.data for edge in state.in_edges(call) if edge.dst_conn == 's']
    assert bound and all(sdfg.arrays[name].storage == dtypes.StorageType.Default for name in bound), bound


@pytest.mark.gpu
def test_a_device_element_staged_for_a_host_body_computes_what_numpy_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = body_reading_a_device_element_on_the_host()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    A = cupy.asarray(np.arange(1, 9, dtype=np.float64))
    out = cupy.zeros(8)
    sdfg(A=A, out=out)
    np.testing.assert_allclose(out.get(), np.full(8, 1.0))
