# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU offload is a step of its own, not part of ``finalize_for_target``.

The device move used to be bundled into the canonicalize-GPU tail, which left no place to run a pass
on a canonicalized-but-still-device-agnostic graph. It is now :func:`offload_to_gpu` (or a caller's
own recipe -- CloudSC keeps the nblocks map sequential and schedules the inner maps), and
``finalize_for_target(sdfg, 'gpu')`` finalizes an already-offloaded graph.

The split has a hazard: everything in the GPU branch of the finalize tail reads the device maps and
``GPU_Global`` arrays an offload creates, so on a host graph each step is a no-op and the result is a
CPU graph wearing a GPU label. :func:`assert_offloaded` turns that into a raise.
"""
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.finalize import assert_offloaded, finalize_for_target, offload_to_gpu


def elementwise_sdfg(name='fin_offload'):
    """``B[i] = A[i] * 2`` as one host map."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [32], dace.float64)
    sdfg.add_array('B', [32], dace.float64)
    state = sdfg.add_state('main')
    state.add_mapped_tasklet('scale', {'i': '0:32'}, {'inp': dace.Memlet('A[i]')},
                             'o = inp * 2.0', {'o': dace.Memlet('B[i]')},
                             external_edges=True)
    sdfg.validate()
    return sdfg


def has_device_map(sdfg):
    return any(
        isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
        for node, _ in sdfg.all_nodes_recursive())


def test_finalize_does_not_offload():
    """``finalize_for_target`` no longer moves anything onto the device by itself."""
    sdfg = elementwise_sdfg('fin_no_offload')
    with pytest.raises(ValueError, match='already-offloaded'):
        finalize_for_target(sdfg, 'gpu')
    assert not has_device_map(sdfg)


def test_assert_offloaded_rejects_host_graph():
    """The guard names the fix rather than failing somewhere downstream."""
    with pytest.raises(ValueError, match='offload_to_gpu'):
        assert_offloaded(elementwise_sdfg('fin_host'))


def test_assert_offloaded_accepts_gpu_global_only():
    """A graph offloaded by storage alone (no map of its own) passes: either signal counts."""
    sdfg = elementwise_sdfg('fin_storage_only')
    sdfg.arrays['A'].storage = dtypes.StorageType.GPU_Global
    assert_offloaded(sdfg)  # must not raise


def sequential_sdfg(name='fin_sequential'):
    """``b = b * 2`` over one loop: no map, no library node, nothing an offload could place."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('B', [32], dace.float64)
    loop = dace.sdfg.state.LoopRegion('rows', 'i < 32', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('scale', {'inp': None}, {'o': None}, 'o = inp * 2.0')
    state.add_edge(state.add_read('B'), None, tasklet, 'inp', dace.Memlet('B[i]'))
    state.add_edge(tasklet, 'o', state.add_write('B'), None, dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def test_assert_offloaded_accepts_a_graph_with_nothing_to_offload():
    """A purely sequential graph (npbench crc16) is host code by right, not a missing offload call.

    The guard exists to catch a forgotten :func:`offload_to_gpu`; with neither a map nor a library
    node anywhere, no offload could have placed anything, so there is nothing it could be about.
    """
    sdfg = sequential_sdfg()
    assert not any(isinstance(node, dace.sdfg.nodes.MapEntry) for node, _ in sdfg.all_nodes_recursive())
    assert_offloaded(sdfg)  # must not raise
    # The graph that HAS a map and was never offloaded is still rejected -- the relaxation is about
    # what could be placed, not about how loudly the guard speaks.
    with pytest.raises(ValueError, match='offload_to_gpu'):
        assert_offloaded(elementwise_sdfg('fin_still_rejected'))


def test_offload_then_finalize_is_the_supported_order():
    """``offload_to_gpu`` schedules the map on the device, after which finalizing is accepted."""
    sdfg = elementwise_sdfg('fin_offload_ok')
    offload_to_gpu(sdfg)
    assert has_device_map(sdfg), 'offload_to_gpu left the map on the host'
    finalize_for_target(sdfg, 'gpu')  # must not raise
    assert has_device_map(sdfg)


def test_cpu_target_unaffected():
    """The CPU tail never consulted the guard; it still finalizes a plain host graph."""
    sdfg = elementwise_sdfg('fin_cpu')
    finalize_for_target(sdfg, 'cpu')
    assert not has_device_map(sdfg)


def test_passes_can_run_between_canonicalization_and_offload():
    """The point of the split: a caller's pass sees a device-agnostic graph, and its effect survives
    the offload. Here the map range is halved before offloading, and the compiled kernel honours it."""
    sdfg = elementwise_sdfg('fin_between')
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            node.map.range = dace.subsets.Range([(0, 15, 1)])
    offload_to_gpu(sdfg)
    entries = [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]
    assert entries, 'no device map after offload'
    assert entries[0].map.range[0][1] == 15, entries[0].map.range


if __name__ == '__main__':
    test_finalize_does_not_offload()
    test_assert_offloaded_rejects_host_graph()
    test_assert_offloaded_accepts_gpu_global_only()
    test_assert_offloaded_accepts_a_graph_with_nothing_to_offload()
    test_offload_then_finalize_is_the_supported_order()
    test_cpu_target_unaffected()
    test_passes_can_run_between_canonicalization_and_offload()
    print('OK')
