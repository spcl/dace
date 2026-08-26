# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU reduce expansion falls back to ``pure``, which must still be device code.

``ExpandReduceGPUAuto`` declines a node carrying no identity and hands it to ``ExpandReducePure``,
whose maps carry the default schedule. Schedule inference has already run by the time a library node
expands, so those maps reach codegen as host loops over ``GPU_Global`` memory (npbench nbody).
"""
import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.libraries.standard.nodes.reduce import Reduce


def gpu_reduce_without_identity() -> dace.SDFG:
    """Row sums of a device array, by a node whose identity nobody set."""
    sdfg = dace.SDFG('gpu_reduce_no_identity')
    sdfg.add_array('A', [8, 256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [8], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=[1], identity=None)
    node.implementation = 'GPUAuto'
    state.add_node(node)
    state.add_edge(state.add_access('A'), None, node, '_in', dace.Memlet('A[0:8, 0:256]'))
    state.add_edge(node, '_out', state.add_access('out'), None, dace.Memlet('out[0:8]'))
    node.add_in_connector('_in')
    node.add_out_connector('_out')
    sdfg.validate()
    return sdfg


def test_the_pure_fallback_is_scheduled_as_a_kernel():
    sdfg = gpu_reduce_without_identity()
    sdfg.expand_library_nodes()

    schedules = {
        node.map.label: (node.map.schedule, state.entry_node(node) is None)
        for node, state in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)
    }
    assert schedules, 'the fallback emitted no map at all'
    outer = {label for label, (_, top) in schedules.items() if top}
    assert outer, 'the fallback emitted no outermost map to launch'
    for label, (schedule, top) in schedules.items():
        expected = dtypes.ScheduleType.GPU_Device if top else dtypes.ScheduleType.Sequential
        assert schedule is expected, f'map {label} is {schedule.name}, not {expected.name}'
    sdfg.validate()


if __name__ == '__main__':
    test_the_pure_fallback_is_scheduled_as_a_kernel()
