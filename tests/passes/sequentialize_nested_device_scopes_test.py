# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU-side resolution of nested device scopes.

:class:`~dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes.
SequentializeNestedDeviceScopes` is what the GPU target runs where CPU runs its fork/join cost
model. The behaviour pinned here is the one the canonicalize finalize tail used to provide for
``target='gpu'``: nesting decides, unconditionally -- no trip counts, no work threshold, because a
``GPU_Device`` map inside a device scope is an illegal in-kernel launch rather than a bad trade.
"""
import dace
import pytest
from dace import dtypes
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.cpu_specialization import SequentializeUnprofitableParallelScopes
from dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes import (
    SequentializeNestedDeviceScopes)

N = dace.symbol('N')


def nested_device_maps():
    """``GPU_Device`` map inside a ``GPU_Device`` map -- an in-kernel kernel launch."""
    sdfg = dace.SDFG('nested_device_maps')
    sdfg.add_array('a', [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', dict(i='0:N'), schedule=dtypes.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map('inner', dict(j='0:N'), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('t', {}, {'out'}, 'out = 1.0')
    access = state.add_access('a')
    state.add_edge(outer_entry, None, inner_entry, None, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, outer_exit, access, src_conn='out', memlet=dace.Memlet('a[i, j]'))
    sdfg.validate()
    return sdfg


def schedules(sdfg):
    """Every map label -> schedule in ``sdfg``."""
    return {n.map.label: n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}


def test_nested_device_map_is_sequentialized():
    """The inner kernel loses the device schedule; the outer one keeps it."""
    sdfg = nested_device_maps()
    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['outer'] == dtypes.ScheduleType.GPU_Device
    assert schedules(sdfg)['inner'] == dtypes.ScheduleType.Sequential


def test_cpu_cost_model_never_touches_device_maps():
    """The CPU band decides CPU schedules only -- a GPU graph is not its business."""
    sdfg = nested_device_maps()
    SequentializeUnprofitableParallelScopes().apply_pass(sdfg, {})
    assert schedules(sdfg)['outer'] == dtypes.ScheduleType.GPU_Device
    assert schedules(sdfg)['inner'] == dtypes.ScheduleType.GPU_Device


@pytest.mark.parametrize('trips', ['N', '4'])
def test_device_library_node_in_any_loop_is_sequentialized(trips):
    """A library node re-entered by a loop would launch per iteration. Unlike the CPU rule, this
    one is NOT trip-count aware: even a provably short loop counts."""
    sdfg = dace.SDFG(f'device_reduce_loop_{trips}')
    sdfg.add_array('row', [N], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    region = LoopRegion('outer',
                        initialize_expr='it = 0',
                        condition_expr=f'it < {trips}',
                        update_expr='it = it + 1',
                        loop_var='it')
    sdfg.add_node(region, is_start_block=True)
    state = region.add_state('body', is_start_block=True)
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    node.schedule = dtypes.ScheduleType.GPU_Device
    state.add_node(node)
    state.add_edge(state.add_access('row'), None, node, '_in', dace.Memlet('row[0:N]'))
    state.add_edge(node, '_out', state.add_access('acc'), None, dace.Memlet('acc[0]'))
    sdfg.validate()

    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})

    assert node.schedule == dtypes.ScheduleType.Sequential


def test_top_level_device_map_is_left_alone():
    """Nothing re-enters a top-level kernel, so it keeps its device schedule."""
    sdfg = dace.SDFG('top_level_device_map')
    sdfg.add_array('a', [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('body', {'i': '0:N'}, {},
                             'out = 1.0', {'out': dace.Memlet('a[i]')},
                             schedule=dtypes.ScheduleType.GPU_Device,
                             external_edges=True)
    sdfg.validate()

    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})

    assert schedules(sdfg)['body_map'] == dtypes.ScheduleType.GPU_Device


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
