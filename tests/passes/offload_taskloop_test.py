# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map that only LAUNCHES work stays on the host, and what it launches becomes the kernels.

Two independent reasons make a map a taskloop, and each gets its own shape here:

* every bit of computation under it sits inside an inner map or a nested SDFG that itself only
  launches -- the ICON blocking shape, where the ``nblks`` map launches and the ``nproma``/``nlev``
  maps compute;
* it encloses a library node that is neither a copy nor a fill, which expands to a device-wide call
  that only host code can issue.

The second reason is not optional -- a kernel cannot contain a cuBLAS call, so it applies whatever
the config says. ``optimizer.gpu_taskloop_heuristics`` gates only the first, and every case here
asserts the knob-off placement as well.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.offloading.taskloop import is_taskloop_map, taskloop_maps

#: The ICON blocking triple the taskloop shape is named after.
NBLKS, NPROMA, NLEV = 4, 8, 6


def uncollapsed_nest() -> dace.SDFG:
    """An ``i`` map whose only child is a ``j`` map holding the tasklet.

    A collapsed ``map[i, j]`` is one kernel and the question never arises; left uncollapsed, ``i``
    launches and ``j`` computes.
    """
    sdfg = dace.SDFG('uncollapsed_nest')
    sdfg.add_array('A', [16, 16], dace.float64)
    sdfg.add_array('B', [16, 16], dace.float64)
    state = sdfg.add_state('body')
    outer_in, outer_out = state.add_map('outer', dict(i='0:16'))
    inner_in, inner_out = state.add_map('inner', dict(j='0:16'))
    tasklet = state.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(state.add_read('A'),
                          outer_in,
                          inner_in,
                          tasklet,
                          dst_conn='inp',
                          memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet,
                          inner_out,
                          outer_out,
                          state.add_write('B'),
                          src_conn='out',
                          memlet=dace.Memlet('B[i, j]'))
    sdfg.validate()
    return sdfg


def map_over_reduce() -> dace.SDFG:
    """A row map whose body stages one row and hands it to a ``Reduce`` library node."""
    sdfg = dace.SDFG('map_over_reduce')
    sdfg.add_array('A', [8, 16], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('row', [16], dace.float64, transient=True)
    state = sdfg.add_state('body')
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    reduce_node = state.add_reduce('lambda a, b: a + b', None, 0.0)
    row = state.add_access('row')
    state.add_memlet_path(state.add_read('A'), entry, row, memlet=dace.Memlet('A[i, 0:16]', other_subset='0:16'))
    state.add_edge(row, None, reduce_node, None, dace.Memlet('row[0:16]'))
    state.add_memlet_path(reduce_node, exit_node, state.add_write('B'), memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def map_over_fill() -> dace.SDFG:
    """The same shape with a ``Fill``: initializing data is not a device-wide call."""
    from dace.libraries.standard.nodes.fill import FillLibraryNode

    sdfg = dace.SDFG('map_over_fill')
    sdfg.add_array('B', [8, 16], dace.float64)
    state = sdfg.add_state('body')
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    fill = FillLibraryNode('fill', value=1.0)
    state.add_node(fill)
    state.add_nedge(entry, fill, dace.Memlet())
    state.add_memlet_path(fill, exit_node, state.add_write('B'), src_conn='_fill_out', memlet=dace.Memlet('B[i, 0:16]'))
    sdfg.validate()
    return sdfg


def blocked_nest() -> dace.SDFG:
    """The ICON shape: an ``nblks`` map whose single child is a nested SDFG of ``(nproma, nlev)`` maps.

    The body is several states joined by an interstate edge that assigns a symbol -- the
    computation is all inside maps, and the edge only prepares the next state's range.
    """
    inner = dace.SDFG('block_body')
    inner.add_array('a', [NPROMA, NLEV], dace.float64)
    inner.add_array('b', [NPROMA, NLEV], dace.float64)
    inner.add_symbol('half', dace.int64)
    first = inner.add_state('scale', is_start_block=True)
    second = inner.add_state('shift')
    inner.add_edge(first, second, dace.InterstateEdge(assignments={'half': str(NLEV // 2)}))
    first.add_mapped_tasklet('scale',
                             dict(jc=f'0:{NPROMA}', jk=f'0:{NLEV}'),
                             dict(inp=dace.Memlet('a[jc, jk]')),
                             'out = inp * 2.0',
                             dict(out=dace.Memlet('b[jc, jk]')),
                             external_edges=True)
    second.add_mapped_tasklet('shift',
                              dict(jc=f'0:{NPROMA}', jk='0:half'),
                              dict(inp=dace.Memlet('b[jc, jk]')),
                              'out = inp + 1.0',
                              dict(out=dace.Memlet('b[jc, jk]')),
                              external_edges=True)

    sdfg = dace.SDFG('blocked_nest')
    sdfg.add_array('A', [NBLKS, NPROMA, NLEV], dace.float64)
    sdfg.add_array('B', [NBLKS, NPROMA, NLEV], dace.float64)
    state = sdfg.add_state('blocks')
    entry, exit_node = state.add_map('nblks', dict(jb=f'0:{NBLKS}'))
    nested = state.add_nested_sdfg(inner, {'a'}, {'b'})
    state.add_memlet_path(state.add_read('A'),
                          entry,
                          nested,
                          dst_conn='a',
                          memlet=dace.Memlet(f'A[jb, 0:{NPROMA}, 0:{NLEV}]'))
    state.add_memlet_path(nested,
                          exit_node,
                          state.add_write('B'),
                          src_conn='b',
                          memlet=dace.Memlet(f'B[jb, 0:{NPROMA}, 0:{NLEV}]'))
    sdfg.validate()
    return sdfg


def blocked_nest_reference(A: np.ndarray) -> np.ndarray:
    B = A * 2.0
    B[:, :, :NLEV // 2] += 1.0
    return B


def offloaded(sdfg: dace.SDFG, heuristics: bool) -> dace.SDFG:
    with dace.config.set_temporary('optimizer', 'gpu_taskloop_heuristics', value=heuristics):
        sdfg.apply_gpu_transformations()
    return sdfg


def map_schedule(sdfg: dace.SDFG, fragment: str) -> dtypes.ScheduleType:
    """The schedule of the one map in the tree whose label contains ``fragment``."""
    found = [
        node.map.schedule for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and fragment in node.map.label
    ]
    assert len(found) == 1, f"{fragment!r} matched {len(found)} maps"
    return found[0]


def libnode_schedule(sdfg: dace.SDFG) -> dtypes.ScheduleType:
    """The schedule of the tree's single library node."""
    found = [node.schedule for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.LibraryNode)]
    assert len(found) == 1, f"expected one library node, found {len(found)}"
    return found[0]


def storages(sdfg: dace.SDFG) -> dict:
    return {name: desc.storage for name, desc in sdfg.arrays.items()}


def test_an_uncollapsed_nest_launches_from_the_host():
    sdfg = offloaded(uncollapsed_nest(), heuristics=True)
    assert map_schedule(sdfg, 'outer') == dtypes.ScheduleType.Sequential
    assert map_schedule(sdfg, 'inner') == dtypes.ScheduleType.GPU_Device


def test_an_uncollapsed_nest_is_one_kernel_without_the_heuristics():
    sdfg = offloaded(uncollapsed_nest(), heuristics=False)
    assert map_schedule(sdfg, 'outer') == dtypes.ScheduleType.GPU_Device
    assert map_schedule(sdfg, 'inner') == dtypes.ScheduleType.Sequential


def test_the_launched_kernel_reads_device_copies():
    """The launcher is host code, but the data it hands down is the kernel's, so it moves once."""
    sdfg = offloaded(uncollapsed_nest(), heuristics=True)
    where = storages(sdfg)
    assert where['A_gpu'] == dtypes.StorageType.GPU_Global
    assert where['B_gpu'] == dtypes.StorageType.GPU_Global
    assert where['A'] != dtypes.StorageType.GPU_Global
    # One copy each way for the whole launcher, not one per iteration.
    labels = [block.label for block in sdfg.states()]
    assert sum('to_gpu' in label for label in labels) == 1, labels
    assert sum('to_host' in label for label in labels) == 1, labels


def test_a_map_around_a_device_wide_library_node_launches_from_the_host():
    sdfg = offloaded(map_over_reduce(), heuristics=True)
    assert map_schedule(sdfg, 'rows') == dtypes.ScheduleType.Sequential
    assert libnode_schedule(sdfg) == dtypes.ScheduleType.GPU_Device


def test_a_map_around_a_library_node_launches_from_the_host_without_the_heuristics_too():
    """Not a heuristic: a device-wide call is issued by host code, so no map around one is a kernel."""
    sdfg = offloaded(map_over_reduce(), heuristics=False)
    assert map_schedule(sdfg, 'rows') == dtypes.ScheduleType.Sequential
    assert libnode_schedule(sdfg) == dtypes.ScheduleType.GPU_Device


def test_the_staged_row_of_a_launched_library_node_is_device_memory():
    sdfg = offloaded(map_over_reduce(), heuristics=True)
    assert storages(sdfg)['row'] == dtypes.StorageType.GPU_Global


def test_a_fill_does_not_make_its_parent_a_taskloop():
    """A fill is data movement, not a device-wide call, so the map around it is still the kernel."""
    sdfg = map_over_fill()
    state = next(iter(sdfg.states()))
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    assert not is_taskloop_map(state, entry, state.scope_children())


def nest_with_a_side_tasklet() -> dace.SDFG:
    """The uncollapsed nest plus a tasklet in the OUTER map's own scope."""
    sdfg = dace.SDFG('nest_with_side_tasklet')
    sdfg.add_array('A', [16, 16], dace.float64)
    sdfg.add_array('B', [16, 16], dace.float64)
    sdfg.add_array('C', [16], dace.float64)
    state = sdfg.add_state('body')
    read = state.add_read('A')
    outer_in, outer_out = state.add_map('outer', dict(i='0:16'))
    inner_in, inner_out = state.add_map('inner', dict(j='0:16'))
    tasklet = state.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(read, outer_in, inner_in, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet,
                          inner_out,
                          outer_out,
                          state.add_write('B'),
                          src_conn='out',
                          memlet=dace.Memlet('B[i, j]'))
    side = state.add_tasklet('side', {'inp'}, {'out'}, 'out = inp + 1.0')
    state.add_memlet_path(read, outer_in, side, dst_conn='inp', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(side, outer_out, state.add_write('C'), src_conn='out', memlet=dace.Memlet('C[i]'))
    sdfg.validate()
    return sdfg


def test_a_map_that_computes_in_its_own_scope_is_not_a_taskloop():
    """A tasklet beside the inner map means the outer one does work, not only launching."""
    sdfg = nest_with_a_side_tasklet()
    state = next(iter(sdfg.states()))
    outer = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.label == 'outer')
    assert not is_taskloop_map(state, outer, state.scope_children())
    assert map_schedule(offloaded(sdfg, heuristics=True), 'outer') == dtypes.ScheduleType.GPU_Device


def test_a_block_map_over_a_nested_sdfg_launches_its_states_kernels():
    """Interstate edges inside the body are symbol preparation, not computation."""
    sdfg = offloaded(blocked_nest(), heuristics=True)
    assert map_schedule(sdfg, 'nblks') == dtypes.ScheduleType.Sequential
    assert map_schedule(sdfg, 'scale') == dtypes.ScheduleType.GPU_Device
    assert map_schedule(sdfg, 'shift') == dtypes.ScheduleType.GPU_Device


def test_a_block_maps_body_is_sequential_without_the_heuristics():
    sdfg = offloaded(blocked_nest(), heuristics=False)
    assert map_schedule(sdfg, 'nblks') == dtypes.ScheduleType.GPU_Device
    assert map_schedule(sdfg, 'scale') == dtypes.ScheduleType.Sequential
    assert map_schedule(sdfg, 'shift') == dtypes.ScheduleType.Sequential


def test_the_body_of_a_block_map_copies_nothing_of_its_own():
    """The launcher already placed the bound arrays, so the body inherits them instead of copying."""
    sdfg = offloaded(blocked_nest(), heuristics=True)
    body = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert [state.label for state in body.states()] == ['scale', 'shift']
    assert body.arrays['a'].storage == dtypes.StorageType.GPU_Global
    assert body.arrays['b'].storage == dtypes.StorageType.GPU_Global


def test_taskloop_maps_finds_the_launchers_of_a_whole_tree():
    sdfg = blocked_nest()
    labels = {entry.map.label for entry in taskloop_maps(sdfg)}
    assert labels == {'nblks'}


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_launched_nest_computes_what_the_single_kernel_computes(heuristics):
    sdfg = offloaded(uncollapsed_nest(), heuristics)
    A = np.random.default_rng(0).random((16, 16))
    B = np.zeros((16, 16))
    sdfg(A=A, B=B)
    assert np.allclose(B, A * 2.0)


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_launched_library_node_computes_what_the_kernel_computes(heuristics):
    sdfg = offloaded(map_over_reduce(), heuristics)
    A = np.random.default_rng(1).random((8, 16))
    B = np.zeros(8)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.sum(axis=1))


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_launched_block_body_computes_what_the_kernel_computes(heuristics):
    sdfg = offloaded(blocked_nest(), heuristics)
    A = np.random.default_rng(2).random((NBLKS, NPROMA, NLEV))
    B = np.zeros((NBLKS, NPROMA, NLEV))
    sdfg(A=A, B=B)
    assert np.allclose(B, blocked_nest_reference(A))
