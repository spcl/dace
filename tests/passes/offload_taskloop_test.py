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
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.properties import CodeBlock
from dace.sdfg import infer_types, nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadToAccelerator
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
    state.add_edge(row, None, reduce_node, '_in', dace.Memlet('row[0:16]'))
    state.add_memlet_path(reduce_node, exit_node, state.add_write('B'), memlet=dace.Memlet('B[i]'), src_conn='_out')
    sdfg.validate()
    return sdfg


def map_over_reduce_beside_a_tasklet() -> dace.SDFG:
    """One state: a kernel fills ``T`` from ``A``; a row map reduces ``T``'s rows into ``B`` and,
    beside the reduce, scales ``T[i, 0]`` into ``U``; a kernel copies ``U`` into ``C``.

    The reduce makes the row map a taskloop, so its scope is host code, while ``T`` and ``U`` are
    touched by kernels in the same state (a hybrid state). The side tasklet reaches both arrays
    straight through the map's entry and exit, with no access node of its own. That is the shape
    npbench cp2k_density_matrix_trs4 reaches on the GPU canonicalize column once a row-wise ``* 3``
    is fused beside a loop body that reduces.
    """
    sdfg = dace.SDFG('map_over_reduce_beside_a_tasklet')
    for name, shape in (('A', [8, 16]), ('B', [8]), ('C', [8])):
        sdfg.add_array(name, shape, dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('T', [8, 16], dace.float64, transient=True)
    sdfg.add_array('U', [8], dace.float64, transient=True)
    sdfg.add_array('row', [16], dace.float64, transient=True)
    state = sdfg.add_state('body')
    scratch = state.add_access('T')
    state.add_mapped_tasklet('fill',
                             dict(j='0:8', k='0:16'), {'a': dace.Memlet('A[j, k]')},
                             't = a + 1', {'t': dace.Memlet('T[j, k]')},
                             input_nodes={'A': state.add_read('A')},
                             output_nodes={'T': scratch},
                             external_edges=True)
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    reduce_node = state.add_reduce('lambda a, b: a + b', None, 0.0)
    row = state.add_access('row')
    state.add_memlet_path(scratch, entry, row, memlet=dace.Memlet('T[i, 0:16]', other_subset='0:16'))
    state.add_edge(row, None, reduce_node, '_in', dace.Memlet('row[0:16]'))
    state.add_memlet_path(reduce_node, exit_node, state.add_write('B'), memlet=dace.Memlet('B[i]'), src_conn='_out')
    scale = state.add_tasklet('scale', {'x'}, {'u'}, 'u = x * 3')
    state.add_memlet_path(scratch, entry, scale, dst_conn='x', memlet=dace.Memlet('T[i, 0]'))
    scaled = state.add_access('U')
    state.add_memlet_path(scale, exit_node, scaled, src_conn='u', memlet=dace.Memlet('U[i]'))
    state.add_mapped_tasklet('copy',
                             dict(m='0:8'), {'v': dace.Memlet('U[m]')},
                             'c = v', {'c': dace.Memlet('C[m]')},
                             input_nodes={'U': scaled},
                             output_nodes={'C': state.add_write('C')},
                             external_edges=True)
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
    state.add_memlet_path(fill,
                          exit_node,
                          state.add_write('B'),
                          src_conn=FillLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet('B[i, 0:16]'))
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


def blocked_nest_beside_a_host_read() -> dace.SDFG:
    """The ICON shape in a hybrid state: a free tasklet beside the ``nblks`` map reads ``A[0, 0, 0]``.

    The host read makes the state hybrid, so the wrapper runs, and the ``nblks`` body -- one nested
    SDFG that only launches -- is what it must leave alone.
    """
    sdfg = blocked_nest()
    state = sdfg.start_block
    sdfg.add_array('first', [1], dace.float64)
    peek = state.add_tasklet('peek', {'x'}, {'y'}, 'y = x')
    state.add_edge(state.add_read('A'), None, peek, 'x', dace.Memlet('A[0, 0, 0]'))
    state.add_edge(peek, 'y', state.add_write('first'), None, dace.Memlet('first[0]'))
    sdfg.validate()
    return sdfg


def rows_over_a_reducing_body(side_tasklet: bool) -> dace.SDFG:
    """The npbench gem shape: a row map whose body is ONE nested SDFG that zeroes ``res`` and reduces.

    The top level is hybrid -- ``negate`` is host code over ``k``, which the ``scale`` kernel reads
    too. The ``Reduce`` makes ``rows`` a taskloop, and inside the nested SDFG the single-element
    ``zero`` sits beside it. With ``side_tasklet`` the body also holds ``side``, scaling ``T[i, 0]``
    into ``U``, which a ``copy`` kernel reads: host work beside the nested SDFG.
    """
    inner = dace.SDFG('row_body')
    inner.add_array('a', [16], dace.float64)
    inner.add_array('res', [1], dace.float64)
    compute = inner.add_state('compute')
    zero = compute.add_tasklet('zero', {}, {'o'}, 'o = 0')
    zeroed = compute.add_access('res')
    row = compute.add_read('a')
    reduce_node = compute.add_reduce('lambda a, b: a + b', None, None)
    compute.add_edge(zero, 'o', zeroed, None, dace.Memlet('res[0]'))
    compute.add_nedge(zeroed, row, dace.Memlet())
    compute.add_edge(row, None, reduce_node, '_in', dace.Memlet('a[0:16]'))
    compute.add_edge(reduce_node, '_out', compute.add_write('res'), None, dace.Memlet('res[0]'))

    sdfg = dace.SDFG('rows_over_a_reducing_body')
    for name, shape in (('A', [8, 16]), ('B', [8]), ('C', [8]), ('k', [1])):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_array('T', [8, 16], dace.float64, transient=True)
    sdfg.add_array('U', [8], dace.float64, transient=True)
    sdfg.add_array('nk', [1], dace.float64, transient=True)
    state = sdfg.add_state('body')
    negate = state.add_tasklet('negate', {'x'}, {'y'}, 'y = -x')
    negated = state.add_access('nk')
    state.add_edge(state.add_read('k'), None, negate, 'x', dace.Memlet('k[0]'))
    state.add_edge(negate, 'y', negated, None, dace.Memlet('nk[0]'))
    scaled = state.add_access('T')
    state.add_mapped_tasklet('scale',
                             dict(j='0:8', m='0:16'), {
                                 'a': dace.Memlet('A[j, m]'),
                                 's': dace.Memlet('nk[0]')
                             },
                             't = a * s', {'t': dace.Memlet('T[j, m]')},
                             input_nodes={
                                 'A': state.add_read('A'),
                                 'nk': negated
                             },
                             output_nodes={'T': scaled},
                             external_edges=True)
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    nested = state.add_nested_sdfg(inner, {'a'}, {'res'})
    state.add_memlet_path(scaled, entry, nested, dst_conn='a', memlet=dace.Memlet('T[i, 0:16]'))
    state.add_memlet_path(nested, exit_node, state.add_write('B'), src_conn='res', memlet=dace.Memlet('B[i]'))
    if side_tasklet:
        side = state.add_tasklet('side', {'x'}, {'u'}, 'u = x * 3')
        state.add_memlet_path(scaled, entry, side, dst_conn='x', memlet=dace.Memlet('T[i, 0]'))
        tripled = state.add_access('U')
        state.add_memlet_path(side, exit_node, tripled, src_conn='u', memlet=dace.Memlet('U[i]'))
        state.add_mapped_tasklet('copy',
                                 dict(q='0:8'), {'v': dace.Memlet('U[q]')},
                                 'c = v', {'c': dace.Memlet('C[q]')},
                                 input_nodes={'U': tripled},
                                 output_nodes={'C': state.add_write('C')},
                                 external_edges=True)
    sdfg.validate()
    return sdfg


def node_is_device_code(sdfg: dace.SDFG, label: str) -> bool:
    """Whether the one tasklet, library node or nested SDFG labelled ``label`` runs inside a kernel."""
    found = [(node, state) for node, state in sdfg.all_nodes_recursive()
             if isinstance(node, (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG)) and node.label == label]
    assert len(found) == 1, f"{label!r} matched {len(found)} nodes"
    node, state = found[0]
    return is_devicelevel_gpu(state.sdfg, state, node)


def rows_over_a_scalar_body_with_inner_reduces() -> dace.SDFG:
    """The npbench cp2k_density_matrix_trs4 shape: a row body of lone single-element states.

    Each of ``flag0``..``flag2`` is its own state writing one element of ``F``, and a last state maps
    ``j`` over a nested ``cell`` that reduces one row of the block -- the device-wide calls sit under
    the body's own loop, not in its states. ``negate`` makes the top level hybrid, as in
    :func:`rows_over_a_reducing_body`.
    """
    cell = dace.SDFG('cell')
    cell.add_array('a', [4], dace.float64)
    cell.add_array('r', [1], dace.float64)
    reduce_state = cell.add_state('reduce')
    reduce_node = reduce_state.add_reduce('lambda a, b: a + b', None, 0.0)
    reduce_state.add_edge(reduce_state.add_read('a'), None, reduce_node, '_in', dace.Memlet('a[0:4]'))
    reduce_state.add_edge(reduce_node, '_out', reduce_state.add_write('r'), None, dace.Memlet('r[0]'))

    body = dace.SDFG('scalar_body')
    body.add_array('blk', [4, 4], dace.float64)
    body.add_array('res', [4], dace.float64)
    body.add_array('flags', [3], dace.float64)
    previous = None
    for index in range(3):
        flag = body.add_state(f'flag{index}', is_start_block=index == 0)
        tasklet = flag.add_tasklet(f'flag{index}', {}, {'o'}, f'o = {index + 1}')
        flag.add_edge(tasklet, 'o', flag.add_write('flags'), None, dace.Memlet(f'flags[{index}]'))
        if previous is not None:
            body.add_edge(previous, flag, dace.InterstateEdge())
        previous = flag
    cells = body.add_state_after(previous, 'cells')
    entry, exit_node = cells.add_map('cells', dict(j='0:4'))
    nested = cells.add_nested_sdfg(cell, {'a'}, {'r'})
    cells.add_memlet_path(cells.add_read('blk'), entry, nested, dst_conn='a', memlet=dace.Memlet('blk[j, 0:4]'))
    cells.add_memlet_path(nested, exit_node, cells.add_write('res'), src_conn='r', memlet=dace.Memlet('res[j]'))

    sdfg = dace.SDFG('rows_over_a_scalar_body_with_inner_reduces')
    for name, shape in (('A', [8, 4, 4]), ('B', [8, 4]), ('F', [8, 3]), ('k', [1])):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_array('T', [8, 4, 4], dace.float64, transient=True)
    sdfg.add_array('nk', [1], dace.float64, transient=True)
    state = sdfg.add_state('body')
    negate = state.add_tasklet('negate', {'x'}, {'y'}, 'y = -x')
    negated = state.add_access('nk')
    state.add_edge(state.add_read('k'), None, negate, 'x', dace.Memlet('k[0]'))
    state.add_edge(negate, 'y', negated, None, dace.Memlet('nk[0]'))
    scaled = state.add_access('T')
    state.add_mapped_tasklet('scale',
                             dict(p='0:8', q='0:4', m='0:4'), {
                                 'a': dace.Memlet('A[p, q, m]'),
                                 's': dace.Memlet('nk[0]')
                             },
                             't = a * s', {'t': dace.Memlet('T[p, q, m]')},
                             input_nodes={
                                 'A': state.add_read('A'),
                                 'nk': negated
                             },
                             output_nodes={'T': scaled},
                             external_edges=True)
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    rows_body = state.add_nested_sdfg(body, {'blk'}, {'res', 'flags'})
    state.add_memlet_path(scaled, entry, rows_body, dst_conn='blk', memlet=dace.Memlet('T[i, 0:4, 0:4]'))
    state.add_memlet_path(rows_body, exit_node, state.add_write('B'), src_conn='res', memlet=dace.Memlet('B[i, 0:4]'))
    state.add_memlet_path(rows_body, exit_node, state.add_write('F'), src_conn='flags', memlet=dace.Memlet('F[i, 0:3]'))
    sdfg.validate()
    return sdfg


def outermost_kernel(sdfg: dace.SDFG, label: str) -> nodes.MapEntry:
    """The outermost GPU_Device map around the one tasklet labelled ``label``, across nested SDFGs."""
    found = [(node, state) for node, state in sdfg.all_nodes_recursive()
             if isinstance(node, nodes.Tasklet) and node.label == label]
    assert len(found) == 1, f"{label!r} matched {len(found)} tasklets"
    node, state = found[0]
    kernel = None
    while state is not None:
        scope = state.scope_dict()[node]
        while scope is not None:
            if scope.map.schedule == dtypes.ScheduleType.GPU_Device:
                kernel = scope
            scope = state.scope_dict()[scope]
        nested = state.sdfg.parent_nsdfg_node
        node, state = nested, state.sdfg.parent
    assert kernel is not None, f"{label!r} runs on the host"
    return kernel


def offloaded(sdfg: dace.SDFG, heuristics: bool) -> dace.SDFG:
    """The offloading pass's own output, which is what every assertion below is about.

    ``simplify`` is left off deliberately. It runs after the offloading and is free to fuse the
    copy states into the body and to drop an upload the kernel overwrites before reading -- both
    correct, both invisible to a caller, and both of which rename or remove the very states these
    tests name. Simplification has its own tests; asking for it here would only make them assert
    its shape instead of the offloader's.
    """
    with dace.config.set_temporary('optimizer', 'gpu_taskloop_heuristics', value=heuristics):
        sdfg.apply_gpu_transformations(simplify=False)
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


def test_a_map_around_an_in_kernel_library_node_is_still_the_kernel():
    """A cub block reduce emits device code and refuses to expand outside a kernel, so the map
    around it is that kernel -- unlike the device-wide reduce, which host code issues."""
    sdfg = map_over_reduce()
    state = next(iter(sdfg.states()))
    reduce_node = next(node for node in state.nodes() if isinstance(node, nodes.LibraryNode))
    reduce_node.implementation = 'CUDA (block atomic)'
    entry = next(node for node in state.nodes() if isinstance(node, nodes.MapEntry))

    assert not is_taskloop_map(state, entry, state.scope_children())
    assert map_schedule(offloaded(sdfg, heuristics=False), 'rows') == dtypes.ScheduleType.GPU_Device


def test_the_staged_row_of_a_launched_library_node_is_device_memory():
    sdfg = offloaded(map_over_reduce(), heuristics=True)
    assert storages(sdfg)['row'] == dtypes.StorageType.GPU_Global


@pytest.mark.parametrize('heuristics', [True, False])
def test_a_tasklet_beside_a_launched_library_node_becomes_a_kernel(heuristics):
    """The tasklet may not stay host code reading a device array: it gets a size-1 kernel.

    The wrapper judged it scalar-only because it looked for access nodes, but a tasklet inside a
    scope reaches its arrays through the scope's entry and exit. It was left on the host,
    and validation rejected the graph: ``stored as StorageType.GPU_Global but accessed on host``.
    """
    sdfg = offloaded(map_over_reduce_beside_a_tasklet(), heuristics)
    assert map_schedule(sdfg, 'rows') == dtypes.ScheduleType.Sequential
    state, scale = next((state, node) for node, state in sdfg.all_nodes_recursive()
                        if isinstance(node, nodes.Tasklet) and node.label == 'scale')
    assert is_devicelevel_gpu(state.sdfg, state, scale), 'the side tasklet stayed host code'
    sdfg.validate()


@pytest.mark.parametrize('side_tasklet', [False, True])
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_nested_body_around_a_reduce_is_not_wrapped_into_a_kernel(heuristics, side_tasklet):
    """The nested SDFG is device work, not host work to lift: its ``Reduce`` stays a host-issued call.

    Wrapped into a size-1 map, the whole body -- the device-wide ``Reduce`` included -- ran in one
    GPU thread per row. npbench gem hit it on the GPU canonicalize column: no result within 420 s,
    against 2.3 s a call without the wrap.
    """
    sdfg = offloaded(rows_over_a_reducing_body(side_tasklet), heuristics)
    assert map_schedule(sdfg, 'rows') == dtypes.ScheduleType.Sequential
    assert not node_is_device_code(sdfg, 'Reduce'), 'the Reduce was wrapped into a kernel'
    sdfg.validate()


@pytest.mark.parametrize('heuristics', [True, False])
def test_the_single_element_tasklet_beside_the_nested_reduce_becomes_a_kernel(heuristics):
    """``zero`` shares its state with the ``Reduce`` -- a hybrid state -- so it gets a size-1 kernel."""
    sdfg = offloaded(rows_over_a_reducing_body(side_tasklet=False), heuristics)
    assert node_is_device_code(sdfg, 'zero'), 'the tasklet beside the Reduce stayed host code'


@pytest.mark.parametrize('heuristics', [True, False])
def test_a_tasklet_beside_a_nested_reducing_body_becomes_a_kernel(heuristics):
    """Host work beside the nested SDFG is the hybrid case: ``side`` may not read a device ``T`` on the host."""
    sdfg = offloaded(rows_over_a_reducing_body(side_tasklet=True), heuristics)
    assert node_is_device_code(sdfg, 'side'), 'the side tasklet stayed host code'
    sdfg.validate()


@pytest.mark.parametrize('heuristics', [True, False])
def test_a_run_of_single_element_states_beside_inner_reduces_is_one_kernel(heuristics):
    """One kernel for the row body, not one per lone single-element state inside it.

    Each lone state wrapped on its own is a launch per state per row: npbench cp2k_density_matrix_trs4
    ran 4x slower on the GPU canonicalize column (210 ms against 51 ms a call at preset M).
    """
    sdfg = offloaded(rows_over_a_scalar_body_with_inner_reduces(), heuristics)
    kernels = {outermost_kernel(sdfg, f'flag{index}') for index in range(3)}
    assert len(kernels) == 1, f'the lone states launch {len(kernels)} kernels'
    sdfg.validate()


def test_a_block_body_in_a_hybrid_state_is_not_wrapped_into_a_kernel():
    """A nested SDFG that only launches keeps its maps as kernels instead of one thread running them."""
    sdfg = offloaded(blocked_nest_beside_a_host_read(), heuristics=True)
    assert map_schedule(sdfg, 'nblks') == dtypes.ScheduleType.Sequential
    assert not node_is_device_code(sdfg, 'block_body'), 'the block body was wrapped into a kernel'
    assert map_schedule(sdfg, 'scale') == dtypes.ScheduleType.GPU_Device


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


def triangular_nest() -> dace.SDFG:
    """The cholesky shape: an inner map whose extent is written in the OUTER map's parameter."""
    sdfg = dace.SDFG('triangular_nest')
    sdfg.add_array('A', [16, 16], dace.float64)
    sdfg.add_array('B', [16, 16], dace.float64)
    state = sdfg.add_state('body')
    read = state.add_read('A')
    outer_in, outer_out = state.add_map('outer', dict(i='0:16'))
    inner_in, inner_out = state.add_map('inner', dict(j='0:i + 1'))
    tasklet = state.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(read, outer_in, inner_in, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet,
                          inner_out,
                          outer_out,
                          state.add_write('B'),
                          src_conn='out',
                          memlet=dace.Memlet('B[i, j]'))
    sdfg.validate()
    return sdfg


def test_a_map_whose_body_extent_names_it_is_not_a_taskloop():
    """An inner extent written in the outer parameter disqualifies the outer map.

    Keeping it on the host would launch a different, shrinking kernel per iteration -- measured as a
    1.12x regression on polybench cholesky and 1.14x on npbench cholesky2, against a 5% floor -- and
    the extent has to reach a launch configuration where the outer parameter is not in scope, which
    is how polybench correlation fails to build at all.
    """
    sdfg = triangular_nest()
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
def test_a_tasklet_beside_a_launched_library_node_computes_its_values(heuristics):
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = offloaded(map_over_reduce_beside_a_tasklet(), heuristics)
    A = np.random.default_rng(3).random((8, 16))
    arrays = {'A': cupy.asarray(A), 'B': cupy.zeros(8), 'C': cupy.zeros(8)}
    sdfg(**arrays)
    assert np.allclose(arrays['B'].get(), (A + 1).sum(axis=1))
    assert np.allclose(arrays['C'].get(), (A[:, 0] + 1) * 3)


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_nested_reducing_body_beside_a_tasklet_computes_its_values(heuristics):
    sdfg = offloaded(rows_over_a_reducing_body(side_tasklet=True), heuristics)
    A = np.random.default_rng(4).random((8, 16))
    B, C = np.zeros(8), np.zeros(8)
    sdfg(A=A.copy(), B=B, C=C, k=np.array([2.0]))
    assert np.allclose(B, (A * -2.0).sum(axis=1))
    assert np.allclose(C, A[:, 0] * -6.0)


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [True, False])
def test_a_launched_block_body_computes_what_the_kernel_computes(heuristics):
    sdfg = offloaded(blocked_nest(), heuristics)
    A = np.random.default_rng(2).random((NBLKS, NPROMA, NLEV))
    B = np.zeros((NBLKS, NPROMA, NLEV))
    sdfg(A=A, B=B)
    assert np.allclose(B, blocked_nest_reference(A))


def region_with_an_ordering_edge() -> tuple:
    """A region whose BOUNDARY an empty memlet crosses -- the edge the wrapper has to rewire.

    ``first`` stays outside the wrapped region and orders ``second``, which goes in, so the wrapper
    must route that edge through the new map entry.
    """
    sdfg = dace.SDFG('ordering_edge')
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    state = sdfg.add_state('body')
    read, write = state.add_read('A'), state.add_write('B')
    first = state.add_tasklet('first', {'inp'}, {'out'}, 'out = inp + 1.0')
    second = state.add_tasklet('second', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(read, None, first, 'inp', dace.Memlet('A[0]'))
    state.add_edge(first, 'out', write, None, dace.Memlet('B[0]'))
    state.add_edge(read, None, second, 'inp', dace.Memlet('A[1]'))
    state.add_edge(second, 'out', write, None, dace.Memlet('B[1]'))
    # The ordering edge: it carries no data, only the guarantee that ``first`` happens before.
    state.add_nedge(first, second, dace.Memlet())
    sdfg.validate()
    return sdfg, state, {second}


def test_the_wrapper_leaves_an_ordering_edge_without_a_connector():
    """An empty memlet ORDERS. Given a connector it becomes a data edge with no data behind it, and
    connector type inference then trips over the ``None`` descriptor (npbench cholesky2)."""
    sdfg, state, region = region_with_an_ordering_edge()
    OffloadToAccelerator()._wrap_region_in_size1_map(state, set(region))

    empty = [edge for edge in state.edges() if edge.data is not None and edge.data.is_empty()]
    assert empty, 'the ordering edge was dropped, not rewired'
    offenders = [(str(edge.src), edge.src_conn, str(edge.dst), edge.dst_conn) for edge in empty
                 if edge.src_conn is not None or edge.dst_conn is not None]
    assert not offenders, f"ordering edges given a connector: {offenders}"

    wrapper = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and 'size1_wrap' in n.map.label)
    assert any(edge.dst is wrapper for edge in empty), 'the ordering edge no longer reaches the wrapper'
    from dace.sdfg import infer_types
    infer_types.infer_connector_types(sdfg)
    sdfg.validate()


def rows_over_a_libnode_body() -> tuple:
    """The npbench spmv shape: a row map whose body reads an array on an INTERSTATE EDGE.

    The ``Reduce`` inside the body makes the row map a taskloop whatever the config says (a
    device-wide call is issued by host code), so the body is host code -- and its edge assignment
    ``stop = bounds[1]`` is host code reading an array the outer level put on the device.
    """
    inner = dace.SDFG('row_body')
    inner.add_array('a', [16], dace.float64)
    inner.add_array('res', [1], dace.float64)
    inner.add_array('bounds', [2], dace.int64)
    inner.add_symbol('stop', dace.int64)
    pick = inner.add_state('pick', is_start_block=True)
    compute = inner.add_state('compute')
    inner.add_edge(pick, compute, dace.InterstateEdge(assignments=dict(stop='bounds[1]')))
    reduce_node = compute.add_reduce('lambda a, b: a + b', None, 0.0)
    compute.add_edge(compute.add_read('a'), None, reduce_node, '_in', dace.Memlet('a[0:stop]'))
    compute.add_edge(reduce_node, '_out', compute.add_write('res'), None, dace.Memlet('res[0]'))

    sdfg = dace.SDFG('rows_over_a_libnode_body')
    sdfg.add_array('A', [8, 16], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('bounds', [2], dace.int64)
    state = sdfg.add_state('rows')
    entry, exit_node = state.add_map('rows', dict(i='0:8'))
    nested = state.add_nested_sdfg(inner, dict(a=None, bounds=None), dict(res=None))
    state.add_memlet_path(state.add_read('A'), entry, nested, dst_conn='a', memlet=dace.Memlet('A[i, 0:16]'))
    state.add_memlet_path(state.add_read('bounds'), entry, nested, dst_conn='bounds', memlet=dace.Memlet('bounds[0:2]'))
    state.add_memlet_path(nested, exit_node, state.add_write('B'), src_conn='res', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg, nested


def interstate_read_storages(sdfg: dace.SDFG) -> dict:
    """Storage of every array an interstate edge of ``sdfg`` reads -- all of it host code."""
    found = {}
    for edge in sdfg.all_interstate_edges():
        for name in edge.data.used_arrays(sdfg.arrays):
            found[name] = sdfg.arrays[name].storage
    return found


def unmapped_body_over_a_kernel() -> tuple:
    """The npbench scattering_self_energies shape: a nested SDFG no map encloses.

    Its own state runs a kernel over ``a``, and its interstate edge reads ``bounds`` -- host code at
    a level the outer analysis only queries for where the body wants its arrays, never places within.
    """
    inner = dace.SDFG('plain_body')
    inner.add_array('a', [16], dace.float64)
    inner.add_array('res', [16], dace.float64)
    inner.add_array('bounds', [2], dace.int64)
    inner.add_symbol('stop', dace.int64)
    pick = inner.add_state('pick', is_start_block=True)
    compute = inner.add_state('compute')
    inner.add_edge(pick, compute, dace.InterstateEdge(assignments=dict(stop='bounds[1]')))
    # The kernel reads ``bounds`` too, so the body's array analysis wants it on the device while
    # the edge above needs it on the host -- the split that only placing within the body resolves.
    compute.add_mapped_tasklet('scale', {'j': '0:16'}, {
        'inp': dace.Memlet('a[j]'),
        'lim': dace.Memlet('bounds[0]')
    },
                               'o = inp * stop + lim', {'o': dace.Memlet('res[j]')},
                               external_edges=True)

    sdfg = dace.SDFG('unmapped_body_over_a_kernel')
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    sdfg.add_array('bounds', [2], dace.int64)
    state = sdfg.add_state('body')
    nested = state.add_nested_sdfg(inner, dict(a=None, bounds=None), dict(res=None))
    state.add_edge(state.add_read('A'), None, nested, 'a', dace.Memlet('A[0:16]'))
    state.add_edge(state.add_read('bounds'), None, nested, 'bounds', dace.Memlet('bounds[0:2]'))
    state.add_edge(nested, 'res', state.add_write('B'), None, dace.Memlet('B[0:16]'))
    sdfg.validate()
    return sdfg, nested


def body_taking_one_element_by_scalar() -> tuple:
    """The npbench azimint_hist shape: a body whose SCALAR input is one element of a device array."""
    inner = dace.SDFG('scalar_body')
    inner.add_scalar('lim', dace.float64)
    inner.add_array('a', [16], dace.float64)
    inner.add_array('res', [16], dace.float64)
    state = inner.add_state('sub')
    # Host code: a tasklet nobody encloses, so it dereferences ``lim`` where it lies.
    tasklet = state.add_tasklet('sub', {'l': None, 'inp': None}, {'o': None}, 'o = inp - l')
    state.add_edge(state.add_read('lim'), None, tasklet, 'l', dace.Memlet('lim[0]'))
    state.add_edge(state.add_read('a'), None, tasklet, 'inp', dace.Memlet('a[0]'))
    state.add_edge(tasklet, 'o', state.add_write('res'), None, dace.Memlet('res[0]'))

    sdfg = dace.SDFG('body_taking_one_element_by_scalar')
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    sdfg.add_array('C', [16], dace.float64)
    kernel = sdfg.add_state('kernel')
    kernel.add_mapped_tasklet('double', {'i': '0:16'}, {'inp': dace.Memlet('A[i]')},
                              'o = inp * 2.0', {'o': dace.Memlet('C[i]')},
                              external_edges=True)
    state = sdfg.add_state_after(kernel, 'body')
    nested = state.add_nested_sdfg(inner, dict(lim=None, a=None), dict(res=None))
    state.add_edge(state.add_read('A'), None, nested, 'lim', dace.Memlet('A[3]'))
    state.add_edge(state.add_read('C'), None, nested, 'a', dace.Memlet('C[0:16]'))
    state.add_edge(nested, 'res', state.add_write('B'), None, dace.Memlet('B[0:16]'))
    sdfg.validate()
    return sdfg, nested


@pytest.mark.parametrize('heuristics', [False, True])
def test_a_scalar_bound_to_device_memory_is_staged_on_the_host(heuristics):
    """A scalar connector reaches the body BY REFERENCE, so host code there reads the outer array.

    Placement cannot see it -- it works on arrays, and this pass asserts no scalar ever enters its
    sets -- so the element itself is copied to a host scalar and the connector rebound onto it.
    """
    sdfg, nested = body_taking_one_element_by_scalar()
    offloaded(sdfg, heuristics=heuristics)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    on_device = {
        name: desc.storage
        for name, desc in nested.sdfg.arrays.items()
        if isinstance(desc, dace.data.Scalar) and desc.storage in GPU_RESIDENT_STORAGES
    }
    assert not on_device, f'the body dereferences device memory through a scalar: {on_device}'
    sdfg.validate()


@pytest.mark.parametrize('heuristics', [False, True])
def test_an_unmapped_body_reads_its_interstate_arrays_on_the_host(heuristics):
    """Every nested SDFG still on the host is its own level, not only the ones a taskloop launches."""
    sdfg, nested = unmapped_body_over_a_kernel()
    offloaded(sdfg, heuristics=heuristics)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    on_device = {
        name: storage
        for name, storage in interstate_read_storages(nested.sdfg).items() if storage in GPU_RESIDENT_STORAGES
    }
    assert not on_device, f"the body's interstate edge reads device memory from host code: {on_device}"
    sdfg.validate()


def body_with_a_sequential_fallback_arm() -> tuple:
    """The bdf_newton_krylov ``newton_matvec`` shape: a body whose host code writes what its kernel reads.

    Canonicalization guards a loop it can only parallelize at run time and keeps the original loop as
    the fallback arm, which stays host code by design. Its writes to ``lap`` must be copied down before
    the kernel reads it, and that copy is placed by the body's own level.
    """
    inner = dace.SDFG('fallback_body')
    inner.add_array('a', [16], dace.float64)
    inner.add_array('res', [16], dace.float64)
    inner.add_array('lap', [16], dace.float64, transient=True)
    inner.add_symbol('n', dace.int64)
    guard = ConditionalBlock('guard')
    inner.add_node(guard, is_start_block=True)
    arm = ControlFlowRegion('sequential_arm', sdfg=inner)
    guard.add_branch(CodeBlock('n > 0'), arm)
    loop = LoopRegion('corners', 'k < 16', 'k', 'k = 0', 'k = k + 1')
    arm.add_node(loop, is_start_block=True)
    corner = loop.add_state('corner', is_start_block=True)
    twice = corner.add_tasklet('twice', {'x': None}, {'o': None}, 'o = 2.0 * x')
    corner.add_edge(corner.add_read('a'), None, twice, 'x', dace.Memlet('a[k]'))
    corner.add_edge(twice, 'o', corner.add_write('lap'), None, dace.Memlet('lap[k]'))
    compute = inner.add_state('compute')
    inner.add_edge(guard, compute, dace.InterstateEdge())
    compute.add_mapped_tasklet('scale', {'j': '0:16'}, {'inp': dace.Memlet('lap[j]')},
                               'o = inp * 2.0', {'o': dace.Memlet('res[j]')},
                               external_edges=True)

    sdfg = dace.SDFG('body_with_a_sequential_fallback_arm')
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    sdfg.add_symbol('n', dace.int64)
    state = sdfg.add_state('body')
    nested = state.add_nested_sdfg(inner, dict(a=None), dict(res=None), symbol_mapping=dict(n='n'))
    state.add_edge(state.add_read('A'), None, nested, 'a', dace.Memlet('A[0:16]'))
    state.add_edge(nested, 'res', state.add_write('B'), None, dace.Memlet('B[0:16]'))
    sdfg.validate()
    return sdfg, nested


def uploads(sdfg: dace.SDFG) -> set:
    """Host containers ``sdfg``'s own states copy into device memory."""
    found = set()
    for state in sdfg.states():
        for edge in state.edges():
            if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
                src, dst = (sdfg.arrays[n.data].storage for n in (edge.src, edge.dst))
                if src not in GPU_RESIDENT_STORAGES and dst in GPU_RESIDENT_STORAGES:
                    found.add(edge.src.data)
    return found


@pytest.mark.parametrize('heuristics', [False, True])
def test_a_body_uploads_what_its_host_fallback_arm_writes(heuristics):
    sdfg, nested = body_with_a_sequential_fallback_arm()
    offloaded(sdfg, heuristics=heuristics)
    sdfg.validate()
    assert 'lap' in uploads(nested.sdfg), uploads(nested.sdfg)


@pytest.mark.gpu
@pytest.mark.parametrize('heuristics', [False, True])
def test_the_kernel_after_a_host_fallback_arm_reads_its_writes(heuristics):
    sdfg, _ = body_with_a_sequential_fallback_arm()
    offloaded(sdfg, heuristics=heuristics)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    A = np.arange(16, dtype=np.float64)
    B = np.zeros(16)
    sdfg(A=A, B=B, n=1)
    np.testing.assert_array_equal(B, 4.0 * A)


@pytest.mark.parametrize('heuristics', [False, True])
def test_a_taskloop_body_reads_its_interstate_arrays_on_the_host(heuristics):
    """A taskloop exists whatever the config says, so its body must be placed whatever it says too.

    Left unplaced, the body keeps reading the outer level's device array from host code, which is
    an invalid SDFG (npbench spmv's ``start = A_indptr[i]``).
    """
    sdfg, nested = rows_over_a_libnode_body()
    offloaded(sdfg, heuristics=heuristics)
    # The step of the real pipeline that makes the omission visible: it propagates the outer
    # binding's storage into the body's own descriptors, so a body nobody placed ends up reading
    # device memory under a host schedule.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    assert map_schedule(sdfg, 'rows') is dtypes.ScheduleType.Sequential
    # ``dtypes.GPU_STORAGES`` lists ``GPU_Shared`` alone, so it would pass this vacuously.
    on_device = {
        name: storage
        for name, storage in interstate_read_storages(nested.sdfg).items() if storage in GPU_RESIDENT_STORAGES
    }
    assert not on_device, f"the body's interstate edge reads device memory from host code: {on_device}"
    sdfg.validate()


def wide_outer_narrow_body() -> dace.SDFG:
    """A large ``i`` map whose body is a small one -- the shape the volume rule declines.

    Made a taskloop this launches 4096 kernels of 8 threads each. Left alone it is one kernel of
    4096, which is what a GPU can actually fill, so the rule has to prefer the second.
    """
    sdfg = dace.SDFG('wide_outer_narrow_body')
    sdfg.add_array('A', [4096, 8], dace.float64)
    state = sdfg.add_state('body')
    outer_in, outer_out = state.add_map('outer', dict(i='0:4096'))
    inner_in, inner_out = state.add_map('inner', dict(j='0:8'))
    read, write = state.add_access('A'), state.add_access('A')
    tasklet = state.add_tasklet('scale', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(read, outer_in, inner_in, tasklet, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet, inner_out, outer_out, write, src_conn='b', memlet=dace.Memlet('A[i, j]'))
    return sdfg


def symbolic_blocked_nest() -> dace.SDFG:
    """ICON's shape with every extent left as a symbol: ``nblks`` outside, ``nproma`` x ``nlev`` in.

    Nothing pins the three at compile time, so the rule has only their shapes to go on -- one axis
    against two. That is what "assume every unpinned symbol is worth the same" is for.
    """
    sdfg = dace.SDFG('symbolic_blocked_nest')
    for name in ('nblks', 'nproma', 'nlev'):
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_array('A', [dace.symbol('nblks'), dace.symbol('nproma'), dace.symbol('nlev')], dace.float64)
    state = sdfg.add_state('body')
    outer_in, outer_out = state.add_map('blocks', dict(b='0:nblks'))
    inner_in, inner_out = state.add_map('columns', dict(p='0:nproma', k='0:nlev'))
    read, write = state.add_access('A'), state.add_access('A')
    tasklet = state.add_tasklet('scale', {'a'}, {'c'}, 'c = a * 2.0')
    state.add_memlet_path(read, outer_in, inner_in, tasklet, dst_conn='a', memlet=dace.Memlet('A[b, p, k]'))
    state.add_memlet_path(tasklet, inner_out, outer_out, write, src_conn='c', memlet=dace.Memlet('A[b, p, k]'))
    return sdfg


def outer_entry(sdfg: dace.SDFG, label: str) -> nodes.MapEntry:
    """The map entry with this label, and the state it lives in."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node.map.label == label:
                return node
    raise AssertionError(f'no map labelled {label}')


def classify(sdfg: dace.SDFG, label: str, overrides=None) -> bool:
    """Would ``label`` be a taskloop, with the heuristics on?"""
    state = next(s for s in sdfg.states() if any(
        isinstance(n, nodes.MapEntry) and n.map.label == label for n in s.nodes()))
    return is_taskloop_map(state, outer_entry(sdfg, label), state.scope_children(), True, overrides)


def test_a_map_wider_than_its_body_keeps_the_kernel_to_itself():
    """4096 outside against 8 inside: the wider launch wins and the map is not a taskloop."""
    assert not classify(wide_outer_narrow_body(), 'outer')


def test_a_map_narrower_than_its_body_hands_the_kernel_down():
    """16 outside against 16 x 16 inside: the body is wider, so the map launches rather than computes."""
    sdfg = uncollapsed_nest()
    assert classify(sdfg, 'outer')


def test_unpinned_extents_are_compared_by_shape_alone():
    """``nblks`` against ``nproma * nlev``: one symbolic axis against two, so the body is wider.

    No extent is known here. Giving every unpinned symbol the same value is what lets the rule
    answer at all, and it answers on the only thing that is actually visible -- the nesting.
    """
    assert classify(symbolic_blocked_nest(), 'blocks')


def test_a_named_map_is_taskloop_or_not_whatever_the_rules_say():
    """``taskloop_overrides`` decides outright, in both directions."""
    wide = wide_outer_narrow_body()
    assert not classify(wide, 'outer')
    assert classify(wide, 'outer', {'outer': True}), 'an override could not force a taskloop on'

    narrow = uncollapsed_nest()
    assert classify(narrow, 'outer')
    assert not classify(narrow, 'outer', {'outer': False}), 'an override could not force a taskloop off'


def test_an_override_reaches_the_pass_through_its_property():
    """The dict handed to ``OffloadToAccelerator`` is what ``find_taskloops`` classifies with."""
    sdfg = wide_outer_narrow_body()
    entry = outer_entry(sdfg, 'outer')
    with dace.config.set_temporary('optimizer', 'gpu_taskloop_heuristics', value=True):
        offloader = OffloadToAccelerator(taskloop_overrides={'outer': True})
        offloader.cache_scopes(sdfg)
        offloader.taskloop_heuristics = True
        offloader.find_taskloops(sdfg)
    assert entry in offloader.taskloops, 'the pass ignored the override it was constructed with'
