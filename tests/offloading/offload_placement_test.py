# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Where ``OffloadToAccelerator`` places data and copies: one graph per defect, plus its numeric companion."""
import numpy as np
import pytest
from ordered_set import OrderedSet

import dace
from dace import data, dtypes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator
from dace.transformation.passes.offloading import offloading_helpers as helpers
from dace.transformation.passes.offloading.offloading_helpers import traverse_IR
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode

#: Extent of the ordering-edge graph.
LENGTH = 16


def host_tasklet_behind_an_interstate_read() -> dace.SDFG:
    """A device map over ``A``, then a state whose HOST tasklet writes ``A[0]``.

    The interstate edge between them reads ``C``, which is what makes the pass build an edge node
    for the second state -- and an edge node carries the same block object as the state node that
    follows it.
    """
    sdfg = dace.SDFG('host_tasklet_behind_an_interstate_read')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('C', [256], dace.int64)
    first = sdfg.add_state('device_work')
    first.add_mapped_tasklet('scale', {'i': '0:256'}, {'inp': dace.Memlet('A[i]')},
                             'out = inp * 2.0', {'out': dace.Memlet('A[i]')},
                             external_edges=True)
    second = sdfg.add_state('host_work')
    tasklet = second.add_tasklet('bump', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    second.add_edge(second.add_read('A'), None, tasklet, 'inp', dace.Memlet('A[0]'))
    second.add_edge(tasklet, 'out', second.add_write('A'), None, dace.Memlet('A[0]'))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'k': 'C[0]'}))
    sdfg.validate()
    return sdfg


def test_an_interstate_read_does_not_hand_the_next_state_the_device_name():
    """An edge node decides about the interstate edges REACHING a block, not about the block.

    It holds the same block object as the state node behind it, so letting its decision fall
    through to the block renamed dataflow the state node had already placed on the host -- tsvc
    ``s315``, where a host tasklet writing ``a`` came out writing ``a_gpu``.
    """
    sdfg = host_tasklet_behind_an_interstate_read()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    host_state = next(state for state in sdfg.states() if state.label == 'host_work')
    written = {edge.data.data for edge in host_state.edges() if not edge.data.is_empty()}
    assert not [name for name in written if sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global
                ], (f'host tasklet left holding a device container: {written}')


def two_host_tasklets_beside_a_kernel_then_an_interstate_read() -> dace.SDFG:
    """Two unconnected host tasklets get two size-1 wrappers that fuse; the next edge reads ``B[0]``."""
    sdfg = dace.SDFG('two_host_tasklets_beside_a_kernel_then_an_interstate_read')
    for name, size in (('A', 16), ('B', 16), ('C', 1), ('D', 1), ('E', 16)):
        sdfg.add_array(name, [size], dace.float64)
    mixed = sdfg.add_state('mixed', is_start_block=True)
    mixed.add_mapped_tasklet('double', {'i': '0:16'}, {'inp': dace.Memlet('A[i]')},
                             'out = inp * 2.0', {'out': dace.Memlet('B[i]')},
                             external_edges=True)
    for label, index, target, offset in (('first', 0, 'C', 1.0), ('second', 1, 'D', 2.0)):
        tasklet = mixed.add_tasklet(label, {'inp'}, {'out'}, f'out = inp + {offset}')
        mixed.add_edge(mixed.add_read('A'), None, tasklet, 'inp', dace.Memlet(f'A[{index}]'))
        mixed.add_edge(tasklet, 'out', mixed.add_write(target), None, dace.Memlet(f'{target}[0]'))
    after = sdfg.add_state('after')
    after.add_mapped_tasklet('shift', {'i': '0:16'}, {'inp': dace.Memlet('B[i]')},
                             'out = inp + k', {'out': dace.Memlet('E[i]')},
                             external_edges=True)
    sdfg.add_edge(mixed, after, dace.InterstateEdge(assignments={'k': 'B[0]'}))
    sdfg.validate()
    # What ``apply_gpu_storage`` does to a signature; the edge now reads device memory until copies exist.
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


def test_fusing_the_wrappers_does_not_validate_before_the_copies_exist():
    """CloudSC's ``pap``: the wrapper fusion validated a graph whose interstate edge still read ``B``."""
    sdfg = two_host_tasklets_beside_a_kernel_then_an_interstate_read()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()

    mixed = next(state for state in sdfg.states() if state.label == 'mixed')
    scopes = mixed.scope_dict()
    wrappers = {
        scopes[node]
        for node in mixed.nodes() if isinstance(node, dace.nodes.Tasklet) and node.label in ('first', 'second')
    }
    assert len(wrappers) == 1, f'the two size-1 wrappers were not fused: {wrappers}'
    assert next(iter(wrappers)).map.schedule == dtypes.ScheduleType.GPU_Device
    edge = next(edge for edge in sdfg.all_interstate_edges() if edge.dst.label == 'after')
    assert set(edge.data.used_arrays(sdfg.arrays)) == {'B_host'}, edge.data.assignments


@pytest.mark.gpu
def test_the_fused_wrappers_compute_what_the_host_tasklets_computed():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = two_host_tasklets_beside_a_kernel_then_an_interstate_read()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    host_a = np.random.default_rng(7).random(16)
    arrays = {
        'A': cupy.asarray(host_a),
        'B': cupy.zeros(16),
        'C': cupy.zeros(1),
        'D': cupy.zeros(1),
        'E': cupy.zeros(16)
    }
    sdfg(**arrays)
    assert np.allclose(arrays['B'].get(), host_a * 2.0)
    assert np.allclose(arrays['C'].get(), [host_a[0] + 1.0])
    assert np.allclose(arrays['D'].get(), [host_a[1] + 2.0])
    assert np.allclose(arrays['E'].get(), host_a * 2.0 + host_a[0] * 2.0)


def fallback_arm_first_then_an_interstate_read() -> dace.SDFG:
    """A guard whose FIRST arm is a host loop, a kernel, then an edge reading ``A[0]``.

    Nothing before the edge touches ``A``, so only propagation carries its location to the edge.
    """
    sdfg = dace.SDFG('fallback_arm_first_then_an_interstate_read')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [4], dace.int64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [16], dace.float64, storage=dtypes.StorageType.GPU_Global)
    start = sdfg.add_state('start', is_start_block=True)

    dispatch = ConditionalBlock('dispatch')
    fallback = ControlFlowRegion('fallback', sdfg=sdfg)
    loop = LoopRegion('seq', 'i < 16', 'i', 'i = 0', 'i = i + 1')
    fallback.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    bump = body.add_tasklet('bump', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    body.add_edge(body.add_read('B'), None, bump, 'inp', dace.Memlet('B[i]'))
    body.add_edge(bump, 'out', body.add_write('B'), None, dace.Memlet('B[i]'))
    dispatch.add_branch(dace.properties.CodeBlock('N < 4'), fallback)
    parallel = ControlFlowRegion('parallel', sdfg=sdfg)
    parallel.add_state('par', is_start_block=True).add_mapped_tasklet('bump_all', {'i': '0:16'},
                                                                      {'inp': dace.Memlet('B[i]')},
                                                                      'out = inp + 1.0', {'out': dace.Memlet('B[i]')},
                                                                      external_edges=True)
    dispatch.add_branch(None, parallel)
    sdfg.add_node(dispatch)
    sdfg.add_edge(start, dispatch, dace.InterstateEdge())

    between = sdfg.add_state('between')
    between.add_mapped_tasklet('double', {'i': '0:16'}, {'inp': dace.Memlet('B[i]')},
                               'out = inp * 2.0', {'out': dace.Memlet('B[i]')},
                               external_edges=True)
    sdfg.add_edge(dispatch, between, dace.InterstateEdge())
    after = sdfg.add_state('after')
    after.add_mapped_tasklet('shift', {'i': '0:16'}, {'inp': dace.Memlet('B[i]')},
                             'out = inp + k', {'out': dace.Memlet('B[i]')},
                             external_edges=True)
    sdfg.add_edge(between, after, dace.InterstateEdge(assignments={'k': 'A[0]'}))
    return sdfg


def test_a_join_hands_on_the_locations_its_later_arm_carries():
    """QE vexx_k: the edge read ``iexx_istart_host``, a copy nobody made.

    The fallback arm's tail does not propagate into the guard's close, so only the parallel arm
    carries ``A`` there. Walked depth-first, the close and everything after it were visited from the
    fallback arm first, before the parallel arm arrived: the edge was renamed onto the host twin
    while its predecessor recorded no location for ``A``, so no copy was placed.
    """
    sdfg = fallback_arm_first_then_an_interstate_read()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()

    edge = next(edge for edge in sdfg.all_interstate_edges() if edge.dst.label == 'after')
    assert set(edge.data.used_arrays(sdfg.arrays)) == {'A_host'}, edge.data.assignments
    assert copy_blocks_for(sdfg, 'A') == ['copy_A_to_host'], 'the host read needs exactly one copy of A'


@pytest.mark.gpu
def test_a_join_hands_on_the_locations_its_later_arm_carries_and_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    b = np.arange(16, dtype=np.float64)
    sdfg = fallback_arm_first_then_an_interstate_read()
    sdfg.apply_gpu_transformations()
    compiled = sdfg.compile()
    for n in (2, 8):  # both arms of the guard
        arrays = {'A': cupy.asarray([5, 0, 0, 0]), 'B': cupy.asarray(b)}
        compiled(**arrays, N=n)
        np.testing.assert_array_equal(arrays['B'].get(), (b + 1.0) * 2.0 + 5.0)


def free_computation_with_a_reading_and_a_sourceless_tasklet() -> dace.SDFG:
    """One state whose top level holds a device map and a free region with two roots.

    The two roots differ in exactly what the bug turned on: ``scale`` reads an array, so wrapping
    the region rewires a real edge for it; ``seed`` reads nothing, so it has no edge to rewire and
    only an ordering edge can put it under the entry. Both feed ``combine``, which is what makes
    them one region rather than two -- and what makes leaving one behind an invalid path rather
    than a missed wrap (tsvc s252's shape).
    """
    sdfg = dace.SDFG('free_computation_with_a_reading_and_a_sourceless_tasklet')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('B', [256], dace.float64)
    sdfg.add_scalar('half', dace.float64, transient=True)
    sdfg.add_scalar('bias', dace.float64, transient=True)
    state = sdfg.add_state('mixed')

    state.add_mapped_tasklet('device', {'i': '0:256'}, {'inp': dace.Memlet('A[i]')},
                             'out = inp * 2.0', {'out': dace.Memlet('B[i]')},
                             external_edges=True)

    scale = state.add_tasklet('scale', {'inp': None}, {'out': None}, 'out = inp * 0.5')
    half = state.add_access('half')
    state.add_edge(state.add_read('A'), None, scale, 'inp', dace.Memlet('A[0]'))
    state.add_edge(scale, 'out', half, None, dace.Memlet('half[0]'))

    seed = state.add_tasklet('seed', {}, {'out': None}, 'out = 1.0')
    bias = state.add_access('bias')
    state.add_edge(seed, 'out', bias, None, dace.Memlet('bias[0]'))

    combine = state.add_tasklet('combine', {'lhs': None, 'rhs': None}, {'out': None}, 'out = lhs + rhs')
    state.add_edge(half, None, combine, 'lhs', dace.Memlet('half[0]'))
    state.add_edge(bias, None, combine, 'rhs', dace.Memlet('bias[0]'))
    state.add_edge(combine, 'out', state.add_write('B'), None, dace.Memlet('B[0]'))
    sdfg.validate()
    return sdfg


def test_a_wrapped_region_puts_every_root_under_its_entry():
    """A size-1 wrapper holds the whole region, including the parts with nothing to rewire.

    Leaving one root outside is not a missing optimization: the rest of the region IS in the scope,
    so the edge between them runs from inside the map to outside it and validation rejects the
    graph -- tsvc ``s252``, ``sink node _Add_ should be a data node``.
    """
    sdfg = free_computation_with_a_reading_and_a_sourceless_tasklet()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    state = next(s for s in sdfg.states() if s.label == 'mixed')
    scopes = state.scope_dict()
    wrapped = next(n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet) and n.label == 'seed')
    assert scopes[wrapped] is not None, 'a tasklet that reads nothing was left outside its wrapper'


ROWS = dace.symbol('ROWS')


@dace.program
def copy_into_a_host_recurrence(x: dace.float64[ROWS], flag: dace.int64, out: dace.float64[ROWS]):
    """``u`` lives on the device (the first arm writes it there); the second copies into it, then scans it on the host."""
    t = x * 2.0
    u = np.empty_like(t)
    if flag > 0:
        u[:] = t + 1.0
    else:
        u[:] = t
        for i in range(1, ROWS):
            u[i] = u[i] + u[i - 1]
    out[:] = u


def test_a_copy_hands_its_destination_the_side_of_its_source():
    sdfg = copy_into_a_host_recurrence.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    loop = next(b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, LoopRegion))
    arm = loop.parent_graph
    fills = [b for b in arm.predecessors(loop) if b.label == 'copy_u_to_host']
    assert fills, [b.label for b in arm.nodes()]


@pytest.mark.gpu
def test_a_copy_before_a_host_recurrence_computes_what_numpy_computes():
    sdfg = copy_into_a_host_recurrence.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    x = np.arange(8, dtype=np.float64)
    for flag, want in ((1, x * 2.0 + 1.0), (0, np.cumsum(x * 2.0))):
        out = np.zeros(8)
        sdfg(x=x, flag=flag, out=out, ROWS=8)
        np.testing.assert_allclose(out, want)


def kernel_then_host_staging_copy() -> dace.SDFG:
    """A kernel writes ``A``; host code then copies ``A[0]`` into the host Scalar ``s``."""
    sdfg = dace.SDFG('kernel_then_host_staging_copy')
    sdfg.add_array('A', [8], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_scalar('s', dace.float64, transient=True)
    state = sdfg.add_state('main', is_start_block=True)
    _, _, exit_node = state.add_mapped_tasklet('fill', {'i': '0:8'}, {},
                                               'a = 1.0', {'a': dace.Memlet('A[i]')},
                                               schedule=dtypes.ScheduleType.GPU_Device,
                                               external_edges=True)
    written = state.out_edges(exit_node)[0].dst
    state.add_edge(written, None, state.add_write('s'), None, dace.Memlet('A[0] -> [0]'))
    return sdfg


def test_a_host_copy_out_of_a_kernel_output_is_not_a_kernel_write():
    """The staged host Scalar downstream of the kernel's output is written by a copy the host issues,
    so it may stay by value (polybench durbin's ``alpha_host``)."""
    sdfg = kernel_then_host_staging_copy()

    written = helpers.data_written_by_device_code(sdfg)

    assert list(written) == ['A']


def kernel_writing_a_scalar_a_later_state_reads() -> dace.SDFG:
    """A hybrid state whose free tasklet writes a scalar, and a second state that reads it.

    ``pick`` reads device data, so the size-1 wrapper pulls it in -- and the ``acc`` access node it
    writes comes with it. Nothing then crosses the ``MapExit``, which is the only boundary the
    placement analysis looked at.
    """
    sdfg = dace.SDFG('kernel_writing_a_scalar_a_later_state_reads')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('B', [256], dace.float64)
    sdfg.add_scalar('acc', dace.float64, transient=True)

    produce = sdfg.add_state('produce')
    produce.add_mapped_tasklet('device', {'i': '0:256'}, {'inp': dace.Memlet('A[i]')},
                               'out = inp * 2.0', {'out': dace.Memlet('B[i]')},
                               external_edges=True)
    pick = produce.add_tasklet('pick', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    acc = produce.add_access('acc')
    produce.add_edge(produce.add_read('B'), None, pick, 'inp', dace.Memlet('B[0]'))
    produce.add_edge(pick, 'out', acc, None, dace.Memlet('acc[0]'))
    # A second consumer INSIDE the region is what makes ``acc`` interior. Without it the access node
    # is the region's last node, the wrapper leaves it outside the ``MapExit``, and the old
    # boundary-only analysis already saw it -- so the shape under test would not be durbin's.
    use = produce.add_tasklet('use', {'inp': None}, {'out': None}, 'out = inp * 3.0')
    produce.add_edge(acc, None, use, 'inp', dace.Memlet('acc[0]'))
    produce.add_edge(use, 'out', produce.add_write('B'), None, dace.Memlet('B[1]'))

    consume = sdfg.add_state_after(produce, 'consume')
    spread = consume.add_tasklet('spread', {'inp': None}, {'out': None}, 'out = inp')
    consume.add_edge(consume.add_read('acc'), None, spread, 'inp', dace.Memlet('acc[0]'))
    consume.add_edge(spread, 'out', consume.add_write('A'), None, dace.Memlet('A[0]'))
    sdfg.validate()
    return sdfg


def test_a_scalar_a_kernel_writes_and_a_later_state_reads_is_device_resident():
    """A scalar goes into a kernel BY VALUE, so a kernel that writes one loses the write.

    No error is raised and no launch fails -- polybench durbin ran to completion and returned wrong
    numbers, because every iteration read the host ``alpha`` the previous kernel had only written to
    its own stack. The write is observable outside the kernel, so the descriptor has to be device
    memory and the parameter a pointer.
    """
    sdfg = kernel_writing_a_scalar_a_later_state_reads()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    written = [name for name, desc in sdfg.arrays.items() if name.startswith('acc') and desc.transient]
    assert written, 'the scalar vanished, so this asserts nothing'
    resident = [name for name in written if sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES]
    assert resident, (f'no device copy of the kernel-written scalar: '
                      f'{[(n, sdfg.arrays[n].storage.name) for n in written]}')

    # Not filtered by ``obj.language``: the CUDA target names the device file's language after the
    # detected backend (``cu`` for CUDA, ``cpp`` for HIP -- ``dace/codegen/targets/cuda.py``), so a
    # language-specific filter here would only run on an NVIDIA host.
    signatures = [line for obj in sdfg.generate_code() for line in obj.clean_code.splitlines() if '__global__' in line]
    assert signatures, 'nothing was emitted as a kernel, so the signature asserts nothing'
    by_value = [line for line in signatures for name in resident if f'double {name}' in line]
    assert not by_value, f'a kernel takes a scalar it writes by value: {by_value}'


def copy_blocks_for(sdfg: dace.SDFG, name: str):
    """The copy states the pass inserted for ``name``, by the label ``create_interstate_copy`` gives."""
    return [
        block.label for block in sdfg.all_control_flow_blocks()
        if block.label.startswith('copy_') and f'copy_{name}_' in block.label
    ]


def indirect_read_only_sdfg() -> dace.SDFG:
    """``idx`` read by a parallel map AND by a sequential data-dependent loop, and written by neither.

    The map is the parallel consumer: it reads ``idx[i]`` to place its result, which is the indirect
    access an index array exists for. The loop is the sequential one: each step's index is the
    PREVIOUS step's value, so it cannot be a map and it is host code. Nothing writes ``idx``, so the
    two sides can never disagree about it -- which is the whole reason one copy is enough.
    """
    sdfg = dace.SDFG('indirect_read_only')
    sdfg.add_array('idx', [16], dace.int64, transient=False, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('data', [16], dace.float64, transient=False, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('out', [16], dace.float64, transient=False, storage=dace.StorageType.GPU_Global)
    sdfg.add_scalar('cursor', dace.int64, transient=True)
    sdfg.add_symbol('k', dace.int64)

    parallel = sdfg.add_state('parallel', is_start_block=True)
    entry, exit_ = parallel.add_map('gather', {'i': '0:16'}, schedule=dace.ScheduleType.GPU_Device)
    gather = parallel.add_tasklet('gather', {'j': None, 'd': None}, {'o': None}, 'o = d * float(j)')
    parallel.add_memlet_path(parallel.add_read('idx'), entry, gather, dst_conn='j', memlet=dace.Memlet('idx[i]'))
    parallel.add_memlet_path(parallel.add_read('data'), entry, gather, dst_conn='d', memlet=dace.Memlet('data[i]'))
    parallel.add_memlet_path(gather, exit_, parallel.add_write('out'), src_conn='o', memlet=dace.Memlet('out[i]'))

    # The pointer chase: sequential by construction, and host code. Not a guarded fallback -- there
    # is no ConditionalBlock above it, so its reads of `idx` are reads every execution performs.
    chase = LoopRegion('chase', 'k < 16', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(chase)
    sdfg.add_edge(parallel, chase, dace.InterstateEdge())
    step = chase.add_state('step', is_start_block=True)
    hop = step.add_tasklet('hop', {'j': None}, {'c': None}, 'c = j')
    step.add_edge(step.add_read('idx'), None, hop, 'j', dace.Memlet('idx[k]'))
    step.add_edge(hop, 'c', step.add_write('cursor'), None, dace.Memlet('cursor[0]'))
    return sdfg


def test_a_read_only_array_both_sides_read_is_copied_exactly_once():
    """The duplicate is made once at entry, not once per crossing.

    ``idx`` is read on both sides and written by neither, so the host copy can never go stale: one
    copy at entry serves every host read for the rest of the run. Counting is the assertion that
    separates this from the read-write placement, which has to copy on every crossing to stay
    coherent -- and the loop here would make that one copy per iteration.
    """
    sdfg = indirect_read_only_sdfg()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    copies = copy_blocks_for(sdfg, 'idx')
    assert len(copies) == 1, f'expected one copy of a read-only array, got {copies}'
    assert copies[0].endswith('_to_host'), f'the copy must bring idx down to the host: {copies[0]}'
    assert sdfg.arrays['idx'].storage == dace.StorageType.GPU_Global, 'the device side keeps the original'
    assert sdfg.arrays['idx_host'].storage != dace.StorageType.GPU_Global, 'the host side must be host memory'


def test_a_host_only_array_is_staged_once_each_way_and_not_wrapped():
    """A container only host code touches gets a home, not a kernel.

    The alternative the pass reaches for -- declaring the state hybrid and lifting the tasklet into
    a size-1 map -- is correct but pays a kernel launch per iteration for a scalar accumulation.
    Staging it costs one copy each way for the whole run, and the write-back is what makes the
    caller's array hold the answer.
    """
    sdfg = indirect_read_only_sdfg()
    # `total` is touched by the host loop alone: no map, no library node, nothing on the device.
    sdfg.add_array('total', [16], dace.float64, transient=False, storage=dace.StorageType.GPU_Global)
    step = next(state for state in sdfg.all_states() if state.label == 'step')
    accumulate = step.add_tasklet('accumulate', {}, {'t': None}, 't = 1.0')
    step.add_edge(accumulate, 't', step.add_write('total'), None, dace.Memlet('total[k]'))

    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    copies = copy_blocks_for(sdfg, 'total')
    assert len(copies) == 2, f'expected one copy each way for a written host-only array, got {copies}'
    assert sorted(c.split('_')[-2] + '_' + c.split('_')[-1] for c in copies) == ['to_gpu', 'to_host']


def cpu_heap_sdfg() -> dace.SDFG:
    """One map writing a non-transient array, with every descriptor on ``CPU_Heap``.

    That is the storage ``auto_optimize(DeviceType.CPU)`` leaves behind, and the shape the rename
    below was blind to.
    """
    sdfg = dace.SDFG('cpu_heap_offload')
    for name in ('A', 'B'):
        sdfg.add_array(name, [32], dace.float64, storage=dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state('compute')
    state.add_mapped_tasklet('double', {'i': '0:32'}, {'a': dace.Memlet('A[i]')},
                             'b = a * 2.0', {'b': dace.Memlet('B[i]')},
                             external_edges=True)
    return sdfg


def test_a_device_access_to_a_cpu_heap_array_is_renamed_to_the_device_copy():
    """A host array the offloading puts on the device must be RENAMED at every device access.

    The rename asked whether the descriptor's storage was ``Default``, which is one host storage of
    several: after ``auto_optimize`` for the CPU every array carries ``CPU_Heap`` instead, so the
    test was False and the kernel kept writing the HOST array, while the copy-back overwrote it with
    a device buffer nothing had written. vadv came back exactly as it went in -- a wrong answer with
    a valid graph, which is why this is asserted on the graph and not only through a result.
    """
    sdfg = cpu_heap_sdfg()
    sdfg.apply_gpu_transformations(simplify=False)
    sdfg.validate()

    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.data_nodes():
            entry = scopes.get(node)
            if entry is None or entry.map.schedule not in dtypes.GPU_SCHEDULES:
                continue
            assert sdfg.arrays[node.data].storage in GPU_RESIDENT_STORAGES, (
                f'{node.data!r} is accessed inside a {entry.map.schedule} map but lives in '
                f'{sdfg.arrays[node.data].storage}')

    kernel_writes = {
        edge.dst.data
        for state in sdfg.states()
        for edge in state.edges()
        if isinstance(edge.src, dace.nodes.MapExit) and isinstance(edge.dst, dace.nodes.AccessNode)
        and state.entry_node(edge.src).map.schedule in dtypes.GPU_SCHEDULES
    }
    assert kernel_writes, 'no kernel write to check'
    for name in kernel_writes:
        assert sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES, (
            f'a GPU map writes {name!r}, which lives in {sdfg.arrays[name].storage}: the write never '
            'reaches the device buffer that is copied back')


#: Blocks in a chain, comfortably past CPython's 1000-frame default. CloudSC canonicalized for the
#: GPU carries about twice this, which is the graph that produced the failure below.
LONG_CHAIN = 5000


def ir_chain(length: int):
    """``(open, [states])`` for one section holding ``length`` states in a row.

    The shape the IR pass builds for straight-line code: ``open -> s0 -> ... -> sN-1 -> close``.
    """
    sdfg = dace.SDFG('ir_chain')
    region = sdfg.add_state('section')
    open_node = OffloadingIRNode.new_open_node(region)
    states = [
        OffloadingIRNode.new_state_node(sdfg.add_state(f's{i}'), OrderedSet(), OrderedSet()) for i in range(length)
    ]
    previous = open_node
    for state in states:
        previous.append_node(state)
        previous = state
    previous.append_node(open_node.close)
    return open_node, states


def test_the_ir_walk_does_not_recurse_once_per_block():
    """The IR holds one node per state and per interstate edge, so its chain is as long as the
    program has blocks. Walking it by RECURSION spends one Python frame per block and overran the
    interpreter stack on the first application-sized kernel: CloudSC canonicalized for the GPU died
    with ``RecursionError: maximum recursion depth exceeded`` inside ``apply_gpu_transformations``,
    which stopped its whole GPU canonicalization (and with it every CPF device render of it).

    A chain is also the case where the answer is obvious, so this asserts the RESULT as well as the
    absence of the crash: the one tail of a straight line is its last state."""
    open_node, states = ir_chain(LONG_CHAIN)
    assert open_node.get_all_tails() == [states[-1]]


def test_traverse_ir_does_not_recurse_once_per_block():
    """:func:`~dace.transformation.passes.offloading.offloading_helpers.traverse_IR` walks the same
    chain and had the same stack cost. It visits every node exactly once, in the order the
    recursion did -- which is what the collected labels check."""
    open_node, states = ir_chain(LONG_CHAIN)
    seen = []
    traverse_IR(open_node, seen.append)
    assert seen == [open_node] + states + [open_node.close]


def test_a_tail_contributes_none_of_its_remaining_siblings():
    """The recursive walk stopped at the FIRST child that was the close node and skipped the rest,
    which is what makes a branching section report one tail per arm rather than per edge. The
    iterative walk has to keep that, so a node with a close child and a further child contributes
    itself and nothing below that child."""
    sdfg = dace.SDFG('ir_branch')
    open_node = OffloadingIRNode.new_open_node(sdfg.add_state('section'))
    head = OffloadingIRNode.new_state_node(sdfg.add_state('head'), OrderedSet(), OrderedSet())
    skipped = OffloadingIRNode.new_state_node(sdfg.add_state('skipped'), OrderedSet(), OrderedSet())
    open_node.append_node(head)
    head.append_node(open_node.close)
    head.append_node(skipped)
    skipped.append_node(open_node.close)
    assert open_node.get_all_tails() == [head]


@pytest.mark.gpu
def test_a_cpu_heap_array_survives_the_round_trip():
    """The same program, run: what the kernel computed has to come back to the caller."""
    sdfg = cpu_heap_sdfg()
    sdfg.apply_gpu_transformations()

    a = np.arange(32, dtype=np.float64)
    b = np.zeros(32, dtype=np.float64)
    sdfg(A=a, B=b)

    assert np.allclose(b, a * 2.0), 'the device result never reached the host array'


def ordering_edge_after_a_kernel() -> dace.SDFG:
    """A kernel writes ``A``; an empty memlet orders the read of scalar ``s`` by a second kernel after it.

    CloudSC's ``zpsupsatsrce`` orders some thirty reads this way, ``ptsphy`` among them.
    """
    sdfg = dace.SDFG('ordering_edge_after_a_kernel')
    sdfg.add_array('A', [LENGTH], dace.float64)
    sdfg.add_array('B', [LENGTH], dace.float64)
    sdfg.add_scalar('s', dace.float64)
    state = sdfg.add_state('ordered')
    _, _, exit_a = state.add_mapped_tasklet('write_a', {'i': f'0:{LENGTH}'}, {},
                                            'o = 1.0', {'o': dace.Memlet('A[i]')},
                                            external_edges=True)
    written = state.out_edges(exit_a)[0].dst
    _, entry_b, _ = state.add_mapped_tasklet('scale_b', {'i': f'0:{LENGTH}'}, {'inp': dace.Memlet('s[0]')},
                                             'o = inp * 2.0', {'o': dace.Memlet('B[i]')},
                                             external_edges=True)
    state.add_nedge(written, state.in_edges(entry_b)[0].src, dace.Memlet())
    sdfg.validate()
    return sdfg


def test_an_ordering_edge_does_not_make_a_scalar_device_written():
    """A scalar no kernel writes stays a by-value scalar: a device copy of it would be copied in mid-run."""
    sdfg = ordering_edge_after_a_kernel()
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert isinstance(sdfg.arrays['s'], data.Scalar), sdfg.arrays['s']
    assert not [name for name in sdfg.arrays if name != 's' and name.endswith('_s')], list(sdfg.arrays)


def test_the_pass_reports_what_it_placed_on_the_device():
    """A Pipeline reads the result as "did anything change", so an offload must not answer None."""
    sdfg = cpu_heap_sdfg()
    placed = OffloadToAccelerator().apply_pass(sdfg, {})
    assert placed is not None and {name.split('.', 1)[1] for name in placed} == {'A_gpu', 'B_gpu'}, placed


def test_a_graph_with_nothing_to_offload_reports_no_change():
    sdfg = dace.SDFG('host_only')
    sdfg.add_scalar('s', dace.float64)
    state = sdfg.add_state('host')
    one = state.add_tasklet('one', {}, {'o'}, 'o = 1.0')
    state.add_edge(one, 'o', state.add_write('s'), None, dace.Memlet('s[0]'))
    assert OffloadToAccelerator().apply_pass(sdfg, {}) is None


def one_element_read_by_two_maps() -> dace.SDFG:
    """``s = A[3]`` on the host, then two maps that both read ``s``."""
    sdfg = dace.SDFG('one_element_read_by_two_maps')
    for name in 'ABC':
        sdfg.add_array(name, [LENGTH], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    state = sdfg.add_state('main')
    source = state.add_read('A')
    element = state.add_access('s')
    state.add_edge(source, None, element, None, dace.Memlet('A[3] -> [0]'))
    for name in 'BC':
        entry, exit_node = state.add_map(f'scale_{name}', {'i': f'0:{LENGTH}'})
        tasklet = state.add_tasklet(f'times_{name}', {'x': None, 'y': None}, {'o': None}, 'o = x * y')
        state.add_memlet_path(element, entry, tasklet, dst_conn='x', memlet=dace.Memlet('s[0]'))
        state.add_memlet_path(source, entry, tasklet, dst_conn='y', memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(tasklet, exit_node, state.add_write(name), src_conn='o', memlet=dace.Memlet(f'{name}[i]'))
    sdfg.validate()
    return sdfg


def test_a_single_element_two_maps_read_stays_outside_both():
    """Moving the copy into the first map left the second reading a node inside the first one's scope,
    and the scope tree put ``times_B`` under ``scale_C``."""
    sdfg = one_element_read_by_two_maps()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    state = next(state for state in sdfg.states() if state.label == 'main')
    scopes = state.scope_dict()
    tasklets = {node.label: scopes[node].map.label for node in state.nodes() if isinstance(node, dace.nodes.Tasklet)}
    assert tasklets == {'times_B': 'scale_B', 'times_C': 'scale_C'}, tasklets
    assert all(scopes[node] is None for node in state.data_nodes() if node.data == 's')


def per_iteration_scratch() -> dace.SDFG:
    """``tmp`` is written and read inside the scope of one map."""
    sdfg = dace.SDFG('per_iteration_scratch')
    sdfg.add_array('A', [LENGTH], dace.float64)
    sdfg.add_array('B', [LENGTH], dace.float64)
    sdfg.add_transient('tmp', [2], dace.float64)
    state = sdfg.add_state('kernel')
    entry, exit_node = state.add_map('twice', {'i': f'0:{LENGTH}'})
    fill = state.add_tasklet('fill', {'a': None}, {'t': None}, 't = a')
    scratch = state.add_access('tmp')
    use = state.add_tasklet('use', {'t': None}, {'b': None}, 'b = 2 * t')
    state.add_memlet_path(state.add_read('A'), entry, fill, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_edge(fill, 't', scratch, None, dace.Memlet('tmp[0]'))
    state.add_edge(scratch, None, use, 't', dace.Memlet('tmp[0]'))
    state.add_memlet_path(use, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def test_a_transient_one_kernel_uses_is_a_register():
    """GPUTransformSDFG made such a transient a register; left in GPU_Global a block-wide reduce refuses it."""
    sdfg = per_iteration_scratch()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    assert sdfg.arrays['tmp'].storage == dtypes.StorageType.Register


ROWS = dace.symbol('ROWS')
COLS = dace.symbol('COLS')


@dace.program
def rows_with_a_small_gather(agg: dace.int64[ROWS], w: dace.float64[ROWS], out: dace.float64[4]):
    """amg_setup's shape: a host loop over every row updates ``acc`` on the host, and a map over a few
    entries reads it every row."""
    acc = np.zeros([ROWS], dtype=np.float64)
    for i in range(ROWS):
        c = agg[i]
        if w[i] >= 0:
            acc[c] = acc[c] + w[i]
        for j in dace.map[0:4]:
            out[j] = out[j] + acc[j]


@dace.program
def rows_with_a_full_stencil(agg: dace.int64[ROWS], w: dace.float64[ROWS], acc: dace.float64[ROWS]):
    """The same loop around a map over ALL of ``acc``: the map moves what the loop would copy."""
    for i in range(ROWS):
        c = agg[i]
        if w[i] >= 0:
            acc[c] = acc[c] + w[i]
        for j in dace.map[0:ROWS]:
            acc[j] = acc[j] * 0.5


@dace.program
def rows_with_an_unranked_gather(agg: dace.int64[ROWS], w: dace.float64[ROWS], out: dace.float64[COLS]):
    """The small gather over another symbol's extent, which no rule may rank against ``ROWS``."""
    acc = np.zeros([ROWS], dtype=np.float64)
    for i in range(ROWS):
        c = agg[i]
        if w[i] >= 0:
            acc[c] = acc[c] + w[i]
        for j in dace.map[0:COLS]:
            out[j] = out[j] + acc[j]


def is_copy_state(sdfg: dace.SDFG, block: dace.sdfg.state.ControlFlowBlock) -> bool:
    """A state of access nodes only, with an edge joining a ``GPU_Global`` container to a host one."""
    if not isinstance(block, dace.SDFGState) or not block.nodes():
        return False
    if not all(isinstance(n, dace.nodes.AccessNode) for n in block.nodes()):
        return False
    gpu = dtypes.StorageType.GPU_Global
    return any(
        (sdfg.arrays[e.src.data].storage is gpu) != (sdfg.arrays[e.dst.data].storage is gpu) for e in block.edges())


def copies_inside_loops(sdfg: dace.SDFG) -> list[str]:
    return [
        b.label for loop in sdfg.all_control_flow_blocks(recursive=True) if isinstance(loop, LoopRegion)
        for b in loop.all_control_flow_blocks(recursive=True) if is_copy_state(sdfg, b)
    ]


def offloaded_program(program: dace.frontend.python.parser.DaceProgram, pin: bool = True) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False, pin_host_loop_maps=pin)
    sdfg.validate()
    return sdfg


def test_a_small_map_in_a_host_loop_stays_on_the_host():
    """Offloaded, the gather copied the whole ``acc`` host<->device on every row: O(rows^2) traffic,
    amg_setup at 344 s per call against 0.8 s on the CPU."""
    sdfg = offloaded_program(rows_with_a_small_gather)
    assert not copies_inside_loops(sdfg), copies_inside_loops(sdfg)
    gather = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.range.num_elements() == 4
    ]
    assert gather and all(n.map.schedule == dtypes.ScheduleType.Sequential for n in gather)


def test_without_pinning_a_small_map_in_a_host_loop_is_a_kernel():
    """Pinning is opt-in: by default every top-level map is a kernel."""
    sdfg = offloaded_program(rows_with_a_small_gather, pin=False)
    gather = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.range.num_elements() == 4
    ]
    assert gather and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in gather)


def test_a_map_over_the_whole_shared_array_stays_a_kernel():
    sdfg = offloaded_program(rows_with_a_full_stencil)
    full = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert full and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in full)


def test_a_map_whose_size_cannot_be_ranked_stays_a_kernel():
    sdfg = offloaded_program(rows_with_an_unranked_gather)
    gather = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and 'COLS' in map(str, n.map.range.free_symbols)
    ]
    assert gather and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in gather)


@pytest.mark.gpu
def test_a_small_map_kept_on_the_host_computes_what_numpy_computes():
    sdfg = offloaded_program(rows_with_a_small_gather)
    agg = np.array([0, 2, 1, 3, 0, 5, 7, 2], dtype=np.int64)
    w = np.arange(8, dtype=np.float64)
    out = np.zeros(4)
    sdfg(agg=agg, w=w, out=out, ROWS=8)
    acc, want = np.zeros(8), np.zeros(4)
    for i in range(8):
        acc[agg[i]] += w[i]
        want += acc[:4]
    np.testing.assert_allclose(out, want)


def kernel_with_a_one_iteration_inner_map() -> dace.SDFG:
    """A map whose body is a single-iteration map writing and reading the length-1 local ``acc``."""
    sdfg = dace.SDFG('kernel_with_a_one_iteration_inner_map')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('B', [256], dace.float64)
    sdfg.add_array('acc', [1], dace.float64, transient=True)
    state = sdfg.add_state('kernel')
    outer_entry, outer_exit = state.add_map('device', {'i': '0:256'})
    inner_entry, inner_exit = state.add_map('once', {'k': '0:1'})
    double = state.add_tasklet('double', {'inp': None}, {'out': None}, 'out = inp * 2.0')
    acc = state.add_access('acc')
    bump = state.add_tasklet('bump', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    state.add_memlet_path(state.add_read('A'),
                          outer_entry,
                          inner_entry,
                          double,
                          dst_conn='inp',
                          memlet=dace.Memlet('A[i]'))
    state.add_edge(double, 'out', acc, None, dace.Memlet('acc[0]'))
    state.add_edge(acc, None, bump, 'inp', dace.Memlet('acc[0]'))
    state.add_memlet_path(bump,
                          inner_exit,
                          outer_exit,
                          state.add_write('B'),
                          src_conn='out',
                          memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def test_a_length_one_local_of_a_one_iteration_map_in_a_kernel_is_a_register():
    """Extended scalarized this after dropping the trivial map; here it only has to live in the kernel."""
    sdfg = kernel_with_a_one_iteration_inner_map()
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    assert sdfg.arrays['acc'].storage == dtypes.StorageType.Register
    state = next(state for state in sdfg.states() if state.label == "kernel")
    kernels = [
        n.map.label for n in state.nodes()
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
    ]
    assert kernels == ['device'], kernels


@pytest.mark.gpu
def test_a_length_one_local_of_a_one_iteration_map_computes_what_numpy_computes():
    sdfg = kernel_with_a_one_iteration_inner_map()
    OffloadToAccelerator().apply_pass(sdfg, {})
    A = np.random.default_rng(3).random(256)
    B = np.zeros(256)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B, A * 2.0 + 1.0)


def host_accumulator_beside_a_kernel_in_a_loop(length: int) -> dace.SDFG:
    """npbench nbody's shape: a kernel on ``A`` and a host tasklet bumping ``PE``, every iteration of a loop,
    with both signature arrays handed over in device memory."""
    sdfg = dace.SDFG(f'host_accumulator_beside_a_kernel_in_a_loop_{length}')
    sdfg.add_array('A', [64], dace.float64)
    sdfg.add_array('PE', [length], dace.float64)
    loop = LoopRegion('steps', 't < 10', 't', 't = 0', 't = t + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    body.add_mapped_tasklet('scale', {'i': '0:64'}, {'a': dace.Memlet('A[i]')},
                            'b = a * 0.5', {'b': dace.Memlet('A[i]')},
                            external_edges=True)
    bump = body.add_tasklet('bump', {'p': None}, {'q': None}, 'q = p + 1.0')
    body.add_edge(body.add_read('PE'), None, bump, 'p', dace.Memlet('PE[0]'))
    body.add_edge(bump, 'q', body.add_write('PE'), None, dace.Memlet('PE[0]'))
    sdfg.validate()
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


@pytest.mark.parametrize('length', [1, 4])
def test_a_host_accumulator_is_copied_once_each_way_and_not_wrapped(length):
    """A length-1 ``PE`` becomes a staged scalar, which inherited ``GPU_Global`` and was then read on the host."""
    sdfg = host_accumulator_beside_a_kernel_in_a_loop(length)
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    body = next(state for state in sdfg.all_states() if state.label == 'body')
    scopes = body.scope_dict()
    bump = next(node for node in body.nodes() if isinstance(node, dace.nodes.Tasklet) and node.label == 'bump')
    assert scopes[bump] is None, 'the host accumulation was wrapped into a kernel'
    assert not copies_inside_loops(sdfg), copies_inside_loops(sdfg)


@pytest.mark.gpu
@pytest.mark.parametrize('length', [1, 4])
def test_a_host_accumulator_beside_a_kernel_computes_what_numpy_computes(length):
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = host_accumulator_beside_a_kernel_in_a_loop(length)
    OffloadToAccelerator().apply_pass(sdfg, {})
    A = cupy.asarray(np.arange(64, dtype=np.float64))
    PE = cupy.zeros(length)
    sdfg(A=A, PE=PE)
    np.testing.assert_allclose(A.get(), np.arange(64) * 0.5**10)
    assert PE.get()[0] == 10.0
