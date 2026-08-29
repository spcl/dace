# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``OffloadToAccelerator`` places a copy where the location CHANGES, not around every kernel.

The shape that separates it from ``GPUTransformSDFG`` is the guarded one canonicalize emits for a
loop it can only parallelize under a runtime condition::

    if <cond>:  <Map>          # parallel arm
    else:       <LoopRegion>   # sequential fallback

With the inputs already resident on the device, the parallel arm needs no copy at all, and the
sequential arm needs one each way. Those copies belong INSIDE that arm: hoisting them to the
enclosing region makes every execution pay for a host round-trip that only the fallback needs, and
that is precisely the cost the old transformation could not avoid.

TSVC ``s171`` is the corpus kernel that canonicalizes to this shape (``a[i * inc] += b[i]``, whose
parallelism turns on ``inc != 0``). Nothing here compiles or runs -- the pass rewrites the SDFG, so
these assertions need no GPU.
"""
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.transformation.passes.offloading import OffloadToAccelerator
from tests.corpus.tsvc import tsvc

GUARDED_KERNEL = 's171_d_single'


def canonicalized_with_gpu_inputs(name: str) -> dace.SDFG:
    """The kernel after ``canonicalize``, with every signature array pinned to the device.

    Pinning is what makes the assertions below about copy PLACEMENT rather than copy count: with
    the inputs already on the GPU, the only copies left in the graph are the ones some branch
    genuinely asked for.
    """
    kernel = next(k for k in tsvc.collect() if k.name == name)
    sdfg = tsvc.to_sdfg(kernel, name, simplify=True)
    canonicalize(sdfg, validate=False, validate_all=False, peel_limit=4, break_anti_dependence=True)
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dtypes.StorageType.GPU_Global
    return sdfg


def is_copy_state(sdfg: dace.SDFG, block) -> bool:
    """A state the pass emitted to move data between host and device.

    Read off the dataflow rather than the label: every node is an access node, and some edge joins
    a ``GPU_Global`` descriptor to one that is not.
    """
    if not isinstance(block, SDFGState):
        return False
    nodes = block.nodes()
    if not nodes or not all(isinstance(n, dace.nodes.AccessNode) for n in nodes):
        return False
    gpu = dtypes.StorageType.GPU_Global
    return any(
        (sdfg.arrays[e.src.data].storage is gpu) != (sdfg.arrays[e.dst.data].storage is gpu) for e in block.edges())


def guard_block(sdfg: dace.SDFG) -> ConditionalBlock:
    """The one conditional carrying a Map arm and a LoopRegion arm."""
    for block in sdfg.all_control_flow_blocks():
        if not isinstance(block, ConditionalBlock):
            continue
        has_map, has_loop = False, False
        for _cond, region in block.branches:
            if any(isinstance(n, dace.nodes.MapEntry) for n, _ in region.all_nodes_recursive()):
                has_map = True
            if any(isinstance(b, LoopRegion) for b in region.all_control_flow_blocks()):
                has_loop = True
        if has_map and has_loop:
            return block
    raise AssertionError('canonicalize did not produce the guarded parallel/sequential pair')


def test_the_guarded_kernel_still_canonicalizes_to_a_parallel_and_a_sequential_arm():
    """Guard on the fixture itself: everything below is vacuous if the shape stops appearing, and a
    silently-vacuous placement test is worse than no test."""
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    branches = guard_block(sdfg).branches
    assert len(branches) == 2, f'expected a parallel arm and a fallback; got {len(branches)} branches'


def test_the_fallback_arm_owns_its_copies():
    """The sequential arm round-trips through the host; the parallel arm and the enclosing region
    stay clear of it."""
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()

    guard = guard_block(sdfg)
    inside = {id(b) for _c, region in guard.branches for b in region.all_control_flow_blocks()}
    outside = [b for b in sdfg.all_control_flow_blocks() if id(b) not in inside and b is not guard]
    hoisted = [b.label for b in outside if is_copy_state(sdfg, b)]
    assert not hoisted, (f'copies hoisted out of the guard: {hoisted}. Every execution then pays a '
                         'host round-trip that only the sequential fallback needs.')

    parallel, sequential = None, None
    for _cond, region in guard.branches:
        if any(isinstance(b, LoopRegion) for b in region.all_control_flow_blocks()):
            sequential = region
        else:
            parallel = region
    assert parallel is not None and sequential is not None

    assert not [b.label for b in parallel.all_control_flow_blocks() if is_copy_state(sdfg, b)
                ], ('the parallel arm reads its inputs where they already are, so it needs no copy')
    assert [b.label for b in sequential.all_control_flow_blocks()
            if is_copy_state(sdfg, b)], ('the sequential arm runs on the host and must copy its inputs down')


def test_the_fallback_copies_in_before_it_runs_and_out_after():
    """Placement is not enough -- the copy has to be ORDERED against the loop. A copy-in scheduled
    after the loop feeds it whatever the device buffer held, and a copy-out scheduled before it
    publishes a stale result."""
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    OffloadToAccelerator().apply_pass(sdfg, {})

    sequential = next(region for _c, region in guard_block(sdfg).branches if any(
        isinstance(b, LoopRegion) for b in region.all_control_flow_blocks()))
    order = list(sequential.bfs_nodes(sequential.start_block))
    loop = next(b for b in order if isinstance(b, LoopRegion))
    copies = [b for b in order if is_copy_state(sdfg, b)]
    assert copies, 'no copy in the sequential arm to order against the loop'

    at = order.index(loop)
    assert order.index(copies[0]) < at, (f'the first copy {copies[0].label!r} does not run before the '
                                         'sequential loop, so the loop reads the device buffer')
    assert order.index(copies[-1]) > at, (f'the last copy {copies[-1].label!r} does not run after the '
                                          'sequential loop, so its result never reaches the device')


@pytest.mark.parametrize('use_new_pass', [True, False])
def test_the_config_knob_selects_the_offloader(use_new_pass):
    """``apply_gpu_transformations`` routes on ``optimizer.new_gpu_offloading_pass``. The two
    offloaders differ in whether a host copy is ever needed, which is what tells them apart: the
    new pass starts from device-resident inputs and stages one, the old one starts on the host."""
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=use_new_pass):
        sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    staged = [name for name in sdfg.arrays if name.endswith('_host')]
    if use_new_pass:
        assert staged, 'the new pass stages a host copy under a `_host` name'
    else:
        assert not staged, 'the old transformation never needs a host copy: it starts on the host'


@pytest.mark.parametrize('use_new_pass', [True, False])
def test_simplify_is_honoured_whichever_offloader_ran(use_new_pass):
    """``simplify`` is ``apply_gpu_transformations``'s contract, not one offloader's.

    The old arm forwards it into ``GPUTransformSDFG``'s options and the new arm has no options to
    forward it to, so the knob used to mean "simplify" on one path and nothing on the other. Both
    offloaders insert copy states, so a caller that asked for a simplified graph and got the new
    pass was handed the unfused ones -- a difference in graph shape decided by a config knob the
    caller never set, which is exactly what the argument exists to make explicit.
    """
    plain = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    simplified = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=use_new_pass):
        plain.apply_gpu_transformations(validate=False, simplify=False)
        simplified.apply_gpu_transformations(validate=False, simplify=True)
    plain.validate()
    simplified.validate()

    def size(sdfg):
        """States, dataflow nodes, dataflow edges. Which one shrinks is not the same on the two
        arms -- the old one fuses states here and leaves the nodes alone, the new one prunes nodes
        and leaves the states alone -- so asserting on any single count would pin one arm's
        incidental shape rather than the contract both share."""
        return (sum(1 for _ in sdfg.all_states()), sum(len(s.nodes()) for s in sdfg.all_states()),
                sum(len(s.edges()) for s in sdfg.all_states()))

    before, after = size(plain), size(simplified)
    assert all(a <= b for a, b in zip(after, before)), (f'simplify=True grew the graph: {before} -> {after}')
    assert after != before, (f'simplify=True left the graph at {before} (states, nodes, edges), identical to '
                             'simplify=False, so the argument did nothing')


def test_an_offloaded_scan_gets_its_device_lowering():
    """A Scan that ends up on the device must be LOWERED for the device.

    It names its device expansion ``CUDA``, which appears in none of ``find_fast_library``'s
    priority lists -- those name vendor BLAS -- so without a rule for it the node falls through to
    ``pure`` and the kernel carries a serial sweep where the CUB one belongs. Selection lives in
    ``set_fast_implementations``, after offloading, because that is the first point at which every
    descriptor's storage is final. Nothing is compiled here: the assertion is on the lowering the
    node is pointed at, which is what codegen would emit.
    """
    sdfg = canonicalized_with_gpu_inputs('s126_d_single')
    scans = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == 'Scan']
    assert scans, 's126 no longer canonicalizes to a Scan; the check below would be vacuous'

    OffloadToAccelerator().apply_pass(sdfg, {})
    set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    sdfg.validate()

    offloaded = [
        n for n, _ in sdfg.all_nodes_recursive()
        if type(n).__name__ == 'Scan' and n.schedule == dtypes.ScheduleType.GPU_Device
    ]
    assert offloaded, 'the Scan was not offloaded, so its lowering says nothing'
    for node in offloaded:
        assert node.implementation in ('CUDA', 'GPUAuto'), (
            f'the Scan runs on the device with implementation {node.implementation!r}; that is a '
            'host lowering, so the kernel would carry a serial sweep')


def test_a_host_pinned_library_node_is_left_to_the_old_transformation():
    """``ScatterConflictCheck`` keeps its flag and its tag scratch on the HOST in every expansion,
    the CUDA one included -- it tags on the device out of the CUB scratch pool and only sizes that
    buffer from the scratch array. The new pass gives a whole state ONE location, so it would move
    both with the rest and the node's own validation rejects the result. Declining is what keeps
    that from becoming a mid-rewrite crash."""
    sdfg = canonicalized_with_gpu_inputs('s4113_d_single')
    checks = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == 'ScatterConflictCheck']
    assert checks, 's4113 no longer canonicalizes to a ScatterConflictCheck'
    assert all(n.host_connectors for n in checks), 'the node must declare which connectors stay on the host'

    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    assert not [name for name in sdfg.arrays if name.endswith('_host')
                ], ('a host-pinned library node must route to the old transformation')


def test_a_find_first_answer_stays_on_the_host():
    """``FindFirst``'s answer is a HOST scalar in every expansion, the device one included.

    The device search leaves its result in CUB scratch; ``find_first_index_device`` copies it back
    and writes ``*out`` on the host, and the CUDA expansion's tasklet assigns the out-connector from
    a host stack variable. Promoting that scalar to device memory has host code write a device
    pointer -- which VALIDATES, and then corrupts, which is why this is asserted on storage rather
    than left to the SDFG checker.
    """
    sdfg = canonicalized_with_gpu_inputs('s481_d_single')
    finds = [(n, st) for n, st in sdfg.all_nodes_recursive() if type(n).__name__ == 'FindFirst']
    assert finds, 's481 no longer canonicalizes to a FindFirst; the check below would be vacuous'
    assert all(n.host_connectors for n, _ in finds), 'FindFirst must declare its answer as host-only'

    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    for node, state in [(n, st) for n, st in sdfg.all_nodes_recursive() if type(n).__name__ == 'FindFirst']:
        for edge in state.out_edges(node):
            if edge.src_conn not in node.host_connectors:
                continue
            storage = sdfg.arrays[edge.data.data].storage
            assert storage != dtypes.StorageType.GPU_Global, (
                f'{edge.data.data} carries the search answer but lives in {storage}; the expansion '
                'writes it from host code')


def scan_with_a_scalar_seed() -> dace.SDFG:
    """A device ``Scan`` whose seed is a ``Scalar`` -- the single-element input the host-preferred
    rule is about."""
    sdfg = dace.SDFG('scan_with_a_scalar_seed')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('B', [256], dace.float64)
    sdfg.add_scalar('seed', dace.float64)
    state = sdfg.add_state()
    node = Scan('scan', op=ScanOp.SUM)
    node.add_in_connector('_scan_in')
    node.add_in_connector('_scan_init')
    node.add_out_connector('_scan_out')
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_scan_in', dace.Memlet('A[0:256]'))
    state.add_edge(state.add_read('seed'), None, node, '_scan_init', dace.Memlet('seed[0]'))
    state.add_edge(node, '_scan_out', state.add_write('B'), None, dace.Memlet('B[0:256]'))
    sdfg.validate()
    return sdfg


def test_a_scalar_operand_never_enters_the_placement_sets():
    """The pass places ARRAYS; it asserts that no scalar ever reaches its cpu/gpu sets.

    The host-preferred rule -- single-element inputs of a device library node are cheaper left on
    the host -- must respect that: a ``Scalar`` operand needs no entry, because nothing places it.
    Naming one trips the pass's own invariant instead (tsvc_2_5 ext_break_capture's ``__ff_KFIND``).
    """
    sdfg = scan_with_a_scalar_seed()
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=True):
        sdfg.apply_gpu_transformations(validate=False, simplify=False)  # the assertion fires inside
    sdfg.validate()


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
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=True):
        sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    host_state = next(state for state in sdfg.states() if state.label == 'host_work')
    written = {edge.data.data for edge in host_state.edges() if not edge.data.is_empty()}
    assert not [name for name in written if sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global
                ], (f'host tasklet left holding a device container: {written}')


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
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=True):
        sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    state = next(s for s in sdfg.states() if s.label == 'mixed')
    scopes = state.scope_dict()
    wrapped = next(n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet) and n.label == 'seed')
    assert scopes[wrapped] is not None, 'a tasklet that reads nothing was left outside its wrapper'


def kernel_with_a_one_iteration_inner_map() -> dace.SDFG:
    """A ``GPU_Device`` map whose body is a single-iteration ``Sequential`` map over a length-1 local.

    The shape the offload pass leaves behind: the inner map is a loop that runs once, and ``acc``
    is the stack slot its body writes through. ``acc`` is a length-1 ARRAY rather than a scalar
    because a kernel receives a scalar by value and would lose the write -- which stops being true
    once the map around it is gone.
    """
    sdfg = dace.SDFG('kernel_with_a_one_iteration_inner_map')
    sdfg.add_array('A', [256], dace.float64)
    sdfg.add_array('B', [256], dace.float64)
    sdfg.add_array('acc', [1], dace.float64, transient=True)
    state = sdfg.add_state('kernel')

    outer_entry, outer_exit = state.add_map('device', {'i': '0:256'}, schedule=dtypes.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map('once', {'k': '0:1'}, schedule=dtypes.ScheduleType.Sequential)

    read = state.add_read('A')
    double = state.add_tasklet('double', {'inp': None}, {'out': None}, 'out = inp * 2.0')
    acc = state.add_access('acc')
    bump = state.add_tasklet('bump', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    write = state.add_write('B')

    state.add_memlet_path(read, outer_entry, inner_entry, double, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_edge(double, 'out', acc, None, dace.Memlet('acc[0]'))
    state.add_edge(acc, None, bump, 'inp', dace.Memlet('acc[0]'))
    state.add_memlet_path(bump, inner_exit, outer_exit, write, src_conn='out', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def eliminate_and_scalarize(sdfg: dace.SDFG) -> None:
    """The pass's post-offload cleanup, on a graph already in post-offload shape."""
    offloader = OffloadToAccelerator()
    offloader.cache_scopes(sdfg)
    offloader.scalarize_locals_of_removed_trivial_maps(sdfg)


def test_a_one_iteration_map_inside_a_kernel_leaves_a_scalar_behind():
    """Removing the map is what licenses the scalar, and the kernel itself is not removable.

    ``TrivialMapElimination`` declines a GPU schedule, so the kernel survives its own trivial-looking
    body being dropped; ``acc`` is then an ordinary local of that kernel and needs no array.
    """
    sdfg = kernel_with_a_one_iteration_inner_map()
    eliminate_and_scalarize(sdfg)
    sdfg.validate()

    state = sdfg.states()[0]
    maps = [n.map for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
    assert [m.label for m in maps] == ['device'], f'expected only the kernel to survive, got {maps}'
    assert maps[0].schedule == dtypes.ScheduleType.GPU_Device
    assert isinstance(sdfg.arrays['acc'], dace.data.Scalar), f'acc stayed {type(sdfg.arrays["acc"]).__name__}'


def test_a_one_iteration_map_outside_a_kernel_keeps_its_array():
    """The same map with no kernel around it: dropped, but its array is NOT device-local.

    Only a kernel's schedule makes the write a register. Outside one the length-1 array may be
    something a caller or another state reads through, so the elimination is allowed and the
    conversion is not.
    """
    sdfg = kernel_with_a_one_iteration_inner_map()
    outer = next(n for n in sdfg.states()[0].nodes()
                 if isinstance(n, dace.sdfg.nodes.MapEntry) and n.map.label == 'device')
    outer.map.schedule = dtypes.ScheduleType.CPU_Multicore
    eliminate_and_scalarize(sdfg)
    sdfg.validate()
    assert isinstance(sdfg.arrays['acc'], dace.data.Array), 'a host-level local was scalarized'


def test_a_trivial_kernel_map_is_never_eliminated():
    """The size-1 wrapper the pass builds for a hybrid state IS the kernel."""
    sdfg = kernel_with_a_one_iteration_inner_map()
    inner = next(n for n in sdfg.states()[0].nodes()
                 if isinstance(n, dace.sdfg.nodes.MapEntry) and n.map.label == 'once')
    inner.map.schedule = dtypes.ScheduleType.GPU_Device
    eliminate_and_scalarize(sdfg)
    sdfg.validate()
    labels = {n.map.label for n in sdfg.states()[0].nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)}
    assert labels == {'device', 'once'}, f'a GPU-scheduled map was eliminated: {labels}'
    assert isinstance(sdfg.arrays['acc'], dace.data.Array), 'a kernel output was scalarized'


def test_a_pinned_host_map_falls_back_to_the_old_transformation():
    """``host_maps`` / ``host_data`` name what must STAY on the host. The new pass derives that
    rather than accepting it, so a caller that pins anything gets the old transformation even with
    the knob on -- silently ignoring the pin would move data the caller said not to move."""
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=True):
        sdfg.apply_gpu_transformations(validate=False, simplify=False, host_data=['a'])
    sdfg.validate()
    assert not [name
                for name in sdfg.arrays if name.endswith('_host')], ('the pinned call must not go through the new pass')


if __name__ == '__main__':
    test_the_guarded_kernel_still_canonicalizes_to_a_parallel_and_a_sequential_arm()
    test_the_fallback_arm_owns_its_copies()
    test_the_fallback_copies_in_before_it_runs_and_out_after()
    test_the_config_knob_selects_the_offloader(True)
    test_the_config_knob_selects_the_offloader(False)
    test_an_offloaded_scan_gets_its_device_lowering()
    test_a_host_pinned_library_node_is_left_to_the_old_transformation()
    test_a_find_first_answer_stays_on_the_host()
    test_a_pinned_host_map_falls_back_to_the_old_transformation()
    test_an_interstate_read_does_not_hand_the_next_state_the_device_name()
    test_a_wrapped_region_puts_every_root_under_its_entry()
    test_a_one_iteration_map_inside_a_kernel_leaves_a_scalar_behind()
    test_a_one_iteration_map_outside_a_kernel_keeps_its_array()
    test_a_trivial_kernel_map_is_never_eliminated()


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
    with dace.config.set_temporary('optimizer', 'new_gpu_offloading_pass', value=True):
        sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    written = [name for name, desc in sdfg.arrays.items() if name.startswith('acc') and desc.transient]
    assert written, 'the scalar vanished, so this asserts nothing'
    resident = [name for name in written if sdfg.arrays[name].storage in GPU_RESIDENT_STORAGES]
    assert resident, (f'no device copy of the kernel-written scalar: '
                      f'{[(n, sdfg.arrays[n].storage.name) for n in written]}')

    signatures = [
        line for obj in sdfg.generate_code() if obj.language == 'cu' for line in obj.clean_code.splitlines()
        if '__global__' in line
    ]
    assert signatures, 'nothing was emitted as a kernel, so the signature asserts nothing'
    by_value = [line for line in signatures for name in resident if f'double {name}' in line]
    assert not by_value, f'a kernel takes a scalar it writes by value: {by_value}'


def test_apply_gpu_storage_leaves_every_scalar_on_the_host():
    """A non-transient scalar stays host-resident even when device code writes it.

    Host code reads a scalar as a loop bound, a branch condition or from a tasklet outside any
    map, and a device-resident one makes every such read invalid. A kernel that writes one gets a
    GPU transient and a copy back from the offload pass instead.
    """
    from dace.transformation.auto import auto_optimize

    sdfg = dace.SDFG('scalar_stays_on_host')
    sdfg.add_scalar('written', dace.float64, transient=False)
    sdfg.add_scalar('read_only', dace.float64, transient=False)
    sdfg.add_array('A', [8], dace.float64, transient=False)
    st = sdfg.add_state('main')
    entry, exit_ = st.add_map('k', {'i': '0:8'}, schedule=dace.ScheduleType.GPU_Device)
    t = st.add_tasklet('w', {'r': None}, {'o': None, 'a': None}, 'o = r\na = r')
    st.add_memlet_path(st.add_read('read_only'), entry, t, dst_conn='r', memlet=dace.Memlet('read_only[0]'))
    st.add_memlet_path(t, exit_, st.add_write('written'), src_conn='o', memlet=dace.Memlet('written[0]'))
    st.add_memlet_path(t, exit_, st.add_write('A'), src_conn='a', memlet=dace.Memlet('A[i]'))

    auto_optimize.apply_gpu_storage(sdfg)

    assert sdfg.arrays['written'].storage is not dace.StorageType.GPU_Global
    assert sdfg.arrays['read_only'].storage is not dace.StorageType.GPU_Global
    assert sdfg.arrays['A'].storage is dace.StorageType.GPU_Global


def test_apply_gpu_storage_leaves_an_array_an_interstate_edge_reads_on_the_host():
    """The scalar rule again, for the case the scalar check cannot see.

    An interstate edge's reads live in its condition and assignments, not on any AccessNode, so a
    pass that walks nodes concludes the array is only ever touched by the maps it can see. Indexing
    one element does not make the container a Scalar, so ``A[0] < N`` on a loop condition took the
    array to the device and left the host reading device memory -- a graph validation refuses, well
    after the pass that shaped it. npbench azimint_hist reaches this through the bin edges
    ``np.histogram`` derives from its ``radius`` argument.
    """
    from dace.transformation.auto import auto_optimize

    sdfg = dace.SDFG('interstate_read_stays_on_host')
    sdfg.add_array('bounds', [2], dace.float64, transient=False)
    sdfg.add_array('data', [8], dace.float64, transient=False)
    body = sdfg.add_state('body')
    entry, exit_ = body.add_map('k', {'i': '0:8'}, schedule=dace.ScheduleType.GPU_Device)
    tasklet = body.add_tasklet('scale', {'d': None}, {'o': None}, 'o = d * 2.0')
    body.add_memlet_path(body.add_read('data'), entry, tasklet, dst_conn='d', memlet=dace.Memlet('data[i]'))
    body.add_memlet_path(tasklet, exit_, body.add_write('data'), src_conn='o', memlet=dace.Memlet('data[i]'))
    after = sdfg.add_state('after')
    # The read that no AccessNode carries.
    sdfg.add_edge(body, after, dace.InterstateEdge(condition='bounds[0] < bounds[1]'))

    auto_optimize.apply_gpu_storage(sdfg)

    assert sdfg.arrays['bounds'].storage is not dace.StorageType.GPU_Global
    assert sdfg.arrays['data'].storage is dace.StorageType.GPU_Global
    sdfg.validate()
