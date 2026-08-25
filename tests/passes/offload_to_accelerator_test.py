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
        assert node.implementation in ('CUDA', 'CUDA_strided', 'GPUAuto'), (
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
