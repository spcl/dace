# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``OffloadToAccelerator`` places a copy where the location CHANGES, not around every kernel.

The shape that exercises it is the guarded one canonicalize emits for a
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
import numpy as np
import pytest

import dace
from dace.transformation import pass_pipeline as ppl
from dace import dtypes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.libraries.standard.nodes.find_first import FindFirst, INDEX_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.merge_node import MergeLibraryNode
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.transformation.passes.offloading import OffloadToAccelerator
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadingIRNode as MonolithIRNode
from dace.transformation.passes.offloading.offloading_helpers import traverse_IR
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.ordered import OrderedSet
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
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
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
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})

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


def test_apply_gpu_transformations_offloads_with_the_pass():
    """``apply_gpu_transformations`` runs ``OffloadToAccelerator``, and nothing else.

    The offloading starts from device-resident inputs and stages a host copy where host code reads
    one; the transformation it replaced started on the host and never needed such a copy, so the
    ``_host`` name is the signature of which offloader ran.
    """
    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    assert [name for name in sdfg.arrays if name.endswith('_host')], \
        'no `_host` staging: apply_gpu_transformations did not run the offloading pass'


def test_simplify_is_honoured_by_the_offloading():
    """``simplify`` is ``apply_gpu_transformations``'s contract, not the offloading's.

    ``OffloadToAccelerator`` takes no such option and leaves the copy states it inserts unfused, so
    a caller that asked for a simplified graph has to be handed one by the method itself.
    """
    plain = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    simplified = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    plain.apply_gpu_transformations(validate=False, simplify=False)
    simplified.apply_gpu_transformations(validate=False, simplify=True)
    plain.validate()
    simplified.validate()

    def size(sdfg):
        """States, dataflow nodes, dataflow edges. Asserting on any single count would pin the
        incidental shape simplify happens to reach rather than the contract that it shrinks."""
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

    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
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


def test_a_host_pinned_library_node_keeps_its_operands_on_the_host():
    """``ScatterConflictCheck`` keeps its flag and its tag scratch on the HOST in every expansion,
    the CUDA one included -- it tags on the device out of the CUB scratch pool and only sizes that
    buffer from the scratch array. The new pass gives a whole state ONE location, so it would move
    both with the rest and the node's own validation rejects the result. Leaving the node's
    operands where it declared them is what keeps that from becoming a mid-rewrite crash."""
    sdfg = canonicalized_with_gpu_inputs('s4113_d_single')
    checks = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == 'ScatterConflictCheck']
    assert checks, 's4113 no longer canonicalizes to a ScatterConflictCheck'
    assert all(n.host_connectors for n in checks), 'the node must declare which connectors stay on the host'

    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    for node, state in sdfg.all_nodes_recursive():
        if type(node).__name__ != 'ScatterConflictCheck':
            continue
        for edge in state.in_edges(node) + state.out_edges(node):
            connector = edge.dst_conn if edge.dst is node else edge.src_conn
            if connector in node.host_connectors:
                assert state.sdfg.arrays[edge.data.data].storage not in GPU_RESIDENT_STORAGES, \
                    f'{connector} ({edge.data.data}) was moved off the host'


def test_a_scatter_dispatcher_copies_only_inside_its_sequential_arm():
    """The scatter guard runs the parallel Map unless the index array collides, then the original
    loop on the host. Only that fallback pays host copies, in and back out inside itself: s4113 ran
    the arm first in the IR and the conditional closed on the host, so the parallel arm copied ``a``
    to the host and the program copied it back after the guard on every execution."""
    sdfg = canonicalized_with_gpu_inputs('s4113_d_single')
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    guard = guard_block(sdfg)
    sequential = next(region for _c, region in guard.branches if any(
        isinstance(b, LoopRegion) and b.pinned_sequential for b in region.nodes()))
    in_arm = set(sequential.all_control_flow_blocks(recursive=True))
    stray = [
        b.label for b in sdfg.all_control_flow_blocks(recursive=True) if is_copy_state(sdfg, b) and b not in in_arm
    ]
    assert not stray, f'copies outside the sequential arm, paid on every execution: {stray}'
    order = list(sequential.bfs_nodes(sequential.start_block))
    loop = next(b for b in order if isinstance(b, LoopRegion))
    copies = [b for b in order if is_copy_state(sdfg, b)]
    assert copies and order.index(copies[0]) < order.index(loop) < order.index(copies[-1]), \
        'the arm must copy in before its loop and back out after it'


def find_first_kernel_with_gpu_inputs() -> dace.SDFG:
    """A standalone ``FindFirst`` search whose signature arrays are already on the device.

    The shape an early-exit search canonicalizes to: one library node, no surrounding map. No pass
    builds this automatically any more -- ``EarlyExitToFindIndex`` (the search-loop lift) was
    deleted because its rewrite never paid for itself; ``FindFirst`` remains exported library API,
    constructed by hand the way ``tests/library/detect_test.py`` and
    ``tests/codegen/cpf/test_emission.py`` already do. Building it directly here, with both arrays
    pre-pinned to the device the way ``canonicalized_with_gpu_inputs`` pins a kernel's signature,
    keeps this test exercising the same placement decision without depending on that lift.
    """
    sdfg = dace.SDFG('find_first_kernel_with_gpu_inputs')
    sdfg.add_array('a', [256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [1], dace.int64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = FindFirst('search', predicate=f'_a[{INDEX_NAME}] > 0.5', begin=0, end=256)
    node.add_in_connector('_a', dace.pointer(dace.float64))
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_a', dace.Memlet('a[0:256]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def test_a_find_first_answer_stays_on_the_host():
    """``FindFirst``'s answer is a HOST scalar in every expansion, the device one included.

    The device search leaves its result in CUB scratch; ``find_first_index_device`` copies it back
    and writes ``*out`` on the host, and the CUDA expansion's tasklet assigns the out-connector from
    a host stack variable. Promoting that scalar to device memory has host code write a device
    pointer -- which VALIDATES, and then corrupts, which is why this is asserted on storage rather
    than left to the SDFG checker.
    """
    sdfg = find_first_kernel_with_gpu_inputs()
    finds = [(n, st) for n, st in sdfg.all_nodes_recursive() if type(n).__name__ == 'FindFirst']
    assert finds, 'the fixture built no FindFirst node; the check below would be vacuous'
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
    sdfg.apply_gpu_transformations(validate=False, simplify=False)  # the assertion fires inside
    sdfg.validate()


def select_with_a_single_element_fallback(state: dace.SDFGState, fallback: dace.nodes.AccessNode, t: str, mask: str,
                                          out: str) -> None:
    """``out = np.where(mask, t, fallback[0])`` as the frontend builds it: one ``MergeLibraryNode``."""
    node = MergeLibraryNode('where')
    state.add_node(node)
    state.add_edge(state.add_read(t), None, node, MergeLibraryNode.TRUE_CONNECTOR_NAME, dace.Memlet(f'{t}[0:64]'))
    state.add_edge(state.add_read(mask), None, node, MergeLibraryNode.MASK_CONNECTOR_NAME, dace.Memlet(f'{mask}[0:64]'))
    state.add_edge(fallback, None, node, MergeLibraryNode.FALSE_CONNECTOR_NAME, dace.Memlet(f'{fallback.data}[0]'))
    state.add_edge(node, MergeLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write(out), None,
                   dace.Memlet(f'{out}[0:64]'))


def where_with_a_host_computed_fallback() -> dace.SDFG:
    """``out = np.where(nonsing, fac, -x)``: a host tasklet computes the length-1 fallback.

    QE vexx_k's ``fac = np.where(nonsing, fac, -exxdiv)`` in ``g2_convolution``.
    """
    sdfg = dace.SDFG('where_with_a_host_computed_fallback')
    sdfg.add_array('fac', [64], dace.float64)
    sdfg.add_array('nonsing', [64], dace.bool_)
    sdfg.add_scalar('x', dace.float64)
    sdfg.add_array('out', [64], dace.float64)
    sdfg.add_array('neg_x', [1], dace.float64, transient=True)
    state = sdfg.add_state()
    usub = state.add_tasklet('usub', {'inp'}, {'res'}, 'res = -inp')
    neg_x = state.add_access('neg_x')
    state.add_edge(state.add_read('x'), None, usub, 'inp', dace.Memlet('x[0]'))
    state.add_edge(usub, 'res', neg_x, None, dace.Memlet('neg_x[0]'))
    select_with_a_single_element_fallback(state, neg_x, 'fac', 'nonsing', 'out')
    sdfg.validate()
    return sdfg


def where_with_a_device_reduced_fallback() -> dace.SDFG:
    """``out = np.where(match, jv, np.max(jv))``: a reduction in one state, the select in the next.

    QE vexx_k's ``jmin = np.min(np.where(match, jv, np.max(jv)))``.
    """
    sdfg = dace.SDFG('where_with_a_device_reduced_fallback')
    sdfg.add_array('jv', [64], dace.int32)
    sdfg.add_array('match', [64], dace.bool_)
    sdfg.add_array('out', [64], dace.int32)
    sdfg.add_scalar('max_jv', dace.int32, transient=True)
    reduce_state = sdfg.add_state('reduce')
    reduce = Reduce('max', wcr='lambda a, b: max(a, b)', axes=None, identity=None)
    reduce_state.add_node(reduce)
    reduce_state.add_edge(reduce_state.add_read('jv'), None, reduce, '_in', dace.Memlet('jv[0:64]'))
    reduce_state.add_edge(reduce, '_out', reduce_state.add_write('max_jv'), None, dace.Memlet('max_jv[0]'))
    select_state = sdfg.add_state_after(reduce_state, 'select')
    select_with_a_single_element_fallback(select_state, select_state.add_read('max_jv'), 'jv', 'match', 'out')
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('build', [where_with_a_host_computed_fallback, where_with_a_device_reduced_fallback])
def test_a_select_kernel_reads_its_single_element_fallback_from_the_device(build):
    """A ``MergeLibraryNode`` expands to a device map, which dereferences ``_mrg_f`` in the kernel.

    The host preference for single-element inputs of a device library node exists for vendor calls,
    which take a host pointer as happily as a device one. Applied to the select, it handed the
    kernel a host address: the reduced ``max_jv`` was copied back to ``max_jv_host`` only for the
    select to read it there, and the host-computed ``-exxdiv`` never left the host. QE vexx_k on the
    DaCe GPU canonicalize column faulted with "Memory access fault by GPU ... on address 0x26326000".
    """
    sdfg = build()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()

    selects = [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, MergeLibraryNode)]
    assert len(selects) == 1
    node, state = selects[0]
    assert node.schedule == dtypes.ScheduleType.GPU_Device
    fallback = next(e for e in state.in_edges(node) if e.dst_conn == MergeLibraryNode.FALSE_CONNECTOR_NAME)
    desc = state.sdfg.arrays[fallback.data.data]
    assert desc.storage in GPU_RESIDENT_STORAGES, (f'the select kernel reads {fallback.data.data} from '
                                                   f'{desc.storage}, a host pointer')


def test_a_vendor_coefficient_may_stay_on_the_host():
    """The permission is per connector: Gemm's runtime ``_alpha``/``_beta`` keep it, the select has none."""
    from dace.libraries.blas.nodes.gemm import Gemm
    assert Gemm.host_or_device_connectors == frozenset({'_alpha', '_beta'})
    assert MergeLibraryNode.host_or_device_connectors == frozenset()


@pytest.mark.gpu
def test_a_select_with_a_host_computed_fallback_runs_on_the_device():
    """The same program, run: no host pointer reaches the kernel, and every masked-out slot is -x."""
    sdfg = where_with_a_host_computed_fallback()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(3)
    fac = rng.random(64)
    nonsing = rng.random(64) > 0.5
    out = np.zeros(64)
    sdfg(fac=fac, nonsing=nonsing, x=1.5, out=out)
    assert np.array_equal(out, np.where(nonsing, fac, -1.5))


@pytest.mark.gpu
def test_a_select_with_a_device_reduced_fallback_runs_on_the_device():
    """The same program, run: the reduced maximum is read where the reduction wrote it."""
    sdfg = where_with_a_device_reduced_fallback()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(5)
    jv = rng.integers(-100, 100, 64).astype(np.int32)
    match = rng.random(64) > 0.5
    out = np.zeros(64, dtype=np.int32)
    sdfg(jv=jv, match=match, out=out)
    assert np.array_equal(out, np.where(match, jv, jv.max()))


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
    """A guard whose FIRST arm is the pinned fallback loop, a kernel, then an edge reading ``A[0]``.

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
    loop.pinned_sequential = True
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


ROWS = dace.symbol('ROWS')
COLS = dace.symbol('COLS')


@dace.program
def scatter_rows_after_a_row_map(agg: dace.int64[ROWS], x: dace.float64[ROWS, COLS], acc: dace.float64[ROWS],
                                 tmp: dace.float64[ROWS, COLS]):
    """amg_setup's shape: the scatter guard's fallback is a row loop with a map in its body."""
    for i in range(ROWS):
        for j in dace.map[0:COLS]:
            tmp[i, j] = x[i, j] * 2.0
        acc[agg[i]] = acc[agg[i]] + tmp[i, 0] + tmp[i, COLS - 1]


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


def copies_inside_loops(sdfg: dace.SDFG) -> list[str]:
    return [
        b.label for loop in sdfg.all_control_flow_blocks(recursive=True) if isinstance(loop, LoopRegion)
        for b in loop.all_control_flow_blocks(recursive=True) if is_copy_state(sdfg, b)
    ]


def test_a_small_map_in_a_host_loop_stays_on_the_host():
    """Offloaded, the gather copied the whole ``acc`` host<->device on every row: O(rows^2) traffic,
    amg_setup at 344 s per call against 0.8 s on the CPU."""
    sdfg = rows_with_a_small_gather.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    assert not copies_inside_loops(sdfg), copies_inside_loops(sdfg)
    gather = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.range.num_elements() == 4
    ]
    assert gather and all(n.map.schedule == dtypes.ScheduleType.Sequential for n in gather)


def test_a_map_over_the_whole_shared_array_stays_a_kernel():
    sdfg = rows_with_a_full_stencil.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    full = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert full and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in full)


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


def test_a_map_whose_size_cannot_be_ranked_stays_a_kernel():
    sdfg = rows_with_an_unranked_gather.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    gather = [
        n for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and parent is not None and 'COLS' in n.map.range.free_symbols
    ]
    assert gather and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in gather)


def offloaded_scatter_rows() -> dace.SDFG:
    sdfg = scatter_rows_after_a_row_map.to_sdfg(simplify=True)
    canonicalize(sdfg, target='gpu')
    offload_to_gpu(sdfg)
    return finalize_for_target(sdfg, 'gpu')


def test_a_scatter_fallback_keeps_its_row_map_on_the_host():
    sdfg = offloaded_scatter_rows()
    fallback = [
        b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, LoopRegion) and b.pinned_sequential
    ]
    assert fallback, 'the scatter no longer canonicalizes to a guarded fallback loop'
    maps = [
        n for loop in fallback for st in loop.all_states() for n in st.nodes() if isinstance(n, dace.nodes.MapEntry)
    ]
    assert maps and all(n.map.schedule not in dtypes.GPU_SCHEDULES for n in maps), [n.map.schedule for n in maps]


@pytest.mark.gpu
def test_a_scatter_fallback_on_the_host_computes_what_numpy_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = offloaded_scatter_rows()
    rng = np.random.default_rng(0)
    for agg in (np.arange(8, dtype=np.int64), np.array([0, 2, 1, 3, 0, 5, 7, 2], dtype=np.int64)):
        x = rng.random((8, 4))
        want = np.zeros(8)
        np.add.at(want, agg, 2 * x[:, 0] + 2 * x[:, 3])
        acc, tmp = cupy.zeros(8), cupy.zeros((8, 4))
        sdfg(agg=cupy.asarray(agg), x=cupy.asarray(x), acc=acc, tmp=tmp, ROWS=8, COLS=4)
        np.testing.assert_allclose(acc.get(), want)


@pytest.mark.gpu
def test_a_small_map_kept_on_the_host_computes_what_numpy_computes():
    sdfg = rows_with_a_small_gather.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    agg = np.array([0, 2, 1, 3, 0, 5, 7, 2], dtype=np.int64)
    w = np.arange(8, dtype=np.float64)
    out = np.zeros(4)
    sdfg(agg=agg, w=w, out=out, ROWS=8)
    acc, want = np.zeros(8), np.zeros(4)
    for i in range(8):
        acc[agg[i]] += w[i]
        want += acc[:4]
    np.testing.assert_allclose(out, want)


if __name__ == '__main__':
    test_the_guarded_kernel_still_canonicalizes_to_a_parallel_and_a_sequential_arm()
    test_the_fallback_arm_owns_its_copies()
    test_the_fallback_copies_in_before_it_runs_and_out_after()
    test_an_offloaded_scan_gets_its_device_lowering()
    test_a_host_pinned_library_node_keeps_its_operands_on_the_host()
    test_a_find_first_answer_stays_on_the_host()
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


def test_the_sequential_arm_of_a_guarded_loop_keeps_its_host_copies():
    """The exception to the rule above: a fallback arm is host code that OWNS its copies.

    Canonicalization emits a loop it can only parallelize under a runtime condition as both arms of
    one ConditionalBlock. The sequential arm's tasklets are free tasklets over device-resident
    arrays too, so the lift above would take them -- and delete the copies that are the entire point
    of the arm. What separates the two is the loop: a fallback is a LoopRegion inside a conditional,
    while nbody's is an ordinary loop with no conditional above it.
    """
    from dace.transformation.passes.offloading.offload_to_accelerator import in_sequential_specialization_arm

    sdfg = canonicalized_with_gpu_inputs(GUARDED_KERNEL)
    conditionals = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]
    assert conditionals, 'the kernel lost its guard, so this test would pass without checking one'

    arms = {state.label: in_sequential_specialization_arm(state) for state in sdfg.all_states()}
    sequential = [label for label, inside in arms.items() if inside]
    assert sequential, 'no state was recognised as the sequential arm'
    for label in sequential:
        state = next(s for s in sdfg.all_states() if s.label == label)
        loops = []
        current = state.parent_graph
        while current is not None:
            loops.append(isinstance(current, LoopRegion))
            current = getattr(current, 'parent_graph', None)
        assert any(loops), f'{label} was called a fallback arm without a loop above it'


def test_apply_gpu_storage_moves_an_array_an_interstate_edge_reads_and_the_offload_stages_it():
    """EVERY non-transient array moves, and the offload is what makes the host read legal.

    An interstate edge's reads live in its condition and assignments, not on any AccessNode, so a
    pass that walks nodes concludes the array is only ever touched by the maps it can see. Indexing
    one element does not make the container a Scalar, so ``bounds[0] < bounds[1]`` on an edge is a
    host read of a container the caller delivers as a device pointer.

    Holding the array back does not fix that -- it only moves the contradiction to the ABI, where
    nothing checks it, and strands any kernel that wanted the array. ``stage_on_host`` is the
    resolution: a host copy, every host use repointed at it, a copy at the boundary. Measured on
    llr-focus40: six kernels were unoffloadable until the array moved.
    """
    from dace.transformation.auto import auto_optimize

    sdfg = dace.SDFG('interstate_read_is_staged')
    sdfg.add_array('bounds', [2], dace.float64, transient=False)
    sdfg.add_array('data', [8], dace.float64, transient=False)
    body = sdfg.add_state('body')
    entry, exit_ = body.add_map('k', {'i': '0:8'}, schedule=dace.ScheduleType.GPU_Device)
    tasklet = body.add_tasklet('scale', {'d': None}, {'o': None}, 'o = d * 2.0')
    body.add_memlet_path(body.add_read('data'), entry, tasklet, dst_conn='d', memlet=dace.Memlet('data[i]'))
    body.add_memlet_path(tasklet, exit_, body.add_write('data'), src_conn='o', memlet=dace.Memlet('data[i]'))

    # A REAL branch, both arms doing work: a lone conditional successor folds away in simplify,
    # and the edge carrying the read goes with it.
    def bump(label: str, by: str) -> dace.SDFGState:
        st = sdfg.add_state(label)
        m_entry, m_exit = st.add_map(label, {'i': '0:8'}, schedule=dace.ScheduleType.GPU_Device)
        node = st.add_tasklet(label, {'d': None}, {'o': None}, f'o = d + {by}')
        st.add_memlet_path(st.add_read('data'), m_entry, node, dst_conn='d', memlet=dace.Memlet('data[i]'))
        st.add_memlet_path(node, m_exit, st.add_write('data'), src_conn='o', memlet=dace.Memlet('data[i]'))
        return st

    # The read that no AccessNode carries.
    sdfg.add_edge(body, bump('lo', '1.0'), dace.InterstateEdge(condition='bounds[0] < bounds[1]'))
    sdfg.add_edge(body, bump('hi', '2.0'), dace.InterstateEdge(condition='not (bounds[0] < bounds[1])'))

    auto_optimize.apply_gpu_storage(sdfg)

    assert sdfg.arrays['bounds'].storage is dace.StorageType.GPU_Global
    assert sdfg.arrays['data'].storage is dace.StorageType.GPU_Global

    sdfg.apply_gpu_transformations()

    # Both places the read can live: on an interstate edge, or -- once simplify raises the branch
    # into a ConditionalBlock -- in that block's own condition, which is what get_meta_read_memlets
    # answers and what replace_meta_accesses had to rewrite.
    read = set(auto_optimize.interstate_read_names(sdfg))
    for block in sdfg.all_control_flow_blocks():
        read |= {m.data for m in block.get_meta_read_memlets() if m is not None and m.data in sdfg.arrays}

    assert 'bounds' not in read, 'host code still reads the device array'
    staged = sorted(name for name in read if name.startswith('bounds'))
    assert staged, f'the condition reads nothing that came from bounds: {sorted(read)}'
    for name in staged:
        assert sdfg.arrays[name].storage is not dace.StorageType.GPU_Global
    sdfg.validate()


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


def ir_chain(node_class, length: int):
    """``(open, [states])`` for one section holding ``length`` states in a row.

    The shape the IR pass builds for straight-line code: ``open -> s0 -> ... -> sN-1 -> close``.
    """
    sdfg = dace.SDFG('ir_chain')
    region = sdfg.add_state('section')
    open_node = node_class.new_open_node(region)
    states = [node_class.new_state_node(sdfg.add_state(f's{i}'), OrderedSet(), OrderedSet()) for i in range(length)]
    previous = open_node
    for state in states:
        previous.append_node(state)
        previous = state
    previous.append_node(open_node.close)
    return open_node, states


@pytest.mark.parametrize('node_class', (OffloadingIRNode, MonolithIRNode))
def test_the_ir_walk_does_not_recurse_once_per_block(node_class):
    """The IR holds one node per state and per interstate edge, so its chain is as long as the
    program has blocks. Walking it by RECURSION spends one Python frame per block and overran the
    interpreter stack on the first application-sized kernel: CloudSC canonicalized for the GPU died
    with ``RecursionError: maximum recursion depth exceeded`` inside ``apply_gpu_transformations``,
    which stopped its whole GPU canonicalization (and with it every CPF device render of it).

    A chain is also the case where the answer is obvious, so this asserts the RESULT as well as the
    absence of the crash: the one tail of a straight line is its last state."""
    open_node, states = ir_chain(node_class, LONG_CHAIN)
    assert open_node.get_all_tails() == [states[-1]]


def test_traverse_ir_does_not_recurse_once_per_block():
    """:func:`~dace.transformation.passes.offloading.offloading_helpers.traverse_IR` walks the same
    chain and had the same stack cost. It visits every node exactly once, in the order the
    recursion did -- which is what the collected labels check."""
    open_node, states = ir_chain(OffloadingIRNode, LONG_CHAIN)
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
