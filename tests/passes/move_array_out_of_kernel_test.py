# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``_tile_extent`` returns the static tile width for a tiled inner-map extent so the
lifted transient's shape does not leak an out-of-scope outer-loop symbol into ``cudaMalloc``."""
import numpy as np
import pytest
import sympy

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.move_array_out_of_kernel import (_prepend_subscript_indices, _tile_extent,
                                                                 MoveArrayOutOfKernel)

NX, NZ = (dace.symbol(s, dtype=dace.int64) for s in ('NX', 'NZ'))


def test_tile_extent_recognises_min_pattern():
    """For a ``Min``-bounded inner-map extent, ``_tile_extent`` returns the static tile width 32."""
    b_i = sympy.Symbol('b_i')
    N = sympy.Symbol('N')
    max_elem = sympy.Min(N - 1, b_i + 31)
    min_elem = b_i
    extent = _tile_extent(max_elem, min_elem)
    assert extent == 32, f"expected 32, got {extent}"
    assert b_i not in extent.free_symbols, f"tile extent leaks outer-loop symbol: {extent.free_symbols}"


def test_tile_extent_falls_back_for_plain_range():
    """No ``Min`` in the upper bound: the symbolic extent is returned unchanged."""
    W = sympy.Symbol('W')
    extent = _tile_extent(W - 1, sympy.Integer(0))
    assert sympy.simplify(extent - W) == 0, f"expected W, got {extent}"


def test_tile_extent_handles_outer_block_strided_loop():
    """Outer strided GPU_Device map ``b_i = 0:N:32``: the fallback returns the host-visible ``N``."""
    N = sympy.Symbol('N')
    # max_element() of a strided range comes back as ``N - 1``; pin that and check there is no leak.
    extent = _tile_extent(N - 1, sympy.Integer(0))
    assert sympy.simplify(extent - N) == 0
    assert sympy.Symbol('b_i') not in extent.free_symbols


def test_get_new_shape_info_multidim_prepend_strides():
    """A GPU map that prepends >1 dimension must yield packed C-layout strides.

    Lifting an ``[64]`` transient out of a 2-D kernel ``map[0:128, 0:32]`` gives shape
    ``[128, 32, 64]``; the packed strides are ``[2048, 64, 1]``. Regression: the stride loop
    inserted the running accumulator *before* multiplying and iterated ``range_size[:-1]``, so
    it produced ``[64, 64, 1]`` -- both prepended dims wrongly shared stride 64.
    """
    sdfg = dace.SDFG('move_array_strides')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:128', j='0:32'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [64])
    new_shape, new_strides, new_total, _new_offsets = MoveArrayOutOfKernel().get_new_shape_info(arr, [me])

    assert [int(s) for s in new_shape] == [128, 32, 64], new_shape
    assert [int(s) for s in new_strides] == [2048, 64, 1], new_strides
    assert int(new_total) == 128 * 32 * 64, new_total


def test_prepend_subscript_indices_rewrites_an_inlined_body():
    """``InlineTaskletConnectors`` bakes the memlet subset into the body TEXT before this pass runs.

    From that point the readable generator reads the body, not the memlet, as the access. Reshaping
    the descriptor without rewriting the body leaves a rank-1 subscript on a rank-3 array; the
    emitter finds the rank mismatch, declines to build an ``arr_idx(...)`` access and emits the
    stale subscript verbatim, so every kernel iteration writes the same leading slice.
    """
    rewritten = _prepend_subscript_indices('tmp[k] = a[i, j, k] * 2.0', 'tmp', ['i', 'j'])
    assert rewritten == 'tmp[i, j, k] = a[i, j, k] * 2.0'

    both = _prepend_subscript_indices('out[k] = tmp[NZ - 1 - k] + 1.0', 'tmp', ['i', 'j'])
    assert both == 'out[k] = tmp[i, j, NZ - 1 - k] + 1.0'


def test_prepend_subscript_indices_leaves_unrelated_bodies_alone():
    """A body that never names the array, and one that cannot be parsed, are both returned unchanged.

    The rewrite is best-effort by design: a body it cannot handle keeps the form it had, which is
    exactly the behaviour that existed before the rewrite.
    """
    assert _prepend_subscript_indices('out[k] = a[k] + 1.0', 'tmp', ['i']) is None
    assert _prepend_subscript_indices('tmp[k = ', 'tmp', ['i']) is None
    assert _prepend_subscript_indices('tmp[k] = 1.0', 'tmp', []) is None


# --------------------------------------------------------------------------------------------------
# The lift prefixes memlets only where it reshaped the descriptor.
# --------------------------------------------------------------------------------------------------
def kernel_with_scratch_below_a_nested_sdfg() -> dace.SDFG:
    """``out[i, k] = 2 * a[i, NZ - 1 - k] + 1`` through a per-iteration ``tmp[NZ]`` that is passed
    one level FURTHER DOWN than where it is defined.

    Three levels, and the middle one is the point: ``tmp`` is defined in ``body`` (inside the
    kernel, so it is lifted), and ``body`` hands it to ``producer`` and ``consumer`` through
    connectors, so each of those holds its own ``tmp`` descriptor at the original rank. The
    reversed read keeps the buffer alive; a straight-through copy is recomputed into the consumer
    and the buffer disappears.
    """
    producer = dace.SDFG('producer')
    producer.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    producer.add_array('tmp', [NZ], dace.float64, storage=dtypes.StorageType.Register)
    producer.add_state('fill',
                       is_start_block=True).add_mapped_tasklet('scale', {'k': '0:NZ'}, {'__in': dace.Memlet('a[i, k]')},
                                                               '__out = __in * 2.0', {'__out': dace.Memlet('tmp[k]')},
                                                               schedule=dtypes.ScheduleType.Sequential,
                                                               external_edges=True)

    consumer = dace.SDFG('consumer')
    consumer.add_array('tmp', [NZ], dace.float64, storage=dtypes.StorageType.Register)
    consumer.add_array('out', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    consumer.add_state('drain', is_start_block=True).add_mapped_tasklet('shift', {'k': '0:NZ'},
                                                                        {'__in': dace.Memlet('tmp[NZ - 1 - k]')},
                                                                        '__out = __in + 1.0',
                                                                        {'__out': dace.Memlet('out[i, k]')},
                                                                        schedule=dtypes.ScheduleType.Sequential,
                                                                        external_edges=True)

    body = dace.SDFG('body')
    body.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    body.add_array('out', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    # Symbolic extent: there is no device-local form for it (a VLA in device code), which is what
    # makes this buffer take the lift rather than the register demotion.
    body.add_array('tmp', [NZ], dace.float64, transient=True, storage=dtypes.StorageType.Register)

    fill = body.add_state('call_producer', is_start_block=True)
    pnode = fill.add_nested_sdfg(producer, {'a'}, {'tmp'}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    fill.add_edge(fill.add_read('a'), None, pnode, 'a', dace.Memlet('a[0:NX, 0:NZ]'))
    fill.add_edge(pnode, 'tmp', fill.add_write('tmp'), None, dace.Memlet('tmp[0:NZ]'))

    drain = body.add_state_after(fill, 'call_consumer')
    cnode = drain.add_nested_sdfg(consumer, {'tmp'}, {'out'}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    drain.add_edge(drain.add_read('tmp'), None, cnode, 'tmp', dace.Memlet('tmp[0:NZ]'))
    drain.add_edge(cnode, 'out', drain.add_write('out'), None, dace.Memlet('out[0:NX, 0:NZ]'))

    sdfg = dace.SDFG('scratch_below_a_nested_sdfg')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('grid', is_start_block=True)
    entry, exit_node = state.add_map('kernel', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(body, {'a'}, {'out'}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    state.add_memlet_path(nsdfg,
                          exit_node,
                          state.add_write('out'),
                          src_conn='out',
                          memlet=dace.Memlet('out[0:NX, 0:NZ]'))
    sdfg.validate()
    return sdfg


def test_lift_leaves_descendant_nested_sdfgs_at_their_own_rank():
    """Every memlet's rank must match the rank of the descriptor it is written against.

    The lift gives the buffer one slice per kernel iteration, which adds leading dimensions to the
    descriptor -- but only in the SDFGs it reshapes: the definition site and its ancestors up to the
    kernel's own SDFG. A nested SDFG BELOW the definition keeps its own descriptor at the original
    rank, and the connector memlet one level up (which does carry the new index) is what selects
    the slice for it. Prefixing inside those descendants too left a rank-2 memlet on a rank-1
    descriptor, which validation rejects with ``Memlet subset does not match node dimension``.

    Checked over every descriptor of the buffer rather than only the one that failed, so a lift
    that reshapes some other level without its memlets is caught by the same assertion.
    """
    from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline

    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})

    ranks = {}
    for nested in sdfg.all_sdfgs_recursive():
        if 'tmp' not in nested.arrays:
            continue
        rank = len(nested.arrays['tmp'].shape)
        ranks[nested.name] = rank
        for state in nested.all_states():
            for edge in state.edges():
                if edge.data.data != 'tmp' or edge.data.subset is None:
                    continue
                assert edge.data.subset.dims() == rank, (
                    f"{nested.name}: memlet {edge.data} has rank {edge.data.subset.dims()} against a "
                    f"rank-{rank} descriptor {nested.arrays['tmp'].shape}")

    assert ranks, 'the scratch buffer vanished entirely'
    assert ranks.get('body') == 2, f'the definition site did not gain the per-iteration index: {ranks}'
    assert ranks.get('producer') == 1 and ranks.get('consumer') == 1, \
        f'a descendant nested SDFG was reshaped, so this no longer covers the case: {ranks}'
    sdfg.validate()


@pytest.mark.gpu
def test_lifted_scratch_below_a_nested_sdfg_computes_the_right_values():
    """Structure is not enough: the slices must also be disjoint per kernel iteration. A shared one
    validates, compiles, and returns another iteration's numbers."""
    cupy = pytest.importorskip('cupy')

    nx, nz = 5, 7
    rng = np.random.default_rng(0)
    host_a = rng.random((nx, nz))
    expected = host_a[:, ::-1] * 2.0 + 1.0

    out = cupy.zeros((nx, nz))
    kernel_with_scratch_below_a_nested_sdfg()(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), expected)


def kernel_over_k_beside_a_nested_k_loop() -> dace.SDFG:
    """Two kernels in one SDFG: one whose map parameter is ``k``, one whose body OWNS a ``k`` loop.

    ``mid[i, k] = 2 * a[NX - 1 - i, k]`` through a per-iteration ``tmp[NX]`` defined one level
    down, then ``out[i, k] = mid[i, k] + 1`` in a sequential sweep. The scratch buffer's extent is
    symbolic, so it is lifted and the lift propagates the first kernel's ``k`` into every nest of
    the SDFG -- including the sweep, which assigns ``k`` itself.
    """
    scratch_body = dace.SDFG('scratch_body')
    scratch_body.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    scratch_body.add_array('mid', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    # Symbolic extent: no device-local form (a VLA in device code), so it takes the lift rather
    # than the register demotion.
    scratch_body.add_array('tmp', [NX], dace.float64, transient=True, storage=dtypes.StorageType.Register)
    fill = scratch_body.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('scale', {'i': '0:NX'}, {'__in': dace.Memlet('a[i, k]')},
                            '__out = __in * 2.0', {'__out': dace.Memlet('tmp[i]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    drain = scratch_body.add_state_after(fill, 'drain')
    # The reversed read keeps the buffer alive; a straight-through copy is recomputed away.
    drain.add_mapped_tasklet('shift', {'i': '0:NX'}, {'__in': dace.Memlet('tmp[NX - 1 - i]')},
                             '__out = __in', {'__out': dace.Memlet('mid[i, k]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)

    sweep = dace.SDFG('sweep')
    sweep.add_array('mid', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sweep.add_array('out', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    loop = LoopRegion('k_sweep',
                      condition_expr='k < NZ',
                      loop_var='k',
                      initialize_expr='k = 0',
                      update_expr='k = k + 1')
    sweep.add_node(loop, is_start_block=True)
    step = loop.add_state('step', is_start_block=True)
    bump = step.add_tasklet('bump', {'__in'}, {'__out'}, '__out = __in + 1.0')
    step.add_edge(step.add_read('mid'), None, bump, '__in', dace.Memlet('mid[i, k]'))
    step.add_edge(bump, '__out', step.add_write('out'), None, dace.Memlet('out[i, k]'))

    sdfg = dace.SDFG('kernel_over_k_beside_a_nested_k_loop')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('mid', [NX, NZ], dace.float64, transient=True, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [NX, NZ], dace.float64, storage=dtypes.StorageType.GPU_Global)

    scratch = sdfg.add_state('scratch', is_start_block=True)
    entry, exit_node = scratch.add_map('kernel_k', dict(k='0:NZ'), schedule=dtypes.ScheduleType.GPU_Device)
    scratch_nsdfg = scratch.add_nested_sdfg(scratch_body, {'a'}, {'mid'}, symbol_mapping=dict(k='k', NX=NX, NZ=NZ))
    scratch.add_memlet_path(scratch.add_read('a'),
                            entry,
                            scratch_nsdfg,
                            dst_conn='a',
                            memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    scratch.add_memlet_path(scratch_nsdfg,
                            exit_node,
                            scratch.add_write('mid'),
                            src_conn='mid',
                            memlet=dace.Memlet('mid[0:NX, 0:NZ]'))

    sweep_state = sdfg.add_state_after(scratch, 'sweep')
    sentry, sexit = sweep_state.add_map('kernel_i', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    sweep_nsdfg = sweep_state.add_nested_sdfg(sweep, {'mid'}, {'out'}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    sweep_state.add_memlet_path(sweep_state.add_read('mid'),
                                sentry,
                                sweep_nsdfg,
                                dst_conn='mid',
                                memlet=dace.Memlet('mid[0:NX, 0:NZ]'))
    sweep_state.add_memlet_path(sweep_nsdfg,
                                sexit,
                                sweep_state.add_write('out'),
                                src_conn='out',
                                memlet=dace.Memlet('out[0:NX, 0:NZ]'))
    sdfg.validate()
    return sdfg


def test_lift_does_not_bind_a_name_the_nest_assigns_itself():
    """The lift must not hand a nest a binding for a symbol that nest gives its own value.

    Codegen filters a nested function's symbol parameters through ``used_symbols`` with
    ``keep_defined_in_mapping``, which drops a name the nest assigns, while the frame skips the
    hoisted ``int64_t k;`` declaration precisely because the name IS in ``symbol_mapping``. The
    sweep's counter then reaches the C++ neither as a parameter nor as a declaration and the
    emitted ``for (k = ...)`` names nothing, which is a hard compile error.
    """
    from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline

    sdfg = kernel_over_k_beside_a_nested_k_loop()
    GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})

    sweeps = [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.NestedSDFG) and node.sdfg.name.startswith('sweep')
    ]
    assert sweeps, 'the sweep nest vanished, so this no longer covers the case'
    for nest in sweeps:
        assert 'k' not in nest.symbol_mapping, f'{nest.label} was bound to the counter it assigns itself'
    sdfg.validate()


def test_lifted_kernel_declares_a_nested_loop_counter():
    """End to end: the emitted device code must declare the sweep's counter, not just name it."""
    device_code = ''.join(obj.clean_code for obj in kernel_over_k_beside_a_nested_k_loop().generate_code())
    assert 'for (k = ' not in device_code, 'the loop counter is used without a declaration in scope'
    assert 'k = ' in device_code, 'the sweep loop disappeared, so this no longer covers the case'


#: A block wider than the kernel, so an index list in the wrong order overruns the buffer.
KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT = 3, 5, 7


def kernel_with_a_thread_block_around_its_scratch():
    """``a[i, t] = tmp[2]`` with ``tmp[2] = i + 10 * t``, where ``tmp`` sits inside a ``GPU_ThreadBlock``
    map over ``t`` inside the ``GPU_Device`` kernel over ``i`` -- the nesting ``WarpTiling`` leaves."""
    sdfg = dace.SDFG('scratch_inside_a_thread_block')
    sdfg.add_array('a', [KERNEL_EXTENT, BLOCK_EXTENT], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('tmp', [SCRATCH_EXTENT], dace.float64, transient=True, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('grid', is_start_block=True)
    kernel_entry, kernel_exit = state.add_map('kernel',
                                              dict(i=f'0:{KERNEL_EXTENT}'),
                                              schedule=dtypes.ScheduleType.GPU_Device)
    block_entry, block_exit = state.add_map('block',
                                            dict(t=f'0:{BLOCK_EXTENT}'),
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    fill = state.add_tasklet('fill', {}, {'__out'}, '__out = i + 10.0 * t')
    use = state.add_tasklet('use', {'__in'}, {'__out'}, '__out = __in')
    scratch = state.add_access('tmp')
    state.add_nedge(kernel_entry, block_entry, dace.Memlet())
    state.add_nedge(block_entry, fill, dace.Memlet())
    state.add_edge(fill, '__out', scratch, None, dace.Memlet('tmp[2]'))
    state.add_edge(scratch, None, use, '__in', dace.Memlet('tmp[2]'))
    state.add_memlet_path(use,
                          block_exit,
                          kernel_exit,
                          state.add_write('a'),
                          src_conn='__out',
                          memlet=dace.Memlet('a[i, t]'))
    sdfg.validate()
    return sdfg, kernel_entry


def test_lifted_indices_follow_the_order_of_the_lifted_dimensions():
    """Every access to a lifted buffer must land inside it, in a slice no other iteration uses.

    The shape gains ``[KERNEL, BLOCK]`` in front, outermost map first, so an access inside both maps
    has to lead with ``[i, t]``. Built innermost first it led with ``[t, i]`` against those strides,
    which overruns the buffer as soon as the block is wider than the kernel: the softmax in
    ``warp_tiling_test`` wrote past its device allocation that way and faulted.

    Checked by evaluating the flat offset of each access for every ``(i, t)``, so a wrong order fails
    on the arithmetic it produces rather than on its spelling.
    """
    sdfg, kernel_entry = kernel_with_a_thread_block_around_its_scratch()
    MoveArrayOutOfKernel().apply_pass(sdfg, kernel_entry, 'tmp')

    desc = sdfg.arrays['tmp']
    assert [int(s) for s in desc.shape] == [KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT], desc.shape
    accesses = [
        e for e in sdfg.start_state.edges()
        if e.data.data == 'tmp' and (isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet))
    ]
    assert len(accesses) == 2, [str(e.data) for e in accesses]
    total = int(desc.total_size)
    for edge in accesses:
        begins = [begin for begin, _, _ in edge.data.subset.ndrange()]
        offsets = {
            sum(int(dace.symbolic.evaluate(b, {
                'i': i,
                't': t
            })) * int(s) for b, s in zip(begins, desc.strides))
            for i in range(KERNEL_EXTENT)
            for t in range(BLOCK_EXTENT)
        }
        assert len(offsets) == KERNEL_EXTENT * BLOCK_EXTENT, f'{edge.data}: two iterations share one slice'
        assert 0 <= min(offsets) and max(offsets) < total, \
            f'{edge.data}: offsets {min(offsets)}..{max(offsets)} leave the {total}-element buffer'


def test_the_lift_moves_one_slice_per_iteration_out_of_the_kernel():
    """Inside the kernel a lifted buffer leaves one iteration's slice at a time.

    The edges the lift adds from the in-kernel access node out through the map exits carried the
    whole buffer. Codegen lowers such an edge to a copy of the buffer onto itself in EVERY iteration,
    so each thread re-reads and re-writes the slices the other threads are filling: a race, and on
    the warp-tiled softmax 32 whole-buffer copies per block.
    """
    sdfg, kernel_entry = kernel_with_a_thread_block_around_its_scratch()
    MoveArrayOutOfKernel().apply_pass(sdfg, kernel_entry, 'tmp')

    state = sdfg.start_state
    moved = {
        e.dst.map.label: int(e.data.subset.num_elements())
        for e in state.edges() if isinstance(e.dst, dace.nodes.MapExit) and e.data.data == 'tmp'
    }
    assert moved == {'block': SCRATCH_EXTENT, 'kernel': BLOCK_EXTENT * SCRATCH_EXTENT}, moved


if __name__ == '__main__':
    test_lift_leaves_descendant_nested_sdfgs_at_their_own_rank()
    test_lifted_scratch_below_a_nested_sdfg_computes_the_right_values()
    test_lift_does_not_bind_a_name_the_nest_assigns_itself()
    test_lifted_kernel_declares_a_nested_loop_counter()
