# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``_tile_extent`` returns the static tile width for a tiled inner-map extent so the
lifted transient's shape does not leak an out-of-scope outer-loop symbol into ``cudaMalloc``."""
import numpy as np
import pytest
import sympy

import dace
from dace import dtypes
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


if __name__ == '__main__':
    test_lift_leaves_descendant_nested_sdfgs_at_their_own_rank()
    test_lifted_scratch_below_a_nested_sdfg_computes_the_right_values()
