# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MoveArrayOutOfKernel`."""
import ast
import re
import warnings

import numpy as np
import pytest
import sympy

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel, prepend_subscript_indices, tile_extent

NX, NZ = (dace.symbol(s, dtype=dace.int64) for s in ('NX', 'NZ'))
GLOBAL = dtypes.StorageType.GPU_Global


def test_tile_extent_recognises_min_pattern():
    """For a ``Min``-bounded inner-map extent, ``tile_extent`` returns the static tile width 32."""
    b_i = sympy.Symbol('b_i')
    N = sympy.Symbol('N')
    max_elem = sympy.Min(N - 1, b_i + 31)
    min_elem = b_i
    extent = tile_extent(max_elem, min_elem)
    assert extent == 32, f"expected 32, got {extent}"
    assert b_i not in extent.free_symbols, f"tile extent leaks outer-loop symbol: {extent.free_symbols}"


def test_tile_extent_falls_back_to_the_range_extent_without_a_min():
    """No ``Min`` in the upper bound (a plain or outer strided map): the extent is the symbolic ``N``."""
    N = sympy.Symbol('N')
    extent = tile_extent(N - 1, sympy.Integer(0))
    assert sympy.simplify(extent - N) == 0, f"expected N, got {extent}"


@pytest.mark.parametrize('ranges, shape, strides, expected_shape, expected_strides', [
    (dict(i='0:128', j='0:32'), [64], None, [128, 32, 64], [2048, 64, 1]),
    (dict(i='0:8'), [4, 16], [1, 4], [8, 4, 16], [64, 1, 4]),
])
def test_lifted_dimensions_are_prepended_slowest_varying_keeping_the_own_layout(ranges, shape, strides, expected_shape,
                                                                                expected_strides):
    """Map dimensions go in front as the slowest axes; a C or Fortran transient keeps its layout on its own axes."""
    state = dace.SDFG('move_array_strides').add_state('s')
    me, _ = state.add_map('kernel', ranges, schedule=dace.dtypes.ScheduleType.GPU_Device)
    arr = dace.data.Array(dace.float64, shape, strides=strides)

    new_shape, new_strides, new_total, _ = MoveArrayOutOfKernel().get_new_shape_info(arr, [me])

    assert [int(s) for s in new_shape] == expected_shape, new_shape
    assert [int(s) for s in new_strides] == expected_strides, new_strides
    assert int(new_total) == int(np.prod(expected_shape)), new_total


def test_get_new_shape_info_rejects_unsupported_layout():
    """Neither packed-C nor packed-Fortran: refuse rather than silently re-lay-out the array."""
    sdfg = dace.SDFG('move_array_strides_bad')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:8'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [4, 16], strides=[32, 2])
    with pytest.raises(NotImplementedError):
        MoveArrayOutOfKernel().get_new_shape_info(arr, [me])


def kernel_with_internal_transient() -> dace.SDFG:
    """``GPU_Device`` map holding a ``GPU_Global`` transient too large to demote to registers."""
    sdfg = dace.SDFG('flat_lift')
    sdfg.add_array('A', [128], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_transient('buf', [1024], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    me, mx = state.add_map('kernel', dict(i='0:128'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    buf = state.add_access('buf')
    produce = state.add_tasklet('produce', {}, {'o': None}, 'o = 1.0')
    state.add_edge(me, None, produce, None, dace.Memlet())
    state.add_edge(produce, 'o', buf, None, dace.Memlet('buf[0]'))
    consume = state.add_tasklet('consume', {'b': None}, {'o': None}, 'o = b')
    state.add_edge(buf, None, consume, 'b', dace.Memlet('buf[0]'))
    state.add_memlet_path(consume, mx, state.add_write('A'), src_conn='o', memlet=dace.Memlet('A[i]'))
    sdfg.validate()
    return sdfg


def transient_body() -> dace.SDFG:
    """Nested body writing ``a_out[0]`` through its own ``buf[1024]`` transient."""
    inner = dace.SDFG('inner')
    inner.add_array('a_out', [1], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_transient('buf', [1024], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('i', is_start_block=True)
    buf = inner_state.add_access('buf')
    produce = inner_state.add_tasklet('produce', {}, {'o': None}, 'o = 1.0')
    inner_state.add_edge(produce, 'o', buf, None, dace.Memlet('buf[0]'))
    consume = inner_state.add_tasklet('consume', {'b': None}, {'o': None}, 'o = b')
    inner_state.add_edge(buf, None, consume, 'b', dace.Memlet('buf[0]'))
    inner_state.add_edge(consume, 'o', inner_state.add_write('a_out'), None, dace.Memlet('a_out[0]'))
    return inner


def kernel_with_transient_behind_a_nested_sdfg() -> dace.SDFG:
    """The same transient, one nested-SDFG boundary below the kernel."""
    inner = transient_body()

    sdfg = dace.SDFG('nested_lift')
    sdfg.add_array('A', [128], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    me, mx = state.add_map('kernel', dict(i='0:128'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {}, {'a_out': None})
    state.add_edge(me, None, nsdfg, None, dace.Memlet())
    state.add_memlet_path(nsdfg, mx, state.add_write('A'), src_conn='a_out', memlet=dace.Memlet('A[i]'))
    sdfg.validate()
    return sdfg


def buf_scopes(sdfg: dace.SDFG) -> list:
    """Enclosing scope of every ``buf`` access node across the hierarchy."""
    return [
        state.scope_dict()[node] for sub in sdfg.all_sdfgs_recursive() for state in sub.states()
        for node in state.data_nodes() if node.data == 'buf'
    ]


def test_flat_transient_is_lifted_out_of_the_kernel():
    """The transient gains a dimension per kernel iteration and reaches an access node outside it."""
    sdfg = kernel_with_internal_transient()
    assert lift(sdfg) == 1

    assert tuple(sdfg.arrays['buf'].shape) == (128, 1024), sdfg.arrays['buf'].shape
    assert tuple(sdfg.arrays['buf'].strides) == (1024, 1), sdfg.arrays['buf'].strides
    assert sdfg.arrays['buf'].transient, 'the lifted array must still be allocated, not expected as input'
    # One access node stays inside the kernel writing its slice, one lands outside it.
    assert None in buf_scopes(sdfg), 'nothing carries the array out of the kernel'
    sdfg.validate()


def test_transient_behind_a_nested_sdfg_is_lifted_through_the_boundary():
    """The descriptor is lifted through the nested SDFG and becomes an outer-level transient."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    assert lift(sdfg) == 1

    assert 'buf' in sdfg.arrays, 'the descriptor never reached the kernel-owning SDFG'
    assert tuple(sdfg.arrays['buf'].shape) == (128, 1024), sdfg.arrays['buf'].shape
    assert sdfg.arrays['buf'].transient
    # Its inner counterpart is now a connector-bound argument rather than an allocation.
    inner = next(s for s in sdfg.all_sdfgs_recursive() if s is not sdfg)
    assert not inner.arrays['buf'].transient, 'the inner copy must be passed in, not allocated in-kernel'
    assert buf_scopes(sdfg) == [None, None], buf_scopes(sdfg)
    sdfg.validate()


def test_small_transient_is_demoted_to_registers_instead():
    """Under the element threshold the array becomes per-thread ``Register`` and is not lifted."""
    sdfg = kernel_with_internal_transient()
    sdfg.arrays['buf'].set_shape((8, ))
    assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1  # demoted: no lift warning

    assert sdfg.arrays['buf'].storage == dace.dtypes.StorageType.Register
    assert tuple(sdfg.arrays['buf'].shape) == (8, ), 'a demoted array keeps its own shape'


def test_a_small_persistent_transient_is_lifted_not_demoted():
    """A register cannot outlive the invocation a persistent array is kept across."""
    sdfg = kernel_with_internal_transient()
    sdfg.arrays['buf'].set_shape((8, ))
    sdfg.arrays['buf'].lifetime = dace.AllocationLifetime.Persistent
    with pytest.warns(UserWarning, match='will be lifted outside the kernel'):
        assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    assert sdfg.arrays['buf'].storage == dace.dtypes.StorageType.GPU_Global
    assert tuple(sdfg.arrays['buf'].shape) == (128, 8)
    sdfg.validate()


def test_lifted_transient_is_renamed_around_a_colliding_descriptor():
    """An unrelated outer descriptor already holds the name, so the lifted one takes a fresh one."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    sdfg.add_array('buf', [4], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    assert lift(sdfg) == 1

    assert tuple(sdfg.arrays['buf'].shape) == (4, ), 'the colliding outer descriptor was overwritten'
    lifted = [name for name, desc in sdfg.arrays.items() if name != 'buf' and tuple(desc.shape) == (128, 1024)]
    assert len(lifted) == 1, sorted(sdfg.arrays)
    assert sdfg.arrays[lifted[0]].transient
    sdfg.validate()


def lift(sdfg: dace.SDFG) -> int:
    """Run the pass with register demotion off, so every in-kernel transient is lifted."""
    sut = MoveArrayOutOfKernel()
    sut.register_demotion_max_elements = 0
    with pytest.warns(UserWarning, match='will be lifted outside the kernel'):
        return sut.apply_pass(sdfg, {})


def flat_offsets(desc: dace.data.Array, subset: dace.subsets.Range, values: dict) -> tuple[int, int]:
    strides = [int(dace.symbolic.evaluate(s, values)) for s in desc.strides]
    corners = (subset.min_element(), subset.max_element())
    return tuple(sum(int(dace.symbolic.evaluate(b, values)) * s for b, s in zip(c, strides)) for c in corners)


def test_a_kernel_starting_above_zero_indexes_its_slice_from_the_first_iteration():
    """The lifted dimension holds the trip count, so iteration ``i`` of ``1:127`` owns row ``i - 1``."""
    sdfg = kernel_with_internal_transient()
    kernel = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.MapEntry))
    kernel.map.range = dace.subsets.Range([(1, 126, 1)])
    lift(sdfg)

    desc = sdfg.arrays['buf']
    assert tuple(desc.shape) == (126, 1024), desc.shape
    total = int(desc.total_size)
    inner = [e for e in sdfg.start_state.edges() if e.data.data == 'buf' and isinstance(e.src, dace.nodes.Tasklet)]
    firsts = {flat_offsets(desc, inner[0].data.subset, {'i': i})[0] for i in range(1, 127)}
    assert min(firsts) == 0 and max(firsts) < total, (min(firsts), max(firsts), total)


KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT = 3, 5, 7


def kernel_with_a_thread_block_around_its_scratch() -> dace.SDFG:
    """``a[i, t] = tmp[2]`` with ``tmp[2] = i + 10 * t``, ``tmp`` inside a thread-block map ``t`` inside kernel ``i``."""
    sdfg = dace.SDFG('scratch_inside_a_thread_block')
    sdfg.add_array('a', [KERNEL_EXTENT, BLOCK_EXTENT], dace.float64, storage=GLOBAL)
    sdfg.add_array('tmp', [SCRATCH_EXTENT], dace.float64, transient=True, storage=GLOBAL)
    state = sdfg.add_state('grid', is_start_block=True)
    kernel_entry, kernel_exit = state.add_map('kernel',
                                              dict(i=f'0:{KERNEL_EXTENT}'),
                                              schedule=dtypes.ScheduleType.GPU_Device)
    block_entry, block_exit = state.add_map('block',
                                            dict(t=f'0:{BLOCK_EXTENT}'),
                                            schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    fill = state.add_tasklet('fill', {}, {'__out': None}, '__out = i + 10.0 * t')
    use = state.add_tasklet('use', {'__in': None}, {'__out': None}, '__out = __in')
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
    return sdfg


def test_lifted_indices_follow_the_order_of_the_lifted_dimensions():
    """Shape gains ``[KERNEL, BLOCK]`` outermost first, so every access leads with ``[i, t]`` and stays inside."""
    sdfg = kernel_with_a_thread_block_around_its_scratch()
    lift(sdfg)

    desc = sdfg.arrays['tmp']
    assert [int(s) for s in desc.shape] == [KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT], desc.shape
    accesses = [
        e for e in sdfg.start_state.edges()
        if e.data.data == 'tmp' and (isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet))
    ]
    assert len(accesses) == 2, [str(e.data) for e in accesses]
    for edge in accesses:
        offsets = {
            flat_offsets(desc, edge.data.subset, {
                'i': i,
                't': t
            })[0]
            for i in range(KERNEL_EXTENT)
            for t in range(BLOCK_EXTENT)
        }
        assert len(offsets) == KERNEL_EXTENT * BLOCK_EXTENT, f'{edge.data}: two iterations share one slice'
        assert 0 <= min(offsets) and max(offsets) < int(desc.total_size), (edge.data, min(offsets), max(offsets))


def test_the_lift_moves_one_slice_per_iteration_out_of_the_kernel():
    """A whole-buffer edge out of a map exit lowers to a copy racing every iteration."""
    sdfg = kernel_with_a_thread_block_around_its_scratch()
    lift(sdfg)

    moved = {
        e.dst.map.label: int(e.data.subset.num_elements())
        for e in sdfg.start_state.edges() if isinstance(e.dst, dace.nodes.MapExit) and e.data.data == 'tmp'
    }
    assert moved == {'block': SCRATCH_EXTENT, 'kernel': BLOCK_EXTENT * SCRATCH_EXTENT}, moved


@pytest.mark.gpu
def test_the_thread_block_scratch_computes_its_values():
    import cupy  # Only present on GPU runners.
    sdfg = kernel_with_a_thread_block_around_its_scratch()
    lift(sdfg)
    a = cupy.zeros((KERNEL_EXTENT, BLOCK_EXTENT))

    sdfg(a=a)

    expected = np.arange(KERNEL_EXTENT)[:, None] + 10.0 * np.arange(BLOCK_EXTENT)[None, :]
    assert np.array_equal(cupy.asnumpy(a), expected)


def kernel_with_a_locally_named_scratch_extent() -> dace.SDFG:
    """Transient extent ``M`` is local to the nested SDFG, bound to the outer ``NZ - 1``."""
    inner = dace.SDFG('inner_local_scratch')
    inner.add_symbol('M', dace.int64)
    inner.add_array('a_in', [NZ], dace.float64, storage=GLOBAL)
    inner.add_array('out_in', [NZ], dace.float64, storage=GLOBAL)
    inner.add_array('tmp', ['M'], dace.float64, transient=True, storage=GLOBAL)
    fill = inner.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('scale', {'m': '0:M'}, {'__in': dace.Memlet('a_in[m]')},
                            '__out = __in * 2.0', {'__out': dace.Memlet('tmp[m]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    drain = inner.add_state_after(fill, 'drain')
    drain.add_mapped_tasklet('shift', {'m': '0:M'}, {'__in': dace.Memlet('tmp[m]')},
                             '__out = __in + 1.0', {'__out': dace.Memlet('out_in[m]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)

    sdfg = dace.SDFG('locally_named_scratch_extent')
    sdfg.add_array('a', [NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array('out', [NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state('grid', is_start_block=True)
    kernel_entry, kernel_exit = state.add_map('kernel', dict(i='0:1'), schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {'a_in': None}, {'out_in': None}, symbol_mapping={'M': NZ - 1})
    state.add_memlet_path(state.add_read('a'), kernel_entry, nsdfg, dst_conn='a_in', memlet=dace.Memlet('a[0:NZ]'))
    state.add_memlet_path(nsdfg,
                          kernel_exit,
                          state.add_write('out'),
                          src_conn='out_in',
                          memlet=dace.Memlet('out[0:NZ]'))
    sdfg.validate()
    return sdfg


def test_lift_translates_a_locally_named_shape_symbol_through_symbol_mapping():
    """The lifted descriptor must say ``NZ - 1``, not the inner-only ``M``."""
    sdfg = kernel_with_a_locally_named_scratch_extent()
    lift(sdfg)

    desc = sdfg.arrays['tmp']
    assert sympy.simplify(desc.shape[-1] - (NZ - 1)) == 0, desc.shape
    assert 'M' not in sdfg.free_symbols
    assert 'M' not in sdfg.arglist()
    sdfg.validate()


def kernel_with_scratch_below_a_nested_sdfg() -> dace.SDFG:
    """``out[i, k] = 2 * a[i, NZ - 1 - k] + 1`` through ``tmp[NZ]`` defined in ``body`` and passed further down."""
    producer = dace.SDFG('producer')
    producer.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    producer.add_array('tmp', [NZ], dace.float64, storage=GLOBAL)
    producer.add_state('fill',
                       is_start_block=True).add_mapped_tasklet('scale', {'k': '0:NZ'}, {'__in': dace.Memlet('a[i, k]')},
                                                               '__out = __in * 2.0', {'__out': dace.Memlet('tmp[k]')},
                                                               schedule=dtypes.ScheduleType.Sequential,
                                                               external_edges=True)

    consumer = dace.SDFG('consumer')
    consumer.add_array('tmp', [NZ], dace.float64, storage=GLOBAL)
    consumer.add_array('out', [NX, NZ], dace.float64, storage=GLOBAL)
    consumer.add_state('drain', is_start_block=True).add_mapped_tasklet('shift', {'k': '0:NZ'},
                                                                        {'__in': dace.Memlet('tmp[NZ - 1 - k]')},
                                                                        '__out = __in + 1.0',
                                                                        {'__out': dace.Memlet('out[i, k]')},
                                                                        schedule=dtypes.ScheduleType.Sequential,
                                                                        external_edges=True)

    body = dace.SDFG('body')
    body.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    body.add_array('out', [NX, NZ], dace.float64, storage=GLOBAL)
    body.add_array('tmp', [NZ], dace.float64, transient=True, storage=GLOBAL)
    mapping = dict(i='i', NX=NX, NZ=NZ)
    fill = body.add_state('call_producer', is_start_block=True)
    pnode = fill.add_nested_sdfg(producer, {'a': None}, {'tmp': None}, symbol_mapping=mapping)
    fill.add_edge(fill.add_read('a'), None, pnode, 'a', dace.Memlet('a[0:NX, 0:NZ]'))
    fill.add_edge(pnode, 'tmp', fill.add_write('tmp'), None, dace.Memlet('tmp[0:NZ]'))
    drain = body.add_state_after(fill, 'call_consumer')
    cnode = drain.add_nested_sdfg(consumer, {'tmp': None}, {'out': None}, symbol_mapping=mapping)
    drain.add_edge(drain.add_read('tmp'), None, cnode, 'tmp', dace.Memlet('tmp[0:NZ]'))
    drain.add_edge(cnode, 'out', drain.add_write('out'), None, dace.Memlet('out[0:NX, 0:NZ]'))

    sdfg = dace.SDFG('scratch_below_a_nested_sdfg')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array('out', [NX, NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state('grid', is_start_block=True)
    entry, exit_node = state.add_map('kernel', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [32, 1, 1]
    nsdfg = state.add_nested_sdfg(body, {'a': None}, {'out': None}, symbol_mapping=mapping)
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    state.add_memlet_path(nsdfg,
                          exit_node,
                          state.add_write('out'),
                          src_conn='out',
                          memlet=dace.Memlet('out[0:NX, 0:NZ]'))
    sdfg.validate()
    return sdfg


def test_lift_leaves_descendant_nested_sdfgs_at_their_own_rank():
    """Only the definition and its ancestors are reshaped; a nest below keeps its rank and its memlets."""
    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    lift(sdfg)

    ranks = {}
    for nested in sdfg.all_sdfgs_recursive():
        if 'tmp' not in nested.arrays:
            continue
        rank = ranks[nested.name] = len(nested.arrays['tmp'].shape)
        for state in nested.all_states():
            for edge in state.edges():
                if edge.data.data == 'tmp':
                    assert edge.data.subset.dims() == rank, (nested.name, str(edge.data))
    assert ranks == {'scratch_below_a_nested_sdfg': 2, 'body': 2, 'producer': 1, 'consumer': 1}, ranks
    sdfg.validate()


@pytest.mark.gpu
def test_lifted_scratch_below_a_nested_sdfg_computes_the_right_values():
    import cupy  # Only present on GPU runners.
    nx, nz = 5, 7
    host_a = np.random.default_rng(0).random((nx, nz))
    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    lift(sdfg)
    out = cupy.zeros((nx, nz))

    sdfg(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), host_a[:, ::-1] * 2.0 + 1.0)


def wrap_kernel_around_a_body_it_never_reaches() -> dace.SDFG:
    """``out[k] = 2 * a[NZ - 1 - k] + 1`` in a body under a size-1 kernel ``w`` the body never names."""
    body = dace.SDFG('wrapped_body')
    body.add_array('a', [NZ], dace.float64, storage=GLOBAL)
    body.add_array('out', [NZ], dace.float64, storage=GLOBAL)
    body.add_array('tmp', [NZ], dace.float64, transient=True, storage=GLOBAL)
    fill = body.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('scale', {'k': '0:NZ'}, {'__in': dace.Memlet('a[k]')},
                            '__out = __in * 2.0', {'__out': dace.Memlet('tmp[k]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    drain = body.add_state_after(fill, 'drain')
    drain.add_mapped_tasklet('shift', {'k': '0:NZ'}, {'__in': dace.Memlet('tmp[NZ - 1 - k]')},
                             '__out = __in + 1.0', {'__out': dace.Memlet('out[k]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)

    sdfg = dace.SDFG('wrap_kernel_scratch')
    sdfg.add_array('a', [NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array('out', [NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state('grid', is_start_block=True)
    entry, exit_node = state.add_map('wrap', dict(w='0:1'), schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(body, {'a': None}, {'out': None}, symbol_mapping=dict(NZ=NZ))
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NZ]'))
    state.add_memlet_path(nsdfg, exit_node, state.add_write('out'), src_conn='out', memlet=dace.Memlet('out[0:NZ]'))
    sdfg.validate()
    return sdfg


def test_a_lift_binds_the_kernel_parameter_it_indexes_by():
    """The new slice index is ``w``, which the body never used, so the nest must be bound to it."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    lift(sdfg)

    nest = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert str(nest.symbol_mapping.get('w')) == 'w', nest.symbol_mapping
    sdfg.validate()


def kernel_over_k_beside_a_nested_k_loop() -> dace.SDFG:
    """Kernel ``k`` lifts a symbolic scratch; a second kernel's nest owns a ``k`` loop of its own."""
    scratch_body = dace.SDFG('scratch_body')
    scratch_body.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    scratch_body.add_array('mid', [NX, NZ], dace.float64, storage=GLOBAL)
    scratch_body.add_array('tmp', [NX], dace.float64, transient=True, storage=GLOBAL)
    fill = scratch_body.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('scale', {'i': '0:NX'}, {'__in': dace.Memlet('a[i, k]')},
                            '__out = __in * 2.0', {'__out': dace.Memlet('tmp[i]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    drain = scratch_body.add_state_after(fill, 'drain')
    drain.add_mapped_tasklet('shift', {'i': '0:NX'}, {'__in': dace.Memlet('tmp[NX - 1 - i]')},
                             '__out = __in', {'__out': dace.Memlet('mid[i, k]')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)

    sweep = dace.SDFG('sweep')
    sweep.add_array('mid', [NX, NZ], dace.float64, storage=GLOBAL)
    sweep.add_array('out', [NX, NZ], dace.float64, storage=GLOBAL)
    loop = LoopRegion('k_sweep',
                      condition_expr='k < NZ',
                      loop_var='k',
                      initialize_expr='k = 0',
                      update_expr='k = k + 1')
    sweep.add_node(loop, is_start_block=True)
    step = loop.add_state('step', is_start_block=True)
    bump = step.add_tasklet('bump', {'__in': None}, {'__out': None}, '__out = __in + 1.0')
    step.add_edge(step.add_read('mid'), None, bump, '__in', dace.Memlet('mid[i, k]'))
    step.add_edge(bump, '__out', step.add_write('out'), None, dace.Memlet('out[i, k]'))

    sdfg = dace.SDFG('kernel_over_k_beside_a_nested_k_loop')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array('mid', [NX, NZ], dace.float64, transient=True, storage=GLOBAL)
    sdfg.add_array('out', [NX, NZ], dace.float64, storage=GLOBAL)

    scratch = sdfg.add_state('scratch', is_start_block=True)
    entry, exit_node = scratch.add_map('kernel_k', dict(k='0:NZ'), schedule=dtypes.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [32, 1, 1]
    snode = scratch.add_nested_sdfg(scratch_body, {'a': None}, {'mid': None}, symbol_mapping=dict(k='k', NX=NX, NZ=NZ))
    scratch.add_memlet_path(scratch.add_read('a'), entry, snode, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    scratch.add_memlet_path(snode,
                            exit_node,
                            scratch.add_write('mid'),
                            src_conn='mid',
                            memlet=dace.Memlet('mid[0:NX, 0:NZ]'))

    sweep_state = sdfg.add_state_after(scratch, 'sweep')
    sentry, sexit = sweep_state.add_map('kernel_i', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    sentry.map.gpu_block_size = [32, 1, 1]
    wnode = sweep_state.add_nested_sdfg(sweep, {'mid': None}, {'out': None}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    sweep_state.add_memlet_path(sweep_state.add_read('mid'),
                                sentry,
                                wnode,
                                dst_conn='mid',
                                memlet=dace.Memlet('mid[0:NX, 0:NZ]'))
    sweep_state.add_memlet_path(wnode,
                                sexit,
                                sweep_state.add_write('out'),
                                src_conn='out',
                                memlet=dace.Memlet('out[0:NX, 0:NZ]'))
    sdfg.validate()
    return sdfg


def test_lift_does_not_bind_a_name_the_nest_assigns_itself():
    """A nest owning a ``k`` loop must not be bound to the kernel's ``k``: codegen then declares neither."""
    sdfg = kernel_over_k_beside_a_nested_k_loop()
    lift(sdfg)

    sweep = next(n for n, _ in sdfg.all_nodes_recursive()
                 if isinstance(n, dace.nodes.NestedSDFG) and n.sdfg.name == 'sweep')
    assert 'k' not in sweep.symbol_mapping, sweep.symbol_mapping
    sdfg.validate()
    code = ''.join(obj.clean_code for obj in sdfg.generate_code())
    assert re.search(r'\w+ k;', code), 'the loop counter is used without a declaration in scope'


def kernel_with_interstate_buffer_read() -> dace.SDFG:
    """``out[i] = a[i, order[0]]``: an interstate edge reads one element of a kernel-local buffer."""
    inner = dace.SDFG('pick_body')
    inner.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    inner.add_array('out', [NX], dace.float64, storage=GLOBAL)
    inner.add_array('order', [NZ], dace.int64, transient=True, storage=GLOBAL)
    inner.add_symbol('sel', dace.int64)
    fill = inner.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('rank', {'k': '0:NZ'}, {'__in': dace.Memlet('a[i, k]')},
                            '__out = (NZ - 1 - k) if (__in > 0.5) else k', {'__out': dace.Memlet('order[k]')},
                            schedule=dtypes.ScheduleType.Sequential,
                            external_edges=True)
    use = inner.add_state('use')
    inner.add_edge(fill, use, dace.InterstateEdge(assignments={'sel': 'order[0]'}))
    pick = use.add_tasklet('pick', {'__in': None}, {'__out': None}, '__out = __in')
    use.add_edge(use.add_read('a'), None, pick, '__in', dace.Memlet('a[i, sel]'))
    use.add_edge(pick, '__out', use.add_write('out'), None, dace.Memlet('out[i]'))

    sdfg = dace.SDFG('kernel_with_interstate_buffer_read')
    sdfg.add_array('a', [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array('out', [NX], dace.float64, storage=GLOBAL)
    state = sdfg.add_state('body', is_start_block=True)
    entry, exit_node = state.add_map('grid', dict(i='0:NX'), schedule=dtypes.ScheduleType.GPU_Device)
    entry.map.gpu_block_size = [32, 1, 1]
    nsdfg = state.add_nested_sdfg(inner, {'a': None}, {'out': None}, symbol_mapping=dict(i='i', NX=NX, NZ=NZ))
    state.add_memlet_path(state.add_read('a'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('a[0:NX, 0:NZ]'))
    state.add_memlet_path(nsdfg, exit_node, state.add_write('out'), src_conn='out', memlet=dace.Memlet('out[0:NX]'))
    sdfg.validate()
    return sdfg


def test_an_interstate_read_of_a_lifted_buffer_gains_the_kernel_index():
    """``order[0]`` on an interstate edge must keep naming one element of the now rank-2 buffer."""
    sdfg = kernel_with_interstate_buffer_read()
    lift(sdfg)

    body = next(s for s in sdfg.all_sdfgs_recursive() if s.name == 'pick_body')
    assert len(body.arrays['order'].shape) == 2, body.arrays['order'].shape
    reads = [
        node for edge in body.all_interstate_edges() for value in edge.data.assignments.values()
        for node in ast.walk(ast.parse(str(value))) if isinstance(node, ast.Subscript)
    ]
    assert [ast.unparse(r) for r in reads] == ['order[i, 0]'], [ast.unparse(r) for r in reads]
    sdfg.generate_code()


def test_a_symbol_mapping_read_of_a_lifted_buffer_gains_the_kernel_index():
    """A nest bound to ``order[1]`` must keep reading one element of the now rank-2 buffer."""
    sdfg = kernel_with_interstate_buffer_read()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    leaf = dace.SDFG('leaf')
    leaf.add_symbol('s', dace.int64)
    leaf.add_array('o', [1], dace.float64, storage=GLOBAL)
    leaf_state = leaf.add_state()
    write = leaf_state.add_tasklet('write', {}, {'x': None}, 'x = s')
    leaf_state.add_edge(write, 'x', leaf_state.add_write('o'), None, dace.Memlet('o[0]'))
    body.add_array('picked', [1], dace.float64, transient=True, storage=dtypes.StorageType.Register)
    use = next(s for s in body.states() if s.label == 'use')
    nest = use.add_nested_sdfg(leaf, {}, {'o': None}, symbol_mapping={'s': 'order[1]'})
    use.add_edge(nest, 'o', use.add_write('picked'), None, dace.Memlet('picked[0]'))

    lift(sdfg)

    assert str(nest.symbol_mapping['s']) == 'order[i, 1]', nest.symbol_mapping
    sdfg.validate()


@pytest.mark.gpu
def test_an_interstate_read_of_a_lifted_buffer_computes_the_right_values():
    import cupy  # Only present on GPU runners.
    nx, nz = 5, 7
    host_a = np.random.default_rng(0).random((nx, nz))
    first = np.where(host_a[:, 0] > 0.5, nz - 1, 0)
    sdfg = kernel_with_interstate_buffer_read()
    lift(sdfg)
    out = cupy.zeros(nx)

    sdfg(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), host_a[np.arange(nx), first])


def test_two_nests_defining_the_same_name_are_each_lifted_once():
    """Two sibling definitions of ``buf`` in one kernel are two arrays, each gaining one kernel dimension."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    state = sdfg.start_state
    first = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    sdfg.add_array('B', [128], dace.float64, storage=GLOBAL)
    second = state.add_nested_sdfg(transient_body(), {}, {'a_out': None})
    kernel = state.entry_node(first)
    state.add_edge(kernel, None, second, None, dace.Memlet())
    state.add_memlet_path(second,
                          state.exit_node(kernel),
                          state.add_write('B'),
                          src_conn='a_out',
                          memlet=dace.Memlet('B[i]'))
    sdfg.validate()

    lift(sdfg)

    lifted = sorted(tuple(desc.shape) for name, desc in sdfg.arrays.items() if name not in ('A', 'B'))
    assert lifted == [(128, 1024), (128, 1024)], {n: d.shape for n, d in sdfg.arrays.items()}
    sdfg.validate()


def test_a_renamed_lift_leaves_a_descendant_its_own_descriptor():
    """The owner is renamed around an outer ``tmp``; the nests below keep their own rank-1 ``tmp``."""
    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    sdfg.add_array('tmp', [4], dace.float64, storage=GLOBAL)
    lift(sdfg)

    assert tuple(sdfg.arrays['tmp'].shape) == (4, ), 'the colliding outer descriptor was overwritten'
    for nested in sdfg.all_sdfgs_recursive():
        if nested.name in ('producer', 'consumer'):
            assert tuple(nested.arrays['tmp'].shape) == (NZ, ), (nested.name, nested.arrays['tmp'].shape)
    sdfg.validate()


def test_a_transient_inside_a_loop_of_the_nested_body_is_lifted():
    """A body state inside a loop region still sits in the kernel, so its transient is lifted."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    inner = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    body = inner.start_block
    loop = LoopRegion('repeat', 'r < 2', 'r', 'r = 0', 'r = r + 1')
    inner.remove_node(body)
    loop.add_node(body, is_start_block=True)
    inner.add_node(loop, is_start_block=True)
    sdfg.validate()

    assert lift(sdfg) == 1

    assert tuple(sdfg.arrays['buf'].shape) == (128, 1024), sdfg.arrays['buf'].shape
    sdfg.validate()


def test_a_nest_giving_the_kernel_parameter_its_own_meaning_keeps_its_transient():
    """Inside a loop over ``w`` the lifted index would name the loop counter, so the transient is left in place."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    loop = LoopRegion('again', 'w < 1', 'w', 'w = 0', 'w = w + 1')
    fill, drain = body.start_block, body.sink_nodes()[0]
    body.remove_nodes_from([fill, drain])
    loop.add_node(fill, is_start_block=True)
    loop.add_node(drain)
    loop.add_edge(fill, drain, dace.InterstateEdge())
    body.add_node(loop, is_start_block=True)
    sdfg.validate()

    sut = MoveArrayOutOfKernel()
    sut.register_demotion_max_elements = 0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert sut.apply_pass(sdfg, {}) is None

    assert 'tmp' not in sdfg.arrays
    assert tuple(body.arrays['tmp'].shape) == (NZ, ) and body.arrays['tmp'].transient
    nest = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert 'tmp' not in nest.out_connectors and 'w' not in nest.symbol_mapping


def test_a_transient_shared_by_two_kernels_is_refused():
    """One allocation used by two kernels has no single per-iteration slicing."""
    sdfg = kernel_with_internal_transient()
    second = sdfg.add_state_after(sdfg.start_state, 'again')
    me, mx = second.add_map('kernel2', dict(i='0:128'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    produce = second.add_tasklet('produce', {}, {'o': None}, 'o = 2.0')
    second.add_edge(me, None, produce, None, dace.Memlet())
    buf = second.add_access('buf')
    second.add_edge(produce, 'o', buf, None, dace.Memlet('buf[1]'))
    second.add_edge(buf, None, mx, None, dace.Memlet())
    sdfg.validate()

    with pytest.raises(NotImplementedError, match='shared by the kernels'):
        MoveArrayOutOfKernel().apply_pass(sdfg, {})


def test_control_flow_reading_a_per_thread_buffer_is_refused():
    """Written inside a thread-block map of its own SDFG, the buffer has no one element for an interstate edge."""
    sdfg = kernel_with_interstate_buffer_read()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    fill = body.start_block
    fill.remove_nodes_from(list(fill.nodes()))
    me, mx = fill.add_map('lanes', dict(k='0:NZ'), schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    write = fill.add_tasklet('write', {}, {'__out': None}, '__out = k')
    order = fill.add_access('order')
    fill.add_edge(me, None, write, None, dace.Memlet())
    fill.add_edge(write, '__out', order, None, dace.Memlet('order[k]'))
    fill.add_edge(order, None, mx, None, dace.Memlet())
    sdfg.validate()

    sut = MoveArrayOutOfKernel()
    sut.register_demotion_max_elements = 0
    with pytest.warns(UserWarning, match='will be lifted'):
        with pytest.raises(NotImplementedError, match='varies per GPU thread'):
            sut.apply_pass(sdfg, {})


def test_code_mentioning_the_name_without_subscripting_it_is_returned_verbatim():
    """A name that only appears as a substring must not count as a rewrite, not even through reformatting."""
    code = 'x+1 if tmp_flag else 0'
    assert prepend_subscript_indices(code, 'tmp', ['i']) is code
    assert prepend_subscript_indices('tmp[0]+1', 'tmp', ['i']) == 'tmp[i, 0] + 1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
