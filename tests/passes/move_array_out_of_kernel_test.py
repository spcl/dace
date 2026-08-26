# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that ``tile_extent`` returns the static tile width for a tiled inner-map extent so the
lifted transient's shape does not leak an out-of-scope outer-loop symbol into ``cudaMalloc``."""
import pytest
import sympy

import dace
from dace.transformation.passes.move_array_out_of_kernel import tile_extent, MoveArrayOutOfKernel


def test_tile_extent_recognises_min_pattern():
    """For a ``Min``-bounded inner-map extent, ``tile_extent`` returns the static tile width 32."""
    b_i = sympy.Symbol('b_i')
    N = sympy.Symbol('N')
    max_elem = sympy.Min(N - 1, b_i + 31)
    min_elem = b_i
    extent = tile_extent(max_elem, min_elem)
    assert extent == 32, f"expected 32, got {extent}"
    assert b_i not in extent.free_symbols, f"tile extent leaks outer-loop symbol: {extent.free_symbols}"


def test_tile_extent_falls_back_for_plain_range():
    """No ``Min`` in the upper bound: the symbolic extent is returned unchanged."""
    W = sympy.Symbol('W')
    extent = tile_extent(W - 1, sympy.Integer(0))
    assert sympy.simplify(extent - W) == 0, f"expected W, got {extent}"


def test_tile_extent_handles_outer_block_strided_loop():
    """Outer strided GPU_Device map ``b_i = 0:N:32``: the fallback returns the host-visible ``N``."""
    N = sympy.Symbol('N')
    # max_element() of a strided range comes back as ``N - 1``; pin that and check there is no leak.
    extent = tile_extent(N - 1, sympy.Integer(0))
    assert sympy.simplify(extent - N) == 0
    assert sympy.Symbol('b_i') not in extent.free_symbols


def test_get_new_shape_info_multidim_prepend_strides():
    """Lifting a ``[64]`` C-packed transient out of ``map[0:128, 0:32]`` gives ``[128, 32, 64]``
    with packed C strides ``[2048, 64, 1]``."""
    sdfg = dace.SDFG('move_array_strides')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:128', j='0:32'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [64])
    new_shape, new_strides, new_total, _new_offsets = MoveArrayOutOfKernel().get_new_shape_info(arr, [me])

    assert [int(s) for s in new_shape] == [128, 32, 64], new_shape
    assert [int(s) for s in new_strides] == [2048, 64, 1], new_strides
    assert int(new_total) == 128 * 32 * 64, new_total


def test_get_new_shape_info_keeps_fortran_layout():
    """A packed-Fortran transient keeps that layout on its own axes; the prepended map dimensions
    become the slowest-varying ones regardless."""
    sdfg = dace.SDFG('move_array_strides_f')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:8'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [4, 16], strides=[1, 4])
    assert arr.is_packed_fortran_strides()
    new_shape, new_strides, new_total, _new_offsets = MoveArrayOutOfKernel().get_new_shape_info(arr, [me])

    assert [int(s) for s in new_shape] == [8, 4, 16], new_shape
    assert [int(s) for s in new_strides] == [64, 1, 4], new_strides
    assert int(new_total) == 8 * 4 * 16, new_total


def test_get_new_shape_info_rejects_unsupported_layout():
    """Neither packed-C nor packed-Fortran: refuse rather than silently re-lay-out the array."""
    sdfg = dace.SDFG('move_array_strides_bad')
    state = sdfg.add_state('s')
    me, _mx = state.add_map('kernel', dict(i='0:8'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    arr = dace.data.Array(dace.float64, [4, 16], strides=[32, 2])
    with pytest.raises(NotImplementedError):
        MoveArrayOutOfKernel().get_new_shape_info(arr, [me])


if __name__ == '__main__':
    test_tile_extent_recognises_min_pattern()
    test_tile_extent_falls_back_for_plain_range()
    test_tile_extent_handles_outer_block_strided_loop()
    test_get_new_shape_info_multidim_prepend_strides()
    test_get_new_shape_info_keeps_fortran_layout()
    test_get_new_shape_info_rejects_unsupported_layout()


def _kernel_with_internal_transient() -> dace.SDFG:
    """``GPU_Device`` map holding a ``GPU_Global`` transient too large to demote to registers."""
    sdfg = dace.SDFG('flat_lift')
    sdfg.add_array('A', [128], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_transient('buf', [1024], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    me, mx = state.add_map('kernel', dict(i='0:128'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    buf = state.add_access('buf')
    produce = state.add_tasklet('produce', {}, {'o'}, 'o = 1.0')
    state.add_edge(me, None, produce, None, dace.Memlet())
    state.add_edge(produce, 'o', buf, None, dace.Memlet('buf[0]'))
    consume = state.add_tasklet('consume', {'b'}, {'o'}, 'o = b')
    state.add_edge(buf, None, consume, 'b', dace.Memlet('buf[0]'))
    state.add_memlet_path(consume, mx, state.add_write('A'), src_conn='o', memlet=dace.Memlet('A[i]'))
    sdfg.validate()
    return sdfg


def _kernel_with_transient_behind_a_nested_sdfg() -> dace.SDFG:
    """The same transient, one nested-SDFG boundary below the kernel."""
    inner = dace.SDFG('inner')
    inner.add_array('a_out', [1], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_transient('buf', [1024], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('i', is_start_block=True)
    buf = inner_state.add_access('buf')
    produce = inner_state.add_tasklet('produce', {}, {'o'}, 'o = 1.0')
    inner_state.add_edge(produce, 'o', buf, None, dace.Memlet('buf[0]'))
    consume = inner_state.add_tasklet('consume', {'b'}, {'o'}, 'o = b')
    inner_state.add_edge(buf, None, consume, 'b', dace.Memlet('buf[0]'))
    inner_state.add_edge(consume, 'o', inner_state.add_write('a_out'), None, dace.Memlet('a_out[0]'))

    sdfg = dace.SDFG('nested_lift')
    sdfg.add_array('A', [128], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    me, mx = state.add_map('kernel', dict(i='0:128'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {}, {'a_out': None})
    state.add_edge(me, None, nsdfg, None, dace.Memlet())
    state.add_memlet_path(nsdfg, mx, state.add_write('A'), src_conn='a_out', memlet=dace.Memlet('A[i]'))
    sdfg.validate()
    return sdfg


def _buf_scopes(sdfg: dace.SDFG) -> list:
    """Enclosing scope of every ``buf`` access node across the hierarchy."""
    return [
        state.scope_dict()[node] for sub in sdfg.all_sdfgs_recursive() for state in sub.states()
        for node in state.data_nodes() if node.data == 'buf'
    ]


def test_flat_transient_is_lifted_out_of_the_kernel():
    """The transient gains a dimension per kernel iteration and reaches an access node outside it."""
    sdfg = _kernel_with_internal_transient()
    assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    assert tuple(sdfg.arrays['buf'].shape) == (128, 1024), sdfg.arrays['buf'].shape
    assert tuple(sdfg.arrays['buf'].strides) == (1024, 1), sdfg.arrays['buf'].strides
    assert sdfg.arrays['buf'].transient, 'the lifted array must still be allocated, not expected as input'
    # One access node stays inside the kernel writing its slice, one lands outside it.
    assert None in _buf_scopes(sdfg), 'nothing carries the array out of the kernel'
    sdfg.validate()


def test_transient_behind_a_nested_sdfg_is_lifted_through_the_boundary():
    """The descriptor is lifted through the nested SDFG and becomes an outer-level transient."""
    sdfg = _kernel_with_transient_behind_a_nested_sdfg()
    assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    assert 'buf' in sdfg.arrays, 'the descriptor never reached the kernel-owning SDFG'
    assert tuple(sdfg.arrays['buf'].shape) == (128, 1024), sdfg.arrays['buf'].shape
    assert sdfg.arrays['buf'].transient
    # Its inner counterpart is now a connector-bound argument rather than an allocation.
    inner = next(s for s in sdfg.all_sdfgs_recursive() if s is not sdfg)
    assert not inner.arrays['buf'].transient, 'the inner copy must be passed in, not allocated in-kernel'
    assert _buf_scopes(sdfg) == [None, None], _buf_scopes(sdfg)
    sdfg.validate()


def test_small_transient_is_demoted_to_registers_instead():
    """Under the element threshold the array becomes per-thread ``Register`` and is not lifted."""
    sdfg = _kernel_with_internal_transient()
    sdfg.arrays['buf'].set_shape((8, ))
    assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    assert sdfg.arrays['buf'].storage == dace.dtypes.StorageType.Register
    assert tuple(sdfg.arrays['buf'].shape) == (8, ), 'a demoted array keeps its own shape'
