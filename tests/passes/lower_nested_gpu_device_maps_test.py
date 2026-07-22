# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`NestedGPUDeviceMapLowering`.

The pass rewrites the ``GPU_Device`` nested inside ``GPU_Device`` pattern that the
experimental CUDA codegen explicitly refuses (``Dynamic parallelism ... not supported``):
the outer kernel's iteration range is union-expanded with the inner kernels' params, each
inner kernel's body is moved into a ``NestedSDFG`` guarded by an if-bound-check, and the
inner ``GPU_Device`` map itself is removed. The result is a single flat ``GPU_Device``
kernel whose body uses if-guards to fan out to each original inner kernel's range.
"""
import dace
import pytest

from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering


def _build_outer_with_two_sibling_inner_gpu_kernels() -> dace.SDFG:
    """``vertical_loop (0:K)`` (GPU_Device) wrapping a NestedSDFG that holds two sibling
    ``horizontal_loop`` ``GPU_Device`` maps of ranges ``(0:J+1, 0:I)`` and ``(0:J, 0:I)``.

    Mirrors the ICON ``native_functions_main`` reproducer in miniature.
    """
    K = dace.symbol('K', dtype=dace.int32)
    J = dace.symbol('J', dtype=dace.int32)
    I = dace.symbol('I', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_gpu_maps')
    sdfg.add_array('A', [K, J + 1, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [K, J, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('a_in', [J + 1, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_out', [J, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    # Inner kernel 1: writes a_in slice via WCR-free overwrite (range J+1, I).
    me_a, mx_a = inner_state.add_map('horizontal_loop_a',
                                     dict(__j='0:J + 1', __i='0:I'),
                                     schedule=dace.dtypes.ScheduleType.GPU_Device)
    t_a = inner_state.add_tasklet('write_a', {}, {'_a': dace.float64}, '_a = 1.0')
    inner_state.add_memlet_path(me_a, t_a, memlet=dace.Memlet())
    inner_state.add_memlet_path(t_a,
                                mx_a,
                                inner_state.add_write('a_in'),
                                src_conn='_a',
                                memlet=dace.Memlet('a_in[__j, __i]'))

    # Inner kernel 2: writes b_out (range J, I).
    me_b, mx_b = inner_state.add_map('horizontal_loop_b',
                                     dict(__j='0:J', __i='0:I'),
                                     schedule=dace.dtypes.ScheduleType.GPU_Device)
    t_b = inner_state.add_tasklet('write_b', {}, {'_b': dace.float64}, '_b = 2.0')
    inner_state.add_memlet_path(me_b, t_b, memlet=dace.Memlet())
    inner_state.add_memlet_path(t_b,
                                mx_b,
                                inner_state.add_write('b_out'),
                                src_conn='_b',
                                memlet=dace.Memlet('b_out[__j, __i]'))

    nsdfg = state.add_nested_sdfg(inner, set(), {'a_in', 'b_out'}, symbol_mapping={'__k': '__k'})
    a_write = state.add_write('A')
    b_write = state.add_write('B')
    state.add_memlet_path(outer_me, nsdfg, memlet=dace.Memlet())
    state.add_memlet_path(nsdfg, outer_mx, a_write, src_conn='a_in', memlet=dace.Memlet('A[__k, 0:J + 1, 0:I]'))
    state.add_memlet_path(nsdfg, outer_mx, b_write, src_conn='b_out', memlet=dace.Memlet('B[__k, 0:J, 0:I]'))
    return sdfg


def _count_gpu_device_maps(sdfg: dace.SDFG) -> tuple[int, int]:
    """Return ``(top_level, inside_nsdfgs)`` counts of ``GPU_Device`` ``MapEntry``s.

    Top-level counts the outer-state maps. Inside-NSDFG counts maps within any
    ``NestedSDFG`` in the SDFG hierarchy.
    """
    top = sum(1 for state in sdfg.states() for n in state.nodes()
              if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    inner = 0
    for s in sdfg.all_sdfgs_recursive():
        if s is sdfg:
            continue
        inner += sum(1 for state in s.states() for n in state.nodes()
                     if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    return top, inner


def test_pass_flattens_nested_gpu_kernels_validates_clean():
    """After the pass: outer ``GPU_Device`` map gains the inner kernels' params; inner
    ``GPU_Device`` maps disappear (their bodies live in if-bound-checked NSDFGs); the SDFG
    validates."""
    sdfg = _build_outer_with_two_sibling_inner_gpu_kernels()

    top_before, inner_before = _count_gpu_device_maps(sdfg)
    assert (top_before, inner_before) == (1, 2), (top_before, inner_before)

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    top_after, inner_after = _count_gpu_device_maps(sdfg)
    assert (top_after, inner_after) == (1, 0), (top_after, inner_after)

    # The outer map now carries the inner kernels' iteration params.
    outer = next(n for state in sdfg.states() for n in state.nodes()
                 if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    assert set(outer.map.params) >= {'__j', '__i'}, outer.map.params

    sdfg.validate()


def _build_nested_kernel_with_internal_inout_node() -> dace.SDFG:
    """Outer ``GPU_Device`` kernel wrapping a NestedSDFG whose inner ``GPU_Device`` map
    has a *non-transient* array as an internal inout ``AccessNode`` (written then read
    inside the kernel body: ``c_read -> map -> t1 -> c_mid -> t2 -> map -> c_write``).

    This is the shape that drives the inout-collection branch of ``_move_map_to_if``.
    """
    K = dace.symbol('K', dtype=dace.int32)
    J = dace.symbol('J', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_inout')
    sdfg.add_array('C', [K, J], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('c_io', [J], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    me, mx = inner_state.add_map('horizontal_loop', dict(__j='0:J'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    c_read = inner_state.add_read('c_io')
    c_mid = inner_state.add_access('c_io')  # internal inout node (non-transient)
    c_write = inner_state.add_write('c_io')
    t1 = inner_state.add_tasklet('bump', {'i'}, {'o'}, 'o = i + 1.0')
    t2 = inner_state.add_tasklet('bump2', {'i'}, {'o'}, 'o = i + 2.0')
    inner_state.add_memlet_path(c_read, me, t1, dst_conn='i', memlet=dace.Memlet('c_io[__j]'))
    inner_state.add_edge(t1, 'o', c_mid, None, dace.Memlet('c_io[__j]'))
    inner_state.add_edge(c_mid, None, t2, 'i', dace.Memlet('c_io[__j]'))
    inner_state.add_memlet_path(t2, mx, c_write, src_conn='o', memlet=dace.Memlet('c_io[__j]'))

    nsdfg = state.add_nested_sdfg(inner, {'c_io'}, {'c_io'}, symbol_mapping={'__k': '__k'})
    c_outer_read = state.add_read('C')
    c_outer_write = state.add_write('C')
    state.add_memlet_path(c_outer_read, outer_me, nsdfg, dst_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    state.add_memlet_path(nsdfg, outer_mx, c_outer_write, src_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    return sdfg


def test_inner_kernel_with_internal_inout_node_lowers_clean():
    """A non-transient array that is an internal inout ``AccessNode`` is collected by
    data name (a ``str``), not by the ``AccessNode`` object -- otherwise the pass leaks a
    node into the NestedSDFG connector set and raises ``KeyError`` on ``sdfg.arrays[node]``."""
    sdfg = _build_nested_kernel_with_internal_inout_node()

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    top, inner = _count_gpu_device_maps(sdfg)
    assert (top, inner) == (1, 0), (top, inner)
    # Every NestedSDFG connector must be a real (str) array name, never an AccessNode.
    for s in sdfg.all_sdfgs_recursive():
        for cf_state in s.states():
            for n in cf_state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    for conn in list(n.in_connectors) + list(n.out_connectors):
                        assert isinstance(conn, str), conn
    sdfg.validate()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
