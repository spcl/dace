# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``LiftSharedOutOfNestedSDFG``, which promotes ``GPU_Shared`` transients out of
NSDFGs inside a ``GPU_Device`` map and wires them to the kernel ``MapEntry`` / ``MapExit`` so the
allocator pins their ``__shared__`` declaration to the kernel scope. Topology asserted directly."""

import dace
from dace import SDFG, dtypes, nodes
from dace.memlet import Memlet
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG


def _build_inner_sdfg_with_shared(name: str, mode: str) -> SDFG:
    """Build a NestedSDFG with one Shared transient, used in the requested
    ``mode``: ``'read'`` (read only), ``'write'`` (write only), ``'both'``
    (read and written), or ``'none'`` (declared but never accessed).
    """
    inner = SDFG(name)
    inner.add_array('shared_arr', [4], dace.float32, storage=dtypes.StorageType.GPU_Shared, transient=True)
    inner.add_array('host_in', [4], dace.float32, storage=dtypes.StorageType.GPU_Global)
    inner.add_array('host_out', [4], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = inner.add_state('inner')

    if mode in ('write', 'both'):
        an_in = state.add_access('host_in')
        an_shared_w = state.add_access('shared_arr')
        state.add_edge(an_in, None, an_shared_w, None, Memlet('shared_arr[0:4]'))
    if mode in ('read', 'both'):
        an_shared_r = state.add_access('shared_arr')
        an_out = state.add_access('host_out')
        state.add_edge(an_shared_r, None, an_out, None, Memlet('host_out[0:4]'))
    return inner


def _wrap_in_gpu_kernel(inner: SDFG, *, with_inputs: bool, with_outputs: bool) -> SDFG:
    """Wrap ``inner`` in an outer SDFG with a GPU_Device map around the NestedSDFG."""
    outer = SDFG('outer')
    outer.add_array('A', [4], dace.float32, storage=dtypes.StorageType.GPU_Global)
    outer.add_array('B', [4], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = outer.add_state('s0')

    inputs = {'host_in'} if with_inputs else set()
    outputs = {'host_out'} if with_outputs else set()
    nsdfg_node = state.add_nested_sdfg(inner, inputs, outputs)
    me, mx = state.add_map('kmap', dict(i='0:1'), schedule=dtypes.ScheduleType.GPU_Device)

    if with_inputs:
        an_a = state.add_access('A')
        state.add_edge(an_a, None, me, 'IN_A', Memlet('A[0:4]'))
        me.add_in_connector('IN_A')
        me.add_out_connector('OUT_A')
        state.add_edge(me, 'OUT_A', nsdfg_node, 'host_in', Memlet('A[0:4]'))
    else:
        # An empty edge to anchor the NestedSDFG inside the kernel scope.
        state.add_edge(me, None, nsdfg_node, None, Memlet())

    if with_outputs:
        an_b = state.add_access('B')
        mx.add_in_connector('IN_B')
        mx.add_out_connector('OUT_B')
        state.add_edge(nsdfg_node, 'host_out', mx, 'IN_B', Memlet('B[0:4]'))
        state.add_edge(mx, 'OUT_B', an_b, None, Memlet('B[0:4]'))
    else:
        state.add_edge(nsdfg_node, None, mx, None, Memlet())

    return outer


def _find_nsdfg_node(outer: SDFG):
    for s in outer.states():
        for n in s.nodes():
            if isinstance(n, nodes.NestedSDFG):
                return n, s
    return None, None


def test_lift_shared_read_and_written():
    inner = _build_inner_sdfg_with_shared('inner_rw', mode='both')
    outer = _wrap_in_gpu_kernel(inner, with_inputs=True, with_outputs=True)

    LiftSharedOutOfNestedSDFG().apply_pass(outer, {})

    # Outer SDFG now owns a transient with the lifted name and Shared storage.
    assert 'shared_arr' in outer.arrays, 'lift should add the descriptor on the outer SDFG'
    out_desc = outer.arrays['shared_arr']
    assert out_desc.transient is True
    assert out_desc.storage == dtypes.StorageType.GPU_Shared

    # Inner descriptor is now non-transient (a connector parameter).
    assert inner.arrays['shared_arr'].transient is False

    # NSDFG node has both connectors (read + write).
    nsdfg_node, state = _find_nsdfg_node(outer)
    assert 'shared_arr' in nsdfg_node.in_connectors
    assert 'shared_arr' in nsdfg_node.out_connectors

    # Dependency edge from MapEntry exists (anchors allocation in kernel scope).
    me = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    mx = state.exit_node(me)
    me_to_an = [e for e in state.out_edges(me) if isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'shared_arr']
    assert len(me_to_an) >= 1, 'expected at least one dep edge MapEntry -> AccessNode(shared_arr)'

    an_to_mx = [e for e in state.in_edges(mx) if isinstance(e.src, nodes.AccessNode) and e.src.data == 'shared_arr']
    assert len(an_to_mx) >= 1, 'expected at least one dep edge AccessNode(shared_arr) -> MapExit'


def test_lift_shared_write_only_anchors_via_map_entry():
    """Write-only path still gets an incoming dep edge from MapEntry."""
    inner = _build_inner_sdfg_with_shared('inner_w', mode='write')
    outer = _wrap_in_gpu_kernel(inner, with_inputs=True, with_outputs=False)

    LiftSharedOutOfNestedSDFG().apply_pass(outer, {})

    assert 'shared_arr' in outer.arrays
    nsdfg_node, state = _find_nsdfg_node(outer)
    assert 'shared_arr' in nsdfg_node.out_connectors
    assert 'shared_arr' not in nsdfg_node.in_connectors

    me = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    me_to_an = [e for e in state.out_edges(me) if isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'shared_arr']
    assert len(me_to_an) == 1, 'write-only path must add the MapEntry->AccessNode anchor edge'


def test_lift_shared_unused_is_skipped():
    """An inner Shared transient that is never read or written is not lifted."""
    inner = _build_inner_sdfg_with_shared('inner_unused', mode='none')
    outer = _wrap_in_gpu_kernel(inner, with_inputs=False, with_outputs=False)

    result = LiftSharedOutOfNestedSDFG().apply_pass(outer, {})

    assert 'shared_arr' not in outer.arrays, 'unused inner Shared should not be lifted'
    assert inner.arrays['shared_arr'].transient is True, 'inner descriptor stays transient when unused'
    # No work means apply_pass returns None.
    assert result is None


def test_lift_shared_idempotent():
    """Two consecutive applications produce the same topology as one."""
    inner = _build_inner_sdfg_with_shared('inner_idem', mode='both')
    outer = _wrap_in_gpu_kernel(inner, with_inputs=True, with_outputs=True)

    LiftSharedOutOfNestedSDFG().apply_pass(outer, {})
    arrays_after_first = set(outer.arrays.keys())
    inner_arrays_after_first = set(inner.arrays.keys())

    LiftSharedOutOfNestedSDFG().apply_pass(outer, {})

    assert set(outer.arrays.keys()) == arrays_after_first
    assert set(inner.arrays.keys()) == inner_arrays_after_first


if __name__ == '__main__':
    test_lift_shared_read_and_written()
    test_lift_shared_write_only_anchors_via_map_entry()
    test_lift_shared_unused_is_skipped()
    test_lift_shared_idempotent()
