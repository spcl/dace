# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test for a crash in the canonicalize finalize tail's nested-SDFG schedule
handling (``dace/transformation/passes/canonicalize/finalize.py``).

``libnode_is_sequential`` used to read ``state.sdfg.parent_nsdfg_node.schedule`` to decide
whether a library node lives inside a "Sequential-scheduled nested SDFG". A ``NestedSDFG``
node carries no ``schedule`` property at all (only ``Map`` and ``LibraryNode`` do), so that
probe raised ``AttributeError: 'NestedSDFG' object has no attribute 'schedule'`` for any
library node nested inside an SDFG that is itself nested inside another SDFG, whenever the
node's own schedule had not already been pinned ``Sequential`` by the earlier
``sequentialize_nested_parallel_scopes`` pass (e.g. a top-level nsdfg call not itself wrapped
in a map -- ``tests/numpy/lift_einsum_matmul_test.py::test_lift_via_canon[gemv_acc]``).

The fix drops the broken probe: :func:`~dace.transformation.helpers.get_parent_map_and_loop_scopes`
already walks OUT across every nested-SDFG boundary to the root and yields every enclosing
``MapEntry`` / ``LoopRegion``, so a parallel map or loop several nsdfg levels up is still found
by the loop that follows -- the "Sequential nested SDFG" case was never a distinct one to probe.
"""
import dace
from dace import dtypes
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.passes.canonicalize.finalize import libnode_is_sequential, sequentialize_nested_parallel_scopes


def _build_nested_reduce_in_parallel_map():
    """A CPU_Multicore outer map whose body is a NestedSDFG; the nested SDFG's body is a
    ``Reduce`` library node left on the storage-derived ``CPU_Multicore`` schedule (as
    ``infer_types.set_default_schedule_and_storage_types`` would leave it -- unaware that the
    node is re-entered per outer iteration). Returns ``(sdfg, reduce_node)``.
    """
    sdfg = dace.SDFG('nsdfg_sched_regress')
    sdfg.add_array('A', [8, 8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    state = sdfg.add_state()

    inner = dace.SDFG('inner')
    inner.add_array('row', [8], dace.float64)
    inner.add_array('acc', [1], dace.float64)
    istate = inner.add_state()
    row_access = istate.add_access('row')
    acc_access = istate.add_access('acc')
    reduce_node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    reduce_node.schedule = dtypes.ScheduleType.CPU_Multicore
    istate.add_node(reduce_node)
    istate.add_edge(row_access, None, reduce_node, None, dace.Memlet('row[0:8]'))
    istate.add_edge(reduce_node, None, acc_access, None, dace.Memlet('acc[0]'))

    nsdfg_node = state.add_nested_sdfg(inner, {'row'}, {'acc'})
    map_entry, map_exit = state.add_map('outer', dict(i='0:8'), schedule=dtypes.ScheduleType.CPU_Multicore)
    a_access = state.add_access('A')
    b_access = state.add_access('B')
    state.add_memlet_path(a_access, map_entry, nsdfg_node, dst_conn='row', memlet=dace.Memlet('A[i, 0:8]'))
    state.add_memlet_path(nsdfg_node, map_exit, b_access, src_conn='acc', memlet=dace.Memlet('B[i:i+1]'))

    sdfg.validate()
    return sdfg, reduce_node


def _find_state(sdfg: dace.SDFG, node) -> dace.SDFGState:
    for n, state in sdfg.all_nodes_recursive():
        if n is node:
            return state
    raise AssertionError('node not found in sdfg')


def test_sequentialize_nested_parallel_scopes_pins_nested_reduce():
    """``sequentialize_nested_parallel_scopes`` must not raise on a NestedSDFG nested inside a
    parallel map, and must pin the nsdfg's body (the Reduce library node) to ``Sequential`` --
    the transitive behaviour its docstring promises."""
    sdfg, reduce_node = _build_nested_reduce_in_parallel_map()
    assert reduce_node.schedule == dtypes.ScheduleType.CPU_Multicore

    sequentialize_nested_parallel_scopes(sdfg, dtypes.DeviceType.CPU)

    assert reduce_node.schedule == dtypes.ScheduleType.Sequential


def test_libnode_is_sequential_survives_unpinned_schedule_in_nested_sdfg():
    """The actual crash site: ``libnode_is_sequential`` called on a library node whose own
    schedule was NOT already pinned ``Sequential`` (``sequentialize_nested_parallel_scopes``
    only pins a node re-entered under a parallel map/loop; a libnode inside an nsdfg called
    from a non-parallel context keeps its storage-derived schedule) but that still lives
    inside a nested SDFG (``state.sdfg.parent_nsdfg_node is not None``). Before the fix this
    raised ``AttributeError: 'NestedSDFG' object has no attribute 'schedule'``."""
    sdfg, reduce_node = _build_nested_reduce_in_parallel_map()
    assert reduce_node.schedule == dtypes.ScheduleType.CPU_Multicore
    state = _find_state(sdfg, reduce_node)
    assert state.sdfg.parent_nsdfg_node is not None

    # No prior sequentialize_nested_parallel_scopes call: the schedule is exactly the
    # storage-derived CPU_Multicore that used to reach the broken parent_nsdfg.schedule probe.
    sequential = libnode_is_sequential(reduce_node, state, sdfg)

    assert sequential is True
