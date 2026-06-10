# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end regression test that reconstructs the ``failed_validation.sdfg`` shape.

The original ``failed_validation.sdfg`` (ICON ``__field_operator_pnabla``) has three
top-level blocks:

* ``metrics_entry`` -- a ``ConditionalBlock`` whose single branch wraps an ``SDFGState``
  containing a host-side ``gt_start_timer`` Tasklet that writes the start time scalar.
* ``stmt_0`` -- a single ``SDFGState`` holding host derefs of connectivity descriptors
  (free Tasklets at the top level) alongside one or more ``GPU_Device`` maps; the kernel
  body contains a ``NestedSDFG`` (``reduce_with_skip_values_*``).
* ``metrics_exit`` -- a sibling ``ConditionalBlock`` with the matching ``gt_stop_timer``.

Under the previous recursive ``insert_sync_tasklets`` walk, the pipeline reached into the
NSDFG inside the ``GPU_Device`` map and spliced a sync state there, planting a
``gpu_streams[0]`` memlet with no inner ``gpu_streams`` array. The validator fired
``Node validation failed: 'gpu_streams'``. The root-only walk fix lifts the host deref out
via :class:`SplitStateByGPUClass` and places the sync state cleanly between ``stmt_0`` and
``metrics_exit`` at the root SDFG level.

This test pins that behaviour end-to-end: build the structural skeleton via the SDFG API,
run the default pipeline, assert sync placement, assert the SDFG validates.
"""
import dace
import pytest

from dace.codegen import common
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline

_N = dace.symbol('N', dtype=dace.int32)


def _add_metrics_timer_block(sdfg: dace.SDFG, label: str, *, start: bool) -> ConditionalBlock:
    """Build a ``ConditionalBlock`` mirroring ICON's metrics timer wrapper.

    The branch's body is a single ``SDFGState`` with one host-side Tasklet using the
    ``side_effects`` property and a CUDA backend ``cudaDeviceSynchronize`` for accuracy.

    :param sdfg: The parent SDFG; the block is added to its top level.
    :param label: ConditionalBlock label (``metrics_entry`` or ``metrics_exit``).
    :param start: ``True`` to build the start-timer block, ``False`` for the stop-timer.
    :return: The new ``ConditionalBlock``.
    """
    cb = ConditionalBlock(label, sdfg=sdfg)
    sdfg.add_node(cb)

    branch = ControlFlowRegion(f'{label}_collect_metrics', sdfg=sdfg)
    state = branch.add_state(f'{label}_collect_metrics', is_start_block=True)

    if start:
        tasklet = state.add_tasklet(
            'gt_start_timer',
            {},
            {'time'},
            ('cudaDeviceSynchronize();\n'
             'auto now = std::chrono::high_resolution_clock::now();\n'
             'time = std::chrono::duration_cast<std::chrono::nanoseconds>(\n'
             '    now.time_since_epoch()).count();\n'),
            language=dace.Language.CPP,
            side_effects=True,
        )
        tasklet.out_connectors = {'time': dace.int64}
        state.add_edge(tasklet, 'time', state.add_write('gt_start_time'), None, dace.Memlet('gt_start_time[0]'))
    else:
        tasklet = state.add_tasklet(
            'gt_stop_timer',
            {'time_start'},
            {'time_total'},
            ('cudaDeviceSynchronize();\n'
             'auto now = std::chrono::high_resolution_clock::now();\n'
             'auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(\n'
             '    now.time_since_epoch()).count();\n'
             'time_total = now_ns - time_start;\n'),
            language=dace.Language.CPP,
            side_effects=True,
        )
        tasklet.in_connectors = {'time_start': dace.int64}
        tasklet.out_connectors = {'time_total': dace.int64}
        state.add_edge(state.add_read('gt_start_time'), None, tasklet, 'time_start', dace.Memlet('gt_start_time[0]'))
        state.add_edge(tasklet, 'time_total', state.add_write('gt_compute_time'), None,
                       dace.Memlet('gt_compute_time[0]'))

    cb.add_branch(CodeBlock('(gt_metrics_level >= 10)'), branch)
    return cb


def _build_failed_validation_shape() -> dace.SDFG:
    """Reconstruct the structural skeleton of ``failed_validation.sdfg``.

    :return: A ready-to-pipeline SDFG with the three-block shape (CB, compute state, CB)
             whose compute state holds a host deref Tasklet alongside a GPU_Device map
             that contains a NestedSDFG.
    """
    sdfg = dace.SDFG('failed_validation_reconstructed')
    sdfg.add_symbol('gt_metrics_level', dace.int32)
    sdfg.add_array('A', [_N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [_N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    # Host-side scalar that the free deref Tasklet writes to (mirrors ICON's host-side
    # connectivity-descriptor derefs).
    sdfg.add_scalar('S_M_0', dace.float64, transient=True)
    sdfg.add_scalar('gt_start_time', dace.int64, transient=True)
    sdfg.add_scalar('gt_compute_time', dace.int64, transient=True)

    metrics_entry = _add_metrics_timer_block(sdfg, 'metrics_entry', start=True)
    sdfg.start_block = sdfg.node_id(metrics_entry)

    # stmt_0 -- the compute state mirroring ICON's stmt_0: a free host-side deref Tasklet
    # alongside a GPU_Device map whose body contains a NestedSDFG.
    stmt_0 = sdfg.add_state('stmt_0')

    deref = stmt_0.add_tasklet('tlet_1_deref', {}, {'out'}, 'out = 1.0;', language=dace.Language.CPP)
    deref.out_connectors = {'out': dace.float64}
    stmt_0.add_edge(deref, 'out', stmt_0.add_write('S_M_0'), None, dace.Memlet('S_M_0[0]'))

    me, mx = stmt_0.add_map('map_0_fieldop', dict(i_Edge='0:N'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    # Per-lane scratch transients chaining the four kernel-body stages.
    sdfg.add_scalar('lane_in', dace.float64, transient=True, storage=dace.dtypes.StorageType.Register)
    sdfg.add_array('lane_buf', [4], dace.float64, transient=True, storage=dace.dtypes.StorageType.Register)
    sdfg.add_scalar('lane_mid', dace.float64, transient=True, storage=dace.dtypes.StorageType.Register)
    sdfg.add_scalar('lane_out', dace.float64, transient=True, storage=dace.dtypes.StorageType.Register)

    # Stage 1 (in-map Tasklet, BEFORE the inner sequential map) -- pre-multiplication.
    tlet_pre = stmt_0.add_tasklet('tlet_pre_deref', {'_a': dace.float64}, {'_l': dace.float64}, '_l = _a * 2.0')
    lane_in_acc = stmt_0.add_access('lane_in')

    # Stage 2: a short ``Sequential`` map nested inside the ``GPU_Device`` map -- mirroring
    # the ICON ``tlet_6_V2E_neighbors_map[0:7]`` neighbour-iteration shape. It writes per-k
    # entries into a small per-lane register buffer.
    seq_me, seq_mx = stmt_0.add_map('seq_neighbors_map', dict(k='0:4'), schedule=dace.dtypes.ScheduleType.Sequential)
    seq_t = stmt_0.add_tasklet('seq_neighbors_inner', {'_li': dace.float64}, {'_bk': dace.float64}, '_bk = _li + k')
    lane_buf_acc = stmt_0.add_access('lane_buf')
    stmt_0.add_memlet_path(lane_in_acc, seq_me, seq_t, dst_conn='_li', memlet=dace.Memlet('lane_in[0]'))
    stmt_0.add_memlet_path(seq_t, seq_mx, lane_buf_acc, src_conn='_bk', memlet=dace.Memlet('lane_buf[k]'))

    # Stage 3 (NestedSDFG ``reduce_with_skip_values_0``) -- the structure that the previous
    # recursive sync walk used to splice a sync state into.
    inner = dace.SDFG('reduce_with_skip_values_0')
    inner.add_array('a_in', [1], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_out', [1], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    init = inner.add_state('init', is_start_block=True)
    reduce_state = inner.add_state('reduce')
    inner.add_edge(init, reduce_state, dace.InterstateEdge())
    rt = reduce_state.add_tasklet('reduce_add', {'_a': dace.float64}, {'_b': dace.float64}, '_b = _a + 1.0')
    reduce_state.add_edge(reduce_state.add_read('a_in'), None, rt, '_a', dace.Memlet('a_in[0]'))
    reduce_state.add_edge(rt, '_b', reduce_state.add_write('b_out'), None, dace.Memlet('b_out[0]'))
    nsdfg = stmt_0.add_nested_sdfg(inner, {'a_in'}, {'b_out'}, {})
    lane_mid_acc = stmt_0.add_access('lane_mid')

    # Stage 4 (in-map Tasklet, AFTER the NSDFG) -- finalize.
    tlet_post = stmt_0.add_tasklet('tlet_post_finalize', {'_m': dace.float64}, {'_o': dace.float64}, '_o = _m - 1.0')
    lane_out_acc = stmt_0.add_access('lane_out')

    a_read = stmt_0.add_read('A')
    b_write = stmt_0.add_write('B')

    # Map -> pre-Tasklet -> lane_in -> Sequential(k=0:4) -> lane_buf -> NSDFG -> lane_mid
    # -> post-Tasklet -> lane_out -> Map exit.
    stmt_0.add_memlet_path(a_read, me, tlet_pre, dst_conn='_a', memlet=dace.Memlet('A[i_Edge]'))
    stmt_0.add_edge(tlet_pre, '_l', lane_in_acc, None, dace.Memlet('lane_in[0]'))
    stmt_0.add_edge(lane_buf_acc, None, nsdfg, 'a_in', dace.Memlet('lane_buf[0]'))
    stmt_0.add_edge(nsdfg, 'b_out', lane_mid_acc, None, dace.Memlet('lane_mid[0]'))
    stmt_0.add_edge(lane_mid_acc, None, tlet_post, '_m', dace.Memlet('lane_mid[0]'))
    stmt_0.add_edge(tlet_post, '_o', lane_out_acc, None, dace.Memlet('lane_out[0]'))
    stmt_0.add_memlet_path(lane_out_acc, mx, b_write, memlet=dace.Memlet('B[i_Edge]'))

    metrics_exit = _add_metrics_timer_block(sdfg, 'metrics_exit', start=False)

    sdfg.add_edge(metrics_entry, stmt_0, dace.InterstateEdge())
    sdfg.add_edge(stmt_0, metrics_exit, dace.InterstateEdge())
    return sdfg


def _sync_tasklets(state):
    """Return the ``cudaStreamSynchronize`` / ``hipStreamSynchronize`` Tasklets in ``state``."""
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


def test_failed_validation_shape_synced_at_root_and_validates():
    """The reconstructed shape compiles down to a single root-level sync state, no syncs
    inside the inner NestedSDFG, and the resulting SDFG must validate."""
    sdfg = _build_failed_validation_shape()
    GPUStreamPipeline().apply_pass(sdfg, {})

    # Exactly one sync tasklet across the SDFG hierarchy, in a root-level state.
    root_syncs = [s for s in sdfg.states() if _sync_tasklets(s)]
    assert len(root_syncs) == 1, (f"Expected exactly one root-level sync state, got "
                                  f"{[s.label for s in root_syncs]}")
    sync_state = root_syncs[0]
    assert sync_state.label.startswith('__gpu_sync_after_'), sync_state.label

    inner_syncs = []
    for inner in sdfg.all_sdfgs_recursive():
        if inner is sdfg:
            continue
        for s in inner.states():
            if _sync_tasklets(s):
                inner_syncs.append(f'{inner.name}::{s.label}')
    assert not inner_syncs, f"No sync tasklets must exist inside any nested SDFG; got {inner_syncs}"

    # Split must have lifted the host deref Tasklet out into a ``*_cpu_before`` state.
    block_labels = [b.label for b in sdfg.nodes()]
    assert any(
        lbl.endswith('_cpu_before')
        for lbl in block_labels), (f"Expected SplitStateByGPUClass to lift the host deref into a *_cpu_before state; "
                                   f"got block labels: {block_labels}")

    sdfg.validate()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
