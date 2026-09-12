# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test for ``AutoSingleStreamGPUScheduler`` sync placement at root level.

Reproduces the shape of ``failed_validation.sdfg`` (an ICON-style program):

  * Three top-level blocks: a CPU host state, a GPU computation state, then a CPU host state.
  * The GPU state contains some free Tasklets/AccessNodes plus a ``GPU_Device`` map.
  * The ``GPU_Device`` map's body has an inner ``GPU_ThreadBlock`` map and a ``NestedSDFG``.

The earlier ``insert_sync_tasklets`` implementation walked the entire CFG recursively, so it
descended into the NSDFG that sits inside the kernel and spliced a sync state there --
planting a ``gpu_streams[0]`` memlet with no inner ``gpu_streams`` array, triggering
``Node validation failed: 'gpu_streams'`` at validation time. The fix restricts sync placement
to the root SDFG's ``.nodes()`` / ``.edges()`` only.

This test pins that contract: build the SDFG by hand to mirror the failing shape, run the
default pipeline, then assert (a) the sync state lands at the outer GPU -> CPU iedge, (b) no
sync tasklets exist anywhere inside the NestedSDFG that lives inside the GPU kernel, and
(c) the SDFG validates.
"""
import dace
import pytest

from dace.codegen import common
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline


def _build_simple_three_state_sdfg() -> dace.SDFG:
    """Simpler ``cpu_pre -> gpu_mid -> cpu_post`` SDFG; ``gpu_mid`` wraps the GPU kernel
    in a NestedSDFG so the outer state has no GPU-Device map of its own."""
    sdfg = dace.SDFG('three_state_cpu_gpu_cpu')
    sdfg.add_array('A', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('host_in', [1], dace.float32)
    sdfg.add_array('host_out', [1], dace.float32)

    cpu_pre = sdfg.add_state('cpu_pre', is_start_block=True)
    t_pre = cpu_pre.add_tasklet('host_init', {}, {'__out': dace.pointer(dace.float32)},
                                '__out = 1.0f;',
                                language=dace.Language.CPP)
    cpu_pre.add_edge(t_pre, '__out', cpu_pre.add_write('host_in'), None, dace.Memlet('host_in[0]'))

    gpu_mid = sdfg.add_state('gpu_mid')
    inner = dace.SDFG('inner_kernel')
    inner.add_array('a_in', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_out', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner_st = inner.add_state('inner_state')
    me, mx = inner_st.add_map('gpu_map', dict(i='0:16'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = inner_st.add_tasklet('mul2', {'_a': dace.float32}, {'_b': dace.float32}, '_b = _a * 2.0')
    inner_st.add_memlet_path(inner_st.add_read('a_in'), me, kt, dst_conn='_a', memlet=dace.Memlet('a_in[i]'))
    inner_st.add_memlet_path(kt, mx, inner_st.add_write('b_out'), src_conn='_b', memlet=dace.Memlet('b_out[i]'))
    nsdfg_node = gpu_mid.add_nested_sdfg(inner, {'a_in'}, {'b_out'}, {})
    gpu_mid.add_edge(gpu_mid.add_read('A'), None, nsdfg_node, 'a_in', dace.Memlet('A[0:16]'))
    gpu_mid.add_edge(nsdfg_node, 'b_out', gpu_mid.add_write('B'), None, dace.Memlet('B[0:16]'))

    cpu_post = sdfg.add_state('cpu_post')
    t_post = cpu_post.add_tasklet('host_fin', {}, {'__out': dace.pointer(dace.float32)},
                                  '__out = 9.0f;',
                                  language=dace.Language.CPP)
    cpu_post.add_edge(t_post, '__out', cpu_post.add_write('host_out'), None, dace.Memlet('host_out[0]'))

    sdfg.add_edge(cpu_pre, gpu_mid, dace.InterstateEdge())
    sdfg.add_edge(gpu_mid, cpu_post, dace.InterstateEdge())
    return sdfg


def _build_failed_validation_shape() -> dace.SDFG:
    """``cpu_pre -> gpu_kernel_state -> cpu_post``; ``gpu_kernel_state`` carries a
    ``GPU_Device`` map containing a ``GPU_ThreadBlock`` map and a ``NestedSDFG``."""
    sdfg = dace.SDFG('failed_validation_shape')
    sdfg.add_array('A', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('host_in', [1], dace.float32)
    sdfg.add_array('host_out', [1], dace.float32)

    # State 1 (CPU): host init.
    cpu_pre = sdfg.add_state('cpu_pre', is_start_block=True)
    t_pre = cpu_pre.add_tasklet('host_init', {}, {'__out': dace.pointer(dace.float32)},
                                '__out = 1.0f;',
                                language=dace.Language.CPP)
    cpu_pre.add_edge(t_pre, '__out', cpu_pre.add_write('host_in'), None, dace.Memlet('host_in[0]'))

    # State 2 (GPU): a GPU_Device map with a thread block map and a NestedSDFG inside.
    gpu_state = sdfg.add_state('gpu_kernel_state')
    a_read = gpu_state.add_read('A')
    b_write = gpu_state.add_write('B')
    dev_me, dev_mx = gpu_state.add_map('gpu_device_map',
                                       dict(blockIdx_x='0:1'),
                                       schedule=dace.dtypes.ScheduleType.GPU_Device)
    blk_me, blk_mx = gpu_state.add_map('gpu_threadblock_map',
                                       dict(threadIdx_x='0:16'),
                                       schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    # NestedSDFG INSIDE the GPU_Device + GPU_ThreadBlock scope -- this is the structure that
    # the broken pipeline used to splice a sync state into.
    inner = dace.SDFG('per_thread_body')
    inner.add_array('a_lane', [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_lane', [1], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner_st = inner.add_state('s')
    in_t = inner_st.add_tasklet('mul2', {'_a': dace.float32}, {'_b': dace.float32}, '_b = _a * 2.0')
    inner_st.add_edge(inner_st.add_read('a_lane'), None, in_t, '_a', dace.Memlet('a_lane[0]'))
    inner_st.add_edge(in_t, '_b', inner_st.add_write('b_lane'), None, dace.Memlet('b_lane[0]'))
    nsdfg_node = gpu_state.add_nested_sdfg(inner, {'a_lane'}, {'b_lane'}, {})

    # Wire the memlet path: A -> Device map -> ThreadBlock map -> NSDFG -> ThreadBlock map -> Device map -> B.
    gpu_state.add_memlet_path(a_read,
                              dev_me,
                              blk_me,
                              nsdfg_node,
                              dst_conn='a_lane',
                              memlet=dace.Memlet('A[threadIdx_x]'))
    gpu_state.add_memlet_path(nsdfg_node,
                              blk_mx,
                              dev_mx,
                              b_write,
                              src_conn='b_lane',
                              memlet=dace.Memlet('B[threadIdx_x]'))

    # State 3 (CPU): host finalize.
    cpu_post = sdfg.add_state('cpu_post')
    t_post = cpu_post.add_tasklet('host_fin', {}, {'__out': dace.pointer(dace.float32)},
                                  '__out = 9.0f;',
                                  language=dace.Language.CPP)
    cpu_post.add_edge(t_post, '__out', cpu_post.add_write('host_out'), None, dace.Memlet('host_out[0]'))

    sdfg.add_edge(cpu_pre, gpu_state, dace.InterstateEdge())
    sdfg.add_edge(gpu_state, cpu_post, dace.InterstateEdge())
    return sdfg


def _sync_tasklets(state):
    """``cudaStreamSynchronize`` / ``hipStreamSynchronize`` tasklets in ``state``."""
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


def test_sync_at_root_with_nsdfg_inside_gpu_device_map():
    """Sync must be placed at the OUTER ``gpu_kernel_state -> cpu_post`` boundary, not
    inside the NestedSDFG that lives inside the ``GPU_Device`` map.

    This is the exact failure mode ``failed_validation.sdfg`` exhibited under the old
    recursive walk: the sync got placed inside the NSDFG body, planting a ``gpu_streams[0]``
    memlet in a sub-SDFG that has no inner ``gpu_streams`` array."""
    sdfg = _build_failed_validation_shape()
    GPUStreamPipeline().apply_pass(sdfg, {})

    # Root SDFG: original three states + one spliced sync state.
    sync_states_at_root = [s for s in sdfg.states() if _sync_tasklets(s)]
    assert len(sync_states_at_root) == 1, (f"Expected exactly one sync state at root; "
                                           f"got {[s.label for s in sync_states_at_root]}")
    sync_state = sync_states_at_root[0]
    assert sync_state.label.startswith('__gpu_sync_after_'), sync_state.label

    # The sync state must be on the path from gpu_kernel_state to cpu_post (after gpu state).
    successors_of_sync = {e.dst.label for e in sdfg.out_edges(sync_state)}
    predecessors_of_sync = {e.src.label for e in sdfg.in_edges(sync_state)}
    assert predecessors_of_sync == {'gpu_kernel_state'}, predecessors_of_sync
    assert successors_of_sync == {'cpu_post'}, successors_of_sync

    # No sync tasklets inside the NestedSDFG (or any other inner SDFG).
    inner_syncs = []
    for inner in sdfg.all_sdfgs_recursive():
        if inner is sdfg:
            continue
        for s in inner.states():
            if _sync_tasklets(s):
                inner_syncs.append(f'{inner.name}::{s.label}')
    assert not inner_syncs, (f"Expected no sync tasklets inside any nested SDFG; "
                             f"found syncs in: {inner_syncs}")

    sdfg.validate()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
