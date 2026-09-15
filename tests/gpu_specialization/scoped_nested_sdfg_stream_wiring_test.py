# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test: the GPU stream reaches a NestedSDFG that sits INSIDE a host map scope.

``samples/simple/spmv.py`` under the transformation tester produces exactly this shape -- a
host ``Sequential``/``CPU_Multicore`` map nest whose body is a NestedSDFG holding the kernel.
Stream propagation used to wire ``gpu_streams`` straight from a global AccessNode into that
NestedSDFG, an edge crossing the map boundary without passing through the MapEntry. The graph
then had no scope path to the MapExit, so ``SDFGState.scope_dict`` refused it with "Leftover
nodes in queue" and codegen died inside its own preprocessing pipeline.
"""
import dace

from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (STREAM_CONNECTOR,
                                                                               get_gpu_stream_array_name)


def _kernel_in_a_host_map_scope() -> dace.SDFG:
    """``map i (host) { nested SDFG { map j (GPU_Device) } }`` -- the spmv shape, minimised."""
    inner = dace.SDFG('inner_kernel')
    inner.add_array('a_in', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_out', [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('inner_state', is_start_block=True)
    kernel_entry, kernel_exit = inner_state.add_map('gpu_map',
                                                    dict(j='0:16'),
                                                    schedule=dace.dtypes.ScheduleType.GPU_Device)
    tasklet = inner_state.add_tasklet('mul2', {'_a': dace.float32}, {'_b': dace.float32}, '_b = _a * 2.0')
    inner_state.add_memlet_path(inner_state.add_read('a_in'),
                                kernel_entry,
                                tasklet,
                                dst_conn='_a',
                                memlet=dace.Memlet('a_in[j]'))
    inner_state.add_memlet_path(tasklet,
                                kernel_exit,
                                inner_state.add_write('b_out'),
                                src_conn='_b',
                                memlet=dace.Memlet('b_out[j]'))

    sdfg = dace.SDFG('kernel_under_host_map')
    sdfg.add_array('A', [4, 16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [4, 16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('launch', is_start_block=True)
    host_entry, host_exit = state.add_map('host_map', dict(i='0:4'), schedule=dace.dtypes.ScheduleType.Sequential)
    nsdfg_node = state.add_nested_sdfg(inner, ['a_in'], ['b_out'])
    state.add_memlet_path(state.add_read('A'),
                          host_entry,
                          nsdfg_node,
                          dst_conn='a_in',
                          memlet=dace.Memlet('A[i, 0:16]'))
    state.add_memlet_path(nsdfg_node,
                          host_exit,
                          state.add_write('B'),
                          src_conn='b_out',
                          memlet=dace.Memlet('B[i, 0:16]'))
    sdfg.validate()
    return sdfg


def test_stream_reaches_a_nested_sdfg_inside_a_host_map():
    sdfg = _kernel_in_a_host_map_scope()
    GPUStreamPipeline().apply_pass(sdfg, {})
    sdfg.validate()

    state = sdfg.states()[0]
    stream_name = get_gpu_stream_array_name()
    nsdfg_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    # By label, not ``entry_node``: the scope lookup is exactly what a mis-wired stream edge
    # corrupts, so asking it here would report the damage as a missing scope instead.
    host_entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.label == 'host_map')

    # Every stream edge reaching the scoped NestedSDFG comes from its enclosing MapEntry, not
    # from a global AccessNode: that is what keeps the scope walkable.
    stream_in_edges = [e for e in state.in_edges(nsdfg_node) if e.data.data == stream_name]
    assert stream_in_edges, 'the NestedSDFG holding the kernel must receive the stream'
    for edge in stream_in_edges:
        assert edge.src is host_entry, f'stream edge bypasses the map scope: {edge.src} -> {edge.dst}'

    # The pass that used to crash here is the one codegen runs first; scope_dict must accept
    # the graph, and the enclosing map must carry the stream pass-through pair.
    state.scope_dict()
    assert f'IN_{STREAM_CONNECTOR}' in host_entry.in_connectors
    assert f'OUT_{STREAM_CONNECTOR}' in host_entry.out_connectors


if __name__ == '__main__':
    test_stream_reaches_a_nested_sdfg_inside_a_host_map()
