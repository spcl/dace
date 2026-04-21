# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.codegen import common
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets

gpu_stream_pipeline = Pipeline([
    InsertExplicitGPUGlobalMemoryCopies(),
    NaiveGPUStreamScheduler(),
    InsertGPUStreams(),
    ConnectGPUStreamsToNodes(),
    InsertGPUStreamSyncTasklets(),
])

backend = common.get_gpu_backend()


def _sync_tasklet(state):
    """Return the single ``{backend}StreamSynchronize`` tasklet in ``state``."""
    sync_tasklets = [
        n for n in state.nodes()
        if isinstance(n, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in n.code.as_string
    ]
    assert len(sync_tasklets) == 1, (f"Exactly one stream-synchronization tasklet is expected, "
                                     f"found {len(sync_tasklets)}.")
    return sync_tasklets[0]


def _stream_in_edges(state, node):
    """Return the in-edges of ``node`` that carry a ``gpu_streams[...]`` memlet."""
    return [e for e in state.in_edges(node) if e.data is not None and str(e.data).startswith('gpu_streams[')]


@pytest.mark.gpu
def test_basic():
    """
    A simple memory copy program.

    Since the SDFG has a single connected component, exactly one GPU stream is
    used and must be synchronized at the end of the state.  The unified
    ``ConnectGPUStreamsToNodes`` pass plus ``InsertGPUStreamSyncTasklets``
    produces a single synchronization tasklet that receives exactly one
    ``gpu_streams[i]`` in-edge per stream used.
    """

    @dace.program
    def simple_copy(A: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:128:1] @ dace.dtypes.ScheduleType.GPU_Device:
            B[i] = A[i]

    sdfg = simple_copy.to_sdfg()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]

    # Exactly one synchronization tasklet, and it is a sink.
    sync = _sync_tasklet(state)
    assert sync in state.sink_nodes(), "The stream-synchronization tasklet must be a sink of the state."

    # Exactly one stream used (single connected component), so exactly one
    # ``gpu_streams[...]`` in-edge feeds the synchronization tasklet.
    stream_edges = _stream_in_edges(state, sync)
    assert len(stream_edges) == 1, (f"Expected one gpu_streams in-edge on the sync tasklet, "
                                    f"got {len(stream_edges)}: {[str(e.data) for e in stream_edges]}")
    assert stream_edges[0].src.desc(state).dtype == dace.dtypes.gpuStream_t, (
        "The gpu_streams in-edge must originate from a gpu_streams AccessNode.")


@pytest.mark.gpu
def test_extended():
    """
    A program that performs two independent memory copies.

    The input arrays reside in host memory, and ``apply_gpu_transformations``
    is applied.  The data is first copied to GPU global memory, after which
    the two copies are executed on the GPU.  Since these copies form two
    independent connected components, the naive scheduler assigns them to
    different GPU streams.

    This test verifies:
      1. Both streams are synchronized at the end of the state (one sync
         tasklet with two distinct ``gpu_streams[i]`` in-edges).
      2. Each asynchronous memory-copy tasklet has exactly one input
         connector (its stream handle) and references it in its code.
    """

    @dace.program
    def independent_copies(A: dace.uint32[128], B: dace.uint32[128], C: dace.uint32[128], D: dace.uint32[128]):
        for i in dace.map[0:128:1]:
            B[i] = A[i]
        for i in dace.map[0:128:1]:
            D[i] = C[i]

    sdfg = independent_copies.to_sdfg()
    sdfg.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]

    # -- 1. Synchronization at end of state covers both streams. --------------
    sync = _sync_tasklet(state)
    stream_edges = _stream_in_edges(state, sync)
    stream_slots = {str(e.data) for e in stream_edges}
    assert stream_slots == {'gpu_streams[0]',
                            'gpu_streams[1]'}, (f"Expected both streams to feed the sync tasklet, got {stream_slots}")
    for e in stream_edges:
        assert e.src.desc(state).dtype == dace.dtypes.gpuStream_t, (
            "Every gpu_streams in-edge must originate from a gpu_streams AccessNode.")

    # -- 2. CopyLibraryNodes wire their stream connector correctly. -----------
    # After the merge of explicit-streams / explicit-gpu-global-copies, the
    # pipeline inserts ``CopyLibraryNode`` instances (lowered to cudaMemcpyAsync
    # at codegen time) rather than raw ``MemcpyAsync`` tasklets.
    copy_libnodes = [n for n in state.nodes() if type(n).__name__ == 'CopyLibraryNode']
    assert copy_libnodes, ("Expected at least one CopyLibraryNode after gpu_transformations + "
                           "InsertExplicitGPUGlobalMemoryCopies.")
    for cn in copy_libnodes:
        assert 'stream' in cn.in_connectors, ("CopyLibraryNode must expose a 'stream' in-connector "
                                              "for the GPU stream handle.")
        stream_edges_cn = _stream_in_edges(state, cn)
        assert len(stream_edges_cn) == 1, (f"CopyLibraryNode '{cn.label}' must have exactly one "
                                           f"gpu_streams in-edge, got {len(stream_edges_cn)}.")
        assert stream_edges_cn[0].dst_conn == 'stream', ("The gpu_streams in-edge must target the "
                                                         "'stream' connector.")
