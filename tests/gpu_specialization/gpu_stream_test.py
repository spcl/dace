# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for GPU stream scheduling (stream count, per-state sync-tasklet fusion)."""
import pytest

import dace
from dace.codegen import common
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline

gpu_stream_pipeline = GPUStreamPipeline()

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


def _all_sync_tasklets(state):
    backend = common.get_gpu_backend()
    return [
        n for n in state.nodes()
        if isinstance(n, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in n.code.as_string
    ]


@pytest.mark.gpu
def test_basic():
    """Single connected component: one stream, one sync tasklet with one gpu_streams in-edge."""

    @dace.program
    def simple_copy(A: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:128:1] @ dace.dtypes.ScheduleType.GPU_Device:
            B[i] = A[i]

    sdfg = simple_copy.to_sdfg()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]

    sync = _sync_tasklet(state)
    assert sync in state.sink_nodes(), "The stream-synchronization tasklet must be a sink of the state."

    stream_edges = _stream_in_edges(state, sync)
    assert len(stream_edges) == 1, (f"Expected one gpu_streams in-edge on the sync tasklet, "
                                    f"got {len(stream_edges)}: {[str(e.data) for e in stream_edges]}")
    assert stream_edges[0].src.desc(state).dtype == dace.dtypes.gpuStream_t, (
        "The gpu_streams in-edge must originate from a gpu_streams AccessNode.")


@pytest.mark.gpu
def test_extended():
    """Two independent components on two streams, fused into one sync tasklet
    per state with one ``__stream_<id>`` connector per stream id."""

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

    syncs = _all_sync_tasklets(state)
    assert len(syncs) == 1, f"Expected one fused sync tasklet (two streams); got {len(syncs)}."
    sync = syncs[0]
    stream_edges = _stream_in_edges(state, sync)
    assert len(stream_edges) == 2, (f"Fused sync tasklet must have one gpu_streams[<i>] edge per stream; "
                                    f"got {len(stream_edges)}: {[str(e.data) for e in stream_edges]}")
    seen_slots = {str(e.data) for e in stream_edges}
    for e in stream_edges:
        assert e.src.desc(state).dtype == dace.dtypes.gpuStream_t
    assert seen_slots == {'gpu_streams[0]', 'gpu_streams[1]'}

    copy_libnodes = [n for n in state.nodes() if type(n).__name__ == 'CopyLibraryNode']
    assert copy_libnodes, ("Expected at least one CopyLibraryNode after gpu_transformations + "
                           "InsertExplicitGPUGlobalMemoryCopies.")
    from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import STREAM_CONNECTOR
    for cn in copy_libnodes:
        assert STREAM_CONNECTOR in cn.in_connectors, (
            f"CopyLibraryNode must expose a {STREAM_CONNECTOR!r} in-connector for the GPU stream handle.")
        stream_edges_cn = _stream_in_edges(state, cn)
        assert len(stream_edges_cn) == 1, (f"CopyLibraryNode '{cn.label}' must have exactly one "
                                           f"gpu_streams in-edge, got {len(stream_edges_cn)}.")
        assert stream_edges_cn[0].dst_conn == STREAM_CONNECTOR
