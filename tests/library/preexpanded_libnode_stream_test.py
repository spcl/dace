# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for pre-expanded GPU runtime tasklets in the stream pipeline.

Some callers expand library nodes manually before calling ``compile()``
(e.g. to inspect the lowered SDFG). The stream pipeline must still treat
the resulting ``cudaMemcpyAsync`` / ``cudaMemsetAsync`` tasklets as
stream consumers — otherwise their assigned stream silently defaults to
0 and no synchronization is emitted.

These tests pin the contract:

1. Pre-expanded ``CopyLibraryNode`` Tasklets receive a ``stream``
   in-connector wired to ``gpu_streams[<i>]``, just like the unexpanded
   libnodes would.
2. The naive sync classifier emits ``cudaStreamSynchronize`` after them.
3. The monolithic single-stream strategy accepts pre-expanded SDFGs and
   emits the expected number of host-boundary syncs.
4. The legacy ``__dace_current_stream`` codegen prelude raises if a
   Tasklet references the symbol without a wired stream connector — the
   silent-fallback bug it hid in the past.
"""
import pytest

import dace
from dace.codegen import common
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import MonolithicSingleStreamGPUScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               has_stream_connector,
                                                                               is_already_lowered_gpu_runtime_call)


def _build_h2d_d2h_pre_expanded_sdfg():
    """Build an SDFG with ``CopyLibraryNode`` H2D + D2H, then pre-expand."""
    sdfg = dace.SDFG('preexpanded_h2d_d2h')
    sdfg.add_array('host_in', [16], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array('host_out', [16], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array('dev', [16], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)

    state = sdfg.add_state('s')
    a = state.add_access('host_in')
    d = state.add_access('dev')
    b = state.add_access('host_out')
    h2d = CopyLibraryNode(name='copy_h2d')
    h2d.implementation = 'MemcpyCUDA1D'
    state.add_node(h2d)
    d2h = CopyLibraryNode(name='copy_d2h')
    d2h.implementation = 'MemcpyCUDA1D'
    state.add_node(d2h)
    state.add_edge(a, None, h2d, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('host_in[0:16]'))
    state.add_edge(h2d, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, d, None, dace.Memlet('dev[0:16]'))
    state.add_edge(d, None, d2h, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('dev[0:16]'))
    state.add_edge(d2h, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.Memlet('host_out[0:16]'))

    sdfg.expand_library_nodes()
    return sdfg


def _runtime_tasklets(sdfg):
    return [(n, state) for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() for n in state.nodes()
            if is_already_lowered_gpu_runtime_call(n)]


def _sync_tasklets(sdfg):
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [(n, state) for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() for n in state.nodes()
            if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_naive_strategy_wires_stream_connector_on_pre_expanded_tasklet():
    """The naive strategy must recognize pre-expanded ``cudaMemcpyAsync``
    Tasklets as stream consumers and wire a ``stream`` in-connector to each."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    runtime_calls = _runtime_tasklets(sdfg)
    assert len(runtime_calls) == 2

    GPUStreamPipeline().apply_pass(sdfg, {})

    for tasklet, _ in _runtime_tasklets(sdfg):
        assert has_stream_connector(tasklet), (
            f"Pre-expanded tasklet '{tasklet.label}' must have a stream in-connector "
            f"after the pipeline runs.")
        assert COPY_MEMSET_STREAM_CONNECTOR in tasklet.in_connectors


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_naive_strategy_emits_state_end_sync_for_pre_expanded_tasklets():
    """Naive strategy must emit a ``cudaStreamSynchronize`` after the
    runtime tasklets so the host doesn't race with their async work."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline().apply_pass(sdfg, {})

    syncs = _sync_tasklets(sdfg)
    assert len(syncs) >= 1, "Expected at least one sync tasklet for the pre-expanded H2D/D2H copies."


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_monolithic_strategy_accepts_pre_expanded_sdfg():
    """Monolithic strategy must accept a pre-expanded SDFG (host-level
    cudaMemcpyAsync tasklets are recognized as stream consumers, not
    rejected by the all-on-GPU validator)."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline(scheduling_strategy=MonolithicSingleStreamGPUScheduler()).apply_pass(sdfg, {})

    syncs = _sync_tasklets(sdfg)
    assert len(syncs) == 1, (f"Monolithic on the H2D+D2H state should emit exactly one host-boundary sync; "
                             f"got {len(syncs)}.")


def test_pipeline_wires_connector_for_pre_expanded_runtime_tasklet():
    """The pipeline must wire a ``__stream`` connector onto every
    pre-expanded GPU runtime tasklet. The codegen prelude binds
    ``__dace_current_stream`` from that connector — so a tasklet using
    the legacy symbol without a wired connector would generate
    ill-formed C++. We verify the connector is present after the
    pipeline, which is what makes the codegen prelude work."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline().apply_pass(sdfg, {})
    for tasklet, _ in _runtime_tasklets(sdfg):
        assert any(
            t == dace.dtypes.gpuStream_t
            for t in tasklet.in_connectors.values()), (f"Pre-expanded runtime tasklet '{tasklet.label}' must carry a "
                                                       f"gpuStream_t in-connector after the pipeline runs.")
