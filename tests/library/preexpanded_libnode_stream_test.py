# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The stream pipeline treats pre-expanded ``cudaMemcpyAsync`` / ``cudaMemsetAsync`` tasklets as
stream consumers (connectors wired, syncs emitted, monolithic strategy accepting)."""
import pytest

import dace
from dace.codegen import common
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import MonolithicSingleStreamGPUScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (STREAM_CONNECTOR, has_stream_connector,
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
    """Naive strategy wires a ``stream`` in-connector on each pre-expanded ``cudaMemcpyAsync`` tasklet."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    runtime_calls = _runtime_tasklets(sdfg)
    assert len(runtime_calls) == 2

    GPUStreamPipeline().apply_pass(sdfg, {})

    for tasklet, _ in _runtime_tasklets(sdfg):
        assert has_stream_connector(tasklet), (
            f"Pre-expanded tasklet '{tasklet.label}' must have a stream in-connector "
            f"after the pipeline runs.")
        assert STREAM_CONNECTOR in tasklet.in_connectors


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_naive_strategy_emits_state_end_sync_for_pre_expanded_tasklets():
    """Naive strategy emits a ``cudaStreamSynchronize`` after the pre-expanded runtime tasklets."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline().apply_pass(sdfg, {})

    syncs = _sync_tasklets(sdfg)
    assert len(syncs) >= 1, "Expected at least one sync tasklet for the pre-expanded H2D/D2H copies."


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_monolithic_strategy_accepts_pre_expanded_sdfg():
    """Monolithic strategy accepts a pre-expanded SDFG (host-level copy tasklets pass the validator)."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline(scheduling_strategy=MonolithicSingleStreamGPUScheduler()).apply_pass(sdfg, {})

    syncs = _sync_tasklets(sdfg)
    assert len(syncs) == 1, (f"Monolithic on the H2D+D2H state should emit exactly one host-boundary sync; "
                             f"got {len(syncs)}.")


def test_pipeline_wires_connector_for_pre_expanded_runtime_tasklet():
    """Pipeline wires a ``gpuStream_t`` in-connector onto every pre-expanded runtime tasklet."""
    sdfg = _build_h2d_d2h_pre_expanded_sdfg()
    GPUStreamPipeline().apply_pass(sdfg, {})
    for tasklet, _ in _runtime_tasklets(sdfg):
        assert any(
            t == dace.dtypes.gpuStream_t
            for t in tasklet.in_connectors.values()), (f"Pre-expanded runtime tasklet '{tasklet.label}' must carry a "
                                                       f"gpuStream_t in-connector after the pipeline runs.")
