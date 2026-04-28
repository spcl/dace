# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``GPUSpecializationPipeline`` — the canonical sequence of passes that
lowers an SDFG to a GPU-codegen-ready form (explicit copies, GPU streams,
stream synchronization tasklets).

Idempotency: detects prior application via the ``gpu_streams`` transient
(created by ``InsertGPUStreams`` as the pipeline's first state-modifying
step) and short-circuits on a re-run. Pre-applying the pipeline and then
letting the codegen's preprocess re-run it would otherwise double-add
the stream array, double-thread connectors, and corrupt per-stream chains
— producing runtime memory faults at execution time.
"""
from typing import Any, Dict

from dace import SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_lowering_applied
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams


class GPUSpecializationPipeline(Pipeline):
    """The standard gpu_specialization lowering, with built-in idempotency."""

    def __init__(self):
        super().__init__([
            InsertExplicitGPUGlobalMemoryCopies(),
            NaiveGPUStreamScheduler(),
            InsertGPUStreams(),
            ConnectGPUStreamsToNodes(),
            InsertGPUStreamSyncTasklets(),
        ])

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        if is_gpu_lowering_applied(sdfg):
            return {}
        return super().apply_pass(sdfg, pipeline_results)
