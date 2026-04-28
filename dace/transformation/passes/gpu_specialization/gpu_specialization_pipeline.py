# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

The lowering is two-phase around ``expand_library_nodes``:

* :class:`GPUStreamPipeline` — pre-expansion. Lifts implicit AccessNode
  copies to explicit ``CopyLibraryNode`` instances, schedules backend GPU
  streams, threads the ``gpu_streams`` array through every SDFG that
  hosts kernel launches, wires per-kernel ``__stream_<i>`` connectors,
  and emits stream synchronization tasklets. Sets the
  ``is_gpu_lowering_applied`` signal (presence of the ``gpu_streams``
  transient) and gates on it for re-application idempotency.

* :class:`GPUPostExpansionPipeline` — post-expansion. Reconnects internal
  GPU consumers of NestedSDFGs born from library expansion to the
  inherited ``stream`` connector, and lifts ``GPU_Shared`` transients
  out of inner NestedSDFGs into the SDFG that owns the kernel
  ``MapEntry``. Structurally idempotent — every sub-pass tolerates a
  re-run as a no-op.

A standalone caller can invoke the two pipelines around an
``expand_library_nodes(recursive=True)`` call to reach the same shape
the codegen produces. ``GPUSpecializationPipeline`` (legacy name) is
kept as a thin alias for :class:`GPUStreamPipeline` so external
references don't break.
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
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG
from dace.transformation.passes.gpu_specialization.reconnect_within_expanded_sdfgs import ReconnectWithinExpandedSDFGs
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages


class GPUStreamPipeline(Pipeline):
    """Pre-expansion gpu_specialization lowering with built-in idempotency.

    Re-application short-circuits via the ``is_gpu_lowering_applied``
    signal: pre-applying this pipeline and then letting the codegen's
    preprocess re-run it would otherwise double-add the stream array,
    double-thread connectors, and corrupt per-stream chains."""

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


class GPUPostExpansionPipeline(Pipeline):
    """Post-expansion gpu_specialization lowering.

    Runs after ``expand_library_nodes(recursive=True)``. Sub-passes are
    structurally idempotent — re-running on an already-lowered SDFG is a
    no-op (no inner Shared transients with ``transient=True`` remain, no
    expanded-libnode NSDFGs without consumers remain, ``Default``
    schedules already replaced) — so no explicit re-entry guard is
    necessary here.

    ``InferDefaultSchedulesAndStorages`` is re-run because expansions
    born from ``expand_library_nodes`` (e.g. ``MappedTasklet`` for a
    CPU↔CPU copy) emit fresh maps with ``ScheduleType.Default`` that
    would otherwise reach codegen unlowered.
    """

    def __init__(self):
        super().__init__([
            ReconnectWithinExpandedSDFGs(),
            LiftSharedOutOfNestedSDFG(),
            InferDefaultSchedulesAndStorages(),
        ])


# Legacy alias preserved so out-of-tree references to the old name keep
# working. New code should pick :class:`GPUStreamPipeline` (pre-expansion)
# or :class:`GPUPostExpansionPipeline` (post-expansion) explicitly.
GPUSpecializationPipeline = GPUStreamPipeline
