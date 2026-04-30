# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

The codegen target invokes :class:`GPUCodegenPreprocessPipeline` once
during ``preprocess``. That pipeline composes every transformation
needed to bring an SDFG to a state the experimental CUDA codegen can
emit code for; the order is declarative and post-expansion-first (no
two-phase wiring around ``expand_library_nodes`` any more).

Two lower-level entry points remain for callers that want to invoke
just the stream pass:

* :class:`GPUStreamPipeline` — wraps one
  :class:`GPUStreamSchedulingStrategy` (default
  :class:`NaiveGPUStreamScheduler`; pass ``scheduling_strategy=…`` to
  opt into another, e.g. :class:`MonolithicSingleStreamGPUScheduler`).

  The strategy owns end-to-end stream lowering: assigns streams,
  allocates ``gpu_streams`` (with propagation to nested SDFGs that need
  it), wires every consumer's stream connector, and emits sync tasklets
  per its own placement policy. Operates on a *post-expansion* SDFG —
  callers must run ``sdfg.expand_library_nodes(recursive=True)`` first
  if their input still contains library nodes.

  Stream scheduling is single-shot. Re-application is rejected with a
  warning — the WCC partition is graph-shape-dependent and re-running
  on an already-wired SDFG would corrupt the wiring. Nested SDFGs share
  the root's decisions; calling the pipeline on a non-root SDFG raises.

* :class:`GPUCodegenPreprocessPipeline` — the codegen-target end-to-end
  pipeline. Composes scheduling/storage inference, scalar promotion,
  implicit-copy lifting, library expansion, stream scheduling,
  Shared-transient lifting, threadblock tiling, and connector-type
  re-inference in one declarative sequence.
"""
import warnings
from typing import Any, Dict, Optional

from dace import SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (GPUStreamSchedulingStrategy,
                                                                                 NaiveGPUStreamScheduler)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_lowering_applied
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages


class GPUStreamPipeline(Pipeline):
    """Post-expansion GPU stream lowering, parametrised by scheduling strategy.

    Pass ``scheduling_strategy=<instance>`` to swap in a different
    strategy (e.g. :class:`MonolithicSingleStreamGPUScheduler`).

    Expects a *post-expansion* SDFG: every kernel ``MapEntry`` and
    runtime call Tasklet (e.g. ``cudaMemcpyAsync``) is visible at its
    final position. The strategy walks them directly, so libnodes must
    have been flattened upstream (call
    ``sdfg.expand_library_nodes(recursive=True)`` if your input still
    contains them; :class:`GPUCodegenPreprocessPipeline` does this for
    you).
    """

    def __init__(self, scheduling_strategy: Optional[GPUStreamSchedulingStrategy] = None):
        if scheduling_strategy is None:
            scheduling_strategy = NaiveGPUStreamScheduler()
        elif not isinstance(scheduling_strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"scheduling_strategy must be a GPUStreamSchedulingStrategy instance, "
                            f"got {type(scheduling_strategy).__name__}.")
        self._scheduling_strategy = scheduling_strategy
        super().__init__([scheduling_strategy])

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        if is_gpu_lowering_applied(sdfg):
            warnings.warn(
                "GPUStreamPipeline: skipping re-application — the SDFG already has the "
                "``gpu_streams`` array, indicating the pipeline has run. Stream "
                "assignment is single-shot and re-running it would corrupt the wiring.",
                UserWarning,
                stacklevel=2)
            return {}
        if sdfg.parent_sdfg is not None:
            raise ValueError(f"GPUStreamPipeline: must run on the root SDFG. Got nested SDFG "
                             f"'{sdfg.name}' (parent '{sdfg.parent_sdfg.name}'). Nested SDFGs share "
                             "the root's decisions; do not invoke the pipeline on them.")
        return super().apply_pass(sdfg, pipeline_results)


# Legacy alias preserved so out-of-tree references keep working.
GPUSpecializationPipeline = GPUStreamPipeline


class GPUCodegenPreprocessPipeline(Pipeline):
    """One-shot GPU-codegen preparation.

    Composes every transformation that brings an SDFG to a state the
    experimental CUDA codegen can emit code for. Order is declarative;
    each step depends on the previous one's invariants but is otherwise
    independent. Replaces the previous hand-stitched sequence in
    ``ExperimentalCUDACodeGen.preprocess``.

    Steps (in order):

    1. :class:`InferDefaultSchedulesAndStorages` — fill in ``Default``
       schedules / storages everywhere.
    2. :class:`PromoteGPUScalarsToArrays` — promote scalars the kernel
       writes into length-1 Arrays (rule 1) and any other Scalar that
       cannot live on the GPU as a Scalar (rule 2).
    3. :class:`GPUStreamPipeline` — schedule streams, wire ``__stream``
       connectors on libnodes, emit sync tasklets. (Currently runs
       **before** library expansion; future iteration will move it
       after.)
    4. :class:`ExpandLibraryNodes` — wraps
       ``sdfg.expand_library_nodes(recursive=True)`` as a Pass.
    5. :class:`GPUPostExpansionPipeline` — reconnect internal GPU
       consumers of expansion-spawned NestedSDFGs to the inherited
       ``stream`` connector and lift ``GPU_Shared`` transients out of
       inner NestedSDFGs.
    6. :class:`AddThreadBlockMaps` — tile every ``GPU_Device`` map with
       an inner ``GPU_ThreadBlock``; computes the
       ``(grid, block)`` dimensions for codegen. Returns
       ``{'kernel_dimensions_map': …, 'tb_inserted_kernels': …}`` — the
       codegen target reads it from ``pipeline_results``.
    7. :class:`InvalidateAndInferConnectorTypes` — reset stale
       Array-vs-Scalar typings on NestedSDFGs and re-infer per
       sub-SDFG.
    8. ``DefaultSharedMemorySync`` (conditional) — auto-insert
       ``__syncthreads``. Pulled in via ``apply_pass`` callsite (uses a
       config flag) rather than via ``Pipeline`` membership.
    """

    def __init__(self):
        # Imports done locally to avoid the circular-import dance in
        # ``dace.transformation`` package init.
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (
            AddThreadBlockMaps, ExpandLibraryNodes, InvalidateAndInferConnectorTypes)
        from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
            InsertExplicitGPUGlobalMemoryCopies)
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import PromoteGPUScalarsToArrays
        # Order: lift implicit copies first (so libnodes exist), expand
        # libnodes, then stream-schedule on the post-expansion SDFG (the
        # strategy sees real kernels + runtime tasklets, not opaque
        # libnodes), then lift Shared transients out of NSDFGs spawned
        # by expansion, then tile late, then refresh connector types.
        # The stream-scheduling strategy is included directly (rather
        # than via :class:`GPUStreamPipeline`) — :class:`Pipeline` is
        # decorated as a ``@dataclass`` and therefore unhashable, so it
        # can't be a child of another :class:`Pipeline`. Strategies
        # extend :class:`Pass` directly and are hashable.
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            PromoteGPUScalarsToArrays(),
            InsertExplicitGPUGlobalMemoryCopies(),
            ExpandLibraryNodes(),
            NaiveGPUStreamScheduler(),
            LiftSharedOutOfNestedSDFG(),
            AddThreadBlockMaps(),
            InvalidateAndInferConnectorTypes(),
        ])
