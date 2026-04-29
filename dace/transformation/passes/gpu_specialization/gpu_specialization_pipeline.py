# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

The lowering is two-phase around ``expand_library_nodes``:

* :class:`GPUStreamPipeline` — pre-expansion stream lowering. Wraps a
  single :class:`GPUStreamSchedulingStrategy` (default
  :class:`NaiveGPUStreamScheduler`; pass ``scheduling_strategy=…`` to
  opt into another, e.g. :class:`MonolithicSingleStreamGPUScheduler`).

  The strategy owns end-to-end stream lowering: assigns streams,
  allocates ``gpu_streams`` (with propagation to nested SDFGs that need
  it), wires every consumer's stream connector, and emits sync tasklets
  per its own placement policy.

  Lifting implicit AccessNode→AccessNode GPU copies to ``CopyLibraryNode``
  instances is a *precondition* (not stream-management policy) and is
  pulled in automatically via the strategy's :meth:`depends_on`.

  Stream scheduling is single-shot. Re-application is rejected with a
  warning — the WCC partition is graph-shape-dependent and re-running on
  an already-wired SDFG would corrupt the wiring. Nested SDFGs share the
  root's decisions; calling the pipeline on a non-root SDFG raises.

* :class:`GPUPostExpansionPipeline` — post-expansion. Reconnects internal
  GPU consumers of NestedSDFGs born from library expansion to the
  inherited ``stream`` connector, and lifts ``GPU_Shared`` transients
  out of inner NestedSDFGs into the SDFG that owns the kernel
  ``MapEntry``. Idempotent.
"""
import warnings
from typing import Any, Dict, Optional

from dace import SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (GPUStreamSchedulingStrategy,
                                                                                 NaiveGPUStreamScheduler)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_lowering_applied
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG
from dace.transformation.passes.gpu_specialization.reconnect_within_expanded_sdfgs import ReconnectWithinExpandedSDFGs
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages


class GPUStreamPipeline(Pipeline):
    """Pre-expansion GPU stream lowering, parametrised by scheduling strategy.

    Pass ``scheduling_strategy=<instance>`` to swap in a different
    strategy (e.g. :class:`MonolithicSingleStreamGPUScheduler`). The
    strategy declares its own preconditions via :meth:`depends_on`; the
    pipeline framework pulls those in automatically (so callers don't
    have to remember to run e.g. ``InsertExplicitGPUGlobalMemoryCopies``
    upstream).
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
                UserWarning, stacklevel=2)
            return {}
        if sdfg.parent_sdfg is not None:
            raise ValueError(
                f"GPUStreamPipeline: must run on the root SDFG. Got nested SDFG "
                f"'{sdfg.name}' (parent '{sdfg.parent_sdfg.name}'). Nested SDFGs share "
                "the root's decisions; do not invoke the pipeline on them.")
        return super().apply_pass(sdfg, pipeline_results)


class GPUPostExpansionPipeline(Pipeline):
    """Post-expansion gpu_specialization lowering.

    Runs after ``expand_library_nodes(recursive=True)``. Idempotent —
    structurally a no-op on an already-lowered SDFG.

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


# Legacy alias preserved so out-of-tree references keep working.
GPUSpecializationPipeline = GPUStreamPipeline
