# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

:class:`GPUCodegenPreprocessPipeline` is the codegen target's one-shot
codegen-preparation pipeline. :class:`GPUStreamPipeline` is a lower-level
entry point that runs the stream scheduler + wirer on a post-expansion
SDFG. Both act on the root SDFG only.
"""
from typing import Optional

from dace.config import Config
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (AutoSingleStreamGPUScheduler,
                                                                                 GPUStreamSchedulingStrategy)
from dace.transformation.passes.gpu_specialization.gpu_stream_wiring import GPUStreamWiring
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages


class GPUStreamPipeline(Pipeline):
    """Post-expansion GPU stream lowering: scheduling -> wiring.

    Pass ``scheduling_strategy=<instance>`` to swap in a different scheduling
    strategy (default :class:`NaiveGPUStreamScheduler`). Expects a
    post-expansion SDFG -- libnodes must be flattened upstream via
    ``sdfg.expand_library_nodes(recursive=True)``.

    The scheduling pass is idempotent (gpu_stream_id is persisted per node);
    the wiring pass is single-shot, gated by
    :func:`is_stream_wiring_applied`. Pipeline-level guards are no longer
    needed -- each pass owns its own re-entry semantics.
    """

    def __init__(self, scheduling_strategy: Optional[GPUStreamSchedulingStrategy] = None):
        if scheduling_strategy is None:
            # The experimental codegen owns the synchronize_on_exit flag and hands it to the
            # scheduling strategy (the strategy itself falls back to the same config when given None).
            scheduling_strategy = AutoSingleStreamGPUScheduler(
                synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        elif not isinstance(scheduling_strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"scheduling_strategy must be a GPUStreamSchedulingStrategy instance, "
                            f"got {type(scheduling_strategy).__name__}.")
        self._scheduling_strategy = scheduling_strategy
        super().__init__([scheduling_strategy, GPUStreamWiring(scheduling_strategy)])


# Legacy alias preserved so out-of-tree references keep working.
GPUSpecializationPipeline = GPUStreamPipeline


class GPUCodegenPreprocessPipeline(Pipeline):
    """One-shot GPU-codegen preparation.

    Declarative ordering of every transformation that brings an SDFG to a state the experimental
    CUDA codegen can emit. See the constructor for the non-obvious sequencing constraints.
    """

    def __init__(self):
        # Imports done locally to avoid the circular-import dance in
        # ``dace.transformation`` package init.
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (AddThreadBlockMaps,
                                                                                             ExpandLibraryNodes,
                                                                                             ReinferConnectorTypes)
        from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
            InsertExplicitGPUGlobalMemoryCopies)
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import PromoteGPUScalarsToArrays
        from dace.transformation.passes.demote_kernel_internal_arrays_to_scalars import (
            DemoteKernelInternalArraysToScalars)
        from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering
        # Order constraints:
        #   * ``NestedGPUDeviceMapLowering`` first: flattens nested ``GPU_Device`` maps into a
        #     single kernel (union of the inner ranges on the outer map + a per-body bound-check
        #     NestedSDFG). Every downstream pass -- scalar promotion, copy insertion, stream
        #     scheduling, thread-block tiling -- assumes one-level kernels.
        #   * ``NaiveGPUStreamScheduler`` after ``ExpandLibraryNodes``: the scheduler walks real
        #     kernel/runtime-call nodes and would miss opaque libnodes.
        #   * ``AddThreadBlockMaps`` after the kernel-internal transient hoist (in
        #     ``InsertExplicitGPUGlobalMemoryCopies``): tiling first leaks the inner-map outer-loop
        #     symbol into host-side ``cudaMalloc`` size expressions for hoisted transients.
        #   * ``DemoteKernelInternalArraysToScalars`` after the structure is final (post stream
        #     wiring / tiling) and before ``ReinferConnectorTypes``: it scalarizes kernel-internal
        #     length-1 arrays and resets their connectors, which the final re-inference then
        #     re-derives as scalar references.
        #   * ``ReinferConnectorTypes`` last: earlier passes mutate descriptors under NestedSDFG
        #     connectors, so connector types must be re-derived for correct codegen signatures.
        # Scheduling pass writes ``Node.gpu_stream_id``; wiring pass reads it
        # and lays down the ``gpu_streams`` array + connector + sync wiring.
        strategy = AutoSingleStreamGPUScheduler(
            synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            NestedGPUDeviceMapLowering(),
            PromoteGPUScalarsToArrays(),
            InsertExplicitGPUGlobalMemoryCopies(),
            ExpandLibraryNodes(),
            # NOTE: This is a bit strange since the `strategy` is called on its own and then as
            #   a proper strategy, i.e. a subobject encapsulating something. Data should be
            #   transmitted through the second argument of `apply_pass()`. Also this explains
            #   the need for the ``naive_fallback`  property.
            strategy,
            GPUStreamWiring(strategy),
            LiftSharedOutOfNestedSDFG(),
            AddThreadBlockMaps(),
            DemoteKernelInternalArraysToScalars(),
            ReinferConnectorTypes(),
        ])
