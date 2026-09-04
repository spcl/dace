# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

:class:`GPUCodegenPreprocessPipeline` is the codegen target's one-shot
codegen-preparation pipeline; :class:`GPUStreamPipeline` runs just the stream
scheduler + wirer on a post-expansion SDFG. Both act on the root SDFG only.
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

    Pass ``scheduling_strategy=<instance>`` to override the default
    :class:`NaiveGPUStreamScheduler`. Expects a post-expansion SDFG (libnodes
    flattened via ``sdfg.expand_library_nodes(recursive=True)``).

    Scheduling is idempotent (gpu_stream_id persisted per node) and wiring is
    single-shot (gated by :func:`is_stream_wiring_applied`), so each pass owns its
    own re-entry semantics and no pipeline-level guard is needed.
    """

    def __init__(self, scheduling_strategy: Optional[GPUStreamSchedulingStrategy] = None):
        if scheduling_strategy is None:
            # Codegen owns synchronize_on_exit and hands it to the strategy (which falls
            # back to the same config when given None).
            scheduling_strategy = AutoSingleStreamGPUScheduler(
                synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        elif not isinstance(scheduling_strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"scheduling_strategy must be a GPUStreamSchedulingStrategy instance, "
                            f"got {type(scheduling_strategy).__name__}.")
        self._scheduling_strategy = scheduling_strategy
        super().__init__([scheduling_strategy, GPUStreamWiring(scheduling_strategy)])


class GPUCodegenPreprocessPipeline(Pipeline):
    """One-shot GPU-codegen preparation.

    Declarative ordering of every transformation that brings an SDFG to a state the experimental
    CUDA codegen can emit. See the constructor for the non-obvious sequencing constraints.
    """

    def __init__(self):
        # Local imports: avoid circular import in ``dace.transformation`` package init.
        # ``AssignmentAndCopyKernelToMemsetAndMemcpy`` is deliberately NOT imported here any more:
        # extended moved it into the canonicalize pipeline, so running it here too would apply it
        # twice.
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (
            AddThreadBlockMaps, ExpandLibraryNodes, NormalizeHostLevelGPUSchedules, NormalizeHostLevelGPUSchedulesEarly,
            ReinferConnectorTypes, SynchronizeStreamUnawareGPUCallbacks)
        from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
            InsertExplicitGPUGlobalMemoryCopies)
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import PromoteGPUScalarsToArrays
        from dace.transformation.passes.demote_kernel_internal_arrays_to_scalars import (
            DemoteKernelInternalArraysToScalars)
        from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering
        from dace.transformation.passes.gpu_specialization.promote_warp_tiles import PromoteWarpTiles
        # Order constraints (why each pass sits where it does):
        #   * ``NestedGPUDeviceMapLowering`` first: flattens nested ``GPU_Device`` maps into one
        #     kernel; every downstream pass assumes one-level kernels.
        #   * scheduler after ``ExpandLibraryNodes``: it walks real kernel/runtime-call nodes and
        #     would miss opaque libnodes.
        #   * ``PromoteWarpTiles`` before ``AddThreadBlockMaps``: it redeems the ``is_warp_tile`` tag
        #     the device offload had to ignore, and a kernel that gets its thread-block level that
        #     way must not then be given a second one.
        #   * ``AddThreadBlockMaps`` after the transient hoist in ``InsertExplicitGPUGlobalMemoryCopies``:
        #     tiling first leaks the inner-map outer-loop symbol into host-side ``cudaMalloc`` sizes.
        #   * ``DemoteKernelInternalArraysToScalars`` after structure is final and before
        #     ``ReinferConnectorTypes``: it scalarizes length-1 arrays and resets connectors, which
        #     re-inference then re-derives as scalar references.
        #   * ``SynchronizeStreamUnawareGPUCallbacks`` after wiring: it fences host callbacks the
        #     stream scheduler cannot order, and its sync tasklet takes no stream connector.
        #   * ``ReinferConnectorTypes`` last: earlier passes mutate NestedSDFG-connector descriptors,
        #     so connector types must be re-derived for correct codegen signatures.
        # Scheduling writes ``Node.gpu_stream_id``; wiring reads it and lays down the
        # ``gpu_streams`` array + connector + sync wiring.
        strategy = AutoSingleStreamGPUScheduler(
            synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            # Fix host-level tasklets touching GPU data that an expansion performed *before* the
            # pipeline already left behind; see the class docstring for why this runs twice.
            NormalizeHostLevelGPUSchedulesEarly(),
            NestedGPUDeviceMapLowering(),
            PromoteGPUScalarsToArrays(),
            InsertExplicitGPUGlobalMemoryCopies(),
            ExpandLibraryNodes(),
            # After all expansions: repair host-level maps that inherited kernel-internal
            # schedules (GPU_ThreadBlock & co.) through nested library expansions -- they are
            # kernels and must be GPU_Device before stream scheduling and tiling see them.
            NormalizeHostLevelGPUSchedules(),
            strategy,
            GPUStreamWiring(strategy),
            SynchronizeStreamUnawareGPUCallbacks(),
            LiftSharedOutOfNestedSDFG(),
            PromoteWarpTiles(),
            AddThreadBlockMaps(),
            DemoteKernelInternalArraysToScalars(),
            ReinferConnectorTypes(),
        ])
