# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines, both acting on the root SDFG only:
:class:`GPUCodegenPreprocessPipeline` prepares an SDFG for the experimental codegen, and
:class:`GPUStreamPipeline` runs just the stream scheduler and wirer on a post-expansion SDFG.
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
    """Post-expansion GPU stream lowering: scheduling, then wiring.

    Expects libnodes already flattened via ``sdfg.expand_library_nodes(recursive=True)``. Each pass
    owns its re-entry semantics -- scheduling is idempotent, wiring is single-shot -- so the pipeline
    needs no guard of its own.
    """

    def __init__(self, scheduling_strategy: Optional[GPUStreamSchedulingStrategy] = None):
        if scheduling_strategy is None:
            scheduling_strategy = AutoSingleStreamGPUScheduler(
                synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        elif not isinstance(scheduling_strategy, GPUStreamSchedulingStrategy):
            raise TypeError(f"scheduling_strategy must be a GPUStreamSchedulingStrategy instance, "
                            f"got {type(scheduling_strategy).__name__}.")
        self._scheduling_strategy = scheduling_strategy
        super().__init__([scheduling_strategy, GPUStreamWiring(scheduling_strategy)])


class GPUCodegenPreprocessPipeline(Pipeline):
    """One-shot GPU-codegen preparation: every transformation that brings an SDFG to a state the
    experimental CUDA codegen can emit. The constructor documents the sequencing constraints."""

    def __init__(self):
        # Local imports: avoid circular import in ``dace.transformation`` package init.
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (AddThreadBlockMaps,
                                                                                             ExpandLibraryNodes,
                                                                                             ReinferConnectorTypes)
        from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
        from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import PromoteGPUScalarsToArrays
        from dace.transformation.passes.demote_kernel_internal_arrays_to_scalars import (
            DemoteKernelInternalArraysToScalars)
        from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering
        # Order constraints:
        #   * NestedGPUDeviceMapLowering first -- everything downstream assumes one-level kernels.
        #   * scheduler after ExpandLibraryNodes -- it would miss opaque libnodes.
        #   * AddThreadBlockMaps after the MoveArrayOutOfKernel hoist -- tiling first leaks the
        #     inner-map outer-loop symbol into host-side cudaMalloc sizes.
        #   * DemoteKernelInternalArraysToScalars before ReinferConnectorTypes -- it resets the
        #     connectors that re-inference then re-derives as scalar references.
        #   * ReinferConnectorTypes last -- earlier passes mutate NestedSDFG connector descriptors.
        strategy = AutoSingleStreamGPUScheduler(
            synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            NestedGPUDeviceMapLowering(),
            PromoteGPUScalarsToArrays(),
            MoveArrayOutOfKernel(),
            InsertExplicitCopies(),
            ExpandLibraryNodes(),
            strategy,
            GPUStreamWiring(strategy),
            LiftSharedOutOfNestedSDFG(),
            AddThreadBlockMaps(),
            DemoteKernelInternalArraysToScalars(),
            ReinferConnectorTypes(),
        ])
