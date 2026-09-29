# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines, both acting on the root SDFG only:
:class:`GPUCodegenPreprocessPipeline` prepares an SDFG for the experimental codegen, and
:class:`GPUStreamPipeline` runs just the stream scheduler and wirer on a post-expansion SDFG.
:func:`gpu_specialize_offloaded` resolves the device schedules of an offloaded graph
(``finalize_for_target('gpu')`` calls it).
"""
from typing import Optional

from dace import SDFG
from dace.sdfg import nodes
from dace.config import Config
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (AutoSingleStreamGPUScheduler,
                                                                                 GPUStreamSchedulingStrategy)
from dace.transformation.passes.gpu_specialization.gpu_stream_wiring import GPUStreamWiring


def gpu_specialize_offloaded(sdfg: SDFG) -> SDFG:
    """Resolve the device schedules of an offloaded ``sdfg``, in place.

    :param sdfg: An offloaded SDFG.
    :returns: The same ``sdfg`` instance.
    """
    from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
    from dace.transformation.passes.gpu_specialization.block_tile_kernels import BlockTileKernels
    from dace.transformation.passes.gpu_specialization.contiguous_axis_to_threads import ContiguousAxisToThreads
    from dace.transformation.passes.gpu_specialization.promote_host_maps_to_kernels import PromoteHostMapsToKernels
    from dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes import (
        SequentializeNestedDeviceScopes)
    # A loop around a single-iteration kernel launches it once per trip; running the loop inside is one launch.
    MoveLoopIntoMapGated(target='gpu', single_iteration_only=True).apply_pass(sdfg, {})
    # A ``specialize`` node (MatMul) becomes the node it stands for during implementation selection;
    # becoming it here lets the promotion below see what will actually run.
    for node, state in list(sdfg.all_nodes_recursive()):
        if isinstance(node, nodes.LibraryNode) and (node.implementation or node.default_implementation) == 'specialize':
            node.expand(state)
    # A host map that only launches device work is the kernel; everything below resolves its nesting.
    PromoteHostMapsToKernels().apply_pass(sdfg, {})
    # For ``map JK { work; map JL }``, ContiguousAxisToThreads makes JL a thread dimension. It runs first
    # because SequentializeNestedDeviceScopes would otherwise pin JL sequential.
    ContiguousAxisToThreads().apply_pass(sdfg, {})
    # What stays nested gets a block per outer iteration and its inner maps across the lanes, where
    # running the body once per lane is sound; the rest is pinned sequential below.
    BlockTileKernels().apply_pass(sdfg, {})
    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})
    return sdfg


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
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (
            AddThreadBlockMaps, ExpandLibraryNodes, InferDefaultSchedulesAndStorages, ReinferConnectorTypes,
            SynchronizeStreamUnawareGPUCallbacks)
        from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
        from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel
        from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays
        from dace.transformation.passes.demote_kernel_internal_arrays_to_scalars import (
            DemoteKernelInternalArraysToScalars)
        from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering
        from dace.transformation.passes.gpu_specialization.promote_warp_tiles import PromoteWarpTiles
        # Order constraints:
        #   * NestedGPUDeviceMapLowering first -- everything downstream assumes one-level kernels.
        #   * scheduler after ExpandLibraryNodes -- it would miss opaque libnodes.
        #   * PromoteWarpTiles before AddThreadBlockMaps -- canonicalize promotes a pending ``is_warp_tile``
        #     before choosing a block size; a graph that reaches codegen with one still pending gets its
        #     thread-block level from the tile, not a second one on top of it.
        #   * AddThreadBlockMaps after the MoveArrayOutOfKernel hoist -- tiling first leaks the
        #     inner-map outer-loop symbol into host-side cudaMalloc sizes.
        #   * DemoteKernelInternalArraysToScalars before ReinferConnectorTypes -- it resets the
        #     connectors that re-inference then re-derives as scalar references.
        #   * SynchronizeStreamUnawareGPUCallbacks after wiring -- its fence takes no stream connector.
        #   * ReinferConnectorTypes last -- earlier passes mutate NestedSDFG connector descriptors.
        strategy = AutoSingleStreamGPUScheduler(
            synchronize_on_exit=Config.get('compiler', 'cuda', 'synchronize_on_exit'))
        scalar_promotion = PromoteScalarOutputsToArrays()
        scalar_promotion.gpu = True
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            NestedGPUDeviceMapLowering(),
            scalar_promotion,
            MoveArrayOutOfKernel(),
            InsertExplicitCopies(),
            ExpandLibraryNodes(),
            strategy,
            GPUStreamWiring(strategy),
            SynchronizeStreamUnawareGPUCallbacks(),
            PromoteWarpTiles(),
            AddThreadBlockMaps(),
            DemoteKernelInternalArraysToScalars(),
            ReinferConnectorTypes(),
        ])
