# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU specialization pipelines.

:class:`GPUCodegenPreprocessPipeline` is the codegen target's one-shot
codegen-preparation pipeline. :class:`GPUStreamPipeline` is a lower-level
entry point that runs just the stream-scheduling strategy on a
post-expansion SDFG. Both are single-shot and act on the root SDFG only.
"""
import warnings
from typing import Any, Dict, Optional

from dace import SDFG
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (GPUStreamSchedulingStrategy,
                                                                                 NaiveGPUStreamScheduler)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_lowering_applied
from dace.transformation.passes.gpu_specialization.lift_shared_out_of_nsdfg import LiftSharedOutOfNestedSDFG
from dace.transformation.passes.gpu_specialization.stream_scheduling import LastWriterDFSStreamScheduler
from dace.transformation.passes.promote_gpu_scalars_to_arrays import InferDefaultSchedulesAndStorages


class GPUStreamPipeline(Pipeline):
    """Post-expansion GPU stream lowering, parametrised by scheduling strategy.

    Pass ``scheduling_strategy=<instance>`` to swap in a different
    strategy (default :class:`NaiveGPUStreamScheduler`). Expects a
    post-expansion SDFG -- libnodes must be flattened upstream via
    ``sdfg.expand_library_nodes(recursive=True)``.
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
                "GPUStreamPipeline: skipping re-application -- the SDFG already has the "
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

    Declarative ordering of every transformation that brings an SDFG to a state the experimental
    CUDA codegen can emit. See the constructor for the non-obvious sequencing constraints.
    """

    def __init__(self):
        # Imports done locally to avoid the circular-import dance in
        # ``dace.transformation`` package init.
        from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
            AssignmentAndCopyKernelToMemsetAndMemcpy)
        from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (AddThreadBlockMaps,
                                                                                             ExpandLibraryNodes,
                                                                                             ReinferConnectorTypes)
        from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
            InsertExplicitGPUGlobalMemoryCopies)
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import PromoteGPUScalarsToArrays
        # Order constraints:
        #   * ``AssignmentAndCopyKernelToMemsetAndMemcpy`` before the stream scheduler: it moves
        #     the map's dynamic-input edges onto the new libnode and a pre-wired ``__stream``
        #     connector would clash.
        #   * Stream scheduler after ``ExpandLibraryNodes``: the scheduler walks real
        #     kernel/runtime-call nodes and would miss opaque libnodes.
        #   * ``AddThreadBlockMaps`` after the kernel-internal transient hoist (in
        #     ``InsertExplicitGPUGlobalMemoryCopies``): tiling first leaks the inner-map outer-loop
        #     symbol into host-side ``cudaMalloc`` size expressions for hoisted transients.
        #   * ``ReinferConnectorTypes`` last: earlier passes mutate descriptors under NestedSDFG
        #     connectors, so connector types must be re-derived for correct codegen signatures.
        super().__init__([
            InferDefaultSchedulesAndStorages(),
            PromoteGPUScalarsToArrays(),
            AssignmentAndCopyKernelToMemsetAndMemcpy(),
            InsertExplicitGPUGlobalMemoryCopies(),
            ExpandLibraryNodes(),
            LastWriterDFSStreamScheduler(),
            LiftSharedOutOfNestedSDFG(),
            AddThreadBlockMaps(),
            ReinferConnectorTypes(),
        ])
