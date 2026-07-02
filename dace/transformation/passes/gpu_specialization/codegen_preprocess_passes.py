# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapper :class:`Pass` classes exposing the ``experimental_cuda.preprocess`` steps as composable
Pipeline members, so codegen-preprocess ordering is declarative and testable.
"""
from typing import Any, Dict, Optional

from dace import SDFG, dtypes, nodes, properties
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class ExpandLibraryNodes(ppl.Pass):
    """Recursive :meth:`SDFG.expand_library_nodes` as a Pipeline Pass."""

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.Symbols)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[bool]:
        from dace.sdfg import infer_types
        sdfg.expand_library_nodes(recursive=True)
        # Expansion can spawn fresh NSDFGs whose inner Maps still carry
        # ``ScheduleType.Default``; the codegen dispatcher rejects those.
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        return True


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeHostLevelGPUSchedules(ppl.Pass):
    """Reset GPU kernel-internal schedules on maps that have no enclosing GPU kernel.

    Library-node expansion seeds the expanded NestedSDFG with the library node's schedule
    (``ExpandTransformation.apply``: ``set_default_schedule_and_storage_types(..., [node.schedule],
    True)``). For a host-level GPU library node (e.g. an ONNX MatMul whose expansion nests
    einsum -> Gemm SDFGs), the recursion maps ``SCOPEDEFAULT_SCHEDULE[GPU_Device]`` to
    ``GPU_ThreadBlock`` for maps in *deeper* nested SDFGs -- even though those maps run on the
    host and launch their own kernels. Such maps (e.g. ``gemm_init_map``) are then invisible to
    ``AddThreadBlockMaps``/``InferGPUGridAndBlockSize`` (which only consider ``GPU_Device``) and
    crash code generation.

    A thread-block-scoped map with no enclosing GPU kernel is, semantically, a kernel: reschedule
    it to ``GPU_Device`` so the regular tiling/launch-dimension pipeline picks it up.

    The same expansion path can also leave bare *tasklets* at the host level that read/write
    ``GPU_Global`` data (e.g. the ONNX Conv expansion's C++ loop-nest tasklet): host code then
    dereferences device pointers and crashes at runtime. Those tasklets are wrapped in a
    one-iteration ``GPU_Device`` map (mirroring ``GPUTransformSDFG`` step 7, which performs the
    same wrapping for graphs whose expansion happened *before* the GPU transform).

    Finally, host-level access-to-access *WCR copies* over GPU-resident data (e.g. the ONNX
    Reshape backward's gradient accumulation) lower to a host-side ``CopyND::Accumulate`` loop
    over device pointers, which also crashes at runtime. Those edges are converted to
    ``GPU_Device`` copy maps (via :func:`~dace.sdfg.memlet_utils.memlet_to_map`) with the WCR
    re-attached, so the accumulation runs as an atomic device kernel.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[bool]:
        from dace.sdfg.scope import is_devicelevel_gpu
        from dace.transformation.helpers import wrap_code_node_in_unit_gpu_map
        from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
            is_already_lowered_gpu_runtime_call, is_pipeline_sync_tasklet)
        kernel_internal_schedules = (dtypes.ScheduleType.GPU_ThreadBlock,
                                     dtypes.ScheduleType.GPU_ThreadBlock_Dynamic, dtypes.ScheduleType.GPU_Warp)
        gpu_storage = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned)
        modified = False

        host_gpu_tasklets = []
        for node, state in sdfg.all_nodes_recursive():
            if (isinstance(node, nodes.MapEntry) and node.map.schedule in kernel_internal_schedules
                    and not is_devicelevel_gpu(state.parent, state, node)):
                node.map.schedule = dtypes.ScheduleType.GPU_Device
                modified = True
            elif (isinstance(node, nodes.Tasklet) and state.entry_node(node) is None
                  and not is_devicelevel_gpu(state.parent, state, node)
                  # Host-side GPU runtime-call tasklets (cudaMemcpyAsync launchers, sync
                  # tasklets) legitimately touch GPU data from the host -- leave them be.
                  and not is_already_lowered_gpu_runtime_call(node) and not is_pipeline_sync_tasklet(node)):
                touches_gpu_data = any(
                    not e.data.is_empty() and state.parent.arrays[e.data.data].storage in gpu_storage
                    for e in state.all_edges(node))
                if touches_gpu_data:
                    host_gpu_tasklets.append((state, node))

        # Wrap after the scan: wrapping mutates the graphs being traversed.
        for state, node in host_gpu_tasklets:
            wrap_code_node_in_unit_gpu_map(state, node)
            modified = True

        # Host-level WCR copies over GPU-resident data -> GPU copy maps with the WCR re-attached.
        from dace.sdfg import memlet_utils as mutils
        wcr_copies = []
        for cursdfg in sdfg.all_sdfgs_recursive():
            for state in cursdfg.states():
                for e in state.edges():
                    if (e.data.wcr is not None and not e.data.is_empty() and isinstance(e.src, nodes.AccessNode)
                            and isinstance(e.dst, nodes.AccessNode) and state.entry_node(e.dst) is None
                            and not is_devicelevel_gpu(state.parent, state, e.dst)
                            and (cursdfg.arrays[e.src.data].storage in gpu_storage
                                 or cursdfg.arrays[e.dst.data].storage in gpu_storage)
                            and mutils.can_memlet_be_turned_into_a_map(
                                edge=e, state=state, sdfg=cursdfg, ignore_strides=True)):
                        wcr_copies.append((cursdfg, state, e))
        for cursdfg, state, e in wcr_copies:
            wcr = e.data.wcr
            _, mx = mutils.memlet_to_map(edge=e, state=state, sdfg=cursdfg, ignore_strides=True)
            # memlet_to_map builds plain overwrite memlets; restore the accumulation
            for out_e in state.in_edges(mx) + state.out_edges(mx):
                out_e.data.wcr = wcr
            modified = True

        return modified or None


@properties.make_properties
@transformation.explicit_cf_compatible
class AddThreadBlockMaps(ppl.Pass):
    """Tile every ``GPU_Device`` map lacking an inner ``GPU_ThreadBlock`` map (via
    :class:`AddThreadBlockMap`) and infer the resulting ``(grid, block)`` dimensions.

    Returns ``{'kernel_dimensions_map': ..., 'tb_inserted_kernels': set(MapEntry)}`` in
    ``pipeline_results``. Tiled late on purpose: tiling first leaks the inner-map outer-loop
    symbol into host-side ``cudaMalloc`` size expressions for kernel-hoisted transients.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
        from dace.transformation.passes.analysis.infer_gpu_grid_and_block_size import InferGPUGridAndBlockSize

        old_nodes = set(node for node, _ in sdfg.all_nodes_recursive())
        sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)
        new_nodes = set(node for node, _ in sdfg.all_nodes_recursive()) - old_nodes
        tb_inserted_kernels = {
            n
            for n in new_nodes if isinstance(n, nodes.MapEntry) and n.schedule == dtypes.ScheduleType.GPU_Device
        }
        kernel_dimensions_map = InferGPUGridAndBlockSize().apply_pass(sdfg, tb_inserted_kernels) or {}
        return {
            'kernel_dimensions_map': kernel_dimensions_map,
            'tb_inserted_kernels': tb_inserted_kernels,
        }


@properties.make_properties
@transformation.explicit_cf_compatible
class ReinferConnectorTypes(ppl.Pass):
    """Clear and re-derive NestedSDFG connector types from their inner descriptors.

    Earlier passes mutate descriptors (e.g. ``PromoteGPUScalarsToArrays`` widens a ``Scalar`` to a
    length-1 ``Array``), leaving stale scalar-typed connectors that miscompile (``T name`` vs.
    ``name[0]``). Re-inference makes them pointer-typed.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Connectors | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        from dace.sdfg import infer_types
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import invalidate_array_connectors
        invalidate_array_connectors(sdfg)
        for nsdfg in sdfg.all_sdfgs_recursive():
            infer_types.infer_connector_types(nsdfg)
        return None
