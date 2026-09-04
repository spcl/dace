# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapper :class:`Pass` classes exposing ``experimental_cuda.preprocess`` steps as composable Pipeline
members so codegen-preprocess ordering is declarative and testable."""
import warnings
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
        # Expansion can spawn NSDFGs whose inner Maps carry ``ScheduleType.Default``; codegen rejects those.
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
    one-iteration ``GPU_Device`` map (as the offloading does for a host-level free tasklet, which performs the
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
            is_already_lowered_gpu_runtime_call, is_host_callback_tasklet, is_pipeline_sync_tasklet)
        kernel_internal_schedules = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
                                     dtypes.ScheduleType.GPU_Warp)
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
                  and not is_already_lowered_gpu_runtime_call(node) and not is_pipeline_sync_tasklet(node)
                  # A host callback invokes a host function pointer: it must stay on the host, and is
                  # ordered against the streams by ``SynchronizeStreamUnawareGPUCallbacks`` instead.
                  and not is_host_callback_tasklet(node)):
                touches_gpu_data = any(not e.data.is_empty() and state.parent.arrays[e.data.data].storage in gpu_storage
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
class NormalizeHostLevelGPUSchedulesEarly(NormalizeHostLevelGPUSchedules):
    """:class:`NormalizeHostLevelGPUSchedules`, scheduled before nested-map lowering.

    The normalization is needed twice. An expansion that ran *before* the pipeline (the ONNX
    Dropout expansion, for one) can leave a host-level tasklet reading GPU-resident data, so the
    SDFG is already invalid on entry; ``NestedGPUDeviceMapLowering`` validates the whole graph at
    the end of its own pass and reports that violation before the normal, post-expansion instance
    can wrap the tasklet. Running it once up front fixes the entry state, and the later instance
    still handles what library expansion introduces.

    A subclass only because ``Pipeline`` rejects two passes of the same type.
    """


@properties.make_properties
@transformation.explicit_cf_compatible
class SynchronizeStreamUnawareGPUCallbacks(ppl.Pass):
    """Fence host callbacks that touch GPU memory without being stream-aware.

    Such a callback enqueues its own device work (a cupy kernel, say) on a stream this SDFG knows
    nothing about, so the asynchronous copies the pipeline emits around it on ``gpu_streams`` are
    unordered with respect to it. The legacy CUDA codegen forces the callback's whole component onto
    the null stream; the experimental stream pipeline has no per-node null-stream slot, so a device-
    wide synchronization tasklet is inserted between the callback and its successors instead, which
    orders the callback against every stream.

    A callback that names ``__dace_current_stream`` orders itself and is left alone.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[bool]:
        from dace.codegen import common
        from dace.sdfg.scope import is_devicelevel_gpu
        from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (DEVICE_SYNC_TASKLET_LABEL,
                                                                                       STREAM_CONNECTOR,
                                                                                       dependency_edge,
                                                                                       is_host_callback_tasklet)
        gpu_storage = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned)

        targets = []
        for cursdfg in sdfg.all_sdfgs_recursive():
            for state in cursdfg.states():
                for node in state.nodes():
                    if not is_host_callback_tasklet(node) or is_devicelevel_gpu(cursdfg, state, node):
                        continue
                    if STREAM_CONNECTOR in node.code.as_string:
                        continue
                    if not any(not e.data.is_empty() and cursdfg.arrays[e.data.data].storage in gpu_storage
                               for e in state.all_edges(node)):
                        continue
                    # Re-application must not stack a second fence (or repeat the warning).
                    if any(succ.label == DEVICE_SYNC_TASKLET_LABEL for succ in state.successors(node)):
                        continue
                    targets.append((state, node))

        backend = common.get_gpu_backend()
        for state, node in targets:
            warnings.warn(
                f'Callback "{node.label}" accesses GPU memory but is not stream-aware, so a full device '
                'synchronization is emitted after it. Add a "dace.current_stream" argument to the callback '
                'and use it (e.g. cupy ExternalStream) to keep the work asynchronous.', UserWarning)
            sync = state.add_tasklet(name=DEVICE_SYNC_TASKLET_LABEL,
                                     inputs=set(),
                                     outputs=set(),
                                     code=f'DACE_GPU_CHECK({backend}DeviceSynchronize());',
                                     language=dtypes.Language.CPP,
                                     side_effects=True)
            # Every new edge gets its own Memlet: never hand two live edges the same object.
            for succ in list(state.successors(node)):
                state.add_edge(sync, None, succ, None, dependency_edge())
            state.add_edge(node, None, sync, None, dependency_edge())

        return bool(targets) or None


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
        # ``Modifies`` has no ``Connectors`` flag; connectors live on the code nodes that carry
        # them. ``infer_connector_types`` retypes ANY dataflow node's connectors -- map entries
        # and exits included -- so this must be ``Nodes``, not just tasklets and nested SDFGs;
        # under-declaring would stop a downstream ``should_reapply(Modifies.Scopes)`` from firing.
        # ``Descriptors`` is kept as a conservative over-declaration (the pass only reads them,
        # but over-declaring costs re-runs, never correctness).
        return ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _connector_types(sdfg: SDFG) -> Dict[Any, Any]:
        """Snapshot every dataflow-node connector type, keyed by ``(node, direction, connector)``.

        Re-inference is the only signal of change available -- neither
        ``invalidate_array_connectors`` nor ``infer_connector_types`` reports what it touched --
        so the pass diffs a before/after snapshot.
        """
        snapshot: Dict[Any, Any] = {}
        for node, _ in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.Node):
                continue
            for cname, ctype in node.in_connectors.items():
                snapshot[(node, 'in', cname)] = ctype
            for cname, ctype in node.out_connectors.items():
                snapshot[(node, 'out', cname)] = ctype
        return snapshot

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Re-derive NestedSDFG connector types from their inner descriptors.

        :returns: Number of connectors whose type changed, or ``None`` if none did.
        """
        from dace.sdfg import infer_types
        from dace.transformation.passes.scalar_promotion import invalidate_array_connectors
        before = self._connector_types(sdfg)
        invalidate_array_connectors(sdfg)
        for nsdfg in sdfg.all_sdfgs_recursive():
            infer_types.infer_connector_types(nsdfg)
        after = self._connector_types(sdfg)

        # Diff over the union of keys with a sentinel: a plain ``before.get(key)`` default of
        # ``None`` would compare a typeclass against ``None``, and ``typeclass.__ne__(None)``
        # returns False -- so an ADDED connector would be silently counted as unchanged. Iterating
        # ``after`` alone would likewise miss a REMOVED one.
        missing = object()
        changed = sum(1 for key in before.keys() | after.keys()
                      if before.get(key, missing) is not after.get(key, missing)
                      and before.get(key, missing) != after.get(key, missing))
        return changed or None
