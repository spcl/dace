# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapper :class:`Pass` classes exposing ``experimental_cuda.preprocess`` steps as composable Pipeline
members so codegen-preprocess ordering is declarative and testable."""
import warnings
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, SDFGState, dtypes, nodes, properties
from dace.codegen import common
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """:func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types` as a Pipeline Pass: the GPU
    passes after it read final schedules and storages."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        from dace.sdfg import infer_types
        infer_types.set_default_schedule_and_storage_types(sdfg, None)


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

    Earlier passes mutate descriptors (e.g. ``PromoteScalarOutputsToArrays`` widens a ``Scalar`` to a
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


#: Label of the device-wide fence placed after a stream-unaware host callback.
DEVICE_SYNC_TASKLET_LABEL = 'gpu_callback_device_synchronization'


def is_host_callback(node: nodes.Node) -> bool:
    """A ``dace.callback`` invocation: a side-effecting tasklet that is not one of the pipeline's own syncs."""
    from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_pipeline_sync_tasklet
    return (isinstance(node, nodes.Tasklet) and node.side_effects is True and not is_pipeline_sync_tasklet(node)
            and node.label != DEVICE_SYNC_TASKLET_LABEL)


def stream_unaware_gpu_callbacks(sdfg: SDFG) -> List[Tuple[SDFGState, nodes.Tasklet]]:
    """Host callbacks touching GPU memory without naming the stream, and not fenced yet."""
    from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import STREAM_CONNECTOR
    found = []
    for cursdfg in sdfg.all_sdfgs_recursive():
        for state in cursdfg.states():
            for node in state.nodes():
                if not is_host_callback(node) or STREAM_CONNECTOR in node.code.as_string:
                    continue
                if is_devicelevel_gpu(cursdfg, state, node):
                    continue
                touches_gpu = any(not e.data.is_empty()
                                  and cursdfg.arrays[e.data.data].storage in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
                                  for e in state.all_edges(node))
                fenced = any(succ.label == DEVICE_SYNC_TASKLET_LABEL for succ in state.successors(node))
                if touches_gpu and not fenced:
                    found.append((state, node))
    return found


@properties.make_properties
@transformation.explicit_cf_compatible
class SynchronizeStreamUnawareGPUCallbacks(ppl.Pass):
    """Fence host callbacks that touch GPU memory without being stream-aware.

    Such a callback issues its device work on a stream the SDFG does not know, so the asynchronous
    work scheduled around it on ``gpu_streams`` is unordered against it. A device-wide synchronization
    after the callback orders it against every stream. A callback naming the stream is left alone.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import dependency_edge
        targets = stream_unaware_gpu_callbacks(sdfg)
        backend = common.get_gpu_backend()
        for state, node in targets:
            warnings.warn(
                f'Callback "{node.label}" accesses GPU memory but is not stream-aware, so a full device '
                'synchronization is emitted after it. Add a "dace.current_stream" argument to the callback '
                'and use it (e.g. cupy ExternalStream) to keep the work asynchronous.', UserWarning)
            fence = state.add_tasklet(DEVICE_SYNC_TASKLET_LABEL, {}, {},
                                      f'DACE_GPU_CHECK({backend}DeviceSynchronize());',
                                      language=dtypes.Language.CPP,
                                      side_effects=True)
            for succ in list(state.successors(node)):
                state.add_edge(fence, None, succ, None, dependency_edge())
            state.add_edge(node, None, fence, None, dependency_edge())
        return len(targets) or None
