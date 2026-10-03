# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The steps of the experimental CUDA preprocessing as pipeline passes."""
import warnings
from typing import Any, Dict, Optional

from dace import SDFG, Memlet, dtypes, nodes, properties
from dace.codegen import common
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """:func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types`: the GPU passes read final schedules."""

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
    """Recursive :meth:`SDFG.expand_library_nodes`."""

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.Symbols)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[bool]:
        from dace.sdfg import infer_types
        sdfg.expand_library_nodes(recursive=True)
        # Expansions spawn nested SDFGs with ``Default``-scheduled maps, which codegen rejects.
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        return True


@properties.make_properties
@transformation.explicit_cf_compatible
class AddThreadBlockMaps(ppl.Pass):
    """Tile every ``GPU_Device`` map without a ``GPU_ThreadBlock`` map and infer the ``(grid, block)`` dimensions.

    Returns ``kernel_dimensions_map`` and ``tb_inserted_kernels``. Tiling comes late: earlier, the outer-loop symbol
    would leak into the host-side allocation sizes of the arrays lifted out of the kernel.
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
        kernel_dimensions_map = InferGPUGridAndBlockSize().infer(sdfg, tb_inserted_kernels) or {}
        return {
            'kernel_dimensions_map': kernel_dimensions_map,
            'tb_inserted_kernels': tb_inserted_kernels,
        }


@properties.make_properties
@transformation.explicit_cf_compatible
class ReinferConnectorTypes(ppl.Pass):
    """Re-derive NestedSDFG connector types from their inner descriptors.

    Passes such as ``PromoteScalarOutputsToArrays`` widen a ``Scalar`` to an ``Array``, leaving stale scalar-typed
    connectors that miscompile.
    """

    def modifies(self) -> ppl.Modifies:
        # Connector inference retypes any node, map entries and exits included.
        return ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def connector_types(sdfg: SDFG) -> Dict[Any, Any]:
        """Every connector type, keyed by ``(node, direction, connector)``; inference does not report its changes."""
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
        """Returns the number of connectors whose type changed, or ``None``."""
        from dace.sdfg import infer_types
        from dace.transformation.passes.scalar_promotion import invalidate_array_connectors
        before = self.connector_types(sdfg)
        invalidate_array_connectors(sdfg)
        for nsdfg in sdfg.all_sdfgs_recursive():
            infer_types.infer_connector_types(nsdfg)
        after = self.connector_types(sdfg)

        # A sentinel, not ``None``: ``typeclass.__ne__(None)`` is False, so an added connector would not count.
        missing = object()
        changed = sum(1 for key in before.keys() | after.keys()
                      if before.get(key, missing) is not after.get(key, missing)
                      and before.get(key, missing) != after.get(key, missing))
        return changed or None


#: Label of the device-wide fence placed after a stream-unaware host callback.
DEVICE_SYNC_TASKLET_LABEL = 'gpu_callback_device_synchronization'


@properties.make_properties
@transformation.explicit_cf_compatible
class SynchronizeStreamUnawareGPUCallbacks(ppl.Pass):
    """Fence host callbacks that touch GPU memory without being stream-aware.

    Such a callback issues device work on a stream the SDFG does not know, so a device-wide synchronization after it
    orders it against every stream. A callback naming the stream is left alone.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        from dace.codegen.targets.cuda import stream_unaware_gpu_callbacks  # Avoid import loop
        targets = [(state, node) for state, node, _ in stream_unaware_gpu_callbacks(sdfg)
                   if not any(succ.label == DEVICE_SYNC_TASKLET_LABEL for succ in state.successors(node))]
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
                state.add_nedge(fence, succ, Memlet())
            state.add_nedge(node, fence, Memlet())
        return len(targets) or None
