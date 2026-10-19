# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The steps of the experimental CUDA preprocessing as pipeline passes."""

import warnings
from typing import Any

from dace import SDFG, Memlet, dtypes, nodes, properties
from dace.codegen import common
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """:func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types`: the GPU passes read final schedules."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> None:
        from dace.sdfg import infer_types

        infer_types.set_default_schedule_and_storage_types(sdfg, None)


@properties.make_properties
@transformation.explicit_cf_compatible
class ExpandLibraryNodes(ppl.Pass):
    """Recursive :meth:`SDFG.expand_library_nodes`."""

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.States
            | ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.Descriptors
            | ppl.Modifies.Symbols
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> bool | None:
        from dace.codegen.python.dialects.cuda_tile_cpp import expand_outside_tile_kernels
        from dace.sdfg import infer_types

        # A CUDA Tile C++ kernel writes its tile nodes itself
        expand_outside_tile_kernels(sdfg)
        # Expansions spawn nested SDFGs with ``Default``-scheduled maps, which codegen rejects.
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        return True


def apply_thread_block_maps(sdfg: SDFG, xform: transformation.SingleStateTransformation) -> int:
    """``sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)``, enumerating by walking.

    The matcher re-derives every match after each application: a multigraph collapse and a VF2 search per state, only
    to find map entries. A one-node pattern matches each ``MapEntry`` in ``state.nodes()`` order, so this walks the
    same regions, states and nodes in the matcher's order, asks the transformation (``can_be_applied``, under the same
    dtypes and with the same match setup) and applies the first accepted match, then restarts as the matcher does.
    The candidates, their order and every verdict are the matcher's; only the search is gone.

    :return: The number of applications.
    """
    from dace import symbolic
    from dace.config import Config
    from dace.sdfg import SDFGState
    from dace.sdfg.validation import InvalidSDFGError
    from dace.transformation.passes.pattern_matching import CfgIds

    applied = 0
    last = None
    while True:
        match = None
        cfg_ids = CfgIds(sdfg)
        for cfr in sdfg.all_control_flow_regions(recursive=True):
            for state_id, state in enumerate(cfr.nodes()):
                if not isinstance(state, SDFGState):
                    continue
                owner = cfr.sdfg
                for node_id, node in enumerate(state.nodes()):
                    if not isinstance(node, nodes.MapEntry):
                        continue
                    try:
                        xform.setup_match(
                            owner, cfg_ids.cfg_id(cfr), state_id, {type(xform).map_entry: node_id}, 0, options=None
                        )
                        xform._pipeline_results = {}
                        with symbolic.serialization_symbol_dtypes(owner.symbols, inherit=True):
                            found = xform.can_be_applied(state, 0, owner, permissive=False)
                    except Exception as e:
                        if Config.get_bool("optimizer", "match_exception"):
                            raise
                        print(
                            f"WARNING: {type(xform).__name__}::can_be_applied triggered a "
                            f"{e.__class__.__name__} exception: {e}"
                        )
                        continue
                    if found:
                        match = (owner, state)
                        break
                if match is not None:
                    break
            if match is not None:
                break
        if match is None:
            break
        owner, state = match
        xform._pipeline_results = {}
        xform.permissive = False
        with symbolic.serialization_symbol_dtypes(owner.symbols, inherit=True):
            xform.apply(state, owner)
        applied += 1
        last = xform
    if applied:
        try:
            sdfg.validate()
        except InvalidSDFGError as err:
            raise InvalidSDFGError(
                f"Validation failed after applying {type(last).__name__} in {sdfg.cfg_list[last.cfg_id].label}.",
                sdfg,
                None,
            ) from err
    return applied


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

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, Any]:
        from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
        from dace.transformation.passes.analysis.infer_gpu_grid_and_block_size import InferGPUGridAndBlockSize

        old_nodes = {node for node, _ in sdfg.all_nodes_recursive()}
        if sdfg.root_sdfg.using_explicit_control_flow and not AddThreadBlockMap.__explicit_cf_compatible__:
            # The matcher's own refusal; it warns and applies nothing.
            sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)
        else:
            apply_thread_block_maps(sdfg, AddThreadBlockMap())
        new_nodes = {node for node, _ in sdfg.all_nodes_recursive()} - old_nodes
        tb_inserted_kernels = {
            n for n in new_nodes if isinstance(n, nodes.MapEntry) and n.schedule == dtypes.ScheduleType.GPU_Device
        }
        kernel_dimensions_map = InferGPUGridAndBlockSize().infer(sdfg, tb_inserted_kernels) or {}
        return {
            "kernel_dimensions_map": kernel_dimensions_map,
            "tb_inserted_kernels": tb_inserted_kernels,
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
    def connector_types(sdfg: SDFG) -> dict[Any, Any]:
        """Every connector type, keyed by ``(node, direction, connector)``; inference does not report its changes."""
        snapshot: dict[Any, Any] = {}
        for node, _ in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.Node):
                continue
            for cname, ctype in node.in_connectors.items():
                snapshot[(node, "in", cname)] = ctype
            for cname, ctype in node.out_connectors.items():
                snapshot[(node, "out", cname)] = ctype
        return snapshot

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
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
        changed = sum(
            1
            for key in before.keys() | after.keys()
            if before.get(key, missing) is not after.get(key, missing)
            and before.get(key, missing) != after.get(key, missing)
        )
        return changed or None


#: Label of the device-wide fence placed after a stream-unaware host callback.
DEVICE_SYNC_TASKLET_LABEL = "gpu_callback_device_synchronization"


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

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        from dace.codegen.targets.cuda import stream_unaware_gpu_callbacks  # Avoid import loop

        targets = [
            (state, node)
            for state, node, _ in stream_unaware_gpu_callbacks(sdfg)
            if not any(succ.label == DEVICE_SYNC_TASKLET_LABEL for succ in state.successors(node))
        ]
        backend = common.get_gpu_backend()
        for state, node in targets:
            warnings.warn(
                f'Callback "{node.label}" accesses GPU memory but is not stream-aware, so a full device '
                'synchronization is emitted after it. Add a "dace.current_stream" argument to the callback '
                "and use it (e.g. cupy ExternalStream) to keep the work asynchronous.",
                UserWarning,
            )
            fence = state.add_tasklet(
                DEVICE_SYNC_TASKLET_LABEL,
                {},
                {},
                f"DACE_GPU_CHECK({backend}DeviceSynchronize());",
                language=dtypes.Language.CPP,
                side_effects=True,
            )
            for succ in list(state.successors(node)):
                state.add_nedge(fence, succ, Memlet())
            state.add_nedge(node, fence, Memlet())
        return len(targets) or None
