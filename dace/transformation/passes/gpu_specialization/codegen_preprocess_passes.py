# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapper :class:`Pass` classes exposing ``experimental_cuda.preprocess`` steps as composable Pipeline
members so codegen-preprocess ordering is declarative and testable."""
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
        from dace.transformation.passes.promote_gpu_scalars_to_arrays import invalidate_array_connectors
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
