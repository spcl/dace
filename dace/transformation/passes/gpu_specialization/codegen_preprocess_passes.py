# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrapper :class:`Pass` classes exposing the ``experimental_cuda.preprocess``
steps as composable Pipeline members.

Each pass wraps one codegen-preprocess operation
(``sdfg.expand_library_nodes``, ``apply_transformations_once_everywhere(AddThreadBlockMap)``,
etc.) so the ordering is declarative and testable.
"""
from typing import Any, Dict, Optional

from dace import SDFG, dtypes, nodes, properties
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class ExpandLibraryNodes(ppl.Pass):
    """Wraps :meth:`SDFG.expand_library_nodes` (recursive) as a Pipeline
    Pass so library-node expansion can be ordered declaratively
    alongside other transformations."""

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
class AddThreadBlockMaps(ppl.Pass):
    """Tile every ``GPU_Device`` map without an inner ``GPU_ThreadBlock``
    map (via :class:`AddThreadBlockMap`) and infer the resulting
    ``(grid, block)`` dimensions for codegen.

    Returns a dict ``{'kernel_dimensions_map': …, 'tb_inserted_kernels':
    set(MapEntry)}`` that callers (the codegen target) read out of
    ``pipeline_results``.

    Tiles late on purpose: the kernel-internal transient hoist
    (``MoveArrayOutOfKernel``) sees user-authored kernel shapes, not
    post-tile shapes — tiling earlier introduces an inner-map range like
    ``Min(N-1, b_i+31) - b_i + 1`` whose ``b_i`` outer-loop symbol then
    leaks into host-side ``cudaMalloc`` size expressions for any
    transient lifted out of the kernel.
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
    """Re-derive NestedSDFG connector types from their inner descriptors.

    Earlier passes mutate descriptors (e.g. ``PromoteGPUScalarsToArrays``
    widens a ``Scalar`` into a length-1 ``Array``); connectors typed at
    construction time still carry the old scalar dtype and the codegen
    emits ``T name`` while the body indexes ``name[0]`` — compile error.
    Clear those connector annotations on every NestedSDFG (recursively),
    then re-infer them so they become pointer-typed.
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
