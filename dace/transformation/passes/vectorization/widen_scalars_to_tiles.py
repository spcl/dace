# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Mechanical descriptor swap: widen transient Scalar / (1,)-Array descriptors
to tile-shape ``Array(widths)`` inside every tile-tagged Map body NSDFG.

Runs after :class:`StageGlobalArrayThroughScalars` has normalized the body to
``source_AN -> Scalar chain through Tasklets -> sink_AN`` shape, so widening
is purely a descriptor swap + memlet-subset rewrite. No analysis, no Tasklet
walk, no recursive chain detection.

Per user direction 2026-06-10: handles BOTH ``data.Scalar`` AND ``(1,)``
``data.Array`` descriptors -- the Python frontend often produces ``(1,)``
arrays for what are semantically scalars (the ``__tmp_*_slice`` family from
Bypass), and a Scalar-only check misses them.

The widening is unconditional for every widenable transient in the body.
Per the staging-first design, loop-invariant data does NOT become a
transient Scalar -- it stays as a direct non-transient read (the
``stage_constant_access`` CONSTANT case of the staging design 3.1). So
every transient Scalar / ``(1,)`` Array we see in the body is, by
construction, lane-dep and must widen. Widening one extra Scalar in the
rare edge case where the staging pass missed a CONSTANT classification is
safe (extra register pressure, no correctness issue -- every lane just
holds the same value).
"""
from typing import Any, Dict, List, Optional, Tuple

from dace import data, dtypes, properties, subsets
from dace.sdfg import SDFG
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map


@properties.make_properties
@transformation.explicit_cf_compatible
class WidenScalarsToTiles(ppl.Pass):
    """Widen Scalar / ``(1,)``-Array transients to tile-shape ``Array(widths)``.

    :ivar widths: Per-tile-dim widths; same shape contract as the walker.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"WidenScalarsToTiles: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Same predicate as :class:`StageInsideBody` and :class:`InferBodyTransientShapes`.
        """
        K = len(self.widths)
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, MapEntry):
                continue
            if not isinstance(parent, SDFGState):
                continue
            try:
                if not is_innermost_map(parent, node):
                    continue
            except (StopIteration, ValueError):
                continue
            if len(node.map.params) < K:
                continue
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                continue
            yield parent, nsdfgs[0], node

    @staticmethod
    def _is_widenable(desc) -> bool:
        """``Scalar`` or ``(1,)``-shape ``Array`` transient -> widenable to tile.

        Per user direction 2026-06-10: ``(1,)`` arrays are common Python
        frontend artifacts that semantically behave as scalars; treat them
        identically to ``Scalar``.
        """
        if isinstance(desc, data.Scalar):
            return True
        if isinstance(desc, data.Array) and tuple(desc.shape) == (1, ):
            return True
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Widen every eligible transient + rewrite touching memlets."""
        widths = tuple(self.widths)
        target_shape = widths
        target_subset = ", ".join(f"0:{w}" for w in widths)
        widened_total = 0
        for _outer_state, nsdfg_node, _map_entry in self._body_nsdfgs(sdfg):
            inner_sdfg = nsdfg_node.sdfg
            # ANALYZE: collect (name, desc) pairs to widen.
            to_widen: List[str] = []
            for name, desc in list(inner_sdfg.arrays.items()):
                if not desc.transient:
                    continue
                if not self._is_widenable(desc):
                    continue
                to_widen.append(name)
            if not to_widen:
                continue
            # APPLY: batched mutation -- swap descriptors, rewrite memlets.
            for name in to_widen:
                old_desc = inner_sdfg.arrays[name]
                inner_sdfg.arrays[name] = data.Array(
                    dtype=old_desc.dtype,
                    shape=target_shape,
                    transient=True,
                    storage=dtypes.StorageType.Register,
                )
                widened_total += 1
            target_range = subsets.Range.from_string(target_subset)
            for name in to_widen:
                for state in inner_sdfg.states():
                    for edge in state.edges():
                        if edge.data is None or edge.data.data != name:
                            continue
                        edge.data.subset = subsets.Range(list(target_range.ranges))
        return widened_total if widened_total else None
