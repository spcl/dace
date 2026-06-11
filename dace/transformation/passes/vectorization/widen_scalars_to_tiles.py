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

The widening is **gated by lane-dependence** per user direction 2026-06-10:
"Scalars should be widened only if they are read from non-const subsets,
same for store. scalar + tile op -> tile, but scalar - scalar -> scalar or
scalar - symbol -> scalar (if symbol is loop-invariant; if loop-variant
meaning tile-dependent -> tile)." A scalar fed purely by other scalars and
loop-invariant symbols stays a scalar; only scalars whose data lineage
reaches a non-CONSTANT non-transient access OR whose computing tasklet
references a tile iter-var symbol get widened.

Propagation rule (forward, iterate to fixed point):

1. For every non-transient ``AccessNode`` adjacent edge, classify its
   AN-side subset via ``classify_tile_access``. Mark the non-transient
   data name as ``lane-dep`` iff any classification is non-CONSTANT.
2. Seed: every transient is initially loop-invariant.
3. For each ``Tasklet``: it produces lane-dep output if ANY of:
     a. an input data name is in the lane-dep set, OR
     b. its code body references a tile iter-var symbol (e.g. ``i``).
   When lane-dep, mark every output data name as lane-dep.
4. Iterate until no new transient is added.
5. Widen only the transients in the lane-dep set.

Because ``StageGlobalArrayThroughScalars`` already ran (this pass is
designed to follow it), there are no direct ``AccessNode -> AccessNode``
chains for the propagator to miss -- every data-flow hop goes through a
Tasklet. The simple Tasklet-walk propagation is correct on this clean
input (unlike its placement before staging, where post-Bypass AN -> AN
chains broke the propagation -- the design tension documented in commit
8a9476e1d).
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import data, dtypes, properties, subsets
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


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

    @staticmethod
    def _classify_non_transient_lane_dep(inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> Set[str]:
        """Return non-transient names with at least one non-CONSTANT access.

        These are the seeds for lane-dep propagation: the data they carry
        differs per lane within a tile.
        """
        lane_dep: Set[str] = set()
        for state in inner_sdfg.states():
            for an in state.nodes():
                if not isinstance(an, AccessNode):
                    continue
                desc = inner_sdfg.arrays.get(an.data)
                if desc is None or desc.transient:
                    continue
                if an.data in lane_dep:
                    continue
                for edge in list(state.out_edges(an)) + list(state.in_edges(an)):
                    try:
                        sub = an_side_subset(edge, an, inner_sdfg)
                    except Exception:  # noqa: BLE001 -- helper may refuse exotic edges
                        lane_dep.add(an.data)  # conservative
                        break
                    try:
                        record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
                    except Exception:  # noqa: BLE001
                        lane_dep.add(an.data)
                        break
                    if not record.per_dim_kind:
                        lane_dep.add(an.data)
                        break
                    if not all(k == PerDimKind.CONSTANT for k in record.per_dim_kind):
                        lane_dep.add(an.data)
                        break
        return lane_dep

    @staticmethod
    def _tasklet_references_iter_var(tasklet: Tasklet, iter_vars: Tuple[str, ...]) -> bool:
        """True iff ``tasklet``'s code body references any tile iter-var name.

        Lightweight token scan -- DaCe tasklet bodies are simple expressions
        and the iter-var names (``i``, ``j``, ...) don't appear in comments
        or string literals in practice. False positives are safe (over-widen).
        """
        import re
        if tasklet.code is None:
            return False
        code_str = tasklet.code.as_string or ""
        if not code_str:
            return False
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", code_str))
        return any(v in tokens for v in iter_vars)

    @staticmethod
    def _data_names_of_edge(edge, side: str) -> List[str]:
        """Collect all data names that ``edge`` references on ``side``.

        Returns the memlet ``data`` field plus the AccessNode endpoint's
        ``data`` (when the endpoint on ``side`` is an AN). Both matter
        because post-Bypass memlets sometimes carry ``data`` pointing at
        a non-transient while the endpoint AN is a transient.
        """
        names = []
        endpoint = edge.src if side == "src" else edge.dst
        if isinstance(endpoint, AccessNode):
            names.append(endpoint.data)
        if edge.data and edge.data.data:
            names.append(edge.data.data)
        return names

    def _propagate_lane_dep(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...], nt_lane_dep: Set[str]) -> Set[str]:
        """Forward-propagate lane-dep through Tasklets to a fixed point.

        :returns: set of transient data names that need tile-shape widening.
        """
        lane_dep_transients: Set[str] = set()
        changed = True
        max_iters = 32
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for state in inner_sdfg.states():
                for node in state.nodes():
                    if not isinstance(node, Tasklet):
                        continue
                    # Tasklet produces lane-dep output iff any input is lane-dep
                    # OR its body references a tile iter-var.
                    is_lane_dep = self._tasklet_references_iter_var(node, iter_vars)
                    if not is_lane_dep:
                        for e in state.in_edges(node):
                            for nm in self._data_names_of_edge(e, "src"):
                                if nm in nt_lane_dep or nm in lane_dep_transients:
                                    is_lane_dep = True
                                    break
                            if is_lane_dep:
                                break
                    if not is_lane_dep:
                        continue
                    for e in state.out_edges(node):
                        for nm in self._data_names_of_edge(e, "dst"):
                            desc = inner_sdfg.arrays.get(nm)
                            if desc is None or not desc.transient:
                                continue
                            if not self._is_widenable(desc):
                                continue
                            if nm not in lane_dep_transients:
                                lane_dep_transients.add(nm)
                                changed = True
        return lane_dep_transients

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Widen lane-dep transient Scalars / ``(1,)`` Arrays + rewrite memlets."""
        K = len(self.widths)
        widths = tuple(self.widths)
        target_shape = widths
        target_subset = ", ".join(f"0:{w}" for w in widths)
        widened_total = 0
        for _outer_state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            inner_sdfg = nsdfg_node.sdfg
            iter_vars = tuple(map_entry.map.params[-K:])
            # ANALYZE: classify non-transients, forward-propagate to find
            # lane-dep transients.
            nt_lane_dep = self._classify_non_transient_lane_dep(inner_sdfg, iter_vars)
            to_widen = self._propagate_lane_dep(inner_sdfg, iter_vars, nt_lane_dep)
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
