# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Insert ``TileLoad`` / ``TileStore`` lib nodes between non-transient
``AccessNode`` boundaries and the body's transient tile chain.

Runs after :class:`StageGlobalArrayThroughScalars` + :class:`WidenScalarsToTiles`
have normalised the body NSDFG to ``source_AN -> Tasklet -> Tile -> Tasklet
-> ... -> Tile -> Tasklet -> sink_AN`` shape. This pass wraps each
**lane-dep** non-transient boundary edge with a tile lib node:

* **Read side** (``source_AN -> Tasklet``):
  * CONSTANT (loop-invariant): leave the direct edge in place. Codegen
    reads as ``kind=Scalar`` (hardware splat).
  * LINEAR / AFFINE / REPLICATE / MODULAR: insert ``source_AN -> TileLoad
    -> tile_bridge -> Tasklet``.
  * GATHER: TileLoad with ``gather_dims`` (deferred to follow-up pass).

* **Write side** (``Tasklet -> sink_AN``): symmetric -- TileStore for
  lane-dep writes, direct edge for CONSTANT.

Per user direction 2026-06-10: ``For scalar load or store we can emit a
Python assignment tasklet`` -- this applies when a Scalar transient bridge
mediates a CONSTANT chain. Direct ``AN -> Tasklet`` edges for CONSTANT
access are left alone (codegen reads as scalar splat).

Local decision per edge: no walker, no per-iter-var dep mask, no
multi-state reasoning. The staging-first pipeline factored that out.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import data, dtypes, properties, subsets
from dace.libraries.tileops import TileLoad, TileStore
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.subsets import an_side_subset
from dace.transformation.passes.vectorization.utils.tile_access import PerDimKind, classify_tile_access


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertTileLoadStore(ppl.Pass):
    """Insert tile lib nodes on every lane-dep non-transient boundary.

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
            raise ValueError(f"InsertTileLoadStore: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Same predicate as :class:`StageInsideBody` / :class:`WidenScalarsToTiles`.
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

    def _is_lane_dep_edge(self, edge, an: AccessNode, inner_sdfg: SDFG,
                          iter_vars: Tuple[str, ...]) -> Tuple[bool, Optional[Tuple]]:
        """Classify the AN-side subset; return ``(is_lane_dep, per_dim_kinds)``."""
        try:
            sub = an_side_subset(edge, an, inner_sdfg)
        except Exception:  # noqa: BLE001
            return True, None  # conservative: assume lane-dep
        try:
            record = classify_tile_access(sub, iter_vars=iter_vars, inner_sdfg=inner_sdfg)
        except Exception:  # noqa: BLE001
            return True, None
        if not record.per_dim_kind:
            return True, None
        kinds = tuple(record.per_dim_kind)
        is_lane_dep = not all(k == PerDimKind.CONSTANT for k in kinds)
        return is_lane_dep, kinds

    def _allocate_tile_bridge(self, inner_sdfg: SDFG, base_name: str, dtype) -> str:
        """Allocate a fresh tile-shape ``Array(widths)`` transient."""
        widths = tuple(self.widths)
        name, _ = inner_sdfg.add_array(
            f"{base_name}_tile",
            shape=widths,
            dtype=dtype,
            transient=True,
            storage=dtypes.StorageType.Register,
            find_new_name=True,
        )
        return name

    def _find_inner_mask_name(self, inner_sdfg: SDFG) -> Optional[str]:
        """Find the body-NSDFG's iteration mask array name, or None if no mask is in scope."""
        from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
        base = TileNameScheme.ITER_MASK
        if base in inner_sdfg.arrays:
            return base
        for name in inner_sdfg.arrays:
            if name.startswith(f"{base}_"):
                return name
        return None

    def _find_mask_producer_an(self, state: SDFGState, mask_name: str) -> Optional[AccessNode]:
        """Find the AccessNode that TileMaskGen writes to (the shared mask source).

        Every downstream consumer's ``_mask`` edge MUST read from this SAME
        AccessNode so the scheduler orders TileMaskGen before the consumers.
        """
        from dace.libraries.tileops import TileMaskGen
        for n in state.nodes():
            if not isinstance(n, TileMaskGen):
                continue
            for out_edge in state.out_edges(n):
                if (out_edge.src_conn == "_o" and isinstance(out_edge.dst, AccessNode)
                        and out_edge.dst.data == mask_name):
                    return out_edge.dst
        return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Insert TileLoad/TileStore on every lane-dep non-transient boundary edge."""
        K = len(self.widths)
        widths = tuple(self.widths)
        tile_subset = ", ".join(f"0:{w}" for w in widths)
        inserted = 0
        for _outer_state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            inner_sdfg = nsdfg_node.sdfg
            iter_vars = tuple(map_entry.map.params[-K:])
            mask_name = self._find_inner_mask_name(inner_sdfg)
            for state in list(inner_sdfg.states()):
                mask_an = self._find_mask_producer_an(state, mask_name) if mask_name else None
                has_mask = mask_an is not None
                # ANALYZE: collect edges to rewrite (read side + write side).
                ans = [n for n in list(state.nodes()) if isinstance(n, AccessNode)]
                read_actions: List[Tuple[AccessNode, Any, Tuple]] = []
                write_actions: List[Tuple[AccessNode, Any, Tuple]] = []
                for an in ans:
                    desc = inner_sdfg.arrays.get(an.data)
                    if desc is None or desc.transient:
                        continue
                    for edge in list(state.out_edges(an)):
                        if isinstance(edge.dst, (TileLoad, TileStore)):
                            continue
                        is_lane_dep, kinds = self._is_lane_dep_edge(edge, an, inner_sdfg, iter_vars)
                        if not is_lane_dep:
                            continue
                        if kinds and PerDimKind.GATHER in kinds:
                            continue  # GATHER deferred to follow-up
                        read_actions.append((an, edge, kinds))
                    for edge in list(state.in_edges(an)):
                        if isinstance(edge.src, (TileLoad, TileStore)):
                            continue
                        is_lane_dep, kinds = self._is_lane_dep_edge(edge, an, inner_sdfg, iter_vars)
                        if not is_lane_dep:
                            continue
                        if kinds and PerDimKind.GATHER in kinds:
                            continue
                        write_actions.append((an, edge, kinds))
                # APPLY: batched mutation.
                mask_subset = ", ".join(f"0:{w}" for w in widths)
                for an, old_edge, _kinds in read_actions:
                    desc = inner_sdfg.arrays[an.data]
                    bridge_name = self._allocate_tile_bridge(inner_sdfg, an.data, desc.dtype)
                    bridge_an = state.add_access(bridge_name)
                    load_node = TileLoad(name=f"load_{an.data}", widths=widths, has_mask=has_mask)
                    state.add_node(load_node)
                    an_subset = an_side_subset(old_edge, an, inner_sdfg)
                    state.add_edge(an, None, load_node, "_src", Memlet(data=an.data, subset=an_subset))
                    state.add_edge(load_node, "_dst", bridge_an, None, Memlet(data=bridge_name, subset=tile_subset))
                    if has_mask:
                        state.add_edge(mask_an, None, load_node, "_mask",
                                       Memlet(data=mask_an.data, subset=mask_subset))
                    state.add_edge(bridge_an, None, old_edge.dst, old_edge.dst_conn,
                                   Memlet(data=bridge_name, subset=tile_subset))
                    state.remove_edge(old_edge)
                    inserted += 1
                for an, old_edge, _kinds in write_actions:
                    desc = inner_sdfg.arrays[an.data]
                    bridge_name = self._allocate_tile_bridge(inner_sdfg, an.data, desc.dtype)
                    bridge_an = state.add_access(bridge_name)
                    store_node = TileStore(name=f"store_{an.data}", widths=widths, has_mask=has_mask)
                    state.add_node(store_node)
                    state.add_edge(old_edge.src, old_edge.src_conn, bridge_an, None,
                                   Memlet(data=bridge_name, subset=tile_subset))
                    state.add_edge(bridge_an, None, store_node, "_src", Memlet(data=bridge_name, subset=tile_subset))
                    if has_mask:
                        state.add_edge(mask_an, None, store_node, "_mask",
                                       Memlet(data=mask_an.data, subset=mask_subset))
                    an_subset = an_side_subset(old_edge, an, inner_sdfg)
                    state.add_edge(store_node, "_dst", an, None, Memlet(data=an.data, subset=an_subset))
                    state.remove_edge(old_edge)
                    inserted += 1
        return inserted if inserted else None
