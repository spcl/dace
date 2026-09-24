# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrap each innermost map body in a single NestedSDFG.

Precondition for remainder-split + iteration-mask passes: every innermost map
body becomes one uniform unit.
"""
from typing import Any

import dace
from dace import properties, symbolic
from dace.ordered import OrderedSet
from dace.sdfg.graph import SubgraphView
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineMultistateSDFG, InlineSDFG
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.vectorization.lower_reduction_wcr import lower_reduction_wcr_in_body
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.arrays import demote_connector_views
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_vectorizable_map,
    map_body_nodes,
)
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)


@properties.make_properties
class NestInnermostMapBodyIntoNSDFG(ppl.Pass):
    """Nest each innermost map body into a NestedSDFG in place.

    Post: every innermost map contains exactly one NestedSDFG (no bare-tasklet
    bodies) and that NestedSDFG holds no further nesting -- the tile passes only
    see one level down. Maps whose innermost trip is provably a multiple of ``vector_width``
    skipped by default (no remainder needed; wrapping perturbs downstream
    strided/gather detection). ``nest_provably_divisible=True`` nests them anyway
    -- masked-tail tile path needs a body NSDFG for the tile iteration mask.
    """

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    tiled_dims = properties.Property(
        dtype=int,
        default=1,
        allow_none=False,
        desc="Number of innermost dims the orchestrator will tile (``len(widths)``). Only forwarded "
        "to the shared ``is_vectorizable_map`` gate, so this pass agrees with the tile passes on "
        "which maps are candidates -- a map this pass declines to nest but a later pass strides is "
        "exactly the desync that gate exists to prevent.")
    nest_provably_divisible = properties.Property(
        dtype=bool,
        default=False,
        desc="Also nest innermost maps whose trip is provably a multiple of "
        "``vector_width`` (default skips them). The masked-tail tile path sets "
        "this: its provably-divisible interior still needs a NestedSDFG body "
        "for the tile iteration mask.")

    def __init__(self, vector_width: int = 8, nest_provably_divisible: bool = False, tiled_dims: int = 1) -> None:
        super().__init__()
        self.vector_width = vector_width
        self.nest_provably_divisible = nest_provably_divisible
        self.tiled_dims = tiled_dims

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _trip_is_provably_divisible(self, map_entry: dace.nodes.MapEntry) -> bool:
        # Innermost-dim trip provably a multiple of ``vector_width``?
        if not map_entry.map.range.ranges:
            return False
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return False
        trip = symbolic.simplify(ub - lb + 1)
        try:
            if bool(symbolic.simplify(trip % self.vector_width) == 0):
                return True
        except Exception:
            pass
        return False

    def _body_is_nested_reduction(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
        # True if the body is already a nested reduction: one NestedSDFG plus only boundary reduction AccessNodes (each
        # with a WCR edge to the MapExit -- the partial ``NormalizeWCRSource`` interposed on ``NSDFG -> AccessNode
        # -[wcr]-> MapExit``).
        map_exit = state.exit_node(map_entry)
        body = [
            k for k in map_body_nodes(state, map_entry) if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
        ]
        if len([k for k in body if isinstance(k, dace.nodes.NestedSDFG)]) != 1:
            return False
        others = [k for k in body if not isinstance(k, dace.nodes.NestedSDFG)]
        if not others:
            return False
        return all(
            isinstance(k, dace.nodes.AccessNode) and any(
                e.dst is map_exit and e.data is not None and e.data.wcr is not None for e in state.out_edges(k))
            for k in others)

    def _strip_boundary_other_subsets(self, state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG) -> None:
        # Drop stale ``other_subset`` on the body-NSDFG's boundary edges.
        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            if mem.other_subset is not None:
                mem.other_subset = None

    @staticmethod
    def expand_body_boundary(state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG) -> None:
        xform = ExpandNestedSDFGInputs()
        xform.setup_match(state.sdfg,
                          state.parent_graph.cfg_id,
                          state.block_id, {ExpandNestedSDFGInputs.nested_sdfg: nsdfg_node},
                          0,
                          override=True)
        if xform.can_be_applied(state, 0, state.sdfg, permissive=False):
            xform.apply(state, state.sdfg)

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Wrap every eligible innermost map body in a NestedSDFG.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Maps nested plus body-interior NestedSDFGs inlined, or ``None`` if neither.
        """
        # Phase 1, select (read-only): classify every innermost map on the unmutated SDFG. ``scan_cache``
        # memoizes the whole-SDFG symbol scan, sound only because nothing is nested in this phase.
        scan_cache: dict = {}
        # Annotated: an untyped list infers its elements as ``Any``, which silently disables the type
        # checker over every loop below that consumes them.
        selected: list[tuple[dace.SDFGState, dace.nodes.MapEntry, OrderedSet[dace.nodes.Node]]] = []
        candidates: list[tuple[dace.SDFGState, dace.nodes.MapEntry]] = []
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, dace.nodes.MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_vectorizable_map(g, n, self.tiled_dims, scan_cache=scan_cache):
                continue
            candidates.append((g, n))
            # Provably-divisible -> no remainder, leave un-nested (matches old
            # behaviour). ``nest_provably_divisible`` overrides: masked-tail interior
            # is divisible by design but needs a body NSDFG for the mask.
            if not self.nest_provably_divisible and self._trip_is_provably_divisible(n):
                continue
            # Body already a single NSDFG -> skip; orchestrator's always-on
            # ExpandNestedSDFGInputs normalises connector subsets regardless.
            # This pass only wraps bare-tasklet bodies.
            if get_single_nsdfg_inside_map(g, n) is not None:
                continue
            # Idempotency: a body already nested by a prior run is ``NSDFG -> AccessNode -[wcr]->
            # MapExit`` (the reduction partial ``NormalizeWCRSource`` interposed). Re-nesting it
            # would pull that boundary WCR back inside the body. Skip when the body is one NSDFG
            # plus only boundary reduction AccessNodes (each ``-[wcr]-> MapExit``).
            if self._body_is_nested_reduction(g, n):
                continue
            # Deliberately ``all_nodes_between``: a body ending in a write-only scratch scalar comes back empty
            # and stays un-nested, since the widener cannot lower it yet; RestoreUntiledMapStride fixes the step.
            body_nodes = OrderedSet(node for node in g.all_nodes_between(n, g.exit_node(n))
                                    if not isinstance(node, (dace.nodes.MapEntry, dace.nodes.MapExit)))
            if not body_nodes:
                continue
            selected.append((g, n, body_nodes))

        # Phase 2 -- NEST (mutate). Each nesting is body-local, so nesting order is independent and
        # the phase-1 node sets stay valid across the loop (``nest_state_subgraph`` leaves sibling
        # maps untouched).
        nested_bodies = []
        for g, n, body_nodes in selected:
            subgraph = SubgraphView(g, body_nodes)
            nsdfg_node = nest_state_subgraph(g.sdfg, g, subgraph, name=f"{n.label}_body")
            self._strip_boundary_other_subsets(g, nsdfg_node)
            demote_connector_views(nsdfg_node)
            # A postamble tail (``__scalar_tail`` / ``__tile_k1_tail``) runs the original body as
            # a step-1 loop the tile emitter skips, so its reduction stays a per-iteration
            # boundary WCR (no in-body TileReduce fold).
            is_tail = n.map.label.endswith(SCALAR_TAIL_MARKER) or n.map.label.endswith(TILE_K1_TAIL_MARKER)
            nested_bodies.append((nsdfg_node, is_tail))
        nested = len(nested_bodies)
        if nested:
            # The WCR sink now flows from the NSDFG; interpose a private scalar (NormalizeWCRSource) so CPU
            # codegen emits the boundary WCR as an OpenMP reduction.
            from dace.transformation.passes.normalize_wcr_source import (NormalizeWCRSource)
            NormalizeWCRSource().apply_pass(sdfg, {})
            # Rewrite the WCR ``nest_state_subgraph`` duplicated onto the inner body edge to ``acc = acc <op> src``
            # (foldable via TileReduce); the boundary WCR stays. Postamble tails keep it.
            for nsdfg_node, is_tail in nested_bodies:
                lower_reduction_wcr_in_body(nsdfg_node.sdfg, tiled=not is_tail)

        # Phase 3, flatten the body interior: the walker never descends into an NSDFG inside a body, so it
        # would tile around compute that runs once per tile (TSVC s4115/s4116). ExpandNestedSDFGInputs first,
        # since InlineMultistateSDFG refuses non-full boundary subsets.
        flattened = 0
        for g, n in candidates:
            for node in map_body_nodes(g, n):
                if isinstance(node, dace.nodes.NestedSDFG):
                    if any(isinstance(inner, dace.nodes.NestedSDFG) for inner, _ in node.sdfg.all_nodes_recursive()):
                        self.expand_body_boundary(g, node)
                    node.sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, permissive=False, validate=False)
                    flattened += node.sdfg.apply_transformations_repeated(
                        [InlineSDFG, InlineMultistateSDFG], permissive=False, validate=False) or 0

        assert_invariant(no_memlet_dim_mismatch(sdfg), "NestInnermostMapBodyIntoNSDFG",
                         "memlet subset and other_subset have matching dimensionality")
        return (nested + flattened) or None
