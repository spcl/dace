# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrap each innermost map body in a single NestedSDFG.

Precondition for remainder-split + iteration-mask passes: every innermost map
body becomes one uniform unit.
"""
from typing import Optional

import dace
from dace import properties, symbolic
from dace.sdfg.graph import SubgraphView
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.passes.vectorization.lower_reduction_wcr import lower_reduction_wcr_in_body
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                   TILE_K1_TAIL_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_vectorizable_map,
)
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)


@properties.make_properties
class NestInnermostMapBodyIntoNSDFG(ppl.Pass):
    """Nest each innermost map body into a NestedSDFG in place.

    Post: every innermost map contains exactly one NestedSDFG (no bare-tasklet
    bodies). Maps whose innermost trip is provably a multiple of ``vector_width``
    skipped by default (no remainder needed; wrapping perturbs downstream
    strided/gather detection). ``nest_provably_divisible=True`` nests them anyway
    — masked-tail tile path needs a body NSDFG for the tile iteration mask.
    """

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    nest_provably_divisible = properties.Property(
        dtype=bool,
        default=False,
        desc="Also nest innermost maps whose trip is provably a multiple of "
        "``vector_width`` (default skips them). The masked-tail tile path sets "
        "this: its provably-divisible interior still needs a NestedSDFG body "
        "for the tile iteration mask.")

    def __init__(self, vector_width: int = 8, nest_provably_divisible: bool = False):
        super().__init__()
        self.vector_width = vector_width
        self.nest_provably_divisible = nest_provably_divisible

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _trip_is_provably_divisible(self, map_entry: dace.nodes.MapEntry) -> bool:
        """Innermost-dim trip provably a multiple of ``vector_width``?

        :param map_entry: map entry whose innermost range is checked.
        :returns: ``True`` iff trip provably divisible by ``vector_width``.
        """
        if not map_entry.map.range.ranges:
            return False
        lb, ub, step = map_entry.map.range[-1]
        if (step != 1) and (str(step) != "1"):
            return False
        trip = symbolic.simplify(ub - lb + 1)
        try:
            if bool((trip % self.vector_width).simplify() == 0):
                return True
        except Exception:
            pass
        return False

    def _body_is_nested_reduction(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
        """True if the body is already a nested reduction: one NestedSDFG plus only boundary
        reduction AccessNodes (each with a WCR edge to the MapExit -- the partial
        ``NormalizeWCRSource`` interposed on ``NSDFG -> AccessNode -[wcr]-> MapExit``).

        Re-nesting such a body pulls the boundary WCR back inside, so the caller skips it -- the
        pass stays idempotent on its own output.

        :param state: the state holding the map.
        :param map_entry: the map to inspect.
        :returns: ``True`` iff the body is one NestedSDFG plus only boundary-reduction AccessNodes.
        """
        map_exit = state.exit_node(map_entry)
        body = [
            k for k in state.all_nodes_between(map_entry, map_exit)
            if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
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
        """Drop stale ``other_subset`` on the body-NSDFG's boundary edges.

        ``nest_state_subgraph`` deep-copies the original boundary memlet for the
        reconnected outer edge. If the body staged a global through a scalar element
        (frontend ``c_slice``-style ``Scalar``), that memlet was a copy *into the
        scalar* (``a[jk, jc] -> c_slice[0]``) carrying ``other_subset=[0]`` (rank-1
        scalar side). The reconnected edge now feeds connector ``a`` (rank-2 ``(1,1)``
        descriptor), and ``validate()`` resolves ``other_subset`` against the
        memlet-path source AccessNode ``a`` (rank 2) -> rank-1 ``[0]`` fails
        "other_subset does not match node dimension".

        Boundary edge is a plain pass-through: inner descriptor already defines the
        connector-side shape, so ``other_subset`` is redundant (same convention as
        NSDFG/lib-node connector edges; downstream
        :class:`~dace.transformation.interstate.expand_nested_sdfg_inputs.ExpandNestedSDFGInputs`
        clears it when widening these memlets). Clearing here keeps the SDFG valid
        immediately rather than transiently malformed.

        ``data is None`` = structural dependency edge, not data movement -> skip.

        :param state: state holding the freshly nested body NSDFG.
        :param nsdfg_node: the body :class:`~dace.sdfg.nodes.NestedSDFG` node.
        """
        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            if mem.other_subset is not None:
                mem.other_subset = None

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Wrap every eligible innermost map body in a NestedSDFG.

        :param sdfg: The SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of maps nested, or ``None`` if none.
        """
        nested = 0
        nested_bodies = []
        # Only innermost maps mutated -> iteration safe under in-place nesting
        # (``nest_state_subgraph`` leaves outer maps untouched).
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, dace.nodes.MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_vectorizable_map(g, n):
                continue
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
            # empty map -> nothing to nest
            body_nodes = {
                node
                for node in g.all_nodes_between(n, g.exit_node(n))
                if not isinstance(node, (dace.nodes.MapEntry, dace.nodes.MapExit))
            }
            if not body_nodes:
                continue
            subgraph = SubgraphView(g, body_nodes)
            nsdfg_node = nest_state_subgraph(g.sdfg, g, subgraph, name=f"{n.label}_body")
            self._strip_boundary_other_subsets(g, nsdfg_node)
            # A postamble tail (``__scalar_tail`` / ``__tile_k1_tail``) runs the original body as
            # a step-1 loop the tile emitter skips, so its reduction stays a per-iteration
            # boundary WCR (no in-body TileReduce fold).
            is_tail = n.map.label.endswith(SCALAR_TAIL_MARKER) or n.map.label.endswith(TILE_K1_TAIL_MARKER)
            nested_bodies.append((nsdfg_node, is_tail))
            nested += 1
        if nested:
            # WCR sink that flowed from a tasklet now flows from the new NSDFG.
            # CPU codegen only emits WCR for AccessNode sources -> interpose a
            # private scalar via :class:`NormalizeWCRSource` to keep it visible. After this the
            # reduction lives on the boundary ``NSDFG -> AccessNode -[wcr]-> MapExit`` chain (the
            # AccessNode is in the PARENT state, so ``no_wcr_inside_nested_sdfgs`` allows it) and
            # lowers to the OpenMP ``reduction(op:acc)`` clause.
            from dace.transformation.passes.normalize_wcr_source import (NormalizeWCRSource)
            NormalizeWCRSource().apply_pass(sdfg, {})
            # ``nest_state_subgraph`` also duplicates that reduction WCR onto the INNER body edge
            # (``src -[wcr]-> acc``). A loose WCR inside the body NSDFG is not tile-foldable, so
            # rewrite it -- ONLY inside each freshly-nested body -- to ``acc = acc <op> src`` (a
            # tile-in + scalar-out reduction the walker folds via ``TileReduce``), leaving the
            # boundary WCR untouched. A postamble tail keeps the per-iteration boundary WCR.
            for nsdfg_node, is_tail in nested_bodies:
                lower_reduction_wcr_in_body(nsdfg_node.sdfg, tiled=not is_tail)
        assert_invariant(no_memlet_dim_mismatch(sdfg), "NestInnermostMapBodyIntoNSDFG",
                         "memlet subset and other_subset have matching dimensionality")
        return nested or None
