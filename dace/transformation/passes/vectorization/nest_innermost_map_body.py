# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrap each innermost map body in a single NestedSDFG.

Precondition for the remainder-split and iteration-mask passes so they
can treat every innermost map body as one uniform unit.
"""
from typing import Optional

import dace
from dace import properties, symbolic
from dace.sdfg.graph import SubgraphView
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_innermost_map,
)
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, no_memlet_dim_mismatch)


@properties.make_properties
class NestInnermostMapBodyIntoNSDFG(ppl.Pass):
    """Nest each innermost map body into a NestedSDFG in place.

    After the pass every innermost map contains exactly one NestedSDFG
    (no bare-tasklet bodies). Maps whose innermost trip count is provably
    a multiple of ``vector_width`` are skipped by default: they need no
    remainder split, and wrapping them would perturb downstream
    strided/gather detection. Set ``nest_provably_divisible=True`` to
    nest them anyway — required by the masked-tail tile path, where the
    provably-divisible interior still needs a NestedSDFG body so the
    tile iteration mask can be attached.
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
        """Check whether the innermost dimension trip count is provably a multiple of ``vector_width``.

        :param map_entry: The map entry whose innermost range is checked.
        :returns: ``True`` iff the trip count is provably divisible by ``vector_width``.
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

    def _strip_boundary_other_subsets(self, state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG) -> None:
        """Drop the stale ``other_subset`` on the body-NSDFG's boundary edges.

        ``nest_state_subgraph`` deep-copies the original map-body boundary memlet
        for the reconnected outer edge. When the body staged a global through a
        scalar element (a frontend ``c_slice``-style ``Scalar``), that original
        memlet was a copy *into the scalar* — e.g. ``a[jk, jc] -> c_slice[0]`` —
        so it carried ``other_subset = [0]`` describing the rank-1 scalar side.
        The reconnected outer edge now feeds the body-NSDFG connector ``a``
        (whose descriptor is the rank-2 ``(1, 1)`` element), and DaCe
        ``validate()`` resolves the edge's ``other_subset`` against the
        memlet-path *source* AccessNode ``a`` (rank 2). The rank-1 ``[0]`` then
        fails ``Memlet other_subset does not match node dimension``.

        The boundary edge is a plain pass-through into the connector: the inner
        descriptor already defines the connector-side shape, so ``other_subset``
        is redundant there (the same convention DaCe uses everywhere for
        NSDFG / lib-node connector edges, and exactly what the downstream
        :class:`~dace.transformation.interstate.expand_nested_sdfg_inputs.ExpandNestedSDFGInputs`
        clears when it widens these memlets). Clearing it here makes the SDFG
        valid immediately after this pass rather than transiently malformed.

        A memlet edge whose ``data is None`` is a structural dependency edge, not
        a data movement — leave it untouched.

        :param state: The state holding the freshly nested body NSDFG.
        :param nsdfg_node: The body :class:`~dace.sdfg.nodes.NestedSDFG` node.
        """
        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            mem = edge.data
            if mem is None or mem.data is None:
                # Structural dependency edge -- skip.
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
        # Walk every map across nested SDFGs. We only mutate innermost
        # maps so the iteration is safe under the in-place nesting
        # (``nest_state_subgraph`` does not touch outer maps).
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, dace.nodes.MapEntry):
                continue
            if not isinstance(g, dace.SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            # Provably-divisible maps need no remainder; leave the body
            # un-nested so the divisible path matches the old behaviour.
            # ``nest_provably_divisible`` overrides this (the masked-tail
            # interior is divisible by design but still needs a body NSDFG
            # for the tile mask).
            if not self.nest_provably_divisible and self._trip_is_provably_divisible(n):
                continue
            # Skip if the body already collapses to a single NestedSDFG.
            # The always-on ``ExpandNestedSDFGInputs`` step in the
            # orchestrator normalises every NSDFG's connector subsets
            # regardless of how the body was produced, so this pass
            # focuses purely on wrapping bare-tasklet bodies.
            if get_single_nsdfg_inside_map(g, n) is not None:
                continue
            # Skip purely empty maps (nothing to nest).
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
            nested += 1
        if nested:
            # After nesting bodies, any WCR sink that previously flowed from a
            # tasklet now flows from the new NSDFG. DaCe cpu codegen only emits
            # WCR for AccessNode sources, so interpose a private scalar via
            # :class:`NormalizeWCRSource` to keep the reduction visible.
            from dace.transformation.passes.normalize_wcr_source import (NormalizeWCRSource)
            NormalizeWCRSource().apply_pass(sdfg, {})
        assert_invariant(no_memlet_dim_mismatch(sdfg), "NestInnermostMapBodyIntoNSDFG",
                         "memlet subset and other_subset have matching dimensionality")
        return nested or None
