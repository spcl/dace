# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``NestInnermostMapBodyIntoNSDFG``, precondition for the remainder-loop
and iteration-mask passes (P2 / P3). For every innermost map whose body
is not already wrapped in a single NestedSDFG, wrap its body in one so
the downstream passes can treat the body as a uniform unit (clone the
nested SDFG, attach mask connectors, etc.).

Reuses:
- ``map_consists_of_single_nsdfg_or_no_nsdfg`` and
  ``get_single_nsdfg_inside_map`` from ``utils.map_predicates`` for the
  shape check.
- ``dace.transformation.helpers.nest_state_subgraph`` for the actual
  in-place wrapping.
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


@properties.make_properties
class NestInnermostMapBodyIntoNSDFG(ppl.Pass):
    """For every innermost map whose body is not already a single
    NestedSDFG, nest the body into a NestedSDFG in-place. After the pass,
    every innermost map satisfies ``map_consists_of_single_nsdfg_or_no_nsdfg``
    AND contains exactly one NestedSDFG (no bare-tasklet bodies).

    Maps whose innermost trip count is *provably* a multiple of
    ``vector_width`` are skipped: they need no remainder, so P2 will not
    split them and the body never has to be cloned.  Wrapping them in an
    NSDFG would needlessly perturb downstream strided/gather/jacobi
    detection — this preserves the pre-remainder-refactor behaviour for
    the divisible case (what the old ``divides_evenly`` mode did by
    skipping P1/P2 entirely).
    """

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)

    def __init__(self, vector_width: int = 8):
        super().__init__()
        self.vector_width = vector_width

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _trip_is_provably_divisible(self, map_entry: dace.nodes.MapEntry) -> bool:
        """``True`` iff the innermost dim's trip count is provably a
        multiple of ``vector_width`` (mirrors the P2 skip check exactly,
        so the two passes agree on which maps get a remainder)."""
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

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
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
            if self._trip_is_provably_divisible(n):
                continue
            # Skip if the body already collapses to a single NestedSDFG.
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
            nest_state_subgraph(g.sdfg, g, subgraph, name=f"{n.label}_body")
            nested += 1
        return nested or None
