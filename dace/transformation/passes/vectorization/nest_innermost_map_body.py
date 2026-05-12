# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``NestInnermostMapBodyIntoNSDFG``, precondition for the remainder-loop
and iteration-mask passes (P2 / P3). For every innermost map whose body
is not already wrapped in a single NestedSDFG, wrap its body in one so
the downstream passes can treat the body as a uniform unit (clone the
nested SDFG, attach mask connectors, etc.).

Reuses:
- ``map_consists_of_single_nsdfg_or_no_nsdfg`` and
  ``get_single_nsdfg_inside_map`` from ``vectorization_utils`` for the
  shape check.
- ``dace.transformation.helpers.nest_state_subgraph`` for the actual
  in-place wrapping.
"""
from typing import Optional

import dace
from dace import properties
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
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
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
