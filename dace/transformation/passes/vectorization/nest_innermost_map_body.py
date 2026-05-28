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


@properties.make_properties
class NestInnermostMapBodyIntoNSDFG(ppl.Pass):
    """Nest each innermost map body into a NestedSDFG in place.

    After the pass every innermost map contains exactly one NestedSDFG
    (no bare-tasklet bodies). Maps whose innermost trip count is provably
    a multiple of ``vector_width`` are skipped by default: they need no
    remainder split, and wrapping them would perturb downstream
    strided/gather detection. Set ``nest_provably_divisible=True`` to
    nest them anyway — required by the SVE-style chain, where the
    per-core block is deliberately divisible (analyze-clean-then-Min)
    yet still needs a NestedSDFG body so the global iteration mask can
    be attached.
    """

    CATEGORY: str = "Vectorization Preparation"

    vector_width = properties.Property(dtype=int, default=8, allow_none=False)
    nest_provably_divisible = properties.Property(
        dtype=bool,
        default=False,
        desc="Also nest innermost maps whose trip is provably a multiple of "
        "``vector_width`` (default skips them). The SVE-style chain sets this: "
        "its per-core block is intentionally divisible but still needs a "
        "NestedSDFG body for the global iteration mask.")

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
            # SVE-style overrides this (its clean block is divisible by
            # design but still needs a body NSDFG for the global mask).
            if not self.nest_provably_divisible and self._trip_is_provably_divisible(n):
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
        if nested:
            # After nesting bodies, any WCR sink that previously flowed from a
            # tasklet now flows from the new NSDFG. DaCe cpu codegen only emits
            # WCR for AccessNode sources, so interpose a private scalar via
            # :class:`NormalizeWCRSource` to keep the reduction visible.
            from dace.transformation.passes.vectorization.normalize_wcr_source import (NormalizeWCRSource)
            NormalizeWCRSource().apply_pass(sdfg, {})
        return nested or None
