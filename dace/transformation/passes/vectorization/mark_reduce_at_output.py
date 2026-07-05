# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tag reduction maps for the ``reduce_at_output`` tile lowering.

A NESTED reduction -- a reduction whose accumulation lands *inside* an innermost
map's body (``Reduce ─[wcr]→ acc`` in the body NSDFG, or a scalar ``AN ─[wcr]→
MapExit`` reduction edge) -- is dropped by the K-dim tile emitters (only a
top-level reduction, e.g. a plain ``sum(a*b)``, vectorises today). This pass marks
each such map with :data:`REDUCE_AT_OUTPUT_MARKER` so the downstream passes (the
WCR asserts, and the tile-reduce relocation) recognise it: the reduction stays in
place through tiling, and a ``TileReduce`` is spliced at the ``NSDFG → WCR-AN``
boundary (partial-tile → scalar → OMP WCR) rather than being asserted away.

Runs AFTER ``LiftEinsum`` so a clean contraction (gemm) stays on the BLAS/Einsum
path and is NOT tagged here.
"""
from typing import Optional

import dace
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import SDFG, SDFGState, nodes as nd
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

#: Appended to an innermost map's label when its body carries a reduction that must be
#: lowered via a boundary ``TileReduce`` (the ``reduce_at_output`` path). Mirrors the
#: ``SCALAR_TAIL_MARKER`` convention so the tile passes can skip / special-case it.
REDUCE_AT_OUTPUT_MARKER = "__reduce_out"


def _has_reduction(state: SDFGState, node) -> bool:
    """True iff ``node`` (or, for a NestedSDFG, its inner SDFG recursively) carries a
    reduction to accumulate: a :class:`Reduce` whose output edge has a WCR, or (for a
    NestedSDFG) a scalar ``─[wcr]→`` edge inside it."""
    if isinstance(node, Reduce):
        return any(e.data is not None and e.data.wcr is not None for e in state.out_edges(node))
    if isinstance(node, nd.NestedSDFG):
        for sd in node.sdfg.all_sdfgs_recursive():
            for st in sd.states():
                for n in st.nodes():
                    if isinstance(n, Reduce) and any(e.data is not None and e.data.wcr is not None
                                                     for e in st.out_edges(n)):
                        return True
                for e in st.edges():
                    if (e.data is not None and e.data.wcr is not None and e.data.subset is not None
                            and e.data.subset.num_elements() == 1):
                        return True
    return False


def _no_nested_maps(state: SDFGState, map_entry: nd.MapEntry) -> bool:
    """True iff ``map_entry`` has NO map anywhere inside it (recursively, through body
    NSDFGs). Only such a *truly innermost* map is vectorized (user direction: "we
    vectorize innermost maps -- no maps inside"), so only it may carry the
    ``reduce_at_output`` reduction. An OUTER map that merely *encloses* a deeper
    reduction map (azimint's ``i`` map over the ``j`` reduction) must NOT be tagged --
    it is not the reduction and tiling it would relocate the wrong dimension."""
    scope = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
    for n in scope.nodes():
        if isinstance(n, nd.MapEntry):
            return False
        if isinstance(n, nd.NestedSDFG):
            for sd in n.sdfg.all_sdfgs_recursive():
                for st in sd.states():
                    if any(isinstance(x, nd.MapEntry) for x in st.nodes()):
                        return False
    return True


def _map_has_body_reduction(state: SDFGState, map_entry: nd.MapEntry) -> bool:
    """True iff ``map_entry``'s body carries a scalar reduction to accumulate: a
    :class:`Reduce` node (or a body NSDFG containing one -- the v3 ``product-fill map +
    Reduce`` shape lives inside a nested SDFG) whose output has a WCR, or a scalar
    ``AN ─[wcr]→ MapExit`` reduction edge (a single-element write under a ``CR`` WCR)."""
    exit_node = state.exit_node(map_entry)
    scope = state.scope_subgraph(map_entry, include_entry=False, include_exit=False)
    for n in scope.nodes():
        if _has_reduction(state, n):
            return True
    # A scalar reduction WCR into the map exit (an un-lifted reduction).
    for e in state.in_edges(exit_node):
        if e.data is None or e.data.wcr is None:
            continue
        if e.data.subset is not None and e.data.subset.num_elements() == 1:
            return True
    return False


class MarkReduceAtOutput(ppl.Pass):
    """Append :data:`REDUCE_AT_OUTPUT_MARKER` to every innermost map whose body carries a
    reduction (see :func:`_map_has_body_reduction`). No-op on maps already marked."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing  # only relabels maps

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = 0
        for n, parent in sdfg.all_nodes_recursive():
            if not isinstance(n, nd.MapEntry) or not isinstance(parent, SDFGState):
                continue
            if n.map.label.endswith(REDUCE_AT_OUTPUT_MARKER):
                continue
            # Only a TRULY INNERMOST map (no maps inside, recursively) is vectorized, so
            # only it may carry the reduce_at_output reduction. Tagging an enclosing map
            # (azimint's ``i`` map over the ``j`` reduction) would relocate the wrong dim
            # and let the flatten/splice mangle a non-reduction map (user rule: vectorize
            # innermost maps).
            if _no_nested_maps(parent, n) and _map_has_body_reduction(parent, n):
                n.map.label = n.map.label + REDUCE_AT_OUTPUT_MARKER
                count += 1
        return count or None
