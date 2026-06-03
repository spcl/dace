# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Eliminate dead AccessNode -> AccessNode copies inside body NSDFGs.

A "dead copy" is an ``AN1 -> AN2`` edge whose destination ``AN2`` is a
transient that is never read anywhere downstream — no out-edge in this
state, no other read in any other state of the same SDFG, no reference
from any interstate edge. The copy produces a value nobody uses; the
edge plus the orphan access node are safe to drop.

Why this matters for vectorization: the upstream
:class:`CleanAccessNodeToScalarSliceToTaskletPattern` cleaner only
rewrites ``AN1 -> AN2 -> Tasklet`` triples where the access node has
``in_degree == 1 and out_degree == 1``. A dead-end copy
(``AN1 -> AN2`` with ``out_degree(AN2) == 0``) doesn't match the
pattern and survives intact. If the surviving copy's memlet carries
``other_subset``, the downstream
:class:`RefuseOtherSubsetInNSDFG` precheck refuses it — even though the
copy itself produces nothing the kernel needs.

This pass runs before the precheck so genuinely dead copies are
deleted (and the surviving NSDFG is potentially tiling-eligible)
rather than killing the descent on a value the kernel never consumes.

Scope: body NSDFGs only (the outer SDFG's AN -> AN edges may be
scatter/gather staging that the legacy 1D detect passes consume).
"""
from typing import Any, Dict, Optional

import dace
from dace import nodes
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class EliminateDeadCopies(ppl.Pass):
    """Delete dead transient AN -> AN copies inside body NSDFGs.

    A copy ``AN1 -> AN2`` is dead iff ``AN2.data`` is a transient with
    no other use in the enclosing SDFG. Re-runs to a fixed point in case
    dropping one copy frees a chain (``A -> B -> C`` where ``C`` is dead
    becomes ``A -> B`` after one pass; ``B`` may then become dead too).
    """
    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every body NSDFG; remove dead AN -> AN copies to a fixed point.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Unused.
        :returns: Number of dead copies removed, or ``None`` if zero.
        """
        total = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            if nsdfg is sdfg:
                continue
            while True:
                removed = self._sweep_nsdfg(nsdfg)
                total += removed
                if removed == 0:
                    break
        return total if total > 0 else None

    def _sweep_nsdfg(self, nsdfg: SDFG) -> int:
        """One sweep over ``nsdfg``; return the number of dead copies removed."""
        removed = 0
        for state in list(nsdfg.states()):
            for edge in list(state.edges()):
                if not self._is_dead_copy(nsdfg, state, edge):
                    continue
                dst = edge.dst
                state.remove_edge(edge)
                # If the destination access node now has zero degree, drop it
                # along with the orphan transient descriptor (when no other
                # state still references it).
                if state.in_degree(dst) == 0 and state.out_degree(dst) == 0:
                    state.remove_node(dst)
                    if not self._referenced_anywhere(nsdfg, dst.data):
                        nsdfg.remove_data(dst.data, validate=False)
                removed += 1
        return removed

    def _is_dead_copy(self, nsdfg: SDFG, state: SDFGState, edge) -> bool:
        """A copy is dead iff its destination is an unread transient.

        Conservative checks:

        - The edge is a non-empty memlet between two AccessNodes.
        - The destination is a transient (renaming a non-transient would
          drop a kernel output).
        - The destination has no outgoing edge in this state.
        - The destination's data name appears in no other state's data
          nodes or memlets, nor in any interstate edge.
        """
        if edge.data is None or edge.data.is_empty():
            return False
        if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
            return False
        dst = edge.dst
        desc = nsdfg.arrays.get(dst.data)
        if desc is None or not desc.transient:
            return False
        if state.out_degree(dst) != 0:
            return False
        if self._referenced_anywhere(nsdfg, dst.data, exclude_state=state):
            return False
        return True

    def _referenced_anywhere(self, nsdfg: SDFG, name: str, exclude_state: Optional[SDFGState] = None) -> bool:
        """True iff ``name`` is read or written anywhere in ``nsdfg``.

        :param nsdfg: SDFG to scan.
        :param name: Data name to look for.
        :param exclude_state: When set, this state is skipped (used by the
            in-state check that has already classified its own access).
        :returns: ``True`` if ``name`` is referenced anywhere else.
        """
        for state in nsdfg.states():
            if state is exclude_state:
                continue
            for dn in state.data_nodes():
                if dn.data == name:
                    return True
            for e in state.edges():
                if e.data is not None and e.data.data == name:
                    return True
        for ise in nsdfg.all_interstate_edges():
            if any(str(s) == name for s in ise.data.free_symbols):
                return True
        return False
