# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Splice out empty boundary ``SDFGState`` s left by map->loop lowering.

``MapToForLoop`` emits empty ``MapState_*_pre_state`` / ``*_post_state``
boundary states around the loop it creates. Inside a ``ConditionalBlock``
branch they make the guarded body look like a heterogeneous chain
``[empty_pre, empty_post, loop]`` instead of just ``[loop]``, which sends
``MoveIfIntoLoop`` down its imperfect-nest path and wraps *empty* states in
trivial loops -- pure churn that degrades to a sticky ``NestedSDFG``.
``StateFusionExtended`` does not remove these (it will not fuse a state into
a ``LoopRegion`` successor).

This is a canonicalization-only cleanup: it removes a state that holds no
dataflow nodes and splices its predecessors onto its successor, merging the
two edges' assignments onto the bypass edge. An empty state pinned only by an
assignment keeps two states apart, which keeps their maps apart, which blocks
fusion -- so merging rather than refusing is what makes this a fusion-prep
pass. The merge is only performed when it is provably value-preserving (see
``_merge_assignments``); anything else is left untouched.
"""
from typing import Any, Dict, Optional, Set

from dace import SDFG, symbolic
from dace.sdfg.graph import Edge
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def _merge_assignments(first: Dict[str, str], second: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Merge the assignments of two consecutive interstate edges onto one edge.

    The assignments on a single interstate edge are **not ordered** -- they are
    all evaluated against the symbol state on entry to the edge. Across the two
    original edges they *are* ordered: ``second`` sees the values ``first``
    wrote. So the merge is value-preserving exactly when no right-hand side in
    ``second`` reads a symbol ``first`` assigns; otherwise that read would
    silently change from the updated value to the stale one.

    A left-hand-side collision needs no guard: ``second`` overwrites ``first``
    both when run in sequence and in the merged dict, so the resulting value is
    the same either way.

    :param first: Assignments of the edge entering the empty state.
    :param second: Assignments of the edge leaving it.
    :returns: The merged assignment dict, or ``None`` if merging is unsound.
    """
    if not second:
        return dict(first)
    written = set(first.keys())
    if written:
        for rhs in second.values():
            if written & {str(s) for s in symbolic.pystr_to_symbolic(rhs).free_symbols}:
                return None
    merged = dict(first)
    merged.update(second)
    return merged


def _elide_one(region: ControlFlowRegion) -> bool:
    """Splice out one empty, trivially-connected ``SDFGState`` of ``region``.

    :param region: The control-flow region to scan.
    :returns: ``True`` if a state was removed (caller should re-scan).
    """
    for st in list(region.nodes()):
        if not isinstance(st, SDFGState) or st.number_of_nodes() != 0:
            continue
        in_e = list(region.in_edges(st))
        out_e = list(region.out_edges(st))
        is_start = region.start_block is st
        # Single-successor splice: every predecessor jumps straight to the
        # (unique) successor; the empty state is bypassed.
        if len(out_e) == 1 and (in_e or is_start):
            succ = out_e[0].dst
            if succ is st:
                continue
            # The successor edge's condition would have to be distributed over
            # every predecessor edge; only an unconditional one splices cleanly.
            if not out_e[0].data.is_unconditional():
                continue
            # Nothing carries the successor edge's assignments when the empty
            # state is the region's entry and has no predecessor.
            if not in_e and out_e[0].data.assignments:
                continue
            merged = [_merge_assignments(e.data.assignments, out_e[0].data.assignments) for e in in_e]
            if any(m is None for m in merged):
                continue
            for e, assignments in zip(in_e, merged):
                region.add_edge(e.src, succ, InterstateEdge(condition=e.data.condition, assignments=assignments))
            for e in in_e + out_e:
                region.remove_edge(e)
            region.remove_node(st)
            if is_start:
                region.start_block = region.node_id(succ)
            return True
        # Empty sink with no successor: a dead tail; drop it and its in-edges.
        # Its assignments are dropped with it, so they must not be observable
        # -- in a nested region an exit-edge symbol can outlive the region.
        if not out_e and in_e and not is_start and not any(e.data.assignments for e in in_e):
            for e in in_e:
                region.remove_edge(e)
            region.remove_node(st)
            return True
    return False


@transformation.explicit_cf_compatible
class EmptyStateElimination(ppl.Pass):
    """Remove empty, trivially-connected boundary states (fixpoint)."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Splice out empty boundary states until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of states removed, or ``None`` if none.
        """
        count = 0
        changed = True
        while changed:
            changed = False
            for region in list(sdfg.all_control_flow_regions(recursive=True)):
                while _elide_one(region):
                    count += 1
                    changed = True
        return count or None
