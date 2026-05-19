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

This is a conservative, canonicalization-only cleanup: it removes a state
that holds no dataflow nodes when every incident interstate edge is
unconditional and assignment-free (so splicing it out is exactly path-
preserving). Anything else is left untouched.
"""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def _trivial_edge(e) -> bool:
    """An interstate edge is trivial iff unconditional and assignment-free."""
    return (e.data.is_unconditional() and not e.data.assignments)


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
        if any(not _trivial_edge(e) for e in in_e + out_e):
            continue
        is_start = region.start_block is st
        # Single-successor splice: every predecessor jumps straight to the
        # (unique) successor; the empty state is bypassed.
        if len(out_e) == 1 and (in_e or is_start):
            succ = out_e[0].dst
            for e in in_e:
                region.add_edge(e.src, succ, e.data)
            for e in in_e + out_e:
                region.remove_edge(e)
            region.remove_node(st)
            if is_start:
                region.start_block = region.node_id(succ)
            return True
        # Empty sink with no successor: a dead tail; drop it and its in-edges.
        if not out_e and in_e and not is_start:
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

    def depends_on(self):
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
