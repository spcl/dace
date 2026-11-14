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
``merge_assignments``); anything else is left untouched.
"""

from typing import Any

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.block_fusion import splice_empty_state


def elide_one(region: ControlFlowRegion) -> bool:
    """Splice out one empty, trivially-connected ``SDFGState`` of ``region``.

    :param region: The control-flow region to scan.
    :returns: ``True`` if a state was removed (caller should re-scan).
    """
    return any(
        isinstance(state, SDFGState) and state.number_of_nodes() == 0 and splice_empty_state(region, state)
        for state in region.nodes()
    )


@transformation.explicit_cf_compatible
class EmptyStateElimination(ppl.Pass):
    """Remove empty, trivially-connected boundary states (fixpoint)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Splice out empty boundary states until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of states removed, or ``None`` if none.
        """
        count = 0
        changed = True
        while changed:
            changed = False
            for region in list(sdfg.all_control_flow_regions(recursive=True)):
                while elide_one(region):
                    count += 1
                    changed = True
        return count or None
