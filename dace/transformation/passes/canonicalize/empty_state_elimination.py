# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Splice out empty boundary ``SDFGState`` s left by map->loop lowering."""

from typing import Any

from dace import SDFG
from dace.sdfg.state import ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.block_fusion import splice_empty_state


def elide_one(region: ControlFlowRegion) -> bool:
    """Splice out one empty, trivially-connected ``SDFGState`` of ``region``."""
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
        """Splice out empty boundary states until none remain."""
        count = 0
        changed = True
        while changed:
            changed = False
            for region in list(sdfg.all_control_flow_regions(recursive=True)):
                while elide_one(region):
                    count += 1
                    changed = True
        return count or None
