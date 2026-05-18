# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Push a conditional block through one level of loop nesting.

The ``LoopRegion`` analogue of ``MoveIfIntoMap``: when a conditional guards a
loop, the guard is pushed inside the loop body so later fission/fusion can
cross it. Not implemented yet -- this is a structurally-complete no-op
placeholder so the canonicalization pipeline has a real, named slot.
"""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class MoveIfIntoLoop(ppl.Pass):
    """Push a guarding conditional into a loop body (no-op until implemented)."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """No-op until the conditional-into-loop rewrite is implemented.

        :param sdfg: The SDFG (unmodified).
        :param pipeline_results: Results from previous passes (unused).
        :returns: ``None`` (nothing changed).
        """
        return None
