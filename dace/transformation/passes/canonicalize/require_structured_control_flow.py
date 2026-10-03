# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stop canonicalization on control flow that ``ControlFlowRaising`` could not structure."""
from typing import Any, Dict, Optional

from dace import SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl


class RequireStructuredControlFlow(ppl.Pass):
    """Raise unless every region of the SDFG branches through ``ConditionalBlock`` and loops through ``LoopRegion``.

    Canonicalization runs ``ControlFlowRaising`` right before this check and supports nothing else.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        sdutil.require_structured_control_flow(sdfg, 'Canonicalization')
        return None
