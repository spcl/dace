# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import copy
from typing import Any, Dict, Optional, Set, Union
from dace import SDFG, ControlFlowRegion
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
import dace.sdfg.utils as sdutil


@transformation.explicit_cf_compatible
class DetectAndLiftReductions(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self):
        return {}

    def _match_accumulation_on_tasklet(cfg: LoopRegion) -> bool:
        # AccessNode 1 of Array A access at offset [i,j]
        # AccessNode 2 of Array A access at offset [i,j]
        # AccessNode 3 of Array B access at offset [..., k, ...]
        # Tasklet that computes AN2 = AN1 op B
        # Inside a ForCFG that that iterates from 0 to K where iterator variable is k
        # Connectivity is AN1 -> | Tasklet |-> AN2
        #                 AN2 -> |         |
        # If the pattern is found remove the ForCFG add a state with Reduction library node
        pass

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        pass

        return None
