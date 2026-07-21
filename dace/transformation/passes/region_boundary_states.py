# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, Optional

from dace import properties
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RegionBoundaryStates(ppl.Pass):
    """
    Brackets every control flow region with an empty state in its parent region.

    Allocation is only emitted inside a state. Without the leading state, data sized by a symbol that the region's
    incoming edge assigns is allocated at the region's predecessor, before the symbol is defined. The trailing state
    gives the matching deallocation a place to go.
    """

    CATEGORY: str = 'Helper'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Reapplying would bracket the brackets, so the pass runs once.
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """
        :param sdfg: The SDFG to modify in-place.
        :return: Number of states inserted, or None if unchanged.
        """
        inserted = 0
        # Branches of a conditional are entered without an inter-state edge, so they need no boundary.
        regions = [
            cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if not isinstance(cfg, ConditionalBlock)
        ]
        for cfg in regions:
            for node in list(cfg.nodes()):
                if not isinstance(node, AbstractControlFlowRegion):
                    continue
                cfg.add_state_before(node, is_start_block=node is cfg.start_block)
                cfg.add_state_after(node)
                inserted += 2
        return inserted or None
