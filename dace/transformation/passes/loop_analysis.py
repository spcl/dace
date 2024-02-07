# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties
from typing import Dict, Set


@properties.make_properties
class LoopCarryDependencyAnalysis(ppl.Pass):
    """
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges)

    def apply_pass(self, top_sdfg: SDFG, _) -> Dict[int, Dict[SDFGState, Set[SDFGState]]]:
        """
        :return: A dictionary ..
        """
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                pass
        return {}
