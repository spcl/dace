# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties
from typing import Dict, Set, Any

from dace.transformation.pass_pipeline import Pass
from dace.transformation.passes.control_flow_region_analysis import CFGDataDependence


@properties.make_properties
class LoopCarryDependencyAnalysis(ppl.Pass):
    """
    Analyze the data dependencies between loop iterations for loop regions.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self) -> Set[type[Pass] | Pass]:
        return {CFGDataDependence}

    def apply_pass(self, top_sdfg: SDFG,
                   pipeline_results: Dict[str, Any]) -> Dict[int, Dict[LoopRegion, Set[Memlet]]]:
        """
        :return: A dictionary ..
        """
        results = defaultdict()

        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                pass

        return results
