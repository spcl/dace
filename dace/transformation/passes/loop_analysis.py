# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, properties
from typing import Dict, Set, Any, Tuple

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

        cfg_dependency_dict: Dict[int, Tuple[Dict[str, Set[Memlet]], Dict[str, Set[Memlet]]]] = pipeline_results[
            CFGDataDependence.__name__
        ]
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                loop_inputs, loop_outputs = cfg_dependency_dict[cfg.cfg_id]
                for data in loop_inputs:
                    if not data in loop_outputs:
                        continue

                    for input in loop_inputs[data]:
                        if cfg.loop_variable and cfg.loop_variable in input.free_symbols:
                            # may be dep. variable dependent carry
                            print('may be carry dependency')
                        else:
                            for output in loop_outputs[data]:
                                if output.subset.intersects(input.src_subset):
                                    print('carry dependency')
                print(cfg)

        return results
