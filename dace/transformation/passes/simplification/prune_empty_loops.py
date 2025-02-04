# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dace import properties
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class PruneEmptyLoops(ppl.Pass):
    """
    Prunes empty (or no-op) loops.
    """

    CATEGORY: str = 'Simplification'

    def __init__(self):
        super().__init__()
        self.apply_to_conditionals = True

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply_pass(self, sdfg, pipeline_results):
        loops = [l for l in sdfg.all_control_flow_regions(recursive=True) if isinstance(l, LoopRegion)]

        removed_loops = 0
        for loop in loops:
            body = loop.nodes()
            if len(body) == 0 or (len(body) == 1 and isinstance(body[0], SDFGState) and len(body[0].nodes()) == 0):
                # Loop is empty / does nothing.
                replacement_node_before = loop.parent_graph.add_state_before(loop)
                replacement_node_after = loop.parent_graph.add_state_after(loop)
                loop.parent_graph.add_edge(replacement_node_before, replacement_node_after, InterstateEdge())
                loop.parent_graph.remove_node(loop)
                removed_loops += 1

        if removed_loops > 0:
            sdfg.reset_cfg_list()
            return removed_loops
        return None
