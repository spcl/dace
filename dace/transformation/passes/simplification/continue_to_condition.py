# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Dict, Optional, Any

from dace import sdfg as sd, properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ContinueBlock, ControlFlowRegion, ConditionalBlock, LoopRegion
from dace.transformation import transformation
from dace.transformation import pass_pipeline as ppl
from dace.sdfg.sdfg import SDFG


@properties.make_properties
@transformation.explicit_cf_compatible
class ContinueToCondition(ppl.Pass):
    """
    Converts all continue statements in a loop to a condition in the loop.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        for node, parent in sdfg.all_nodes_recursive():
            if self.can_be_applied(node):
                self.apply(node, sdfg)

    def can_be_applied(self, cb: ContinueBlock) -> bool:
        # Must be a continue block...
        if not isinstance(cb, ContinueBlock):
            return False 

        # ...in a conditional block...
        pg = cb.parent_graph.parent_graph
        if not isinstance(pg, ConditionalBlock):
            return False
        if len(pg.branches) != 1:
            return False

        # ...with a single branch...
        code, cfg = pg.branches[0]
        if len(cfg.nodes()) != 1:
            return False
        if cfg.nodes()[0] is not cb:
            return False

        # ...and the parent graph must have a single successor..
        outer_cfg = pg.parent_graph
        if len(outer_cfg.successors(pg)) > 1:
            return False
        
        # ..and the conditional block must be directly nested in a loop.
        if not isinstance(outer_cfg, LoopRegion):
            return False

        return True

    def apply(self, cb: ContinueBlock, sdfg: sd.SDFG):
        # If there are no successors, we can just remove the continue node
        pg: ConditionalBlock = cb.parent_graph.parent_graph
        outer_cfg = pg.parent_graph
        if len(outer_cfg.successors(pg)) == 0:
            outer_cfg.add_state_after(pg)  # Replacement
            outer_cfg.remove_node(pg)
            return

        # Eliminate the continue node
        cfg = pg.branches[0][1]
        cfg.remove_node(cb)

        # Flip the condition
        pg.branches[0][0].as_string = f"not({pg.branches[0][0].as_string})"

        # Insert all the nodes after the conditional block into the conditional block
        to_process = list(outer_cfg.successors(pg))
        old_new_mapping = {}
        to_remove = []
        while to_process:
            node = to_process.pop(0)
            to_process.extend(outer_cfg.successors(node))
            new_node = copy.deepcopy(node)
            old_new_mapping[node] = new_node
            cfg.add_node(new_node)

            for edge in outer_cfg.in_edges(node):
                if edge.src is pg:
                    continue
                cfg.add_edge(
                    old_new_mapping[edge.src],
                    new_node,
                    copy.deepcopy(edge.data),
                )
            to_remove.append(node)

        for node in to_remove:
            outer_cfg.remove_node(node)

        # Fix sdfg parents
        sdutil.set_nested_sdfg_parent_references(sdfg)
