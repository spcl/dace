# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Condition normalization transformation"""

import copy

from dace import sdfg as sd, properties, InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, ConditionalBlock
from dace.transformation import transformation as xf

# FIXME: This transformation seems to not terminate.

@properties.make_properties
@xf.explicit_cf_compatible
class ConditionNesting(xf.MultiStateTransformation):
    """
    Nests the graphs before and after a conditional block into each branch of the conditional block.
    """

    cblck = xf.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cblck)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if len(graph.successors(self.cblck)) != 1:
            return False
        
        succ = graph.successors(self.cblck)[0]
        if len(graph.predecessors(succ)) != 1:
            return False

        if any([isinstance(n, ConditionalBlock) for n in graph.successors(self.cblck)]):
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        self.nest_successor(self.cblck)

    def nest_successor(self, cblck: ConditionalBlock):
        # Ensure each branch has a single sink node and sink node is empty
        assert all([len(cfg.sink_nodes()) == 1 for _, cfg in cblck.branches])

        # There should be exactly one or no else branches in each conditional block
        cblck_elses = len([True for cnd, cfg in cblck.branches if cnd is None])
        assert cblck_elses <= 1, "Multiple else branches in cblck"

        # Exactly one successor to the conditional block and predecessor to the successor
        outer_cfg = cblck.parent_graph
        assert len(outer_cfg.successors(cblck)) == 1, "Multiple successors to cblck"
        succ = outer_cfg.successors(cblck)[0]
        assert len(outer_cfg.predecessors(succ)) == 1, "Multiple predecessors to succ"

        # Add an else branch if there is none
        if cblck_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck.add_branch(None, cfg)

        # First any else branch conditons with not(cond_blck condition)
        cond_string = ""
        for cnd, cfg in cblck.branches:
            if cnd is not None:
                assert cnd.as_string != "1", "Branch condition is always true"
                if cond_string == "":
                    cond_string = f"not({cnd.as_string})"
                else:
                    cond_string = f"{cond_string} and not({cnd.as_string})"

        for i, (cnd, cfg) in enumerate(cblck.branches):
            if cnd is None:
                cblck.branches[i][0] = CodeBlock(cond_string)

        # Move the nodes after the conditional block into each branch
        sinks = {cfg: cfg.sink_nodes()[0] for _, cfg in cblck.branches}
        for _, cfg in cblck.branches:
            new_node = copy.deepcopy(succ)
            cfg.add_node(new_node)
            edge = outer_cfg.in_edges(succ)[0]
            cfg.add_edge(sinks[cfg], new_node, copy.deepcopy(edge.data))

        # Reconnect the edges
        for e in outer_cfg.out_edges(succ):
            outer_cfg.add_edge(cblck, e.dst, copy.deepcopy(e.data))
        outer_cfg.remove_node(succ)

        # Fix sdfg parents
        for _, cfg in cblck.branches:
            for st in cfg.nodes():
                for node in st.nodes():
                    if isinstance(node, sd.nodes.NestedSDFG):
                        node.sdfg.parent_sdfg = cfg.sdfg
                st.sdfg = cfg.sdfg
            cfg.reset_cfg_list()
