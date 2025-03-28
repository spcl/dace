# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dace import sdfg as sd, properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import (
    ControlFlowRegion,
    ConditionalBlock,
    ControlFlowBlock,
    CodeBlock,
)
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class ConditionNesting(xf.MultiStateTransformation):
    """
    Nests the graphs before and after a conditional block into each branch of the conditional block.
    """

    cfb = xf.PatternNode(ControlFlowBlock)
    cblck = xf.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.cblck, cls.cfb),
            sdutil.node_path_graph(cls.cfb, cls.cblck),
        ]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Skip ConditinalBlocks
        if isinstance(self.cfb, ConditionalBlock):
            return False

        # Case 1: Nest successor
        if expr_index == 0:
            if len(graph.successors(self.cblck)) != 1:
                return False
            return True

        # Case 2: Nest predecessor
        if expr_index == 1:
            if len(graph.predecessors(self.cblck)) != 1:
                return False
            if len(graph.edges_between(self.cfb, self.cblck)) != 1:
                return False

            edge = graph.edges_between(self.cfb, self.cblck)[0]
            if edge.data.condition.as_string != "1":
                return False
            if any(
                [
                    cnd is not None
                    and cnd.get_free_symbols() & edge.data.assignments.keys() != set()
                    for cnd, _ in self.cblck.branches
                ]
            ):
                return False
            return True

        return False

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        if self.expr_index == 0:
            self.nest_successor(sdfg, self.cblck)
        elif self.expr_index == 1:
            self.nest_predecessor(sdfg, self.cblck)

    def nest_successor(self, sdfg: sd.SDFG, cblck: ConditionalBlock):
        # Ensure each branch has a single sink node and sink node is empty
        assert all([len(cfg.sink_nodes()) == 1 for _, cfg in cblck.branches])

        # Exactly one successor to the conditional block and predecessor to the successor
        outer_cfg = cblck.parent_graph
        assert len(outer_cfg.successors(cblck)) == 1, "Multiple successors to cblck"
        succ = outer_cfg.successors(cblck)[0]
        assert len(outer_cfg.predecessors(succ)) == 1, "Multiple predecessors to succ"

        # There should be exactly one or no else branches in the parent conditional block
        cblck_elses = len([True for cnd, cfg in cblck.branches if cnd is None])
        assert cblck_elses <= 1, "Multiple else branches in cblck"

        # Add an else branch if there is none
        if cblck_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck.add_branch(None, cfg)

        # Move the nodes after the conditional block into each branch
        sinks = {cfg: cfg.sink_nodes()[0] for _, cfg in cblck.branches}
        for _, cfg in cblck.branches:
            new_node = copy.deepcopy(succ)
            cfg.add_node(new_node)

            for edge in outer_cfg.in_edges(succ):
                cfg.add_edge(sinks[cfg], new_node, copy.deepcopy(edge.data))

        # Reconnect the ConditionalBlock edges
        for e in outer_cfg.out_edges(succ):
            outer_cfg.add_edge(cblck, e.dst, copy.deepcopy(e.data))
        outer_cfg.remove_node(succ)

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblck.branches):
            cfg.label = f"{cblck.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"

        # Fix SDFG parents
        sdutil.set_nested_sdfg_parent_references(sdfg)

    def nest_predecessor(self, sdfg: sd.SDFG, cblck: ConditionalBlock):
        # Ensure each branch has a single source node and source node is empty
        assert all([len(cfg.source_nodes()) == 1 for _, cfg in cblck.branches])

        # Exactly one predecessor to the conditional block and successor to the predecessor
        outer_cfg = cblck.parent_graph
        assert len(outer_cfg.predecessors(cblck)) == 1, "Multiple predecessors to cblck"
        pred = outer_cfg.predecessors(cblck)[0]
        assert len(outer_cfg.successors(pred)) == 1, "Multiple successors to pred"

        # There should be exactly one or no else branches in the parent conditional block
        cblck_elses = len([True for cnd, cfg in cblck.branches if cnd is None])
        assert cblck_elses <= 1, "Multiple else branches in cblck"

        # Add an else branch if there is none
        if cblck_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck.add_branch(None, cfg)

        # Move the nodes before the conditional block into each branch
        sources = {cfg: cfg.source_nodes()[0] for _, cfg in cblck.branches}
        for cnd, cfg in cblck.branches:
            new_node = copy.deepcopy(pred)
            cfg.add_node(new_node, is_start_block=True)

            for edge in outer_cfg.out_edges(pred):
                cfg.add_edge(new_node, sources[cfg], copy.deepcopy(edge.data))

                # Edge may not contain assignments that cblck depends on
                assert (
                    cnd is None
                    or cnd.get_free_symbols() & edge.data.assignments.keys() == set()
                ), "Assignments in edge are used in cblck"

        # Reconnect the ConditionalBlock edges
        for e in outer_cfg.in_edges(pred):
            outer_cfg.add_edge(e.src, cblck, copy.deepcopy(e.data))
        outer_cfg.remove_node(pred)

        # If cblck doesn't have predecessory anymore, set start block of parent
        if len(outer_cfg.predecessors(cblck)) == 0:
            outer_cfg.start_block = outer_cfg.node_id(cblck)

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblck.branches):
            cfg.label = f"{cblck.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"

        # Fix SDFG parents
        sdutil.set_nested_sdfg_parent_references(sdfg)
