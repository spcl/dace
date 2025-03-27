# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Condition normalization transformation"""

import copy

from dace import sdfg as sd, properties
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, ConditionalBlock
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class ConditionFusion(xf.MultiStateTransformation):
    """
    Fuses conditional blocks that are either nested or consecutive.
    """

    cblck1 = xf.PatternNode(ConditionalBlock)
    cblck2 = xf.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.cblck1, cls.cblck2),
            sdutil.node_path_graph(cls.cblck1),
        ]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Case 1: Consecutive conditional blocks
        if expr_index == 0:
            if len(graph.successors(self.cblck1)) != 1:
                return False
            if len(graph.predecessors(self.cblck2)) != 1:
                return False
            if len(graph.edges_between(self.cblck1, self.cblck2)) != 1:
                return False

            edge = graph.edges_between(self.cblck1, self.cblck2)[0]
            if edge.data.condition.as_string != "1":
                return False
            if any(
                [
                    cnd is not None
                    and cnd.get_free_symbols() & edge.data.assignments.keys() != set()
                    for cnd, _ in self.cblck2.branches
                ]
            ):
                return False

            return True

        # Case 2: Nested conditional blocks
        if expr_index == 1:
            if len(graph.predecessors(self.cblck1)) != 0:
                return False
            if len(graph.successors(self.cblck1)) != 0:
                return False

            parent_cfg = self.cblck1.parent_graph
            if not hasattr(parent_cfg, "parent_graph"):
                return False
            parent_cfg = parent_cfg.parent_graph
            if not isinstance(parent_cfg, ConditionalBlock):
                return False
            return True

        return False

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        if self.expr_index == 0:
            self.fuse_consecutive_conditions(self.cblck1, self.cblck2)
        elif self.expr_index == 1:
            self.fuse_nested_conditions(self.cblck1)
        sdutil.set_nested_sdfg_parent_references(sdfg)

    def fuse_consecutive_conditions(
        self, cblck1: ConditionalBlock, cblck2: ConditionalBlock
    ):
        # Check if cblck1 has a single sink node for each branch
        assert all([len(cfg.sink_nodes()) == 1 for _, cfg in cblck1.branches])

        # Check if it only has one successor and that successor is a conditional block
        outer_cfg = cblck1.parent_graph
        assert (
            len(outer_cfg.successors(cblck1)) == 1
        ), "Conditional block has no or multiple successors"
        assert (
            outer_cfg.successors(cblck1)[0] == cblck2
        ), "Consecutive conditional block is not a successor"

        # Check if cblck2 has a single predecessor
        assert (
            len(outer_cfg.predecessors(cblck2)) == 1
        ), "Conditional block has no or multiple predecessors"

        # Edge between cblck1 and cblck2 should not have any conditions
        assert (
            len(outer_cfg.edges_between(cblck1, cblck2)) == 1
        ), "Multiple edges between conditional blocks"

        cblck_edge = outer_cfg.edges_between(cblck1, cblck2)[0]
        assert (
            cblck_edge.data.condition.as_string == "1"
        ), "Edge between conditional blocks has conditions"

        # Edge between cblck1 and cblck2 may have assignments, but only if none of the conditions in cblck2 depend on them
        assert all(
            [
                cnd is None
                or cnd.get_free_symbols() & cblck_edge.data.assignments.keys() == set()
                for cnd, _ in cblck2.branches
            ]
        ), "Assignments in edge are used in cblck2"

        # There should be exactly one or no else branches in each conditional block
        cblck1_elses = len([True for cnd, cfg in cblck1.branches if cnd is None])
        cblck2_elses = len([True for cnd, cfg in cblck2.branches if cnd is None])
        assert cblck1_elses <= 1, "Multiple else branches in cblck1"
        assert cblck2_elses <= 1, "Multiple else branches in cblck2"

        # Add an else branch if there is none
        if cblck1_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck1.add_branch(None, cfg)
        if cblck2_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck2.add_branch(None, cfg)

        # First any else branch conditons with not(cond_blck condition)
        for cblck in [cblck1, cblck2]:
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

        # Clone each branch of cblck1
        orig_blck1_branches = len(cblck1.branches)
        for _ in range(len(cblck2.branches) - 1):
            for cnd, cfg in list(cblck1.branches):
                cnd2 = copy.deepcopy(cnd)
                cfg2 = copy.deepcopy(cfg)
                cblck1.add_branch(cnd2, cfg2)

        # Add the conditons of cblck2 to cblck1 and copy the cfgs
        for i, (cnd, cfg) in enumerate(cblck2.branches):
            for j in range(orig_blck1_branches):
                off = orig_blck1_branches * i + j
                cblck1.branches[off][
                    0
                ].as_string = (
                    f"({cblck1.branches[off][0].as_string}) and ({cnd.as_string})"
                )

                old_new_mapping = {}
                for node in cfg.nodes():
                    new_node = copy.deepcopy(node)
                    old_new_mapping[node] = new_node
                    cblck1.branches[off][1].add_node(new_node)

                for node in cfg.nodes():
                    new_node = old_new_mapping[node]
                    if node is cfg.start_block:
                        cblck1.branches[off][1].add_edge(
                            cblck1.branches[off][1].sink_nodes()[0],
                            new_node,
                            copy.deepcopy(cblck_edge.data),
                        )

                    for edge in cfg.in_edges(node):
                        cblck1.branches[off][1].add_edge(
                            old_new_mapping[edge.src],
                            new_node,
                            copy.deepcopy(edge.data),
                        )

        # Remove cblck2
        for e in outer_cfg.out_edges(cblck2):
            outer_cfg.add_edge(cblck1, e.dst, copy.deepcopy(e.data))
        outer_cfg.remove_node(cblck2)

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblck1.branches):
            cfg.label = f"{cblck1.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"

    def fuse_nested_conditions(self, cblck1: ConditionalBlock):
        nbranch = cblck1.parent_graph

        # Check if cblck1 has no predecessors and no successors
        assert len(nbranch.predecessors(cblck1)) == 0
        assert len(nbranch.successors(cblck1)) == 0

        # Check if cblck1 is nested in another conditional block
        assert hasattr(nbranch, "parent_graph")
        assert isinstance(nbranch.parent_graph, ConditionalBlock)
        cblckp = nbranch.parent_graph

        # There should be exactly one or no else branches in the parent conditional block
        cblck1_elses = len([True for cnd, cfg in cblck1.branches if cnd is None])
        cblckp_elses = len([True for cnd, cfg in cblckp.branches if cnd is None])
        assert cblck1_elses <= 1, "Multiple else branches in cblck1"
        assert cblckp_elses <= 1, "Multiple else branches in cblckp"

        # Add an else branch if there is none
        if cblck1_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck1.add_branch(None, cfg)
        if cblckp_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblckp.add_branch(None, cfg)

        # First any else branch conditons with not(cond_blck condition)
        for cblck in [cblck1, cblckp]:
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

        # Find condition of cblck1 in cblckp
        cond = None
        for cnd, cfg in cblckp.branches:
            if cfg == nbranch:
                cond = cnd
                break
        assert cond is not None

        # For each branch of cblck1, add a branch to cblckp
        for cnd1, cfg1 in cblck1.branches:
            cnd2 = copy.deepcopy(cnd1)
            cnd2.as_string = f"({cond.as_string}) and ({cnd2.as_string})"
            cfg2 = copy.deepcopy(cfg1)
            cblckp.add_branch(cnd2, cfg2)

        # Remove original branch from cblckp
        cblckp.remove_branch(nbranch)

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblckp.branches):
            cfg.label = f"{cblckp.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"
