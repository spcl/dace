# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Optional
from dace import properties
from dace.frontend.python import astutils
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class PruneEmptyConditionalBranches(ppl.ControlFlowRegionPass):
    """
    Prunes empty (or no-op) conditional branches from conditional blocks.
    """

    CATEGORY: str = 'Simplification'

    def __init__(self):
        super().__init__()
        self.apply_to_conditionals = True

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply(self, region: ControlFlowRegion, _) -> Optional[int]:
        if not isinstance(region, ConditionalBlock):
            return None
        removed_branches = 0
        all_branches = region.branches
        has_else = all_branches[-1][0] is None
        new_else_cond = None
        for cond, branch in all_branches:
            branch_nodes = branch.nodes()
            if (len(branch_nodes) == 0 or (len(branch_nodes) == 1 and isinstance(branch_nodes[0], SDFGState) and
                                           len(branch_nodes[0].nodes()) == 0)):
                # Found a branch we can eliminate.
                if has_else and branch is not all_branches[-1][1]:
                    # If this conditional has an else branch and that is not the branch being eliminated, we need to
                    # change that else branch to a conditional else-if branch that negates the current branch's
                    # condition.
                    negated_condition = astutils.negate_expr(cond.code[0])
                    if new_else_cond is None:
                        new_else_cond = properties.CodeBlock([negated_condition])
                    else:
                        combined_cond = astutils.and_expr(negated_condition, new_else_cond.code[0])
                        new_else_cond = properties.CodeBlock([combined_cond])
                    region.remove_branch(branch)
                else:
                    # Simple case, eliminate the branch.
                    region.remove_branch(branch)
                removed_branches += 1
        # If the else branch remains, make sure it now has the new negate-all condition.
        if region.branches and new_else_cond is not None and region.branches[-1][0] is None:
            region._branches[-1] = (new_else_cond, region._branches[-1][1])

        if len(region.branches) == 0:
            # The conditional has become entirely empty, remove it.
            replacement_node_before = region.parent_graph.add_state_before(region)
            replacement_node_after = region.parent_graph.add_state_after(region)
            region.parent_graph.add_edge(replacement_node_before, replacement_node_after, InterstateEdge())
            region.parent_graph.remove_node(region)

        if removed_branches > 0:
            region.reset_cfg_list()
            return removed_branches
        return None

