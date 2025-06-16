# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from typing import List, Optional, Tuple

import networkx as nx
import sympy

from dace import properties
from dace.frontend.python import astutils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, ReturnBlock
from dace.sdfg.utils import dfs_conditional
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.loop_lifting import LoopLifting


@properties.make_properties
@transformation.explicit_cf_compatible
class ControlFlowRaising(ppl.Pass):
    """
    Raises all detectable control flow that can be expressed with native SDFG structures, such as loops and branching.
    """

    CATEGORY: str = 'Simplification'

    raise_sink_node_returns = properties.Property(
        dtype=bool,
        default=False,
        desc='Whether or not to lift sink nodes in an SDFG context to explicit return blocks.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def _lift_returns(self, sdfg: SDFG) -> int:
        """
        Make any implicit early program exits explicit by inserting return blocks.
        An implicit early program exit is a control flow block with not at least one unconditional edge leading out of
        it, or where there is no 'catchall' condition that negates all other conditions. For any such transition, if
        the condition(s) is / are not met, the SDFG halts.
        This method detects such situations and inserts an explicit transition to a return block for each such missing
        unconditional edge or 'catchall' condition. Note that this is only performed on the top-level control flow
        region, i.e., the SDFG itself. Any implicit early stops inside nested regions only end the context of that
        region, and not the entire SDFG.

        :param sdfg: The SDFG in which to lift returns
        :returns: The number of return blocks lifted
        """
        returns_lifted = 0
        for nd in sdfg.nodes():
            # Existing returns can be skipped.
            if isinstance(nd, ReturnBlock):
                continue

            # First check if there is an unconditional outgoing edge.
            has_unconditional = False
            full_cond_expression: Optional[List[ast.AST]] = None
            oedges = sdfg.out_edges(nd)
            for oe in oedges:
                if oe.data.is_unconditional():
                    has_unconditional = True
                    break
                else:
                    if full_cond_expression is None:
                        full_cond_expression = oe.data.condition.code[0]
                    else:
                        full_cond_expression = astutils.and_expr(full_cond_expression, oe.data.condition.code[0])
            # If there is no unconditional outgoing edge, there may be a catchall that is the negation of all other
            # conditions.
            # NOTE: Checking that for the general case is expensive. For now, we check it for the case of two outgoing
            #       edges, where the two edges are a negation of one another, which is cheap. In any other case, an
            #       explicit return is added with the negation of everything. This is conservative and always correct,
            #       but may insert a stray (and unreachable) return in rare cases. That case should hardly ever occur
            #       and does not lead to any negative side effects.
            if has_unconditional:
                insert_return = False
            else:
                if len(oedges) == 2 and oedges[0].data.condition_sympy() == sympy.Not(oedges[1].data.condition_sympy()):
                    insert_return = False
                else:
                    insert_return = True

            if insert_return:
                if full_cond_expression is None:
                    # If there is no condition, there are no outgoing edges - so this is already an explicit program
                    # exit by being a sink node.
                    if self.raise_sink_node_returns:
                        ret_block = ReturnBlock(sdfg.name + '_return')
                        sdfg.add_node(ret_block, ensure_unique_name=True)
                        sdfg.add_edge(nd, ret_block, InterstateEdge())
                        returns_lifted += 1
                else:
                    ret_block = ReturnBlock(nd.label + '_return')
                    sdfg.add_node(ret_block, ensure_unique_name=True)
                    catchall_condition_expression = astutils.negate_expr(full_cond_expression)
                    ret_edge = InterstateEdge(condition=properties.CodeBlock([catchall_condition_expression]))
                    sdfg.add_edge(nd, ret_block, ret_edge)
                    returns_lifted += 1

        return returns_lifted

    def _lift_conditionals(self, sdfg: SDFG) -> int:
        cfgs = list(sdfg.all_control_flow_regions())
        n_cond_regions_pre = len([x for x in sdfg.all_control_flow_blocks() if isinstance(x, ConditionalBlock)])

        for region in cfgs:
            if isinstance(region, ConditionalBlock):
                continue

            # If there are multiple sinks, create a dummy exit node for finding branch merges. If there is at least one
            # non-return block sink, do not count return blocks as sink nodes. Doing so could cause branches to inter-
            # connect unnecessarily, thus preventing lifting.
            non_return_sinks = [s for s in region.sink_nodes() if not isinstance(s, ReturnBlock)]
            sinks = non_return_sinks if len(non_return_sinks) > 0 else region.sink_nodes()
            dummy_exit = None
            if len(sinks) > 1:
                dummy_exit = region.add_state('__DACE_DUMMY')
                for s in sinks:
                    region.add_edge(s, dummy_exit, InterstateEdge())
            idom = nx.immediate_dominators(region.nx, region.start_block)
            alldoms = cfg_analysis.all_dominators(region, idom)
            branch_merges = cfg_analysis.branch_merges(region, idom, alldoms)

            for block in region.nodes():
                graph = block.parent_graph
                oedges = graph.out_edges(block)
                if len(oedges) > 1 and block in branch_merges:
                    merge_block = branch_merges[block]

                    # Construct the branching block.
                    conditional = ConditionalBlock('conditional_' + block.label, sdfg, graph)
                    graph.add_node(conditional)
                    # Connect it.
                    graph.add_edge(block, conditional, InterstateEdge())

                    # Populate branches.
                    for i, oe in enumerate(oedges):
                        branch_name = 'branch_' + str(i) + '_' + block.label
                        branch = ControlFlowRegion(branch_name, sdfg)
                        conditional.add_branch(oe.data.condition, branch)
                        if oe.dst is merge_block:
                            # Empty branch.
                            branch.add_state('noop')
                            graph.remove_edge(oe)
                            continue

                        branch_nodes = set(dfs_conditional(graph, [oe.dst], lambda _, x: x is not merge_block))
                        branch_start = branch.add_state(branch_name + '_start', is_start_block=True)
                        branch.add_nodes_from(branch_nodes)
                        branch.add_edge(branch_start, oe.dst, InterstateEdge(assignments=oe.data.assignments))
                        added = set()
                        for e in graph.all_edges(*branch_nodes):
                            if not (e in added):
                                added.add(e)
                                if e is oe:
                                    continue
                                elif e.dst is merge_block:
                                    if e.data.assignments or not e.data.is_unconditional():
                                        branch.add_edge(e.src, branch.add_state(branch_name + '_end'), e.data)
                                else:
                                    branch.add_edge(e.src, e.dst, e.data)
                        graph.remove_nodes_from(branch_nodes)

                    # Connect to the end of the branch / what happens after.
                    if dummy_exit is None or merge_block is not dummy_exit:
                        graph.add_edge(conditional, merge_block, InterstateEdge())
            if dummy_exit is not None:
                region.remove_node(dummy_exit)

        n_cond_regions_post = len([x for x in sdfg.all_control_flow_blocks() if isinstance(x, ConditionalBlock)])
        lifted = n_cond_regions_post - n_cond_regions_pre
        return lifted

    def apply_pass(self, top_sdfg: SDFG, _) -> Optional[Tuple[int, int, int]]:
        lifted_returns = 0
        lifted_loops = 0
        lifted_branches = 0
        for sdfg in top_sdfg.all_sdfgs_recursive():
            lifted_returns += self._lift_returns(sdfg)
            lifted_loops += sdfg.apply_transformations_repeated([LoopLifting], validate_all=False, validate=False)
            lifted_branches += self._lift_conditionals(sdfg)
        if lifted_branches == 0 and lifted_loops == 0:
            return None
        top_sdfg.reset_cfg_list()
        return lifted_returns, lifted_loops, lifted_branches

    def report(self, pass_retval: Optional[Tuple[int, int, int]]):
        if pass_retval and any([x > 0 for x in pass_retval]):
            return f'Lifted {pass_retval[0]} returns, {pass_retval[1]} loops, and {pass_retval[2]} conditional blocks'
        else:
            return 'No control flow lifted'
