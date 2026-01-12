# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from typing import Dict, List, Optional, Tuple
import warnings

import networkx as nx
import sympy

from dace import properties
from dace.frontend.python import astutils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, ReturnBlock, UnstructuredControlFlow
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
        :return: The number of return blocks lifted
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
            if isinstance(region, (ConditionalBlock, UnstructuredControlFlow)):
                continue

            if region.has_cycles():
                # Do not lift conditionals if there are cycles present, since lifting conditionals requires an acyclic
                # dominance frontier for the analysis. This may lead to incorrect results if cycles are present.
                # Note that the combination of loop raising and unstructured control flow lifting should
                # already have lifted all loops, so this should not occur in practice and this warning would be cause
                # for closer inspection.
                warnings.warn(
                    f'Control flow raising: Skipping lifting conditionals for region {region.name} with cycles.')
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
                    full_cond_expression: Optional[sympy.Basic] = None
                    uncond_generated = False
                    for i, oe in enumerate(oedges):
                        branch_name = 'branch_' + str(i) + '_' + block.label
                        branch = ControlFlowRegion(branch_name, sdfg)

                        if not oe.data.is_unconditional():
                            if i == len(oedges) - 1 and oe.data.condition_sympy() == sympy.Not(full_cond_expression):
                                if uncond_generated:
                                    warnings.warn(
                                        f'Control flow raising: Found multiple unconditional branches in {block.label}')
                                uncond_generated = True
                                cond = None
                            else:
                                cond = oe.data.condition
                                if full_cond_expression is None:
                                    full_cond_expression = oe.data.condition_sympy()
                                else:
                                    full_cond_expression = sympy.And(full_cond_expression, oe.data.condition_sympy())
                        else:
                            if uncond_generated:
                                warnings.warn(
                                    f'Control flow raising: Found multiple unconditional branches in {block.label}')
                            uncond_generated = True
                            cond = None

                        conditional.add_branch(cond, branch)
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
                    if (dummy_exit is None or merge_block is not dummy_exit) and merge_block is not None:
                        graph.add_edge(conditional, merge_block, InterstateEdge())
            if dummy_exit is not None:
                region.remove_node(dummy_exit)

        n_cond_regions_post = len([x for x in sdfg.all_control_flow_blocks() if isinstance(x, ConditionalBlock)])
        lifted = n_cond_regions_post - n_cond_regions_pre
        if lifted:
            sdfg.reset_cfg_list()
            sdfg.root_sdfg.using_explicit_control_flow = True
        return lifted

    def _lift_unstructured(self, sdfg: SDFG) -> int:
        """
        Lift regions of unstructured control flow.
        When this is called, it is assumed that loops have already been lifted. This implies that any remaining
        cycles represent unstructured control flow.

        :param sdfg: The SDFG in which to lift unstructured control flow
        :return: The number of unstructured control flow blocks lifted
        """
        lifted = 0
        for cfg in sdfg.all_control_flow_regions():
            if isinstance(cfg, (UnstructuredControlFlow, ConditionalBlock)):
                continue

            if not cfg.has_cycles():
                # No cycles, no unstructured control flow.
                continue

            # Compute immediate dominators
            idom: Dict[ControlFlowBlock, ControlFlowBlock] = nx.immediate_dominators(cfg.nx, cfg.start_block)

            back_edges = set([(e.src, e.dst) for e in cfg_analysis.back_edges(cfg, idom)])

            # DFS tree edges
            dfs_tree_edges = set(nx.dfs_edges(cfg.nx, cfg.start_block))

            # Structured edges: DFS tree edges + back edges
            structured_edges = dfs_tree_edges | back_edges

            # Unstructured edges: all edges not in structured set
            unstructured_edges = set(cfg.nx.edges) - structured_edges

            # Find the single entry / single exit region around the unstructured edges and turn it into a region
            # of unstructured control flow.
            if len(unstructured_edges) > 0:
                tgt_nodes = set()
                for u, v in unstructured_edges:
                    if u not in tgt_nodes:
                        tgt_nodes.add(u)
                    if v not in tgt_nodes:
                        tgt_nodes.add(v)
                unstructured_nodes, region_entry, region_exit = cfg_analysis.find_sese_region(cfg, tgt_nodes)

                unstructured_region = UnstructuredControlFlow('unstructured_' + str(cfg.name) + '_' + str(lifted))
                unstructured_region.add_node(region_entry, is_start_block=True)
                for edge in cfg.edges():
                    if edge.src in unstructured_nodes and edge.dst in unstructured_nodes or edge.dst is region_exit:
                        unstructured_region.add_edge(edge.src, edge.dst, edge.data)
                if cfg.in_degree(region_entry) == 0:
                    # If there is no incoming edge, this is a start block.
                    cfg.add_node(unstructured_region, is_start_block=True)
                else:
                    for iedge in cfg.in_edges(region_entry):
                        if iedge.src not in unstructured_nodes:
                            cfg.add_edge(iedge.src, unstructured_region, iedge.data)
                if region_exit is not None:
                    for oedge in cfg.out_edges(region_exit):
                        if oedge.dst not in unstructured_nodes:
                            cfg.add_edge(unstructured_region, oedge.dst, oedge.data)
                for node in unstructured_nodes:
                    cfg.remove_node(node)

                lifted += 1

                sdfg.reset_cfg_list()
        return lifted

    def apply_pass(self, top_sdfg: SDFG, _) -> Optional[Tuple[int, int, int]]:
        lifted_returns = 0
        lifted_loops = 0
        lifted_unstructured = 0
        lifted_branches = 0
        for sdfg in top_sdfg.all_sdfgs_recursive():
            lifted_returns += self._lift_returns(sdfg)
            lifted_loops += sdfg.apply_transformations_repeated([LoopLifting], validate_all=False, validate=False)
            lifted_unstructured += self._lift_unstructured(sdfg)
            lifted_branches += self._lift_conditionals(sdfg)
        if lifted_branches == 0 and lifted_loops == 0 and lifted_unstructured == 0 and lifted_returns == 0:
            return None
        top_sdfg.reset_cfg_list()
        return lifted_returns, lifted_loops, lifted_branches, lifted_unstructured

    def report(self, pass_retval: Optional[Tuple[int, int, int]]):
        if pass_retval and any([x > 0 for x in pass_retval]):
            return (f'Lifted {pass_retval[0]} returns, {pass_retval[1]} loops, {pass_retval[2]} conditional blocks, ' +
                    f'and {pass_retval[3]} unstructured control flow regions')
        else:
            return 'No control flow lifted'
