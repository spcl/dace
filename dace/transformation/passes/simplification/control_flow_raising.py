# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Optional, Tuple
import networkx as nx
from dace import properties
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
from dace.sdfg.utils import dfs_conditional
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.loop_lifting import LoopLifting


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class ControlFlowRaising(ppl.Pass):
    """
    Raises all detectable control flow that can be expressed with native SDFG structures, such as loops and branching.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def _lift_conditionals(self, sdfg: SDFG) -> int:
        cfgs = list(sdfg.all_control_flow_regions())
        n_cond_regions_pre = len([x for x in sdfg.all_control_flow_blocks() if isinstance(x, ConditionalBlock)])

        for region in cfgs:
            sinks = region.sink_nodes()
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
                            continue

                        branch_nodes = set(dfs_conditional(graph, [oe.dst], lambda _, x: x is not merge_block))
                        branch_start = branch.add_state(branch_name + '_start', is_start_block=True)
                        branch.add_nodes_from(branch_nodes)
                        branch_end = branch.add_state(branch_name + '_end')
                        branch.add_edge(branch_start, oe.dst, InterstateEdge(assignments=oe.data.assignments))
                        added = set()
                        for e in graph.all_edges(*branch_nodes):
                            if not (e in added):
                                added.add(e)
                                if e is oe:
                                    continue
                                elif e.dst is merge_block:
                                    branch.add_edge(e.src, branch_end, e.data)
                                else:
                                    branch.add_edge(e.src, e.dst, e.data)
                        graph.remove_nodes_from(branch_nodes)

                    # Connect to the end of the branch / what happens after.
                    if merge_block is not dummy_exit:
                        graph.add_edge(conditional, merge_block, InterstateEdge())
            region.remove_node(dummy_exit)

        n_cond_regions_post = len([x for x in sdfg.all_control_flow_blocks() if isinstance(x, ConditionalBlock)])
        return n_cond_regions_post - n_cond_regions_pre

    def apply_pass(self, top_sdfg: SDFG, _) -> Optional[Tuple[int, int]]:
        lifted_loops = 0
        lifted_branches = 0
        for sdfg in top_sdfg.all_sdfgs_recursive():
            lifted_loops += sdfg.apply_transformations_repeated([LoopLifting], validate_all=False, validate=False)
            lifted_branches += self._lift_conditionals(sdfg)
        if lifted_branches == 0 and lifted_loops == 0:
            return None
        return lifted_loops, lifted_branches
