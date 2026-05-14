# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import ControlFlowRegion


def move_branch_cfg_up_discard_conditions(if_block: ConditionalBlock, body_to_take: ControlFlowRegion):
    """
    Moves a branch of a conditional block up in the control flow graph (if_block) (CFG),
    replacing the conditional with the selected branch (body_to_take), discarding
    the conditional check and other branches. Connects the body to previous in and out neighbors of the CFG.
    """
    bodies = {b for _, b in if_block.branches}
    assert body_to_take in bodies
    assert isinstance(if_block, ConditionalBlock)

    graph = if_block.parent_graph

    node_map = dict()
    new_start_block = None
    new_end_block = None

    for node in body_to_take.nodes():
        copynode = copy.deepcopy(node)
        node_map[node] = copynode
        start_block_case = (body_to_take.start_block == node) and (graph.start_block == if_block)
        if body_to_take.start_block == node:
            assert new_start_block is None
            new_start_block = copynode
        if body_to_take.out_degree(node) == 0:
            assert new_end_block is None
            new_end_block = copynode
        graph.add_node(copynode, is_start_block=start_block_case)

    for edge in body_to_take.edges():
        src = node_map[edge.src]
        dst = node_map[edge.dst]
        graph.add_edge(src, dst, copy.deepcopy(edge.data))

    for ie in graph.in_edges(if_block):
        graph.add_edge(ie.src, new_start_block, copy.deepcopy(ie.data))
    for oe in graph.out_edges(if_block):
        graph.add_edge(new_end_block, oe.dst, copy.deepcopy(oe.data))

    graph.remove_node(if_block)
