# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as snodes
from dace.sdfg.analysis.schedule_tree import treenodes as tnodes
from dace.sdfg.graph import NodeNotFoundError
from typing import Tuple

__df_nodes = (
    # tnodes.ViewNode, tnodes.RefSetNode,
    # tnodes.CopyNode, tnodes.DynScopeCopyNode,
    tnodes.TaskletNode,
    tnodes.LibraryCall,
    tnodes.MapScope,
    tnodes.ConsumeScope,
    tnodes.PipelineScope)


def find_tnode_in_sdfg(tnode: tnodes.ScheduleTreeNode, top_level_sdfg: SDFG) -> Tuple[snodes.Node, SDFGState, SDFG]:
    if tnode not in __df_nodes:
        raise NotImplementedError(f"The `find_dfnode_in_sdfg` does not support {type(tnode)} nodes.")
    for n, s in top_level_sdfg.all_nodes_recursive():
        if n is tnode.node:
            return n, s, s.parent
    raise NodeNotFoundError(f"Node {tnode} not found in SDFG.")


def find_snode_in_tree(snode: snodes.Node,
                       tree: tnodes.ScheduleTreeNode) -> Tuple[tnodes.ScheduleTreeScope, tnodes.ScheduleTreeNode]:
    pnode = None
    cnode = None
    frontier = [(tree, child) for child in tree.children]
    while frontier:
        parent, child = frontier.pop()
        if hasattr(child, 'node') and child.node is snode:
            pnode = parent
            cnode = child
            break
        frontier.extend([(child, c) for c in child.children])
    if not pnode:
        raise NodeNotFoundError(f"Node {snode} not found in ScheduleTree.")
    return pnode, cnode


def find_parent(tnode: tnodes.ScheduleTreeNode, tree: tnodes.ScheduleTreeNode) -> tnodes.ScheduleTreeScope:
    pnode = None
    frontier = [(tree, child) for child in tree.children]
    while frontier:
        parent, child = frontier.pop()
        if child is tnode:
            pnode = parent
            break
        frontier.extend([(child, c) for c in child.children])
    if not pnode:
        raise NodeNotFoundError(f"Node {tnode} not found in ScheduleTree.")
    return pnode
