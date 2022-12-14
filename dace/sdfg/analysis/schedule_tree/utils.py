# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as snodes
from dace.sdfg.analysis.schedule_tree import treenodes as tnodes
from dace.sdfg.graph import NodeNotFoundError
from typing import List, Tuple, Union


def partition_scope_body(scope: tnodes.ScheduleTreeScope) -> List[Union[tnodes.ScheduleTreeNode, List[tnodes.ScheduleTreeNode]]]:
    """
    Partitions a scope's body to ScheduleTree nodes, when they define their own sub-scope, and lists of ScheduleTree
    nodes that are children to the same sub-scope. For example, IfScopes, ElifScopes, and ElseScopes are generally
    children to a general "If-Elif-Else-Scope".

    :param scope: The scope.
    :return: A list of (lists of) ScheduleTree nodes.
    """

    num_children = len(scope.children)
    partition = []
    i = 0
    while i < num_children:
        child = scope.children[i]
        if isinstance(child, tnodes.IfScope):
            # Start If-Elif-Else-Scope.
            ifelse = [child]
            i += 1
            while i < num_children and isinstance(scope.children[i], (tnodes.ElifScope, tnodes.ElseScope)):
                ifelse.append(child)
                i += 1
            partition.append(ifelse)
        else:
            partition.append(child)
            i += 1
    return partition
