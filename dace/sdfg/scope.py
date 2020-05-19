from typing import Dict, List

NodeType = 'dace.graph.nodes.Node'
EntryNodeType = 'dace.graph.nodes.EntryNode'
ExitNodeType = 'dace.graph.nodes.ExitNode'
ScopeDictType = Dict[NodeType, List[NodeType]]


class ScopeTree(object):
    """ A class defining a scope, its parent and children scopes, variables, and
        scope entry/exit nodes. """
    def __init__(self, entrynode: EntryNodeType, exitnode: ExitNodeType):
        self.parent: 'ScopeTree' = None
        self.children: List['ScopeTree'] = []
        self.defined_vars: List[str] = []
        self.entry: EntryNodeType = entrynode
        self.exit: ExitNodeType = exitnode


def scope_contains_scope(sdict: ScopeDictType, node: NodeType,
                         other_node: NodeType) -> bool:
    """ 
    Returns true iff scope of `node` contains the scope of  `other_node`.
    """
    curnode = other_node
    nodescope = sdict[node]
    while curnode is not None:
        curnode = sdict[curnode]
        if curnode == nodescope:
            return True
    return False
