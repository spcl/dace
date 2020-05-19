from typing import Dict, List, Tuple

from dace import dtypes, symbolic
from dace.config import Config

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


def is_devicelevel(sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState',
                   node: NodeType) -> bool:
    """ Tests whether a node in an SDFG is contained within GPU device-level
        code.
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in device-level code, False otherwise.
    """
    from dace.graph import nodes as nd
    from dace.sdfg.sdfg import SDFGState

    while sdfg is not None:
        sdict = state.scope_dict()
        scope = sdict[node]
        while scope is not None:
            if scope.schedule in dtypes.GPU_SCHEDULES:
                return True
            scope = sdict[scope]
        # Traverse up nested SDFGs
        if sdfg.parent is not None:
            if isinstance(sdfg.parent, SDFGState):
                parent = sdfg.parent.parent
            else:
                parent = sdfg.parent
            state, node = next(
                (s, n) for s in parent.nodes() for n in s.nodes()
                if isinstance(n, nd.NestedSDFG) and n.sdfg.name == sdfg.name)
        else:
            parent = sdfg.parent
        sdfg = parent
    return False


def devicelevel_block_size(sdfg: 'dace.sdfg.SDFG',
                           state: 'dace.sdfg.SDFGState',
                           node: NodeType) -> Tuple[symbolic.SymExpr]:
    """ Returns the current thread-block size if the given node is enclosed in
        a GPU kernel, or None otherwise.
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: A tuple of sizes or None if the node is not in device-level 
                 code.
    """
    from dace.sdfg.sdfg import SDFGState
    from dace.graph import nodes as nd

    while sdfg is not None:
        sdict = state.scope_dict()
        scope = sdict[node]
        while scope is not None:
            if scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                return tuple(scope.map.range.size())
            elif scope.schedule == dtypes.ScheduleType.GPU_Device:
                # No thread-block map, use config default
                return tuple(
                    int(s) for s in Config.get(
                        'compiler', 'cuda', 'default_block_size').split(','))
            elif scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
                # Dynamic thread-block map, use configured value
                return tuple(
                    int(s)
                    for s in Config.get('compiler', 'cuda',
                                        'dynamic_map_block_size').split(','))

            scope = sdict[scope]
        # Traverse up nested SDFGs
        if sdfg.parent is not None:
            if isinstance(sdfg.parent, SDFGState):
                parent = sdfg.parent.parent
            else:
                parent = sdfg.parent
            state, node = next(
                (s, n) for s in parent.nodes() for n in s.nodes()
                if isinstance(n, nd.NestedSDFG) and n.sdfg.name == sdfg.name)
        else:
            parent = sdfg.parent
        sdfg = parent
    return None