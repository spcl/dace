# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
from typing import Any, Dict, List, Tuple

import dace
from dace import dtypes, symbolic
from dace.config import Config
from dace.sdfg import nodes as nd
from dace.sdfg.state import StateSubgraphView

NodeType = 'dace.sdfg.nodes.Node'
EntryNodeType = 'dace.sdfg.nodes.EntryNode'
ExitNodeType = 'dace.sdfg.nodes.ExitNode'
ScopeDictType = Dict[NodeType, List[NodeType]]


class ScopeTree(object):
    """ A class defining a scope, its parent and children scopes, and
        scope entry/exit nodes. """

    def __init__(self, entrynode: EntryNodeType, exitnode: ExitNodeType):
        self.parent: 'ScopeTree' = None
        self.children: List['ScopeTree'] = []
        self.entry: EntryNodeType = entrynode
        self.exit: ExitNodeType = exitnode


class ScopeSubgraphView(StateSubgraphView):
    """ An extension to SubgraphView that enables the creation of scope
        dictionaries in subgraphs and free symbols. """

    def __init__(self, graph, subgraph_nodes, entry_node):
        super().__init__(graph, subgraph_nodes)
        self.entry = entry_node

    @property
    def parent(self):
        return self._graph.parent

    def top_level_transients(self):
        """ Iterate over top-level transients of this subgraph. """
        schildren = self.scope_children()
        sdfg = self.parent
        result = set()
        for node in schildren[self.entry]:
            if isinstance(node, nd.AccessNode) and node.desc(sdfg).transient:
                result.add(node.data)
        return result


def _scope_subgraph(graph, entry_node, include_entry, include_exit) -> ScopeSubgraphView:
    if not isinstance(entry_node, nd.EntryNode):
        raise TypeError("Received {}: should be dace.nodes.EntryNode".format(type(entry_node).__name__))
    node_to_children = graph.scope_children()
    if include_exit:
        children_nodes = set(node_to_children[entry_node])
    else:
        children_nodes = set(n for n in node_to_children[entry_node] if not isinstance(n, nd.ExitNode))
    map_nodes = [node for node in children_nodes if isinstance(node, nd.EntryNode)]
    while len(map_nodes) > 0:
        next_map_nodes = []
        # Traverse children map nodes
        for map_node in map_nodes:
            # Get child map subgraph (1 level)
            more_nodes = set(node_to_children[map_node])
            # Unionize children_nodes with new nodes
            children_nodes |= more_nodes
            # Add nodes of the next level to next_map_nodes
            next_map_nodes.extend([node for node in more_nodes if isinstance(node, nd.EntryNode)])
        map_nodes = next_map_nodes

    if include_entry:
        children_nodes.add(entry_node)

    # Preserve order of nodes
    return ScopeSubgraphView(graph, [n for n in graph.nodes() if n in children_nodes], entry_node)


def _scope_dict_inner(graph, node_queue, current_scope, node_to_children, result):
    """ Returns a queue of nodes that are external to the current scope. """
    # Initialize an empty list, if necessary
    if node_to_children and current_scope not in result:
        result[current_scope] = []

    external_queue = collections.deque()

    visited = set()
    while len(node_queue) > 0:
        node = node_queue.popleft()

        # If this node has been visited already, skip it
        if node in visited:
            continue
        visited.add(node)

        # Set the node parent (or its parent's children)
        if not node_to_children:
            result[node] = current_scope
        else:
            result[current_scope].append(node)

        successors = [n for n in graph.successors(node) if n not in visited]

        # If this is an Entry Node, we need to recurse further
        if isinstance(node, nd.EntryNode):
            node_queue.extend(_scope_dict_inner(graph, collections.deque(successors), node, node_to_children, result))
        # If this is an Exit Node, we push the successors to the external queue
        elif isinstance(node, nd.ExitNode):
            external_queue.extend(successors)
        # Otherwise, it is a plain node, and we push its successors to the same queue
        else:
            node_queue.extend(successors)

    return external_queue


def _scope_dict_to_ids(state: 'dace.sdfg.SDFGState', scope_dict: ScopeDictType):
    """ Return a JSON-serializable dictionary of a scope dictionary,
        using integral node IDs instead of object references. """

    def node_id_or_none(node):
        if node is None: return -1
        return state.node_id(node)

    res = {}
    for k, v in scope_dict.items():
        res[node_id_or_none(k)] = [node_id_or_none(vi) for vi in v] if v is not None else []
    return res


def scope_contains_scope(sdict: ScopeDictType, node: NodeType, other_node: NodeType) -> bool:
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


def _scope_path(sdict: ScopeDictType, scope: NodeType) -> List[NodeType]:
    result = []
    curnode = scope
    while curnode is not None:
        curnode = sdict[scope]
        result.append(curnode)
    return result


def common_parent_scope(sdict: ScopeDictType, scope_a: NodeType, scope_b: NodeType) -> NodeType:
    """
    Finds a common parent scope for both input scopes, or None if the scopes
    are in different connected components.

    :param sdict: Scope parent dictionary.
    :param scope_a: First scope.
    :param scope_b: Second scope.
    :return: Scope node or None for top-level scope.
    """
    if scope_a is scope_b:
        return scope_a

    # Scope B is in scope A
    if scope_contains_scope(sdict, scope_a, scope_b):
        return scope_a
    # Scope A is in scope B
    if scope_contains_scope(sdict, scope_b, scope_a):
        return scope_b

    # Disjoint scopes: prepare two paths and traverse in reversed fashion
    spath_a = _scope_path(sdict, scope_a)
    spath_b = _scope_path(sdict, scope_b)
    common = None
    for spa, spb in reversed(zip(spath_a, spath_b)):
        if spa is spb:
            common = spa
        else:
            break
    return common


def is_in_scope(sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState', node: NodeType,
                schedules: List[dtypes.ScheduleType]) -> bool:
    """ Tests whether a node in an SDFG is contained within a certain set of 
        scope schedules.
        
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in device-level code, False otherwise.
    """
    while sdfg is not None:
        if state is not None and node is not None:
            sdict = state.scope_dict()
            scope = sdict[node]
            while scope is not None:
                if scope.schedule in schedules:
                    return True
                scope = sdict[scope]
        # Traverse up nested SDFGs
        if sdfg.parent is not None:
            parent = sdfg.parent_sdfg
            state = sdfg.parent
            node = sdfg.parent_nsdfg_node
            if node.schedule in schedules:
                return True
        else:
            parent = sdfg.parent
        sdfg = parent
    return False


def is_devicelevel_gpu(sdfg: 'dace.sdfg.SDFG',
                       state: 'dace.sdfg.SDFGState',
                       node: NodeType,
                       with_gpu_default: bool = False) -> bool:
    """ Tests whether a node in an SDFG is contained within GPU device-level code.

        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in device-level code, False otherwise.
    """
    if with_gpu_default:
        schedules = dtypes.GPU_SCHEDULES + [dtypes.ScheduleType.GPU_Default]
    else:
        schedules = dtypes.GPU_SCHEDULES
    return is_in_scope(
        sdfg,
        state,
        node,
        schedules,
    )


def is_devicelevel_gpu_kernel(sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState', node: NodeType) -> bool:
    """ Tests whether a node in an SDFG is contained within an actual GPU kernel.
        The main difference from :func:`is_devicelevel_gpu` is that it returns False for NestedSDFGs that have a GPU
        device-level schedule, but are not within an actual GPU kernel.
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in GPU kernel code, False otherwise.
    """
    is_parent_nested = (sdfg.parent is not None)
    if is_parent_nested:
        return is_devicelevel_gpu(sdfg.parent.parent, sdfg.parent, sdfg.parent_nsdfg_node, with_gpu_default=True)
    else:
        return is_devicelevel_gpu(state.sdfg, state, node, with_gpu_default=True)


def is_devicelevel_fpga(sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState', node: NodeType) -> bool:
    """ Tests whether a node in an SDFG is contained within FPGA device-level
        code.

        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: True if node is in device-level code, False otherwise.
    """
    from dace.sdfg.utils import is_fpga_kernel
    return (is_in_scope(sdfg, state, node, [dtypes.ScheduleType.FPGA_Device])
            or (state is not None and is_fpga_kernel(sdfg, state)))


def devicelevel_block_size(sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState',
                           node: NodeType) -> Tuple[symbolic.SymExpr]:
    """ Returns the current thread-block size if the given node is enclosed in
        a GPU kernel, or None otherwise.
        
        :param sdfg: The SDFG in which the node resides.
        :param state: The SDFG state in which the node resides.
        :param node: The node in question
        :return: A tuple of sizes or None if the node is not in device-level 
                 code.
    """
    from dace.sdfg import nodes as nd
    from dace.sdfg.sdfg import SDFGState

    while sdfg is not None:
        sdict = state.scope_dict()
        scope = sdict[node]
        while scope is not None:
            if scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                return tuple(scope.map.range.size())
            elif scope.schedule == dtypes.ScheduleType.GPU_Device:
                # No thread-block map, use config default
                return tuple(int(s) for s in Config.get('compiler', 'cuda', 'default_block_size').split(','))
            elif scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
                # Dynamic thread-block map, use configured value
                return tuple(int(s) for s in Config.get('compiler', 'cuda', 'dynamic_map_block_size').split(','))

            scope = sdict[scope]
        # Traverse up nested SDFGs
        if sdfg.parent is not None:
            if isinstance(sdfg.parent, SDFGState):
                parent = sdfg.parent.sdfg
            else:
                parent = sdfg.parent
            state, node = next((s, n) for s in parent.nodes() for n in s.nodes()
                               if isinstance(n, nd.NestedSDFG) and n.sdfg.name == sdfg.name)
        else:
            parent = sdfg.parent
        sdfg = parent
    return None
