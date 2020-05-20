import copy
import collections
from typing import Any, Dict, List, Tuple

from dace import dtypes, symbolic
from dace.config import Config
from dace.sdfg import nodes as nd
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import MemletTrackingView

NodeType = 'dace.sdfg.nodes.Node'
EntryNodeType = 'dace.sdfg.nodes.EntryNode'
ExitNodeType = 'dace.sdfg.nodes.ExitNode'
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


class ScopeSubgraphView(SubgraphView, MemletTrackingView):
    """ An extension to SubgraphView that enables the creation of scope
        dictionaries in subgraphs and free symbols. """
    def __init__(self, graph, subgraph_nodes):
        super(ScopeSubgraphView, self).__init__(graph, subgraph_nodes)
        self._clear_scopedict_cache()

    @property
    def parent(self):
        return self._graph.parent

    def _clear_scopedict_cache(self):
        """ Clears the cached results for the scope_dict function.

            For use when the graph mutates (e.g., new edges/nodes, deletions).
        """
        self._scope_dict_toparent_cached = None
        self._scope_dict_tochildren_cached = None

    def scope_dict(self,
                   node_to_children=False,
                   return_ids=False,
                   validate=True):
        """ Returns a dictionary that segments an SDFG state into
            entry-node/exit-node scopes.

            :param node_to_children: If False (default), returns a mapping
                                     of each node to its parent scope
                                     (ScopeEntry) node. If True, returns a
                                     mapping of each parent node to a list of
                                     children nodes.
            :type node_to_children: bool
            :param return_ids: Return node ID numbers instead of node objects.
            :type return_ids: bool
            :param validate: Ensure that the graph is not malformed when
                 computing dictionary.
            :return: The mapping from a node to its parent scope node, or the
                     mapping from a node to a list of children nodes.
            :rtype: dict(Node, Node) or dict(Node, list(Node))
        """
        result = None
        if not node_to_children and self._scope_dict_toparent_cached is not None:
            result = copy.copy(self._scope_dict_toparent_cached)
        elif node_to_children and self._scope_dict_tochildren_cached is not None:
            result = copy.copy(self._scope_dict_tochildren_cached)

        if result is None:
            result = {}
            node_queue = collections.deque(self.source_nodes())
            eq = _scope_dict_inner(self, node_queue, None, node_to_children,
                                   result)

            # Sanity check
            if validate:
                assert len(eq) == 0

            # Cache result
            if node_to_children:
                self._scope_dict_tochildren_cached = result
            else:
                self._scope_dict_toparent_cached = result

            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    def scope_subgraph(self,
                       entry_node,
                       include_entry=True,
                       include_exit=True):
        """ Returns a subgraph that only contains the scope, defined by the
            given entry node.
        """
        return _scope_subgraph(self, entry_node, include_entry, include_exit)

    def top_level_transients(self):
        from dace.sdfg.state import top_level_transients
        return top_level_transients(self)

    def all_transients(self):
        from dace.sdfg.state import all_transients
        return all_transients(self)

    def entry_node(self, node: nd.Node) -> nd.EntryNode:
        """ Returns the entry node that wraps the current node, or None if
            it is top-level in a state. """
        return self.scope_dict()[node]

    def exit_node(self, entry_node: nd.EntryNode) -> nd.ExitNode:
        """ Returns the exit node leaving the context opened by
            the given entry node. """
        node_to_children = self.scope_dict(True)
        return next(v for v in node_to_children[entry_node]
                    if isinstance(v, nd.ExitNode))

    def data_symbols(self):
        """Returns all symbols used in data nodes."""
        from dace.sdfg.sdfg import data_symbols
        return data_symbols(self)

    def scope_symbols(self):
        """Returns all symbols defined by scopes within this state."""
        from dace.sdfg.sdfg import scope_symbols
        return scope_symbols(self)

    def interstate_symbols(self):
        """Returns all symbols (assigned, used) in interstate edges in nested
           SDFGs within this subgraph."""
        from dace.sdfg.sdfg import interstate_symbols
        return interstate_symbols(self)

    def undefined_symbols(self, sdfg, include_scalar_data):
        from dace.sdfg.sdfg import undefined_symbols
        return undefined_symbols(sdfg, self, include_scalar_data)

    def all_nodes_recursive(self):
        for node in self.nodes():
            yield node, self
            if isinstance(node, nd.NestedSDFG):
                yield from node.sdfg.all_nodes_recursive()

    def all_edges_recursive(self):
        for e in self.edges():
            yield e, self
        for node in self.nodes():
            if isinstance(node, nd.NestedSDFG):
                yield from node.sdfg.all_edges_recursive()


def _scope_subgraph(graph, entry_node, include_entry, include_exit):
    if not isinstance(entry_node, nd.EntryNode):
        raise TypeError("Received {}: should be dace.nodes.EntryNode".format(
            type(entry_node).__name__))
    node_to_children = graph.scope_dict(True)
    if include_exit:
        children_nodes = set(node_to_children[entry_node])
    else:
        children_nodes = set(n for n in node_to_children[entry_node]
                             if not isinstance(n, nd.ExitNode))
    map_nodes = [
        node for node in children_nodes if isinstance(node, nd.EntryNode)
    ]
    while len(map_nodes) > 0:
        next_map_nodes = []
        # Traverse children map nodes
        for map_node in map_nodes:
            # Get child map subgraph (1 level)
            more_nodes = set(node_to_children[map_node])
            # Unionize children_nodes with new nodes
            children_nodes |= more_nodes
            # Add nodes of the next level to next_map_nodes
            next_map_nodes.extend([
                node for node in more_nodes if isinstance(node, nd.EntryNode)
            ])
        map_nodes = next_map_nodes

    if include_entry:
        children_nodes.add(entry_node)

    # Preserve order of nodes
    return ScopeSubgraphView(graph,
                             [n for n in graph.nodes() if n in children_nodes])


def _scope_dict_inner(graph, node_queue, current_scope, node_to_children,
                      result):
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
            node_queue.extend(
                _scope_dict_inner(graph, collections.deque(successors), node,
                                  node_to_children, result))
        # If this is an Exit Node, we push the successors to the external
        # queue
        elif isinstance(node, nd.ExitNode):
            external_queue.extend(successors)
        # Otherwise, it is a plain node, and we push its successors to the
        # same queue
        else:
            node_queue.extend(successors)

    return external_queue


def _scope_dict_to_ids(state: 'dace.sdfg.SDFGState',
                       scope_dict: Dict[Any, List[Any]]):
    """ Return a JSON-serializable dictionary of a scope dictionary,
        using integral node IDs instead of object references. """
    def node_id_or_none(node):
        if node is None: return -1
        return state.node_id(node)

    return {
        node_id_or_none(k): [node_id_or_none(vi) for vi in v]
        for k, v in scope_dict.items()
    }


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
    from dace.sdfg import nodes as nd
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
    from dace.sdfg import nodes as nd

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