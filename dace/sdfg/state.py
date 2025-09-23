# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes of a single SDFG state and dataflow subgraphs. """

import ast
import abc
import collections
import copy
import inspect
import itertools
import warnings
import sympy
from typing import (TYPE_CHECKING, Any, AnyStr, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type,
                    Union, overload)

import dace
from dace.frontend.python import astutils
from dace.sdfg.replace import replace_in_codeblock
import dace.serialize
from dace import data as dt
from dace import dtypes
from dace import memlet as mm
from dace import serialize
from dace import subsets as sbs
from dace import symbolic
from dace.properties import (CodeBlock, DebugInfoProperty, DictProperty, EnumProperty, Property, SubsetProperty,
                             SymbolicProperty, CodeProperty, make_properties)
from dace.sdfg import nodes as nd
from dace.sdfg.graph import (MultiConnectorEdge, NodeNotFoundError, OrderedMultiDiConnectorGraph, SubgraphView,
                             OrderedDiGraph, Edge, generate_element_id)
from dace.sdfg.propagation import propagate_memlet
from dace.sdfg.validation import validate_state
from dace.subsets import Range, Subset

if TYPE_CHECKING:
    import dace.sdfg.scope
    from dace.sdfg import SDFG

NodeT = Union[nd.Node, 'ControlFlowBlock']
EdgeT = Union[MultiConnectorEdge[mm.Memlet], Edge['dace.sdfg.InterstateEdge']]
GraphT = Union['ControlFlowRegion', 'SDFGState']


def _getdebuginfo(old_dinfo=None) -> dtypes.DebugInfo:
    """ Returns a DebugInfo object for the position that called this function.

        :param old_dinfo: Another DebugInfo object that will override the
                          return value of this function
        :return: DebugInfo containing line number and calling file.
    """
    if old_dinfo is not None:
        return old_dinfo

    caller = inspect.getframeinfo(inspect.stack()[2][0], context=0)
    return dtypes.DebugInfo(caller.lineno, 0, caller.lineno, 0, caller.filename)


def _make_iterators(ndrange):
    # Input can either be a dictionary or a list of pairs
    if isinstance(ndrange, list):
        params = [k for k, _ in ndrange]
        ndrange = {k: v for k, v in ndrange}
    else:
        params = list(ndrange.keys())

    # Parse each dimension separately
    ranges = []
    for p in params:
        prange: Union[str, sbs.Subset, Tuple[symbolic.SymbolicType]] = ndrange[p]
        if isinstance(prange, sbs.Subset):
            rng = prange.ndrange()[0]
        elif isinstance(prange, tuple):
            rng = prange
        else:
            rng = SubsetProperty.from_string(prange)[0]
        ranges.append(rng)
    map_range = sbs.Range(ranges)

    return params, map_range


class BlockGraphView(object):
    """
    Read-only view interface of an SDFG control flow block, containing methods for memlet tracking, traversal, subgraph
    creation, queries, and replacements. ``ControlFlowBlock`` and ``StateSubgraphView`` inherit from this class to share
    methods.
    """

    ###################################################################
    # Typing overrides

    @overload
    def nodes(self) -> List[NodeT]:
        ...

    @overload
    def edges(self) -> List[EdgeT]:
        ...

    @overload
    def in_degree(self, node: NodeT) -> int:
        ...

    @overload
    def out_degree(self, node: NodeT) -> int:
        ...

    @property
    def sdfg(self) -> 'SDFG':
        ...

    ###################################################################
    # Traversal methods

    @abc.abstractmethod
    def all_nodes_recursive(self,
                            predicate: Optional[Callable[[NodeT, GraphT],
                                                         bool]] = None) -> Iterator[Tuple[NodeT, GraphT]]:
        """
        Iterate over all nodes in this graph or subgraph.
        This includes control flow blocks, nodes in those blocks, and recursive control flow blocks and nodes within
        nested SDFGs. It returns tuples of the form (node, parent), where the node is either a dataflow node, in which
        case the parent is an SDFG state, or a control flow block, in which case the parent is a control flow graph
        (i.e., an SDFG or a scope block).

        :param predicate: An optional predicate function that decides on whether the traversal should recurse or not.
        If the predicate returns False, traversal is not recursed any further into the graph found under NodeT for
        a given [NodeT, GraphT] pair.
        """
        return []

    @abc.abstractmethod
    def all_edges_recursive(self) -> Iterator[Tuple[EdgeT, GraphT]]:
        """
        Iterate over all edges in this graph or subgraph.
        This includes dataflow edges, inter-state edges, and recursive edges within nested SDFGs. It returns tuples of
        the form (edge, parent), where the edge is either a dataflow edge, in which case the parent is an SDFG state, or
        an inter-stte edge, in which case the parent is a control flow graph (i.e., an SDFG or a scope block).
        """
        return []

    @abc.abstractmethod
    def data_nodes(self) -> List[nd.AccessNode]:
        """
        Returns all data nodes (i.e., AccessNodes, arrays) present in this graph or subgraph.
        Note: This does not recurse into nested SDFGs.
        """
        return []

    @abc.abstractmethod
    def entry_node(self, node: nd.Node) -> Optional[nd.EntryNode]:
        """ Returns the entry node that wraps the current node, or None if it is top-level in a state. """
        return None

    @abc.abstractmethod
    def exit_node(self, entry_node: nd.EntryNode) -> Optional[nd.ExitNode]:
        """ Returns the exit node leaving the context opened by the given entry node. """
        raise None

    ###################################################################
    # Memlet-tracking methods

    @abc.abstractmethod
    def memlet_path(self, edge: MultiConnectorEdge[mm.Memlet]) -> List[MultiConnectorEdge[mm.Memlet]]:
        """
        Given one edge, returns a list of edges representing a path between its source and sink nodes.
        Used for memlet tracking.

        :note: Behavior is undefined when there is more than one path involving this edge.
        :param edge: An edge within a state (memlet).
        :return: A list of edges from a source node to a destination node.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def memlet_tree(self, edge: MultiConnectorEdge) -> mm.MemletTree:
        """
        Given one edge, returns a tree of edges between its node source(s) and sink(s).
        Used for memlet tracking.

        :param edge: An edge within a state (memlet).
        :return: A tree of edges whose root is the source/sink node (depending on direction) and associated children
                 edges.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def in_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        """
        Returns a generator over edges entering the given connector of the given node.

        :param node: Destination node of edges.
        :param connector: Destination connector of edges.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def out_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        """
        Returns a generator over edges exiting the given connector of the given node.

        :param node: Source node of edges.
        :param connector: Source connector of edges.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        """
        Returns a generator over edges entering or exiting the given connector of the given node.

        :param node: Source/destination node of edges.
        :param connector: Source/destination connector of edges.
        """
        raise NotImplementedError()

    ###################################################################
    # Query, subgraph, and replacement methods

    @abc.abstractmethod
    def used_symbols(self,
                     all_symbols: bool,
                     keep_defined_in_mapping: bool = False,
                     with_contents: bool = True) -> Set[str]:
        """
        Returns a set of symbol names that are used in the graph.

        :param all_symbols: If False, only returns symbols that are needed as arguments (only used in generated code).
        :param keep_defined_in_mapping: If True, symbols defined in inter-state edges that are in the symbol mapping
                                        will be removed from the set of defined symbols.
        :param with_contents: Compute the symbols used including the ones used by the contents of the graph. If set to
                              False, only symbols used on the BlockGraphView itself are returned. The latter may
                              include symbols used in the conditions of conditional blocks, loops, etc. Defaults to
                              True.
        """
        return set()

    @property
    def free_symbols(self) -> Set[str]:
        """
        Returns a set of symbol names that are used, but not defined, in this graph view.
        In the case of an SDFG, this property is used to determine the symbolic parameters of the SDFG and
        verify that ``SDFG.symbols`` is complete.

        :note: Assumes that the graph is valid (i.e., without undefined or overlapping symbols).
        """
        return self.used_symbols(all_symbols=True)

    @abc.abstractmethod
    def read_and_write_sets(self) -> Tuple[Set[AnyStr], Set[AnyStr]]:
        """
        Determines what data is read and written in this graph.
        Does not include reads to subsets of containers that have previously been written within the same state.

        :return: A two-tuple of sets of things denoting ({data read}, {data written}).
        """
        return set(), set()

    @abc.abstractmethod
    def unordered_arglist(self,
                          defined_syms=None,
                          shared_transients=None) -> Tuple[Dict[str, dt.Data], Dict[str, dt.Data]]:
        return {}, {}

    def arglist(self, defined_syms=None, shared_transients=None) -> Dict[str, dt.Data]:
        """
        Returns an ordered dictionary of arguments (names and types) required to invoke this subgraph.

        The arguments differ from SDFG.arglist, but follow the same order,
        namely: <sorted data arguments>, <sorted scalar arguments>.

        Data arguments contain:
            * All used non-transient data containers in the subgraph
            * All used transient data containers that were allocated outside.
              This includes data from memlets, transients shared across multiple states, and transients that could not
              be allocated within the subgraph (due to their ``AllocationLifetime`` or according to the
              ``dtypes.can_allocate`` function).

        Scalar arguments contain:
            * Free symbols in this state/subgraph.
            * All transient and non-transient scalar data containers used in this subgraph.

        This structure will create a sorted list of pointers followed by a sorted list of PoDs and structs.

        :return: An ordered dictionary of (name, data descriptor type) of all the arguments, sorted as defined here.
        """
        data_args, scalar_args = self.unordered_arglist(defined_syms, shared_transients)

        # Fill up ordered dictionary
        result = collections.OrderedDict()
        for k, v in itertools.chain(sorted(data_args.items()), sorted(scalar_args.items())):
            result[k] = v

        return result

    def signature_arglist(self, with_types=True, for_call=False):
        """ Returns a list of arguments necessary to call this state or subgraph, formatted as a list of C definitions.

            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when calling the SDFG.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        return [v.as_arg(name=k, with_types=with_types, for_call=for_call) for k, v in self.arglist().items()]

    @abc.abstractmethod
    def top_level_transients(self) -> Set[str]:
        """Iterate over top-level transients of this graph."""
        return set()

    @abc.abstractmethod
    def all_transients(self) -> List[str]:
        """Iterate over all transients in this graph."""
        return []

    @abc.abstractmethod
    def replace(self, name: str, new_name: str):
        """
        Finds and replaces all occurrences of a symbol or array in this graph.

        :param name: Name to find.
        :param new_name: Name to replace.
        """
        pass

    @abc.abstractmethod
    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
        """
        Finds and replaces all occurrences of a set of symbols or arrays in this graph.

        :param repl: Mapping from names to replacements.
        :param symrepl: Optional symbolic version of ``repl``.
        """
        pass


@make_properties
class DataflowGraphView(BlockGraphView, abc.ABC):

    def __init__(self, *args, **kwargs):
        # Ensure that the cache for the scope related function exists.
        self._clear_scopedict_cache()

    ###################################################################
    # Typing overrides

    @overload
    def nodes(self) -> List[nd.Node]:
        ...

    @overload
    def edges(self) -> List[MultiConnectorEdge[mm.Memlet]]:
        ...

    ###################################################################
    # Traversal methods

    def all_nodes_recursive(self, predicate=None) -> Iterator[Tuple[NodeT, GraphT]]:
        for node in self.nodes():
            yield node, self
            if isinstance(node, nd.NestedSDFG) and node.sdfg:
                if predicate is None or predicate(node, self):
                    yield from node.sdfg.all_nodes_recursive(predicate)

    def all_edges_recursive(self) -> Iterator[Tuple[EdgeT, GraphT]]:
        for e in self.edges():
            yield e, self
        for node in self.nodes():
            if isinstance(node, nd.NestedSDFG):
                yield from node.sdfg.all_edges_recursive()

    def data_nodes(self) -> List[nd.AccessNode]:
        """ Returns all data_nodes (arrays) present in this state. """
        return [n for n in self.nodes() if isinstance(n, nd.AccessNode)]

    def entry_node(self, node: nd.Node) -> Optional[nd.EntryNode]:
        """ Returns the entry node that wraps the current node, or None if
            it is top-level in a state. """
        return self.scope_dict()[node]

    def exit_node(self, entry_node: nd.EntryNode) -> Optional[nd.ExitNode]:
        """ Returns the exit node leaving the context opened by
            the given entry node. """
        node_to_children = self.scope_children()
        return next(v for v in node_to_children[entry_node] if isinstance(v, nd.ExitNode))

    ###################################################################
    # Memlet-tracking methods

    def memlet_path(self, edge: MultiConnectorEdge[mm.Memlet]) -> List[MultiConnectorEdge[mm.Memlet]]:
        """ Given one edge, returns a list of edges representing a path
            between its source and sink nodes. Used for memlet tracking.

            :note: Behavior is undefined when there is more than one path
                   involving this edge.
            :param edge: An edge within this state.
            :return: A list of edges from a source node to a destination node.
            """
        result = [edge]

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # If empty memlet, return itself as the path
        if (edge.src_conn is None and edge.dst_conn is None and edge.data.is_empty()):
            return result

        # Prepend incoming edges until reaching the source node
        curedge = edge
        visited = set()
        while not isinstance(curedge.src, (nd.CodeNode, nd.AccessNode)):
            visited.add(curedge)
            # Trace through scopes using OUT_# -> IN_#
            if isinstance(curedge.src, (nd.EntryNode, nd.ExitNode)):
                if curedge.src_conn is None:
                    raise ValueError("Source connector cannot be None for {}".format(curedge.src))
                assert curedge.src_conn.startswith("OUT_")
                next_edge = next(e for e in state.in_edges(curedge.src) if e.dst_conn == "IN_" + curedge.src_conn[4:])
                result.insert(0, next_edge)
                curedge = next_edge
                if curedge in visited:
                    raise ValueError('Cycle encountered while reading memlet path')

        # Append outgoing edges until reaching the sink node
        curedge = edge
        visited.clear()
        while not isinstance(curedge.dst, (nd.CodeNode, nd.AccessNode)):
            visited.add(curedge)
            # Trace through scope entry using IN_# -> OUT_#
            if isinstance(curedge.dst, (nd.EntryNode, nd.ExitNode)):
                if curedge.dst_conn is None:
                    raise ValueError("Destination connector cannot be None for {}".format(curedge.dst))
                if not curedge.dst_conn.startswith("IN_"):  # Map variable
                    break
                next_edge = next(e for e in state.out_edges(curedge.dst) if e.src_conn == "OUT_" + curedge.dst_conn[3:])
                result.append(next_edge)
                curedge = next_edge
                if curedge in visited:
                    raise ValueError('Cycle encountered while reading memlet path')

        return result

    def memlet_tree(self, edge: MultiConnectorEdge) -> mm.MemletTree:
        propagate_forward = False
        propagate_backward = False
        if ((isinstance(edge.src, nd.EntryNode) and edge.src_conn is not None) or
            (isinstance(edge.dst, nd.EntryNode) and edge.dst_conn is not None and edge.dst_conn.startswith('IN_'))):
            propagate_forward = True
        if ((isinstance(edge.src, nd.ExitNode) and edge.src_conn is not None)
                or (isinstance(edge.dst, nd.ExitNode) and edge.dst_conn is not None)):
            propagate_backward = True

        # If either both are False (no scopes involved) or both are True
        # (invalid SDFG), we return only the current edge as a degenerate tree
        if propagate_forward == propagate_backward:
            return mm.MemletTree(edge)

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # Find tree root
        curedge = edge
        visited = set()
        if propagate_forward:
            while (isinstance(curedge.src, nd.EntryNode) and curedge.src_conn is not None):
                visited.add(curedge)
                assert curedge.src_conn.startswith('OUT_')
                cname = curedge.src_conn[4:]
                curedge = next(e for e in state.in_edges(curedge.src) if e.dst_conn == 'IN_%s' % cname)
                if curedge in visited:
                    raise ValueError('Cycle encountered while reading memlet path')
        elif propagate_backward:
            while (isinstance(curedge.dst, nd.ExitNode) and curedge.dst_conn is not None):
                visited.add(curedge)
                assert curedge.dst_conn.startswith('IN_')
                cname = curedge.dst_conn[3:]
                curedge = next(e for e in state.out_edges(curedge.dst) if e.src_conn == 'OUT_%s' % cname)
                if curedge in visited:
                    raise ValueError('Cycle encountered while reading memlet path')
        tree_root = mm.MemletTree(curedge, downwards=propagate_forward)

        # Collect children (recursively)
        def add_children(treenode):
            if propagate_forward:
                if not (isinstance(treenode.edge.dst, nd.EntryNode) and treenode.edge.dst_conn
                        and treenode.edge.dst_conn.startswith('IN_')):
                    return
                conn = treenode.edge.dst_conn[3:]
                treenode.children = [
                    mm.MemletTree(e, downwards=True, parent=treenode) for e in state.out_edges(treenode.edge.dst)
                    if e.src_conn == 'OUT_%s' % conn
                ]
            elif propagate_backward:
                if (not isinstance(treenode.edge.src, nd.ExitNode) or treenode.edge.src_conn is None):
                    return
                conn = treenode.edge.src_conn[4:]
                treenode.children = [
                    mm.MemletTree(e, downwards=False, parent=treenode) for e in state.in_edges(treenode.edge.src)
                    if e.dst_conn == 'IN_%s' % conn
                ]

            for child in treenode.children:
                add_children(child)

        # Start from root node (obtained from above parent traversal)
        add_children(tree_root)

        # Find edge in tree
        def traverse(node):
            if node.edge == edge:
                return node
            for child in node.children:
                res = traverse(child)
                if res is not None:
                    return res
            return None

        # Return node that corresponds to current edge
        return traverse(tree_root)

    def in_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        return (e for e in self.in_edges(node) if e.dst_conn == connector)

    def out_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        return (e for e in self.out_edges(node) if e.src_conn == connector)

    def edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        return itertools.chain(self.in_edges_by_connector(node, connector),
                               self.out_edges_by_connector(node, connector))

    ###################################################################
    # Scope-related methods

    def _clear_scopedict_cache(self):
        """
        Clears the cached results for the scope_dict function.
        For use when the graph mutates (e.g., new edges/nodes, deletions).
        """
        self._scope_dict_toparent_cached = None
        self._scope_dict_tochildren_cached = None
        self._scope_tree_cached = None
        self._scope_leaves_cached = None

    def scope_tree(self) -> Dict[Union[None, nd.Node], 'dace.sdfg.scope.ScopeTree']:
        """Get the scope trees.

        :note: That the result is cached inside the state, thus it is not allowed to modify the returned value.
            However, the `ScopeTree` can be safely shallow copied.
        """
        from dace.sdfg.scope import ScopeTree

        if self._scope_tree_cached is not None:
            return copy.copy(self._scope_tree_cached)

        sdp = self.scope_dict()
        sdc = self.scope_children()

        result = {}

        # Get scopes
        for node, scopenodes in sdc.items():
            if node is None:
                exit_node = None
            else:
                exit_node = next(v for v in scopenodes if isinstance(v, nd.ExitNode))
            scope = ScopeTree(node, exit_node)
            result[node] = scope

        # Scope parents and children
        for node, scope in result.items():
            if node is not None:
                scope.parent = result[sdp[node]]
            scope.children = [result[n] for n in sdc[node] if isinstance(n, nd.EntryNode)]

        self._scope_tree_cached = result

        return copy.copy(self._scope_tree_cached)

    def scope_leaves(self) -> List['dace.sdfg.scope.ScopeTree']:
        """Return the list of scope leaves.

        :note: That the result is cached inside the state, thus it is not allowed to modify the returned value.
            However, the `ScopeTree` can be safely shallow copied.
        """
        if self._scope_leaves_cached is not None:
            return copy.copy(self._scope_leaves_cached)

        st = self.scope_tree()
        self._scope_leaves_cached = [scope for scope in st.values() if len(scope.children) == 0]
        return copy.copy(self._scope_leaves_cached)

    def scope_dict(self,
                   return_ids: bool = False,
                   validate: bool = True) -> Dict[nd.Node, Union['SDFGState', nd.Node, None]]:
        """
        Return the scope dict, i.e. map every node inside the state to its enclosing scope or `None` if at global scope.

        :note: The result is cached inside the state, but the returned `dict` is only shallow copied.
        """
        from dace.sdfg.scope import _scope_dict_inner, _scope_dict_to_ids

        result = copy.copy(self._scope_dict_toparent_cached)

        if result is None:
            result = {}
            node_queue = collections.deque(self.source_nodes())
            eq = _scope_dict_inner(self, node_queue, None, False, result)

            # Sanity checks
            if validate and len(eq) != 0:
                cycles = list(self.find_cycles())
                if cycles:
                    raise ValueError('Found cycles in state %s: %s' % (self.label, cycles))
                raise RuntimeError("Leftover nodes in queue: {}".format(eq))

            if validate and len(result) != self.number_of_nodes():
                cycles = list(self.find_cycles())
                if cycles:
                    raise ValueError('Found cycles in state %s: %s' % (self.label, cycles))
                leftover_nodes = set(self.nodes()) - result.keys()
                raise RuntimeError("Some nodes were not processed: {}".format(leftover_nodes))

            # Cache result
            self._scope_dict_toparent_cached = result
            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    def scope_children(self,
                       return_ids: bool = False,
                       validate: bool = True) -> Dict[Union[nd.Node, 'SDFGState', None], List[nd.Node]]:
        """For every scope node returns the list of nodes that are inside that scope.

        The global scope is denoted by `None`. It is essentially the inversion of `scope_dict`.

        :note: The result is cached inside the state thus it is not allowed to modify the returned values.
        """
        from dace.sdfg.scope import _scope_dict_inner, _scope_dict_to_ids

        result = None
        if self._scope_dict_tochildren_cached is not None:
            # NOTE: Why do we shallow copy the `dict` but not the lists?
            result = copy.copy(self._scope_dict_tochildren_cached)

        if result is None:
            result = {}
            node_queue = collections.deque(self.source_nodes())
            eq = _scope_dict_inner(self, node_queue, None, True, result)

            # Sanity checks
            if validate and len(eq) != 0:
                cycles = list(self.find_cycles())
                if cycles:
                    raise ValueError('Found cycles in state %s: %s' % (self.label, cycles))
                raise RuntimeError("Leftover nodes in queue: {}".format(eq))

            entry_nodes = set(n for n in self.nodes() if isinstance(n, nd.EntryNode)) | {None}
            if (validate and len(result) != len(entry_nodes)):
                cycles = list(self.find_cycles())
                if cycles:
                    raise ValueError('Found cycles in state %s: %s' % (self.label, cycles))
                raise RuntimeError("Some nodes were not processed: {}".format(entry_nodes - result.keys()))

            # Cache result
            self._scope_dict_tochildren_cached = result
            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    ###################################################################
    # Query, subgraph, and replacement methods

    def is_leaf_memlet(self, e):
        if isinstance(e.src, nd.ExitNode) and e.src_conn and e.src_conn.startswith('OUT_'):
            return False
        if isinstance(e.dst, nd.EntryNode) and e.dst_conn and e.dst_conn.startswith('IN_'):
            return False
        return True

    def used_symbols(self,
                     all_symbols: bool,
                     keep_defined_in_mapping: bool = False,
                     with_contents: bool = True) -> Set[str]:
        if not with_contents:
            return set()

        state = self.graph if isinstance(self, SubgraphView) else self
        sdfg = state.sdfg
        new_symbols = set()
        freesyms = set()

        # Free symbols from nodes
        for n in self.nodes():
            if isinstance(n, nd.EntryNode):
                new_symbols |= set(n.new_symbols(sdfg, self, {}).keys())
            elif isinstance(n, nd.AccessNode):
                # Add data descriptor symbols
                freesyms |= set(map(str, n.desc(sdfg).used_symbols(all_symbols)))
            elif isinstance(n, nd.Tasklet):
                if n.language == dtypes.Language.Python:
                    # Consider callbacks defined as symbols as free
                    for stmt in n.code.code:
                        for astnode in ast.walk(stmt):
                            if (isinstance(astnode, ast.Call) and isinstance(astnode.func, ast.Name)
                                    and astnode.func.id in sdfg.symbols):
                                freesyms.add(astnode.func.id)
                else:
                    # Find all string tokens and filter them to sdfg.symbols, while ignoring connectors
                    codesyms = symbolic.symbols_in_code(
                        n.code.as_string,
                        potential_symbols=sdfg.symbols.keys(),
                        symbols_to_ignore=(n.in_connectors.keys() | n.out_connectors.keys() | n.ignored_symbols),
                    )
                    freesyms |= codesyms
                    continue

            if hasattr(n, 'used_symbols'):
                freesyms |= n.used_symbols(all_symbols)
            else:
                freesyms |= n.free_symbols

        # Free symbols from memlets
        for e in self.edges():
            # If used for code generation, only consider memlet tree leaves
            if not all_symbols and not self.is_leaf_memlet(e):
                continue

            freesyms |= e.data.used_symbols(all_symbols, e)

        # Do not consider SDFG constants as symbols
        new_symbols.update(set(sdfg.constants.keys()))
        return freesyms - new_symbols

    def defined_symbols(self) -> Dict[str, dt.Data]:
        """
        Returns a dictionary that maps currently-defined symbols in this SDFG
        state or subgraph to their types.
        """
        state = self.graph if isinstance(self, SubgraphView) else self
        sdfg = state.sdfg

        # Start with SDFG global symbols
        defined_syms = {k: v for k, v in sdfg.symbols.items()}

        def update_if_not_none(dic, update):
            update = {k: v for k, v in update.items() if v is not None}
            dic.update(update)

        # Add data-descriptor free symbols
        for desc in sdfg.arrays.values():
            for sym in desc.free_symbols:
                if sym.dtype is not None:
                    defined_syms[str(sym)] = sym.dtype

        # Add inter-state symbols
        if isinstance(sdfg.start_block, AbstractControlFlowRegion):
            update_if_not_none(defined_syms, sdfg.start_block.new_symbols(defined_syms))
        for edge in sdfg.all_interstate_edges():
            update_if_not_none(defined_syms, edge.data.new_symbols(sdfg, defined_syms))
            if isinstance(edge.dst, AbstractControlFlowRegion):
                update_if_not_none(defined_syms, edge.dst.new_symbols(defined_syms))

        # Add scope symbols all the way to the subgraph
        sdict = state.scope_dict()
        scope_nodes = []
        for source_node in self.source_nodes():
            curnode = source_node
            while sdict[curnode] is not None:
                curnode = sdict[curnode]
                scope_nodes.append(curnode)

        for snode in dtypes.deduplicate(list(reversed(scope_nodes))):
            update_if_not_none(defined_syms, snode.new_symbols(sdfg, state, defined_syms))

        return defined_syms

    def _read_and_write_sets(self) -> Tuple[Dict[AnyStr, List[Subset]], Dict[AnyStr, List[Subset]]]:
        """
        Determines what data is read and written in this subgraph, returning
        dictionaries from data containers to all subsets that are read/written.
        """
        from dace.sdfg import utils  # Avoid cyclic import

        # Ensures that the `{src,dst}_subset` are properly set.
        #  TODO: find where the problems are
        for edge in self.edges():
            edge.data.try_initialize(self.sdfg, self, edge)

        read_set = collections.defaultdict(list)
        write_set = collections.defaultdict(list)

        # NOTE: In a previous version a _single_ read (i.e. leaving Memlet) that was
        #   fully covered by a single write (i.e. an incoming Memlet) was removed from
        #   the read set and only the write survived. However, this was never fully
        #   implemented nor correctly implemented and caused problems.
        #   So this filtering was removed.

        for subgraph in utils.concurrent_subgraphs(self):
            subgraph_read_set = collections.defaultdict(list)  # read and write set of this subgraph.
            subgraph_write_set = collections.defaultdict(list)
            for n in utils.dfs_topological_sort(subgraph, sources=subgraph.source_nodes()):
                if not isinstance(n, nd.AccessNode):
                    # Read and writes can only be done through access nodes,
                    #  so ignore every other node.
                    continue

                # Get a list of all incoming (writes) and outgoing (reads) edges of the
                #  access node, ignore all empty memlets as they do not carry any data.
                in_edges = [in_edge for in_edge in subgraph.in_edges(n) if not in_edge.data.is_empty()]
                out_edges = [out_edge for out_edge in subgraph.out_edges(n) if not out_edge.data.is_empty()]

                # Extract the subsets that describes where we read and write the data
                #  and store them for the later filtering.
                # NOTE: In certain cases the corresponding subset might be None, in this case
                #   we assume that the whole array is written, which is the default behaviour.
                ac_desc = n.desc(self.sdfg)
                in_subsets = {
                    in_edge:
                    (sbs.Range.from_array(ac_desc) if in_edge.data.dst_subset is None else in_edge.data.dst_subset)
                    for in_edge in in_edges
                }
                out_subsets = {
                    out_edge:
                    (sbs.Range.from_array(ac_desc) if out_edge.data.src_subset is None else out_edge.data.src_subset)
                    for out_edge in out_edges
                }

                # Update the read and write sets of the subgraph.
                if in_edges:
                    subgraph_write_set[n.data].extend(in_subsets.values())
                if out_edges:
                    subgraph_read_set[n.data].extend(out_subsets[out_edge] for out_edge in out_edges)

            # Add the subgraph's read and write set to the final ones.
            for data, accesses in subgraph_read_set.items():
                read_set[data] += accesses
            for data, accesses in subgraph_write_set.items():
                write_set[data] += accesses

        return copy.deepcopy((read_set, write_set))

    def read_and_write_sets(self) -> Tuple[Set[AnyStr], Set[AnyStr]]:
        """
        Determines what data is read and written in this subgraph.

        :return: A two-tuple of sets of things denoting
                 ({data read}, {data written}).
        """
        read_set, write_set = self._read_and_write_sets()
        return set(read_set.keys()), set(write_set.keys())

    def unordered_arglist(self,
                          defined_syms=None,
                          shared_transients=None) -> Tuple[Dict[str, dt.Data], Dict[str, dt.Data]]:
        sdfg: 'SDFG' = self.sdfg
        shared_transients = shared_transients or sdfg.shared_transients()
        sdict = self.scope_dict()

        data_args = {}
        scalar_args = {}

        # Gather data descriptors from nodes
        descs = {}
        descs_with_nodes = {}
        scalars_with_nodes = set()
        for node in self.nodes():
            if isinstance(node, nd.AccessNode):
                descs[node.data] = node.desc(sdfg)
                # NOTE: In case of multiple nodes of the same data this will
                #   override previously found nodes.
                descs_with_nodes[node.data] = node
                if isinstance(node.desc(sdfg), dt.Scalar):
                    scalars_with_nodes.add(node.data)

        # If a subgraph, and a node appears outside the subgraph as well,
        # it is externally allocated
        if isinstance(self, SubgraphView):
            outer_nodes = set(self.graph.nodes()) - set(self.nodes())
            for node in outer_nodes:
                if isinstance(node, nd.AccessNode) and node.data in descs:
                    desc = descs[node.data]
                    if isinstance(desc, dt.Scalar):
                        scalar_args[node.data] = desc
                    else:
                        data_args[node.data] = desc

        # Add data arguments from memlets, if do not appear in any of the nodes (i.e., originate externally)
        #  TODO: Investigate is scanning the adjacent edges of the input and output connectors is better.
        for edge in self.edges():
            if edge.data.is_empty():
                continue

            elif edge.data.data not in descs:
                # The edge reads data from the outside, and the Memlet is directly indicating what is read.
                if (isinstance(edge.src, nd.CodeNode) and isinstance(edge.dst, nd.CodeNode)):
                    continue  # Ignore code->code edges.
                additional_descs = {edge.data.data: sdfg.arrays[edge.data.data]}

            elif isinstance(edge.dst, (nd.AccessNode, nd.CodeNode)) and isinstance(edge.src, nd.EntryNode):
                # Special case from the above; An AccessNode reads data from the Outside, but
                #  the Memlet references the data on the inside. Thus we have to follow the data
                #  to where it originates from.
                # NOTE: We have to use a memlet path, because we have to go "against the flow"
                #   Furthermore, in a valid SDFG the data will only come from one source anyway.
                top_source_edge = self.graph.memlet_path(edge)[0]
                if not isinstance(top_source_edge.src, nd.AccessNode):
                    continue
                additional_descs = ({
                    top_source_edge.src.data: top_source_edge.src.desc(sdfg)
                } if top_source_edge.src.data not in descs else {})

            elif isinstance(edge.dst, nd.ExitNode) and isinstance(edge.src, (nd.AccessNode, nd.CodeNode)):
                # Same case as above, but for outgoing Memlets.
                # NOTE: We have to use a memlet tree here, because the data could potentially
                #   go to multiple sources. We have to do it this way, because if we would call
                #   `memlet_tree()` here, then we would just get the edge back.
                additional_descs = {}
                connector_to_look = "OUT_" + edge.dst_conn[3:]
                for oedge in self.graph.out_edges_by_connector(edge.dst, connector_to_look):
                    if ((not oedge.data.is_empty()) and (oedge.data.data not in descs)
                            and (oedge.data.data not in additional_descs)):
                        additional_descs[oedge.data.data] = sdfg.arrays[oedge.data.data]

            else:
                # Case is ignored.
                continue

            # Now processing the list of newly found data.
            for aname, additional_desc in additional_descs.items():
                if isinstance(additional_desc, dt.Scalar):
                    scalar_args[aname] = additional_desc
                else:
                    data_args[aname] = additional_desc

        # Loop over locally-used data descriptors
        for name, desc in descs.items():
            if name in data_args or name in scalar_args:
                continue
            # If scalar, always add if there are no scalar nodes
            if isinstance(desc, dt.Scalar) and name not in scalars_with_nodes:
                scalar_args[name] = desc
            # If array/stream is not transient, then it is external
            elif not desc.transient:
                data_args[name] = desc
            # Check for shared transients
            elif name in shared_transients:
                data_args[name] = desc
            # Check allocation lifetime for external transients:
            #   1. If a full state, Global, SDFG, and Persistent
            elif (not isinstance(self, SubgraphView)
                  and desc.lifetime not in (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.State)):
                data_args[name] = desc
            #   2. If a subgraph, State also applies
            elif isinstance(self, SubgraphView):
                if (desc.lifetime != dtypes.AllocationLifetime.Scope):
                    data_args[name] = desc
                # Check for allocation constraints that would
                # enforce array to be allocated outside subgraph
                elif desc.lifetime == dtypes.AllocationLifetime.Scope:
                    curnode = sdict[descs_with_nodes[name]]
                    while curnode is not None:
                        if dtypes.can_allocate(desc.storage, curnode.schedule):
                            break
                        curnode = sdict[curnode]
                    else:
                        # If no internal scope can allocate node,
                        # mark as external
                        data_args[name] = desc
        # End of data descriptor loop

        # Add scalar arguments from free symbols
        defined_syms = defined_syms or self.defined_symbols()
        scalar_args.update({
            k: dt.Scalar(defined_syms[k]) if k in defined_syms else sdfg.arrays[k]
            for k in self.used_symbols(all_symbols=False)
            if not k.startswith('__dace') and k not in sdfg.constants and (k in defined_syms or k in sdfg.arrays)
        })

        # Add scalar arguments from free symbols of data descriptors
        for arg in data_args.values():
            scalar_args.update({
                str(k): dt.Scalar(k.dtype)
                for k in arg.used_symbols(all_symbols=False)
                if not str(k).startswith('__dace') and str(k) not in sdfg.constants
            })

        return data_args, scalar_args

    def signature_arglist(self, with_types=True, for_call=False):
        """ Returns a list of arguments necessary to call this state or
            subgraph, formatted as a list of C definitions.

            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        return [v.as_arg(name=k, with_types=with_types, for_call=for_call) for k, v in self.arglist().items()]

    def scope_subgraph(self, entry_node, include_entry=True, include_exit=True):
        from dace.sdfg.scope import _scope_subgraph
        return _scope_subgraph(self, entry_node, include_entry, include_exit)

    def top_level_transients(self):
        """Iterate over top-level transients of this state."""
        schildren = self.scope_children()
        sdfg = self.sdfg
        result = set()
        for node in schildren[None]:
            if isinstance(node, nd.AccessNode) and node.desc(sdfg).transient:
                result.add(node.data)
        return result

    def all_transients(self) -> List[str]:
        """Iterate over all transients in this state."""
        return dtypes.deduplicate(
            [n.data for n in self.nodes() if isinstance(n, nd.AccessNode) and n.desc(self.sdfg).transient])

    def replace(self, name: str, new_name: str):
        """ Finds and replaces all occurrences of a symbol or array in this
            state.

            :param name: Name to find.
            :param new_name: Name to replace.
        """
        from dace.sdfg.replace import replace
        replace(self, name, new_name)

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
        from dace.sdfg.replace import replace_dict
        replace_dict(self, repl, symrepl)


@make_properties
class ControlGraphView(BlockGraphView, abc.ABC):

    ###################################################################
    # Typing overrides

    @overload
    def nodes(self) -> List['ControlFlowBlock']:
        ...

    @overload
    def edges(self) -> List[Edge['dace.sdfg.InterstateEdge']]:
        ...

    @overload
    def in_edges(self, node: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        ...

    @overload
    def out_edges(self, node: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        ...

    @overload
    def all_edges(self, node: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        ...

    ###################################################################
    # Traversal methods

    def all_nodes_recursive(self, predicate=None) -> Iterator[Tuple[NodeT, GraphT]]:
        for node in self.nodes():
            yield node, self
            if predicate is None or predicate(node, self):
                yield from node.all_nodes_recursive(predicate)

    def all_edges_recursive(self) -> Iterator[Tuple[EdgeT, GraphT]]:
        for e in self.edges():
            yield e, self
        for node in self.nodes():
            yield from node.all_edges_recursive()

    def data_nodes(self) -> List[nd.AccessNode]:
        data_nodes = []
        for node in self.nodes():
            data_nodes.extend(node.data_nodes())
        return data_nodes

    def entry_node(self, node: nd.Node) -> Optional[nd.EntryNode]:
        for block in self.nodes():
            if node in block.nodes():
                return block.entry_node(node)
        return None

    def exit_node(self, entry_node: nd.EntryNode) -> Optional[nd.ExitNode]:
        for block in self.nodes():
            if entry_node in block.nodes():
                return block.exit_node(entry_node)
        return None

    ###################################################################
    # Memlet-tracking methods

    def memlet_path(self, edge: MultiConnectorEdge[mm.Memlet]) -> List[MultiConnectorEdge[mm.Memlet]]:
        for block in self.nodes():
            if edge in block.edges():
                return block.memlet_path(edge)
        return []

    def memlet_tree(self, edge: MultiConnectorEdge) -> mm.MemletTree:
        for block in self.nodes():
            if edge in block.edges():
                return block.memlet_tree(edge)
        return mm.MemletTree(edge)

    def in_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        for block in self.nodes():
            if node in block.nodes():
                return block.in_edges_by_connector(node, connector)
        return []

    def out_edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        for block in self.nodes():
            if node in block.nodes():
                return block.out_edges_by_connector(node, connector)
        return []

    def edges_by_connector(self, node: nd.Node, connector: AnyStr) -> Iterable[MultiConnectorEdge[mm.Memlet]]:
        for block in self.nodes():
            if node in block.nodes():
                return block.edges_by_connector(node, connector)

    ###################################################################
    # Query, subgraph, and replacement methods

    @abc.abstractmethod
    def _used_symbols_internal(self,
                               all_symbols: bool,
                               defined_syms: Optional[Set] = None,
                               free_syms: Optional[Set] = None,
                               used_before_assignment: Optional[Set] = None,
                               keep_defined_in_mapping: bool = False,
                               with_contents: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        raise NotImplementedError()

    def used_symbols(self,
                     all_symbols: bool,
                     keep_defined_in_mapping: bool = False,
                     with_contents: bool = True) -> Set[str]:
        return self._used_symbols_internal(all_symbols,
                                           keep_defined_in_mapping=keep_defined_in_mapping,
                                           with_contents=with_contents)[0]

    def read_and_write_sets(self) -> Tuple[Set[AnyStr], Set[AnyStr]]:
        read_set = set()
        write_set = set()
        for block in self.nodes():
            for edge in self.in_edges(block):
                read_set |= edge.data.free_symbols & self.sdfg.arrays.keys()
            rs, ws = block.read_and_write_sets()
            read_set.update(rs)
            write_set.update(ws)
        return read_set, write_set

    def unordered_arglist(self,
                          defined_syms=None,
                          shared_transients=None) -> Tuple[Dict[str, dt.Data], Dict[str, dt.Data]]:
        data_args = {}
        scalar_args = {}
        for block in self.nodes():
            n_data_args, n_scalar_args = block.unordered_arglist(defined_syms, shared_transients)
            data_args.update(n_data_args)
            scalar_args.update(n_scalar_args)
        return data_args, scalar_args

    def top_level_transients(self) -> Set[str]:
        res = set()
        for block in self.nodes():
            res.update(block.top_level_transients())
        return res

    def all_transients(self) -> List[str]:
        res = []
        for block in self.nodes():
            res.extend(block.all_transients())
        return dtypes.deduplicate(res)

    def replace(self, name: str, new_name: str):
        for n in self.nodes():
            n.replace(name, new_name)
        for e in self.edges():
            e.data.replace(name, new_name)

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None,
                     replace_in_graph: bool = True,
                     replace_keys: bool = False):
        symrepl = symrepl or {
            symbolic.symbol(k): symbolic.pystr_to_symbolic(v) if isinstance(k, str) else v
            for k, v in repl.items()
        }

        if replace_in_graph:
            # Replace in inter-state edges
            for edge in self.edges():
                edge.data.replace_dict(repl, replace_keys=replace_keys)

            # Replace in states
            for state in self.nodes():
                state.replace_dict(repl, symrepl)


@make_properties
class ControlFlowBlock(BlockGraphView, abc.ABC):

    guid = Property(dtype=str, allow_none=False)

    is_collapsed = Property(dtype=bool, desc='Show this block as collapsed', default=False)

    pre_conditions = DictProperty(key_type=str, value_type=list, desc='Pre-conditions for this block')
    post_conditions = DictProperty(key_type=str, value_type=list, desc='Post-conditions for this block')
    invariant_conditions = DictProperty(key_type=str, value_type=list, desc='Invariant conditions for this block')
    ranges = DictProperty(key_type=str,
                          value_type=Range,
                          default={},
                          desc='Variable ranges across this block, typically within loops')

    executions = SymbolicProperty(default=0,
                                  desc="The number of times this block gets executed (0 stands for unbounded)")
    dynamic_executions = Property(dtype=bool, default=True, desc="The number of executions of this block is dynamic")

    _label: str

    _default_lineinfo: Optional[dace.dtypes.DebugInfo] = None
    _sdfg: Optional['SDFG'] = None
    _parent_graph: Optional['ControlFlowRegion'] = None

    def __init__(self, label: str = '', sdfg: Optional['SDFG'] = None, parent: Optional['ControlFlowRegion'] = None):
        super(ControlFlowBlock, self).__init__()
        self._label = label
        self._default_lineinfo = None
        self._sdfg = sdfg
        self._parent_graph = parent
        self.is_collapsed = False
        self.pre_conditions = {}
        self.post_conditions = {}
        self.invariant_conditions = {}

        self.guid = generate_element_id(self)

    def nodes(self):
        return []

    def edges(self):
        return []

    def sub_regions(self) -> List['AbstractControlFlowRegion']:
        return []

    def set_default_lineinfo(self, lineinfo: dace.dtypes.DebugInfo):
        """
        Sets the default source line information to be lineinfo, or None to
        revert to default mode.
        """
        self._default_lineinfo = lineinfo

    def view(self):
        from dace.sdfg.analysis.cutout import SDFGCutout
        cutout = SDFGCutout.multistate_cutout(self, make_side_effects_global=False, override_start_block=self)
        cutout.view()

    def to_json(self, parent=None):
        tmp = {
            'type': self.__class__.__name__,
            'collapsed': self.is_collapsed,
            'label': self._label,
            'id': parent.node_id(self) if parent is not None else None,
            'attributes': serialize.all_properties_to_json(self),
        }
        return tmp

    @classmethod
    def from_json(cls, json_obj, context=None):
        context = context or {'sdfg': None, 'parent_graph': None}
        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError("Class type mismatch")

        ret = cls(label=json_obj['label'], sdfg=context['sdfg'])

        dace.serialize.set_properties_from_json(ret, json_obj)

        return ret

    def __str__(self):
        return self._label

    def __repr__(self) -> str:
        return f'ControlFlowBlock ({self.label})'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('_parent_graph', '_sdfg', '_cfg_list', 'guid'):  # Skip derivative attributes and GUID
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        for k in ('_parent_graph', '_sdfg'):
            if id(getattr(self, k)) in memo:
                setattr(result, k, memo[id(getattr(self, k))])
            else:
                setattr(result, k, None)

        return result

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    @property
    def sdfg(self) -> 'SDFG':
        return self._sdfg

    @sdfg.setter
    def sdfg(self, sdfg: 'SDFG'):
        self._sdfg = sdfg

    @property
    def parent_graph(self) -> 'ControlFlowRegion':
        return self._parent_graph

    @parent_graph.setter
    def parent_graph(self, parent: Optional['ControlFlowRegion']):
        self._parent_graph = parent

    @property
    def block_id(self) -> int:
        return self.parent_graph.node_id(self)


@make_properties
class SDFGState(OrderedMultiDiConnectorGraph[nd.Node, mm.Memlet], ControlFlowBlock, DataflowGraphView):
    """ An acyclic dataflow multigraph in an SDFG, corresponding to a
        single state in the SDFG state machine. """

    nosync = Property(dtype=bool, default=False, desc="Do not synchronize at the end of the state")

    instrument = EnumProperty(dtype=dtypes.InstrumentationType,
                              desc="Measure execution statistics with given method",
                              default=dtypes.InstrumentationType.No_Instrumentation)

    symbol_instrument = EnumProperty(dtype=dtypes.DataInstrumentationType,
                                     desc="Instrument symbol values when this state is executed",
                                     default=dtypes.DataInstrumentationType.No_Instrumentation)
    symbol_instrument_condition = CodeProperty(desc="Condition under which to trigger the symbol instrumentation",
                                               default=CodeBlock("1", language=dtypes.Language.CPP))

    location = DictProperty(key_type=str,
                            value_type=symbolic.pystr_to_symbolic,
                            desc='Full storage location identifier (e.g., rank, GPU ID)')

    def __repr__(self) -> str:
        return f"SDFGState ({self.label})"

    def __init__(self, label=None, sdfg=None, debuginfo=None, location=None):
        """ Constructs an SDFG state.

            :param label: Name for the state (optional).
            :param sdfg: A reference to the parent SDFG.
            :param debuginfo: Source code locator for debugging.
        """
        OrderedMultiDiConnectorGraph.__init__(self)
        ControlFlowBlock.__init__(self, label, sdfg)
        super(SDFGState, self).__init__()
        self._label = label
        self._graph = self  # Allowing MemletTrackingView mixin to work
        self._clear_scopedict_cache()
        self._debuginfo = debuginfo
        self.nosync = False
        self.location = location if location is not None else {}
        self._default_lineinfo = None

    @property
    def parent(self):
        """ Returns the parent SDFG of this state. """
        return self.sdfg

    @parent.setter
    def parent(self, value):
        self.sdfg = value

    def is_empty(self):
        return self.number_of_nodes() == 0

    def validate(self) -> None:
        validate_state(self)

    def nodes(self) -> List[nd.Node]:  # Added for type hints
        return super().nodes()

    def all_edges_and_connectors(self, *nodes):
        """
        Returns an iterable to incoming and outgoing Edge objects, along
        with their connector types.
        """
        for node in nodes:
            for e in self.in_edges(node):
                yield e, (node.in_connectors[e.dst_conn] if e.dst_conn else None)
            for e in self.out_edges(node):
                yield e, (node.out_connectors[e.src_conn] if e.src_conn else None)

    def add_node(self, node):
        if not isinstance(node, nd.Node):
            raise TypeError("Expected Node, got " + type(node).__name__ + " (" + str(node) + ")")
        # Correct nested SDFG's parent attributes
        if isinstance(node, nd.NestedSDFG) and node.sdfg is not None:
            node.sdfg.parent = self
            node.sdfg.parent_sdfg = self.sdfg
            node.sdfg.parent_nsdfg_node = node
        self._clear_scopedict_cache()
        return super(SDFGState, self).add_node(node)

    def remove_node(self, node):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_node(node)

    def add_edge(self, u, u_connector, v, v_connector, memlet):
        if not isinstance(u, nd.Node):
            raise TypeError("Source node is not of type nd.Node (type: %s)" % str(type(u)))
        if u_connector is not None and not isinstance(u_connector, str):
            raise TypeError("Source connector is not string (type: %s)" % str(type(u_connector)))
        if not isinstance(v, nd.Node):
            raise TypeError("Destination node is not of type nd.Node (type: " + "%s)" % str(type(v)))
        if v_connector is not None and not isinstance(v_connector, str):
            raise TypeError("Destination connector is not string (type: %s)" % str(type(v_connector)))
        if not isinstance(memlet, mm.Memlet):
            raise TypeError("Memlet is not of type Memlet (type: %s)" % str(type(memlet)))

        if u_connector and isinstance(u, nd.AccessNode) and u_connector not in u.out_connectors:
            u.add_out_connector(u_connector, force=True)
        if v_connector and isinstance(v, nd.AccessNode) and v_connector not in v.in_connectors:
            v.add_in_connector(v_connector, force=True)

        self._clear_scopedict_cache()
        result = super(SDFGState, self).add_edge(u, u_connector, v, v_connector, memlet)
        memlet.try_initialize(self.sdfg, self, result)
        return result

    def remove_edge(self, edge):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_edge(edge)

    def remove_edge_and_connectors(self, edge):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_edge(edge)
        if edge.src_conn in edge.src.out_connectors:
            edge.src.remove_out_connector(edge.src_conn)
        if edge.dst_conn in edge.dst.in_connectors:
            edge.dst.remove_in_connector(edge.dst_conn)

    def to_json(self, parent=None):
        # Create scope dictionary with a failsafe
        try:
            scope_dict = {k: sorted(v) for k, v in sorted(self.scope_children(return_ids=True).items())}
        except (RuntimeError, ValueError):
            scope_dict = {}

        # Try to initialize edges before serialization
        for edge in self.edges():
            edge.data.try_initialize(self.sdfg, self, edge)

        ret = {
            'type': type(self).__name__,
            'label': self.name,
            'id': parent.node_id(self) if parent is not None else None,
            'collapsed': self.is_collapsed,
            'scope_dict': scope_dict,
            'nodes': [n.to_json(self) for n in self.nodes()],
            'edges':
            [e.to_json(self) for e in sorted(self.edges(), key=lambda e: (e.src_conn or '', e.dst_conn or ''))],
            'attributes': serialize.all_properties_to_json(self),
        }

        return ret

    @classmethod
    def from_json(cls, json_obj, context={'sdfg': None}, pre_ret=None):
        """ Loads the node properties, label and type into a dict.

            :param json_obj: The object containing information about this node.
                             NOTE: This may not be a string!
            :return: An SDFGState instance constructed from the passed data
        """

        _type = json_obj['type']
        if _type != cls.__name__:
            raise Exception("Class type mismatch")

        attrs = json_obj['attributes']
        nodes = json_obj['nodes']
        edges = json_obj['edges']

        ret = pre_ret if pre_ret is not None else SDFGState(
            label=json_obj['label'], sdfg=context['sdfg'], debuginfo=None)

        rec_ci = {
            'sdfg': context['sdfg'],
            'sdfg_state': ret,
            'callback': context['callback'] if 'callback' in context else None
        }
        serialize.set_properties_from_json(ret, json_obj, rec_ci)

        for n in nodes:
            nret = serialize.from_json(n, context=rec_ci)
            ret.add_node(nret)

        # Connect using the edges
        for e in edges:
            eret = serialize.from_json(e, context=rec_ci)

            ret.add_edge(eret.src, eret.src_conn, eret.dst, eret.dst_conn, eret.data)

        # Fix potentially broken scopes
        for n in nodes:
            if isinstance(n, nd.MapExit):
                n.map = ret.entry_node(n).map
            elif isinstance(n, nd.ConsumeExit):
                n.consume = ret.entry_node(n).consume

        # Reinitialize memlets
        for edge in ret.edges():
            edge.data.try_initialize(context['sdfg'], ret, edge)

        return ret

    def _repr_html_(self):
        """ HTML representation of a state, used mainly for Jupyter
            notebooks. """
        # Create dummy SDFG with this state as the only one
        from dace.sdfg import SDFG
        arrays = set(n.data for n in self.data_nodes())
        sdfg = SDFG(self.label)
        sdfg._arrays = dace.sdfg.NestedDict({k: self.sdfg.arrays[k] for k in arrays})
        sdfg.add_node(self)

        return sdfg._repr_html_()

    def __deepcopy__(self, memo):
        result: SDFGState = ControlFlowBlock.__deepcopy__(self, memo)

        for node in result.nodes():
            if isinstance(node, nd.NestedSDFG):
                try:
                    node.sdfg.parent = result
                except AttributeError:
                    # NOTE: There are cases where a NestedSDFG does not have `sdfg` attribute.
                    # TODO: Investigate why this happens.
                    pass
        return result

    def symbols_defined_at(self, node: nd.Node) -> Dict[str, dtypes.typeclass]:
        """
        Returns all symbols available to a given node.
        The symbols a node can access are a combination of the global SDFG
        symbols, symbols defined in inter-state paths to its state,
        and symbols defined in scope entries in the path to this node.

        :param node: The given node.
        :return: A dictionary mapping symbol names to their types.
        """
        from dace.sdfg.sdfg import SDFG

        if node is None:
            return collections.OrderedDict()

        sdfg: SDFG = self.sdfg

        # Start with global symbols
        symbols = collections.OrderedDict(sdfg.symbols)
        for desc in sdfg.arrays.values():
            symbols.update([(str(s), s.dtype) for s in desc.free_symbols])

        # Add symbols from inter-state edges along the path to the state
        try:
            start_state = sdfg.start_state
            for e in sdfg.predecessor_state_transitions(start_state):
                symbols.update(e.data.new_symbols(sdfg, symbols))
        except ValueError:
            # Cannot determine starting state (possibly some inter-state edges
            # do not yet exist)
            for e in sdfg.edges():
                symbols.update(e.data.new_symbols(sdfg, symbols))

        # Find scopes this node is situated in
        sdict = self.scope_dict()
        scope_list = []
        curnode = node
        while sdict[curnode] is not None:
            curnode = sdict[curnode]
            scope_list.append(curnode)

        # Add the scope symbols top-down
        for scope_node in reversed(scope_list):
            symbols.update(scope_node.new_symbols(sdfg, self, symbols))

        return symbols

    # Dynamic SDFG creation API
    ##############################
    def add_read(self, array_or_stream_name: str, debuginfo: Optional[dtypes.DebugInfo] = None) -> nd.AccessNode:
        """
        Adds an access node to this SDFG state (alias of ``add_access``).

        :param array_or_stream_name: The name of the array/stream.
        :param debuginfo: Source line information for this access node.
        :return: An array access node.
        :see: add_access
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        return self.add_access(array_or_stream_name, debuginfo=debuginfo)

    def add_write(self, array_or_stream_name: str, debuginfo: Optional[dtypes.DebugInfo] = None) -> nd.AccessNode:
        """
        Adds an access node to this SDFG state (alias of ``add_access``).

        :param array_or_stream_name: The name of the array/stream.
        :param debuginfo: Source line information for this access node.
        :return: An array access node.
        :see: add_access
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        return self.add_access(array_or_stream_name, debuginfo=debuginfo)

    def add_access(self, array_or_stream_name: str, debuginfo: Optional[dtypes.DebugInfo] = None) -> nd.AccessNode:
        """ Adds an access node to this SDFG state.

            :param array_or_stream_name: The name of the array/stream.
            :param debuginfo: Source line information for this access node.
            :return: An array access node.
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        node = nd.AccessNode(array_or_stream_name, debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_tasklet(
        self,
        name: str,
        inputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        code: str,
        language: dtypes.Language = dtypes.Language.Python,
        state_fields: Optional[List[str]] = None,
        code_global: str = "",
        code_init: str = "",
        code_exit: str = "",
        location: dict = None,
        side_effects: Optional[bool] = None,
        debuginfo=None,
    ):
        """ Adds a tasklet to the SDFG state. """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)

        # Make dictionary of autodetect connector types from set
        if isinstance(inputs, (set, collections.abc.KeysView)):
            inputs = {k: None for k in inputs}
        if isinstance(outputs, (set, collections.abc.KeysView)):
            outputs = {k: None for k in outputs}

        tasklet = nd.Tasklet(
            name,
            inputs,
            outputs,
            code,
            language,
            state_fields=state_fields,
            code_global=code_global,
            code_init=code_init,
            code_exit=code_exit,
            location=location,
            side_effects=side_effects,
            debuginfo=debuginfo,
        ) if language != dtypes.Language.SystemVerilog else nd.RTLTasklet(
            name,
            inputs,
            outputs,
            code,
            language,
            state_fields=state_fields,
            code_global=code_global,
            code_init=code_init,
            code_exit=code_exit,
            location=location,
            side_effects=side_effects,
            debuginfo=debuginfo,
        )
        self.add_node(tasklet)
        return tasklet

    def add_nested_sdfg(
        self,
        sdfg: Optional['SDFG'],
        inputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        symbol_mapping: Dict[str, Any] = None,
        name=None,
        schedule=dtypes.ScheduleType.Default,
        location: Optional[Dict[str, symbolic.SymbolicType]] = None,
        debuginfo: Optional[dtypes.DebugInfo] = None,
        external_path: Optional[str] = None,
    ):
        """
        Adds a nested SDFG to the SDFG state.

        :param sdfg: The SDFG to nest. Can be None if ``external_path`` is provided.
        :param inputs: Input connectors of the nested SDFG. Can be a set of connector names
                       (types will be auto-detected) or a dict mapping connector names to data types.
        :param outputs: Output connectors of the nested SDFG. Can be a set of connector names
                        (types will be auto-detected) or a dict mapping connector names to data types.
        :param symbol_mapping: A dictionary mapping nested SDFG symbol names to expressions in the
                               parent SDFG's scope. If None, symbols are mapped to themselves.
        :param name: Name of the nested SDFG node. If None, uses the nested SDFG's label.
        :param schedule: Schedule type for the nested SDFG node. Defaults to ``ScheduleType.Default``. This argument
                         is deprecated and will be removed in the future.
        :param location: Execution location descriptor for the nested SDFG.
        :param debuginfo: Debug information for the nested SDFG node.
        :param external_path: Path to an external SDFG file. Used when ``sdfg`` parameter is None.
        :return: The created NestedSDFG node.
        :raises ValueError: If neither sdfg nor external_path is provided, or if required symbols
                           are missing from the symbol mapping.
        """
        if name is None:
            name = sdfg.label
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)

        if sdfg is None and external_path is None:
            raise ValueError('Neither an SDFG nor an external SDFG path has been provided')

        if schedule != dtypes.ScheduleType.Default:
            warnings.warn(
                "The 'schedule' argument is deprecated and will be removed in the future.",
                DeprecationWarning,
            )

        if sdfg is not None:
            sdfg.parent = self
            sdfg.parent_sdfg = self.sdfg

            sdfg.update_cfg_list([])

        # Make dictionary of autodetect connector types from set
        if isinstance(inputs, (set, collections.abc.KeysView)):
            inputs = {k: None for k in inputs}
        if isinstance(outputs, (set, collections.abc.KeysView)):
            outputs = {k: None for k in outputs}

        s = nd.NestedSDFG(
            name,
            sdfg,
            inputs,
            outputs,
            symbol_mapping=symbol_mapping,
            schedule=schedule,
            location=location,
            debuginfo=debuginfo,
            path=external_path,
        )
        self.add_node(s)

        if sdfg is not None:
            sdfg.parent_nsdfg_node = s

            # Add "default" undefined symbols if None are given
            symbols = sdfg.free_symbols
            if symbol_mapping is None:
                symbol_mapping = {s: s for s in symbols}
                s.symbol_mapping = symbol_mapping

            # Validate missing symbols
            missing_symbols = [s for s in symbols if s not in symbol_mapping]
            if missing_symbols and self.sdfg is not None:
                # If symbols are missing, try to get them from the parent SDFG
                parent_mapping = {s: s for s in missing_symbols if s in self.sdfg.symbols}
                symbol_mapping.update(parent_mapping)
                s.symbol_mapping = symbol_mapping
                missing_symbols = [s for s in symbols if s not in symbol_mapping]
            if missing_symbols:
                raise ValueError('Missing symbols on nested SDFG "%s": %s' % (name, missing_symbols))

            # Add new global symbols to nested SDFG
            from dace.codegen.tools.type_inference import infer_expr_type
            for sym, symval in s.symbol_mapping.items():
                if sym not in sdfg.symbols:
                    # TODO: Think of a better way to avoid calling
                    # symbols_defined_at in this moment
                    sdfg.add_symbol(sym, infer_expr_type(symval, self.sdfg.symbols) or dtypes.typeclass(int))

        return s

    def add_map(
        self,
        name,
        ndrange: Union[Dict[str, Union[str, sbs.Subset]], List[Tuple[str, Union[str, sbs.Subset]]]],
        schedule=dtypes.ScheduleType.Default,
        unroll=False,
        debuginfo=None,
    ) -> Tuple[nd.MapEntry, nd.MapExit]:
        """ Adds a map entry and map exit.

            :param name:      Map label
            :param ndrange:   Mapping between range variable names and their
                              subsets (parsed from strings)
            :param schedule:  Map schedule type
            :param unroll:    True if should unroll the map in code generation

            :return: (map_entry, map_exit) node 2-tuple
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        map = nd.Map(name, *_make_iterators(ndrange), schedule=schedule, unroll=unroll, debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, map_exit])
        return map_entry, map_exit

    def add_consume(self,
                    name,
                    elements: Tuple[str, str],
                    condition: str = None,
                    schedule=dtypes.ScheduleType.Default,
                    chunksize=1,
                    debuginfo=None,
                    language=dtypes.Language.Python) -> Tuple[nd.ConsumeEntry, nd.ConsumeExit]:
        """ Adds consume entry and consume exit nodes.

            :param name:      Label
            :param elements:  A 2-tuple signifying the processing element
                              index and number of total processing elements
            :param condition: Quiescence condition to finish consuming, or
                              None (by default) to consume until the stream
                              is empty for the first time. If false, will
                              consume forever.
            :param schedule:  Consume schedule type.
            :param chunksize: Maximal number of elements to consume at a time.
            :param debuginfo: Source code line information for debugging.
            :param language:  Code language for ``condition``.

            :return: (consume_entry, consume_exit) node 2-tuple
        """
        if len(elements) != 2:
            raise TypeError("Elements must be a 2-tuple of "
                            "(PE_index, num_PEs)")
        pe_tuple = (elements[0], SymbolicProperty.from_string(elements[1]))

        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        if condition is not None:
            condition = CodeBlock(condition, language)
        consume = nd.Consume(name, pe_tuple, condition, schedule, chunksize, debuginfo=debuginfo)
        entry = nd.ConsumeEntry(consume)
        exit = nd.ConsumeExit(consume)

        self.add_nodes_from([entry, exit])
        return entry, exit

    def add_mapped_tasklet(self,
                           name: str,
                           map_ranges: Union[Dict[str, Union[str, sbs.Subset]], List[Tuple[str, Union[str,
                                                                                                      sbs.Subset]]]],
                           inputs: Dict[str, mm.Memlet],
                           code: str,
                           outputs: Dict[str, mm.Memlet],
                           schedule=dtypes.ScheduleType.Default,
                           unroll_map=False,
                           location=None,
                           language=dtypes.Language.Python,
                           debuginfo=None,
                           external_edges=False,
                           input_nodes: Optional[Union[Dict[str, nd.AccessNode], List[nd.AccessNode],
                                                       Set[nd.AccessNode]]] = None,
                           output_nodes: Optional[Union[Dict[str, nd.AccessNode], List[nd.AccessNode],
                                                        Set[nd.AccessNode]]] = None,
                           propagate=True) -> Tuple[nd.Tasklet, nd.MapEntry, nd.MapExit]:
        """ Convenience function that adds a map entry, tasklet, map exit,
            and the respective edges to external arrays.

            :param name:       Tasklet (and wrapping map) name
            :param map_ranges: Mapping between variable names and their
                               subsets
            :param inputs:     Mapping between input local variable names and
                               their memlets
            :param code:       Code (written in `language`)
            :param outputs:    Mapping between output local variable names and
                               their memlets
            :param schedule:   Map schedule
            :param unroll_map: True if map should be unrolled in code
                               generation
            :param location:   Execution location indicator.
            :param language:   Programming language in which the code is
                               written
            :param debuginfo:  Source line information
            :param external_edges: Create external access nodes and connect
                                   them with memlets automatically
            :param input_nodes: Mapping between data names and corresponding
                                input nodes to link to, if external_edges is
                                True.
            :param output_nodes: Mapping between data names and corresponding
                                 output nodes to link to, if external_edges is
                                 True.
            :param propagate: If True, computes outer memlets via propagation.
                              False will run faster but the SDFG may not be
                              semantically correct.
            :return: tuple of (tasklet, map_entry, map_exit)
        """
        map_name = name + "_map"
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)

        # Create appropriate dictionaries from inputs
        tinputs = {k: None for k, v in inputs.items()}
        toutputs = {k: None for k, v in outputs.items()}

        if isinstance(input_nodes, (list, set)):
            input_nodes = {input_node.data: input_node for input_node in input_nodes}
        if isinstance(output_nodes, (list, set)):
            output_nodes = {output_node.data: output_node for output_node in output_nodes}

        tasklet = nd.Tasklet(
            name,
            tinputs,
            toutputs,
            code,
            language=language,
            location=location,
            debuginfo=debuginfo,
        )
        map = nd.Map(map_name, *_make_iterators(map_ranges), schedule=schedule, unroll=unroll_map, debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, tasklet, map_exit])

        # Create access nodes
        inpdict = {}
        outdict = {}
        if external_edges:
            input_nodes = input_nodes or {}
            output_nodes = output_nodes or {}
            input_data = dtypes.deduplicate([memlet.data for memlet in inputs.values()])
            output_data = dtypes.deduplicate([memlet.data for memlet in outputs.values()])
            for inp in sorted(input_data):
                if inp in input_nodes:
                    inpdict[inp] = input_nodes[inp]
                else:
                    inpdict[inp] = self.add_read(inp)
            for out in sorted(output_data):
                if out in output_nodes:
                    outdict[out] = output_nodes[out]
                else:
                    outdict[out] = self.add_write(out)

        edges: List[Edge[dace.Memlet]] = []

        # Connect inputs from map to tasklet
        tomemlet = {}
        for name, memlet in sorted(inputs.items()):
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            edges.append(self.add_edge(map_entry, None, tasklet, name, memlet))
            tomemlet[memlet.data] = memlet

        # If there are no inputs, add empty memlet
        if len(inputs) == 0:
            self.add_edge(map_entry, None, tasklet, None, mm.Memlet())

        if external_edges:
            for inp, inpnode in sorted(inpdict.items()):
                # Add external edge
                if propagate:
                    outer_memlet = propagate_memlet(self, tomemlet[inp], map_entry, True)
                else:
                    outer_memlet = tomemlet[inp]
                edges.append(self.add_edge(inpnode, None, map_entry, "IN_" + inp, outer_memlet))

                # Add connectors to internal edges
                for e in self.out_edges(map_entry):
                    if e.data.data == inp:
                        e._src_conn = "OUT_" + inp

                # Add connectors to map entry
                map_entry.add_in_connector("IN_" + inp)
                map_entry.add_out_connector("OUT_" + inp)

        # Connect outputs from tasklet to map
        tomemlet = {}
        for name, memlet in sorted(outputs.items()):
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            edges.append(self.add_edge(tasklet, name, map_exit, None, memlet))
            tomemlet[memlet.data] = memlet

        # If there are no outputs, add empty memlet
        if len(outputs) == 0:
            self.add_edge(tasklet, None, map_exit, None, mm.Memlet())

        if external_edges:
            for out, outnode in sorted(outdict.items()):
                # Add external edge
                if propagate:
                    outer_memlet = propagate_memlet(self, tomemlet[out], map_exit, True)
                else:
                    outer_memlet = tomemlet[out]
                edges.append(self.add_edge(map_exit, "OUT_" + out, outnode, None, outer_memlet))

                # Add connectors to internal edges
                for e in self.in_edges(map_exit):
                    if e.data.data == out:
                        e._dst_conn = "IN_" + out

                # Add connectors to map entry
                map_exit.add_in_connector("IN_" + out)
                map_exit.add_out_connector("OUT_" + out)

        # Try to initialize memlets
        for edge in edges:
            edge.data.try_initialize(self.sdfg, self, edge)

        return tasklet, map_entry, map_exit

    def add_reduce(
        self,
        wcr,
        axes,
        identity=None,
        schedule=dtypes.ScheduleType.Default,
        debuginfo=None,
    ) -> 'dace.libraries.standard.Reduce':
        """ Adds a reduction node.

            :param wcr: A lambda function representing the reduction operation
            :param axes: A tuple of axes to reduce the input memlet from, or
                         None for all axes
            :param identity: If not None, initializes output memlet values
                                 with this value
            :param schedule: Reduction schedule type

            :return: A Reduce node
        """
        import dace.libraries.standard as stdlib  # Avoid import loop
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        result = stdlib.Reduce('Reduce', wcr, axes, identity, schedule=schedule, debuginfo=debuginfo)
        self.add_node(result)
        return result

    def add_pipeline(self,
                     name,
                     ndrange,
                     init_size=0,
                     init_overlap=False,
                     drain_size=0,
                     drain_overlap=False,
                     additional_iterators={},
                     schedule=dtypes.ScheduleType.FPGA_Device,
                     debuginfo=None,
                     **kwargs) -> Tuple[nd.PipelineEntry, nd.PipelineExit]:
        """ Adds a pipeline entry and pipeline exit. These are used for FPGA
            kernels to induce distinct behavior between an "initialization"
            phase, a main streaming phase, and a "draining" phase, which require
            a additive number of extra loop iterations (i.e., N*M + I + D),
            where I and D are the number of initialization/drain iterations.
            The code can detect which phase it is in by querying the
            init_condition() and drain_condition() boolean variable.

            :param name:          Pipeline label
            :param ndrange:       Mapping between range variable names and
                                  their subsets (parsed from strings)
            :param init_size:     Number of iterations of initialization phase.
            :param init_overlap:  Whether the initialization phase overlaps
                                  with the "main" streaming phase of the loop.
            :param drain_size:    Number of iterations of draining phase.
            :param drain_overlap: Whether the draining phase overlaps with
                                  the "main" streaming phase of the loop.
            :param additional_iterators: a dictionary containing additional
                                  iterators that will be created for this scope and that are not
                                  automatically managed by the scope code.
                                  The dictionary takes the form 'variable_name' -> init_value
            :return: (map_entry, map_exit) node 2-tuple
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        pipeline = nd.PipelineScope(name,
                                    *_make_iterators(ndrange),
                                    init_size=init_size,
                                    init_overlap=init_overlap,
                                    drain_size=drain_size,
                                    drain_overlap=drain_overlap,
                                    additional_iterators=additional_iterators,
                                    schedule=schedule,
                                    debuginfo=debuginfo,
                                    **kwargs)
        pipeline_entry = nd.PipelineEntry(pipeline)
        pipeline_exit = nd.PipelineExit(pipeline)
        self.add_nodes_from([pipeline_entry, pipeline_exit])
        return pipeline_entry, pipeline_exit

    def add_edge_pair(
        self,
        scope_node,
        internal_node,
        external_node,
        internal_memlet,
        external_memlet=None,
        scope_connector=None,
        internal_connector=None,
        external_connector=None,
    ):
        """ Adds two edges around a scope node (e.g., map entry, consume
            exit).

            The internal memlet (connecting to the internal node) has to be
            specified. If external_memlet (i.e., connecting to the node out
            of the scope) is not specified, it is propagated automatically
            using internal_memlet and the scope.

            :param scope_node: A scope node (for example, map exit) to add
                               edges around.
            :param internal_node: The node within the scope to connect to. If
                                  `scope_node` is an entry node, this means
                                  the node connected to the outgoing edge,
                                  else incoming.
            :param external_node: The node out of the scope to connect to.
            :param internal_memlet: The memlet on the edge to/from
                                    internal_node.
            :param external_memlet: The memlet on the edge to/from
                                    external_node (optional, will propagate
                                    internal_memlet if not specified).
            :param scope_connector: A scope connector name (or a unique
                                    number if not specified).
            :param internal_connector: The connector on internal_node to
                                       connect to.
            :param external_connector: The connector on external_node to
                                       connect to.
            :return: A 2-tuple representing the (internal, external) edges.
        """
        if not isinstance(scope_node, (nd.EntryNode, nd.ExitNode)):
            raise ValueError("scope_node is not a scope entry/exit")

        # Autodetermine scope connector ID
        if scope_connector is None:
            # Pick out numbered connectors that do not lead into the scope range
            conn_id = 1
            for conn in (scope_node.in_connectors.keys() | scope_node.out_connectors.keys()):
                if conn.startswith("IN_") or conn.startswith("OUT_"):
                    conn_name = conn[conn.find("_") + 1:]
                    try:
                        cid = int(conn_name)
                        if cid >= conn_id:
                            conn_id = cid + 1
                    except (TypeError, ValueError):
                        pass
            scope_connector = str(conn_id)

        # Add connectors
        scope_node.add_in_connector("IN_" + scope_connector)
        scope_node.add_out_connector("OUT_" + scope_connector)
        ##########################

        # Add internal edge
        if isinstance(scope_node, nd.EntryNode):
            iedge = self.add_edge(
                scope_node,
                "OUT_" + scope_connector,
                internal_node,
                internal_connector,
                internal_memlet,
            )
        else:
            iedge = self.add_edge(
                internal_node,
                internal_connector,
                scope_node,
                "IN_" + scope_connector,
                internal_memlet,
            )

        # Add external edge
        if external_memlet is None:
            # If undefined, propagate
            external_memlet = propagate_memlet(self, internal_memlet, scope_node, True)

        if isinstance(scope_node, nd.EntryNode):
            eedge = self.add_edge(
                external_node,
                external_connector,
                scope_node,
                "IN_" + scope_connector,
                external_memlet,
            )
        else:
            eedge = self.add_edge(
                scope_node,
                "OUT_" + scope_connector,
                external_node,
                external_connector,
                external_memlet,
            )

        # Try to initialize memlets
        iedge.data.try_initialize(self.sdfg, self, iedge)
        eedge.data.try_initialize(self.sdfg, self, eedge)

        return (iedge, eedge)

    def add_memlet_path(self, *path_nodes, memlet=None, src_conn=None, dst_conn=None, propagate=True):
        """
        Adds a path of memlet edges between the given nodes, propagating
        from the given innermost memlet.

        :param path_nodes: Nodes participating in the path (in the given order).
        :param memlet: (mandatory) The memlet at the innermost scope
                       (e.g., the incoming memlet to a tasklet (last
                       node), or an outgoing memlet from an array
                       (first node), followed by scope exits).
        :param src_conn: Connector at the beginning of the path.
        :param dst_conn: Connector at the end of the path.
        """
        if memlet is None:
            raise TypeError("Innermost memlet cannot be None")
        if len(path_nodes) < 2:
            raise ValueError("Memlet path must consist of at least 2 nodes")

        src_node = path_nodes[0]
        dst_node = path_nodes[-1]

        # Add edges first so that scopes can be understood
        edges = [
            self.add_edge(path_nodes[i], None, path_nodes[i + 1], None, mm.Memlet())
            for i in range(len(path_nodes) - 1)
        ]

        if not isinstance(memlet, mm.Memlet):
            raise TypeError("Expected Memlet, got: {}".format(type(memlet).__name__))

        if any(isinstance(n, nd.EntryNode) for n in path_nodes):
            propagate_forward = False
        else:  # dst node's scope is higher than src node, propagate out
            propagate_forward = True

        # Innermost edge memlet
        cur_memlet = memlet

        cur_memlet._is_data_src = (isinstance(src_node, nd.AccessNode) and src_node.data == cur_memlet.data)

        # Verify that connectors exist
        if (not memlet.is_empty() and hasattr(edges[0].src, "out_connectors") and isinstance(edges[0].src, nd.CodeNode)
                and not isinstance(edges[0].src, nd.LibraryNode)
                and (src_conn is None or src_conn not in edges[0].src.out_connectors)):
            raise ValueError("Output connector {} does not exist in {}".format(src_conn, edges[0].src.label))
        if (not memlet.is_empty() and hasattr(edges[-1].dst, "in_connectors")
                and isinstance(edges[-1].dst, nd.CodeNode) and not isinstance(edges[-1].dst, nd.LibraryNode)
                and (dst_conn is None or dst_conn not in edges[-1].dst.in_connectors)):
            raise ValueError("Input connector {} does not exist in {}".format(dst_conn, edges[-1].dst.label))

        path = edges if propagate_forward else reversed(edges)
        last_conn = None
        # Propagate and add edges
        for i, edge in enumerate(path):
            # Figure out source and destination connectors
            if propagate_forward:
                next_conn = edge.dst.next_connector(memlet.data)
                sconn = src_conn if i == 0 else "OUT_" + last_conn
                dconn = dst_conn if i == len(edges) - 1 else "IN_" + next_conn
            else:
                next_conn = edge.src.next_connector(memlet.data)
                sconn = src_conn if i == len(edges) - 1 else "OUT_" + next_conn
                dconn = dst_conn if i == 0 else "IN_" + last_conn

            last_conn = next_conn

            if cur_memlet.is_empty():
                if propagate_forward:
                    sconn = src_conn if i == 0 else None
                    dconn = dst_conn if i == len(edges) - 1 else None
                else:
                    sconn = src_conn if i == len(edges) - 1 else None
                    dconn = dst_conn if i == 0 else None

            # Modify edge to match memlet path
            edge._src_conn = sconn
            edge._dst_conn = dconn
            edge._data = cur_memlet

            # Add connectors to edges
            if propagate_forward:
                if dconn is not None:
                    edge.dst.add_in_connector(dconn)
                if sconn is not None:
                    edge.src.add_out_connector(sconn)
            else:
                if dconn is not None:
                    edge.dst.add_in_connector(dconn)
                if sconn is not None:
                    edge.src.add_out_connector(sconn)

            # Propagate current memlet to produce the next one
            if i < len(edges) - 1:
                snode = edge.dst if propagate_forward else edge.src
                if not cur_memlet.is_empty():
                    if propagate:
                        cur_memlet = propagate_memlet(self, cur_memlet, snode, True)
        # Try to initialize memlets
        for edge in edges:
            edge.data.try_initialize(self.sdfg, self, edge)

    def remove_memlet_path(self, edge: MultiConnectorEdge, remove_orphans: bool = True) -> None:
        """ Removes all memlets and associated connectors along a path formed
            by a given edge. Undefined behavior if the path is ambiguous.
            Orphaned entry and exit nodes will be connected with empty edges to
            maintain connectivity of the graph.

            :param edge: An edge that is part of the path that should be
                         removed, which will be passed to `memlet_path` to
                         determine the edges to be removed.
            :param remove_orphans: Remove orphaned data nodes from the graph if
                                   they become orphans from removing this memlet
                                   path.
        """

        path = self.memlet_path(edge)

        is_read = isinstance(path[0].src, nd.AccessNode)
        if is_read:
            # Traverse from connector to access node, so we can check if it's
            # safe to delete edges going out of a scope
            path = reversed(path)

        for edge in path:

            self.remove_edge(edge)

            # Check if there are any other edges exiting the source node that
            # use the same connector
            for e in self.out_edges(edge.src):
                if e.src_conn is not None and e.src_conn == edge.src_conn:
                    other_outgoing = True
                    break
            else:
                other_outgoing = False
                edge.src.remove_out_connector(edge.src_conn)

            # Check if there are any other edges entering the destination node
            # that use the same connector
            for e in self.in_edges(edge.dst):
                if e.dst_conn is not None and e.dst_conn == edge.dst_conn:
                    other_incoming = True
                    break
            else:
                other_incoming = False
                edge.dst.remove_in_connector(edge.dst_conn)

            if isinstance(edge.src, nd.EntryNode):
                # If removing this edge orphans the entry node, replace the
                # edge with an empty edge
                # NOTE: The entry node is an orphan iff it has no other outgoing edges.
                if self.out_degree(edge.src) == 0:
                    self.add_nedge(edge.src, edge.dst, mm.Memlet())
                if other_outgoing:
                    # If other inner memlets use the outer memlet, we have to
                    # stop the deletion here
                    break

            if isinstance(edge.dst, nd.ExitNode):
                # If removing this edge orphans the exit node, replace the
                # edge with an empty edge
                # NOTE: The exit node is an orphan iff it has no other incoming edges.
                if self.in_degree(edge.dst) == 0:
                    self.add_nedge(edge.src, edge.dst, mm.Memlet())
                if other_incoming:
                    # If other inner memlets use the outer memlet, we have to
                    # stop the deletion here
                    break

            # Prune access nodes
            if remove_orphans:
                if (isinstance(edge.src, nd.AccessNode) and self.degree(edge.src) == 0):
                    self.remove_node(edge.src)
                if (isinstance(edge.dst, nd.AccessNode) and self.degree(edge.dst) == 0):
                    self.remove_node(edge.dst)

    # DEPRECATED FUNCTIONS
    ######################################
    def add_array(self,
                  name,
                  shape,
                  dtype,
                  storage=dtypes.StorageType.Default,
                  transient=False,
                  strides=None,
                  offset=None,
                  lifetime=dtypes.AllocationLifetime.Scope,
                  debuginfo=None,
                  total_size=None,
                  find_new_name=False,
                  alignment=0):
        """ :note: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_array" API is deprecated, please '
            'use "SDFG.add_array" and "SDFGState.add_access"', DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.sdfg._arrays:
            del self.sdfg._arrays[name]
        self.sdfg.add_array(name,
                            shape,
                            dtype,
                            storage=storage,
                            transient=transient,
                            strides=strides,
                            offset=offset,
                            lifetime=lifetime,
                            debuginfo=debuginfo,
                            find_new_name=find_new_name,
                            total_size=total_size,
                            alignment=alignment)
        return self.add_access(name, debuginfo)

    def add_stream(
        self,
        name,
        dtype,
        buffer_size=1,
        shape=(1, ),
        storage=dtypes.StorageType.Default,
        transient=False,
        offset=None,
        lifetime=dtypes.AllocationLifetime.Scope,
        debuginfo=None,
    ):
        """ :note: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_stream" API is deprecated, please '
            'use "SDFG.add_stream" and "SDFGState.add_access"', DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.sdfg._arrays:
            del self.sdfg._arrays[name]
        self.sdfg.add_stream(
            name,
            dtype,
            buffer_size,
            shape,
            storage,
            transient,
            offset,
            lifetime,
            debuginfo,
        )
        return self.add_access(name, debuginfo)

    def add_scalar(
        self,
        name,
        dtype,
        storage=dtypes.StorageType.Default,
        transient=False,
        lifetime=dtypes.AllocationLifetime.Scope,
        debuginfo=None,
    ):
        """ :note: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_scalar" API is deprecated, please '
            'use "SDFG.add_scalar" and "SDFGState.add_access"', DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.sdfg._arrays:
            del self.sdfg._arrays[name]
        self.sdfg.add_scalar(name, dtype, storage, transient, lifetime, debuginfo)
        return self.add_access(name, debuginfo)

    def add_transient(self,
                      name,
                      shape,
                      dtype,
                      storage=dtypes.StorageType.Default,
                      strides=None,
                      offset=None,
                      lifetime=dtypes.AllocationLifetime.Scope,
                      debuginfo=None,
                      total_size=None,
                      alignment=0):
        """ :note: This function is deprecated. """
        return self.add_array(name,
                              shape,
                              dtype,
                              storage=storage,
                              transient=True,
                              strides=strides,
                              offset=offset,
                              lifetime=lifetime,
                              debuginfo=debuginfo,
                              total_size=total_size,
                              alignment=alignment)

    def fill_scope_connectors(self):
        """ Creates new "IN_%d" and "OUT_%d" connectors on each scope entry
            and exit, depending on array names. """
        for nid, node in enumerate(self.nodes()):
            ####################################################
            # Add connectors to scope entries
            if isinstance(node, nd.EntryNode):
                # Find current number of input connectors
                num_inputs = len(
                    [e for e in self.in_edges(node) if e.dst_conn is not None and e.dst_conn.startswith("IN_")])

                conn_to_data = {}

                # Append input connectors and get mapping of connectors to data
                for edge in self.in_edges(node):
                    if edge.data.data in conn_to_data:
                        raise NotImplementedError(
                            f"Cannot fill scope connectors in SDFGState {self.label} because EntryNode {node.label} "
                            f"has multiple input edges from data {edge.data.data}.")
                    # We're only interested in edges without connectors
                    if edge.dst_conn is not None or edge.data.data is None:
                        continue
                    edge._dst_conn = "IN_" + str(num_inputs + 1)
                    node.add_in_connector(edge.dst_conn)
                    conn_to_data[edge.data.data] = num_inputs + 1

                    num_inputs += 1

                # Set the corresponding output connectors
                for edge in self.out_edges(node):
                    if edge.src_conn is not None:
                        continue
                    if edge.data.data is None:
                        continue
                    edge._src_conn = "OUT_" + str(conn_to_data[edge.data.data])
                    node.add_out_connector(edge.src_conn)
            ####################################################
            # Same treatment for scope exits
            if isinstance(node, nd.ExitNode):
                # Find current number of output connectors
                num_outputs = len(
                    [e for e in self.out_edges(node) if e.src_conn is not None and e.src_conn.startswith("OUT_")])

                conn_to_data = {}

                # Append output connectors and get mapping of connectors to data
                for edge in self.out_edges(node):
                    if edge.src_conn is not None and edge.src_conn.startswith("OUT_"):
                        conn_to_data[edge.data.data] = edge.src_conn[4:]

                    # We're only interested in edges without connectors
                    if edge.src_conn is not None or edge.data.data is None:
                        continue
                    edge._src_conn = "OUT_" + str(num_outputs + 1)
                    node.add_out_connector(edge.src_conn)
                    conn_to_data[edge.data.data] = num_outputs + 1

                    num_outputs += 1

                # Set the corresponding input connectors
                for edge in self.in_edges(node):
                    if edge.dst_conn is not None:
                        continue
                    if edge.data.data is None:
                        continue
                    edge._dst_conn = "IN_" + str(conn_to_data[edge.data.data])
                    node.add_in_connector(edge.dst_conn)

    def expand_library_node(self, node: nd.LibraryNode, implementation: str, **expansion_kwargs) -> str:
        """
        Expand a library node with a specific implementation.

        This is a convenience method that provides a clean interface for expanding
        library nodes from the state level. It automatically handles validation
        and calls the library node's expand method.

        :param node: The library node to expand
        :param implementation: The implementation to use for expansion
        :param expansion_kwargs: Additional keyword arguments for expansion
        :return: The name of the expanded implementation

        Example:
            result = state.expand_library_node(gemm_node, 'MKL')
        """
        # Check that the node is actually in this state
        if node not in self.nodes():
            raise ValueError(f"Node {node} is not in this state")

        # Check that implementation exists
        if implementation not in node.implementations:
            raise KeyError(f"Unknown implementation for node {type(node).__name__}: {implementation}")

        # Use the new expand interface
        return node.expand(self, implementation, **expansion_kwargs)


@make_properties
class ContinueBlock(ControlFlowBlock):
    """ Special control flow block to represent a continue inside of loops. """

    def __repr__(self):
        return f'ContinueBlock ({self.label})'

    def to_json(self, parent=None):
        tmp = super().to_json(parent)
        tmp['nodes'] = []
        tmp['edges'] = []
        return tmp


@make_properties
class BreakBlock(ControlFlowBlock):
    """ Special control flow block to represent a continue inside of loops or switch / select blocks. """

    def __repr__(self):
        return f'BreakBlock ({self.label})'

    def to_json(self, parent=None):
        tmp = super().to_json(parent)
        tmp['nodes'] = []
        tmp['edges'] = []
        return tmp


@make_properties
class ReturnBlock(ControlFlowBlock):
    """ Special control flow block to represent an early return out of the SDFG or a nested procedure / SDFG. """

    def __repr__(self):
        return f'ReturnBlock ({self.label})'

    def to_json(self, parent=None):
        tmp = super().to_json(parent)
        tmp['nodes'] = []
        tmp['edges'] = []
        return tmp


class StateSubgraphView(SubgraphView, DataflowGraphView):
    """ A read-only subgraph view of an SDFG state. """

    def __init__(self, graph, subgraph_nodes):
        super().__init__(graph, subgraph_nodes)

    @property
    def sdfg(self) -> 'SDFG':
        state: SDFGState = self.graph
        return state.sdfg


@make_properties
class AbstractControlFlowRegion(OrderedDiGraph[ControlFlowBlock, 'dace.sdfg.InterstateEdge'], ControlGraphView,
                                ControlFlowBlock, abc.ABC):
    """
    Abstract superclass to represent all kinds of control flow regions in an SDFG.
    This is consequently one of the three main classes of control flow graph nodes, which include ``ControlFlowBlock``s,
    ``SDFGState``s, and nested ``AbstractControlFlowRegion``s. An ``AbstractControlFlowRegion`` can further be either a
    region that directly contains a control flow graph (``ControlFlowRegion``s and subclasses thereof), or something
    that acts like and has the same utilities as a control flow region, including the same API, but is itself not
    directly a single graph. An example of this is the ``ConditionalBlock``, which acts as a single control flow region
    to the outside, but contains multiple actual graphs (one per branch). As such, there are very few but important
    differences between the subclasses of ``AbstractControlFlowRegion``s, such as how traversals are performed, how many
    start blocks there are, etc.
    """

    def __init__(self,
                 label: str = '',
                 sdfg: Optional['SDFG'] = None,
                 parent: Optional['AbstractControlFlowRegion'] = None):
        OrderedDiGraph.__init__(self)
        ControlGraphView.__init__(self)
        ControlFlowBlock.__init__(self, label, sdfg, parent)

        self._labels: Set[str] = set()
        self._start_block: Optional[int] = None
        self._cached_start_block: Optional[ControlFlowBlock] = None
        self._cfg_list: List['ControlFlowRegion'] = [self]

    def get_meta_codeblocks(self) -> List[CodeBlock]:
        """
        Get a list of codeblocks used by the control flow region.
        This may include things such as loop control statements or conditions for branching etc.
        """
        return []

    def get_meta_read_memlets(self) -> List[mm.Memlet]:
        """
        Get read memlets used by the control flow region itself, such as in condition checks for conditional blocks, or
        in loop conditions for loops etc.
        """
        return []

    def replace_meta_accesses(self, replacements: Dict[str, str]) -> None:
        """
        Replace accesses to specific data containers in reads or writes performed by the control flow region itself in
        meta accesses, such as in condition checks for conditional blocks or in loop conditions for loops, etc.

        :param replacements: A dictionary mapping the current data container names to the names of data containers with
                             which accesses to them should be replaced.
        """
        pass

    @property
    def root_sdfg(self) -> 'SDFG':
        from dace.sdfg.sdfg import SDFG  # Avoid import loop
        if not isinstance(self.cfg_list[0], SDFG):
            raise RuntimeError('Root CFG is not of type SDFG')
        return self.cfg_list[0]

    def reset_cfg_list(self) -> List['AbstractControlFlowRegion']:
        """
        Reset the CFG list when changes have been made to the SDFG's CFG tree.
        This collects all control flow graphs recursively and propagates the collection to all CFGs as the new CFG list.

        :return: The newly updated CFG list.
        """
        if isinstance(self, dace.SDFG) and self.parent_sdfg is not None:
            return self.parent_sdfg.reset_cfg_list()
        elif self._parent_graph is not None:
            return self._parent_graph.reset_cfg_list()
        else:
            # Propagate new CFG list to all children
            all_cfgs = list(self.all_control_flow_regions(recursive=True))
            for g in all_cfgs:
                g._cfg_list = all_cfgs
        return self._cfg_list

    def update_cfg_list(self, cfg_list):
        """
        Given a collection of CFGs, add them all to the current SDFG's CFG list.
        Any CFGs already in the list are skipped, and the newly updated list is propagated across all CFGs in the CFG
        tree.

        :param cfg_list: The collection of CFGs to add to the CFG list.
        """
        # TODO: Refactor
        sub_cfg_list = self._cfg_list
        for g in cfg_list:
            if g not in sub_cfg_list:
                sub_cfg_list.append(g)
        ptarget = None
        if isinstance(self, dace.SDFG) and self.parent_sdfg is not None:
            ptarget = self.parent_sdfg
        elif self._parent_graph is not None:
            ptarget = self._parent_graph
        if ptarget is not None:
            ptarget.update_cfg_list(sub_cfg_list)
            self._cfg_list = ptarget.cfg_list
            for g in sub_cfg_list:
                g._cfg_list = self._cfg_list
        else:
            self._cfg_list = sub_cfg_list

    def state(self, state_id: int) -> SDFGState:
        node = self.node(state_id)
        if not isinstance(node, SDFGState):
            raise TypeError(f'The node with id {state_id} is not an SDFGState')
        return node

    def inline(self, lower_returns: bool = False) -> Tuple[bool, Any]:
        """
        Inlines the control flow region into its parent control flow region (if it exists).

        :param lower_returns: Whether or not to remove explicit return blocks when inlining where possible. Defaults to
                              False.
        :return: True if the inlining succeeded, false otherwise.
        """
        parent = self.parent_graph
        if parent:

            # Add all region states and make sure to keep track of all the ones that need to be connected in the end.
            to_connect: Set[ControlFlowBlock] = set()
            ends_context: Set[ControlFlowBlock] = set()
            block_to_state_map: Dict[ControlFlowBlock, SDFGState] = dict()
            for node in self.nodes():
                node.label = self.label + '_' + node.label
                if isinstance(node, ReturnBlock) and lower_returns and isinstance(parent, dace.SDFG):
                    # If a return block is being inlined into an SDFG, convert it into a regular state. Otherwise it
                    # remains as-is.
                    newnode = parent.add_state(node.label)
                    block_to_state_map[node] = newnode
                    if self.out_degree(node) == 0:
                        to_connect.add(newnode)
                        ends_context.add(newnode)
                else:
                    parent.add_node(node, ensure_unique_name=True)
                    if self.out_degree(node) == 0:
                        to_connect.add(node)
                        if isinstance(node, (BreakBlock, ContinueBlock, ReturnBlock)):
                            ends_context.add(node)

            # Add all region edges.
            for edge in self.edges():
                src = block_to_state_map[edge.src] if edge.src in block_to_state_map else edge.src
                dst = block_to_state_map[edge.dst] if edge.dst in block_to_state_map else edge.dst
                parent.add_edge(src, dst, edge.data)

            # Redirect all edges to the region to the internal start state.
            for b_edge in parent.in_edges(self):
                parent.add_edge(b_edge.src, self.start_block, b_edge.data)
                parent.remove_edge(b_edge)

            end_state = None
            if len(to_connect) > 0:
                end_state = parent.add_state(self.label + '_end')
                # Redirect all edges exiting the region to instead exit the end state.
                for a_edge in parent.out_edges(self):
                    parent.add_edge(end_state, a_edge.dst, a_edge.data)
                    parent.remove_edge(a_edge)

                for node in to_connect:
                    if node in ends_context:
                        parent.add_edge(node, end_state, dace.InterstateEdge(condition='False'))
                    else:
                        parent.add_edge(node, end_state, dace.InterstateEdge())
            # Remove the original control flow region (self) from the parent graph.
            parent.remove_node(self)

            sdfg = parent if isinstance(parent, dace.SDFG) else parent.sdfg
            sdfg.reset_cfg_list()

            return True, end_state

        return False, None

    def new_symbols(self, symbols: Dict[str, dtypes.typeclass]) -> Dict[str, dtypes.typeclass]:
        """
        Returns a mapping between the symbol defined by this control flow region and its type, if it exists.
        """
        return {}

    ###################################################################
    # CFG API methods

    def add_return(self, label=None) -> ReturnBlock:
        label = self._ensure_unique_block_name(label)
        block = ReturnBlock(label)
        self._labels.add(label)
        self.add_node(block)
        return block

    def add_edge(self, src: ControlFlowBlock, dst: ControlFlowBlock, data: 'dace.sdfg.InterstateEdge'):
        """ Adds a new edge to the graph. Must be an InterstateEdge or a subclass thereof.

            :param u: Source node.
            :param v: Destination node.
            :param edge: The edge to add.
        """
        if not isinstance(src, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(src)))
        if not isinstance(dst, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(dst)))
        if not isinstance(data, dace.sdfg.InterstateEdge):
            raise TypeError('Expected InterstateEdge, got ' + str(type(data)))
        if dst is self._cached_start_block:
            self._cached_start_block = None
        return super().add_edge(src, dst, data)

    def _ensure_unique_block_name(self, proposed: Optional[str] = None) -> str:
        if self._labels is None or len(self._labels) != self.number_of_nodes():
            self._labels = set(s.label for s in self.nodes())
        return dt.find_new_name(proposed or 'block', self._labels)

    def add_node(self,
                 node,
                 is_start_block: bool = False,
                 ensure_unique_name: bool = False,
                 *,
                 is_start_state: bool = None):
        if not isinstance(node, ControlFlowBlock):
            raise TypeError('Expected ControlFlowBlock, got ' + str(type(node)))

        if ensure_unique_name:
            node.label = self._ensure_unique_block_name(node.label)

        super().add_node(node)
        self._cached_start_block = None
        node.parent_graph = self
        if isinstance(self, dace.SDFG):
            sdfg = self
        else:
            sdfg = self.sdfg
        node.sdfg = sdfg
        if isinstance(node, AbstractControlFlowRegion):
            for n in node.all_control_flow_blocks():
                n.sdfg = self.sdfg
        start_block = is_start_block
        if is_start_state is not None:
            warnings.warn('is_start_state is deprecated, use is_start_block instead', DeprecationWarning)
            start_block = is_start_state

        if start_block:
            self.start_block = len(self.nodes()) - 1
            self._cached_start_block = node

    def add_state(self, label=None, is_start_block=False, *, is_start_state: Optional[bool] = None) -> SDFGState:
        label = self._ensure_unique_block_name(label)
        state = SDFGState(label)
        self._labels.add(label)
        start_block = is_start_block
        if is_start_state is not None:
            warnings.warn('is_start_state is deprecated, use is_start_block instead', DeprecationWarning)
            start_block = is_start_state
        self.add_node(state, is_start_block=start_block)
        return state

    def add_state_before(self,
                         state: SDFGState,
                         label=None,
                         is_start_block=False,
                         condition: Optional[CodeBlock] = None,
                         assignments: Optional[Dict] = None,
                         *,
                         is_start_state: Optional[bool] = None) -> SDFGState:
        """ Adds a new SDFG state before an existing state, reconnecting predecessors to it instead.

            :param state: The state to prepend the new state before.
            :param label: State label.
            :param is_start_block: If True, resets scope block starting state to this state.
            :param condition: Transition condition of the newly created edge between state and the new state.
            :param assignments: Assignments to perform upon transition.
            :return: A new SDFGState object.
        """
        new_state = self.add_state(label, is_start_block=is_start_block, is_start_state=is_start_state)
        # Reconnect
        for e in self.in_edges(state):
            self.remove_edge(e)
            self.add_edge(e.src, new_state, e.data)
        # Add the new edge
        self.add_edge(new_state, state, dace.sdfg.InterstateEdge(condition=condition, assignments=assignments))
        return new_state

    def add_state_after(self,
                        state: SDFGState,
                        label=None,
                        is_start_block=False,
                        condition: Optional[CodeBlock] = None,
                        assignments: Optional[Dict] = None,
                        *,
                        is_start_state: Optional[bool] = None) -> SDFGState:
        """ Adds a new SDFG state after an existing state, reconnecting it to the successors instead.

            :param state: The state to append the new state after.
            :param label: State label.
            :param is_start_block: If True, resets scope block starting state to this state.
            :param condition: Transition condition of the newly created edge between state and the new state.
            :param assignments: Assignments to perform upon transition.
            :return: A new SDFGState object.
        """
        new_state = self.add_state(label, is_start_block=is_start_block, is_start_state=is_start_state)
        # Reconnect
        for e in self.out_edges(state):
            self.remove_edge(e)
            self.add_edge(new_state, e.dst, e.data)
        # Add the new edge
        self.add_edge(state, new_state, dace.sdfg.InterstateEdge(condition=condition, assignments=assignments))
        return new_state

    ###################################################################
    # Traversal methods

    def all_control_flow_regions(self,
                                 recursive=False,
                                 load_ext=False,
                                 parent_first=True) -> Iterator['AbstractControlFlowRegion']:
        """ Iterate over this and all nested control flow regions. """
        if parent_first:
            yield self
        for block in self.nodes():
            if isinstance(block, SDFGState) and recursive:
                for node in block.nodes():
                    if isinstance(node, nd.NestedSDFG):
                        if node.sdfg:
                            yield from node.sdfg.all_control_flow_regions(recursive=recursive,
                                                                          load_ext=load_ext,
                                                                          parent_first=parent_first)
                        elif load_ext:
                            node.load_external(block)
                            yield from node.sdfg.all_control_flow_regions(recursive=recursive,
                                                                          load_ext=load_ext,
                                                                          parent_first=parent_first)
            elif isinstance(block, AbstractControlFlowRegion):
                yield from block.all_control_flow_regions(recursive=recursive,
                                                          load_ext=load_ext,
                                                          parent_first=parent_first)
        if not parent_first:
            yield self

    def all_sdfgs_recursive(self, load_ext=False) -> Iterator['SDFG']:
        """ Iterate over this and all nested SDFGs. """
        for cfg in self.all_control_flow_regions(recursive=True, load_ext=load_ext):
            if isinstance(cfg, dace.SDFG):
                yield cfg

    def all_states(self) -> Iterator[SDFGState]:
        """ Iterate over all states in this control flow graph. """
        for block in self.nodes():
            if isinstance(block, SDFGState):
                yield block
            elif isinstance(block, AbstractControlFlowRegion):
                yield from block.all_states()

    def all_control_flow_blocks(self, recursive=False) -> Iterator[ControlFlowBlock]:
        """ Iterate over all control flow blocks in this control flow graph. """
        for cfg in self.all_control_flow_regions(recursive=recursive):
            for block in cfg.nodes():
                yield block

    def all_interstate_edges(self, recursive=False) -> Iterator[Edge['dace.sdfg.InterstateEdge']]:
        """ Iterate over all interstate edges in this control flow graph. """
        for cfg in self.all_control_flow_regions(recursive=recursive):
            for edge in cfg.edges():
                yield edge

    ###################################################################
    # Inherited / Overrides

    def _used_symbols_internal(self,
                               all_symbols: bool,
                               defined_syms: Optional[Set] = None,
                               free_syms: Optional[Set] = None,
                               used_before_assignment: Optional[Set] = None,
                               keep_defined_in_mapping: bool = False,
                               with_contents: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        defined_syms = set() if defined_syms is None else defined_syms
        free_syms = set() if free_syms is None else free_syms
        used_before_assignment = set() if used_before_assignment is None else used_before_assignment

        if with_contents:
            try:
                ordered_blocks = self.bfs_nodes(self.start_block)
            except ValueError:  # Failsafe (e.g., for invalid or empty SDFGs)
                ordered_blocks = self.nodes()

            for block in ordered_blocks:
                state_symbols = set()
                if isinstance(block, (ControlFlowRegion, ConditionalBlock)):
                    b_free_syms, b_defined_syms, b_used_before_syms = block._used_symbols_internal(
                        all_symbols, defined_syms, free_syms, used_before_assignment, keep_defined_in_mapping,
                        with_contents)
                    free_syms |= b_free_syms
                    defined_syms |= b_defined_syms
                    used_before_assignment |= b_used_before_syms
                    state_symbols = b_free_syms
                else:
                    state_symbols = block.used_symbols(all_symbols, keep_defined_in_mapping, with_contents)
                    free_syms |= state_symbols

                # Add free inter-state symbols
                for e in self.out_edges(block):
                    # NOTE: First we get the true InterstateEdge free symbols, then we compute the newly defined symbols
                    # by subracting the (true) free symbols from the edge's assignment keys. This way we can correctly
                    # compute the symbols that are used before being assigned.
                    efsyms = e.data.used_symbols(all_symbols)
                    # collect symbols representing data containers
                    dsyms = {sym for sym in efsyms if sym in self.sdfg.arrays}
                    for d in dsyms:
                        efsyms |= {str(sym) for sym in self.sdfg.arrays[d].used_symbols(all_symbols)}
                    defined_syms |= set(e.data.assignments.keys()) - (efsyms | state_symbols)
                    used_before_assignment.update(efsyms - defined_syms)
                    free_syms |= efsyms

        # Remove symbols that were used before they were assigned.
        defined_syms -= used_before_assignment

        if isinstance(self, dace.SDFG):
            # Remove from defined symbols those that are in the symbol mapping
            if self.parent_nsdfg_node is not None and keep_defined_in_mapping:
                defined_syms -= set(self.parent_nsdfg_node.symbol_mapping.keys())

            # Add the set of SDFG symbol parameters
            # If all_symbols is False, those symbols would only be added in the case of non-Python tasklets
            if all_symbols:
                free_syms |= set(self.symbols.keys())

        # Subtract symbols defined in inter-state edges and constants from the list of free symbols.
        free_syms -= defined_syms

        return free_syms, defined_syms, used_before_assignment

    def to_json(self, parent=None):
        graph_json = OrderedDiGraph.to_json(self)
        block_json = ControlFlowBlock.to_json(self, parent)
        graph_json.update(block_json)

        graph_json['cfg_list_id'] = int(self.cfg_id)
        graph_json['start_block'] = self._start_block

        return graph_json

    @classmethod
    def from_json(cls, json_obj, context=None):
        context = context or {'sdfg': None, 'parent_graph': None}
        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError("Class type mismatch")

        nodes = json_obj['nodes']
        edges = json_obj['edges']

        ret = cls(label=json_obj['label'], sdfg=context['sdfg'])

        dace.serialize.set_properties_from_json(ret, json_obj)

        nodelist = []
        for n in nodes:
            nci = copy.copy(context)
            nci['parent_graph'] = ret

            block = dace.serialize.from_json(n, context=nci)
            ret.add_node(block)
            nodelist.append(block)

        for e in edges:
            e = dace.serialize.from_json(e)
            ret.add_edge(nodelist[int(e.src)], nodelist[int(e.dst)], e.data)

        if 'start_block' in json_obj:
            ret._start_block = json_obj['start_block']

        return ret

    ###################################################################
    # Getters, setters, and builtins

    def __str__(self):
        return ControlFlowBlock.__str__(self)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} ({self.label})'

    @property
    def cfg_list(self) -> List['ControlFlowRegion']:
        return self._cfg_list

    @property
    def cfg_id(self) -> int:
        """
        Returns the unique index of the current CFG within the current tree of CFGs (Top-level CFG/SDFG is 0, nested
        CFGs/SDFGs are greater).
        """
        return self.cfg_list.index(self)

    @property
    def start_block(self):
        """ Returns the starting block of this ControlFlowGraph. """
        if self._cached_start_block is not None:
            return self._cached_start_block

        source_nodes = self.source_nodes()
        if len(source_nodes) == 1:
            self._cached_start_block = source_nodes[0]
            return source_nodes[0]
        # If the starting block is ambiguous allow manual override.
        if self._start_block is not None:
            self._cached_start_block = self.node(self._start_block)
            return self._cached_start_block
        raise ValueError('Ambiguous or undefined starting block for ControlFlowGraph, '
                         'please use "is_start_block=True" when adding the '
                         'starting block with "add_state" or "add_node"')

    @start_block.setter
    def start_block(self, block_id):
        """ Manually sets the starting block of this ControlFlowGraph.

            :param block_id: The node ID (use `node_id(block)`) of the block to set.
        """
        if block_id < 0 or block_id >= self.number_of_nodes():
            raise ValueError('Invalid state ID')
        self._start_block = block_id
        self._cached_start_block = None


@make_properties
class ControlFlowRegion(AbstractControlFlowRegion):
    """
    A ``ControlFlowRegion`` represents a control flow graph node that itself contains a control flow graph.
    This can be an arbitrary control flow graph, but may also be a specific type of control flow region with additional
    semantics, such as a loop or a function call.
    """

    def __init__(self, label='', sdfg=None, parent=None):
        super().__init__(label, sdfg, parent)


@make_properties
class LoopRegion(ControlFlowRegion):
    """
    A control flow region that represents a loop.

    Like in traditional programming languages, a loop has a condition that is checked before each iteration.
    It may have zero or more initialization statements that are executed before the first loop iteration, and zero or
    more update statements that are executed after each iteration. For example, a loop with only a condition and neither
    an initialization nor an update statement is equivalent to a while loop, while a loop with initialization and update
    statements represents a for loop. Loops may additionally be inverted, meaning that the condition is checked after
    the first iteration instead of before.

    A loop region, like any other control flow region, has a single distinct entry / start block, and one or more
    exit blocks. Exit blocks are blocks that have no outgoing edges or only conditional outgoing edges. Whenever an
    exit block finshes executing, one iteration of the loop is completed.

    Loops may have an arbitrary number of break states. Whenever a break state finishes executing, the loop is exited
    immediately. A loop may additionally have an arbitrary number of continue states. Whenever a continue state finishes
    executing, the next iteration of the loop is started immediately (with execution of the update statement(s), if
    present).
    """

    update_statement = CodeProperty(serialize_if=lambda ustmnt: ustmnt is not None,
                                    allow_none=True,
                                    default=None,
                                    desc='The loop update statement. May be None if the update happens elsewhere.')
    init_statement = CodeProperty(serialize_if=lambda istmnt: istmnt is not None,
                                  allow_none=True,
                                  default=None,
                                  desc='The loop init statement. May be None if the initialization happens elsewhere.')
    loop_condition = CodeProperty(allow_none=True, default=None, desc='The loop condition')
    inverted = Property(dtype=bool,
                        default=False,
                        desc='If True, the loop condition is checked after the first iteration.')
    update_before_condition = Property(dtype=bool,
                                       default=True,
                                       desc='If False, the loop condition is checked before the update statement is' +
                                       ' executed. This only applies to inverted loops, turning them from a typical ' +
                                       'do-while style into a while(true) with a break before the update (at the end ' +
                                       'of an iteration) if the condition no longer holds.')
    loop_variable = Property(dtype=str, default='', desc='The loop variable, if given')

    def __init__(self,
                 label: str,
                 condition_expr: Optional[Union[str, CodeBlock]] = None,
                 loop_var: Optional[str] = None,
                 initialize_expr: Optional[Union[str, CodeBlock]] = None,
                 update_expr: Optional[Union[str, CodeBlock]] = None,
                 inverted: bool = False,
                 sdfg: Optional['SDFG'] = None,
                 update_before_condition=True):
        super(LoopRegion, self).__init__(label, sdfg)

        if initialize_expr is not None:
            if isinstance(initialize_expr, CodeBlock):
                self.init_statement = initialize_expr
            else:
                self.init_statement = CodeBlock(initialize_expr)
        else:
            self.init_statement = None

        if condition_expr:
            if isinstance(condition_expr, CodeBlock):
                self.loop_condition = condition_expr
            else:
                self.loop_condition = CodeBlock(condition_expr)
        else:
            self.loop_condition = CodeBlock('True')

        if update_expr is not None:
            if isinstance(update_expr, CodeBlock):
                self.update_statement = update_expr
            else:
                self.update_statement = CodeBlock(update_expr)
        else:
            self.update_statement = None

        self.loop_variable = loop_var or ''
        self.inverted = inverted
        self.update_before_condition = update_before_condition

    def inline(self, lower_returns: bool = False) -> Tuple[bool, Any]:
        """
        Inlines the loop region into its parent control flow region.

        :param lower_returns: Whether or not to remove explicit return blocks when inlining where possible. Defaults to
                              False.
        :return: True if the inlining succeeded, false otherwise.
        """
        parent = self.parent_graph
        if not parent:
            raise RuntimeError('No top-level SDFG present to inline into')

        # Avoid circular imports
        from dace.frontend.python import astutils

        # Check that the loop initialization and update statements each only contain assignments, if the loop has any.
        if self.init_statement is not None:
            if isinstance(self.init_statement.code, list):
                for stmt in self.init_statement.code:
                    if not isinstance(stmt, astutils.ast.Assign):
                        return False, None
        if self.update_statement is not None:
            if isinstance(self.update_statement.code, list):
                for stmt in self.update_statement.code:
                    if not isinstance(stmt, astutils.ast.Assign):
                        return False, None

        # First recursively inline any other contained control flow regions other than loops to ensure break, continue,
        # and return are inlined correctly.
        def recursive_inline_cf_regions(region: ControlFlowRegion) -> None:
            for block in region.nodes():
                if ((isinstance(block, ControlFlowRegion) or isinstance(block, ConditionalBlock))
                        and not isinstance(block, LoopRegion)):
                    recursive_inline_cf_regions(block)
                    block.inline(lower_returns=lower_returns)

        recursive_inline_cf_regions(self)

        # Add all boilerplate loop states necessary for the structure.
        init_state = parent.add_state(self.label + '_init')
        guard_state = parent.add_state(self.label + '_guard')
        end_state = parent.add_state(self.label + '_end')
        loop_latch_state = parent.add_state(self.label + '_latch')

        # Add all loop states and make sure to keep track of all the ones that need to be connected in the end.
        # Return blocks are inlined as-is. If the parent graph is an SDFG, they are converted to states, otherwise
        # they are left as explicit exit blocks.
        connect_to_latch: Set[SDFGState] = set()
        connect_to_end: Set[SDFGState] = set()
        block_to_state_map: Dict[ControlFlowBlock, SDFGState] = dict()
        for node in self.nodes():
            node.label = self.label + '_' + node.label
            if isinstance(node, BreakBlock):
                newnode = parent.add_state(node.label)
                connect_to_end.add(newnode)
                block_to_state_map[node] = newnode
            elif isinstance(node, ContinueBlock):
                newnode = parent.add_state(node.label)
                connect_to_latch.add(newnode)
                block_to_state_map[node] = newnode
            elif isinstance(node, ReturnBlock) and isinstance(parent, dace.SDFG):
                newnode = parent.add_state(node.label)
                block_to_state_map[node] = newnode
            else:
                if self.out_degree(node) == 0:
                    connect_to_latch.add(node)
                parent.add_node(node, ensure_unique_name=True)

        # Add all internal loop edges.
        for edge in self.edges():
            src = block_to_state_map[edge.src] if edge.src in block_to_state_map else edge.src
            dst = block_to_state_map[edge.dst] if edge.dst in block_to_state_map else edge.dst
            parent.add_edge(src, dst, edge.data)

        # Redirect all edges to the loop to the init state.
        for b_edge in parent.in_edges(self):
            parent.add_edge(b_edge.src, init_state, b_edge.data)
            parent.remove_edge(b_edge)
        # Redirect all edges exiting the loop to instead exit the end state.
        for a_edge in parent.out_edges(self):
            parent.add_edge(end_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        # Add an initialization edge that initializes the loop variable if applicable.
        init_edge = dace.InterstateEdge()
        if self.init_statement is not None:
            init_edge.assignments = {}
            for stmt in self.init_statement.code:
                assign: astutils.ast.Assign = stmt
                init_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
        if self.inverted:
            parent.add_edge(init_state, self.start_block, init_edge)
        else:
            parent.add_edge(init_state, guard_state, init_edge)

        # Connect the loop tail.
        update_edge = dace.InterstateEdge()
        if self.update_statement is not None:
            update_edge.assignments = {}
            for stmt in self.update_statement.code:
                assign: astutils.ast.Assign = stmt
                update_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
        parent.add_edge(loop_latch_state, guard_state, update_edge)

        # Add condition checking edges and connect the guard state.
        cond_expr = self.loop_condition.code
        parent.add_edge(guard_state, end_state, dace.InterstateEdge(CodeBlock(astutils.negate_expr(cond_expr)).code))
        parent.add_edge(guard_state, self.start_block, dace.InterstateEdge(CodeBlock(cond_expr).code))

        # Connect any end states from the loop's internal state machine to the tail state so they end a
        # loop iteration. Do the same for any continue states, and connect any break states to the end of the loop.
        for node in connect_to_latch:
            parent.add_edge(node, loop_latch_state, dace.InterstateEdge())
        for node in connect_to_end:
            parent.add_edge(node, end_state, dace.InterstateEdge())

        parent.remove_node(self)

        sdfg = parent if isinstance(parent, dace.SDFG) else parent.sdfg
        sdfg.reset_cfg_list()

        return True, (init_state, guard_state, end_state)

    def can_normalize(self) -> Tuple[bool, bool]:
        """
        Checks if the loop region can be normalized, meaning that it starts from 0 and increments by 1.

        :return: A tuple of two booleans indicating if the loop init and step can be normalized.
        """

        # Avoid cyclic import
        from dace.transformation.passes.analysis import loop_analysis

        # If loop information cannot be determined, we cannot normalize
        start = loop_analysis.get_init_assignment(self)
        step = loop_analysis.get_loop_stride(self)
        itervar = self.loop_variable
        if start is None or step is None or itervar is None:
            return False, False

        # If we cannot symbolically match the loop condition, we cannot normalize
        condition = symbolic.pystr_to_symbolic(self.loop_condition.as_string)
        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sympy.Wild('a')
        if (condition.match(itersym < a) is None and condition.match(itersym <= a) is None
                and condition.match(itersym > a) is None and condition.match(itersym >= a) is None):
            return False, False

        # Get a list of all defined symbols in the loop body
        defined_syms = set()
        for edge, _ in self.all_edges_recursive():
            if isinstance(edge.data, dace.InterstateEdge):
                defined_syms.update(edge.data.assignments.keys())

        # Check if we can normalize loop init
        # Iteration variable not altered in the loop body and Init is not zero
        can_norm_init = (itervar not in defined_syms and symbolic.resolve_symbol_to_constant(start, self.sdfg) != 0)

        # Check if we can normalize loop step
        # Iteration variable not altered in the loop body, increment not altered in body, step does not contain iteration variable, and Step is not one
        step_free_syms = set([str(s) for s in step.free_symbols])
        can_norm_step = (itervar not in defined_syms and step_free_syms.isdisjoint(defined_syms)
                         and step_free_syms.isdisjoint({itervar})
                         and symbolic.resolve_symbol_to_constant(step, self.sdfg) != 1)

        # Return the results
        return can_norm_init, can_norm_step

    def normalize(self) -> bool:
        """
        Normalizes the loop region, meaning that it starts from 0 and increments by 1, if possible.
        Partially normalizes if only one of the two is possible.

        :return: True if the loop was normalized, False otherwise.
        """

        # Avoid cyclic import
        from dace.transformation.passes.analysis import loop_analysis

        # Check if the loop can be normalized
        norm_init, norm_step = self.can_normalize()
        if not (norm_init or norm_step):
            return False

        start = loop_analysis.get_init_assignment(self)
        step = loop_analysis.get_loop_stride(self)
        itervar = self.loop_variable

        # Create the conversion expression
        if norm_init and norm_step:
            val = f"{itervar} * {step} + {start}"
        elif norm_init:
            val = f"{itervar} + {start}"
        elif norm_step:
            val = f"{itervar} * {step}"

        # Replace each occurrence of the old iteration variable with the new one in the loop body, but not in the loop header
        new_iter = self.sdfg.find_new_symbol(f"{itervar}_norm")
        old_loop_init = copy.deepcopy(self.init_statement)
        old_loop_cond = copy.deepcopy(self.loop_condition)
        old_loop_step = copy.deepcopy(self.update_statement)

        self.replace_dict({itervar: new_iter}, replace_keys=False)
        self.init_statement = old_loop_init
        self.loop_condition = old_loop_cond
        self.update_statement = old_loop_step

        # Add new state before the loop to compute the new iteration symbol
        start_state = self.start_block
        self.add_state_before(start_state, is_start_block=True, assignments={new_iter: val})

        # Adjust loop header
        if norm_init:
            self.init_statement = CodeBlock(f"{itervar} = 0")
        if norm_step:
            self.update_statement = CodeBlock(f"{itervar} = {itervar} + 1")

        # Compute new condition
        condition = symbolic.pystr_to_symbolic(self.loop_condition.as_string)
        itersym = symbolic.pystr_to_symbolic(itervar)

        # Find condition by matching expressions
        end: Optional[sympy.Expr] = None
        a = sympy.Wild('a')
        op = ''
        match = condition.match(itersym < a)
        if match:
            op = '<'
            end = match[a]
        if end is None:
            match = condition.match(itersym <= a)
            if match:
                op = '<='
                end = match[a]
        if end is None:
            match = condition.match(itersym > a)
            if match:
                op = '>'
                end = match[a]
        if end is None:
            match = condition.match(itersym >= a)
            if match:
                op = '>='
                end = match[a]
        if len(op) == 0:
            raise ValueError('Cannot match loop condition for loop normalization')

        # Invert the operator for reverse loops
        is_reverse = step < 0
        if is_reverse:
            if op == '<':
                op = '>='
            elif op == '<=':
                op = '>'
            elif op == '>':
                op = '<='
            elif op == '>=':
                op = '<'

            # swap start and end
            start, end = end, start

            # negate step
            step = -step

        if norm_init and norm_step:
            new_condition = f"{itersym} {op} (({end}) - ({start})) / {step}"
        elif norm_init:
            new_condition = f"{itersym} {op} ({end}) - ({start})"
        elif norm_step:
            new_condition = f"{itersym} {op} ({end}) / {step}"

        if is_reverse:
            new_condition = f"{new_condition} + 1"

        self.loop_condition = CodeBlock(new_condition)
        return True

    def get_meta_codeblocks(self):
        codes = [self.loop_condition]
        if self.init_statement:
            codes.append(self.init_statement)
        if self.update_statement:
            codes.append(self.update_statement)
        return codes

    def get_meta_read_memlets(self) -> List[mm.Memlet]:
        # Avoid cyclic imports.
        from dace.sdfg.sdfg import memlets_in_ast
        read_memlets = memlets_in_ast(self.loop_condition.code[0], self.sdfg.arrays)
        if self.init_statement:
            read_memlets.extend(memlets_in_ast(self.init_statement.code[0], self.sdfg.arrays))
        if self.update_statement:
            read_memlets.extend(memlets_in_ast(self.update_statement.code[0], self.sdfg.arrays))
        return read_memlets

    def replace_meta_accesses(self, replacements):
        if self.loop_variable in replacements:
            self.loop_variable = replacements[self.loop_variable]
        replace_in_codeblock(self.loop_condition, replacements)
        if self.init_statement:
            replace_in_codeblock(self.init_statement, replacements)
        if self.update_statement:
            replace_in_codeblock(self.update_statement, replacements)

    def _used_symbols_internal(self,
                               all_symbols: bool,
                               defined_syms: Optional[Set] = None,
                               free_syms: Optional[Set] = None,
                               used_before_assignment: Optional[Set] = None,
                               keep_defined_in_mapping: bool = False,
                               with_contents: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        defined_syms = set() if defined_syms is None else defined_syms
        free_syms = set() if free_syms is None else free_syms
        used_before_assignment = set() if used_before_assignment is None else used_before_assignment

        defined_syms.add(self.loop_variable)
        if self.init_statement is not None:
            free_syms |= self.init_statement.get_free_symbols()
        if self.update_statement is not None:
            free_syms |= self.update_statement.get_free_symbols()
        cond_free_syms = self.loop_condition.get_free_symbols()
        if self.loop_variable and self.loop_variable in cond_free_syms:
            cond_free_syms.remove(self.loop_variable)

        b_free_symbols, b_defined_symbols, b_used_before_assignment = super()._used_symbols_internal(
            all_symbols, keep_defined_in_mapping=keep_defined_in_mapping, with_contents=with_contents)
        outside_defined = defined_syms - used_before_assignment
        used_before_assignment |= ((b_used_before_assignment - {self.loop_variable}) - outside_defined)
        free_syms |= b_free_symbols
        defined_syms |= b_defined_symbols

        defined_syms -= used_before_assignment
        free_syms -= defined_syms
        free_syms |= cond_free_syms

        return free_syms, defined_syms, used_before_assignment

    def new_symbols(self, symbols) -> Dict[str, dtypes.typeclass]:
        # Avoid cyclic import
        from dace.codegen.tools.type_inference import infer_expr_type
        from dace.transformation.passes.analysis import loop_analysis

        if self.init_statement and self.loop_variable:
            alltypes = copy.copy(symbols)
            alltypes.update({k: v.dtype for k, v in self.sdfg.arrays.items()})
            l_end = loop_analysis.get_loop_end(self)
            l_start = loop_analysis.get_init_assignment(self)
            l_step = loop_analysis.get_loop_stride(self)
            inferred_type = dtypes.result_type_of(infer_expr_type(l_start, alltypes), infer_expr_type(l_step, alltypes),
                                                  infer_expr_type(l_end, alltypes))
            init_rhs = loop_analysis.get_init_assignment(self)
            if self.loop_variable not in symbolic.free_symbols_and_functions(init_rhs):
                return {self.loop_variable: inferred_type}
        return {}

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None,
                     replace_in_graph: bool = True,
                     replace_keys: bool = True):
        if replace_keys:
            if self.loop_variable and self.loop_variable in repl:
                self.loop_variable = repl[self.loop_variable]

        from dace.sdfg.replace import replace_properties_dict
        replace_properties_dict(self, repl, symrepl)

        super().replace_dict(repl, symrepl, replace_in_graph)

    def add_break(self, label=None) -> BreakBlock:
        label = self._ensure_unique_block_name(label)
        block = BreakBlock(label)
        self._labels.add(label)
        self.add_node(block)
        return block

    def add_continue(self, label=None) -> ContinueBlock:
        label = self._ensure_unique_block_name(label)
        block = ContinueBlock(label)
        self._labels.add(label)
        self.add_node(block)
        return block

    @property
    def has_continue(self) -> bool:
        for node, _ in self.all_nodes_recursive(lambda n, _: not isinstance(n, (LoopRegion, SDFGState))):
            if isinstance(node, ContinueBlock):
                return True
        return False

    @property
    def has_break(self) -> bool:
        for node, _ in self.all_nodes_recursive(lambda n, _: not isinstance(n, (LoopRegion, SDFGState))):
            if isinstance(node, BreakBlock):
                return True
        return False

    @property
    def has_return(self) -> bool:
        for node, _ in self.all_nodes_recursive(lambda n, _: not isinstance(n, (LoopRegion, SDFGState))):
            if isinstance(node, ReturnBlock):
                return True
        return False


@make_properties
class ConditionalBlock(AbstractControlFlowRegion):

    _branches: List[Tuple[Optional[CodeBlock], ControlFlowRegion]]

    def __init__(self, label: str = '', sdfg: Optional['SDFG'] = None, parent: Optional['ControlFlowRegion'] = None):
        super().__init__(label, sdfg, parent)
        self._branches = []

    def sub_regions(self):
        return [b for _, b in self.branches]

    def replace_meta_accesses(self, replacements):
        for c, _ in self.branches:
            if c is not None:
                replace_in_codeblock(c, replacements)

    def __str__(self):
        return self._label

    def __repr__(self) -> str:
        return f'ConditionalBlock ({self.label})'

    @property
    def branches(self) -> List[Tuple[Optional[CodeBlock], ControlFlowRegion]]:
        return self._branches

    def add_branch(self, condition: Optional[Union[CodeBlock, str]], branch: ControlFlowRegion):
        if condition is not None and not isinstance(condition, CodeBlock):
            condition = CodeBlock(condition)
        self._branches.append([condition, branch])
        branch.parent_graph = self
        branch.sdfg = self.sdfg

    def remove_branch(self, branch: ControlFlowRegion):
        self._branches = [(c, b) for c, b in self._branches if b is not branch]

    def get_meta_codeblocks(self):
        codes = []
        for c, _ in self.branches:
            if c is not None:
                codes.append(c)
        return codes

    def get_meta_read_memlets(self) -> List[mm.Memlet]:
        # Avoid cyclic imports.
        from dace.sdfg.sdfg import memlets_in_ast
        read_memlets = []
        for c, _ in self.branches:
            if c is not None:
                read_memlets.extend(memlets_in_ast(c.code[0], self.sdfg.arrays))
        return read_memlets

    def _used_symbols_internal(self,
                               all_symbols: bool,
                               defined_syms: Optional[Set] = None,
                               free_syms: Optional[Set] = None,
                               used_before_assignment: Optional[Set] = None,
                               keep_defined_in_mapping: bool = False,
                               with_contents: bool = True) -> Tuple[Set[str], Set[str], Set[str]]:
        defined_syms = set() if defined_syms is None else defined_syms
        free_syms = set() if free_syms is None else free_syms
        used_before_assignment = set() if used_before_assignment is None else used_before_assignment

        for condition, region in self._branches:
            if condition is not None:
                free_syms |= condition.get_free_symbols(defined_syms)
            b_free_symbols, b_defined_symbols, b_used_before_assignment = region._used_symbols_internal(
                all_symbols, defined_syms, free_syms, used_before_assignment, keep_defined_in_mapping, with_contents)
            free_syms |= b_free_symbols
            defined_syms |= b_defined_symbols
            used_before_assignment |= b_used_before_assignment

        defined_syms -= used_before_assignment
        free_syms -= defined_syms

        return free_syms, defined_syms, used_before_assignment

    def replace_dict(self,
                     repl: Dict[str, str],
                     symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None,
                     replace_in_graph: bool = True,
                     replace_keys: bool = True):
        # Avoid circular imports
        from dace.sdfg.replace import replace_in_codeblock

        if replace_keys:
            from dace.sdfg.replace import replace_properties_dict
            replace_properties_dict(self, repl, symrepl)

        for cond, region in self._branches:
            region.replace_dict(repl, symrepl, replace_in_graph)
            if cond is not None:
                replace_in_codeblock(cond, repl)

    def to_json(self, parent=None):
        json = super().to_json(parent)
        json['branches'] = [(condition.to_json() if condition is not None else None, cfg.to_json())
                            for condition, cfg in self._branches]
        return json

    @classmethod
    def from_json(cls, json_obj, context=None):
        context = context or {'sdfg': None, 'parent_graph': None}
        _type = json_obj['type']
        if _type != cls.__name__:
            raise TypeError('Class type mismatch')

        ret = cls(label=json_obj['label'], sdfg=context['sdfg'])

        dace.serialize.set_properties_from_json(ret, json_obj)

        for condition, region in json_obj['branches']:
            if condition is not None:
                ret.add_branch(CodeBlock.from_json(condition), ControlFlowRegion.from_json(region, context))
            else:
                ret.add_branch(None, ControlFlowRegion.from_json(region, context))
        return ret

    def inline(self, lower_returns: bool = False) -> Tuple[bool, Any]:
        """
        Inlines the conditional region into its parent control flow region.

        :param lower_returns: Whether or not to remove explicit return blocks when inlining where possible. Defaults to
                              False.
        :return: True if the inlining succeeded, false otherwise.
        """
        parent = self.parent_graph
        if not parent:
            raise RuntimeError('No top-level SDFG present to inline into')

        # Add all boilerplate states necessary for the structure.
        guard_state = parent.add_state(self.label + '_guard')
        end_state = parent.add_state(self.label + '_end')

        # Redirect all edges to the region to the init state.
        for b_edge in parent.in_edges(self):
            parent.add_edge(b_edge.src, guard_state, b_edge.data)
            parent.remove_edge(b_edge)
        # Redirect all edges exiting the region to instead exit the end state.
        for a_edge in parent.out_edges(self):
            parent.add_edge(end_state, a_edge.dst, a_edge.data)
            parent.remove_edge(a_edge)

        from dace.sdfg.sdfg import InterstateEdge
        else_branch = None
        full_cond_expression: Optional[List[ast.AST]] = None
        for condition, region in self._branches:
            if condition is None:
                else_branch = region
            else:
                if full_cond_expression is None:
                    full_cond_expression = condition.code[0]
                else:
                    full_cond_expression = astutils.and_expr(full_cond_expression, condition.code[0])
                parent.add_node(region)
                parent.add_edge(guard_state, region, InterstateEdge(condition=condition))
                parent.add_edge(region, end_state, InterstateEdge())
                region.inline(lower_returns=lower_returns)
        if full_cond_expression is not None:
            negative_full_cond = astutils.negate_expr(full_cond_expression)
            negative_cond = CodeBlock([negative_full_cond])
        else:
            negative_cond = CodeBlock('1')

        if else_branch is not None:
            parent.add_node(else_branch)
            parent.add_edge(guard_state, else_branch, InterstateEdge(condition=negative_cond))
            parent.add_edge(else_branch, end_state, InterstateEdge())
            else_branch.inline(lower_returns=lower_returns)
        else:
            parent.add_edge(guard_state, end_state, InterstateEdge(condition=negative_cond))

        parent.remove_node(self)

        sdfg = parent if isinstance(parent, dace.SDFG) else parent.sdfg
        sdfg.reset_cfg_list()

        return True, (guard_state, end_state)

    # Abstract control flow region overrides

    @property
    def start_block(self):
        return None

    @start_block.setter
    def start_block(self, _):
        pass

    # Graph API overrides.

    def node_id(self, node: 'ControlFlowBlock') -> int:
        try:
            return next(i for i, (_, b) in enumerate(self._branches) if b is node)
        except StopIteration:
            raise NodeNotFoundError(node)

    def nodes(self) -> List['ControlFlowBlock']:
        return [node for _, node in self._branches]

    def number_of_nodes(self):
        return len(self._branches)

    def edges(self) -> List[Edge['dace.sdfg.InterstateEdge']]:
        return []

    def in_edges(self, _: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        return []

    def out_edges(self, _: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        return []

    def all_edges(self, _: 'ControlFlowBlock') -> List[Edge['dace.sdfg.InterstateEdge']]:
        return []


@make_properties
class UnstructuredControlFlow(ControlFlowRegion):
    """ Special control flow region to represent a region of unstructured control flow. """

    def __repr__(self):
        return f'UnstructuredCF ({self.label})'


@make_properties
class NamedRegion(ControlFlowRegion):

    debuginfo = DebugInfoProperty()

    def __init__(self, label: str, sdfg: Optional['SDFG'] = None, debuginfo: Optional[dtypes.DebugInfo] = None):
        super().__init__(label, sdfg)
        self.debuginfo = debuginfo


@make_properties
class FunctionCallRegion(NamedRegion):

    arguments = DictProperty(str, str)

    def __init__(self,
                 label: str,
                 arguments: Dict[str, str] = {},
                 sdfg: 'SDFG' = None,
                 debuginfo: Optional[dtypes.DebugInfo] = None):
        super().__init__(label, sdfg, debuginfo)
        self.arguments = arguments
