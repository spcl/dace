# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes of a single SDFG state and dataflow subgraphs. """

import collections
import copy
from dace import (data as dt, dtypes, memlet as mm, serialize, subsets as sbs,
                  symbolic)
from dace.sdfg import nodes as nd
from dace.sdfg.graph import (OrderedMultiDiConnectorGraph, MultiConnectorEdge,
                             SubgraphView)
from dace.sdfg.propagation import propagate_memlet
from dace.sdfg.validation import validate_state
from dace.properties import (Property, DictProperty, SubsetProperty,
                             SymbolicProperty, CodeBlock, make_properties)
from inspect import getframeinfo, stack
import itertools
from typing import Any, Dict, Optional, List, Set, Tuple, Union
import warnings


def _getdebuginfo(old_dinfo=None) -> dtypes.DebugInfo:
    """ Returns a DebugInfo object for the position that called this function.
        :param old_dinfo: Another DebugInfo object that will override the
                          return value of this function
        :return: DebugInfo containing line number and calling file.
    """
    if old_dinfo is not None:
        return old_dinfo

    caller = getframeinfo(stack()[2][0])
    return dtypes.DebugInfo(caller.lineno, 0, caller.lineno, 0, caller.filename)


class StateGraphView(object):
    """
    Read-only view interface of an SDFG state, containing methods for memlet
    tracking, traversal, subgraph creation, queries, and replacements.
    ``SDFGState`` and ``StateSubgraphView`` inherit from this class to share
    methods.
    """
    def __init__(self, *args, **kwargs):
        self._clear_scopedict_cache()

    ###################################################################
    # Traversal methods

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

    def data_nodes(self):
        """ Returns all data_nodes (arrays) present in this state. """
        return [n for n in self.nodes() if isinstance(n, nd.AccessNode)]

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

    ###################################################################
    # Memlet-tracking methods

    def memlet_path(self, edge: MultiConnectorEdge) -> List[MultiConnectorEdge]:
        """ Given one edge, returns a list of edges representing a path
            between its source and sink nodes. Used for memlet tracking.

            @note: Behavior is undefined when there is more than one path
                   involving this edge.
            :param edge: An edge within this state.
            :return: A list of edges from a source node to a destination node.
            """
        result = [edge]

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # If empty memlet, return itself as the path
        if (edge.src_conn is None and edge.dst_conn is None
                and edge.data.is_empty()):
            return result

        # Prepend incoming edges until reaching the source node
        curedge = edge
        while not isinstance(curedge.src, (nd.CodeNode, nd.AccessNode)):
            # Trace through scopes using OUT_# -> IN_#
            if isinstance(curedge.src, (nd.EntryNode, nd.ExitNode)):
                if curedge.src_conn is None:
                    raise ValueError(
                        "Source connector cannot be None for {}".format(
                            curedge.src))
                assert curedge.src_conn.startswith("OUT_")
                next_edge = next(e for e in state.in_edges(curedge.src)
                                 if e.dst_conn == "IN_" + curedge.src_conn[4:])
                result.insert(0, next_edge)
                curedge = next_edge

        # Prepend outgoing edges until reaching the sink node
        curedge = edge
        while not isinstance(curedge.dst, (nd.CodeNode, nd.AccessNode)):
            # Trace through scope entry using IN_# -> OUT_#
            if isinstance(curedge.dst, (nd.EntryNode, nd.ExitNode)):
                if curedge.dst_conn is None:
                    raise ValueError(
                        "Destination connector cannot be None for {}".format(
                            curedge.dst))
                if not curedge.dst_conn.startswith("IN_"):  # Map variable
                    break
                next_edge = next(e for e in state.out_edges(curedge.dst)
                                 if e.src_conn == "OUT_" + curedge.dst_conn[3:])
                result.append(next_edge)
                curedge = next_edge

        return result

    def memlet_tree(self, edge: MultiConnectorEdge) -> mm.MemletTree:
        """ Given one edge, returns a tree of edges between its node source(s)
            and sink(s). Used for memlet tracking.

            :param edge: An edge within this state.
            :return: A tree of edges whose root is the source/sink node
                     (depending on direction) and associated children edges.
            """
        propagate_forward = False
        propagate_backward = False
        if ((isinstance(edge.src, nd.EntryNode) and edge.src_conn is not None)
                or
            (isinstance(edge.dst, nd.EntryNode) and edge.dst_conn is not None
             and edge.dst_conn.startswith('IN_'))):
            propagate_forward = True
        if ((isinstance(edge.src, nd.ExitNode) and edge.src_conn is not None) or
            (isinstance(edge.dst, nd.ExitNode) and edge.dst_conn is not None)):
            propagate_backward = True

        # If either both are False (no scopes involved) or both are True
        # (invalid SDFG), we return only the current edge as a degenerate tree
        if propagate_forward == propagate_backward:
            return mm.MemletTree(edge)

        # Obtain the full state (to work with paths that trace beyond a scope)
        state = self._graph

        # Find tree root
        curedge = edge
        if propagate_forward:
            while (isinstance(curedge.src, nd.EntryNode)
                   and curedge.src_conn is not None):
                assert curedge.src_conn.startswith('OUT_')
                cname = curedge.src_conn[4:]
                curedge = next(e for e in state.in_edges(curedge.src)
                               if e.dst_conn == 'IN_%s' % cname)
        elif propagate_backward:
            while (isinstance(curedge.dst, nd.ExitNode)
                   and curedge.dst_conn is not None):
                assert curedge.dst_conn.startswith('IN_')
                cname = curedge.dst_conn[3:]
                curedge = next(e for e in state.out_edges(curedge.dst)
                               if e.src_conn == 'OUT_%s' % cname)
        tree_root = mm.MemletTree(curedge)

        # Collect children (recursively)
        def add_children(treenode):
            if propagate_forward:
                if not (isinstance(treenode.edge.dst, nd.EntryNode)
                        and treenode.edge.dst_conn
                        and treenode.edge.dst_conn.startswith('IN_')):
                    return
                conn = treenode.edge.dst_conn[3:]
                treenode.children = [
                    mm.MemletTree(e, parent=treenode)
                    for e in state.out_edges(treenode.edge.dst)
                    if e.src_conn == 'OUT_%s' % conn
                ]
            elif propagate_backward:
                if (not isinstance(treenode.edge.src, nd.ExitNode)
                        or treenode.edge.src_conn is None):
                    return
                conn = treenode.edge.src_conn[4:]
                treenode.children = [
                    mm.MemletTree(e, parent=treenode)
                    for e in state.in_edges(treenode.edge.src)
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

    def scope_tree(self) -> 'dace.sdfg.scope.ScopeTree':
        from dace.sdfg.scope import ScopeTree

        if (hasattr(self, '_scope_tree_cached')
                and self._scope_tree_cached is not None):
            return copy.copy(self._scope_tree_cached)

        sdp = self.scope_dict(node_to_children=False)
        sdc = self.scope_dict(node_to_children=True)

        result = {}

        sdfg_symbols = self.parent.symbols.keys()

        # Get scopes
        for node, scopenodes in sdc.items():
            if node is None:
                exit_node = None
            else:
                exit_node = next(v for v in scopenodes
                                 if isinstance(v, nd.ExitNode))
            scope = ScopeTree(node, exit_node)
            scope.defined_vars = set(
                symbolic.pystr_to_symbolic(s)
                for s in (self.symbols_defined_at(node).keys()
                          | sdfg_symbols))
            result[node] = scope

        # Scope parents and children
        for node, scope in result.items():
            if node is not None:
                scope.parent = result[sdp[node]]
            scope.children = [
                result[n] for n in sdc[node] if isinstance(n, nd.EntryNode)
            ]

        self._scope_tree_cached = result

        return copy.copy(self._scope_tree_cached)

    def scope_leaves(self) -> List['dace.sdfg.scope.ScopeTree']:
        if (hasattr(self, '_scope_leaves_cached')
                and self._scope_leaves_cached is not None):
            return copy.copy(self._scope_leaves_cached)
        st = self.scope_tree()
        self._scope_leaves_cached = [
            scope for scope in st.values() if len(scope.children) == 0
        ]
        return copy.copy(self._scope_leaves_cached)

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
        from dace.sdfg.scope import _scope_dict_inner, _scope_dict_to_ids
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
            if validate and len(eq) != 0:
                raise RuntimeError("Leftover nodes in queue: {}".format(eq))

            # Cache result
            if node_to_children:
                self._scope_dict_tochildren_cached = result
            else:
                self._scope_dict_toparent_cached = result

            result = copy.copy(result)

        if return_ids:
            return _scope_dict_to_ids(self, result)
        return result

    ###################################################################
    # Query, subgraph, and replacement methods

    @property
    def free_symbols(self) -> Set[str]:
        """
        Returns a set of symbol names that are used, but not defined, in
        this graph view (SDFG state or subgraph thereof).
        :note: Assumes that the graph is valid (i.e., without undefined or
               overlapping symbols).
        """
        sdfg = self.parent
        new_symbols = set()
        freesyms = set()

        # Free symbols from nodes
        for n in self.nodes():
            if isinstance(n, nd.EntryNode):
                new_symbols |= set(n.new_symbols(sdfg, self, {}).keys())
            elif isinstance(n, nd.AccessNode):
                # Add data descriptor symbols
                freesyms |= set(map(str, n.desc(sdfg).free_symbols))

            freesyms |= n.free_symbols

        # Free symbols from memlets
        for e in self.edges():
            freesyms |= e.data.free_symbols

        # Do not consider SDFG constants as symbols
        new_symbols.update(set(sdfg.constants.keys()))

        return freesyms - new_symbols

    def defined_symbols(self) -> Dict[str, dt.Data]:
        """
        Returns a dictionary that maps currently-defined symbols in this SDFG
        state or subgraph to their types.
        """
        state = self.graph if isinstance(self, SubgraphView) else self
        sdfg = self.parent

        # Start with SDFG global symbols
        defined_syms = {k: v for k, v in sdfg.symbols.items()}

        # Add data-descriptor free symbols
        for desc in sdfg.arrays.values():
            for sym in desc.free_symbols:
                defined_syms[str(sym)] = sym.dtype

        # Add inter-state symbols
        # NOTE: A DFS such as in validate_sdfg can be invoked here, but may
        #       be time consuming.
        for edge in sdfg.edges():
            defined_syms.update(edge.data.new_symbols(defined_syms))

        # Add scope symbols all the way to the subgraph
        sdict = self.scope_dict()
        scope_nodes = []
        for source_node in self.source_nodes():
            curnode = source_node
            while sdict[curnode] is not None:
                curnode = sdict[curnode]
                scope_nodes.append(curnode)

        for snode in dtypes.deduplicate(list(reversed(scope_nodes))):
            defined_syms.update(snode.new_symbols(defined_syms))

        return defined_syms

    def arglist(self) -> Dict[str, dt.Data]:
        """
        Returns an ordered dictionary of arguments (names and types) required
        to invoke this SDFG state or subgraph thereof.

        The arguments differ from SDFG.arglist, but follow the same order,
        namely: <sorted data arguments>, <sorted scalar arguments>.

        Data arguments contain:
            * All used non-transient data containers in the subgraph
            * All used transient data containers that were allocated outside.
              This includes data from memlets, transients shared across multiple
              states, and transients that could not be allocated within the
              subgraph (due to their ``AllocationLifetime`` or according to the
              ``dtypes.can_allocate`` function).

        Scalar arguments contain:
            * Free symbols in this state/subgraph.
            * All transient and non-transient scalar data containers used in
              this subgraph.

        This structure will create a sorted list of pointers followed by a
        sorted list of PoDs and structs.

        :return: An ordered dictionary of (name, data descriptor type) of all
                 the arguments, sorted as defined here.
        """
        sdfg: 'dace.sdfg.SDFG' = self.parent
        shared_transients = sdfg.shared_transients()
        sdict = self.scope_dict()

        data_args = {}
        scalar_args = {}

        # Gather data descriptors from nodes
        descs = {}
        for node in self.nodes():
            if isinstance(node, nd.AccessNode):
                descs[node.data] = node.desc(sdfg)

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

        # Add data arguments from memlets, if do not appear in any of the nodes
        # (i.e., originate externally)
        for edge in self.edges():
            if edge.data.data is not None and edge.data.data not in descs:
                desc = sdfg.arrays[edge.data.data]
                if isinstance(desc, dt.Scalar):
                    # Ignore code->code edges.
                    if (isinstance(edge.src, nd.CodeNode)
                            and isinstance(edge.dst, nd.CodeNode)):
                        continue

                    scalar_args[edge.data.data] = desc
                else:
                    data_args[edge.data.data] = desc

        # Loop over locally-used data descriptors
        for name, desc in descs.items():
            if name in data_args or name in scalar_args:
                continue
            # If scalar, always add
            if isinstance(desc, dt.Scalar):
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
                  and desc.lifetime not in (dtypes.AllocationLifetime.Scope,
                                            dtypes.AllocationLifetime.State)):
                data_args[name] = desc
            #   2. If a subgraph, State also applies
            elif isinstance(self, SubgraphView):
                if (desc.lifetime != dtypes.AllocationLifetime.Scope):
                    data_args[name] = desc
            # Check for allocation constraints that would
            # enforce array to be allocated outside subgraph
            elif desc.lifetime == dtypes.AllocationLifetime.Scope:
                curnode = sdict[node]
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
        defined_syms = self.defined_symbols()
        scalar_args.update({
            k:
            dt.Scalar(defined_syms[k]) if k in defined_syms else sdfg.arrays[k]
            for k in self.free_symbols
            if not k.startswith('__dace') and k not in sdfg.constants
        })

        # Add scalar arguments from free symbols of data descriptors
        for arg in data_args.values():
            scalar_args.update({
                str(k): dt.Scalar(k.dtype)
                for k in arg.free_symbols if not str(k).startswith('__dace')
                and str(k) not in sdfg.constants
            })

        # Fill up ordered dictionary
        result = collections.OrderedDict()
        for k, v in itertools.chain(sorted(data_args.items()),
                                    sorted(scalar_args.items())):
            result[k] = v

        return result

    def signature_arglist(self, with_types=True, for_call=False):
        """ Returns a list of arguments necessary to call this state or
            subgraph, formatted as a list of C definitions.
            :param with_types: If True, includes argument types in the result.
            :param for_call: If True, returns arguments that can be used when
                             calling the SDFG.
            :return: A list of strings. For example: `['float *A', 'int b']`.
        """
        return [
            v.as_arg(name=k, with_types=with_types, for_call=for_call)
            for k, v in self.arglist().items()
        ]

    def scope_subgraph(self, entry_node, include_entry=True, include_exit=True):
        from dace.sdfg.scope import _scope_subgraph
        return _scope_subgraph(self, entry_node, include_entry, include_exit)

    def top_level_transients(self):
        """Iterate over top-level transients of this state."""
        sdict = self.scope_dict(node_to_children=True)
        sdfg = self.parent
        result = set()
        for node in sdict[None]:
            if isinstance(node, nd.AccessNode) and node.desc(sdfg).transient:
                result.add(node.data)
        return result

    def all_transients(self) -> List[str]:
        """Iterate over all transients in this state."""
        return dtypes.deduplicate([
            n.data for n in self.nodes()
            if isinstance(n, nd.AccessNode) and n.desc(self.parent).transient
        ])

    def replace(self, name: str, new_name: str):
        """ Finds and replaces all occurrences of a symbol or array in this
            state.
            :param name: Name to find.
            :param new_name: Name to replace.
        """
        from dace.sdfg.sdfg import replace
        replace(self, name, new_name)


@make_properties
class SDFGState(OrderedMultiDiConnectorGraph, StateGraphView):
    """ An acyclic dataflow multigraph in an SDFG, corresponding to a
        single state in the SDFG state machine. """

    is_collapsed = Property(dtype=bool,
                            desc="Show this node/scope/state as collapsed",
                            default=False)

    nosync = Property(dtype=bool,
                      default=False,
                      desc="Do not synchronize at the end of the state")

    instrument = Property(choices=dtypes.InstrumentationType,
                          desc="Measure execution statistics with given method",
                          default=dtypes.InstrumentationType.No_Instrumentation)

    location = DictProperty(
        key_type=str,
        value_type=symbolic.pystr_to_symbolic,
        desc='Full storage location identifier (e.g., rank, GPU ID)')

    def __init__(self, label=None, sdfg=None, debuginfo=None, location=None):
        """ Constructs an SDFG state.
            :param label: Name for the state (optional).
            :param sdfg: A reference to the parent SDFG.
            :param debuginfo: Source code locator for debugging.
        """
        super(SDFGState, self).__init__()
        self._label = label
        self._parent = sdfg
        self._graph = self  # Allowing MemletTrackingView mixin to work
        self._clear_scopedict_cache()
        self._debuginfo = debuginfo
        self.is_collapsed = False
        self.nosync = False
        self.location = location if location is not None else {}
        self._default_lineinfo = None

    @property
    def parent(self):
        """ Returns the parent SDFG of this state. """
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    def __str__(self):
        return self._label

    @property
    def label(self):
        return self._label

    @property
    def name(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def is_empty(self):
        return self.number_of_nodes() == 0

    def validate(self) -> None:
        validate_state(self)

    def set_default_lineinfo(self, lineinfo: dtypes.DebugInfo):
        """
        Sets the default source line information to be lineinfo, or None to
        revert to default mode. 
        """
        self._default_lineinfo = lineinfo

    def nodes(self) -> List[nd.Node]:  # Added for type hints
        return super().nodes()

    def all_edges_and_connectors(self, *nodes):
        """
        Returns an iterable to incoming and outgoing Edge objects, along
        with their connector types.
        """
        for node in nodes:
            for e in self.in_edges(node):
                yield e, (node.in_connectors[e.dst_conn]
                          if e.dst_conn else None)
            for e in self.out_edges(node):
                yield e, (node.out_connectors[e.src_conn]
                          if e.src_conn else None)

    def add_node(self, node):
        if not isinstance(node, nd.Node):
            raise TypeError("Expected Node, got " + str(type(node)) + " (" +
                            str(node) + ")")
        self._clear_scopedict_cache()
        return super(SDFGState, self).add_node(node)

    def remove_node(self, node):
        self._clear_scopedict_cache()
        super(SDFGState, self).remove_node(node)

    def add_edge(self, u, u_connector, v, v_connector, memlet):
        if not isinstance(u, nd.Node):
            raise TypeError("Source node is not of type nd.Node (type: %s)" %
                            str(type(u)))
        if u_connector is not None and not isinstance(u_connector, str):
            raise TypeError("Source connector is not string (type: %s)" %
                            str(type(u_connector)))
        if not isinstance(v, nd.Node):
            raise TypeError("Destination node is not of type nd.Node (type: " +
                            "%s)" % str(type(v)))
        if v_connector is not None and not isinstance(v_connector, str):
            raise TypeError("Destination connector is not string (type: %s)" %
                            str(type(v_connector)))
        if not isinstance(memlet, mm.Memlet):
            raise TypeError("Memlet is not of type Memlet (type: %s)" %
                            str(type(memlet)))

        self._clear_scopedict_cache()
        result = super(SDFGState, self).add_edge(u, u_connector, v, v_connector,
                                                 memlet)
        memlet.try_initialize(self.parent, self, result)
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
            scope_dict = {
                k: sorted(v)
                for k, v in sorted(
                    self.scope_dict(node_to_children=True,
                                    return_ids=True).items())
            }
        except RuntimeError:
            scope_dict = {}

        # Try to initialize edges before serialization
        for edge in self.edges():
            edge.data.try_initialize(self.parent, self, edge)

        ret = {
            'type':
            type(self).__name__,
            'label':
            self.name,
            'id':
            parent.node_id(self) if parent is not None else None,
            'collapsed':
            self.is_collapsed,
            'scope_dict':
            scope_dict,
            'nodes': [n.to_json(self) for n in self.nodes()],
            'edges': [
                e.to_json(self) for e in sorted(
                    self.edges(),
                    key=lambda e: (e.src_conn or '', e.dst_conn or ''))
            ],
            'attributes':
            serialize.all_properties_to_json(self),
        }

        return ret

    @classmethod
    def from_json(cls, json_obj, context={'sdfg': None}):
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

        ret = SDFGState(label=json_obj['label'],
                        sdfg=context['sdfg'],
                        debuginfo=None)

        rec_ci = {
            'sdfg': context['sdfg'],
            'sdfg_state': ret,
            'callback': context['callback'] if 'callback' in context else None
        }
        serialize.set_properties_from_json(ret, json_obj, rec_ci)

        for n in nodes:
            nret = serialize.loads(serialize.dumps(n), context=rec_ci)
            ret.add_node(nret)

        # Connect using the edges
        for e in edges:
            eret = serialize.loads(serialize.dumps(e), context=rec_ci)

            ret.add_edge(eret.src, eret.src_conn, eret.dst, eret.dst_conn,
                         eret.data)

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
        sdfg._arrays = {k: self._parent.arrays[k] for k in arrays}
        sdfg.add_node(self)

        return sdfg._repr_html_()

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

        sdfg: SDFG = self.parent

        # Start with global symbols
        symbols = collections.OrderedDict(sdfg.symbols)
        for desc in sdfg.arrays.values():
            symbols.update([(str(s), s.dtype) for s in desc.free_symbols])

        # Add symbols from inter-state edges along the path to the state
        try:
            start_state = sdfg.start_state
            for path in sdfg.all_simple_paths(start_state, self, as_edges=True):
                for e in path:
                    symbols.update(e.data.new_symbols(symbols))
        except ValueError:
            # Cannot determine starting state (possibly some inter-state edges
            # do not yet exist)
            for e in sdfg.edges():
                symbols.update(e.data.new_symbols(symbols))

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
    def add_read(self,
                 array_or_stream_name: str,
                 debuginfo=None) -> nd.AccessNode:
        """ Adds a read-only access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        node = nd.AccessNode(array_or_stream_name,
                             dtypes.AccessType.ReadOnly,
                             debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_write(self,
                  array_or_stream_name: str,
                  debuginfo=None) -> nd.AccessNode:
        """ Adds a write-only access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        node = nd.AccessNode(array_or_stream_name,
                             dtypes.AccessType.WriteOnly,
                             debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_access(self,
                   array_or_stream_name: str,
                   debuginfo=None) -> nd.AccessNode:
        """ Adds a general (read/write) access node to this SDFG state.
            :param array_or_stream_name: The name of the array/stream.
            :return: An array access node.
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        node = nd.AccessNode(array_or_stream_name,
                             dtypes.AccessType.ReadWrite,
                             debuginfo=debuginfo)
        self.add_node(node)
        return node

    def add_tasklet(
        self,
        name: str,
        inputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        code: str,
        language: dtypes.Language = dtypes.Language.Python,
        location: dict = None,
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
            location=location,
            debuginfo=debuginfo,
        )
        self.add_node(tasklet)
        return tasklet

    def add_nested_sdfg(
        self,
        sdfg: 'dace.sdfg.SDFG',
        parent,
        inputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dtypes.typeclass]],
        symbol_mapping: Dict[str, Any] = None,
        name=None,
        schedule=dtypes.ScheduleType.Default,
        location=None,
        debuginfo=None,
    ):
        """ Adds a nested SDFG to the SDFG state. """
        if name is None:
            name = sdfg.label
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)

        sdfg.parent = self
        sdfg.parent_sdfg = self.parent

        sdfg.update_sdfg_list([])

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
        )
        self.add_node(s)

        sdfg.parent_nsdfg_node = s

        # Add "default" undefined symbols if None are given
        symbols = sdfg.free_symbols
        if symbol_mapping is None:
            symbol_mapping = {s: s for s in symbols}
            s.symbol_mapping = symbol_mapping

        # Validate missing symbols
        missing_symbols = [s for s in symbols if s not in symbol_mapping]
        if missing_symbols:
            raise ValueError('Missing symbols on nested SDFG "%s": %s' %
                             (name, missing_symbols))

        # Add new global symbols to nested SDFG
        from dace.codegen.tools.type_inference import infer_expr_type
        for sym, symval in s.symbol_mapping.items():
            if sym not in sdfg.symbols:
                # TODO: Think of a better way to avoid calling
                # symbols_defined_at in this moment
                sdfg.add_symbol(
                    sym,
                    infer_expr_type(symval, self.parent.symbols)
                    or dtypes.typeclass(int))

        return s

    def _make_iterators(self, ndrange):
        # Input can either be a dictionary or a list of pairs
        if isinstance(ndrange, list):
            params = [k for k, v in ndrange]
            ndrange = {k: v for k, v in ndrange}
        else:
            params = list(ndrange.keys())
        map_range = SubsetProperty.from_string(", ".join(
            [ndrange[p] for p in params]))
        return params, map_range

    def add_map(
            self,
            name,
            ndrange: Union[Dict[str, str], List[Tuple[str, str]]],
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
        map = nd.Map(name,
                     *self._make_iterators(ndrange),
                     schedule=schedule,
                     unroll=unroll,
                     debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, map_exit])
        return map_entry, map_exit

    def add_consume(
        self,
        name,
        elements: Tuple[str, str],
        condition: str = None,
        schedule=dtypes.ScheduleType.Default,
        chunksize=1,
        debuginfo=None,
        language=dtypes.Language.Python
    ) -> Tuple[nd.ConsumeEntry, nd.ConsumeExit]:
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
        consume = nd.Consume(name,
                             pe_tuple,
                             CodeBlock(condition, language),
                             schedule,
                             chunksize,
                             debuginfo=debuginfo)
        entry = nd.ConsumeEntry(consume)
        exit = nd.ConsumeExit(consume)

        self.add_nodes_from([entry, exit])
        return entry, exit

    def add_mapped_tasklet(
            self,
            name: str,
            map_ranges: Dict[str, sbs.Subset],
            inputs: Dict[str, mm.Memlet],
            code: str,
            outputs: Dict[str, mm.Memlet],
            schedule=dtypes.ScheduleType.Default,
            unroll_map=False,
            location=None,
            language=dtypes.Language.Python,
            debuginfo=None,
            external_edges=False,
            input_nodes: Optional[Dict[str, nd.AccessNode]] = None,
            output_nodes: Optional[Dict[str, nd.AccessNode]] = None,
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
            :param debuginfo:  Debugging information (mostly for DIODE)
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

        tasklet = nd.Tasklet(
            name,
            tinputs,
            toutputs,
            code,
            language=language,
            location=location,
            debuginfo=debuginfo,
        )
        map = nd.Map(map_name,
                     *self._make_iterators(map_ranges),
                     schedule=schedule,
                     unroll=unroll_map,
                     debuginfo=debuginfo)
        map_entry = nd.MapEntry(map)
        map_exit = nd.MapExit(map)
        self.add_nodes_from([map_entry, tasklet, map_exit])

        # Create access nodes
        inpdict = {}
        outdict = {}
        if external_edges:
            input_nodes = input_nodes or {}
            output_nodes = output_nodes or {}
            input_data = set(memlet.data for memlet in inputs.values())
            output_data = set(memlet.data for memlet in outputs.values())
            for inp in input_data:
                if inp in input_nodes:
                    inpdict[inp] = input_nodes[inp]
                else:
                    inpdict[inp] = self.add_read(inp)
            for out in output_data:
                if out in output_nodes:
                    outdict[out] = output_nodes[out]
                else:
                    outdict[out] = self.add_write(out)

        # Connect inputs from map to tasklet
        tomemlet = {}
        for name, memlet in inputs.items():
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            self.add_edge(map_entry, None, tasklet, name, memlet)
            tomemlet[memlet.data] = memlet

        # If there are no inputs, add empty memlet
        if len(inputs) == 0:
            self.add_edge(map_entry, None, tasklet, None, mm.Memlet())

        if external_edges:
            for inp, inpnode in inpdict.items():
                # Add external edge
                if propagate:
                    outer_memlet = propagate_memlet(self, tomemlet[inp],
                                                    map_entry, True)
                else:
                    outer_memlet = tomemlet[inp]
                self.add_edge(inpnode, None, map_entry, "IN_" + inp,
                              outer_memlet)

                # Add connectors to internal edges
                for e in self.out_edges(map_entry):
                    if e.data.data == inp:
                        e._src_conn = "OUT_" + inp

                # Add connectors to map entry
                map_entry.add_in_connector("IN_" + inp)
                map_entry.add_out_connector("OUT_" + inp)

        # Connect outputs from tasklet to map
        tomemlet = {}
        for name, memlet in outputs.items():
            # Set memlet local name
            memlet.name = name
            # Add internal memlet edge
            self.add_edge(tasklet, name, map_exit, None, memlet)
            tomemlet[memlet.data] = memlet

        # If there are no outputs, add empty memlet
        if len(outputs) == 0:
            self.add_edge(tasklet, None, map_exit, None, mm.Memlet())

        if external_edges:
            for out, outnode in outdict.items():
                # Add external edge
                if propagate:
                    outer_memlet = propagate_memlet(self, tomemlet[out],
                                                    map_exit, True)
                else:
                    outer_memlet = tomemlet[out]
                self.add_edge(map_exit, "OUT_" + out, outnode, None,
                              outer_memlet)

                # Add connectors to internal edges
                for e in self.in_edges(map_exit):
                    if e.data.data == out:
                        e._dst_conn = "IN_" + out

                # Add connectors to map entry
                map_exit.add_in_connector("IN_" + out)
                map_exit.add_out_connector("OUT_" + out)

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
        result = stdlib.Reduce(wcr,
                               axes,
                               identity,
                               schedule=schedule,
                               debuginfo=debuginfo)
        self.add_node(result)
        return result

    def add_pipeline(self,
                     name,
                     ndrange,
                     init_size=0,
                     init_overlap=False,
                     drain_size=0,
                     drain_overlap=False,
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
            :return: (map_entry, map_exit) node 2-tuple
        """
        debuginfo = _getdebuginfo(debuginfo or self._default_lineinfo)
        pipeline = nd.Pipeline(name,
                               *self._make_iterators(ndrange),
                               init_size=init_size,
                               init_overlap=init_overlap,
                               drain_size=drain_size,
                               drain_overlap=drain_overlap,
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
            for conn in (scope_node.in_connectors.keys()
                         | scope_node.out_connectors.keys()):
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
            external_memlet = propagate_memlet(self, internal_memlet,
                                               scope_node, True)

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

        return (iedge, eedge)

    def add_memlet_path(self,
                        *path_nodes,
                        memlet=None,
                        src_conn=None,
                        dst_conn=None,
                        propagate=True):
        """ Adds a path of memlet edges between the given nodes, propagating
            from the given innermost memlet.

            :param *path_nodes: Nodes participating in the path (in the given
                                order).
            :keyword memlet: (mandatory) The memlet at the innermost scope
                             (e.g., the incoming memlet to a tasklet (last
                             node), or an outgoing memlet from an array
                             (first node), followed by scope exits).
            :keyword src_conn: Connector at the beginning of the path.
            :keyword dst_conn: Connector at the end of the path.
        """
        if memlet is None:
            raise TypeError("Innermost memlet cannot be None")
        if len(path_nodes) < 2:
            raise ValueError("Memlet path must consist of at least 2 nodes")

        src_node = path_nodes[0]
        dst_node = path_nodes[-1]

        # Add edges first so that scopes can be understood
        edges = [
            self.add_edge(path_nodes[i], None, path_nodes[i + 1], None,
                          mm.Memlet()) for i in range(len(path_nodes) - 1)
        ]

        if not isinstance(memlet, mm.Memlet):
            raise TypeError("Expected Memlet, got: {}".format(
                type(memlet).__name__))

        if any(isinstance(n, nd.EntryNode) for n in path_nodes):
            propagate_forward = False
        else:  # dst node's scope is higher than src node, propagate out
            propagate_forward = True

        # Innermost edge memlet
        cur_memlet = memlet

        # Verify that connectors exist
        if (not memlet.is_empty() and hasattr(edges[0].src, "out_connectors")
                and isinstance(edges[0].src, nd.CodeNode)
                and not isinstance(edges[0].src, nd.LibraryNode) and
            (src_conn is None or src_conn not in edges[0].src.out_connectors)):
            raise ValueError("Output connector {} does not exist in {}".format(
                src_conn, edges[0].src.label))
        if (not memlet.is_empty() and hasattr(edges[-1].dst, "in_connectors")
                and isinstance(edges[-1].dst, nd.CodeNode)
                and not isinstance(edges[-1].dst, nd.LibraryNode) and
            (dst_conn is None or dst_conn not in edges[-1].dst.in_connectors)):
            raise ValueError("Input connector {} does not exist in {}".format(
                dst_conn, edges[-1].dst.label))

        path = edges if propagate_forward else reversed(edges)
        # Propagate and add edges
        for i, edge in enumerate(path):
            # Figure out source and destination connectors
            if propagate_forward:
                sconn = src_conn if i == 0 else ("OUT_" +
                                                 edge.src.last_connector())
                dconn = (dst_conn if i == len(edges) - 1 else
                         ("IN_" + edge.dst.next_connector()))
            else:
                sconn = (src_conn if i == len(edges) - 1 else
                         ("OUT_" + edge.src.next_connector()))
                dconn = dst_conn if i == 0 else ("IN_" +
                                                 edge.dst.last_connector())

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
                        cur_memlet = propagate_memlet(self, cur_memlet, snode,
                                                      True)
        # Try to initialize memlets
        for edge in edges:
            edge.data.try_initialize(self.parent, self, edge)

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
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_array" API is deprecated, please '
            'use "SDFG.add_array" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_array(name,
                              shape,
                              dtype,
                              storage,
                              transient,
                              strides,
                              offset,
                              lifetime,
                              debuginfo,
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
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_stream" API is deprecated, please '
            'use "SDFG.add_stream" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_stream(
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
        """ @attention: This function is deprecated. """
        warnings.warn(
            'The "SDFGState.add_scalar" API is deprecated, please '
            'use "SDFG.add_scalar" and "SDFGState.add_access"',
            DeprecationWarning)
        # Workaround to allow this legacy API
        if name in self.parent._arrays:
            del self.parent._arrays[name]
        self.parent.add_scalar(name, dtype, storage, transient, lifetime,
                               debuginfo)
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
        """ @attention: This function is deprecated. """
        return self.add_array(name,
                              shape,
                              dtype,
                              storage,
                              True,
                              strides,
                              offset,
                              lifetime,
                              debuginfo,
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
                num_inputs = len([
                    e for e in self.in_edges(node)
                    if e.dst_conn is not None and e.dst_conn.startswith("IN_")
                ])

                conn_to_data = {}

                # Append input connectors and get mapping of connectors to data
                for edge in self.in_edges(node):
                    if edge.dst_conn is not None and edge.dst_conn.startswith(
                            "IN_"):
                        conn_to_data[edge.data.data] = edge.dst_conn[3:]

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
                num_outputs = len([
                    e for e in self.out_edges(node)
                    if e.src_conn is not None and e.src_conn.startswith("OUT_")
                ])

                conn_to_data = {}

                # Append output connectors and get mapping of connectors to data
                for edge in self.out_edges(node):
                    if edge.src_conn is not None and edge.src_conn.startswith(
                            "OUT_"):
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


class StateSubgraphView(SubgraphView, StateGraphView):
    """ A read-only subgraph view of an SDFG state. """
    def __init__(self, graph, subgraph_nodes):
        super().__init__(graph, subgraph_nodes)
