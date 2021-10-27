# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformation helper API. """
import copy
import itertools
from networkx import MultiDiGraph

from dace.subsets import Range, Subset, union
import dace.subsets as subsets
from typing import Dict, List, Optional, Tuple, Set, Union

from dace import data, dtypes, symbolic
from dace.sdfg import nodes, utils
from dace.sdfg.graph import SubgraphView, MultiConnectorEdge
from dace.sdfg.scope import ScopeSubgraphView, ScopeTree
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg import graph
from dace.memlet import Memlet


def nest_state_subgraph(sdfg: SDFG,
                        state: SDFGState,
                        subgraph: SubgraphView,
                        name: Optional[str] = None,
                        full_data: bool = False) -> nodes.NestedSDFG:
    """ Turns a state subgraph into a nested SDFG. Operates in-place.
        :param sdfg: The SDFG containing the state subgraph.
        :param state: The state containing the subgraph.
        :param subgraph: Subgraph to nest.
        :param name: An optional name for the nested SDFG.
        :param full_data: If True, nests entire input/output data.
        :return: The nested SDFG node.
        :raise KeyError: Some or all nodes in the subgraph are not located in
                         this state, or the state does not belong to the given
                         SDFG.
        :raise ValueError: The subgraph is contained in more than one scope.
    """
    if state.parent != sdfg:
        raise KeyError('State does not belong to given SDFG')
    if subgraph is not state and subgraph.graph is not state:
        raise KeyError('Subgraph does not belong to given state')

    # Find the top-level scope
    scope_tree = state.scope_tree()
    scope_dict = state.scope_dict()
    scope_dict_children = state.scope_children()
    top_scopenode = -1  # Initialized to -1 since "None" already means top-level

    for node in subgraph.nodes():
        if node not in scope_dict:
            raise KeyError('Node not found in state')

        # If scope entry/exit, ensure entire scope is in subgraph
        if isinstance(node, nodes.EntryNode):
            scope_nodes = scope_dict_children[node]
            if any(n not in subgraph.nodes() for n in scope_nodes):
                raise ValueError('Subgraph contains partial scopes (entry)')
        elif isinstance(node, nodes.ExitNode):
            entry = state.entry_node(node)
            scope_nodes = scope_dict_children[entry] + [entry]
            if any(n not in subgraph.nodes() for n in scope_nodes):
                raise ValueError('Subgraph contains partial scopes (exit)')

        scope_node = scope_dict[node]
        if scope_node not in subgraph.nodes():
            if top_scopenode != -1 and top_scopenode != scope_node:
                raise ValueError('Subgraph is contained in more than one scope')
            top_scopenode = scope_node

    scope = scope_tree[top_scopenode]
    ###

    # Consolidate edges in top scope
    utils.consolidate_edges(sdfg, scope)
    snodes = subgraph.nodes()

    # Collect inputs and outputs of the nested SDFG
    inputs: List[MultiConnectorEdge] = []
    outputs: List[MultiConnectorEdge] = []
    for node in snodes:
        for edge in state.in_edges(node):
            if edge.src not in snodes:
                inputs.append(edge)
        for edge in state.out_edges(node):
            if edge.dst not in snodes:
                outputs.append(edge)

    # Collect transients not used outside of subgraph (will be removed of
    # top-level graph)
    data_in_subgraph = set(n.data for n in subgraph.nodes()
                           if isinstance(n, nodes.AccessNode))
    # Find other occurrences in SDFG
    other_nodes = set(
        n.data for s in sdfg.nodes() for n in s.nodes()
        if isinstance(n, nodes.AccessNode) and n not in subgraph.nodes())
    subgraph_transients = set()
    for data in data_in_subgraph:
        datadesc = sdfg.arrays[data]
        if datadesc.transient and data not in other_nodes:
            subgraph_transients.add(data)

    # All transients of edges between code nodes are also added to nested graph
    for edge in subgraph.edges():
        if (isinstance(edge.src, nodes.CodeNode)
                and isinstance(edge.dst, nodes.CodeNode)):
            subgraph_transients.add(edge.data.data)

    # Collect data used in access nodes within subgraph (will be referenced in
    # full upon nesting)
    input_arrays = set()
    output_arrays = {}
    for node in subgraph.nodes():
        if (isinstance(node, nodes.AccessNode)
                and node.data not in subgraph_transients):
            if node.has_reads(state):
                input_arrays.add(node.data)
            if node.has_writes(state):
                output_arrays[node.data] = state.in_edges(node)[0].data.wcr

    # Create the nested SDFG
    nsdfg = SDFG(name or 'nested_' + state.label)

    # Transients are added to the nested graph as-is
    for name in subgraph_transients:
        nsdfg.add_datadesc(name, sdfg.arrays[name])

    # Input/output data that are not source/sink nodes are added to the graph
    # as non-transients
    for name in (input_arrays | output_arrays.keys()):
        datadesc = copy.deepcopy(sdfg.arrays[name])
        datadesc.transient = False
        nsdfg.add_datadesc(name, datadesc)

    # Connected source/sink nodes outside subgraph become global data
    # descriptors in nested SDFG
    input_names = {}
    output_names = {}
    global_subsets: Dict[str, Tuple[str, Subset]] = {}
    for edge in inputs:
        if edge.data.data is None:  # Skip edges with an empty memlet
            continue
        name = edge.data.data
        if name not in global_subsets:
            datadesc = copy.deepcopy(sdfg.arrays[edge.data.data])
            datadesc.transient = False
            if not full_data:
                datadesc.shape = edge.data.subset.size()
            new_name = nsdfg.add_datadesc(name, datadesc, find_new_name=True)
            global_subsets[name] = (new_name, edge.data.subset)
        else:
            new_name, subset = global_subsets[name]
            if not full_data:
                new_subset = union(subset, edge.data.subset)
                if new_subset is None:
                    new_subset = Range.from_array(sdfg.arrays[name])
                global_subsets[name] = (new_name, new_subset)
                nsdfg.arrays[new_name].shape = new_subset.size()
        input_names[edge] = new_name
    for edge in outputs:
        if edge.data.data is None:  # Skip edges with an empty memlet
            continue
        name = edge.data.data
        if name not in global_subsets:
            datadesc = copy.deepcopy(sdfg.arrays[edge.data.data])
            datadesc.transient = False
            if not full_data:
                datadesc.shape = edge.data.subset.size()
            new_name = nsdfg.add_datadesc(name, datadesc, find_new_name=True)
            global_subsets[name] = (new_name, edge.data.subset)
        else:
            new_name, subset = global_subsets[name]
            if not full_data:
                new_subset = union(subset, edge.data.subset)
                if new_subset is None:
                    new_subset = Range.from_array(sdfg.arrays[name])
                global_subsets[name] = (new_name, new_subset)
                nsdfg.arrays[new_name].shape = new_subset.size()
        output_names[edge] = new_name
    ###################

    # Add scope symbols to the nested SDFG
    defined_vars = set(
        symbolic.pystr_to_symbolic(s)
        for s in (state.symbols_defined_at(top_scopenode).keys()
                  | sdfg.symbols))
    for v in defined_vars:
        if v in sdfg.symbols:
            sym = sdfg.symbols[v]
            nsdfg.add_symbol(v, sym.dtype)

    # Add constants to nested SDFG
    for cstname, cstval in sdfg.constants.items():
        nsdfg.add_constant(cstname, cstval)

    # Create nested state
    nstate = nsdfg.add_state()

    # Add subgraph nodes and edges to nested state
    nstate.add_nodes_from(subgraph.nodes())
    for e in subgraph.edges():
        nstate.add_edge(e.src, e.src_conn, e.dst, e.dst_conn,
                        copy.deepcopy(e.data))

    # Modify nested SDFG parents in subgraph
    for node in subgraph.nodes():
        if isinstance(node, nodes.NestedSDFG):
            node.sdfg.parent = nstate
            node.sdfg.parent_sdfg = nsdfg
            node.sdfg.parent_nsdfg_node = node

    # Add access nodes and edges as necessary
    edges_to_offset = []
    for edge, name in input_names.items():
        node = nstate.add_read(name)
        new_edge = copy.deepcopy(edge.data)
        new_edge.data = name
        edges_to_offset.append(
            (edge, nstate.add_edge(node, None, edge.dst, edge.dst_conn,
                                   new_edge)))
    for edge, name in output_names.items():
        node = nstate.add_write(name)
        new_edge = copy.deepcopy(edge.data)
        new_edge.data = name
        edges_to_offset.append(
            (edge, nstate.add_edge(edge.src, edge.src_conn, node, None,
                                   new_edge)))

    # Offset memlet paths inside nested SDFG according to subsets
    for original_edge, new_edge in edges_to_offset:
        for edge in nstate.memlet_tree(new_edge):
            edge.data.data = new_edge.data.data
            if not full_data:
                edge.data.subset.offset(
                    global_subsets[original_edge.data.data][1], True)

    # Add nested SDFG node to the input state
    nested_sdfg = state.add_nested_sdfg(
        nsdfg, None,
        set(input_names.values()) | input_arrays,
        set(output_names.values()) | output_arrays.keys())

    # Reconnect memlets to nested SDFG
    reconnected_in = set()
    reconnected_out = set()
    empty_input = None
    empty_output = None
    for edge in inputs:
        if edge.data.data is None:
            empty_input = edge
            continue

        name = input_names[edge]
        if name in reconnected_in:
            continue
        if full_data:
            data = Memlet.from_array(edge.data.data,
                                     sdfg.arrays[edge.data.data])
        else:
            data = copy.deepcopy(edge.data)
            data.subset = global_subsets[edge.data.data][1]
        state.add_edge(edge.src, edge.src_conn, nested_sdfg, name, data)
        reconnected_in.add(name)

    for edge in outputs:
        if edge.data.data is None:
            empty_output = edge
            continue

        name = output_names[edge]
        if name in reconnected_out:
            continue
        if full_data:
            data = Memlet.from_array(edge.data.data,
                                     sdfg.arrays[edge.data.data])
        else:
            data = copy.deepcopy(edge.data)
            data.subset = global_subsets[edge.data.data][1]
        data.wcr = edge.data.wcr
        state.add_edge(nested_sdfg, name, edge.dst, edge.dst_conn, data)
        reconnected_out.add(name)

    # Connect access nodes to internal input/output data as necessary
    entry = scope.entry
    exit = scope.exit
    for name in input_arrays:
        node = state.add_read(name)
        if entry is not None:
            state.add_nedge(entry, node, Memlet())
        state.add_edge(node, None, nested_sdfg, name,
                       Memlet.from_array(name, sdfg.arrays[name]))
    for name, wcr in output_arrays.items():
        node = state.add_write(name)
        if exit is not None:
            state.add_nedge(node, exit, Memlet())
        state.add_edge(nested_sdfg, name, node, None, Memlet(data=name,
                                                             wcr=wcr))

    # Graph was not reconnected, but needs to be
    if state.in_degree(nested_sdfg) == 0 and empty_input is not None:
        state.add_edge(empty_input.src, empty_input.src_conn, nested_sdfg, None,
                       empty_input.data)
    if state.out_degree(nested_sdfg) == 0 and empty_output is not None:
        state.add_edge(nested_sdfg, None, empty_output.dst,
                       empty_output.dst_conn, empty_output.data)

    # Remove subgraph nodes from graph
    state.remove_nodes_from(subgraph.nodes())

    # Remove subgraph transients from top-level graph
    for transient in subgraph_transients:
        del sdfg.arrays[transient]

    # Remove newly isolated nodes due to memlet consolidation
    for edge in inputs:
        if state.in_degree(edge.src) + state.out_degree(edge.src) == 0:
            state.remove_node(edge.src)
    for edge in outputs:
        if state.in_degree(edge.dst) + state.out_degree(edge.dst) == 0:
            state.remove_node(edge.dst)

    return nested_sdfg


def state_fission(sdfg: SDFG, subgraph: graph.SubgraphView) -> SDFGState:
    '''
    Given a subgraph, adds a new SDFG state before the state that contains it,
    removes the subgraph from the original state, and connects the two states.
    :param subgraph: the subgraph to remove.
    :return: the newly created SDFG state.
    '''

    state: SDFGState = subgraph.graph
    newstate = sdfg.add_state_before(state)

    # Save edges before removing nodes
    orig_edges = subgraph.edges()

    # Mark boundary access nodes to keep after fission
    nodes_to_remove = set(subgraph.nodes())
    boundary_nodes = [
        n for n in subgraph.nodes()
        if len(state.out_edges(n)) > len(subgraph.out_edges(n))
    ] + [
        n for n in subgraph.nodes()
        if len(state.in_edges(n)) > len(subgraph.in_edges(n))
    ]

    # Make dictionary of nodes to add to new state
    new_nodes = {n: n for n in subgraph.nodes()}
    new_nodes.update({b: copy.deepcopy(b) for b in boundary_nodes})

    nodes_to_remove -= set(boundary_nodes)
    state.remove_nodes_from(nodes_to_remove)

    for n in new_nodes.values():
        if isinstance(n, nodes.NestedSDFG):
            # Set the new parent state
            n.sdfg.parent = newstate

    newstate.add_nodes_from(new_nodes.values())

    for e in orig_edges:
        newstate.add_edge(new_nodes[e.src], e.src_conn, new_nodes[e.dst],
                          e.dst_conn, e.data)

    return newstate


def _get_internal_subset(internal_memlet: Memlet,
                         external_memlet: Memlet,
                         use_src_subset: bool = False,
                         use_dst_subset: bool = False) -> subsets.Subset:
    if (internal_memlet.data != external_memlet.data
            and internal_memlet.other_subset is not None):
        return internal_memlet.other_subset
    if not use_src_subset and not use_dst_subset:
        return internal_memlet.subset
    if use_src_subset and use_dst_subset:
        raise ValueError('Source and destination subsets cannot be '
                         'specified at the same time')
    if use_src_subset:
        return internal_memlet.src_subset
    if use_dst_subset:
        return internal_memlet.dst_subset
    return internal_memlet.subset


def unsqueeze_memlet(internal_memlet: Memlet,
                     external_memlet: Memlet,
                     preserve_minima: bool = False,
                     use_src_subset: bool = False,
                     use_dst_subset: bool = False) -> Memlet:
    """ Unsqueezes and offsets a memlet, as per the semantics of nested
        SDFGs.
        :param internal_memlet: The internal memlet (inside nested SDFG)
                                before modification.
        :param external_memlet: The external memlet before modification.
        :param preserve_minima: Do not change the subset's minimum elements.
        :param use_src_subset: If both sides of the memlet refer to same array,
                               prefer source subset.
        :param use_dst_subset: If both sides of the memlet refer to same array,
                               prefer destination subset.
        :return: Offset Memlet to set on the resulting graph.
    """
    internal_subset = _get_internal_subset(internal_memlet, external_memlet,
                                           use_src_subset, use_dst_subset)
    result = copy.deepcopy(internal_memlet)
    result.data = external_memlet.data
    result.other_subset = None
    result.subset = copy.deepcopy(internal_subset)

    shape = external_memlet.subset.size()
    if len(internal_subset) < len(external_memlet.subset):
        ones = [i for i, d in enumerate(shape) if d == 1]

        # Special case: If internal memlet is one element and the top
        # memlet uses all its dimensions, ignore the internal element
        # TODO: There must be a better solution
        if (len(internal_subset) == 1 and ones == list(range(len(shape))) and
            (internal_subset[0] == (0, 0, 1) or internal_subset[0] == 0)):
            to_unsqueeze = ones[1:]
        else:
            to_unsqueeze = ones

        result.subset.unsqueeze(to_unsqueeze)
    elif len(internal_subset) > len(external_memlet.subset):
        # Try to squeeze internal memlet
        result.subset.squeeze()
        if len(result.subset) != len(external_memlet.subset):
            raise ValueError('Unexpected extra dimensions in internal memlet '
                             'while un-squeezing memlet.\nExternal memlet: %s\n'
                             'Internal memlet: %s' %
                             (external_memlet, internal_memlet))

    result.subset.offset(external_memlet.subset, False)

    if preserve_minima:
        if len(result.subset) != len(external_memlet.subset):
            raise ValueError(
                'Memlet specifies reshape that cannot be un-squeezed.\n'
                'External memlet: %s\nInternal memlet: %s' %
                (external_memlet, internal_memlet))

        original_minima = external_memlet.subset.min_element()
        for i in set(range(len(original_minima))):
            rb, re, rs = result.subset.ranges[i]
            result.subset.ranges[i] = (original_minima[i], re, rs)

    # TODO: Offset rest of memlet according to other_subset
    if external_memlet.other_subset is not None:
        raise NotImplementedError

    return result


def replicate_scope(sdfg: SDFG, state: SDFGState,
                    scope: ScopeSubgraphView) -> ScopeSubgraphView:
    """
    Replicates a scope subgraph view within a state, reconnecting all external
    edges to the same nodes.
    :param sdfg: The SDFG in which the subgraph scope resides.
    :param state: The SDFG state in which the subgraph scope resides.
    :param scope: The scope subgraph to replicate.
    :return: A reconnected replica of the scope.
    """
    exit_node = state.exit_node(scope.entry)

    # Replicate internal graph
    new_nodes = []
    new_entry = None
    new_exit = None
    to_find_new_names: Set[nodes.AccessNode] = set()
    for node in scope.nodes():
        node_copy = copy.deepcopy(node)
        if node == scope.entry:
            new_entry = node_copy
        elif node == exit_node:
            new_exit = node_copy

        if (isinstance(node, nodes.AccessNode)
                and node.desc(sdfg).lifetime == dtypes.AllocationLifetime.Scope
                and node.desc(sdfg).transient):
            to_find_new_names.add(node_copy)
        state.add_node(node_copy)
        new_nodes.append(node_copy)

    for edge in scope.edges():
        src = scope.nodes().index(edge.src)
        dst = scope.nodes().index(edge.dst)
        state.add_edge(new_nodes[src], edge.src_conn, new_nodes[dst],
                       edge.dst_conn, copy.deepcopy(edge.data))

    # Reconnect external scope nodes
    for edge in state.in_edges(scope.entry):
        state.add_edge(edge.src, edge.src_conn, new_entry, edge.dst_conn,
                       copy.deepcopy(edge.data))
    for edge in state.out_edges(exit_node):
        state.add_edge(new_exit, edge.src_conn, edge.dst, edge.dst_conn,
                       copy.deepcopy(edge.data))

    # Set the exit node's map to match the entry node
    new_exit.map = new_entry.map

    # Replicate all temporary transients within scope
    for node in to_find_new_names:
        desc = node.desc(sdfg)
        new_name = sdfg.add_datadesc(node.data,
                                     copy.deepcopy(desc),
                                     find_new_name=True)
        node.data = new_name
        for edge in state.all_edges(node):
            for e in state.memlet_tree(edge):
                e.data.data = new_name

    return ScopeSubgraphView(state, new_nodes, new_entry)


def offset_map(sdfg: SDFG,
               state: SDFGState,
               entry: nodes.MapEntry,
               dim: int,
               offset: symbolic.SymbolicType,
               negative: bool = True):
    """
    Offsets a map parameter and its contents by a value.
    :param sdfg: The SDFG in which the map resides.
    :param state: The state in which the map resides.
    :param entry: The map entry node.
    :param dim: The map dimension to offset.
    :param offset: The value to offset by.
    :param negative: If True, offsets by ``-offset``.
    """
    entry.map.range.offset(offset, negative, indices=[dim])
    param = entry.map.params[dim]
    subgraph = state.scope_subgraph(entry)
    # Offset map param by -offset, contents by +offset and vice versa
    if negative:
        subgraph.replace(param, f'({param} + {offset})')
    else:
        subgraph.replace(param, f'({param} - {offset})')


def split_interstate_edges(sdfg: SDFG) -> None:
    """
    Splits all inter-state edges into edges with conditions and edges with
    assignments. This procedure helps in nested loop detection.
    :param sdfg: The SDFG to split
    :note: Operates in-place on the SDFG.
    """
    for e in sdfg.edges():
        if e.data.assignments and not e.data.is_unconditional():
            tmpstate = sdfg.add_state()
            sdfg.add_edge(e.src, tmpstate,
                          InterstateEdge(condition=e.data.condition))
            sdfg.add_edge(tmpstate, e.dst,
                          InterstateEdge(assignments=e.data.assignments))
            sdfg.remove_edge(e)


def is_symbol_unused(sdfg: SDFG, sym: str) -> bool:
    """
    Checks for uses of symbol in an SDFG, and if there are none returns False.
    :param sdfg: The SDFG to search.
    :param sym: The symbol to test.
    :return: True if the symbol can be removed, False otherwise.
    """
    for desc in sdfg.arrays.values():
        if sym in map(str, desc.free_symbols):
            return False
    for state in sdfg.nodes():
        if sym in state.free_symbols:
            return False
    for e in sdfg.edges():
        if sym in e.data.free_symbols:
            return False

    # Not found, symbol can be removed
    return True


def are_subsets_contiguous(subset_a: subsets.Subset,
                           subset_b: subsets.Subset,
                           dim: int = None) -> bool:

    if dim is not None:
        # A version that only checks for contiguity in certain
        # dimension (e.g., to prioritize stride-1 range)
        if (not isinstance(subset_a, subsets.Range)
                or not isinstance(subset_b, subsets.Range)):
            raise NotImplementedError('Contiguous subset check only '
                                      'implemented for ranges')

        # Other dimensions must be equal
        for i, (s1, s2) in enumerate(zip(subset_a.ranges, subset_b.ranges)):
            if i == dim:
                continue
            if s1[0] != s2[0] or s1[1] != s2[1] or s1[2] != s2[2]:
                return False

        # Set of conditions for contiguous dimension
        ab = (subset_a[dim][1] + 1) == subset_b[dim][0]
        a_overlap_b = subset_a[dim][1] >= subset_b[dim][0]
        ba = (subset_b[dim][1] + 1) == subset_a[dim][0]
        b_overlap_a = subset_b[dim][1] >= subset_a[dim][0]
        # NOTE: Must check with "==" due to sympy using special types
        return (ab == True or a_overlap_b == True or ba == True
                or b_overlap_a == True)

    # General case
    bbunion = subsets.bounding_box_union(subset_a, subset_b)
    try:
        if bbunion.num_elements() == (subset_a.num_elements() +
                                      subset_b.num_elements()):
            return True
    except TypeError:
        pass

    return False


def find_contiguous_subsets(subset_list: List[subsets.Subset],
                            dim: int = None) -> Set[subsets.Subset]:
    """ 
    Finds the set of largest contiguous subsets in a list of subsets. 
    :param subsets: Iterable of subset objects.
    :param dim: Check for contiguity only for the specified dimension.
    :return: A list of contiguous subsets.
    """
    # Currently O(n^3) worst case. TODO: improve
    subset_set = set(
        subsets.Range.from_indices(s) if isinstance(s, subsets.Indices) else s
        for s in subset_list)
    while True:
        for sa, sb in itertools.product(subset_set, subset_set):
            if sa is sb:
                continue
            if sa.covers(sb):
                subset_set.remove(sb)
                break
            elif sb.covers(sa):
                subset_set.remove(sa)
                break
            elif are_subsets_contiguous(sa, sb, dim):
                subset_set.remove(sa)
                subset_set.remove(sb)
                subset_set.add(subsets.bounding_box_union(sa, sb))
                break
        else:  # No modification performed
            break
    return subset_set


def constant_symbols(sdfg: SDFG) -> Set[str]:
    """ 
    Returns a set of symbols that will never change values throughout the course
    of the given SDFG. Specifically, these are the input symbols (i.e., not
    defined in a particular scope) that are never set by interstate edges.
    :param sdfg: The input SDFG.
    :return: A set of symbol names that remain constant throughout the SDFG.
    """
    interstate_symbols = {
        k
        for e in sdfg.edges() for k in e.data.assignments.keys()
    }
    return set(sdfg.symbols) - interstate_symbols


def simplify_state(state: SDFGState,
                   remove_views: bool = False) -> MultiDiGraph:
    """
    Returns a networkx MultiDiGraph object that contains all the access nodes
    and corresponding edges of an SDFG state. The removed code nodes and map
    scopes are replaced by edges that connect their ancestor and succesor access
    nodes.
    :param state: The input SDFG state.
    :return: The MultiDiGraph object.
    """

    sdfg = state.parent

    # Copy the whole state
    G = MultiDiGraph()
    for n in state.nodes():
        G.add_node(n)
    for n in state.nodes():
        for e in state.all_edges(n):
            G.add_edge(e.src, e.dst)
    # Collapse all mappings and their scopes into one node
    scope_children = state.scope_children()
    for n in scope_children[None]:
        if isinstance(n, nodes.EntryNode):
            G.add_edges_from([(n, x)
                              for (y, x) in G.out_edges(state.exit_node(n))])
            G.remove_nodes_from(scope_children[n])
    # Remove all nodes that are not AccessNodes or have incoming
    # wcr edges and connect their predecessors and successors
    for n in state.nodes():
        if n in G.nodes():
            if (not isinstance(n, nodes.AccessNode) or
                (remove_views and isinstance(sdfg.arrays[n.data], data.View))):
                for p in G.predecessors(n):
                    for c in G.successors(n):
                        G.add_edge(p, c)
                G.remove_node(n)
            else:
                for e in state.all_edges(n):
                    if e.data.wcr is not None:
                        for p in G.predecessors(n):
                            for s in G.successors(n):
                                G.add_edge(p, s)
                        G.remove_node(n)
                        break

    return G


def tile(sdfg: SDFG, map_entry: nodes.MapEntry, divides_evenly: bool,
         skew: bool, **tile_sizes: symbolic.SymbolicType):
    """ 
    Helper function that tiles a Map scope by the given sizes, in the 
    given order.
    :param sdfg: The SDFG where the map resides.
    :param map_entry: The map entry node to tile.
    :param divides_evenly: If True, skips pre/postamble for cases
                           where the map dimension is not a multiplier
                           of the tile size.
    :param skew: If True, skews the tiled map to start from zero. Helps
                 compilers improve performance in certain cases.
    :param tile_sizes: An ordered dictionary of the map parameter names
                       to tile and their respective tile size (which can be
                       symbolic expressions).
    """
    # Avoid import loop
    from dace.transformation.dataflow import StripMining

    for k, v in tile_sizes.items():
        StripMining.apply_to(sdfg,
                             dict(dim_idx=map_entry.params.index(k),
                                  tile_size=str(v),
                                  divides_evenly=divides_evenly,
                                  skew=skew),
                             _map_entry=map_entry)


def permute_map(map_entry: nodes.MapEntry, perm: List[int]):
    """ Permutes indices of a map according to a given list of integers. """
    map_entry.map.params = [map_entry.map.params[p] for p in perm]
    map_entry.map.range = [map_entry.map.range[p] for p in perm]


def extract_map_dims(sdfg: SDFG, map_entry: nodes.MapEntry,
                     dims: List[int]) -> Tuple[nodes.MapEntry, nodes.MapEntry]:
    """ 
    Helper function that extracts specific map dimensions into an outer map.
    :param sdfg: The SDFG where the map resides.
    :param map_entry: Map entry node to extract.
    :param dims: A list of dimension indices to extract.
    :return: A 2-tuple containing the extracted map and the remainder map.
    """
    # Avoid import loop
    from dace.transformation.dataflow import MapCollapse, MapExpansion

    # Make extracted dimensions first
    permute_map(
        map_entry,
        dims + [i for i in range(len(map_entry.map.params)) if i not in dims])
    # Expand map
    if len(map_entry.map.params) > 1:
        entries = MapExpansion.apply_to(sdfg, map_entry=map_entry)

        # Collapse extracted maps
        extracted_map = entries[0]
        for idx in range(len(dims) - 1):
            extracted_map, _ = MapCollapse.apply_to(
                sdfg,
                _outer_map_entry=extracted_map,
                _inner_map_entry=entries[idx + 1],
            )

        # Collapse remaining maps
        map_to_collapse = entries[len(dims)]
        for idx in range(len(dims), len(entries) - 1):
            map_to_collapse, _ = MapCollapse.apply_to(
                sdfg,
                _outer_map_entry=map_to_collapse,
                _inner_map_entry=entries[idx + 1],
            )
    else:
        extracted_map = map_entry
        map_to_collapse = map_entry

    return extracted_map, map_to_collapse


def scope_tree_recursive(state: SDFGState,
                         entry: Optional[nodes.EntryNode] = None) -> ScopeTree:
    """ 
    Returns a scope tree that includes scopes from nested SDFGs. 
    :param state: The state that contains the root of the scope tree.
    :param entry: A scope entry node to set as root, otherwise the state is 
                  the root if None is given.
    """
    stree = state.scope_tree()[entry]
    stree.state = state  # Annotate state in tree

    # Add nested SDFGs as children
    def traverse(state: SDFGState, treenode: ScopeTree):
        snodes = state.scope_children()[treenode.entry]
        for node in snodes:
            if isinstance(node, nodes.NestedSDFG):
                for nstate in node.sdfg.nodes():
                    ntree = nstate.scope_tree()[None]
                    ntree.state = nstate
                    treenode.children.append(ntree)
        for child in treenode.children:
            traverse(getattr(child, 'state', state), child)

    traverse(state, stree)
    return stree


def get_internal_scopes(
        state: SDFGState,
        entry: nodes.EntryNode,
        immediate: bool = False) -> List[Tuple[SDFGState, nodes.EntryNode]]:
    """ 
    Returns all internal scopes within a given scope, including if they 
    reside in nested SDFGs.
    :param state: State in which entry node resides.
    :param entry: The entry node to start from.
    :param immediate: If True, only returns the scopes that are immediately
                      nested in the map.
    """
    stree = scope_tree_recursive(state, entry)
    result = []

    def traverse(state: SDFGState, treenode: ScopeTree):
        for child in treenode.children:
            if child.entry is not None:
                result.append((state, child.entry))
                if not immediate:
                    traverse(state, child)
            else:  # Nested SDFG
                traverse(child.state, child)

    traverse(state, stree)
    return result


def gpu_map_has_explicit_threadblocks(state: SDFGState,
                                      entry: nodes.EntryNode) -> bool:
    """ 
    Returns True if GPU_Device map has explicit thread-block maps nested within.
    """
    internal_maps = get_internal_scopes(state, entry)
    if any(m.schedule in (dtypes.ScheduleType.GPU_ThreadBlock,
                          dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
           for _, m in internal_maps):
        return True
    imm_maps = get_internal_scopes(state, entry, immediate=True)
    if any(m.schedule == dtypes.ScheduleType.Default for _, m in imm_maps):
        return True

    return False


def reconnect_edge_through_map(
    state: SDFGState, edge: graph.MultiConnectorEdge[Memlet],
    new_node: Union[nodes.EntryNode, nodes.ExitNode], keep_src: bool
) -> Tuple[graph.MultiConnectorEdge[Memlet], graph.MultiConnectorEdge[Memlet]]:
    """
    Reconnects an edge through a map scope, removes old edge, and returns the 
    two new edges.
    :param state: The state in which the edge and map reside.
    :param edge: The edge to reconnect and remove.
    :param new_node: The scope (map) entry or exit to reconnect through.
    :param keep_src: If True, keeps the source of the edge intact, otherwise
                     keeps destination of edge.
    :return: A 2-tuple of (incoming edge, outgoing edge).
    """
    if keep_src:
        result = state.add_edge_pair(new_node,
                                     edge.dst,
                                     edge.src,
                                     edge.data,
                                     internal_connector=edge.dst_conn,
                                     external_connector=edge.src_conn)
    else:
        result = state.add_edge_pair(new_node,
                                     edge.src,
                                     edge.dst,
                                     edge.data,
                                     internal_connector=edge.src_conn,
                                     external_connector=edge.dst_conn)
    state.remove_edge(edge)
    return result


def contained_in(state: SDFGState, node: nodes.Node,
                 scope: nodes.EntryNode) -> bool:
    """
    Returns true if the specified node is contained within the scope opened
    by the given entry node (including through nested SDFGs).
    """
    # A node is contained within itself
    if node is scope:
        return True
    cursdfg = state.parent
    curstate = state
    curscope = state.entry_node(node)
    while cursdfg is not None:
        while curscope is not None:
            if curscope is scope:
                return True
            curscope = curstate.entry_node(curscope)
        curstate = cursdfg.parent
        curscope = cursdfg.parent_nsdfg_node
        cursdfg = cursdfg.parent_sdfg
    return False


def get_parent_map(
    state: SDFGState,
    node: Optional[nodes.Node] = None
) -> Optional[Tuple[nodes.EntryNode, SDFGState]]:
    """
    Returns the map in which the state (and node) are contained in, or None if
    it is free.
    :param state: The state to test or parent of the node to test.
    :param node: The node to test (optional).
    :return: A tuple of (entry node, state) or None.
    """
    cursdfg = state.parent
    curstate = state
    curscope = node
    while cursdfg is not None:
        if curscope is not None:
            curscope = curstate.entry_node(curscope)
            if curscope is not None:
                return curscope, curstate
        curstate = cursdfg.parent
        curscope = cursdfg.parent_nsdfg_node
        cursdfg = cursdfg.parent_sdfg
    return None


def redirect_edge(
        state: SDFGState,
        edge: graph.MultiConnectorEdge[Memlet],
        new_src: Optional[nodes.Node] = None,
        new_dst: Optional[nodes.Node] = None,
        new_src_conn: Optional[str] = None,
        new_dst_conn: Optional[str] = None,
        new_data: Optional[str] = None,
        new_memlet: Optional[Memlet] = None
) -> graph.MultiConnectorEdge[Memlet]:
    """
    Redirects an edge in a state. Choose which elements to override by setting
    the keyword arguments.
    :param state: The SDFG state in which the edge resides.
    :param edge: The edge to redirect.
    :param new_src: If provided, redirects the source of the new edge.
    :param new_dst: If provided, redirects the destination of the new edge.
    :param new_src_conn: If provided, renames the source connector of the edge.
    :param new_dst_conn: If provided, renames the destination connector of the 
                         edge.
    :param new_data: If provided, changes the data on the memlet of the edge,
                     and the entire associated memlet tree.
    :param new_memlet: If provided, changes only the memlet of the new edge.
    :return: The new, redirected edge.
    :note: ``new_data`` and ``new_memlet`` cannot be used at the same time.
    """
    if new_data is not None and new_memlet is not None:
        raise ValueError('new_data and new_memlet cannot both be given.')
    mtree = None
    if new_data is not None:
        mtree = state.memlet_tree(edge)
    state.remove_edge(edge)
    if new_data is not None:
        memlet = copy.deepcopy(edge.data)
        memlet.data = new_data
        # Rename on full memlet tree
        for e in mtree:
            e.data.data = new_data
    else:
        memlet = new_memlet or edge.data
    new_edge = state.add_edge(new_src or edge.src, new_src_conn
                              or edge.src_conn, new_dst or edge.dst,
                              new_dst_conn or edge.dst_conn, memlet)
    return new_edge
