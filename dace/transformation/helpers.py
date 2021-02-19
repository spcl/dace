# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformation helper API. """
import copy
import itertools

from dace.subsets import Range, Subset, union
import dace.subsets as subsets
from typing import Dict, List, Optional, Tuple, Set

from dace import symbolic
from dace.sdfg import nodes, utils
from dace.sdfg.graph import SubgraphView, MultiConnectorEdge
from dace.sdfg.scope import ScopeSubgraphView
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

    # Collect inputs and outputs of the nested SDFG
    inputs: List[MultiConnectorEdge] = []
    outputs: List[MultiConnectorEdge] = []
    for node in subgraph.source_nodes():
        inputs.extend(state.in_edges(node))
    for node in subgraph.sink_nodes():
        outputs.extend(state.out_edges(node))

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
    input_names = []
    output_names = []
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
        input_names.append(new_name)
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
        output_names.append(new_name)
    ###################

    # Add scope symbols to the nested SDFG
    for v in scope.defined_vars:
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
        nstate.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

    # Modify nested SDFG parents in subgraph
    for node in subgraph.nodes():
        if isinstance(node, nodes.NestedSDFG):
            node.sdfg.parent = nstate
            node.sdfg.parent_sdfg = nsdfg
            node.sdfg.parent_nsdfg_node = node

    # Add access nodes and edges as necessary
    edges_to_offset = []
    for name, edge in zip(input_names, inputs):
        node = nstate.add_read(name)
        new_edge = copy.deepcopy(edge.data)
        new_edge.data = name
        edges_to_offset.append(
            (edge, nstate.add_edge(node, None, edge.dst, edge.dst_conn,
                                   new_edge)))
    for name, edge in zip(output_names, outputs):
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
        set(input_names) | input_arrays,
        set(output_names) | output_arrays.keys())

    # Reconnect memlets to nested SDFG
    reconnected_in = set()
    reconnected_out = set()
    for name, edge in zip(input_names, inputs):
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

    for name, edge in zip(output_names, outputs):
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
    nodes_to_remove -= set(n for n in subgraph.source_nodes()
                           if state.out_degree(n) > 1)
    nodes_to_remove -= set(n for n in subgraph.sink_nodes()
                           if state.in_degree(n) > 1)
    state.remove_nodes_from(nodes_to_remove)

    for n in subgraph.nodes():
        if isinstance(n, nodes.NestedSDFG):
            # Set the new parent state
            n.sdfg.parent = newstate

    newstate.add_nodes_from(subgraph.nodes())

    for e in orig_edges:
        newstate.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, e.data)

    return newstate


def unsqueeze_memlet(internal_memlet: Memlet,
                     external_memlet: Memlet,
                     preserve_minima: bool = False) -> Memlet:
    """ Unsqueezes and offsets a memlet, as per the semantics of nested
        SDFGs.
        :param internal_memlet: The internal memlet (inside nested SDFG)
                                before modification.
        :param external_memlet: The external memlet before modification.
        :param preserve_minima: Do not change the subset's minimum elements.
        :return: Offset Memlet to set on the resulting graph.
    """
    result = copy.deepcopy(internal_memlet)
    result.data = external_memlet.data

    shape = external_memlet.subset.size()
    if len(internal_memlet.subset) < len(external_memlet.subset):
        ones = [i for i, d in enumerate(shape) if d == 1]

        # Special case: If internal memlet is one element and the top
        # memlet uses all its dimensions, ignore the internal element
        # TODO: There must be a better solution
        if (len(internal_memlet.subset) == 1 and ones == list(range(len(shape)))
                and (internal_memlet.subset[0] == (0, 0, 1)
                     or internal_memlet.subset[0] == 0)):
            to_unsqueeze = ones[1:]
        else:
            to_unsqueeze = ones

        result.subset.unsqueeze(to_unsqueeze)
    elif len(internal_memlet.subset) > len(external_memlet.subset):
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
    for node in scope.nodes():
        node_copy = copy.deepcopy(node)
        if node == scope.entry:
            new_entry = node_copy
        elif node == exit_node:
            new_exit = node_copy

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
