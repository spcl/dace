# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functionality that allows users to "cut out" parts of an SDFG in a smart way (i.e., memory preserving) for localized
testing or optimization.
"""
from collections import deque
import copy
from re import I
from typing import Deque, Dict, List, Optional, Set, Tuple, Union
from dace import data, Memlet
from dace.sdfg import nodes as nd, SDFG, SDFGState, utils as sdutil
from dace.sdfg.graph import SubgraphView, MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView


def _stateset_frontier(states: Set[SDFGState]) -> Tuple[Set[SDFGState], Set]:
    """
    For a set of states, return the frontier.
    The frontier in this case refers to the predecessor states leading into the given set of states.
    :param states: The set of states for which to gather the frontier.
    :return: A 2-tuple with the state frontier, and all corresponding frontier edges.
    """
    frontier = set()
    frontier_edges = set()
    for state in states:
        for iedge in state.parent.in_edges(state):
            if iedge.src not in states:
                if iedge.src not in frontier:
                    frontier.add(iedge.src)
                if iedge not in frontier_edges:
                    frontier_edges.add(iedge)
    return frontier, frontier_edges


def multistate_cutout(*states: SDFGState, inserted_states: Dict[SDFGState, SDFGState] = None) -> SDFG:
    """
    Cut out a multi-state subgraph from an SDFG to run separately for localized testing or optimization.
    The subgraph defined by the list of states will be extended to include any further states necessary to make the
    resulting cutout executable, i.e, to ensure that there is a distinct start state. This is achieved by gradually
    adding more states from the cutout's state machine frontier until a distinct, single entry state is obtained.
    :param states: The states in the subgraph to cut out.
    :param inserted_states: A dictionary that provides a mapping from the original states to their cutout counterparts.
    """
    create_element = copy.deepcopy

    # Check that all states are inside the same SDFG.
    sdfg = list(states)[0].parent
    if any(i.parent != sdfg for i in states):
        raise Exception('Not all cutout states reside in the same SDFG')

    cutout_states: Set[SDFGState] = set(states)

    # Determine the start state and ensure there IS a unique start state. If there is no unique
    # start state, keep adding states from the predecessor frontier in the state machine until
    # a unique start state can be determined.
    start_state: SDFGState = None
    for state in cutout_states:
        if state == sdfg.start_state:
            start_state = state
            break

    if start_state is None:
        bfs_queue = deque()
        bfs_queue.append(_stateset_frontier(cutout_states))

        while len(bfs_queue) > 0:
            frontier, frontier_edges = bfs_queue.popleft()
            if len(frontier_edges) == 0:
                # No explicit start state, but also no frontier to select from.
                return sdfg
            elif len(frontier_edges) == 1:
                # The destination of the only frontier edge must be the
                # start state, since only one edge leads into the subgraph.
                start_state = list(frontier_edges)[0].dst
            else:
                if len(frontier) == 0:
                    # No explicit start state, but also no frontier to select from.
                    return sdfg
                if len(frontier) == 1:
                    # For many frontier edges but only one frontier state,
                    # the frontier state is the new start state and is
                    # included in the cutout.
                    start_state = list(frontier)[0]
                    cutout_states.add(start_state)
                else:
                    for s in frontier:
                        cutout_states.add(s)
                    bfs_queue.append(_stateset_frontier(cutout_states))

    subgraph: SubgraphView = SubgraphView(sdfg, cutout_states)

    # Make a new SDFG with the included constants, used symbols, and data containers.
    new_sdfg = SDFG(f'{sdfg.name}_cutout', sdfg.constants_prop)
    defined_symbols: Dict[str, data.Data] = dict()
    free_symbols: Set[str] = set()
    for state in cutout_states:
        free_symbols |= state.free_symbols
        state_defined_symbols = state.defined_symbols()
        for sym in state_defined_symbols:
            defined_symbols[sym] = state_defined_symbols[sym]
    for sym in free_symbols:
        new_sdfg.add_symbol(sym, defined_symbols[sym])

    for state in cutout_states:
        for dnode in state.data_nodes():
            if dnode.data in new_sdfg.arrays:
                continue
            new_desc = sdfg.arrays[dnode.data].clone()
            new_sdfg.add_datadesc(dnode.data, new_desc)

    # Add all states and state transitions required to the new cutout SDFG by traversing the state machine edges.
    if inserted_states is None:
        inserted_states: Dict[SDFGState, SDFGState] = {}
    for is_edge in subgraph.edges():
        if is_edge.src not in inserted_states:
            inserted_states[is_edge.src] = create_element(is_edge.src)
            new_sdfg.add_node(inserted_states[is_edge.src], is_start_state=(is_edge.src == start_state))
            inserted_states[is_edge.src].parent = new_sdfg
        if is_edge.dst not in inserted_states:
            inserted_states[is_edge.dst] = create_element(is_edge.dst)
            new_sdfg.add_node(inserted_states[is_edge.dst], is_start_state=(is_edge.dst == start_state))
            inserted_states[is_edge.dst].parent = new_sdfg
        new_sdfg.add_edge(
            inserted_states[is_edge.src],
            inserted_states[is_edge.dst],
            create_element(is_edge.data)
        )

    # Add remaining necessary states.
    for state in subgraph.nodes():
        if state not in inserted_states:
            inserted_states[state] = create_element(state)
            new_sdfg.add_node(inserted_states[state], is_start_state=(state == start_state))
            inserted_states[state].parent = new_sdfg

    return new_sdfg


def cutout_state(
    state: SDFGState, *nodes: nd.Node, make_copy: bool = True,
    inserted_nodes: Dict[Union[nd.Node, SDFGState], Union[nd.Node, SDFGState]] = None
) -> SDFG:
    """
    Cut out a subgraph of a state from an SDFG to run separately for localized testing or optimization.
    The subgraph defined by the list of nodes will be extended to include access nodes of data containers necessary
    to run the graph separately. In addition, all transient data containers created outside the cut out graph will
    become global.
    :param state: The SDFG state in which the subgraph resides.
    :param nodes: The nodes in the subgraph to cut out.
    :param make_copy: If True, deep-copies every SDFG element in the copy. Otherwise, original references are kept.
    :param inserted_nodes: A dictionary that maps nodes from the original SDFG to their cutout counterparts.
    """
    create_element = copy.deepcopy if make_copy else (lambda x: x)
    sdfg = state.parent
    subgraph: StateSubgraphView = StateSubgraphView(state, nodes)
    subgraph = _extend_subgraph_with_access_nodes(state, subgraph)
    other_arrays = _containers_defined_outside(sdfg, state, subgraph)

    # Make a new SDFG with the included constants, used symbols, and data containers
    new_sdfg = SDFG(f'{state.parent.name}_cutout', sdfg.constants_prop)
    defined_syms = subgraph.defined_symbols()
    freesyms = subgraph.free_symbols
    for sym in freesyms:
        new_sdfg.add_symbol(sym, defined_syms[sym])

    for dnode in subgraph.data_nodes():
        if dnode.data in new_sdfg.arrays:
            continue
        new_desc = sdfg.arrays[dnode.data].clone()
        # If transient is defined outside, it becomes a global
        if dnode.data in other_arrays:
            new_desc.transient = False
        new_sdfg.add_datadesc(dnode.data, new_desc)

    # Add a single state with the extended subgraph
    new_state = new_sdfg.add_state(state.label, is_start_state=True)
    if inserted_nodes is None:
        inserted_nodes: Dict[nd.Node, nd.Node] = {}
    for e in subgraph.edges():
        if e.src not in inserted_nodes:
            inserted_nodes[e.src] = create_element(e.src)
        if e.dst not in inserted_nodes:
            inserted_nodes[e.dst] = create_element(e.dst)
        new_state.add_edge(inserted_nodes[e.src], e.src_conn, inserted_nodes[e.dst], e.dst_conn, create_element(e.data))

    # Insert remaining isolated nodes
    for n in subgraph.nodes():
        if n not in inserted_nodes:
            inserted_nodes[n] = create_element(n)
            new_state.add_node(inserted_nodes[n])

    # Remove remaining dangling connectors from scope nodes and add new data
    # containers and corresponding accesses for dangling connectors on other
    # nodes
    for orig_node in inserted_nodes.keys():
        new_node = inserted_nodes[orig_node]
        if isinstance(orig_node, (nd.EntryNode, nd.ExitNode)):
            used_connectors = set(e.dst_conn for e in new_state.in_edges(new_node))
            for conn in (new_node.in_connectors.keys() - used_connectors):
                new_node.remove_in_connector(conn)
            used_connectors = set(e.src_conn for e in new_state.out_edges(new_node))
            for conn in (new_node.out_connectors.keys() - used_connectors):
                new_node.remove_out_connector(conn)
        else:
            used_connectors = set(e.dst_conn for e in new_state.in_edges(new_node))
            for conn in (new_node.in_connectors.keys() - used_connectors):
                for e in state.in_edges(orig_node):
                    if e.dst_conn and e.dst_conn == conn:
                        _create_alibi_access_node_for_edge(
                            new_sdfg, new_state, sdfg, e, None, None, new_node,
                            conn
                        )
                        prune = False
                        break
                if prune:
                    new_node.remove_in_connector(conn)
            used_connectors = set(e.src_conn for e in new_state.out_edges(new_node))
            for conn in (new_node.out_connectors.keys() - used_connectors):
                prune = True
                for e in state.out_edges(orig_node):
                    if e.src_conn and e.src_conn == conn:
                        _create_alibi_access_node_for_edge(
                            new_sdfg, new_state, sdfg, e, new_node, conn, None,
                            None
                        )
                        prune = False
                        break
                if prune:
                    new_node.remove_out_connector(conn)

    inserted_nodes[state] = new_state

    return new_sdfg


def _create_alibi_access_node_for_edge(
    target_sdfg: SDFG, target_state: SDFGState, original_sdfg: SDFG,
    original_edge: MultiConnectorEdge[Memlet], from_node: Union[nd.Node, None],
    from_connector: Union[str, None], to_node: Union[nd.Node, None],
    to_connector: Union[str, None]
) -> data.Data:
    """ Add an alibi data container and access node to a dangling connector inside of scopes. """
    original_edge.data
    access_size = original_edge.data.subset.size_exact()
    container_name = '__cutout_' + str(original_edge.data.data)
    container_name = data.find_new_name(
        container_name, target_sdfg._arrays.keys()
    )
    original_array = original_sdfg._arrays[original_edge.data.data]
    memlet_str = ''
    if original_edge.data.subset.num_elements_exact() > 1:
        access_size = original_edge.data.subset.size_exact()
        target_sdfg.add_array(
            container_name, access_size, original_array.dtype
        )
        memlet_str = container_name + '['
        sep = None
        for dim_len in original_edge.data.subset.bounding_box_size():
            if sep is not None:
                memlet_str += ','
            if dim_len > 1:
                memlet_str += '0:' + str(dim_len - 1)
            else:
                memlet_str += '0'
            sep = ','
        memlet_str += ']'
    else:
        target_sdfg.add_scalar(container_name, original_array.dtype)
        memlet_str = container_name + '[0]'
    alibi_access_node = target_state.add_access(container_name)
    if from_node is None:
        target_state.add_edge(
            alibi_access_node, None, to_node, to_connector, Memlet(
                memlet_str
            )
        )
    else:
        target_state.add_edge(
            from_node, from_connector, alibi_access_node, None, Memlet(
                memlet_str
            )
        )


def _extend_subgraph_with_access_nodes(state: SDFGState, subgraph: StateSubgraphView) -> StateSubgraphView:
    """ Expands a subgraph view to include necessary input/output access nodes, using memlet paths. """
    sdfg = state.parent
    result: List[nd.Node] = copy.copy(subgraph.nodes())
    queue: Deque[nd.Node] = deque(subgraph.nodes())

    # Add all nodes in memlet paths
    while len(queue) > 0:
        node = queue.pop()
        if isinstance(node, nd.AccessNode):
            if isinstance(node.desc(sdfg), data.View):
                vnode = sdutil.get_view_node(state, node)
                result.append(vnode)
                queue.append(vnode)
            continue
        for e in state.in_edges(node):
            # Special case: IN_* connectors are not traversed further
            if isinstance(e.dst, (nd.EntryNode, nd.ExitNode)) and (e.dst_conn is None or e.dst_conn.startswith('IN_')):
                continue

            # We don't want to extend access nodes over scope entry nodes, but rather we
            # want to introduce alibi data containers for the correct subset instead. Handled separately.
            if isinstance(e.src, nd.EntryNode) and e.src not in result and state.exit_node(e.src) not in result:
                continue
            else:
                mpath = state.memlet_path(e)
                new_nodes = [mpe.src for mpe in mpath if mpe.src not in result]
                result.extend(new_nodes)
                # Memlet path may end in a code node, continue traversing and expanding graph
                queue.extend(new_nodes)

        for e in state.out_edges(node):
            # Special case: OUT_* connectors are not traversed further
            if isinstance(e.src, (nd.EntryNode, nd.ExitNode)) and (e.src_conn is None or e.src_conn.startswith('OUT_')):
                continue

            # We don't want to extend access nodes over scope entry nodes, but rather we
            # want to introduce alibi data containers for the correct subset instead. Handled separately.
            if isinstance(e.dst, nd.ExitNode) and e.dst not in result and state.entry_node(e.dst) not in result:
                continue
            else:
                mpath = state.memlet_path(e)
                new_nodes = [mpe.dst for mpe in mpath if mpe.dst not in result]
                result.extend(new_nodes)
                # Memlet path may end in a code node, continue traversing and expanding graph
                queue.extend(new_nodes)

    # Check for mismatch in scopes
    for node in result:
        enode = None
        if isinstance(node, nd.EntryNode) and state.exit_node(node) not in result:
            enode = state.exit_node(node)
        if isinstance(node, nd.ExitNode) and state.entry_node(node) not in result:
            enode = state.entry_node(node)
        if enode is not None:
            raise ValueError(f'Cutout cannot expand graph implicitly since "{node}" is in the graph and "{enode}" is '
                             'not. Please provide more nodes in the subgraph as necessary.')

    return StateSubgraphView(state, result)


def _containers_defined_outside(sdfg: SDFG, state: SDFGState, subgraph: StateSubgraphView) -> Set[str]:
    """ Returns a list of containers set outside the given subgraph. """
    # Since we care about containers that are written to, we only need to look at access nodes rather than interstate
    # edges
    result: Set[str] = set()
    for ostate in sdfg.nodes():
        for node in ostate.data_nodes():
            if ostate is not state or node not in subgraph.nodes():
                if ostate.in_degree(node) > 0:
                    result.add(node.data)

    # Add all new sink nodes of new subgraph
    for dnode in subgraph.data_nodes():
        if subgraph.out_degree(dnode) == 0 and state.out_degree(dnode) > 0:
            result.add(dnode.data)

    return result


def cutout(
    *nodes: Union[nd.Node, SDFGState], translation: Dict[Union[nd.Node, SDFGState], Union[nd.Node, SDFGState]],
    state: Optional[SDFGState] = None
) -> SDFG:
    if state is not None:
        if any([isinstance(n, SDFGState) for n in nodes]):
            raise Exception('Mixing cutout nodes of type Node and SDFGState is not allowed')
        new_sdfg = cutout_state(state, *nodes, make_copy=True, inserted_nodes=translation)
    else:
        if any([isinstance(n, nd.Node) for n in nodes]):
            raise Exception('Mixing cutout nodes of type Node and SDFGState is not allowed')
        new_sdfg = multistate_cutout(*nodes, inserted_states=translation)
            
    # Ensure the parent relationships and SDFG list is correct.
    for s in new_sdfg.states():
        for node in s.nodes():
            if isinstance(node, nd.NestedSDFG):
                node.sdfg._parent_sdfg = new_sdfg
    new_sdfg.reset_sdfg_list()

    return new_sdfg
