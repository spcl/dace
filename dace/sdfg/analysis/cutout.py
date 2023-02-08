# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functionality that allows users to "cut out" parts of an SDFG in a smart way (i.e., memory preserving) for localized
testing or optimization.
"""
from collections import deque
import copy
from typing import Deque, Dict, List, Set, Tuple, Union
from dace import data
from dace.sdfg import nodes as nd, SDFG, SDFGState, utils as sdutil, InterstateEdge
from dace.sdfg.graph import Edge
from dace.sdfg.state import StateSubgraphView


def _stateset_frontier(states: Set[SDFGState]) -> Tuple[Set[SDFGState], Set[Edge[InterstateEdge]]]:
    """
    For a set of states, return the frontier.
    The frontier in this case refers to the predecessor states leading into the given set of states.

    :param states: The set of states to find the frontier of.
    :return: A tuple of the frontier states and the frontier edges.
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
    pass


def cutout_state(state: SDFGState,
                 *nodes: nd.Node,
                 make_copy: bool = True,
                 inserted_nodes: Dict[Union[nd.Node, SDFGState], Union[nd.Node, SDFGState]]) -> SDFG:
    """
    Cut out a subgraph of a state from an SDFG to run separately for localized testing or optimization.
    The subgraph defined by the list of nodes will be extended to include access nodes of data containers necessary
    to run the graph separately. In addition, all transient data containers created outside the cut out graph will
    become global.
    
    :param state: The SDFG state in which the subgraph resides.
    :param nodes: The nodes in the subgraph to cut out.
    :param make_copy: If True, deep-copies every SDFG element in the copy. Otherwise, original references are kept.
    :param inserted_nodes: A dictionary mapping the original nodes to the new nodes in the cutout SDFG.
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

    for edge in subgraph.edges():
        if edge.data is None or edge.data.data is None:
            continue

        memlet = edge.data
        if memlet.data in new_sdfg.arrays:
            continue
        new_desc = sdfg.arrays[memlet.data].clone()
        # If transient is defined outside, it becomes a global
        if memlet.data in other_arrays:
            new_desc.transient = False
        new_sdfg.add_datadesc(memlet.data, new_desc)

    # Add a single state with the extended subgraph
    new_state = new_sdfg.add_state(state.label, is_start_state=True)
    if inserted_nodes is None:
        inserted_nodes = {}
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

    # Remove remaining dangling connectors from scope nodes
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
                        # TODO: create alibi node.
                        prune = False
                        break
                if prune:
                    new_node.remove_in_connector(conn)
            used_connectors = set(e.src_conn for e in new_state.out_edges(new_node))
            for conn in (new_node.out_connectors.keys() - used_connectors):
                for e in state.out_edges(orig_node):
                    if e.src_conn and e.src_conn == conn:
                        # TODO: create alibi node.
                        prune = False
                        break
                if prune:
                    new_node.remove_out_connector(conn)

    inserted_nodes[state] = new_state

    return new_sdfg


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
            mpath = state.memlet_path(e)
            new_nodes = [mpe.src for mpe in mpath if mpe.src not in result]
            result.extend(new_nodes)
            # Memlet path may end in a code node, continue traversing and expanding graph
            queue.extend(new_nodes)

        for e in state.out_edges(node):
            # Special case: OUT_* connectors are not traversed further
            if isinstance(e.src, (nd.EntryNode, nd.ExitNode)) and (e.src_conn is None or e.src_conn.startswith('OUT_')):
                continue
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
