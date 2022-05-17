# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functionality that allows users to "cut out" parts of an SDFG in a smart way (i.e., memory preserving) for localized
testing or optimization.
"""
from collections import deque
import copy
from typing import Deque, Dict, List, Set
from dace import data
from dace.sdfg import nodes as nd, SDFG, SDFGState, utils as sdutil
from dace.sdfg.state import StateSubgraphView


def cutout_state(state: SDFGState, *nodes: nd.Node, make_copy: bool = True) -> SDFG:
    """
    Cut out a subgraph of a state from an SDFG to run separately for localized testing or optimization.
    The subgraph defined by the list of nodes will be extended to include access nodes of data containers necessary
    to run the graph separately. In addition, all transient data containers created outside the cut out graph will
    become global.
    :param state: The SDFG state in which the subgraph resides.
    :param nodes: The nodes in the subgraph to cut out.
    :param make_copy: If True, deep-copies every SDFG element in the copy. Otherwise, original references are kept.
    """
    create_element = copy.deepcopy if make_copy else (lambda x: x)
    sdfg = state.parent
    subgraph: StateSubgraphView = StateSubgraphView(state, nodes)
    subgraph = _extend_subgraph_with_access_nodes(state, subgraph)
    other_arrays = _containers_defined_outside(sdfg, state, subgraph)

    # Make a new SDFG with the included constants, used symbols, and data containers
    sdfg_name = f'{sdfg.name}_cutout_{sdfg.node_id(state)}'
    new_sdfg = SDFG(sdfg_name, sdfg.constants_prop)
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

    # Remove remaining dangling connectors from scope nodes
    for node in inserted_nodes.values():
        used_connectors = set(e.dst_conn for e in new_state.in_edges(node))
        for conn in (node.in_connectors.keys() - used_connectors):
            node.remove_in_connector(conn)
        used_connectors = set(e.src_conn for e in new_state.out_edges(node))
        for conn in (node.out_connectors.keys() - used_connectors):
            node.remove_out_connector(conn)

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
            if isinstance(e.dst, (nd.EntryNode, nd.ExitNode)) and e.dst_conn.startswith('IN_'):
                continue
            mpath = state.memlet_path(e)
            new_nodes = [mpe.src for mpe in mpath if mpe.src not in result]
            result.extend(new_nodes)
            # Memlet path may end in a code node, continue traversing and expanding graph
            queue.extend(new_nodes)

        for e in state.out_edges(node):
            # Special case: OUT_* connectors are not traversed further
            if isinstance(e.src, (nd.EntryNode, nd.ExitNode)) and e.src_conn.startswith('OUT_'):
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
