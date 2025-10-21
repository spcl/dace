from typing import List, Any, Set
import dace

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data as dt, symbolic, SDFG
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
import copy
import itertools


def offset_memlets(state: dace.SDFGState, offsets: List[Any], entry_node: dace.nodes.MapEntry, name: str):
    all_nodes_between = state.all_nodes_between(entry_node, state.exit_node(entry_node))
    all_edges = state.all_edges(*all_nodes_between)
    for edge in set(all_edges):
        if edge.data.data is None:
            continue

        if edge.data.data != name:
            continue

        old_subset = [(b, e, s) for (b, e, s) in edge.data.subset]

        if len(offsets) > len(old_subset):
            for _ in range(len(offsets) - len(old_subset)):
                old_subset.insert(0, (dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(0), 1))

        if len(offsets) == len(old_subset):
            new_subset = [(b + offset, e + offset, s) for offset, (b, e, s) in zip(offsets, old_subset)]
        else:
            raise Exception("hmm")

        edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_subset))


def move_access_node_up(state: dace.SDFGState, access_node: dace.nodes.AccessNode, do_offset_memlets: bool):
    # Scope Lifetime array where size matches the map dimensions
    # Move it to the upper map, make the size match the parent map, replace all subsets
    assert state.in_degree(access_node) == 1

    lvl2_ie = state.in_edges(access_node)[0]
    lvl2_oe = state.out_edges(access_node)[0]
    lvl2_map_entry: dace.nodes.MapEntry = lvl2_ie.src
    lvl1_map_entry: dace.nodes.MapEntry = state.entry_node(lvl2_map_entry)
    # Add node to lvl1
    in_conn = lvl2_ie.src_conn.replace("OUT_", "IN_")
    lvl1_ie = list(state.in_edges_by_connector(lvl2_map_entry, in_conn))[0]

    new_size = [(e + 1 - b) for (b, e, s) in lvl1_ie.data.subset]

    sdfg = state.sdfg
    dataname = access_node.data
    copydesc = sdfg.arrays[dataname]
    sdfg.remove_data(dataname, False)

    sdfg.add_array(name=dataname,
                   shape=tuple(new_size),
                   dtype=copydesc.dtype,
                   storage=copydesc.storage,
                   location=copydesc.location,
                   transient=copydesc.transient,
                   lifetime=copydesc.lifetime)

    # Remove node from lvl2
    state.remove_node(access_node)
    lvl2_memlet = copy.deepcopy(lvl2_oe.data)
    state.add_edge(lvl2_ie.src, lvl2_ie.src_conn, lvl2_oe.dst, lvl2_oe.dst_conn, lvl2_memlet)

    new_access = state.add_access(dataname)

    lvl1_ie_memlet = copy.deepcopy(lvl1_ie.data)
    lvl1_ie_memlet.data = new_access.data
    state.add_edge(new_access, None, lvl1_ie.dst, lvl1_ie.dst_conn, lvl1_ie_memlet)
    state.remove_edge(lvl1_ie)
    state.add_edge(lvl1_ie.src, lvl1_ie.src_conn, new_access, None, copy.deepcopy(lvl1_ie.data))

    new_begs = [b for (b, e, s) in lvl2_ie.data.subset]
    print(new_begs)
    if do_offset_memlets:
        offset_memlets(state, new_begs, lvl2_map_entry, dataname)



def find_next(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for n in state.bfs_nodes(map_entry):
        if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) == map_entry:
            return n
    return None

def offset_tblock_param(sdfg: dace.SDFG, params: Set[str]):
    repldict = {str(p) : 0 for p in params}
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                param_set = {str(s) for s in node.map.params} # If you have a matching map with params
                if param_set == params:
                    #print("Matched:", {node})
                    nodes_between = state.all_nodes_between(node, state.exit_node(node))
                    for edge in state.all_edges(*nodes_between):
                        if edge.src == node or edge.dst == state.exit_node(node) or edge.dst == node or edge.src == state.exit_node(node): # If connects to the entry / exit skio
                            continue
                        # Replace all occurences of the params with 0
                        if edge.data.data is None:
                            continue
                        #print("R1", edge.data)
                        ndata = copy.deepcopy(edge.data)
                        ndata.replace(repldict)
                        edge.data = ndata
                        #print("R2", edge.data)
                    
                    for n2 in nodes_between:
                        if isinstance(n2, dace.nodes.MapEntry):
                            nlist = []
                            
                            for (b,e,s) in n2.map.range:
                                nlist.append(
                                    [
                                        b.subs(repldict),
                                        e.subs(repldict),
                                        s.subs(repldict)
                                    ]
                                )
                            
                            n2.map.range = dace.subsets.Range(nlist)


def move_exit_access_node_down(state: dace.SDFGState, access_node: dace.nodes.AccessNode, do_offset_memlets: bool):
    # Scope Lifetime array where size matches the map dimensions
    # Move it to the upper map, make the size match the parent map, replace all subsets
    assert state.in_degree(access_node) == 1

    lvl2_ie = state.in_edges(access_node)[0]
    lvl2_oe = state.out_edges(access_node)[0]

    # Lvl3 Map -> AN ->  Lvl2 Map -> Lvl1 Map
    #          | Lvl2_ie
    #                | Lvl2_oe

    lvl2_map_exit: dace.nodes.MapExit = lvl2_oe.dst
    lvl2_map_entry = {
        n
        for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and state.exit_node(n) == lvl2_map_exit
    }.pop()
    lvl1_map_entry = state.entry_node(lvl2_map_entry)
    lvl1_map_exit: dace.nodes.MapExit = state.exit_node(lvl1_map_entry)
    # Add node to lvl1
    out_conn = lvl2_oe.dst_conn.replace("IN_", "OUT_")

    lvl1_oe = list(state.out_edges_by_connector(lvl2_map_exit, out_conn))[0]

    new_size = [(e + 1 - b) for (b, e, s) in lvl1_oe.data.subset]

    sdfg = state.sdfg
    dataname = access_node.data
    copydesc = sdfg.arrays[dataname]
    sdfg.remove_data(dataname, False)

    sdfg.add_array(name=dataname,
                   shape=tuple(new_size),
                   dtype=copydesc.dtype,
                   storage=copydesc.storage,
                   location=copydesc.location,
                   transient=copydesc.transient,
                   lifetime=copydesc.lifetime)

    # Remove node from lvl2
    state.remove_node(access_node)
    lvl2_memlet = copy.deepcopy(lvl2_ie.data)
    lvl2_memlet.data = dataname
    state.add_edge(lvl2_ie.src, lvl2_ie.src_conn, lvl2_oe.dst, lvl2_oe.dst_conn, lvl2_memlet)

    new_access = state.add_access(dataname)

    lvl1_oe_memlet = copy.deepcopy(lvl1_oe.data)
    lvl1_oe_memlet.data = new_access.data
    state.add_edge(lvl1_oe.src, lvl1_oe.src_conn, new_access, None, lvl1_oe_memlet)
    state.remove_edge(lvl1_oe)
    state.add_edge(new_access, None, lvl1_oe.dst, lvl1_oe.dst_conn, copy.deepcopy(lvl1_oe.data))

    new_begs = [b for (b, e, s) in lvl2_oe.data.subset]
    print(new_begs)
    if do_offset_memlets:
        offset_memlets(state, new_begs, lvl2_map_entry, dataname)
