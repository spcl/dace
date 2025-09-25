# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
from typing import Set

def _get_new_connector_name(edge, repldict, state):
    """Determine new connector name for an edge."""
    arr = state.sdfg.arrays[edge.data.data]
    data_shape = arr.shape

    # Full subset?
    full_range = dace.subsets.Range([(0, dim - 1, 1) for dim in data_shape])
    is_complete_subset = edge.data.subset == full_range

    if is_complete_subset:
        return edge.data.data
    else:
        candidate_name = f"{edge.data.data}_slice"
        i = 1
        while f"{candidate_name}_{i}" in repldict.values():
            i += 1
        return f"{candidate_name}_{i}"

def dealias(sdfg: dace.SDFG):
    recurse_in : Set[dace.nodes.NestedSDFG] = set()

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                recurse_in.add(node)

                in_edges = state.in_edges(node)
                out_edges = state.out_edges(node)

                # Gather all replacements we need
                # E.g.                      A[::] -> tmpxceX (NestedSDFG)
                # Needs to be replaced with A[::] -> A_slice (NestedSDFG)
                # A_slice is chosen if the subset is different than the complete shape A
                # Otherwise A is chosen
                # Also consider the case where A[i] -> tmp1 (NestedSDFG)
                #                              A[j] -> tmp2 
                # In this case we need not map them to A twice but to A_slice1, A_slice2
                input_repldict = dict()
                output_repldict = dict()
                for in_edge in in_edges:
                    if in_edge.data is not None and in_edge.data.data != in_edge.dst_conn:
                        new_connector = _get_new_connector_name(in_edge, input_repldict, state)
                        input_repldict[in_edge.dst_conn] = new_connector

                for out_edge in out_edges:
                    if out_edge.data is not None and out_edge.data.data != out_edge.src_conn:
                        new_connector = _get_new_connector_name(out_edge, output_repldict, state)
                        output_repldict[out_edge.src_conn] = new_connector

                # Replace connectors rm tmpxceX connector with A
                for dst_name in set(input_repldict.keys()):
                    rmed = node.remove_in_connector(dst_name)
                    assert rmed
                for dst_name in set(output_repldict.keys()):
                    rmed = node.remove_out_connector(dst_name)
                    assert rmed
                for src_name in set(input_repldict.values()):
                    added = node.add_in_connector(src_name)
                    assert added
                for src_name in set(output_repldict.values()):
                    added = node.add_out_connector(src_name)
                    assert added

                # Update edges
                for in_edge in state.in_edges(node):
                    if in_edge.dst_conn in input_repldict:
                        state.remove_edge(in_edge)
                        state.add_edge(
                            in_edge.src,
                            in_edge.src_conn,
                            in_edge.dst,
                            input_repldict[in_edge.dst_conn],
                            copy.deepcopy(in_edge.data)
                        )
                for out_edge in state.out_edges(node):
                    if out_edge.src_conn in output_repldict:
                        state.remove_edge(out_edge)
                        state.add_edge(
                            out_edge.src,
                            output_repldict[out_edge.src_conn],
                            out_edge.dst,
                            out_edge.dst_conn,
                            copy.deepcopy(out_edge.data)
                        )

                # Replace the data containers
                # If data / access nodes are not manually changed before hand
                # Dace will try to assign to scalars from a symbolic value and crash the thing
                for dst_name, src_name in (input_repldict | output_repldict).items():
                    desc : dace.data.Data = node.sdfg.arrays[dst_name]
                    node.sdfg.remove_data(dst_name, validate=False)
                    node.sdfg.add_datadesc(name=src_name, datadesc=desc, find_new_name=False)

                for dst_name, src_name in (input_repldict | output_repldict).items():
                    assert src_name in node.sdfg.arrays
                    assert dst_name not in node.sdfg.arrays
