# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
from typing import Set, Dict

from dace.sdfg.graph import MultiConnectorEdge
from dace import SDFGState

SLICE_SUFFIX = "slice"


def _get_new_connector_name(edge: MultiConnectorEdge, repldict: Dict[str, str], other_repldict: Dict[str, str],
                            state: SDFGState, nsdfg_node: dace.nodes.NestedSDFG) -> str:
    """
    Determine new connector name for an edge based on data access patterns.
    Following the description in the dealias routine

    Args:
        edge: The edge containing data access information
        repldict: Dictionary of existing replacements to avoid name conflicts
        state: The SDFG state containing the edge

    Returns:
        str: New connector name - either the original array name (for full access)
             or a unique slice name (for partial access)
    """
    nested_sdfg: dace.SDFG = nsdfg_node.sdfg
    arr = state.sdfg.arrays[edge.data.data]
    data_shape = arr.shape

    # Full subset?
    full_range = dace.subsets.Range([(0, dim - 1, 1) for dim in data_shape])
    is_complete_subset = edge.data.subset == full_range

    combined_repldict = repldict | other_repldict

    array_name = set(nested_sdfg.arrays.keys()).union(combined_repldict.values()).union(nested_sdfg.symbols)
    if is_complete_subset:
        candidate_name = dace.data.find_new_name(edge.data.data, array_name)
        return candidate_name
    else:
        candidate_name = dace.data.find_new_name(f"{edge.data.data}_{SLICE_SUFFIX}", array_name)
        return candidate_name


def find_readable_connector_names_for_nested_sdfgs(sdfg: dace.SDFG):
    """
    Remove aliasing in nested SDFG connectors by replacing temporary names with meaningful ones.

    Temporary connector names (e.g., tmpxceX) are replaced with names that reflect the actual data
    being accessed (e.g. <data_name>_slice_<id> or <data_name>). Depending on applicability

    The function handles two main cases:
    1. Full array access: A[::] -> connector gets named 'A'
    2. Partial array access: A[i:j] -> connector gets named 'A_slice_<id>' <id> is needed in
    case multiple slices of the same array are used.


    Args:
        sdfg (dace.SDFG): Modified in-place.
    """

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):

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
                    # Skip "__return"
                    if in_edge.data is not None and in_edge.data.data == "__return":
                        continue
                    if in_edge.data is not None and in_edge.data.data != in_edge.dst_conn:
                        new_connector = _get_new_connector_name(in_edge, input_repldict, output_repldict, state,
                                                                node.sdfg)
                        input_repldict[in_edge.dst_conn] = new_connector

                for out_edge in out_edges:
                    if out_edge.data is not None and out_edge.data.data == "__return":
                        continue
                    if out_edge.data is not None and out_edge.data.data != out_edge.src_conn:
                        # If the name exists in the input_repldcit reuse
                        # to avoid having input dict having a name that is a subset
                        if out_edge.src_conn in input_repldict:
                            new_connector = input_repldict[out_edge.src_conn]
                            output_repldict[out_edge.src_conn] = new_connector
                        else:
                            new_connector = _get_new_connector_name(out_edge, output_repldict, input_repldict, state,
                                                                    node.sdfg)
                            output_repldict[out_edge.src_conn] = new_connector

                # Replace connectors rm tmpxceX connector with A
                for dst_name in set(input_repldict.keys()):
                    rmed = node.remove_in_connector(dst_name)
                    assert rmed, f"Could not removed in connector that is not used anymore: {dst_name}"
                for dst_name in set(output_repldict.keys()):
                    rmed = node.remove_out_connector(dst_name)
                    assert rmed, f"Could not removed out connector that is not used anymore: {dst_name}"
                for src_name in set(input_repldict.values()):
                    added = node.add_in_connector(src_name, force=True)
                    assert added, f"Could add the new in connector to the nested sdfg: {src_name}"
                for src_name in set(output_repldict.values()):
                    added = node.add_out_connector(src_name, force=True)
                    assert added, f"Could add the new out connector to the nested sdfg: {src_name}"

                # Update edges
                for in_edge in state.in_edges(node):
                    if in_edge.dst_conn in input_repldict:
                        in_edge.dst_conn = input_repldict[in_edge.dst_conn]
                for out_edge in state.out_edges(node):
                    if out_edge.src_conn in output_repldict:
                        out_edge.src_conn = output_repldict[out_edge.src_conn]

                # Replace the data containers
                # If data / access nodes are not manually changed before hand
                # Dace will try to assign to scalars from a symbolic value and crash the thing
                replace_dict = (input_repldict | output_repldict)
                added_arrays: Set[str] = set()
                for dst_name, src_name in replace_dict.items():
                    desc: dace.data.Data = node.sdfg.arrays[dst_name]
                    added_arrays.add(src_name)
                    if src_name in node.sdfg.arrays:
                        assert src_name in added_arrays, f"{src_name} is in sdfg.arrays but has not been added by dealias for replacements: {replace_dict}."
                    else:
                        node.sdfg.remove_data(dst_name, validate=False)
                        node.sdfg.add_datadesc(name=src_name, datadesc=desc, find_new_name=False)

                # Necessary for DaCe to try assign the value to the missing access node from a tasklet
                for inner_state in node.sdfg.all_states():
                    for inner_node in inner_state.nodes():
                        if isinstance(inner_node, dace.nodes.AccessNode) and inner_node.data in replace_dict:
                            inner_node.data = replace_dict[inner_node.data]

                node.sdfg.replace_dict(repldict=replace_dict)

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                find_readable_connector_names_for_nested_sdfgs(node.sdfg)
