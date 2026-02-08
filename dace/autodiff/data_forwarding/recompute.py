# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import List

# DaCe imports
import dace
import dace.sdfg.nodes as nodes
from dace.sdfg import SDFG, SDFGState, state as dstate
from dace.sdfg.state import LoopRegion

# Autodiff imports
from dace.autodiff.base_abc import AutoDiffException
import dace.autodiff.utils as ad_utils


def resolve_overwrite_with_recomputation(
    recomputation_nsdfg: nodes.NestedSDFG,
    forward_state: SDFGState,
    backward_state: SDFGState,
    target_an: nodes.AccessNode,
    target_node: nodes.Node,
    starting_edge: dstate.MultiConnectorEdge,
):
    """
    Experimental! Use recomputation in the backward pass to compute data that was overwritten in the forward pass.
    """

    # Add the nsdfg where it is required
    _connect_recomputation_nsdfg(forward_state=forward_state,
                                 backward_state=backward_state,
                                 nsdfg=recomputation_nsdfg,
                                 target_an=target_an,
                                 target_node=target_node,
                                 starting_edge=starting_edge)


def _connect_recomputation_nsdfg(bwd_generator: 'BackwardPassGenerator', forward_state: SDFGState,
                                 backward_state: SDFGState, target_an: nodes.AccessNode, target_node: nodes.Node,
                                 nsdfg: nodes.NestedSDFG, starting_edge: dstate.MultiConnectorEdge):
    """

    """
    # Connect all the SDFG inputs to the nested SDFG
    # First, add the nested sdfg
    for input in nsdfg.in_connectors.keys():
        # For each argument
        input_name = input if "recomputation_" not in input else input[14:]

        # Get the first instance of this AN in the SDFG
        first_instance = None
        for node, parent in bwd_generator.forward_sdfg.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode) and node.data == input:
                first_instance = node
                first_node_state = parent
                break

        assert first_instance

        new_an = nodes.AccessNode(input_name)
        backward_state.add_node(new_an)

        # Create a memlet passing all the data to the nested-SDFG
        memlet = bwd_generator.forward_sdfg.make_array_memlet(input_name)

        # Add the connection to the nested SDFG
        backward_state.add_edge(new_an, None, nsdfg, input, memlet)

    # Write the data to a new access node in the backward state
    # Add a new AccessNode and array to the forward pass
    # First, check if a recomputed array with this name already exists
    if "recomputed_" + target_an.data not in bwd_generator.backward_sdfg.arrays:
        new_recomp_node_name = "recomputed_" + target_an.data
    else:
        i = 0
        while True:
            if f"recomputed_{i}_" + target_an.data not in bwd_generator.backward_sdfg.arrays:
                new_recomp_node_name = f"recomputed_{i}_" + target_an.data
                break
            i += 1

    # Get the new array shape
    # This will be the shape of the current array
    shape: List[int] = list(bwd_generator.forward_sdfg.arrays[target_an.data].shape)

    # Add the array descriptor and AccessNode to the forward state
    original_desc = target_an.desc(forward_state)
    backward_state.sdfg.add_array(
        name=new_recomp_node_name,
        shape=shape,
        dtype=original_desc.dtype,
        transient=True,
    )
    new_recomp_node = backward_state.add_access(new_recomp_node_name)
    new_recomp_node.setzero = True

    # Create a memlet passing all the data to the nested-SDFG
    memlet = bwd_generator.forward_sdfg.make_array_memlet(new_recomp_node.data)

    nsdfg_out_conn = list(nsdfg.out_connectors.keys())
    assert len(nsdfg_out_conn) == 1
    nsdfg_out_conn = nsdfg_out_conn[0]

    # Connect the output of the NestedSDFG
    backward_state.add_edge(nsdfg, nsdfg_out_conn, new_recomp_node, None, memlet)

    # Connect the new AccessNode to the required computation
    bwd_generator._connect_forward_accessnode_not_overwritten(forward_state=forward_state,
                                                              backward_state=backward_state,
                                                              forward_node=target_an,
                                                              target_node=target_node,
                                                              starting_edge=starting_edge,
                                                              replicated_node=new_recomp_node)


def _prune_descendants_recomputation_nsdfg(forward_state: SDFGState, target_an: nodes.AccessNode,
                                           nsdfg: nodes.NestedSDFG):
    """
    1: From this Nested-SDFG, we remove everything that will be executed after the target access node to be recomputed
    2: Prune the unnecessary computation inside the forward state
        Note: this is even necessary sometimes since the output could be overwritten in the same state
    """

    # 1
    # Get the states order for the nested_sdfg
    states_order: List[SDFGState] = ad_utils.get_state_topological_order(nsdfg.sdfg)
    state_index = states_order.index(forward_state)
    descendant_states: List[SDFGState] = states_order[state_index:]
    assert descendant_states.pop(0) == forward_state

    # Check if the target state is within a loop
    target_within_loop, target_loop = ad_utils.state_within_loop(forward_state)

    # We will save the states that are within the same loop because they require special treatement
    same_loop_states: List[SDFGState] = []
    for state in descendant_states:
        # We want to avoid removing the descendant states that are inside the same loop region
        if target_within_loop:
            descendant_within_loop, descendant_loop = ad_utils.state_within_loop(state)
            if descendant_within_loop and descendant_loop == target_loop:
                # If the state is within the same loop, we don't remove it
                same_loop_states.add(state)
                continue

        # Remove the state from the nested_sdfg
        parent = state.parent_graph
        parent.remove_node(state)

    # Cleanup empty LoopRegions if any
    for node in nsdfg.sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion) and len(node.nodes()) == 0:
            parent = node.parent_graph
            parent.remove_node(node)

    # 2
    # Within the same state
    if target_within_loop:
        # For now we keep all of the computation inside the loop
        # TODO: if there is an overwrite to the same array in the decendnat computation
        # We need to make a special case for the last iteration of the loop where the
        # else branch of this if is executed and a special version of the loop is added
        raise AutoDiffException("Recomputation with overwrites within loops is not supported yet.")
    else:
        # If the target state is not within a loop
        # We remove all the descendant computation from the graph

        # Do a reverse bfs to get all the necessary computation
        backward_nodes = {n for e in forward_state.edge_bfs(target_an, reverse=True) for n in [e.src, e.dst]}

        # Remove everything else
        descendant_nodes = set(forward_state.nodes()) - backward_nodes

        for node in descendant_nodes:
            if node is not target_an:
                forward_state.remove_node(node)


def _prune_recomputation_sdfg(forward_state: SDFGState, target_an: nodes.AccessNode, nsdfg: nodes.NestedSDFG):
    """
    1: From this Nested-SDFG, we remove everything that will be executed after the target access node to be recomputed
    2: Prune the unnecessary computation inside the forward state
        Note: this is even necessary sometimes since the output could be overwritten in the same state
    3: TODO: From the target access node, we go backward in the graph and see what elements are required to get this array
    """

    # 1 and 2
    _prune_descendants_recomputation_nsdfg(forward_state=forward_state, target_an=target_an, nsdfg=nsdfg)


def _rename_descriptors_for_recomputation_nsdfg(forward_sdfg: SDFG, nsdfg: nodes.NestedSDFG):
    """
    """
    # Get all the nodes to rename in the NestedSDFG
    to_rename = []
    for inp in nsdfg.in_connectors:
        for node, parent in nsdfg.sdfg.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode) and node.data == inp and parent.in_degree(node) > 0:
                # This is an input that will be written to in the SDFG we need to rename it
                to_rename.append(inp)
                break

    if len(to_rename) > 0:
        # Add a new state to copy the data at the start of the SDFG
        initi_state = nsdfg.sdfg.add_state_before(nsdfg.sdfg.start_state, label=f"init_{nsdfg.label}")

    # Rename the descriptors in the nested SDFG in addition to the in connector
    for name in to_rename:
        # Create a new array
        new_name = f"recomputation_{name}"

        # Change the accessnodes in the NestedSDFG
        for node, parent in nsdfg.sdfg.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode) and node.data == name:
                node.data = new_name

        # Change the memlets in the SDFG
        for edge, parent in nsdfg.sdfg.all_edges_recursive():
            # Skip interstate edges
            if isinstance(edge.data, dace.InterstateEdge):
                continue

            if edge.data.data == name:
                edge.data.data = new_name

        # Add the desciptor
        old_desc = nsdfg.sdfg.arrays[name]
        new_desc = copy.deepcopy(old_desc)

        # Check if this is the output of the recomputation block
        if name not in nsdfg.out_connectors:
            new_desc.transient = True
        else:
            new_desc.transient = False

        nsdfg.sdfg.add_datadesc(name=new_name, datadesc=new_desc)

        # Add a copy operation between the input node and the new descriptor
        input_node = nodes.AccessNode(name)
        new_node = nodes.AccessNode(new_name)
        initi_state.add_node(input_node)
        initi_state.add_node(new_node)

        # Add memory copy edge
        initi_state.add_edge(input_node, None, new_node, None, forward_sdfg.make_array_memlet(name))

        # Change the output if necessary
        if name in nsdfg.out_connectors:
            nsdfg.remove_out_connector(name)
            nsdfg.add_out_connector(new_name)


def get_recomputation_nsdfg(bwd_generator: 'BackwardPassGenerator', forward_state: SDFGState,
                            target_an: nodes.AccessNode) -> nodes.NestedSDFG:
    """
    Given an AccessNode for data that needs to be forwarded from the forward pass to the backward pass,
    Return a nested SDFG that recomputes this data from input data.
    """
    nsdfg_label = "recomputation_nsdfg_" + target_an.data

    # Initially, we will replicate the whole SDFG into a Nested-SDFG and connect it
    # TODO: we likely need a copy of the SDFG before starting AD if separate_sdfgs
    nsdfg = nodes.NestedSDFG(label=nsdfg_label,
                             sdfg=copy.deepcopy(bwd_generator.sdfg),
                             inputs=bwd_generator.sdfg.arg_names,
                             outputs=[target_an.data])

    # We need to make sure the output inside the NestedSDFG is not a transient (anymore)
    nsdfg.sdfg.arrays[target_an.data].transient = False

    # Find the same target node and state in the nsdfg
    nsdfg_forward_state: SDFGState = None
    nb_occurrences = 0
    for state in nsdfg.sdfg.states():
        if state.label == forward_state.label:
            nsdfg_forward_state = state
            nb_occurrences += 1

    # Sanity check
    assert nb_occurrences == 1
    assert nsdfg_forward_state

    # Find the target AccessNode within the state
    nsdfg_target_node: nodes.AccessNode = None
    nb_occurrences = 0
    for node in nsdfg_forward_state.nodes():
        if isinstance(node, nodes.AccessNode) and node.data == target_an.data and nsdfg_forward_state.node_id(
                node) == forward_state.node_id(target_an):
            nsdfg_target_node = node
            nb_occurrences += 1

    # Sanity check
    assert nb_occurrences == 1
    assert nsdfg_target_node

    _prune_recomputation_sdfg(nsdfg=nsdfg, forward_state=nsdfg_forward_state, target_an=nsdfg_target_node)

    # Change descriptors if the inputs are written to
    _rename_descriptors_for_recomputation_nsdfg(forward_sdfg=bwd_generator.sdfg, nsdfg=nsdfg)

    return nsdfg
