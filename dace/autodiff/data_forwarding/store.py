# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import List, Tuple
import sympy as sp

# DaCe imports
import dace.sdfg.nodes as nodes
from dace import dtypes, data as dt, symbolic
from dace.sdfg import SDFGState, graph as dgraph, state as dstate
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion

# Autodiff imports
from dace.autodiff.base_abc import AutoDiffException
import dace.autodiff.utils as ad_utils


def resolve_overwrite_with_store(bwd_generator: 'BackwardPassGenerator', forward_state: SDFGState,
                                 backward_state: SDFGState, forward_node: nodes.AccessNode, target_node: nodes.Node,
                                 starting_edge: dstate.MultiConnectorEdge):
    """
    Given the AccessNode pointing to the data required by the backward pass,
    We will save the values of this array in a new array and forward it to the backward pass.
    """

    # Modify the forward pass to save the data in a new array
    new_stored_array, memlets = _store_data(bwd_generator=bwd_generator,
                                            forward_state=forward_state,
                                            backward_state=backward_state,
                                            forward_an=forward_node,
                                            target_node=target_node,
                                            edge=starting_edge)

    # Check if this data needs to be forwarded through NestedSDFGs
    if bwd_generator.separate_sdfgs or forward_state.sdfg.parent_sdfg is not None:
        # We need to make sure the new array is forwarded to the backward SDFG
        if new_stored_array.data not in bwd_generator.backward_input_arrays:
            # If the data is needed inside a NestedSDFG
            # This will make sure the added array is correctly forwarded
            # and an in connector to the NestedSDFG is added
            data_desc = new_stored_array.desc(forward_state)
            bwd_generator.backward_input_arrays[new_stored_array.data] = data_desc

    # Connect the new array to the target node
    _connect_stored_data_to_target(bwd_generator=bwd_generator,
                                   forward_state=forward_state,
                                   backward_state=backward_state,
                                   source_node=new_stored_array,
                                   forward_node=forward_node,
                                   starting_edge=starting_edge,
                                   memlets=memlets,
                                   target_node=target_node)


def _store_data(bwd_generator: 'BackwardPassGenerator', forward_state: SDFGState, backward_state: SDFGState,
                forward_an: nodes.AccessNode, target_node: nodes.Node,
                edge: dgraph.MultiConnectorEdge) -> Tuple[nodes.AccessNode, List[Memlet]]:
    """
    Given an edge leading an AccessNode or a map to the target node in the forward state,
    add a path from the connector for this AccessNode to store its values for all iterations.
    This can increase the dimension of the array. i.e. the size of the stored array is
    greater or equal to the size of the original array.

    :param edge: the edge connecting the AccessNode to save data from to a map node.
    :return: the new AccessNode which contains the stored data,
             a list of memlets connecting an assign tasklet to this new AccessNode.
    """

    # Get the connector and edge to save
    if isinstance(edge.src, nodes.AccessNode) and edge.src is not forward_an:

        # Get the incoming edge to this AccessNode
        in_edges = forward_state.in_edges(edge.src)

        # There should only be one incoming edge
        assert len(in_edges) == 1

        # Get the memlet path for the edge incoming to this AccessNode
        memlet_path = forward_state.memlet_path(in_edges[0])

        # The start of this path should be the forward AccessNode
        assert forward_an is memlet_path[0].src

        # The last edge in the memlet path has the connector we want to save
        edge = memlet_path[-1]

    # Add a new AccessNode and array to the forward pass
    # First, check if a stored array with this name already exists
    new_store_node_name = forward_state.sdfg._find_new_name("stored_" + forward_an.data)

    # Get the new array shape
    # This will be the shape of the current array
    shape: List[int] = list(bwd_generator.sdfg.arrays[forward_an.data].shape)

    # If the shape is an expression:
    free_symbols_dict = {sym: None for sym in bwd_generator.sdfg.free_symbols}
    if any(symbolic.issymbolic(s, free_symbols_dict) for s in shape):
        # Otherwise, replace all the loop dependent allocations with the max length of the loop
        # For example, an array of size [i+1] in a range(2, 10) loop will be stored in a [10, 10] array (1)
        # Additionally, an array of size [32-i] in the same loop will be stored in a [10, 30]  (2)
        loops = _get_all_enclosing_loops(forward_state)

        if len(loops) > 0:
            # Loop over the shape dimensions
            for i, s in enumerate(shape):
                if ad_utils.shape_has_symbols_to_replace(bwd_generator.sdfg, s):
                    loop_size, loop_index = _get_symbol_upper_bound_from_loop(bwd_generator, s, loops)
                    # Replace the symbol with the loop size and evaluate the expression
                    # Check if loop size can be converted to an integer
                    loop_index_sym = symbolic.pystr_to_symbolic(loop_index)
                    loop_size_sym = loop_size if isinstance(loop_size, int) else symbolic.pystr_to_symbolic(loop_size)
                    shape[i] = s.subs(loop_index_sym, loop_size_sym)

    # Plus the size of any enclosing loops
    enclosed, _ = ad_utils.state_within_loop(forward_state=forward_state)
    nb_enclosing_loops = 0
    loop_param_list = []
    if enclosed:
        # Get all enclosing loops
        all_encolsing_loops = _get_all_enclosing_loops(forward_state=forward_state)
        nb_enclosing_loops = len(all_encolsing_loops)
        # Get the size of each loop and add it to the list
        for loop in all_encolsing_loops:
            # Get the end of the loop
            start, end = ad_utils.extract_loop_region_info(loop)

            # Check if the loop is increasing or decreasing
            # First, try to convert the strings to ints if possible
            # Note that we look for the start or end of the loop
            # And not the size of the loop.
            # This is because we access using the loop indices
            # Using the loop sizes instead would require shifting accesses
            _, new_dim = ad_utils.get_loop_end(start, end, loop)

            # First we check if the new dimension contains symbols
            # These will need to be replaced with scalars for correct allocation
            # The sdfg symbols are allowed to be in the shape
            if ad_utils.shape_has_symbols_to_replace(bwd_generator.sdfg, new_dim):
                # Take the expression to sympy for easier processing
                if isinstance(new_dim, str):
                    new_dim = symbolic.pystr_to_symbolic(new_dim)

                # Try to replace the symbols with the loop size
                loop_size, loop_index = _get_symbol_upper_bound_from_loop(bwd_generator, new_dim, all_encolsing_loops)
                loop_index_sym = symbolic.pystr_to_symbolic(loop_index)
                loop_size_sym = loop_size if isinstance(loop_size, int) else symbolic.pystr_to_symbolic(loop_size)
                new_dim = new_dim.subs(loop_index_sym, loop_size_sym)
            shape.insert(0, new_dim)
            loop_param_list.insert(0, loop.loop_variable)

    # Add the array descriptor and AccessNode to the forward state
    original_desc = forward_an.desc(forward_state)

    # We make a special case for a memlet of the type A[i, j] in an i, j loop
    # In this case we only need an array of the same size as the forward node
    if enclosed and edge.data.data == forward_an.data and len(edge.data.subset) == nb_enclosing_loops:
        # Check that the memlet subset matches perfectly the order of loop nest
        # Make sure the subset elements are (i,i,1) and (j,j,1)
        # Then check if this matches the loop indices
        if all(
                str(subset[0]) == loop_param_list[i] and subset[0] == subset[1] and subset[2] == 1
                for i, subset in enumerate(edge.data.subset)):
            # We only use the loop accesses
            # Both should work since shape[:nb_enclosing_loops] == shape[nb_enclosing_loops:]
            shape = shape[nb_enclosing_loops:]

            # We want to build the memlet as if this was not in a a loop
            nb_enclosing_loops = 0

    forward_state.sdfg.add_array(
        name=new_store_node_name,
        shape=shape,
        dtype=original_desc.dtype,
        transient=True,
    )
    new_store_node = forward_state.add_access(new_store_node_name)

    # Connect the edge source and connector to the new access node
    # We will save the memlets we create and return them
    # This is useful to make the connections for the backward state
    memlets_stack = []

    # The loop accesses will be the same within the state
    # Prepare them for all edges
    loop_access = ','.join([f'{loop_param_list[i]}' for i in range(nb_enclosing_loops)])

    # In the other cases, we need to route the storing through maps
    all_edges = ad_utils.get_all_path_edges(forward_state, forward_an, edge)

    # Get the map nest memlet information
    start_range, param_list, shape_list, param_dict = ad_utils.get_map_nest_information(all_edges)

    # The parameters to add for the current memlet in the loop
    # At first we will use all of the parameters that are used in the memlet
    # param_dict = {key: val for key, val in param_dict.items() if key in edge.data.free_symbols}
    new_param_dict = {}

    # Iterate through the subset
    for index, element in enumerate(edge.data.subset):
        if str(element[0]) in edge.data.free_symbols and str(element[0]) in param_dict.keys():
            # Add the range from the param_dict
            new_param_dict.update({str(element[0]): param_dict[str(element[0])]})
        else:
            # Add the range from the param_dict
            new_param_dict.update({index: element})

    params_to_add = new_param_dict
    # First, we need to add an assign tasklet
    assign_tasklet_node, assign_tasklet_node_out_connector = _get_assign_tasklet(forward_state=forward_state,
                                                                                 node=forward_an,
                                                                                 stored_node=new_store_node,
                                                                                 last_edge=edge,
                                                                                 loop_iterators=loop_access)

    # Start iterating
    previous_node = assign_tasklet_node
    previous_node_out_connector = assign_tasklet_node_out_connector
    map_exist = None
    for edge in reversed(all_edges):
        if isinstance(edge.src, nodes.MapEntry):
            # Get the corresponding map exit
            map_exist = _find_map_exist_for_map_entry(map_entry=edge.src, state=forward_state)

            # Add the Connectors to the map
            map_exit_in_connector = f"IN_stored_{new_store_node.label}"
            map_exit_out_connector = f"OUT_stored_{new_store_node.label}"
            added = map_exist.add_in_connector(map_exit_in_connector)
            assert added
            added = map_exist.add_out_connector(map_exit_out_connector)
            assert added

            # Prepare the memlet data for this edge
            access_list = []
            for key, val in new_param_dict.items():
                if isinstance(key, str):
                    if key in params_to_add.keys():
                        access_list.append(key)
                    else:
                        start = val[0]
                        end = val[1]
                        access_list.append(f'{start}:{end}')
                elif isinstance(key, int):
                    start = val[0]
                    end = val[1] + 1
                    access_list.append(f'{start}:{end}')
                else:
                    raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

            in_state_access = ','.join(access_list)

            memlet_data = Memlet(
                expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
                    expr=f"{new_store_node.data}[{in_state_access}]")

            # Save the memlet for later
            memlets_stack.append(memlet_data)

            # Connect the previous node to this map exist
            forward_state.add_edge(previous_node, previous_node_out_connector, map_exist, map_exit_in_connector,
                                   memlet_data)

            previous_node = map_exist
            previous_node_out_connector = map_exit_out_connector

            # Remove the parameters seen in the current map
            # Since they will become out of scope in the next iteration
            params_to_add = {}
            for key, val in new_param_dict.items():
                if isinstance(key, str):
                    if key not in edge.src.params:
                        start = val[0]
                        end = val[1]
                        params_to_add.update({key: (start, end)})
                elif isinstance(key, int):
                    params_to_add.update({key: val})
                else:
                    raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

        else:
            # Prepare the memlet data for this edge
            access_list = []
            for key, val in new_param_dict.items():
                if isinstance(key, str):
                    start = val[0]
                    end = val[1]
                    access_list.append(f'{start}:{end}')
                elif isinstance(key, int):
                    start = val[0]
                    end = val[1] + 1
                    access_list.append(f'{start}:{end}')
                else:
                    raise AutoDiffException("Found unexepected type in memlet parameters dictionary")

            in_state_access = ','.join(access_list)

            # Get the memlet data for the connection between the last map exit and the new store AccessNode
            memlet_data = Memlet(
                expr=f"{new_store_node.data}[{loop_access},{in_state_access}]") if loop_access else Memlet(
                    expr=f"{new_store_node.data}[{in_state_access}]")

            memlets_stack.append(memlet_data)

            # This should be the last connection
            forward_state.add_edge(previous_node, previous_node_out_connector, new_store_node, None, memlet_data)
            break

    # We need to add an empty memlet from the new store AccessNode to make sure the data is stored before it is
    # potentially altered
    # First, we check if this can be avoided
    # We do a BFS exploration to see if the data we are trying to store is overwritten within the same execution state
    bfs_nodes = list(forward_state.bfs_nodes(source=forward_an))

    # We make sure that views are also compared with their original array to check for conflicts
    conflict_arrays = [forward_an.data]
    # Check if the access node is a view
    if isinstance(forward_an.desc(forward_state), dt.View):
        # Get the original array name
        viewed_array = next(forward_state.in_edges_by_connector(forward_an, "views")).data.data
        conflict_arrays.append(viewed_array)

    if any(isinstance(n, nodes.AccessNode) and n.data in conflict_arrays and n is not forward_an for n in bfs_nodes):
        to_connect = []
        for out_edge in forward_state.out_edges(forward_an):
            # Get the destination of the edge
            dst = out_edge.dst
            if not isinstance(dst, nodes.MapEntry) and dst is not assign_tasklet_node:
                # This will not be necessary for maps since the storing is added to the same map
                # We also don't connect the newly created assign tasklet to avoid creating a cycle
                if dst not in to_connect:
                    # We only need to make a single connection to the new stored data
                    to_connect.append(dst)

        for node in to_connect:
            # Connect the new store AccessNode to assure the store happens first
            # If there isn't already a connnection between these two nodes
            if not any(e.dst == node for e in forward_state.out_edges(new_store_node)):
                forward_state.add_edge(new_store_node, None, node, None, Memlet())

        # Another case for making sure data is stored before it is altered is when the map we save from writes itself to the data we want to save
        # In this case this would depend on the codegen order of the tasklets within the map and is thus not safe
        # Detect if this is the case
        if map_exist:
            # Check if this map exit writes to the data we want to save
            if any(
                    isinstance(e.dst, nodes.AccessNode) and e.dst.data == forward_an.data
                    for e in forward_state.out_edges(map_exist)):
                # Get the map entry of this map exit
                tasklet_in_edges = forward_state.in_edges(assign_tasklet_node)
                assert len(tasklet_in_edges) == 1
                tasklet_in_edge = tasklet_in_edges[0]

                # Safety check
                if not isinstance(tasklet_in_edge.src, nodes.MapEntry):
                    raise AutoDiffException(
                        "The map exit writes to the data we want to save, but the storing strcuture is not what we expect"
                    )

                # Get all the edges coming out of this specific in connector
                collusion_edges = [
                    e for e in forward_state.out_edges(tasklet_in_edge.src)
                    if e.src_conn == tasklet_in_edge.src_conn and e.dst != assign_tasklet_node
                ]

                # We need to add an empty memlet from the new store tasklet to everything else that reads from that connector
                for out_edge in collusion_edges:
                    forward_state.add_edge(assign_tasklet_node, None, out_edge.dst, None, Memlet())

    return new_store_node, memlets_stack


def _connect_stored_data_to_target(bwd_generator: 'BackwardPassGenerator', forward_state: SDFGState,
                                   backward_state: SDFGState, source_node: nodes.AccessNode,
                                   forward_node: nodes.AccessNode, target_node: nodes.Node, memlets: List[Memlet],
                                   starting_edge: dgraph.MultiConnectorEdge):
    """
        Connect the source node to the sink target node (both in the backawrd state) through a set of maps using the parameter memelets.
        We use the forward_sink_edge to track which maps to make this connection through.
        :param source_node: the source node of the new memlet path
        :param sink_node: the sink node of the new memlet path
        :param memlets: the set of memlets to use for the edges in the path
        :param forward_sink_edge: the sink edge connecting the original nodes in the forward state
        """
    # First, if the stored data is not already in the sdfg descriptors, add it
    # This is the case for NestedSDFGs
    if source_node.data not in backward_state.sdfg.arrays:
        # Get the data descriptor from the original sdfg
        data_desc = copy.deepcopy(bwd_generator.sdfg.arrays[source_node.data])
        data_desc.transient = False  # The stored data will be forwarded
        backward_state.sdfg.add_datadesc(source_node.data, data_desc)

    # Get the memlet path from the forward state
    all_edges = ad_utils.get_all_path_edges(forward_state, forward_node, starting_edge)
    assert len(all_edges) > 0

    # We will iterate and connect parent -> child
    reversed_child_node = bwd_generator.reverse_map[target_node]
    child_node = reversed_child_node
    child_node_in_connector = all_edges[-1].dst_conn

    # Iterate through the maps in the path in reverse
    for edge in reversed(all_edges):
        edge_src = edge.src
        if isinstance(edge_src, nodes.MapEntry):
            # Get the correponding map exist
            map_exit = _find_map_exist_for_map_entry(map_entry=edge_src, state=forward_state)

            # Use the lookup table to get the map entry in the backward state corresponding to this map exist in the forward state
            # Sanity check: this map entry should already exist in the backward state
            assert map_exit in bwd_generator.reverse_map
            bwd_map_entry = bwd_generator.reverse_map[map_exit]

            # Get a new connector id
            next_conn = bwd_map_entry.next_connector()

            # Add a new in connector to the mapexit
            parent_node_in_connector = "IN_stored_" + source_node.data + "_" + next_conn
            added = bwd_map_entry.add_in_connector(parent_node_in_connector)
            assert added

            # Add a new out connector to the mapexit
            parent_node_out_connector = "OUT_stored_" + source_node.data + "_" + next_conn
            added = bwd_map_entry.add_out_connector(parent_node_out_connector)
            assert added

            memlet_data = copy.deepcopy(memlets.pop(0))

            # Add the edge with the corresponding memlet
            backward_state.add_edge(bwd_map_entry, parent_node_out_connector, child_node, child_node_in_connector,
                                    memlet_data)

            child_node = bwd_map_entry
            child_node_in_connector = parent_node_in_connector

        if isinstance(edge_src, nodes.AccessNode):
            # The connection from the stored data will be made here
            assert edge_src == forward_node
            memlet_data = copy.deepcopy(memlets.pop(0))

            # Replicate the source stored node
            replicated_source_node = copy.deepcopy(source_node)
            backward_state.add_node(replicated_source_node)

            # Change the memlet data to read from the stored data and not the original data
            memlet_data.data = replicated_source_node.data

            # Add the final connection to the source node
            backward_state.add_edge(replicated_source_node, None, child_node, child_node_in_connector, memlet_data)

            # If this connection was made to a NestedSDFG and the forward node was a view,
            # We need to change the strides in the data descriptor this points to
            # Since the stored data is not a view
            # For example, if the stride of A is 5 (because it points to a column in  a 2d array),
            # The stored data will only contain the row and the stride for it should be one
            # This is only a problem if the view points to a NestedSDFG input,
            # that expects a descriptor with the original view stride
            if isinstance(child_node, nodes.NestedSDFG) and isinstance(forward_node.desc(bwd_generator.sdfg), dt.View):
                # Get the strides of the stored data
                stored_data_desc = bwd_generator.sdfg.arrays[source_node.data]
                stored_strides = stored_data_desc.strides

                # Get the NestedSDFG input descriptor
                input_desc = child_node.sdfg.arrays[child_node_in_connector]

                # Set the strides to be the last elements of the stored strides
                # We take the last elements since we might add loop indices to the shape
                # Sanity check the strides for this desc should be less than or equal to the stored strides
                assert len(input_desc.strides) <= len(stored_strides)
                input_desc.strides = stored_strides[-len(input_desc.shape):]

    # There should be the same number of memlets through the new path
    assert len(memlets) == 0


def _get_assign_tasklet(forward_state: SDFGState,
                        node: nodes.AccessNode,
                        stored_node: nodes.AccessNode,
                        last_edge: dgraph.MultiConnectorEdge,
                        loop_iterators: str,
                        cuda: bool = False):
    """
        """
    # Create the assign tasklet
    assign_tasklet_node_in_connector = "in_stored_" + node.data
    assign_tasklet_node_out_connector = "out_stored_" + node.data

    # Create the memlet for the assignment
    # This will be the same as the memlet going to the tasklet
    assign_memlet_data = copy.deepcopy(last_edge.data)
    param_dict = {}
    memlet_access_iterators = []

    # We check the incoming memlet volume
    if assign_memlet_data.volume != 1:
        # We need to add a map to iterate through the missing dimensions
        # For this we will create an assign block containing a map

        # First, Get the missing dimensions
        # Iterate through the subset
        for element in last_edge.data.subset:
            if str(element[0]) in last_edge.data.free_symbols:
                # This is a symbol we will keep in the store memlet
                memlet_access_iterators.append(str(element[0]))
            else:
                # This is a range tuple we need to add an iterator for
                # Create a random new free symbol
                free_symbol = forward_state.sdfg.find_new_symbol("si")

                # Add the new symbol here so that find_new_symbol doesn't return it again
                forward_state.sdfg.add_symbol(free_symbol, dtypes.int64)
                memlet_access_iterators.append(free_symbol)
                param_dict.update({free_symbol: element})

        # Build the memlets for input and output
        in_state_access = ','.join(memlet_access_iterators)
        input_memlet = Memlet(expr=f"{last_edge.data.data}[{in_state_access}]")
        if loop_iterators:
            output_memlet = Memlet(expr=f"{stored_node.data}[{loop_iterators},{in_state_access}]")
        else:
            output_memlet = Memlet(expr=f"{stored_node.data}[{in_state_access}]")

        assign_tasklet_node, map_entry, map_exit = forward_state.add_mapped_tasklet(
            name=f"__store_{node.data}_assign_",
            map_ranges=param_dict,
            inputs={assign_tasklet_node_in_connector: input_memlet},
            code=f"{assign_tasklet_node_out_connector} = {assign_tasklet_node_in_connector}",
            outputs={assign_tasklet_node_out_connector: output_memlet},
            schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
            external_edges=False)

        # Add the necessary connectors for external connections
        map_entry.add_in_connector("IN_store_block")
        map_exit.add_out_connector("OUT_store_block")

        # Update the internal edges to route through the new connectors
        # Find and update the edge from map_entry to tasklet
        for e in list(forward_state.out_edges(map_entry)):
            if e.dst == assign_tasklet_node:
                # Update the source connector to route through our external connector
                forward_state.remove_edge(e)
                forward_state.add_edge(map_entry, "OUT_store_block", assign_tasklet_node,
                                       assign_tasklet_node_in_connector, e.data)
                map_entry.add_out_connector("OUT_store_block")
                break

        # Find and update the edge from tasklet to map_exit
        for e in list(forward_state.in_edges(map_exit)):
            if e.src == assign_tasklet_node:
                # Update the destination connector to route through our external connector
                forward_state.remove_edge(e)
                forward_state.add_edge(assign_tasklet_node, assign_tasklet_node_out_connector, map_exit,
                                       "IN_store_block", e.data)
                map_exit.add_in_connector("IN_store_block")
                break

        # Make sure this block is connected correctly
        assign_block = map_entry
        assign_block_in_connector = "IN_store_block"
        return_node = map_exit
        return_connector = "OUT_store_block"
    else:
        # Volume is 1, create a simple tasklet without a map
        assign_tasklet_node = nodes.Tasklet(
            label=f"__store_{node.data}_assign_",
            inputs={assign_tasklet_node_in_connector},
            outputs={assign_tasklet_node_out_connector},
            code=f"{assign_tasklet_node_out_connector} = {assign_tasklet_node_in_connector}",
        )

        # Add it to the state
        forward_state.add_node(assign_tasklet_node)

        assign_block = assign_tasklet_node
        assign_block_in_connector = assign_tasklet_node_in_connector
        return_node = assign_tasklet_node
        return_connector = assign_tasklet_node_out_connector

    # Get the last map
    last_map = last_edge.src
    last_map_connector = last_edge.src_conn

    # Add the new edge from the last map entrance to the new assign block
    forward_state.add_edge(last_map, last_map_connector, assign_block, assign_block_in_connector, assign_memlet_data)
    return return_node, return_connector


def _find_map_exist_for_map_entry(map_entry: nodes.MapEntry, state: SDFGState) -> nodes.MapExit:
    """
    Find the map exist that corresponds to the input map entry
    """
    src_candidates = [node for node in state.nodes() if isinstance(node, nodes.MapExit) and node.map == map_entry.map]
    if len(src_candidates) != 1:
        # this shouldn't happen; if we are within a scope, the exit nodes
        # for the scope should already exist in the backward pass
        raise AutoDiffException("Invalid graph")

    return src_candidates[0]


def _get_symbol_upper_bound_from_loop(bwd_generator: 'DataForwardingbwd_generator', s: sp.Symbol,
                                      loops: List[LoopRegion]) -> int:
    """
    Given a symbol and a list of loops, get the upper bound of the symbol from the loops.
    Raises an error if the symbol is not a loop index or the upper bound cannot be extracted correctly.
    """
    # Get the symbol to match
    if isinstance(s, (sp.Symbol, sp.Expr)):
        # We don't want to match global SDFG symbols
        loop_indices = {symb for symb in s.free_symbols if str(symb) not in bwd_generator.sdfg.free_symbols}
        if len(loop_indices) != 1:
            raise AutoDiffException(f"Symbol dimension {s} couldn't be parsed correctly during storing")
        loop_index = str(list(loop_indices)[0])
    elif isinstance(s, str):
        # Convert the string to a symbolic expression and extract free symbols
        try:
            expr = sp.sympify(s)
        except (sp.SympifyError, TypeError, ValueError) as e:
            raise AutoDiffException(f"Symbol dimension {s} couldn't be parsed as a symbolic expression: {e}")

        # We don't want to match global SDFG symbols
        loop_indices = {symb for symb in expr.free_symbols if str(symb) not in bwd_generator.sdfg.free_symbols}
        if len(loop_indices) != 1:
            raise AutoDiffException(f"Symbol dimension {s} couldn't be parsed correctly during storing")
        loop_index = str(list(loop_indices)[0])
    else:
        raise AutoDiffException(f"Symbol dimension {s} is not a string and not a sympy symbol")

    # If the loop bound can be directly extracted from the interstate edges
    if loop_index in bwd_generator.interstate_symbols:
        loop_size = bwd_generator.interstate_symbols[loop_index]
    else:
        # Get the loop range for this symbol
        loop_size = None
        for l in loops:
            # Convert the sympy symbol to string to check if it macthes the loop variable
            if loop_index in l.loop_variable:
                # Get the max loop range
                start, end = ad_utils.extract_loop_region_info(l)

                # Check if the loop variable has a negative coefficient
                # by extracting the coefficient from the affine expression
                s_expr = sp.sympify(s) if isinstance(s, str) else s
                # Find the actual symbol in the expression that matches loop_index by name
                loop_symbol = None
                for sym in s_expr.free_symbols:
                    if str(sym) == loop_index:
                        loop_symbol = sym
                        break

                # Extract the coefficient of the loop variable
                if loop_symbol is not None:
                    coeff = s_expr.coeff(loop_symbol)
                    # If coefficient is negative we need to use smallest instead of largest
                    matched = coeff is not None and (coeff < 0) == True
                else:
                    # Loop variable not found in expression
                    matched = False
                smallest, largest = ad_utils.get_loop_end(start, end, l)
                if not matched:
                    loop_size = largest
                else:
                    loop_size = smallest

    if loop_size is None:
        raise AutoDiffException(
            f"Can't figure out how to save the data inside: {l.label} because of its symbol shape {s}")

    # We will call this function recusrively until loop size is numeric or it is a global SDFG symbol
    if ad_utils.shape_has_symbols_to_replace(bwd_generator.sdfg, loop_size):
        loop_size, _ = _get_symbol_upper_bound_from_loop(bwd_generator, loop_size, loops)
    return loop_size, loop_index


def _get_all_enclosing_loops(forward_state: SDFGState) -> List[LoopRegion]:
    """
        Check if this state will be executed several times within a loop.
        We check if any of the parents of this state is a loop region.
        """
    all_loops = []
    parent = forward_state.parent_graph
    while parent is not None:
        if isinstance(parent, LoopRegion):
            all_loops.append(parent)
        parent = parent.parent_graph
    return all_loops
