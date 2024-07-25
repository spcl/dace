# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import sympy
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import dtypes, subsets
import copy

@make_properties
class BlockTiling(transformation.SingleStateTransformation):
    """
    Block tiling transformation aims to tile sequential work loops of a GPU kernel to tiled loops to align the memory
    movement sized resulting from the loop hierarchy to the cache hierarchy or to pass it to shared memory / other memory locations.

    For example for matrix multiplication as sum of inner products, after tiling the K work-loop with a size of 16 it will become:

    # Before Block Tiling
    for i, j in dace.map[0:M, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        tmp = 0.0
        for k in dace.map[0:K] @ dace.dtypes.ScheduleType.Sequential:
            tmp = tmp + A[i, k] * B[k, j]
        C[i, j] = tmp

    # After Block Tiling
    for i, j in dace.map[0:M, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        for k in dace.map[0:K:16] @ dace.dtypes.ScheduleType.Sequential:
            # Move the blocks i:i+1, k:k+16 and k:k+16,j:j+1 from main to shared memory
            tmp = 0.0
            for kk in dace.map[k:k+16] @ dace.dtypes.ScheduleType.Sequential:
                tmp = tmp + A[i, kk] * B[kk, j]
            C[i, j] = tmp

    It enables better memory movement transformations.
    For the naive matrix multiplication the i:i+X, j:j+Y tile sizes can be changed by applying Thread Coarsening transformation.

    The transformation is designed in the set of "AMM-guided transformations". It assumes that BlockCoarsening, ThreadCoarsening
    and AddThreadBlockMap (or consequented ChangeThreadBlockMap) transformations have been applied, even if the tiling sizes are trivial (1,1,...).
    """
    block_coarsened_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)
    sequential_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    block_tile_sizes = Property(dtype=tuple, default=(16,), desc="")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.block_coarsened_map_entry, cls.thread_block_map_entry, cls.sequential_map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        if state.entry_node(self.sequential_map_entry) != self.thread_block_map_entry:
            return False
        if state.entry_node(self.thread_block_map_entry) != self.block_coarsened_map_entry:
            return False
        if self.thread_block_map_entry.schedule != dtypes.ScheduleType.GPU_ThreadBlock or \
            self.sequential_map_entry.schedule != dtypes.ScheduleType.Sequential or \
            self.block_coarsened_map_entry.schedule != dtypes.ScheduleType.Sequential:
            return False

        return True

    def update_names():
        pass

    def find_next_map_entry(self, state: SDFGState, node : nodes.MapEntry):
        nodes_to_check = [v for u, u_conn, v, v_conn, memlet in state.out_edges(node)]
        while (len(nodes_to_check) != 0):
            node_to_check = nodes_to_check.pop()
            if isinstance(node_to_check, nodes.MapEntry):
                return node_to_check
            if not isinstance(nodes_to_check, nodes.MapExit):
                nodes_to_check += [v for u, u_conn, v, v_conn, memlet in state.out_edges(node_to_check)]
        return None

    def find_next_map_exit(self, state: SDFGState, node : nodes.MapExit):
        nodes_to_check = [v for u, u_conn, v, v_conn, memlet in state.out_edges(node)]
        while (len(nodes_to_check) != 0):
            node_to_check = nodes_to_check.pop()
            if isinstance(node_to_check, nodes.MapExit):
                return node_to_check
            if not isinstance(nodes_to_check, nodes.MapEntry):
                nodes_to_check += [v for u, u_conn, v, v_conn, memlet in state.out_edges(node_to_check)]
        return None

    def return_access_node(self, state: SDFGState, array_name: str):
        for node in sdutil.dfs_topological_sort(state):
            if isinstance(node, nodes.AccessNode):
                if node.data == array_name:
                    return node
        # Create new
        access_node = nodes.AccessNode(data=array_name)
        state.add_node(access_node)
        return access_node

    def replace_connectors(self, state: SDFGState, map_node, match_str : str, replace_str: str):
        new_in_conns = []
        new_out_conns = []

        for in_conn in map_node.in_connectors:
            if in_conn[3:] == match_str:
                new_in_conns.append("IN_" + replace_str)
                for in_edge in state.in_edges(map_node):
                    u, u_conn, v, v_conn, memlet = in_edge
                    if v_conn == in_conn:
                        state.remove_edge(in_edge)
                        state.add_edge(u, u_conn, v, "IN_" + replace_str, memlet)
            else:
                new_in_conns.append(in_conn)
        for in_conn in copy.deepcopy(map_node.in_connectors):
            map_node.remove_in_connector(in_conn)
        for new_in_conn in new_in_conns:
            map_node.add_in_connector(new_in_conn)

        for out_conn in map_node.out_connectors:
            if out_conn[4:] == match_str:
                new_out_conns.append("OUT_" + replace_str)
                for out_edge in state.out_edges(map_node):
                    u, u_conn, v, v_conn, memlet = out_edge
                    if u_conn == out_conn:
                        state.remove_edge(out_edge)
                        state.add_edge(u, "OUT_" + replace_str, v, v_conn, memlet)
            else:
                new_out_conns.append(out_conn)
        for out_conn in copy.deepcopy(map_node.out_connectors):
            map_node.remove_out_connector(out_conn)
        for new_out_conn in new_out_conns:
            map_node.add_out_connector(new_out_conn)

    def replace_subsets(self, state: SDFGState, 
                        map_node, 
                        match_subset : subsets.Range, 
                        replace_subset: subsets.Range):
        edges_to_check = state.out_edges(map_node)
        while len(edges_to_check) > 0:
            edge = edges_to_check.pop()
            u, u_conn, v, v_conn, memlet = edge
            new_range_list = []
            if memlet.subset:
                for range in memlet.subset:
                    if range == match_subset:
                        new_range_list.append(replace_subset)
                    else:
                        new_range_list.append(range)
            else:
                new_range_list = None
            new_memlet = Memlet(data=memlet.data, subset=subsets.Range(new_range_list) if new_range_list != None else new_range_list)
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            edges_to_check += [e for e in state.out_edges(v) if e != state.exit_node(map_node)] 

    def replace_subsets_from_data(self, state: SDFGState,
                                  begin_node,
                                  end_node,
                                  match_data : str,
                                  replace_data : str,
                                  replace_range):
        edges_to_check = set(state.out_edges(begin_node))
        while len(edges_to_check) > 0:
            edge = edges_to_check.pop()
            u, u_conn, v, v_conn, memlet = edge
            new_range_list = []
            if memlet.data == match_data:
                if replace_range != None:
                    new_memlet = Memlet(data=replace_data, subset=replace_range)
                else:
                    new_memlet = Memlet(data=replace_data, subset=memlet.subset)
            else:
                new_memlet = memlet
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            edge = u, u_conn, v, v_conn, new_memlet
            edges_to_check = edges_to_check.union(set([(_u, _uc, _v, _vc, _memlet) for (_u, _uc, _v, _vc, _memlet) in state.out_edges(v) if _v != end_node]))

    def apply(self, state: SDFGState, sdfg: SDFG):
        work_map_entry : nodes.MapEntry = self.sequential_map_entry
        work_map : nodes.Map = work_map_entry.map
        thread_block_map_entry : nodes.MapEntry = self.thread_block_map_entry
        thread_block_map_exit : nodes.MapExit = state.exit_node(thread_block_map_entry)
        thread_block_map : nodes.Map = thread_block_map_entry.map
        block_coarsened_map_entry : nodes.MapEntry = self.block_coarsened_map_entry
        block_coarsened_map_exit  : nodes.MapExit = state.exit_node(self.block_coarsened_map_entry)
        block_coarsened_map : nodes.Map = block_coarsened_map_entry.map
        device_map_entry : nodes.MapEntry = state.entry_node(self.block_coarsened_map_entry)
        device_map_exit : nodes.MapExit = state.exit_node(device_map_entry)
        device_map : nodes.Map = device_map_entry.map
        assert(device_map_entry.map.schedule == dtypes.ScheduleType.GPU_Device)
        tiling_params : tuple = self.block_tile_sizes
        # Create the the new map after the device map
        # Tiling params start from the x dimension, which is last iterator variable
        work_map_len = len(work_map.params)
        matching_tiling_params = [1] * work_map_len
        matching_tiling_params[-min(len(tiling_params), work_map_len):] = tiling_params[:min(len(tiling_params), work_map_len)]

        # Calculate the new range for the new map
        outer_work_map_range_list = []
        for tiling_param, (beg, end, step) in zip(matching_tiling_params, work_map.range):
            outer_work_map_range_list.append((beg, end, step*tiling_param))
        outer_work_map_range = subsets.Range(outer_work_map_range_list)
        outer_work_map = nodes.Map(label=f"bt_{work_map.label}", 
                                   params=[f"t{param}" for param in work_map.params], 
                                   ndrange=outer_work_map_range, schedule=dtypes.ScheduleType.Sequential)
        outer_work_map_entry = nodes.MapEntry(outer_work_map)
        outer_work_map_exit = nodes.MapExit(outer_work_map)
        work_map.unroll = True

        # Map structure: Device Map -> Block Coarsened Map -> Outer Work Map(s) -> Thread Block Map -> Thread Coarsened Map -> Inner Work Map(s)
        for out_conn in block_coarsened_map_entry.out_connectors:
            outer_work_map_entry.add_out_connector(out_conn)
        for in_conn in block_coarsened_map_entry.in_connectors:
            outer_work_map_entry.add_in_connector(in_conn)
        for out_conn in block_coarsened_map_exit.out_connectors:
            outer_work_map_exit.add_out_connector(out_conn)
        for in_conn in block_coarsened_map_exit.in_connectors:
            outer_work_map_exit.add_in_connector(in_conn)
        state.add_node(outer_work_map_entry)
        state.add_node(outer_work_map_exit)

        # Nove replicate the edges as they are
        for out_edge in state.out_edges(block_coarsened_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            state.add_edge(outer_work_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
            state.add_edge(u, u_conn, outer_work_map_entry, v_conn, copy.deepcopy(memlet))
            state.remove_edge(out_edge)
        for in_edge in state.in_edges(block_coarsened_map_exit):
            u, u_conn, v, v_conn, memlet = in_edge
            state.add_edge(outer_work_map_exit, u_conn, v, v_conn, copy.deepcopy(memlet))
            state.add_edge(u, u_conn, outer_work_map_exit, v_conn, copy.deepcopy(memlet))
            state.remove_edge(in_edge)

        # Update the ranges of the inner map
        work_map_range_list = []
        old_work_map_ranges = []
        underapproximated_work_map_range_list = []
        for tiling_param, (beg, end, step), outer_work_map_param in zip(matching_tiling_params, work_map.range, outer_work_map.params):
            old_work_map_ranges.append((beg, end, step))
            work_map_range_list.append((sympy.Symbol(outer_work_map_param), sympy.Min(end, sympy.Symbol(outer_work_map_param)+step*tiling_param-1), step))
            underapproximated_work_map_range_list.append((sympy.Symbol(outer_work_map_param), sympy.Symbol(outer_work_map_param)+step*tiling_param-1, step))
        work_map_range = subsets.Range(work_map_range_list)
        work_map.range = work_map_range

        # Update any memlet outgoing from the inner map that has the form 0:K to tk:tk+tiling_param
        for old_range, new_range in zip(old_work_map_ranges, underapproximated_work_map_range_list):
            print(old_range, "->", new_range)
            self.replace_subsets(state, outer_work_map_entry, old_range, new_range)

        # Move temporary arrays after the outer tiled work map before the thread block schedule (the edges too)
        assert(thread_block_map_entry.schedule == dtypes.ScheduleType.GPU_ThreadBlock)
        edges_to_add = []
        access_nodes = []
        for out_edge in state.out_edges(thread_block_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            if isinstance(v, nodes.AccessNode):
                access_nodes.append(v)
                in_u, in_u_conn, in_v, in_v_conn, in_memlet = out_edge
                assert(len(state.out_edges(v)) == 1)
                out_u, out_u_conn, out_v, out_v_conn, out_memlet = state.out_edges(v)[0]
                assert(v == out_u)
                assert(v == in_v)
                # Move above the thread block tiled map
                old_in_u_conn = in_u_conn
                in_u_conn = "OUT_" + out_v_conn[3:]
                if old_in_u_conn != None:
                    block_coarsened_map_entry.add_out_connector(in_u_conn)
                outer_work_map_entry.add_out_connector(in_u_conn)
                outer_work_map_entry.add_in_connector(out_v_conn)
                thread_block_map_entry.add_out_connector(in_u_conn)
                thread_block_map_entry.add_in_connector(out_v_conn)
                next_map_after_thread_block = self.find_next_map_entry(state, thread_block_map_entry)
                # From block coarsened to outer work map
                edges_to_add.append((block_coarsened_map_entry, old_in_u_conn, v, None, copy.deepcopy(in_memlet)))
                edges_to_add.append((v, None, outer_work_map_entry, out_v_conn, copy.deepcopy(out_memlet)))
                # From Work map to thread block
                edges_to_add.append((outer_work_map_entry, in_u_conn, thread_block_map_entry, out_v_conn, copy.deepcopy(out_memlet)))
                edges_to_add.append((thread_block_map_entry, in_u_conn, next_map_after_thread_block, out_v_conn, copy.deepcopy(out_memlet)))
        for out_edge in state.out_edges(thread_block_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            if u_conn == None and v_conn == None:
                state.remove_edge(out_edge)
        for v in access_nodes:
            for e in state.in_edges(v) + state.out_edges(v):
                state.remove_edge(e)
        for e in edges_to_add:
            state.add_edge(*e)


        # Prevent calling syncthreads in every iteration of the block tiled loop
        thread_block_map_entry.map.gpu_syncthreads = False

        # If there was an assignment after the previous K loop, then it will be assigned every iteration of the tiled tK loop.
        # This means many more assignments to global memory we need to fix that by remove any assignment node
        # 
        # We have moved some access nodes above the thread block schedule, to avoid writing wrong results, 
        # They have to be accumulated in the temporary variable after the inner work map (not the case) and assigned to the global array
        # after the outer work map.
        # The access nodes moved are saved in ther variable access_nodes = []
        # To ensure this, one has to the following:
        # 1. Go through all vertices edges between the outer_work_map_entry and outer_work_map_exit and look for any of the moved access nodes
        #    Followed by an assignment node
        # 2. Replace the assignment node to the global array with the temporary array self (or remove the assignment totally)
        # 3. Save where it was assigned to, assign the temporary array to the global array after the outer work map
        # 4. Updates memlets between inner work and outer work map to access tmp and not C
        # 1: Go through nodes
        work_map_exit = state.exit_node(work_map_entry)
        nodes_to_check = set(v for (u, u_conn, v, v_conn, memlet) in state.out_edges(work_map_exit))
        assignments_removed = []
        edges_to_remove = set()
        nodes_to_remove = set()
        edges_to_add = set()
        while len(nodes_to_check) != 0:
            node = nodes_to_check.pop()
            # 2.1 Check if an access node is followed by an assignment tasklet
            if isinstance(node, nodes.AccessNode):
                assert(len(state.out_edges(node)) == 1)
                assert(len(state.in_edges(node)) == 1)
                out_edge = state.out_edges(node)[0]
                in_edge = state.in_edges(node)[0]
                (u, u_conn, v, _, _) = out_edge
                (iu, iu_conn, _, _, memlet) = in_edge
                if isinstance(v, nodes.Tasklet) and "assign" in v.label:
                    assert(len(state.out_edges(v)) == 1)
                    tasklet_out_edge = state.out_edges(v)[0]
                    (tu, tu_conn, tv, tv_conn, tmemlet) = tasklet_out_edge
                    # 2.2 Save what we have removed when the do the final assignment after the thread block map
                    assignments_removed.append((in_edge, node, out_edge, v, tasklet_out_edge))
                    # 2.3 Remove the assignment, pass the edge directly to the next map
                    edges_to_remove.add(in_edge)
                    edges_to_remove.add(out_edge)
                    edges_to_remove.add(tasklet_out_edge)
                    nodes_to_remove.add(node)
                    nodes_to_remove.add(v)
                    edges_to_add.add((iu, iu_conn, tv, tv_conn, tmemlet))
            if node != thread_block_map_exit:
                nodes_to_check = nodes_to_check.union(set([v for (u, u_conn, v, v_conn, memlet) in state.out_edges(node)]))

        for edge in edges_to_remove:
            state.remove_edge(edge)
        for node in nodes_to_remove:
            state.remove_node(node)
        for edge in edges_to_add:
            state.add_edge(*edge)

        # 3.1 Anything removed needs to be added the thread block map
        for (access_node_in_edge, acces_node, access_node_out_edge, tasklet, tasklet_out_edge) in assignments_removed:
            # Form is out_conn -> None, Access Node, None -> in_conn, Tasklet, out_conn -> in_conn
            # Copy the out_conn of first edge and in_conn of last edge and then reuse the tasklet node to assign the variables in a map
            outer_work_map_exit
            _,_,_,_,local_src_memlet = access_node_out_edge
            local_src = local_src_memlet.data
            _,_,_,_,glb_dst_memlet = tasklet_out_edge
            glb_dst = glb_dst_memlet.data
            subset_list = [(0, end-1, 1) for end in sdfg.arrays[local_src].shape]
            assign_map_entry, assign_map_exit = state.add_map(name=f"assign_map_{local_src}_{glb_dst}",
                                                                ndrange=[(f"assign_map_{i}", v) for i, v in enumerate(subset_list)],
                                                                schedule=dtypes.ScheduleType.Sequential,
                                                                unroll=True)
            _, _, _, in_conn, _ = access_node_in_edge
            _, out_conn, _, _, _ = tasklet_out_edge

            assign_map = assign_map_entry.map
            
            #for map_node in [outer_work_map_exit, assign_map_entry, assign_map_exit]:

            # Right now eveything up to the outer work map is C 
            #local_access=subsets.Range([(0, end, 1) for end in sdfg.arrays[local_src].shape])
            self.replace_subsets_from_data(state, work_map_exit, outer_work_map_exit, glb_dst, local_src, subsets.Range(subset_list))

            pass

        """
        
        nodes_to_check = [v for (u, u_conn, v, v_conn, memlet) in state.out_edges(work_map_exit)]
        while len(nodes_to_check) != 0:
            node = nodes_to_check.pop()
            if isinstance(node, nodes.AccessNode):
                if len(state.out_edges(node)) == 1:
                    # Assignment node, if the assignment to another array
                    edge = state.out_edges(node)[0]
                    u, u_conn, v, v_conn, memlet = edge
                    if isinstance(v, nodes.Tasklet) and "assign" in v.label:
                        state.remove_edge(edge)
                        assert(len(state.out_edges(v)) == 1)
                        _, _, ov, ov_conn, omemlet = state.out_edges(v)[0]
                        assign_node = v
                        # We have made for example Access Node (tmp) -> Assignment -> Map to
                        # Access Node -> Map, where all the memlets after the assignment need to be updated.
                        in_conn = ov_conn
                        out_conn = "OUT_" + ov_conn[3:]
                        data_to_replace = memlet.data
                        global_array = None
                        assert(isinstance(ov, nodes.MapExit))
                        next_map_exit = ov
                        state.remove_node(assign_node)
                        state.add_edge(u, u_conn, ov, ov_conn, Memlet(data=data_to_replace, subset=memlet.subset))
                        while next_map_exit != None and next_map_exit != outer_work_map_exit:
                            print("C", next_map_exit)
                            # Assume maps are directly connected for now, and update the maps until the outer work map to
                            # Use the local variable
                            for out_edge in state.out_edges(next_map_exit):
                                _u, _u_conn, _v, _v_conn, _memlet = out_edge
                                print(_u_conn,out_conn,_v_conn, in_conn)
                                if _u_conn == out_conn and _v_conn == in_conn:
                                    print(data_to_replace)
                                    print(sdfg.arrays[data_to_replace])
                                    subset_list = [(0, end-1, 1) for end in sdfg.arrays[data_to_replace].shape]
                                    assert(global_array == None or global_array == _memlet.data)
                                    global_array = _memlet.data
                                    m = Memlet(data=data_to_replace, subset=subsets.Range(subset_list))
                                    state.remove_edge(out_edge)
                                    state.add_edge(_u, _u_conn, _v, _v_conn, m)
                            next_map_exit = self.find_next_map_exit(state, next_map_exit)
                        # Here the edges use the global array but the Map -> gEdge needs to become Map -> localEdge -> gEdge
                        # While also using local offset and global offset
                        # Update any out connector of form OUT_C to OUT_tmp, where c is not accessed anymore
                        thread_coarsened_map_entry = self.find_next_map_entry(state, outer_work_map_entry)
                        thread_coarsened_map_exit = state.exit_node(thread_coarsened_map_entry)
                        self.replace_connectors(state, outer_work_map_exit, global_array, data_to_replace)
                        self.replace_connectors(state, thread_coarsened_map_exit, global_array, data_to_replace)
                        for out_edge in state.out_edges(outer_work_map_exit):
                            _u, _u_conn, _v, _v_conn, _memlet = out_edge
                            print("EEEEEEEEEEEEEEEEE", _u_conn, out_conn, _v_conn, in_conn)
                            print("OOOOOOOOOOO", _u_conn[4:], data_to_replace,_v_conn[3:],global_array)
                            if _u_conn[4:] == data_to_replace  and _v_conn[3:] == global_array:
                                subset_list = [(0, end-1, 1) for end in sdfg.arrays[data_to_replace].shape]
                                m = Memlet(data=data_to_replace, subset=subsets.Range(subset_list))
                                m2 = Memlet(data=data_to_replace, subset=subsets.Range(subset_list), other_subset=_memlet.subset)
                                #access_node_tmp = self.return_access_node(state, data_to_replace)
                                #access_node_glb = self.return_access_node(state, global_array)
                                access_node_tmp = nodes.AccessNode(data=data_to_replace)
                                access_node_glb = nodes.AccessNode(data=global_array)
                                # Need to create a map with a tasklet
                                assign_map_entry, assign_map_exit = state.add_map(name=f"assign_map_{data_to_replace}_{global_array}",
                                                                                  ndrange=[(f"assign_map_{i}", v) for i, v in enumerate(subset_list)],
                                                                                  schedule=dtypes.ScheduleType.Sequential,
                                                                                  unroll=True)
                                assign_map = assign_map_entry.map

                                for in_conn, out_conn in zip(outer_work_map_exit.in_connectors, outer_work_map_exit.out_connectors):
                                    assign_map_entry.add_in_connector(in_conn)
                                    assign_map_entry.add_out_connector(out_conn)
                                for in_conn, out_conn in zip(thread_block_map_exit.in_connectors, thread_block_map_exit.out_connectors):
                                    assign_map_exit.add_in_connector(in_conn)
                                    assign_map_exit.add_out_connector(out_conn)

                                u_conn_tmp = f"OUT_{data_to_replace}"
                                v_conn_tmp = f"IN_{data_to_replace}"
                                u_conn_c = f"OUT_{global_array}"
                                v_conn_c = f"IN_{global_array}"

                                state.add_node(assign_node)
                                state.remove_edge(out_edge)
                                state.add_edge(_u, u_conn_tmp, assign_map_entry, v_conn_tmp, m)
                                assign_expr_tmp = [(sympy.Symbol(param),sympy.Symbol(param),1) for param in assign_map.params]
                                assign_expr_glb = [(sympy.Symbol(param) + sympy.Symbol(param2),sympy.Symbol(param) + sympy.Symbol(param2),1) 
                                                   for param, param2 in zip(assign_map.params, thread_block_map.params)]
                                m3 = Memlet(data=data_to_replace, subset=subsets.Range(assign_expr_tmp))
                                m4 = Memlet(data=global_array, subset=subsets.Range(assign_expr_glb))
                                assert(len(assign_node.in_connectors)==1)
                                assert(len(assign_node.out_connectors)==1)
                                assign_in_conn = next(iter(assign_node.in_connectors.items()))[0]
                                assign_out_conn = next(iter(assign_node.out_connectors.items()))[0]
                                state.add_edge(assign_map_entry, u_conn_tmp, assign_node, assign_in_conn, m3)
                                state.add_edge(assign_node, assign_out_conn , assign_map_exit, v_conn_c, m4)
                                state.add_edge(assign_map_exit, u_conn_c, thread_block_map_exit, v_conn_c, _memlet)


            if node != thread_block_map_exit:
                nodes_to_check += [v for (u, u_conn, v, v_conn, memlet) in state.out_edges(node)]
        """

    @staticmethod
    def annotates_memlets():
        return True
