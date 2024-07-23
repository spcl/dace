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
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)
    sequential_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    block_tile_sizes = Property(dtype=tuple, default=(16,), desc="")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.sequential_map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        if state.entry_node(self.sequential_map_entry) != self.thread_block_map_entry:
            return False
        if state.entry_node(self.thread_block_map_entry) != self.thread_block_map_entry:
            return False
        if self.thread_block_map_entry.schedule != dtypes.ScheduleType.GPU_ThreadBlock or \
            self.sequential_map_entry.schedule != dtypes.ScheduleType.Sequential:
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

    def replace_memlet_subsets(self, state: SDFGState, subset_to_match : tuple, 
                               subset_to_replace : tuple, begin_node : nodes.MapEntry, 
                               end_node : nodes.MapExit):
        nodes_to_check = [begin_node]

        while (len(nodes_to_check) != 0):
            node_to_check = nodes_to_check.pop()
            for edge_to_check in state.out_edges(node_to_check):
                u, u_conn, v, v_conn, memlet = edge_to_check
                new_ranges_list = []
                for (beg, end, step) in memlet.subset:
                    if (beg, end, step) == subset_to_match:
                        new_ranges_list.append(subset_to_replace)
                    else:
                        new_ranges_list.append((beg, end, step))
                memlet = Memlet(subset=subsets.Range(new_ranges_list), data=memlet.data)
                state.remove_edge(edge_to_check)
                edge_to_check = state.add_edge(u, u_conn, v, v_conn, memlet)
            if node_to_check != end_node:
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

    def replace_connectors(self, state: SDFGState, map_node: nodes.MapEntry | nodes.MapExit, match_str : str, replace_str: str):
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

    def apply(self, state: SDFGState, sdfg: SDFG):
        work_map_entry : nodes.MapEntry = self.sequential_map_entry
        work_map : nodes.Map = work_map_entry.map
        thread_block_map_entry : nodes.MapEntry = self.thread_block_map_entry
        thread_block_map_exit : nodes.MapExit = state.exit_node(thread_block_map_entry)
        thread_block_map : nodes.Map = thread_block_map_entry.map
        tiling_params : tuple = self.block_tile_sizes
        print(work_map, work_map.range)

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
        print(work_map, outer_work_map_range_list, outer_work_map_range)
        outer_work_map = nodes.Map(label=f"bt_{work_map.label}", 
                                   params=[f"t{param}" for param in work_map.params], 
                                   ndrange=outer_work_map_range, schedule=dtypes.ScheduleType.Sequential)
        outer_work_map_entry = nodes.MapEntry(outer_work_map)
        outer_work_map_exit = nodes.MapExit(outer_work_map)
        work_map.unroll = True

        # Map structure: Device Map -> Block Coarsened Map -> Thread Block Map -> Outer Work Map(s) -> Thread Coarsened Map -> Inner Work Map(s)
        inner_map_entry : nodes.MapEntry = self.find_next_map_entry(state, thread_block_map_entry)
        assert(inner_map_entry != None)
        inner_map_exit : nodes.MapExit = state.exit_node(inner_map_entry)
        assert(inner_map_exit != None)
        for out_conn in thread_block_map_entry.out_connectors:
            outer_work_map_entry.add_out_connector(out_conn)
        for in_conn in thread_block_map_entry.in_connectors:
            outer_work_map_entry.add_in_connector(in_conn)
        for out_conn in thread_block_map_exit.out_connectors:
            outer_work_map_exit.add_out_connector(out_conn)
        for in_conn in thread_block_map_exit.in_connectors:
            outer_work_map_exit.add_in_connector(in_conn)
        state.add_node(outer_work_map_entry)
        state.add_node(outer_work_map_exit)

        # Nove replicate the edges as they are
        for out_edge in state.out_edges(thread_block_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            state.add_edge(outer_work_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
            state.add_edge(u, u_conn, outer_work_map_entry, v_conn, copy.deepcopy(memlet))
            state.remove_edge(out_edge)
        for in_edge in state.in_edges(thread_block_map_exit):
            u, u_conn, v, v_conn, memlet = in_edge
            state.add_edge(outer_work_map_exit, u_conn, v, v_conn, copy.deepcopy(memlet))
            state.add_edge(u, u_conn, outer_work_map_exit, v_conn, copy.deepcopy(memlet))
            state.remove_edge(in_edge)

        # Update the ranges of the inner map
        work_map_range_list = []
        old_work_map_ranges = []
        for tiling_param, (beg, end, step), outer_work_map_param in zip(matching_tiling_params, work_map.range, outer_work_map.params):
            old_work_map_ranges.append((beg, end, step))
            work_map_range_list.append((sympy.Symbol(outer_work_map_param), sympy.Min(end, sympy.Symbol(outer_work_map_param)+step*tiling_param-1), step))
        work_map_range = subsets.Range(work_map_range_list)
        work_map.range = work_map_range

        # Move temporary arrays after the outer tiled work map before the thread block schedule (the edges too)
        outer_map_entry = state.entry_node(thread_block_map_entry)
        print("AAAAAAAAAAAAAA", outer_map_entry)
        assert(thread_block_map_entry.schedule == dtypes.ScheduleType.GPU_ThreadBlock)
        edges_to_add = []
        access_nodes = []
        for out_edge in state.out_edges(outer_work_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            if isinstance(v, nodes.AccessNode):
                access_nodes.append(v)
                in_u, in_u_conn, in_v, in_v_conn, in_memlet = out_edge
                assert(len(state.out_edges(v)) == 1)
                out_u, out_u_conn, out_v, out_v_conn, out_memlet = state.out_edges(v)[0]
                assert(v == out_u)
                assert(v == in_v)
                # Move above the thread block tiled map
                old_in_u_conn = None
                if in_u_conn == None: 
                    # It was created after tblock map
                    in_u_conn = "OUT_" + out_v_conn[3:]
                else:
                    old_in_u_conn = in_u_conn
                outer_work_map_entry.add_out_connector(in_u_conn)
                thread_block_map_entry.add_out_connector(in_u_conn)
                outer_work_map_entry.add_in_connector(out_v_conn)
                thread_block_map_entry.add_in_connector(out_v_conn)
                edges_to_add.append((outer_map_entry, old_in_u_conn, v, None, copy.deepcopy(in_memlet)))
                edges_to_add.append((v, None, thread_block_map_entry, out_v_conn, copy.deepcopy(out_memlet)))
                edges_to_add.append((outer_work_map_entry, in_u_conn, out_v, out_v_conn, copy.deepcopy(out_memlet)))
                edges_to_add.append((thread_block_map_entry, in_u_conn, outer_work_map_entry, out_v_conn, copy.deepcopy(out_memlet)))
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

        """
        # The map before tiling had memlets of form of beg:end:step,
        # After the tiling the map is basically beg:beg+tilesize*step:step
        # And therefore any memlet range that has the form of beg:end:step needs to be changed to beg:beg+tilesize*step:step
        # (For shared memory transformation to work correctly)
        # To be safe update every memlet until we reach the map exit of block tiled map
        for old_range, new_range, outer_work_map_param in zip(old_work_map_ranges, outer_work_map_entry.range, outer_work_map.params):
            old_beg, old_end, old_step = old_range
            new_beg, new_end, new_step = new_range
            outer_work_map_param = sympy.Symbol(outer_work_map_param)
            new_beg = outer_work_map_param
            new_end = new_beg + new_step - 1
            self.replace_memlet_subsets(state, (old_beg, old_end, 1), (new_beg, new_end, 1), outer_work_map_entry, outer_work_map_exit)
        """

        # If there was an assignment after the previous K loop, then it will be assigned every iteration of the tiled tK loop.
        # This means many more assignments to global memory we need to fix that by remove any assignment node
        work_map_exit = state.exit_node(work_map_entry)
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

    @staticmethod
    def annotates_memlets():
        return True
