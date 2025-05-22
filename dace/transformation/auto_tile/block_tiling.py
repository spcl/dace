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
from dace.symbolic import SymExpr, symbol
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
    work_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    block_tile_sizes = Property(dtype=tuple, default=(16,), desc="")

    global_application_number = 0

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.work_map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        #if state.entry_node(self.work_map_entry) != self.thread_block_map_entry:
        #    return False
        #if self.thread_block_map_entry.schedule != dtypes.ScheduleType.GPU_ThreadBlock or \
        #    self.work_map_entry.schedule != dtypes.ScheduleType.Sequential:
        #    return False

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
            new_memlet = Memlet(data=memlet.data, subset=subsets.Range(new_range_list) if new_range_list != None else new_range_list,
                                wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
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
                    new_memlet = Memlet(data=replace_data, subset=replace_range, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
                else:
                    new_memlet = Memlet(data=replace_data, subset=memlet.subset, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
            else:
                new_memlet = memlet
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            edge = u, u_conn, v, v_conn, new_memlet
            edges_to_check = edges_to_check.union(set([(_u, _uc, _v, _vc, _memlet) for (_u, _uc, _v, _vc, _memlet) in state.out_edges(v) if _v != end_node]))

    def apply(self, state: SDFGState, sdfg: SDFG):
        work_map_entry : nodes.MapEntry = self.work_map_entry
        work_map : nodes.Map = work_map_entry.map
        thread_coarsened_map_entry : nodes.MapEntry = state.entry_node(work_map_entry)
        thread_block_map_entry : nodes.MapEntry = self.thread_block_map_entry
        thread_block_map_exit : nodes.MapExit = state.exit_node(thread_block_map_entry)
        thread_block_map : nodes.Map = thread_block_map_entry.map
        device_map_entry : nodes.MapEntry = state.entry_node(self.thread_block_map_entry)
        device_map_exit : nodes.MapExit = state.exit_node(device_map_entry)
        device_map : nodes.Map = device_map_entry.map
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
                                   params=[f"{param}" for param in work_map.params],
                                   ndrange=outer_work_map_range, schedule=dtypes.ScheduleType.Sequential)
        outer_work_map_entry = nodes.MapEntry(outer_work_map)
        outer_work_map_exit = nodes.MapExit(outer_work_map)
        work_map.unroll = True

        # Update params with prefix
        new_params = [f"t{param}" for param in work_map.params]
        work_map.params = new_params

        # Map structure: Device Map ->
        #   Outer Work Map(s) ->
        #   Thread Block Map ->
        #   Thread Coarsened Map ->
        #   Inner Work Map(s)
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

        #sdfg.save("a.sdfg")

        # Update the ranges of the inner map
        work_map_range_list = []
        inner_work_map_range_list = []
        old_work_map_ranges = []
        for tiling_param, (beg, end, step), outer_work_map_param, (obeg, oend, ostep) in zip(matching_tiling_params, work_map.range, outer_work_map.params, outer_work_map.range):
            old_work_map_ranges.append((beg, end, step))
            sym_outer_work_map_param = symbol(outer_work_map_param)
            fail = False
            t = False
            try:
                if (oend+1-obeg)//ostep <= step*tiling_param-1:
                    work_map_range_list.append((sym_outer_work_map_param, sym_outer_work_map_param + ((oend+1-obeg)//ostep), ((oend+1-obeg)//ostep)))
                    inner_work_map_range_list.append((0, (oend+1-obeg)//ostep, step))
                    t = True
            except Exception as e:
                fail = True
                pass
            if fail or t:
                work_map_range_list.append((sym_outer_work_map_param, sym_outer_work_map_param + step*tiling_param-1, step))
                inner_work_map_range_list.append((0, step*tiling_param-1,step))
        work_map_range = subsets.Range(inner_work_map_range_list)
        work_map.range = work_map_range

        # Update any memlet outgoing from the inner map that has the form 0:K to tk:tk+tiling_param
        for old_range, new_range in zip(old_work_map_ranges, work_map_range_list):
            self.replace_subsets(state, outer_work_map_entry, old_range, new_range)

        # Outer Work Loop is K=0, K<?, K+=STEP
        # Inner Work Loop is tk=0, tk<STEP, tk+=1
        # If any access involves k, we need to make it to (k+tk)
        edges_to_check = set(state.out_edges(work_map_entry))
        while edges_to_check:
            edge = edges_to_check.pop()
            u, u_conn, v, v_conn, memlet = edge
            new_ranges = []
            for beg, end, step in memlet.subset:
                for param, new_param in zip(outer_work_map.params, new_params):
                    param_symbol = symbol(param)
                    new_param_symbol = symbol(new_param)

                    beg = beg.subs(param_symbol, SymExpr(param_symbol + new_param_symbol))
                    end = end.subs(param_symbol, SymExpr(param_symbol + new_param_symbol))
                    step = step.subs(param_symbol, new_param_symbol)

                new_ranges.append((beg, end, step))
            new_memlet = Memlet(subset=subsets.Range(new_ranges), data=memlet.data, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            if not isinstance(v, nodes.MapExit):
                edges_to_check.union(set(state.out_edges(v)))

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
        exchange_pairs = []
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
        edges_to_remove.clear()
        nodes_to_remove.clear()
        edges_to_add.clear()

        # Create the assign map after the work loop
        # 3.1 Anything removed needs to be added the thread block map

        # Replace any C[A,B] with tmp[D,E] C is the out edge from assign map, tmp is input edge from assign map
        for (access_node_in_edge, access_node, access_node_out_edge, tasklet, tasklet_out_edge) in assignments_removed:
            _, _, _, in_conn, m1 = access_node_in_edge
            _, out_conn, _, _, m2 = tasklet_out_edge
            for out_edge in state.out_edges(work_map_exit):
                u,uc,v,vc,memlet = out_edge
                _,_,_,_,mm = tasklet_out_edge
                if memlet.data == m2.data:
                    ss = None
                    for oe in state.out_edges(thread_coarsened_map_entry):
                        _,_,_,_,_m = oe
                        if _m.data == m1.data:
                            ss = _m.subset
                    tmp_memlet = Memlet(subset=ss, data=m1.data, wcr=m1.wcr, wcr_nonatomic=m1.wcr_nonatomic, allow_oob=m1.allow_oob, debuginfo=m1.debuginfo)
                    state.remove_edge(out_edge)
                    state.add_edge(u,uc,v,vc,tmp_memlet)
            for out_edge in state.out_edges(state.exit_node(thread_coarsened_map_entry)):
                u,uc,v,vc,memlet = out_edge
                _,_,_,_,mm = tasklet_out_edge
                if memlet.data == m2.data:
                    lens = [(0,end-1,1) for end in sdfg.arrays[m1.data].shape]
                    tmp_memlet = Memlet(subset=subsets.Range(lens), data=m1.data, wcr=m1.wcr, wcr_nonatomic=m1.wcr_nonatomic, allow_oob=m1.allow_oob, debuginfo=m1.debuginfo)
                    state.remove_edge(out_edge)
                    state.add_edge(u,uc,v,vc,tmp_memlet)

        # Move tmp allocation before the outer work Map
        for (access_node_in_edge, access_node, access_node_out_edge, tasklet, tasklet_out_edge) in assignments_removed:
            _, _, _, in_conn, m1 = access_node_in_edge
            for out_edge in state.out_edges(outer_work_map_entry):
                u, u_conn, v, v_conn, memlet = out_edge
                if isinstance(v, nodes.AccessNode) and v.data == m1.data:
                    assert(len(state.out_edges(v)) == 1)
                    access_out_edge = state.out_edges(v)[0]
                    ou, ou_conn, ov, ov_conn, omemlet = access_out_edge
                    # Tmp (Transient) variable found move up
                    if u_conn == None:
                        outer_work_map_entry.add_out_connector("OUT_" + ov_conn[3:])
                        uu_conn = "OUT_" + ov_conn[3:]
                    else:
                        uu_conn = u_conn
                    state.remove_edge(access_out_edge)
                    state.add_edge(outer_work_map_entry, uu_conn, ov, ov_conn, copy.deepcopy(omemlet))
                    state.remove_edge(out_edge)
                    state.remove_node(v)

                    # Now re-add
                    state.add_node(v)
                    v.setzero = True
                    if u_conn != None:
                        thread_block_map_entry.add_out_connector(u_conn)
                    state.add_edge(thread_block_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
                    outer_work_map_entry.add_in_connector(ov_conn)
                    state.add_edge(ou, ou_conn, outer_work_map_entry, ov_conn, copy.deepcopy(omemlet))

        # The edge after outer work map is M1 -> C -> M2, should be M1 -> tmp -> C -> M2
        for (access_node_in_edge, access_node, access_node_out_edge, tasklet, tasklet_out_edge) in assignments_removed:
            _, _, _, in_conn, m1 = access_node_in_edge
            _, out_conn, _, _, m2 = tasklet_out_edge
            assert(access_node.data == m1.data)
            an = nodes.AccessNode(data=access_node.data)
            subset = subsets.Range([(0,end-1,1) for end in sdfg.arrays[access_node.data].shape])
            for out_edge in state.out_edges(outer_work_map_exit):
                u,uc,v,vc,memlet = out_edge
                if memlet.data == m2.data:
                    state.add_node(an)
                    state.add_edge(u,uc,an,None,Memlet(subset=subset,data=access_node.data,wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo))
                    state.remove_edge(out_edge)
                    state.add_edge(an,None,v,vc,memlet)

        for out_edge in state.out_edges(thread_block_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            if  v == outer_work_map_entry and u_conn == None and v_conn == None:
                state.remove_edge(out_edge)

        for m in [work_map_entry.map, outer_work_map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d

        work_map.label = f"InnerWorkMapNo{BlockTiling.global_application_number}"
        outer_work_map.label = f"OuterWorkMapNo{BlockTiling.global_application_number}"

        BlockTiling.global_application_number += 1

    @staticmethod
    def annotates_memlets():
        return True
