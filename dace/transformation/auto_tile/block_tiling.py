# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import sympy
from dace.data import ListProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import dtypes, subsets
from dace.symbolic import SymExpr, symbol
import dace
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

    unroll = Property(dtype=bool, default=True, desc="Unroll the outer work map")
    unroll_mask = ListProperty(element_type=bool, default=None, desc="Which dimensions to unroll")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.thread_block_map_entry, cls.work_map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

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
                for _range in memlet.subset:
                    if _range == match_subset:
                        new_range_list.append(replace_subset)
                    else:
                        new_range_list.append(_range)
            else:
                new_range_list = None
            new_memlet = Memlet(data=memlet.data, subset=subsets.Range(new_range_list) if new_range_list != None else new_range_list,
                                wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            #print("AE8", u, u_conn, v, v_conn, new_memlet)
            edges_to_check += [e for e in state.out_edges(v) if e != state.exit_node(map_node)]

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
        work_map.unroll = self.unroll

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
            print("AE10", outer_work_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
            print("AE11", u, u_conn, outer_work_map_entry, v_conn, copy.deepcopy(memlet))
            state.remove_edge(out_edge)
        for in_edge in state.in_edges(thread_block_map_exit):
            u, u_conn, v, v_conn, memlet = in_edge
            state.add_edge(outer_work_map_exit, u_conn, v, v_conn, copy.deepcopy(memlet))
            state.add_edge(u, u_conn, outer_work_map_exit, v_conn, copy.deepcopy(memlet))
            print("AE12", outer_work_map_exit, u_conn, v, v_conn, copy.deepcopy(memlet))
            print("AE13", u, u_conn, outer_work_map_exit, v_conn, copy.deepcopy(memlet))
            state.remove_edge(in_edge)

        # Update the ranges of the inner map
        work_map_range_list = []
        inner_work_map_range_list = []
        old_work_map_ranges = []
        print(matching_tiling_params, work_map.range, outer_work_map.params, outer_work_map.range)
        for tiling_param, (beg, end, step), outer_work_map_param, (obeg, oend, ostep) in zip(matching_tiling_params, work_map.range, outer_work_map.params, outer_work_map.range):
            old_work_map_ranges.append((beg, end, step))
            sym_outer_work_map_param = symbol(outer_work_map_param)
            print((oend+1-obeg)//ostep, step*tiling_param-1)
            if ostep <= step*tiling_param-1:
                work_map_range_list.append((sym_outer_work_map_param, sym_outer_work_map_param + ((oend+1-obeg)//ostep), ((oend+1-obeg)//ostep)))
                inner_work_map_range_list.append((0, (oend+1-obeg)//ostep, step))
            else:
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
            print("AE1:",u, u_conn, v, v_conn, new_memlet)
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

        # Move the assignment that connects to thread coarsened map, to after the otuer work map.
        # Add -(tmp2[0])-> tmp2 -(tmp2[0])-> assign -(C[...])-> thread coarsened
        # it continues as C[1, 1] -> C[32, 32] -> C[256, 256] etc.
        # What we want is to have:
        # Add -(tmp2[0])-> tmp2 -(tmp2[0])-> assign -(tmp[1, 1])-> thread coarsened and then
        # thread coarsened -(tmp[32, 32])-> block_tiled_outer -> access node -(tmp[32,32])-> access node C
        # and the susbets are not changed afterwards

        """
        thread_coarsened_map_exit = state.exit_node(thread_coarsened_map_entry)
        for ie in state.in_edges(thread_coarsened_map_exit):
            if isinstance(ie.src, dace.nodes.Tasklet) and "assign" in ie.src.label:
                print(ie.src, ie)
                # We have found the assignment tasklet
                u, u_conn, v, v_conn, memlet = ie
                # try to find tmp name
                assert len(state.out_edges(state.exit_node(work_map_entry))) == 1
                tmp_name = None
                for oe in state.out_edges(state.exit_node(work_map_entry)):
                    if isinstance(oe.dst, dace.nodes.AccessNode):
                        tmp_name = oe.dst.data
                    else:
                        raise Exception("uwu todo")
                tmp_access = state.add_access(tmp_name)

                state.remove_edge(ie)
                access_str = ", ".join([f"{v}" for v in thread_coarsened_map_entry.map.params])
                state.add_edge(u, u_conn, tmp_access, None, dace.memlet.Memlet(f"{tmp_name}[{access_str}]"))
                state.add_edge(tmp_access, None, v, v_conn, dace.memlet.Memlet(f"{tmp_name}[{access_str}]"))

                # Between thread coarsened and outerwork replace with full access to tmp
                tmp_access = dace.subsets.Range.from_string(tmp_name)
                access_name = "C"
                #self.replace_subsets2(state, thread_block_map_exit, outer_work_map_exit, access_name)

                edges_to_check = set(state.out_edges(thread_coarsened_map_exit))
                edges_to_add = []
                edges_to_rm = []
                while edges_to_check:
                    e = edges_to_check.pop()
                    if e.data and e.data.data == access_name:
                        ne = (e.src, e.src_conn, e.dst, e.dst_conn, dace.memlet.Memlet(tmp_name))
                        edges_to_add.append(ne)
                        edges_to_rm.append(e)

                    if e.dst != outer_work_map_exit:
                       edges_to_check = edges_to_check.union(state.out_edges(e.dst))

                for e in edges_to_rm:
                    state.remove_edge(e)
                for e in edges_to_add:
                    state.add_edge(*e)

                edges_to_add = []
                edges_to_rm = []
                # After outer work map add access node to tmp and then copy to C
                for oe in state.out_edges(outer_work_map_exit):
                    if oe.data and oe.data.data == access_name:
                        edges_to_rm.append(oe)
                        local = state.add_access(tmp_name)
                        glb = state.add_access(access_name)
                        edges_to_add.append((oe.src, oe.src_conn, local, None, dace.memlet.Memlet(tmp_name)))
                        edges_to_add.append((local, None, glb, None, dace.memlet.Memlet(tmp_name)))
                        edges_to_add.append((glb, None, oe.dst, oe.dst_conn, oe.data))

                for e in edges_to_rm:
                    state.remove_edge(e)
                for e in edges_to_add:
                    state.add_edge(*e)
        """
        """
        work_map_exit = state.exit_node(work_map_entry)
        thread_coarsened_map_exit = state.exit_node(thread_coarsened_map_entry)
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
                (u, u_conn, v, v_conn_, _) = out_edge
                (iu, iu_conn, _, iv_conn, memlet) = in_edge
                if isinstance(v, nodes.Tasklet) and "assign" in v.label:
                    #raise Exception(node, v)
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
                    edges_to_add.add((iu, iu_conn, tv, tv_conn, memlet))
                    print("T", tu, tu_conn, iu_conn, v_conn)
                    print("IE", out_edge)
                    print("AE-1",(iu, iu_conn, tv, tv_conn, memlet))

            if node != thread_coarsened_map_exit:
                nodes_to_check = nodes_to_check.union(set([v for (u, u_conn, v, v_conn, memlet) in state.out_edges(node)]))

        for edge in edges_to_remove:
            state.remove_edge(edge)
        for node in nodes_to_remove:
            state.remove_node(node)
        for edge in edges_to_add:
            state.add_edge(*edge)
            print("AE2:",*edge)
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
                print(memlet.data, m2.data)
                if memlet.data == m2.data:
                    ss = None
                    for oe in state.out_edges(thread_coarsened_map_entry):
                        _,_,_,_,_m = oe
                        if _m.data == m1.data:
                            ss = _m.subset
                    tmp_memlet = Memlet(subset=ss, data=m1.data, wcr=m1.wcr, wcr_nonatomic=m1.wcr_nonatomic, allow_oob=m1.allow_oob, debuginfo=m1.debuginfo)
                    state.remove_edge(out_edge)
                    state.add_edge(u,uc,v,vc,tmp_memlet)
                    print("AE3:",u,uc,v,vc,tmp_memlet)
            for out_edge in state.out_edges(state.exit_node(thread_coarsened_map_entry)):
                u,uc,v,vc,memlet = out_edge
                _,_,_,_,mm = tasklet_out_edge
                if memlet.data == m2.data:
                    lens = [(0,end-1,1) for end in sdfg.arrays[m1.data].shape]
                    tmp_memlet = Memlet(subset=subsets.Range(lens), data=m1.data, wcr=m1.wcr, wcr_nonatomic=m1.wcr_nonatomic, allow_oob=m1.allow_oob, debuginfo=m1.debuginfo)
                    state.remove_edge(out_edge)
                    state.add_edge(u,uc,v,vc,tmp_memlet)
                    print("AE4:",u,uc,v,vc,tmp_memlet)

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
                    print("AE4:",outer_work_map_entry, uu_conn, ov, ov_conn, copy.deepcopy(omemlet))
                    state.remove_edge(out_edge)
                    state.remove_node(v)

                    # Now re-add
                    state.add_node(v)
                    v.setzero = True
                    if u_conn != None:
                        thread_block_map_entry.add_out_connector(u_conn)
                    state.add_edge(thread_block_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
                    print("AE5:", thread_block_map_entry, u_conn, v, v_conn, copy.deepcopy(memlet))
                    outer_work_map_entry.add_in_connector(ov_conn)
                    state.add_edge(ou, ou_conn, outer_work_map_entry, ov_conn, copy.deepcopy(omemlet))
                    print("AE5:", ou, ou_conn, outer_work_map_entry, ov_conn, copy.deepcopy(omemlet))


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
                    print("AE6", u,uc,an,None,Memlet(subset=subset,data=access_node.data,wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo))
                    state.remove_edge(out_edge)
                    state.add_edge(an,None,v,vc,memlet)
                    print("AE7", an,None,v,vc,memlet)

        for out_edge in state.out_edges(thread_block_map_entry):
            u, u_conn, v, v_conn, memlet = out_edge
            if  v == outer_work_map_entry and u_conn == None and v_conn == None:
                state.remove_edge(out_edge)

        for m in [work_map_entry.map, outer_work_map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d
        """

        # We need to move any transient allocation before the outer work map entry
        # We need to move any assignment to transients after the outer work map exit (with a sequential assignment map)

        # Any access node created after the outer work map should be moved before the outer work map
        # 1.
        nodes_to_check = state.all_nodes_between(outer_work_map_entry, thread_coarsened_map_entry)

        # A tmp data is created if we have an edge going to it has no connectors
        # Start 1a.
        for node in nodes_to_check:
            if (isinstance(node, dace.nodes.AccessNode) and
                len(state.in_edges(node)) == 1 and
                len(state.out_edges(node)) == 1 and
                state.in_edges(node)[0].src_conn == None and
                state.in_edges(node)[0].dst_conn == None):
                assert state.in_edges(node)[0].src == outer_work_map_entry
                assert state.out_edges(node)[0].dst == thread_coarsened_map_entry
                # Move the node to before the outer work map
                # Add the node before the outer work map
                ie = state.in_edges(node)[0]
                oe = state.out_edges(node)[0]
                state.remove_node(node)
                state.add_node(node)
                # Add the edge to the node
                outer_map = state.entry_node(outer_work_map_entry)
                state.add_edge(outer_map, None, node, None, copy.deepcopy(ie.data))
                # Add the edge from the node
                state.add_edge(node, None, outer_work_map_entry, oe.dst_conn, copy.deepcopy(oe.data))
                # Put an edge between the next two maps now
                outer_work_map_entry.add_in_connector(oe.dst_conn)
                outer_work_map_entry.add_out_connector(oe.dst_conn.replace("IN_", "OUT_"))
                state.add_edge(outer_work_map_entry, oe.dst_conn.replace("IN_", "OUT_"),
                            thread_coarsened_map_entry, oe.dst_conn, copy.deepcopy(oe.data))
        # End 1a.

        # Move any assignment to any tmp-variable after the outer work map
        # Check all outoing edges from the inner work map, if tmp is accessed, move remaining chain after the otuer work map
        # Start 2a.
        work_map_exit = state.exit_node(work_map_entry)
        thread_coarsened_map_exit = state.exit_node(thread_coarsened_map_entry)
        outer_work_map_exit = state.exit_node(outer_work_map_entry)
        edges_to_rm = set()
        edges_to_add = set()
        __i = 0
        for oe in state.out_edges(work_map_exit):
            u, u_conn, v, v_conn, memlet = oe
            om = state.exit_node(state.entry_node(outer_work_map_entry))
            # 1 assignment chain per output
            if isinstance(v, dace.nodes.AccessNode):
                # 1 assignment map per assignment is needed
                thread_coarsened_map_entry.map.range
                d = {thread_coarsened_map_entry.map.params[i]: thread_coarsened_map_entry.map.range[i]
                    for i in range(len(thread_block_map_entry.map.params))}
                amap_entry, amap_exit = state.add_map(name=f"assignment_map_{__i}",
                              ndrange=d,
                              schedule=dace.ScheduleType.Sequential,
                              unroll=self.unroll)
                __i += 1

                # Find the data accessed in this chain
                _nodes = state.all_nodes_between(v, thread_coarsened_map_exit)
                # Edges that directly come from the thread-coarsened map are not used
                # in the inner work map, we need to re-route them to by-pass the new inner-work map
                # Start 2b.
                # Filter edges from thread-coarsened-map
                edges_from_thread_coarsened = set()
                for node in _nodes:
                    for ie in state.in_edges(node):
                        if ie.src == thread_coarsened_map_entry:
                            edges_from_thread_coarsened.add(ie)

                # Filter the data that has been accessed.
                # These edges need to be removed, but they now go directly to the assignment map
                # entry and exit
                needed_accesses = set()
                for e in edges_from_thread_coarsened:
                    if e.data is not None:
                        if e.data.data is not None:
                            needed_accesses.add(e.data.data)
                            edges_to_rm.add(e)
                            edges_to_add.add((amap_entry,
                                            e.src_conn,
                                            e.dst,
                                            e.dst_conn,
                                            copy.deepcopy(e.data)))
                            amap_entry.add_out_connector(e.src_conn)
                            edges_to_add.add((amap_entry, e.src_conn, e.dst, e.dst_conn,
                                            copy.deepcopy(e.data)))

                # For all needed accesses add an edge in form of:
                # thread block map entry -> assignment map
                for access_name in needed_accesses:
                    edges_from_outer_work_map_exit_to_tblock_map_exit = [
                        e for e in state.out_edges(outer_work_map_exit)
                        if e.dst == om
                    ]
                    assert om == thread_block_map_exit
                    edges_to_accessnode = [e for e in edges_from_outer_work_map_exit_to_tblock_map_exit
                                  if e.data is not None and e.data.data == access_name]
                    assert len(edges_to_accessnode) == 1
                    data_copy_edge = edges_to_accessnode[0]
                    edges_to_rm.add(data_copy_edge)
                    # From thread block map to assignment map directly
                    edges_to_add.add((thread_block_map_entry,
                                      data_copy_edge.src_conn,
                                      amap_entry,
                                      data_copy_edge.dst_conn,
                                      copy.deepcopy(data_copy_edge.data)))
                    # From assignment map exit to outer maps, propagate data
                    edges_to_add.add((amap_exit, data_copy_edge.src_conn,
                                      om, data_copy_edge.dst_conn,
                                      copy.deepcopy(data_copy_edge.data)))

                    # Update connectors
                    amap_entry.add_in_connector(data_copy_edge.dst_conn)
                    amap_exit.add_out_connector(data_copy_edge.src_conn)

                    # Rm any in connector referencing the data, or edges from there
                    # Rm them (and hope it will work out) - rm connectors only from the maps
                    _nodes2 =  state.all_nodes_between(outer_work_map_entry, outer_work_map_exit)
                    for _n in set.union(_nodes2, [outer_work_map_entry, outer_work_map_exit]):
                        for _ie in state.in_edges(_n):
                            if _ie.data is not None and _ie.data.data == access_name:
                                edges_to_rm.add(_ie)
                                if (isinstance(_n, nodes.MapEntry) or
                                    isinstance(_n, nodes.MapExit)):
                                    _n.remove_in_connector(_ie.dst_conn)
                                if (isinstance(_ie.src, nodes.MapEntry) or
                                    isinstance(_ie.src, nodes.MapExit)):
                                    _ie.src.remove_out_connector(_ie.src_conn)
                        for _oe in state.out_edges(_n):
                            if _oe.data is not None and _oe.data.data == access_name:
                                edges_to_rm.add(_oe)
                                if (isinstance(_n, nodes.MapEntry) or
                                    isinstance(_n, nodes.MapExit)):
                                    _n.remove_out_connector(_oe.src_conn)
                                if (isinstance(_oe.dst, nodes.MapEntry) or
                                    isinstance(_oe.dst, nodes.MapExit)):
                                    _oe.dst.remove_in_connector(_oe.dst_conn)

                    thread_block_map_entry.add_out_connector(data_copy_edge.src_conn)
                    thread_block_map_exit.add_in_connector(data_copy_edge.dst_conn)
                # End 2b.

                # The threads that originally connect to the thread coarsened map
                # should connected to the assignment map now
                _nodes_to_thread_coarsened = [n for n in _nodes if thread_coarsened_map_exit in
                                              [e.dst for e in state.out_edges(n)] ]
                for node in _nodes_to_thread_coarsened:
                    assert state.out_degree(node) == 1
                    oe2 = state.out_edges(node)[0]
                    edges_to_rm.add(oe2)
                    edges_to_add.add((oe2.src, oe2.src_conn, amap_exit, oe2.dst_conn, copy.deepcopy(oe2.data)))
                    amap_exit.add_in_connector(oe2.dst_conn)

                # If we are dealing with the access node used
                # Then we need to remove to edge going to int, going out from it
                # and add edges that go:
                # outer work map exit -> access node -> assignment map entry
                # and then assignment map exit -> outer map (thread block)
                assert state.out_degree(v) == 1
                edges_to_rm.add(oe)
                edges_to_rm.add(state.out_edges(v)[0])
                oe3 = state.out_edges(v)[0]

                conn_name = None
                for ie in state.in_edges(work_map_exit):
                    if ie.data is not None and ie.data.data == v.data:
                        conn_name = ie.dst_conn[3:] # Remove the IN_
                        break

                out_conn = "OUT_" + conn_name
                in_conn = "IN_" + conn_name

                # Outer work map exit -> access node
                edges_to_add.add((outer_work_map_exit, out_conn,
                                  v, None,
                                  dace.memlet.Memlet(v.data)))
                # Access node -> assignment map entry
                edges_to_add.add((v, None,
                                  amap_entry, in_conn,
                                  dace.memlet.Memlet(v.data)))
                # Assignment map entry -> whatever came after the access node originally
                edges_to_add.add((amap_entry, out_conn,
                                  oe3.dst, oe3.dst_conn, copy.deepcopy(oe3.data)))

                # The missing edges from inner work map -> thread coarsened map and
                # thread coarsened map -> outer work map need to be added
                edges_to_add.add((work_map_exit, out_conn,
                                  thread_coarsened_map_exit, in_conn,
                                  dace.memlet.Memlet(v.data)))
                edges_to_add.add((thread_coarsened_map_exit, out_conn,
                                  outer_work_map_exit, in_conn,
                                  dace.memlet.Memlet(v.data)))

                # Add missing connectors
                thread_coarsened_map_exit.add_in_connector(in_conn)
                thread_coarsened_map_exit.add_out_connector(out_conn)
                outer_work_map_exit.add_in_connector(in_conn)
                outer_work_map_exit.add_out_connector(out_conn)
                amap_entry.add_in_connector(in_conn)
                amap_entry.add_out_connector(out_conn)
        # End 2a.

        # Now update the edges
        for e in edges_to_rm:
            state.remove_edge(e)
        for e in edges_to_add:
            state.add_edge(*e)


        work_map.label = f"InnerWorkMapNo{BlockTiling.global_application_number}"
        outer_work_map.label = f"OuterWorkMapNo{BlockTiling.global_application_number}"

        BlockTiling.global_application_number += 1

    @staticmethod
    def annotates_memlets():
        return True
