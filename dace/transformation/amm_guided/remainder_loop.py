# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
from dace import symbolic
import dace
import copy

@make_properties
class RemainderLoop(transformation.SingleStateTransformation):
    inner_work_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.inner_work_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def update_names():
        pass

    def has_remainder(self, numerator, denominator):
        if numerator.is_integer and denominator.is_integer:
            return numerator % denominator != 0
        else:
            return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        """
        other_tiled_maps = []
        cur_map_entry = state.entry_node(self.map_entry)
        dev_map_entry = None
        thread_block_map_entry = None
        while cur_map_entry:
            if cur_map_entry.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock and \
                any([step != 1 for (_,_,step) in cur_map_entry.map.range]):
                other_tiled_maps.append(cur_map_entry)
            if cur_map_entry.map.schedule == dtypes.ScheduleType.GPU_Device:
                dev_map_entry = cur_map_entry
            if cur_map_entry.map.schedule == dtypes.ScheduleTyoe.GPU_ThreadBlock:
                thread_block_map_entry = cur_map_entry
            cur_map_entry = state.entry_node(cur_map_entry)

        s = ""
        for map_entry in other_tiled_maps:
            for (beg,end,step) in map_entry.map.range:
                if step != 1:
                    
        pass
        """
        # Workflow:
        # 1. Save all unique data accesses inside the micro kernel
        # e.g. A[b_i+d_i,t+tK], B[t+tK,b_j+d_j], tmp[i,j]
        # 2. Find the ranges of these parameters
        # e.g. tK = 0:16:1, k = 0:K:16, i = 0:4
        # 3. Check if any variable can go out of bound (if end-beg % step != 0)
        #    then put them in a unique if condition
        # 4. Go through the unique loop variables appearing in the condition
        # e.g. k, b_i, b_j
        # 5. Find the outermost location where one can apply the if else
        # 6. Create if condition, check if bound check is ensurable 
        # e.g, Meaning not at last iteration of k, not last iteration of b_i, b_j
        #    6.1 Yes -> statically unrolled loops (same)
        #    6.2 No -> Copy code change end condition with min(end, var+step)
        inner_work_map_entry = self.inner_work_map_entry

        # 1.
        edges_to_check = state.out_edges(inner_work_map_entry)
        data_accesses = set()
        while edges_to_check:
            u, u_conn, v, v_conn, memlet = edges_to_check.pop()
            if not (memlet.data, memlet.subset) in data_accesses:
                data_accesses.add((memlet.data, memlet.subset))
            if v != state.exit_node(inner_work_map_entry):
                edges_to_check += state.out_edges(v)

        # 2.
        param_and_ranges = dict()
        cur_map_entry = self.inner_work_map_entry
        while cur_map_entry:
            for param, (beg,end,step) in zip(cur_map_entry.map.params, cur_map_entry.map.range):
                assert(not param in param_and_ranges.keys())
                param_and_ranges[param] = (beg,end,step)
            cur_map_entry = state.entry_node(cur_map_entry)
        print(param_and_ranges)

        # 3.
        # Let us look at B[k + tk, b_j + d_j + j] 
        # k=0:K:16, tk=0:16:1, b_j=0:N:64, d_j=0:64:4, j=0:4:1
        # k + tk -> K % 16 -> can overflow
        # b_j + d_j + j -> N % 16 -> can overflow (the other params can't overflow)
        can_out_of_bound = dict()
        symbols_to_ensure_in_scope = set()
        for (data, subset) in data_accesses:
            for (beg,end,step) in subset:
                free_symbols = set.union(beg.free_symbols, end.free_symbols, step.free_symbols)
                for symbol in free_symbols:
                    (beg,end,step) = param_and_ranges[str(symbol)]
                    if self.has_remainder(end+1-beg, step):
                        can_out_of_bound[(data,subset)] = True
                        symbols_to_ensure_in_scope.add(str(symbol))
                if not (data,subset) in can_out_of_bound.keys():
                    can_out_of_bound[(data,subset)] = False
                print(free_symbols)
        print(can_out_of_bound)
        print(symbols_to_ensure_in_scope)
        
        # 4. Set up the condition
        added_conditions = set()
        for (data, range), b in set(can_out_of_bound.items()):
            if b:
                for i, (beg,end,step) in enumerate(range):
                    added_conditions.add(f"({beg} < {sdfg.arrays[data].shape[i]})")
        condition = "(" + " and ".join(added_conditions) + ")"
        print(condition)

        # 5. Go up until all the variables are defined
        map_before_split = None
        cur_map_entry = self.inner_work_map_entry
        while cur_map_entry:
            if any([param in cur_map_entry.map.params for param in symbols_to_ensure_in_scope]):
                map_before_split = cur_map_entry
                break
            cur_map_entry = state.entry_node(cur_map_entry)
        map_after_split = state.exit_node(map_before_split)
        print(map_before_split)

        # 6. Create if condition
        inner_kernel_entry : nodes.MapEntry = map_before_split
        inner_kernel_exit : nodes.MapExit = state.exit_node(inner_kernel_entry)

        sub_sdfg : SDFG = dace.SDFG('nested_sub', parent=sdfg)
        inner_loop_kernel_sdfg : SDFG = dace.SDFG('innerLoopKernel', parent=sub_sdfg)
        remainder_loop_kernel_sdfg : SDFG  = dace.SDFG('outerLoopKernel', parent=sub_sdfg)

        # Declare arrays for nested SDFG (the names can differ from the top-level SDFG)
        # In this case, we read only one element out of the full arrays
        
        # Inputs for NestedSDFG
        ins = set()
        for in_edge in state.out_edges(map_before_split):
            _,_,_,_,memlet = in_edge
            ins.add(memlet.data)
        # Outputs for NestedSDFG
        outs = set()
        for out_edge in state.in_edges(map_after_split):
            _,_,_,_,memlet = out_edge
            outs.add(memlet.data)

        for (prefix, _sdfg) in [
            ("", sub_sdfg),
            ("", inner_loop_kernel_sdfg),
            ("", remainder_loop_kernel_sdfg)
        ]:
            for in_arr in set.union(ins, outs):
                _sdfg.add_array(name=prefix + in_arr, 
                                shape=sdfg.arrays[in_arr].shape,
                                transient=False,#sdfg.arrays[in_arr].transient, 
                                dtype=sdfg.arrays[in_arr].dtype)
            #for in_arr in set.difference(set(sdfg.arrays.keys()), set.union(ins, outs)):
            #    _sdfg.add_array(name=prefix + in_arr, 
            #                    shape=sdfg.arrays[in_arr].shape,
            #                    transient=sdfg.arrays[in_arr].transient, 
            #                    dtype=sdfg.arrays[in_arr].dtype)

        # Create nested states

        # Save the in and out edges of the map scope encompassing the inner kernels
        # This is necessary to creaet edges from and to access nodes within the sub kernel
        special_in_edges  = []
        special_out_edges = []
        for out_edge in state.out_edges(map_before_split):
            special_in_edges.append(out_edge)
        for out_edge in state.in_edges(map_after_split):
            special_out_edges.append(out_edge)

        state0 = sub_sdfg.add_state('if_guard')
        state1 = sub_sdfg.add_state('innerLoopKernel')
        state2 = sub_sdfg.add_state('remainderLoopKernel')
        state3 = sub_sdfg.add_state('complete')

        sub_sdfg.add_edge(state0, state1, dace.InterstateEdge(condition=f"{condition}"))
        sub_sdfg.add_edge(state0, state2, dace.InterstateEdge(condition=f"not {condition}"))
        sub_sdfg.add_edge(state1, state3, dace.InterstateEdge())
        sub_sdfg.add_edge(state2, state3, dace.InterstateEdge())

        sym_map = dict()
        for sym in sdfg.free_symbols:
            sym_map[sym] = sym

        lnsdfg : nodes.NestedSDFG = state1.add_nested_sdfg(inner_loop_kernel_sdfg, sub_sdfg, ins, outs)
        rnsdfg : nodes.NestedSDFG = state2.add_nested_sdfg(remainder_loop_kernel_sdfg, sub_sdfg, ins, outs)
        nsdfg  : nodes.NestedSDFG = state.add_nested_sdfg(sub_sdfg, sdfg,  ins, outs)

        lkernel : SDFGState = inner_loop_kernel_sdfg.add_state('kernel')
        rkernel : SDFGState = remainder_loop_kernel_sdfg.add_state('kernel')

        # Add necessary input and output access nodes
        for in_arr in ins:
            for (prefix, _state, _sdfg, node) in [
                ("", state1, inner_loop_kernel_sdfg, lnsdfg),
                ("", state2, remainder_loop_kernel_sdfg, rnsdfg)
            ]:
                an = nodes.AccessNode(data=in_arr)
                # Extract the same edge as of the element
                edge = next(((u, u_conn, v, v_conn, memlet) 
                             for (u, u_conn, v, v_conn, memlet) in 
                             state.out_edges(map_before_split) if memlet.data == in_arr), 
                            None)
                _state.add_node(an)
                u,u_conn,v, v_conn,memlet = edge
                _state.add_edge(an, None, node, prefix + in_arr, Memlet(data=in_arr,subset=memlet.subset))
        for out_arr in outs:
            for (prefix, _state, _sdfg, node) in [
                ("", state1, inner_loop_kernel_sdfg, lnsdfg),
                ("", state2, remainder_loop_kernel_sdfg, rnsdfg)
            ]:
                an = nodes.AccessNode(data=out_arr)
                # Extract the same edge as of the element
                edge = next(((u, u_conn, v, v_conn, memlet) 
                             for (u, u_conn, v, v_conn, memlet) in 
                             state.in_edges(map_after_split) if memlet.data == out_arr), 
                            None)
                _state.add_node(an)
                u,u_conn,v, v_conn,memlet = edge
                _state.add_edge(node, prefix + out_arr, an, None, Memlet(data=out_arr,subset=memlet.subset))


        for out_edge in state.out_edges(map_before_split):
            u,u_conn,v,v_conn,memlet = out_edge
            state.add_edge(u, u_conn, nsdfg, memlet.data, copy.deepcopy(memlet))

        for in_edge in state.in_edges(map_after_split):
            u,u_conn,v,v_conn,memlet = in_edge
            state.add_edge(nsdfg, memlet.data, v, v_conn, copy.deepcopy(memlet))

        #for n in nodes_to_copyover:
        #    lkernel.add_node(copy.deepcopy(node))
        #for e in edges_to_copyover:
        #    lkernel.add_edge(*copy.deepcopy(e))

        nodes_to_copyover = set()
        edges_to_copyover = set()
        
        # Go through access nodes, add needed transient arrays
        # Save all nodes and edges between the maps, remove them and copy them into the state
        ncheck = [map_before_split]
        while ncheck:
            n = ncheck.pop()
            if n != nsdfg and n != map_after_split:
                if n != map_before_split:
                    nodes_to_copyover.add(n)
                ncheck += [v for (u,uc,v,vc,m) in state.out_edges(n)]
                edges_to_copyover = edges_to_copyover.union(set([e # Unpacking causes problems in removing
                                                                 for e,(u,uc,v,vc,m) in 
                                                                 zip(state.out_edges(n), state.out_edges(n))
                                                                 if v != nsdfg and v != nsdfg]))

        print("\n".join([str(s) for s in nodes_to_copyover]))
        print("----------------------")
        print("\n".join([str(s) for s in edges_to_copyover]))


        for e in edges_to_copyover:
            state.remove_edge(e)
        for n in nodes_to_copyover:
            state.remove_node(n)

        for kernel, kernel_sdfg in [(lkernel, inner_loop_kernel_sdfg), (rkernel, remainder_loop_kernel_sdfg)]:
            added_labels = dict()
            # Update this to use uuids
            for n in nodes_to_copyover:
                nn = copy.deepcopy(n)
                kernel.add_node(nn)
                added_labels[n.__str__()] = nn
            print(added_labels)
            for e in edges_to_copyover:
                u,uc,v,vc,memlet = e
                uu = added_labels[u.__str__()]
                vv = added_labels[v.__str__()]
                kernel.add_edge(uu,uc,vv,vc,copy.deepcopy(memlet))
            sdfg.save("uwu.sdfg")
            
            first_node = next(dace.sdfg.utils.dfs_topological_sort(kernel))
            if not isinstance(first_node, nodes.MapEntry):
                raise Exception(f"First node in the map currently needs to be a MapEntry it is: {type(first_node)}")
            last_node = kernel.exit_node(first_node)

            for e in special_in_edges:
                u,uc,v,vc,memlet = e
                new_memlet = Memlet(data=memlet.data, subset=memlet.subset)
                an = nodes.AccessNode(data=memlet.data)
                kernel.add_node(an)
                kernel.add_edge(an,None,first_node,vc,new_memlet)
            for e in special_out_edges:
                u,uc,v,vc,memlet = e
                new_memlet = Memlet(data=memlet.data, subset=memlet.subset)
                an = nodes.AccessNode(data=memlet.data)
                kernel.add_node(an)
                kernel.add_edge(last_node,uc,an,None,new_memlet)

            # Re-allocate transients that are first accessed in the inner kernel
            for n in dace.sdfg.utils.dfs_topological_sort(kernel):
                if isinstance(n, nodes.AccessNode):
                    if not n.data in kernel_sdfg.arrays.keys():
                        arr = sdfg.arrays[n.data]
                        assert(arr.transient)
                        if isinstance(arr, dace.data.Array):
                            kernel_sdfg.add_array(
                                name=arr.name,
                                shape=arr.shape,
                                transient=True,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                lifetime=arr.lifetime
                            )
                        else:
                            assert(isinstance(arr, dace.data.Scalar))
                            kernel_sdfg.add_scalar(
                                name=n.data,
                                transient=True,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                lifetime=arr.lifetime
                            )

        print(sdfg.used_symbols(all_symbols=True))
        print(sdfg.free_symbols)
        print(lkernel.free_symbols)
        print(sdfg.symbols)

        # Remove any free symbol that is only used in the sub-kernel
        symbols_to_remove = set.difference(set(sdfg.free_symbols), set(lkernel.free_symbols))
        #for symbol in symbols_to_remove:
        #    sdfg.remove_symbol(symbol)
        #sdfg.free_symbols = sdfg.free_symbols.difference(symbols_to_remove)

    @staticmethod
    def annotates_memlets():
        return False
