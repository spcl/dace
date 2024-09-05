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
from dace.subsets import Range

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
        # Workflow:
        # 1. Save all unique global data accesses
        # e.g. A[b_i+d_i,tk+k], B[tk+k,b_j+d_j], tmp[i,j]
        # but if the kernel has been moves from glb to for example shared then keep track of it
        # (0.) as  shrA -> A
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
        #
        # S2. one special case if memory is moved from global to shared memory
        #     then we need preserve the mapping of shrA -> glbA when checking any entry
        #     this can be done by going through library nodes whr data was moved from glb to shr
        inner_work_map_entry = self.inner_work_map_entry
        map_entry = self.inner_work_map_entry
        dev_entry = None
        while map_entry:
            dev_entry = map_entry
            map_entry = state.entry_node(map_entry)
        assert(dev_entry.map.schedule == dtypes.ScheduleType.GPU_Device)

        # 0. Sort memory-moved array into groups
        access_groups = list()
        for n in dace.sdfg.utils.dfs_topological_sort(state):
            if isinstance(n, nodes.LibraryNode):
                ies = state.in_edges(n)
                oes = state.out_edges(n)
                if len(ies) == 1 and len(oes) == 1:
                    ie = ies[0]
                    oe = oes[0]
                    iu,iuc,iv,ivc,imemlet = ie
                    ou,ouc,ov,ovc,omemlet = oe
                    if sdfg.arrays[imemlet.data].storage != sdfg.arrays[omemlet.data].storage:
                        added = False
                        for i, group in enumerate(access_groups):
                            if imemlet.data in group or omemlet.data in group:
                                added = True
                                updated_group = group.union([(imemlet.data, imemlet.subset), (omemlet.data, omemlet.subset)])
                                access_groups[i] = updated_group
                        if not added:
                            access_groups.append(set([(imemlet.data, imemlet.subset), (omemlet.data, omemlet.subset)]))
        print("ACCESS GROUPS", access_groups)

        # 1.
        edges_to_check = state.out_edges(dev_entry)
        data_accesses = set()
        while edges_to_check:
            u, u_conn, v, v_conn, memlet = edges_to_check.pop()
            # When smth is allocated memlet data and memlet subset can be none
            if memlet.data != None and memlet.subset != None and (not (memlet.data, memlet.subset) in data_accesses):
                data_accesses.add((memlet.data, memlet.subset))
            if v != state.exit_node(inner_work_map_entry):
                edges_to_check += state.out_edges(v)
        print("DATA ACCESSES", data_accesses)

        # 2.
        param_and_ranges = dict()
        cur_map_entry = self.inner_work_map_entry
        while cur_map_entry:
            for param, (beg,end,step) in zip(cur_map_entry.map.params, cur_map_entry.map.range):
                assert(not param in param_and_ranges.keys())
                param_and_ranges[param] = (beg,end,step)
            cur_map_entry = state.entry_node(cur_map_entry)
        print("PARAM AND RANGES", param_and_ranges)
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
                print("free symbols", free_symbols)
                for symbol in free_symbols:
                    if str(symbol) in param_and_ranges.keys():
                        (beg,end,step) = param_and_ranges[str(symbol)]
                        if self.has_remainder(end+1-beg, step):
                            can_out_of_bound[(data,subset)] = True
                            symbols_to_ensure_in_scope.add(str(symbol))
                if not (data,subset) in can_out_of_bound.keys():
                    can_out_of_bound[(data,subset)] = False
        # If one element in a access group can out of bound then all such elements need to be set to true
        for (data,subset), bool in can_out_of_bound.items():
            for group in access_groups:
                group_can_out_of_bound = True
                for (_d, _s) in group:
                    if (_d,_s) in can_out_of_bound.keys() and can_out_of_bound[(_d,_s)]:
                        group_can_out_of_bound = True
                        break

                if group_can_out_of_bound:
                    for (_d,_s) in group:
                        if (data,subset) == (_d,_s):
                            can_out_of_bound[(data,subset)] = True

        print("CAN OUT OF BOUND", can_out_of_bound)
        print("SYMS TO ENSURE IN SCOPE", symbols_to_ensure_in_scope)
        # 5. Go up until all the variables are defined
        map_before_split = None
        print("INNER WORK MAP ENTRY AND SYMYS TO ENSURE", self.inner_work_map_entry, symbols_to_ensure_in_scope)
        cur_map_entry = self.inner_work_map_entry
        while cur_map_entry:
            if any([param in cur_map_entry.map.params for param in symbols_to_ensure_in_scope]):
                map_before_split = cur_map_entry
                break
            cur_map_entry = state.entry_node(cur_map_entry)
        sdfg.save("aaaa.sdfg")
        print("MAP BEFORE SPLIT", map_before_split)


        map_after_split = state.exit_node(map_before_split)


        ncheck = [v for (u,uc,v,vc,m) in state.out_edges(dev_entry)]
        loop_vars_and_ranges = dict()
        while ncheck:
            n = ncheck.pop()
            if isinstance(n, nodes.MapEntry):
                for param, range in zip(n.map.params, n.map.range):
                    beg,end,step = range
                    if n == map_before_split:
                        map_len = step
                    elif n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                        map_len = step
                    else:
                        map_len = ((end+1)-beg)
                    if (map_len.is_integer):
                        #raise Exception("For RemainderLoop transformation to work inner maps need to have static ranges")
                        loop_vars_and_ranges[param] = map_len
            if n != state.exit_node(dev_entry):
                ncheck += [v for (u,uc,v,vc,m) in state.out_edges(n)]

        print("INLP2", loop_vars_and_ranges)
        # 4. Set up the condition
        added_conditions = set()
        conditions_and_ranges = dict()
        for (data, range), b in set(can_out_of_bound.items()):
            if b and sdfg.arrays[data].storage == dtypes.StorageType.GPU_Shared:
                # Find the corresponding globala array
                glb_arr_name = None
                glb_arr = None
                accumulated_beg = [0] * len(range)
                
                for group in access_groups:
                    if (data, range) in group:
                        for _d, _r in group:
                            if sdfg.arrays[_d].storage == dtypes.StorageType.GPU_Global:
                                glb_arr_name = _d
                                glb_arr = sdfg.arrays[_d]
                                for i,(_b,_,_) in enumerate(_r):
                                    accumulated_beg[i] += _b

                for i, (beg,end,step) in enumerate(range):
                    ns = []
                    lim = 0
                    print("A1", data, range, beg.free_symbols)
                    for sym in beg.free_symbols:
                        if str(sym) in loop_vars_and_ranges.keys():
                            print("A2", loop_vars_and_ranges[str(sym)])
                            ns.append(str(accumulated_beg[i]))
                            lim = loop_vars_and_ranges[str(sym)]
                            conditions_and_ranges[str(sym)] = (ns, glb_arr.shape[i], lim)
                        elif str(sym) in map_before_split.map.params:
                            raise Exception("TODO")
                    # It is a an access to the part that was shortened using BlockTiling (of a work map)
                    if not beg.free_symbols:
                        # In this the "beg" in the global array should be always a symbol
                        ns.append(str(accumulated_beg[i]))
                        lim = loop_vars_and_ranges[str(accumulated_beg[i])]
                        param_id = -1
                        for j, param in enumerate(map_before_split.map.params):
                            if param == str(accumulated_beg[j]):
                                param_id = j
                                break
                        conditions_and_ranges[self.inner_work_map_entry.map.params[param_id]] = (ns, glb_arr.shape[i], lim)
                    added_conditions.add(f"({" + ".join(ns + [str(lim)])} <= {glb_arr.shape[i]})")
        """
        conditions_and_ranges = dict()
        for (data, range), b in set(can_out_of_bound.items()):
            print("d", data, range)
            if b and sdfg.arrays[data].storage == dtypes.StorageType.GPU_Global:
                # Special case, we need to combine all offsets of
                for i, (beg,end,step) in enumerate(range):
                    ns = []
                    ns_sym = []
                    var = None
                    lim = None
                    print("d2", data, ", ", range)
                    for sym in beg.free_symbols:
                        print("d3", sym, ", ", beg.free_symbols)
                        print("d33", str(sym), ", ", inner_loop_vars_and_ranges.keys())
                        if str(sym) in inner_loop_vars_and_ranges.keys():
                            ns.append(str(inner_loop_vars_and_ranges[str(sym)]))
                            #ns_sym.append(inner_loop_vars_and_ranges[str(sym)])
                            lim = inner_loop_vars_and_ranges[str(sym)]
                            var = sym
                        elif str(sym) in map_before_split.map.params:
                            for par, range in zip(map_before_split.map.params, map_before_split.map.range):
                                if par == str(sym):
                                    b,e,s = range
                                    ns.append(str(e+1-b))
                                    #ns_sym.append(inner_loop_vars_and_ranges[str(sym)])
                                    lim = e+1-b
                                    var = sym
                        else:
                            ns.append(str(sym))
                            ns_sym.append(sym)
                    #conditions_and_ranges[sym] = (ns, sdfg.arrays[data].shape[i])
                    added_conditions.add(f"({" + ".join(ns)} < {sdfg.arrays[data].shape[i]})")
                    conditions_and_ranges[var] = (ns_sym, sdfg.arrays[data].shape[i], lim)
        """
        print("Added conditions", added_conditions)
        condition = "(" + " and ".join(added_conditions) + ")"

        # For example range for k is 0:K:16,
        # In this last iteration the inner tk goes from 0:16
        # In the remainder loop it should go from 0:16 to 0:Min(K-(Sum all previous var),16)
        print("Conditions and ranges", conditions_and_ranges)

        # 6. Create if condition
        inner_kernel_entry : nodes.MapEntry = map_before_split
        inner_kernel_exit : nodes.MapExit = state.exit_node(inner_kernel_entry)

        sub_sdfg : SDFG = dace.SDFG('nested_sub', parent=sdfg)
        inner_loop_kernel_sdfg : SDFG = dace.SDFG('innerLoopKernel', parent=sub_sdfg)
        remainder_loop_kernel_sdfg : SDFG  = dace.SDFG('outerLoopKernel', parent=sub_sdfg)

        # Declare arrays for nested SDFG (the names can differ from the top-level SDFG)
        # In this case, we read only one element out of the full arrays


        # Inputs for NestedSDFG
        # Create nested states
        first_node_in_microkernel = None
        for e in state.out_edges(map_before_split):
            u,uc,v,vc,m = e
            if isinstance(v, nodes.MapEntry):
                first_node_in_microkernel = v
                break
        if not first_node_in_microkernel:
            raise Exception("At least on edge needs to directly connect the map to be split before and the next map")
        last_node_in_microkernel = state.exit_node(first_node_in_microkernel)

        # I think this part is super fragile
        # It basically maps an shrA[d_i + i,...] access to the limits of the varaible d_i, as we cant catch the limits of i from the implementation
        # TODO find a mapping where the range limit for d_i also holds for i.
        # Since my transformations uses d_ as the thread coarsened prefix I can just apply that
        adders = []
        for (var, (loop_vars, shape, lim)) in conditions_and_ranges.items():
            if var.startswith("d_"):
               adders.append((var[2:], (loop_vars, shape, lim)))
        for k,v in adders:
            conditions_and_ranges[k] = v

        new_ranges = dict()
        for (var, (loop_vars, shape, lim)) in conditions_and_ranges.items():
            new_ranges[var] = \
                ( (0,symbolic.SymExpr(f"Min({shape} - ({'+'.join([str(s) for s in loop_vars])}), {lim})-1"),1),
                  (0,lim-1,1) )

        ins = set()
        for in_edge in state.in_edges(first_node_in_microkernel):
            _,_,_,_,memlet = in_edge
            ins.add(memlet.data)
        # Outputs for NestedSDFG
        outs = set()
        for out_edge in state.out_edges(last_node_in_microkernel):
            _,_,_,_,memlet = out_edge
            outs.add(memlet.data)

        for ( _sdfg) in [
            (sub_sdfg),
            (inner_loop_kernel_sdfg),
            (remainder_loop_kernel_sdfg)
        ]:
            for in_arr in set.union(ins, outs):
                _sdfg.add_array(name=in_arr, 
                                shape=sdfg.arrays[in_arr].shape,
                                transient=False, 
                                dtype=sdfg.arrays[in_arr].dtype)

        # S1. We need to calculate the offsets we need to substract from all data containers passed to sub graphs, we can do it through going through in and out arrays
        offsets = self.create_offsets(state.in_edges(first_node_in_microkernel) + state.out_edges(last_node_in_microkernel))

        # Save the in and out edges of the map scope encompassing the inner kernels
        # This is necessary to creaet edges from and to access nodes within the sub kernel
        special_in_edges  = []
        special_out_edges = []
        for in_edge in state.in_edges(first_node_in_microkernel):
            special_in_edges.append(in_edge)
        for out_edge in state.out_edges(last_node_in_microkernel):
            special_out_edges.append(out_edge)

        state0 = sub_sdfg.add_state('if_guard')
        state1 = sub_sdfg.add_state('innerLoopKernel')
        lkernel_parent_state = state1
        state2 = sub_sdfg.add_state('remainderLoopKernel')
        rkernel_parent_state = state2
        state3 = sub_sdfg.add_state('complete')

        sub_sdfg.add_edge(state0, state1, dace.InterstateEdge(condition=f"{condition}"))
        sub_sdfg.add_edge(state0, state2, dace.InterstateEdge(condition=f"not {condition}"))
        sub_sdfg.add_edge(state1, state3, dace.InterstateEdge())
        sub_sdfg.add_edge(state2, state3, dace.InterstateEdge())

        lkernel : SDFGState = inner_loop_kernel_sdfg.add_state('kernel')
        rkernel : SDFGState = remainder_loop_kernel_sdfg.add_state('kernel')

        nsdfg  : nodes.NestedSDFG = state.add_nested_sdfg(sub_sdfg, sdfg, ins, outs)

        lnsdfg : nodes.NestedSDFG = state1.add_nested_sdfg(inner_loop_kernel_sdfg, sub_sdfg, ins, outs, copy.deepcopy(nsdfg.symbol_mapping))
        rnsdfg : nodes.NestedSDFG = state2.add_nested_sdfg(remainder_loop_kernel_sdfg, sub_sdfg, ins, outs, copy.deepcopy(nsdfg.symbol_mapping))

        # Add necessary input and output access nodes
        for in_arr in ins:
            for (_state, _sdfg, node) in [
                (state1, inner_loop_kernel_sdfg, lnsdfg),
                (state2, remainder_loop_kernel_sdfg, rnsdfg)
            ]:
                an = nodes.AccessNode(data=in_arr)
                # Extract the same edge as of the element
                edge = next(((u, u_conn, v, v_conn, memlet) 
                             for (u, u_conn, v, v_conn, memlet) in 
                             state.in_edges(first_node_in_microkernel) if memlet.data == in_arr), 
                            None)
                _state.add_node(an)
                u,u_conn,v, v_conn,memlet = edge
                _state.add_edge(an, None, node, in_arr, Memlet(data=in_arr,subset=Range(memlet.subset)))
        for out_arr in outs:
            for (_state, _sdfg, node) in [
                (state1, inner_loop_kernel_sdfg, lnsdfg),
                (state2, remainder_loop_kernel_sdfg, rnsdfg)
            ]:
                an = nodes.AccessNode(data=out_arr)
                # Extract the same edge as of the element
                edge = next(((u, u_conn, v, v_conn, memlet) 
                             for (u, u_conn, v, v_conn, memlet) in 
                             state.out_edges(last_node_in_microkernel) if memlet.data == out_arr), 
                            None)
                _state.add_node(an)
                u,u_conn,v, v_conn,memlet = edge
                _state.add_edge(node, out_arr, an, None, Memlet(data=out_arr,subset=Range(memlet.subset)))

        # Edges that go to and come out nested sdfg
        for in_edge in state.in_edges(first_node_in_microkernel):
            u,u_conn,v,v_conn,memlet = in_edge
            state.add_edge(u, u_conn, nsdfg, memlet.data, Memlet(data=memlet.data,subset=Range(memlet.subset)))

        for out_edge in state.out_edges(last_node_in_microkernel):
            u,u_conn,v,v_conn,memlet = out_edge
            state.add_edge(nsdfg, memlet.data, v, v_conn, Memlet(data=memlet.data,subset=Range(memlet.subset)))

        nodes_to_copyover = set()
        edges_to_copyover = set()
        
        # Go through access nodes, add needed transient arrays
        # Save all nodes and edges between the maps, remove them and copy them into the state
        ncheck = [first_node_in_microkernel]
        while ncheck:
            n = ncheck.pop()
            if n != nsdfg:
                if n != map_before_split and n != map_after_split:
                    nodes_to_copyover.add(n)

                    edges_to_copyover = edges_to_copyover.union(set([e # Unpacking causes problems in removing
                                                                 for e,(u,uc,v,vc,m) in 
                                                                 zip(state.out_edges(n), state.out_edges(n))
                                                                 if v != nsdfg and u != last_node_in_microkernel]))
                if n != last_node_in_microkernel:
                    ncheck += [v for (u,uc,v,vc,m) in state.out_edges(n)]

        for e in edges_to_copyover:
            state.remove_edge(e)
        for n in nodes_to_copyover:
            state.remove_node(n)

        for kernel, kernel_sdfg in [(lkernel, inner_loop_kernel_sdfg), (rkernel, remainder_loop_kernel_sdfg)]:
            added_labels = dict()

            for n in nodes_to_copyover:
                nn : nodes.Node = copy.deepcopy(n)
                kernel.add_node(nn)
                added_labels[n.guid] = nn
            for e in edges_to_copyover:
                u,uc,v,vc,memlet = e
                uu = added_labels[u.guid]
                vv = added_labels[v.guid]
                kernel.add_edge(uu,uc,vv,vc,copy.deepcopy(memlet))

            first_node = next(dace.sdfg.utils.dfs_topological_sort(kernel))
            if not isinstance(first_node, nodes.MapEntry):
                sdfg.save("uwuowo.sdfg")
                raise Exception(f"First node in the map currently needs to be a MapEntry it is: {type(first_node)}")
            last_node = kernel.exit_node(first_node)

            for e in special_in_edges:
                u,uc,v,vc,memlet = e
                new_memlet = Memlet(data=memlet.data, subset=Range(memlet.subset))
                an = nodes.AccessNode(data=memlet.data)
                kernel.add_node(an)
                kernel.add_edge(an,None,first_node,vc,new_memlet)
            for e in special_out_edges:
                u,uc,v,vc,memlet = e
                new_memlet = Memlet(data=memlet.data, subset=Range(memlet.subset))
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

        # Now we have two valid SDFGS
        # 6.2 Arrange ranges of the remainder Loop
        for n in dace.sdfg.utils.dfs_topological_sort(rkernel):
            if isinstance(n, nodes.MapEntry) or isinstance(n, nodes.MapExit):
                nr = []
                for i, (p, r) in enumerate(zip(n.map.params, n.map.range)):
                    print("PP", p, r, new_ranges, p in new_ranges.keys())
                    if p in new_ranges.keys():
                        new_range, old_range = new_ranges[p]
                        rb,re,rs = r
                        orb,ore,ors = old_range
                        if (rb != orb or re != ore or rs != ors):
                            raise Exception(f"For the transformation to work any map range needs to have the format 0:N:1, had: {r} and {old_range}")
                        nr.append(new_range)
                    else:
                        nr.append(r)
                print("NR", nr)
                n.map.range = Range(nr)
                print("MR", n.map.range)

        # Now need to apply it also on any assignments
        counter = 0
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode):
                can_out_of_bound = False
                i_memlet = None
                o_memlet = None
                if (len(state.in_edges(n)) > 1 or len(state.out_edges(n)) > 1):
                    raise Exception("Transformation can be applied if all assignment edges have at max in and out degree 1 each")
                if (len(state.in_edges(n)) == 0 or len(state.out_edges(n)) == 0):
                    continue
                for ie, oe in zip(state.in_edges(n), state.out_edges(n)):
                    _,_,_,_,i_memlet = ie
                    _,_,_,_,o_memlet = oe
                    print(i_memlet, o_memlet)
                    if i_memlet.data == None or o_memlet.data == None:
                        continue
                    for ((i_beg,i_end,i_step), (o_beg,o_end,o_step), i_dim, o_dim) in zip(i_memlet.subset, o_memlet.subset, sdfg.arrays[i_memlet.data].shape, sdfg.arrays[o_memlet.data].shape):
                        il = (i_end+1-i_beg)/i_step
                        ol = (o_end+1-o_beg)/o_step
                        print(self.has_remainder(i_dim, il), self.has_remainder(o_dim, ol))
                        print(sdfg.arrays[i_memlet.data].transient, sdfg.arrays[o_memlet.data].transient)
                        if (self.has_remainder(i_dim, il) or self.has_remainder(o_dim, ol)) and \
                            sdfg.arrays[i_memlet.data].transient and not sdfg.arrays[o_memlet.data].transient:
                            can_out_of_bound = True
                            break
                if can_out_of_bound:
                    counter += 1
                    assign_sub_sdfg : SDFG = dace.SDFG(f'nestedAssignment{counter}', parent=sdfg)
                    assign_inner_sdfg : SDFG = dace.SDFG(f'innerAssignment{counter}', parent=sub_sdfg)
                    assign_remainder_sdfg : SDFG  = dace.SDFG(f'remainderAssignment{counter}', parent=sub_sdfg)

                    # To split
                    # Inputs for NestedSDFG
                    ins = set([i_memlet.data])
                    # Outputs for NestedSDFG
                    outs = set([o_memlet.data])

                    for ( _sdfg) in [
                        (assign_sub_sdfg),
                        (assign_inner_sdfg),
                        (assign_remainder_sdfg)
                    ]:
                        for in_arr in set.union(ins, outs):
                            _sdfg.add_array(name=in_arr, 
                                            shape=sdfg.arrays[in_arr].shape,
                                            transient=False, 
                                            dtype=sdfg.arrays[in_arr].dtype)

                    # Create nested states

                    # Save the in and out edges of the map scope encompassing the inner kernels
                    # This is necessary to creaet edges from and to access nodes within the sub kernel
                    special_in_edges  = [ie]
                    special_out_edges = [oe]

                    assign_offsets = self.create_offsets([ie,oe])

                    state0 = assign_sub_sdfg.add_state('if_guard')
                    state1 = assign_sub_sdfg.add_state('innerAssignment')
                    lassign_parent_state = state1
                    state2 = assign_sub_sdfg.add_state('remainderAssignment')
                    rassign_parent_state = state2
                    state3 = assign_sub_sdfg.add_state('complete')

                    conditions = []
                    for (beg,end,step), dim in zip(o_memlet.subset, sdfg.arrays[o_memlet.data].shape):
                        ol = (o_end+1-o_beg)/o_step
                        cond = f"{str(beg)} + {str(ol)} <= {dim}"
                        conditions.append(cond)
                    condition = "(" + " and ".join(conditions) + ")"

                    assign_sub_sdfg.add_edge(state0, state1, dace.InterstateEdge(condition=f"{condition}"))
                    assign_sub_sdfg.add_edge(state0, state2, dace.InterstateEdge(condition=f"not {condition}"))
                    assign_sub_sdfg.add_edge(state1, state3, dace.InterstateEdge())
                    assign_sub_sdfg.add_edge(state2, state3, dace.InterstateEdge())

                    lassign : SDFGState = assign_inner_sdfg.add_state('assignment')
                    rassign : SDFGState = assign_remainder_sdfg.add_state('assignment')

                    assign_nsdfg  : nodes.NestedSDFG = state.add_nested_sdfg(assign_sub_sdfg, sdfg, ins, outs)

                    lnsdfg : nodes.NestedSDFG = state1.add_nested_sdfg(assign_inner_sdfg, assign_sub_sdfg, ins, outs, copy.deepcopy(assign_nsdfg.symbol_mapping))
                    rnsdfg : nodes.NestedSDFG = state2.add_nested_sdfg(assign_remainder_sdfg, assign_sub_sdfg, ins, outs, copy.deepcopy(assign_nsdfg.symbol_mapping))

                    i_u,i_uc,i_v,i_vc,imemlet = ie
                    o_u,o_uc,o_v,o_vc,omemlet = oe
                    state.remove_node(i_v)
                    #state.remove_edge(ie) # Remove node removes these edges?
                    #state.remove_edge(oe)
                    iarr_name = imemlet.data
                    iarr = sdfg.arrays[imemlet.data]
                    state.add_edge(i_u,i_uc,assign_nsdfg,imemlet.data,Memlet(subset=imemlet.subset, data=imemlet.data))
                    oarr_name = omemlet.data
                    oarr = sdfg.arrays[omemlet.data]
                    state.add_edge(assign_nsdfg,omemlet.data,o_v,o_vc,Memlet(subset=omemlet.subset, data=omemlet.data))

                    for s, sub_s in [(state1,lnsdfg), (state2,rnsdfg)]:
                        an_in = nodes.AccessNode(data=iarr_name)
                        an_out = nodes.AccessNode(data=oarr_name)
                        s.add_node(an_in)
                        s.add_node(an_out)
                        s.add_edge(an_in,None,sub_s,iarr_name,Memlet(subset=imemlet.subset, data=iarr_name))
                        s.add_edge(sub_s,oarr_name,an_out,None,Memlet(subset=omemlet.subset, data=oarr_name))

                    for sub_s in [lassign, rassign]:
                        an_in = nodes.AccessNode(data=iarr_name)
                        an_out = nodes.AccessNode(data=oarr_name)
                        sub_s.add_node(an_out)
                        sub_s.add_edge(an_in,None,an_out,None,
                                       Memlet(subset=omemlet.subset, data=oarr_name))

                    # Update ranges in the rassign
                    if len(rassign.edges()) != 1:
                        raise Exception("Assignment sub sdfg should have 1 edges")
                    e0 = rassign.edges()[0]
                    u,uc,v,vc,memlet = e0
                    if memlet.data == oarr_name:
                        new_assign_ranges = []
                        for (beg,end,step), dim in zip(memlet.subset, sdfg.arrays[memlet.data].shape):
                            ol = (o_end+1-o_beg)/o_step
                            new_range = (beg,beg+symbolic.SymExpr(f"Min({dim} - ({str(beg)}), {ol})-1"),1)
                            new_assign_ranges.append(new_range)
                        new_memlet = Memlet(subset=Range(new_assign_ranges), data=memlet.data)
                        rassign.remove_edge(e0)
                        rassign.add_edge(u,uc,v,vc,new_memlet)

        # S1.2 Update memlet subsets
        for kernel_parent, kernel, used_offsets in [(lkernel_parent_state, lkernel, offsets), (rkernel_parent_state, rkernel, offsets),
                                                    (lassign_parent_state, lassign, assign_offsets), (rassign_parent_state, rassign, assign_offsets)]:
            self.substract_offsets(kernel_parent, used_offsets)
            kernel_parent_offsets = self.create_offsets(kernel_parent.edges())
            self.substract_offsets(kernel, used_offsets)
            self.substract_offsets(kernel, kernel_parent_offsets)
        # We need to recursively update offets, the structure is as follows.
        # We have Main SDFG -> innerLoopKernel -> kernel
        #         Main SDFG -> remainderLoopKernel -> kernel
        # At ever edge one needs to calculate and upate the offsets

    def create_offsets(self, edges):
        offsets = dict()
        for edge in edges:
            _,_,_,_,memlet = edge
            data_offsets = [beg for (beg,end,step) in memlet.subset]
            if not memlet.data in offsets:
                offsets[memlet.data] = data_offsets
            else:
                if offsets[memlet.data] != data_offsets:
                    raise Exception("The transformations supports 1 offset per data container")
        return offsets

    def substract_offsets(self, state, offsets):
        edges_to_check = set()
        for n in dace.sdfg.utils.dfs_topological_sort(state):
            edges_to_check = edges_to_check.union(state.out_edges(n))
        for edge in edges_to_check:
            u,uc,v,vc,memlet = edge
            if memlet.data in offsets.keys():
                data_offsets = offsets[memlet.data]
                new_range = [(beg-data_offset,end-data_offset,step) for data_offset, (beg,end,step) in zip(data_offsets, memlet.subset)]
                new_memlet = Memlet(subset=Range(new_range), data=memlet.data)
                state.remove_edge(edge)
                state.add_edge(u,uc,v,vc,new_memlet)

    @staticmethod
    def annotates_memlets():
        return True
