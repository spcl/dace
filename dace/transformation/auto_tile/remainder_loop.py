# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from typing import List, Set, Union
from dace.data import Property
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import transformation
from dace import dtypes
from dace import symbolic
import dace
import copy
from dace.subsets import Range
from dace.sdfg.analysis.cutout import SDFGCutout


@make_properties
class RemainderLoop(transformation.SingleStateTransformation):
    inner_work_map_entry = transformation.PatternNode(nodes.MapEntry)
    tblock_type = Property(dtype=dtypes.ScheduleType, default=dtypes.ScheduleType.GPU_ThreadBlock, allow_none=False)
    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.inner_work_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def update_names():
        pass

    def has_remainder(self, numerator, denominator):
        if (isinstance(numerator, int) or numerator.is_integer) and \
                (isinstance(denominator, int) or denominator.is_integer):
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
        sdfg.save("rb.sdfg")

        inner_work_map_entry = self.inner_work_map_entry
        map_entry = self.inner_work_map_entry
        dev_entry = None
        while map_entry:
            dev_entry = map_entry
            map_entry = state.entry_node(map_entry)
        #assert (dev_entry.map.schedule == dtypes.ScheduleType.GPU_Device)

        # 0. Sort memory-moved array into groups
        access_groups = list()
        for n in dace.sdfg.utils.dfs_topological_sort(state):
            if isinstance(n, nodes.LibraryNode):
                ies = state.in_edges(n)
                oes = state.out_edges(n)
                if len(ies) == 1 and len(oes) == 1:
                    ie = ies[0]
                    oe = oes[0]
                    iu, iuc, iv, ivc, imemlet = ie
                    ou, ouc, ov, ovc, omemlet = oe
                    if sdfg.arrays[imemlet.data].storage != sdfg.arrays[omemlet.data].storage:
                        added = False
                        for i, group in enumerate(access_groups):
                            if imemlet.data in group or omemlet.data in group:
                                added = True
                                updated_group = group.union(
                                    [(imemlet.data, imemlet.subset), (omemlet.data, omemlet.subset)])
                                access_groups[i] = updated_group
                        if not added:
                            access_groups.append(
                                set([(imemlet.data, imemlet.subset), (omemlet.data, omemlet.subset)]))

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

        # 2.
        param_and_ranges = dict()
        cur_map_entry = self.inner_work_map_entry
        while cur_map_entry:
            for param, (beg, end, step) in zip(cur_map_entry.map.params, cur_map_entry.map.range):
                assert (not param in param_and_ranges.keys())
                param_and_ranges[param] = (beg, end, step)
            cur_map_entry = state.entry_node(cur_map_entry)

        # 3.
        # Let us look at B[k + tk, b_j + d_j + j]
        # k=0:K:16, tk=0:16:1, b_j=0:N:64, d_j=0:64:4, j=0:4:1
        # k + tk -> K % 16 -> can overflow
        # b_j + d_j + j -> N % 16 -> can overflow (the other params can't overflow)
        can_out_of_bound = dict()
        symbols_to_ensure_in_scope = set()
        for (data, subset) in data_accesses:
            for (beg, end, step) in subset:
                free_symbols = set.union(
                    beg.free_symbols, end.free_symbols, step.free_symbols)
                for symbol in free_symbols:
                    if str(symbol) in param_and_ranges.keys():
                        (beg, end, step) = param_and_ranges[str(symbol)]
                        if self.has_remainder(end+1-beg, step):
                            can_out_of_bound[(data, subset)] = True
                            symbols_to_ensure_in_scope.add(str(symbol))
                if not (data, subset) in can_out_of_bound.keys():
                    can_out_of_bound[(data, subset)] = False
        # If one element in a access group can out of bound then all such elements need to be set to true

        for (data, subset), b in can_out_of_bound.items():
            for group in access_groups:
                group_can_out_of_bound = True
                for (_d, _s) in group:
                    if (_d, _s) in can_out_of_bound.keys() and can_out_of_bound[(_d, _s)]:
                        group_can_out_of_bound = True
                        break

                if group_can_out_of_bound:
                    for (_d, _s) in group:
                        if (data, subset) == (_d, _s):
                            can_out_of_bound[(data, subset)] = True
                            for (beg, end, step) in subset:
                                free_symbols = set.union(
                                    beg.free_symbols, end.free_symbols, step.free_symbols)
                                for symbol in free_symbols:
                                    symbols_to_ensure_in_scope.add(str(symbol))

        thread_block_map_entry = [v for v in state.all_nodes_between(dev_entry, state.exit_node(dev_entry)) if isinstance(
            v, nodes.MapEntry) and v.schedule == self.tblock_type]
        assert(len(thread_block_map_entry) == 1)

        for param in thread_block_map_entry[0].map.params:
            symbols_to_ensure_in_scope.add(param)

        # 5. Go up until all the variables are defined (remove the vars as iterating the scopes ap)
        map_before_split = None
        cur_map_entry = self.inner_work_map_entry

        b = False
        while cur_map_entry:
            if b:
                break
            for param in cur_map_entry.map.params:
                if b:
                    break
                if param in symbols_to_ensure_in_scope:
                    map_before_split = cur_map_entry
                    b = True
                    break
            cur_map_entry = state.entry_node(cur_map_entry)

        # raise Exception(symbols_to_ensure_in_scope, can_out_of_bound)

        map_after_split = state.exit_node(map_before_split)

        ncheck = [v for (u, uc, v, vc, m) in state.out_edges(dev_entry)]
        loop_vars_and_ranges = dict()
        while ncheck:
            n = ncheck.pop()
            if isinstance(n, nodes.MapEntry):
                for param, range in zip(n.map.params, n.map.range):
                    beg, end, step = range
                    if n == map_before_split:
                        map_len = step
                    elif n.map.schedule == self.tblock_type:
                        map_len = step
                    else:
                        map_len = ((end+1)-beg)
                    if (map_len.is_integer):
                        # raise Exception("For RemainderLoop transformation to work inner maps need to have static ranges")
                        loop_vars_and_ranges[param] = map_len
            if n != state.exit_node(dev_entry):
                ncheck += [v for (u, uc, v, vc, m) in state.out_edges(n)]

        # 4. Set up the condition
        added_conditions = set()
        conditions_and_ranges = dict()

        # TODO: improve (should go through all state)
        if True:  # len(conditions_and_ranges) == 0:
            for n in sdutil.dfs_topological_sort(state, sources=dev_entry):
                if isinstance(n, nodes.MapEntry) and n.map.label.startswith("ThreadCoarsened"):
                    for param, (beg, end, step) in zip(n.map.params, n.map.range):
                        l = (end+1-beg)/step
                        dev_param = f"b_{param}"
                        block_param = f"d_{param}"
                        for dp, (db, de, ds) in zip(dev_entry.map.params, dev_entry.map.range):
                            if dp == dev_param:
                                lim = de+1
                                conditions_and_ranges[param] = (
                                    [block_param, dev_param], de+1, l)
                                added_conditions.add(
                                    f"{block_param} + {dev_param} <= {de+1} - {l}")
                                break
            for n in state.nodes():
                if isinstance(n, nodes.MapEntry) and n.map.label.startswith("InnerWorkMap"):
                    for param, (beg, end, step) in zip(n.map.params, n.map.range):
                        l = (end+1-beg)/step
                        outer_work_map_param = param[1:]
                        block_param = f"d_{param}"
                        for nn in state.nodes():
                            if isinstance(nn, nodes.MapEntry) and nn.map.label.startswith("OuterWorkMap"):
                                for op, (ob, oe, os) in zip(nn.map.params, nn.map.range):
                                    if op == outer_work_map_param:
                                        conditions_and_ranges[param] = (
                                            [outer_work_map_param], oe+1, l)
                                        added_conditions.add(
                                            f"{outer_work_map_param} <= {oe+1} - {l}")
                                        break

        condition = "(" + " and ".join(added_conditions) + ")"
        if condition == "()":
            raise Exception(condition)

        # For example range for k is 0:K:16,
        # In this last iteration the inner tk goes from 0:16
        # In the remainder loop it should go from 0:16 to 0:Min(K-(Sum all previous var),16)

        # 6. Create if condition
        inner_kernel_entry: nodes.MapEntry = map_before_split
        inner_kernel_exit: nodes.MapExit = state.exit_node(inner_kernel_entry)
        #raise Exception(inner_kernel_entry, added_conditions,
        #                symbols_to_ensure_in_scope)

        sub_sdfg: SDFG = dace.SDFG('nested_subsdfg', parent=sdfg)
        inner_loop_kernel_sdfg: SDFG = dace.SDFG(
            'inner_loop_sub_sdfg', parent=sub_sdfg)
        remainder_loop_kernel_sdfg: SDFG = dace.SDFG(
            'outer_loop_sub_sdfg', parent=sub_sdfg)

        # Declare arrays for nested SDFG (the names can differ from the top-level SDFG)
        # In this case, we read only one element out of the full arrays

        # Inputs for NestedSDFG
        # Create nested states
        first_node_in_microkernel = None
        for v in state.bfs_nodes(map_before_split):
            if isinstance(v, nodes.MapEntry) and v != map_before_split:
                first_node_in_microkernel = v
                break
        # if not first_node_in_microkernel:
        #    raise Exception(
        #        "At least on edge needs to directly connect the map to be split before and the next map")
        last_node_in_microkernel = state.exit_node(first_node_in_microkernel)

        # I think this part is super fragile
        # It basically maps an shrA[d_i + i,...] access to the limits of the varaible d_i, as we cant catch the limits of i from the implementation
        # TODO find a mapping where the range limit for d_i also holds for i.
        # Since my transformations uses d_ as the thread coarsened prefix I can just apply that
        adders = []
        for (var, (loop_vars, shape, lim)) in conditions_and_ranges.items():
            if var.startswith("d_"):
                adders.append((var[2:], (loop_vars, shape, lim)))
        for k, v in adders:
            conditions_and_ranges[k] = v

        new_ranges = dict()
        for (var, (loop_vars, shape, lim)) in conditions_and_ranges.items():
            new_ranges[var] = \
                ((0, symbolic.SymExpr(f"Min({shape} - ({'+'.join([str(s) for s in loop_vars])}), {lim})-1"), 1),
                 (0, lim-1, 1))

        ins = set()
        for in_edge in state.in_edges(first_node_in_microkernel):
            _, _, _, _, memlet = in_edge
            if memlet.data is not None:
                ins.add(memlet.data)
        # Outputs for NestedSDFG
        outs = set()
        for out_edge in state.out_edges(last_node_in_microkernel):
            _, _, _, _, memlet = out_edge
            if memlet.data is not None:
                outs.add(memlet.data)
        assert (not (None in ins))
        assert (not (None in outs))

        for (_sdfg) in [
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
        offsets = self.create_offsets(state.in_edges(
            first_node_in_microkernel) + state.out_edges(last_node_in_microkernel))

        # Save the in and out edges of the map scope encompassing the inner kernels
        # This is necessary to creaet edges from and to access nodes within the sub kernel
        special_in_edges = []
        special_out_edges = []
        for in_edge in state.in_edges(first_node_in_microkernel):
            special_in_edges.append(in_edge)
        for out_edge in state.out_edges(last_node_in_microkernel):
            special_out_edges.append(out_edge)

        state0 = sub_sdfg.add_state('if_guard_state')
        state1 = sub_sdfg.add_state('inner_loop_kernel_state')
        lkernel_parent_state = state1
        state2 = sub_sdfg.add_state('remainder_loop_kernel_state')
        rkernel_parent_state = state2
        state3 = sub_sdfg.add_state('completed_state')

        sub_sdfg.add_edge(state0, state1, dace.InterstateEdge(
            condition=f"{condition}"))
        sub_sdfg.add_edge(state0, state2, dace.InterstateEdge(
            condition=f"not {condition}"))
        sub_sdfg.add_edge(state1, state3, dace.InterstateEdge())
        sub_sdfg.add_edge(state2, state3, dace.InterstateEdge())

        lkernel: SDFGState = inner_loop_kernel_sdfg.add_state('lkernel_state')
        rkernel: SDFGState = remainder_loop_kernel_sdfg.add_state(
            'rkernel_state')

        symmap = dict()
        _, _, v, _, _ = state.out_edges(map_before_split)[0]
        for name, typeclass in state.symbols_defined_at(v).items():
            symmap[name] = symbolic.symbol(name, typeclass)
        d = state.symbols_defined_at(v)

        symmap_ns = dict()
        _, _, v, _, _ = state.out_edges(map_before_split)[0]
        for name, typeclass in state.symbols_defined_at(v).items():
            symmap_ns[name] = symbolic.symbol(name, typeclass)
            # raise Exception(symmap, ins, outs, sdfg.symbols, sub_sdfg.symbols, d)

        nsdfg: nodes.NestedSDFG = state.add_nested_sdfg(
            sub_sdfg, sdfg, ins, outs, symbol_type_mapping=copy.deepcopy(d))

        # raise Exception(syms_defiend)
        # raise Exception(nsdfg.symbol_mapping, state.symbols_defined_at(map_before_split), map_before_split)

        lnsdfg: nodes.NestedSDFG = state1.add_nested_sdfg(
            inner_loop_kernel_sdfg, sub_sdfg, ins, outs, nsdfg.symbol_mapping, symbol_type_mapping=copy.deepcopy(d))
        rnsdfg: nodes.NestedSDFG = state2.add_nested_sdfg(
            remainder_loop_kernel_sdfg, sub_sdfg, ins, outs, nsdfg.symbol_mapping, symbol_type_mapping=copy.deepcopy(d))

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
                u, u_conn, v, v_conn, memlet = edge
                _state.add_edge(an, None, node, in_arr, Memlet(
                    data=in_arr, subset=Range(memlet.subset)))
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
                u, u_conn, v, v_conn, memlet = edge
                _state.add_edge(node, out_arr, an, None, copy.deepcopy(memlet))

        # Edges that go to and come out nested sdfg
        for in_edge in state.in_edges(first_node_in_microkernel):
            u, u_conn, v, v_conn, memlet = in_edge
            if memlet is None:
                state.add_edge(u, u_conn, nsdfg, None, None)
            else:
                state.add_edge(u, u_conn, nsdfg, memlet.data,
                               copy.deepcopy(memlet))

        for out_edge in state.out_edges(last_node_in_microkernel):
            u, u_conn, v, v_conn, memlet = out_edge
            if memlet is None:
                state.add_edge(u, u_conn, nsdfg, None, None)
            else:
                state.add_edge(nsdfg, memlet.data, v,
                               v_conn, copy.deepcopy(memlet))

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

                    edges_to_copyover = edges_to_copyover.union(set([e  # Unpacking causes problems in removing
                                                                     for e, (u, uc, v, vc, m) in
                                                                     zip(state.out_edges(
                                                                         n), state.out_edges(n))
                                                                     if v != nsdfg and u != last_node_in_microkernel]))
                if n != last_node_in_microkernel:
                    ncheck += [v for (u, uc, v, vc, m) in state.out_edges(n)]

        for e in edges_to_copyover:
            state.remove_edge(e)
        for n in nodes_to_copyover:
            state.remove_node(n)

        for kernel, kernel_sdfg in [(lkernel, inner_loop_kernel_sdfg), (rkernel, remainder_loop_kernel_sdfg)]:
            added_labels = dict()

            for n in nodes_to_copyover:
                nn: nodes.Node = copy.deepcopy(n)
                kernel.add_node(nn)
                added_labels[n.guid] = nn
            for e in edges_to_copyover:
                u, uc, v, vc, memlet = e
                uu = added_labels[u.guid]
                vv = added_labels[v.guid]
                kernel.add_edge(uu, uc, vv, vc, copy.deepcopy(memlet))

            first_node = next(dace.sdfg.utils.dfs_topological_sort(kernel))
            if not isinstance(first_node, nodes.MapEntry):
                raise Exception(
                    f"First node in the map currently needs to be a MapEntry it is: {type(first_node)}")
            last_node = kernel.exit_node(first_node)

            for e in special_in_edges:
                u, uc, v, vc, memlet = e
                new_memlet = copy.deepcopy(memlet)
                if memlet.data is not None:
                    an = nodes.AccessNode(data=memlet.data)
                    kernel.add_node(an)
                    kernel.add_edge(an, None, first_node, vc, new_memlet)
                # else:
                #    kernel.add_edge(u, None, first_node, vc, new_memlet)
            for e in special_out_edges:
                u, uc, v, vc, memlet = e
                new_memlet = copy.deepcopy(memlet)
                if memlet.data is not None:
                    an = nodes.AccessNode(data=memlet.data)
                    kernel.add_node(an)
                    kernel.add_edge(last_node, uc, an, None, new_memlet)

            # Re-allocate transients that are first accessed in the inner kernel
            for n in dace.sdfg.utils.dfs_topological_sort(kernel):
                if isinstance(n, nodes.AccessNode):
                    if not n.data in kernel_sdfg.arrays.keys():
                        arr = sdfg.arrays[n.data]
                        assert (arr.transient)
                        if isinstance(arr, dace.data.Array):
                            kernel_sdfg.add_array(
                                name=n.data,
                                shape=arr.shape,
                                stride=arr.stride,
                                transient=True,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                lifetime=arr.lifetime
                            )
                        else:
                            assert (isinstance(arr, dace.data.Scalar))
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
                    if p in new_ranges.keys():
                        new_range, old_range = new_ranges[p]
                        rb, re, rs = r
                        orb, ore, ors = old_range
                        # if (rb != orb or re != ore or rs != ors):
                        #    raise Exception(
                        #        f"For the transformation to work any map range needs to have the format 0:N:1, had: {r} and {old_range}")
                        nr.append(new_range)
                    else:
                        nr.append(r)
                n.map.range = Range(nr)

        # Add missing arrays (e.g. transient scalars only accessed on some edges)
        for kernel_sdfg in [inner_loop_kernel_sdfg, remainder_loop_kernel_sdfg]:
            self.add_missing_arrays(sdfg, kernel_sdfg, None)

        inner_loop_kernel_sdfg.validate()
        remainder_loop_kernel_sdfg.validate()

        counter = 0
        for n in state.bfs_nodes(dev_entry):
            if n == state.exit_node(dev_entry):
                break
            if isinstance(n, nodes.AccessNode):
                can_out_of_bound = False
                i_memlet = None
                o_memlet = None
                if (len(state.in_edges(n)) > 1 or len(state.out_edges(n)) > 1):
                    raise Exception(
                        "Transformation can be applied if all assignment edges have at max in and out degree 1 each")
                if (len(state.in_edges(n)) == 0 or len(state.out_edges(n)) == 0):
                    continue
                for ie, oe in zip(state.in_edges(n), state.out_edges(n)):
                    _, _, _, _, i_memlet = ie
                    _, _, _, _, o_memlet = oe
                    if i_memlet.data == None or o_memlet.data == None:
                        continue
                    for ((i_beg, i_end, i_step), (o_beg, o_end, o_step), i_dim, o_dim) in zip(i_memlet.subset, o_memlet.subset, sdfg.arrays[i_memlet.data].shape, sdfg.arrays[o_memlet.data].shape):
                        il = (i_end+1-i_beg)/i_step
                        ol = (o_end+1-o_beg)/o_step
                        if (self.has_remainder(i_dim, il) or self.has_remainder(o_dim, ol)) and \
                                sdfg.arrays[i_memlet.data].transient and not sdfg.arrays[o_memlet.data].transient:
                            can_out_of_bound = True
                            break
                if can_out_of_bound:
                    counter += 1
                    assign_sub_sdfg: SDFG = dace.SDFG(
                        f'nestedAssignment{counter}', parent=sdfg)
                    assign_inner_sdfg: SDFG = dace.SDFG(
                        f'innerAssignment{counter}', parent=sub_sdfg)
                    assign_remainder_sdfg: SDFG = dace.SDFG(
                        f'remainderAssignment{counter}', parent=sub_sdfg)

                    # To split
                    # Inputs for NestedSDFG
                    ins = set([i_memlet.data])
                    # Outputs for NestedSDFG
                    outs = set([o_memlet.data])

                    for (_sdfg) in [
                        (assign_sub_sdfg),
                        (assign_inner_sdfg),
                        (assign_remainder_sdfg)
                    ]:
                        for in_arr in set.union(ins, outs):
                            _sdfg.add_array(name=in_arr,
                                            shape=sdfg.arrays[in_arr].shape,
                                            strides=sdfg.arrays[in_arr].strides,
                                            transient=False,
                                            dtype=sdfg.arrays[in_arr].dtype)

                    # Create nested states

                    # Save the in and out edges of the map scope encompassing the inner kernels
                    # This is necessary to creaet edges from and to access nodes within the sub kernel
                    special_in_edges = [ie]
                    special_out_edges = [oe]

                    assign_offsets = self.create_offsets([ie, oe])

                    state0 = assign_sub_sdfg.add_state('if_guard_state_state')
                    state1 = assign_sub_sdfg.add_state(
                        'inner_assignment_state')
                    lassign_parent_state = state1
                    state2 = assign_sub_sdfg.add_state(
                        'remainder_assignment_state')
                    rassign_parent_state = state2
                    state3 = assign_sub_sdfg.add_state('completed_state')

                    conditions = []
                    for (beg, end, step), dim in zip(o_memlet.subset, sdfg.arrays[o_memlet.data].shape):
                        l = (end+1-beg)/step
                        cond = f"{str(beg)} + {str(l)} <= {dim}"
                        conditions.append(cond)
                    condition = "(" + " and ".join(conditions) + ")"

                    assign_sub_sdfg.add_edge(
                        state0, state1, dace.InterstateEdge(condition=f"{condition}"))
                    assign_sub_sdfg.add_edge(
                        state0, state2, dace.InterstateEdge(condition=f"not {condition}"))
                    assign_sub_sdfg.add_edge(
                        state1, state3, dace.InterstateEdge())
                    assign_sub_sdfg.add_edge(
                        state2, state3, dace.InterstateEdge())

                    lassign: SDFGState = assign_inner_sdfg.add_state(
                        'lassignment')
                    rassign: SDFGState = assign_remainder_sdfg.add_state(
                        'rassignment')

                    d = dict()
                    for sym, t in state.symbols_defined_at(n).items():
                        d[sym] = t

                    assign_nsdfg: nodes.NestedSDFG = state.add_nested_sdfg(
                        assign_sub_sdfg, sdfg, ins, outs, symbol_type_mapping=d)

                    lnsdfg: nodes.NestedSDFG = state1.add_nested_sdfg(
                        assign_inner_sdfg, assign_sub_sdfg, ins, outs, copy.deepcopy(assign_nsdfg.symbol_mapping), symbol_type_mapping=d)
                    rnsdfg: nodes.NestedSDFG = state2.add_nested_sdfg(
                        assign_remainder_sdfg, assign_sub_sdfg, ins, outs, copy.deepcopy(assign_nsdfg.symbol_mapping), symbol_type_mapping=d)

                    i_u, i_uc, i_v, i_vc, imemlet = ie
                    o_u, o_uc, o_v, o_vc, omemlet = oe
                    state.remove_node(i_v)
                    # state.remove_edge(ie) # Remove node removes these edges?
                    # state.remove_edge(oe)
                    iarr_name = imemlet.data
                    iarr = sdfg.arrays[imemlet.data]
                    aan = nodes.AccessNode(data=imemlet.data)
                    state.add_node(aan)
                    state.add_edge(i_u, i_uc, aan, None, Memlet(
                        subset=imemlet.subset, data=imemlet.data))
                    state.add_edge(aan, None, assign_nsdfg, imemlet.data,  Memlet(
                        subset=imemlet.subset, data=imemlet.data, wcr=imemlet.wcr, wcr_nonatomic=imemlet.wcr_nonatomic, allow_oob=imemlet.allow_oob, debuginfo=imemlet.debuginfo))
                    oarr_name = omemlet.data
                    oarr = sdfg.arrays[omemlet.data]
                    state.add_edge(assign_nsdfg, omemlet.data, o_v, o_vc, Memlet(
                        subset=omemlet.subset, data=omemlet.data, wcr=omemlet.wcr, wcr_nonatomic=omemlet.wcr_nonatomic, allow_oob=omemlet.allow_oob, debuginfo=omemlet.debuginfo))

                    for s, sub_s in [(state1, lnsdfg), (state2, rnsdfg)]:
                        an_in = nodes.AccessNode(data=iarr_name)
                        an_out = nodes.AccessNode(data=oarr_name)
                        s.add_node(an_in)
                        s.add_node(an_out)
                        s.add_edge(an_in, None, sub_s, iarr_name, Memlet(
                            subset=imemlet.subset, data=iarr_name, wcr=imemlet.wcr, wcr_nonatomic=imemlet.wcr_nonatomic, allow_oob=imemlet.allow_oob, debuginfo=imemlet.debuginfo))
                        s.add_edge(sub_s, oarr_name, an_out, None, Memlet(
                            subset=omemlet.subset, data=oarr_name, wcr=omemlet.wcr, wcr_nonatomic=omemlet.wcr_nonatomic, allow_oob=omemlet.allow_oob, debuginfo=omemlet.debuginfo))

                    for sub_s in [lassign, rassign]:
                        an_in = nodes.AccessNode(data=iarr_name)
                        an_out = nodes.AccessNode(data=oarr_name)
                        sub_s.add_node(an_out)
                        sub_s.add_edge(an_in, None, an_out, None,
                                       Memlet(subset=omemlet.subset, data=oarr_name, wcr=omemlet.wcr, wcr_nonatomic=omemlet.wcr_nonatomic, allow_oob=omemlet.allow_oob, debuginfo=omemlet.debuginfo))

                    # Update ranges in the rassign
                    if len(rassign.edges()) != 1:
                        raise Exception(
                            "Assignment sub sdfg should have 1 edges")
                    e0 = rassign.edges()[0]
                    u, uc, v, vc, memlet = e0
                    if memlet.data == oarr_name:
                        new_assign_ranges = []
                        for (beg, end, step), dim in zip(memlet.subset, sdfg.arrays[memlet.data].shape):
                            l = (end+1-beg)/step
                            new_range = (
                                beg, beg+symbolic.SymExpr(f"Min({dim} - ({str(beg)}), {l})-1"), 1)
                            new_assign_ranges.append(new_range)
                        new_memlet = Memlet(subset=Range(
                            new_assign_ranges), data=memlet.data, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
                        rassign.remove_edge(e0)
                        rassign.add_edge(u, uc, v, vc, new_memlet)

                    for kernel_parent, kernel, used_offsets in [(lassign_parent_state, lassign, assign_offsets), (rassign_parent_state, rassign, assign_offsets)]:
                        self.substract_offsets(kernel_parent, used_offsets)
                        kernel_parent_offsets = self.create_offsets(
                            kernel_parent.edges())
                        self.substract_offsets(kernel, used_offsets)
                        self.substract_offsets(kernel, kernel_parent_offsets)

        # S1.2 Update memlet subsets
        for kernel_parent, kernel, used_offsets in [(lkernel_parent_state, lkernel, offsets), (rkernel_parent_state, rkernel, offsets)]:
            if kernel_parent and kernel and used_offsets:
                self.substract_offsets(kernel_parent, used_offsets)
                kernel_parent_offsets = self.create_offsets(
                    kernel_parent.edges())
                self.substract_offsets(kernel, used_offsets)
                self.substract_offsets(kernel, kernel_parent_offsets)
            # Add the edges that are still missing

        self.prune_unused_data(sdfg)
        self.prune_unused_data(sub_sdfg)

        # We need to recursively update offets, the structure is as follows.
        # We have Main SDFG -> innerLoopKernel -> kernel
        #         Main SDFG -> remainderLoopKernel -> kernel
        # At ever edge one needs to calculate and upate the offsets

        # 7
        # Get all loop params, any loop param inside a nested SDFG becomes an int and not long long
        params = set()
        for n in dace.sdfg.utils.dfs_topological_sort(state):
            if isinstance(n, nodes.MapEntry):
                params = params.union(n.map.params)

        sdfg.save("rl.sdfg")

    def create_offsets(self, edges):
        offsets = dict()
        for edge in edges:
            _, _, _, _, memlet = edge
            if memlet is not None and memlet.subset is not None:
                data_offsets = [beg for (beg, end, step) in memlet.subset]
                if not memlet.data in offsets:
                    offsets[memlet.data] = data_offsets
                else:
                    if offsets[memlet.data] != data_offsets:
                        raise Exception(
                            "The transformations supports 1 offset per data container")
        return offsets

    def substract_offsets(self, state, offsets):
        edges_to_check = set()
        for n in dace.sdfg.utils.dfs_topological_sort(state):
            edges_to_check = edges_to_check.union(state.out_edges(n))
        for edge in edges_to_check:
            u, uc, v, vc, memlet = edge
            if memlet.data in offsets.keys():
                data_offsets = offsets[memlet.data]
                new_range = [(beg-data_offset, end-data_offset, step)
                             for data_offset, (beg, end, step) in zip(data_offsets, memlet.subset)]
                new_memlet = Memlet(subset=Range(new_range), data=memlet.data, wcr=memlet.wcr,
                                    wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
                state.remove_edge(edge)
                state.add_edge(u, uc, v, vc, new_memlet)

    def add_arr(self, parent_sdfg: SDFG, sub_sdfg: SDFG, arr_name: str, convert_to_transient: bool = True):
        if not arr_name in sub_sdfg.arrays:
            arr = parent_sdfg.arrays[arr_name]
            if isinstance(arr, dace.data.Array):
                sub_sdfg.add_array(
                    name=arr_name,
                    shape=arr.shape,
                    transient=convert_to_transient or arr.transient,
                    strides=arr.shape,
                    dtype=arr.dtype,
                    storage=arr.storage,
                    lifetime=arr.lifetime
                )
            else:
                sub_sdfg.add_scalar(
                    name=arr_name,
                    transient=convert_to_transient or arr.transient,
                    dtype=arr.dtype,
                    storage=arr.storage,
                    lifetime=arr.lifetime
                )

    def add_missing_arrays(self, parent_sdfg: SDFG, sub_sdfg: SDFG, state):
        arrays_to_remove_from_parent = set()
        for s in sub_sdfg.states() if state == None else [state]:
            for n in s.nodes():
                if isinstance(n, nodes.AccessNode):
                    arr_name = n.data
                    if not arr_name in sub_sdfg.arrays:
                        self.add_arr(parent_sdfg, sub_sdfg, arr_name)
                        arrays_to_remove_from_parent.add(arr_name)
                for e in s.out_edges(n):
                    _, _, _, _, memlet = e
                    arr_name = memlet.data
                    if arr_name is not None:
                        if not arr_name in sub_sdfg.arrays:
                            self.add_arr(parent_sdfg, sub_sdfg, arr_name)
                            arrays_to_remove_from_parent.add(arr_name)

        return arrays_to_remove_from_parent

    def rm_arrays(self, sdfg: SDFG, arr_names: Union[List[str], Set[str]]):
        for arr_name in arr_names:
            if arr_name in sdfg.arrays:
                sdfg.arrays.pop(arr_name)

    def prune_unused_data(self, sdfg: SDFG):
        used_arrays = set()
        for state in sdfg.states():
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode):
                    arr_name = n.data
                    if arr_name in sdfg.arrays:
                        used_arrays.add(arr_name)
                for e in state.out_edges(n):
                    _, _, _, _, memlet = e
                    arr_name = memlet.data
                    if arr_name in sdfg.arrays:
                        used_arrays.add(arr_name)
        all_arrays = set(sdfg.arrays.keys())
        unused_arrays = set.difference(all_arrays, used_arrays)

        for unused_arr_name in unused_arrays:
            sdfg.arrays.pop(unused_arr_name)

        return unused_arrays

    @staticmethod
    def annotates_memlets():
        return True
