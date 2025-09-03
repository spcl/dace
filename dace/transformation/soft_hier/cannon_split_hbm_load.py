# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy

from dace import data, sdfg as sd, subsets, symbolic, SDFGState, Memlet, dtypes, InterstateEdge
from dace.sdfg import nodes, graph as GF
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, Property, SymbolicProperty, CodeBlock
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion


@make_properties
class SplitHBMLoad(transformation.SingleStateTransformation):
    """ Implements the double buffering pattern, which pipelines reading
        and processing data by creating a second copy of the memory.
        In particular, the transformation takes a 1D map and all internal
        (directly connected) transients, adds an additional dimension of size 2,
        and turns the map into a for loop that processes and reads the data in a
        double-buffered manner. Other memlets will not be transformed.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    schedule_nsdfg_node = transformation.PatternNode(nodes.NestedSDFG)

    # Properties
    npe = Property(default=None, allow_none=True, desc="Number of processing elements")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.schedule_nsdfg_node)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        # Only one dimensional maps are allowed
        if len(map_entry.map.params) != 1:
            return False

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        NPE = self.npe
        gi = self.gi
        gj = self.gj

        schedule_nsdfg_node = self.schedule_nsdfg_node
        schedule_nsdfg = schedule_nsdfg_node.sdfg

        map_param = map_entry.map.params[0]  # Assuming one dimensional

        schedule_nsdfg.add_symbol(map_param, stype=(symbolic.pystr_to_symbolic(map_param)).dtype)
        schedule_nsdfg_node.symbol_mapping.update({f"{map_param}" : symbolic.pystr_to_symbolic(map_param)})
        
        ##############################
        # Change condition of loop to one fewer iteration (so that the
        # final one reads from the last buffer)
        map_rstart, map_rend, map_rstride = map_entry.map.range[0]
        tK = map_rstride // NPE
        # map_rend = symbolic.pystr_to_symbolic('(%s) - (%s)' % (map_rend, map_rstride))
        # map_entry.map.range = subsets.Range([(map_rstart, map_rend, map_rstride)])

        ##############################
        # collect all streams and transients inside the nested SDFG
        local_streams = set()
        for node, parent in schedule_nsdfg.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode):
                node_desc = node.desc(parent)
                if isinstance(node_desc, data.Stream) and node.data not in local_streams:
                    local_streams.add(node.data)
                    
        print(local_streams)

        transients_to_modify = {}
        global_arrays = {}
        for edge, parent in schedule_nsdfg.all_edges_recursive():
            src_node = edge.src
            dst_node = edge.dst
            if isinstance(edge.src, nodes.AccessNode) and edge.src.data in local_streams:
                if isinstance(edge.dst, nodes.AccessNode):
                    dst_node_desc = edge.dst.desc(parent)
                    if dst_node_desc.transient is True:
                        transients_to_modify[edge.src.data] = edge.dst.data
            elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data in local_streams:
                if isinstance(edge.src, nodes.AccessNode):
                    src_node_desc = edge.src.desc(parent)
                    if src_node_desc.transient is False:
                        global_arrays[edge.dst.data] = edge.src.data
        
        print(transients_to_modify)
        print(global_arrays)
        
        # Modify dimension to transients and modify memlets
        for transient in transients_to_modify.values():
            desc: data.Array = schedule_nsdfg.arrays[transient]
            # Using non-python syntax to ensure properties change
            desc.shape = [4] + list(desc.shape[1:])
            desc.total_size = (desc.total_size // 2) * 3
            
        for stream in local_streams:
            desc: data.Stream = schedule_nsdfg.arrays[stream]
            # Using non-python syntax to ensure properties change
            desc.shape = list(desc.shape[:2]) +[4] + list(desc.shape[3:])


        ##############################
        # Modify memlets to use map parameter as buffer index
        modified_subsets = []  # Store modified memlets for final state
        for edge, parent in schedule_nsdfg.all_edges_recursive():
            if isinstance(edge, GF.MultiConnectorEdge):
                if edge.data.data in transients_to_modify.values():
                    edge.data.subset = self._modify_memlet(schedule_nsdfg, edge.data.subset, edge.data.data, param_name='__db_local_array_param')
                    modified_subsets.append(edge.data.subset)
                else:  # Could be other_subset
                    path = parent.memlet_path(edge)
                    src_node = path[0].src
                    dst_node = path[-1].dst

                    # other_subset could be None. In that case, recreate from array
                    dataname = None
                    if (isinstance(src_node, nodes.AccessNode) and src_node.data in transients_to_modify.values()):
                        dataname = src_node.data
                    elif (isinstance(dst_node, nodes.AccessNode) and dst_node.data in transients_to_modify.values()):
                        dataname = dst_node.data
                    if dataname is not None:
                        subset = (edge.data.other_subset or subsets.Range.from_array(schedule_nsdfg.arrays[dataname]))
                        edge.data.other_subset = self._modify_memlet(schedule_nsdfg, subset, dataname, param_name='__db_local_array_param')
                        modified_subsets.append(edge.data.other_subset)
                        
                if edge.data.data in local_streams:
                    edge.data.subset = self._modify_memlet(schedule_nsdfg, edge.data.subset, edge.data.data, param_name='__db_local_stream_param')
                    modified_subsets.append(edge.data.subset)
                else:  # Could be other_subset
                    path = parent.memlet_path(edge)
                    src_node = path[0].src
                    dst_node = path[-1].dst

                    # other_subset could be None. In that case, recreate from array
                    dataname = None
                    if (isinstance(src_node, nodes.AccessNode) and src_node.data in local_streams):
                        dataname = src_node.data
                    elif (isinstance(dst_node, nodes.AccessNode) and dst_node.data in local_streams):
                        dataname = dst_node.data
                    if dataname is not None:
                        subset = edge.data.other_subset
                        edge.data.other_subset = self._modify_memlet(schedule_nsdfg, subset, dataname, param_name='__db_local_stream_param')
                        modified_subsets.append(edge.data.other_subset)



        ##############################
        # Turn map into for loop
        from dace.sdfg import SDFG, SDFGState
        map_to_for = MapToForLoop()
        map_to_for.setup_match(sdfg, self.cfg_id, self.state_id,
                               {MapToForLoop.map_entry: graph.node_id(self.map_entry)}, self.expr_index)
        nsdfg_node, nstate = map_to_for.apply(graph, sdfg)
        nsdfg: SDFG = nsdfg_node.sdfg

        ##############################
        for array in nsdfg.arrays.keys():
            if array in global_arrays.values():
                # copy the shape of the global arrays into the nested state
                desc_nsdfg: data.Array = nsdfg.arrays[array]
                desc_graph: data.Array = sdfg.arrays[array]
                desc_nsdfg.shape = copy.deepcopy(desc_graph.shape)

        nsdfg_node.symbol_mapping.update({"gi": gi, "gj": gj})
        for stream in local_streams:
            st_desc = copy.deepcopy(schedule_nsdfg.arrays[stream])
            nsdfg.add_stream(name=stream, dtype=st_desc.dtype, shape=st_desc.shape, buffer_size=st_desc.buffer_size, transient=st_desc.transient, storage=st_desc.storage)
            transient_array = transients_to_modify[stream]
            schedule_nsdfg_arr_desc = schedule_nsdfg.arrays[transient_array]

            arr_desc = copy.deepcopy(schedule_nsdfg_arr_desc)
            nsdfg.add_array(name=transient_array, shape=arr_desc.shape, dtype=arr_desc.dtype, storage=arr_desc.storage, transient=arr_desc.transient)

            acc_trans_in = nstate.add_access(transient_array)
            schedule_nsdfg_node.add_in_connector(transient_array, force=True)
            input_edge = nstate.add_edge(acc_trans_in, None, schedule_nsdfg_node, transient_array, Memlet(f"{transient_array}"))
            # an_st = nstate.add_access(stream)
            # nstate.add_edge(an_st, None, acc_trans_in, None, input_edge.data)

            acc_trans_out = nstate.add_access(transient_array)
            schedule_nsdfg_node.add_out_connector(transient_array, force=True)
            output_edge_data = copy.deepcopy(input_edge.data)
            output_edge_data.subset = subsets.Range.from_array(nsdfg.arrays[transient_array])
            nstate.add_edge(schedule_nsdfg_node, transient_array, acc_trans_out, None, output_edge_data)

            schedule_nsdfg_arr_desc.transient = False
            
            
            
        for e in nstate.in_edges(schedule_nsdfg_node):
            if e.src.data in global_arrays.values():
                for idx, rng in enumerate(e.data.subset.ranges):
                    if f"{rng[0]}" == f"{map_param}":
                        e.data.subset.ranges[idx] = (map_rstart, map_rend, map_rend + 1)   



        ##############################
        # Add initial reads to initial nested state
        loop_block = nsdfg_node.sdfg.start_block
        initial_state = nsdfg_node.sdfg.add_state_before(loop_block, '%s_init' % map_entry.map.label)

        initial_state.add_nodes_from(schedule_nsdfg.start_state.nodes())
        for e in schedule_nsdfg.start_state.edges():
            initial_state.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))

        

        # Remove cannon initial state
        old_start_state = schedule_nsdfg.start_state
        new_start_state = schedule_nsdfg.add_state_after(schedule_nsdfg.start_state, label=old_start_state.label, is_start_block=True)
        schedule_nsdfg.remove_node(old_start_state)
        

        # Get the loop region of the nested SDFG
        loop_region = None
        for n in schedule_nsdfg.nodes():
            if isinstance(n, LoopRegion):
                loop_region = n
                break
        if loop_region is None:
            raise ValueError("Loop region not found in nested SDFG")
        lr_param = loop_region.loop_variable

        # hbm load expr
        init_hbm_load_expr = f"0"
        


        # Add the HBM split load state before last state in loop region
        sync_state = loop_region.nodes()[-1]
        loop_region.remove_node(sync_state)
        last_state = loop_region.nodes()[-1]

        
        cb_hbm_split_load = ConditionalBlock(
            label="hbm_split_load",
            sdfg=schedule_nsdfg,
            parent=loop_region,
        )

        loop_region.add_edge(last_state, cb_hbm_split_load, InterstateEdge(None, None))

        cb_hbm_split_load_cfg1 = ControlFlowRegion(
            label="hbm_split_load_s1"
        )

        # condition when to communicate
        comm_hbm_split_load_cond = CodeBlock(
            code=f"({map_param} < {map_rend} - {map_rstride} )",
            language=dtypes.Language.Python
        )

        cb_hbm_split_load.add_branch(
            condition=comm_hbm_split_load_cond,
            branch=cb_hbm_split_load_cfg1
        )

        hbm_split_load_start = cb_hbm_split_load_cfg1.add_state(f"hbm_split_load_start", is_start_block=True)

        local_cb_list = []
        for transient in transients_to_modify.values():
            desc: data.Array = schedule_nsdfg.arrays[transient]
            for edge in initial_state.edges():
                if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == transient:
                    hbm_src = copy.deepcopy(edge.src)           
                    hbm_edge = copy.deepcopy(edge)        
                    cb_communication_local = ConditionalBlock(
                        label=f"hbm_load_{transient}",
                        sdfg=schedule_nsdfg,
                        parent=cb_hbm_split_load_cfg1
                    )
                    
                    local_cb_list.append(cb_communication_local)           
                    
                    if transient == "local_A":
                        # Load from HBM
                        comm_cfg1_local_cond_hbm = CodeBlock(
                            code=f"({gj} == {NPE} - {lr_param} - 1)",
                            language=dtypes.Language.Python
                        )
                    
                    if transient == "local_B":
                        # Load from HBM
                        comm_cfg1_local_cond_hbm = CodeBlock(
                            code=f"({gi} == {NPE} - {lr_param} - 1)",
                            language=dtypes.Language.Python
                        )
                    
                    # Load from HBM
                    cb_comm_local_hbm_cfg = ControlFlowRegion(
                        label=f"cb_{transient}_hbm"
                    )
                    
                    cb_communication_local.add_branch(
                        condition=comm_cfg1_local_cond_hbm,
                        branch=cb_comm_local_hbm_cfg
                    )
                    
                    local_hbm_state = cb_comm_local_hbm_cfg.add_state(f"{transient}_hbm")
                    hbm_an = local_hbm_state.add_access(hbm_src.data)
                    local_san = local_hbm_state.add_access(f"{transient}")
                    if transient == "local_A":
                        range_expr = symbolic.pystr_to_symbolic('(%s + %s)' % (map_param, map_rstride)) 
                        (r_start, r_end, r_stride) = hbm_edge.data.subset.ranges[1] 
                        new_r_start = r_start + range_expr
                        new_r_end = r_end + range_expr
                        hbm_edge.data.subset.ranges[1] = (new_r_start, new_r_end, r_stride)
                        
                    elif transient == "local_B":
                        range_expr = symbolic.pystr_to_symbolic('(%s + %s)' % (map_param, map_rstride)) 
                        (r_start, r_end, r_stride) = hbm_edge.data.subset.ranges[0] 
                        new_r_start = r_start + range_expr
                        new_r_end = r_end + range_expr
                        hbm_edge.data.subset.ranges[0] = (new_r_start, new_r_end, r_stride)
                    local_hbm_state.add_edge(hbm_an, None, local_san, None, hbm_edge.data)
                    hbm_load_expr = f"({map_param} / {map_rstride} + 1) % 4"
                    # print(hbm_load_expr)
                    sd.replace(local_hbm_state, "__db_local_array_param", hbm_load_expr)
                    

        for idx, cb in enumerate(local_cb_list):
            if idx == 0:
                cb_hbm_split_load_cfg1.add_edge(hbm_split_load_start, cb, InterstateEdge(None, None))
            else:
                cb_hbm_split_load_cfg1.add_edge(local_cb_list[idx-1], cb, InterstateEdge(None, None))



        loop_region.add_edge(cb_hbm_split_load, sync_state, InterstateEdge(None, None))

        sd.replace(initial_state, "__db_local_array_param", init_hbm_load_expr)

        for state in loop_region.all_states():
            if state.name == "compute":
                current_hbm_load_expr = f"({map_param} / {map_rstride} + 1) % 4"
                current_tcdm_expr = f"({current_hbm_load_expr} + 3 + ({lr_param} % 2) * 3) % 4"
                sd.replace(state, "__db_local_stream_param", current_tcdm_expr)
                sd.replace(state, "__db_local_array_param", current_tcdm_expr)
            elif state.name == "communication":
                current_hbm_load_expr = f"({map_param} / {map_rstride} + 1) % 4"
                current_tcdm_expr = f"({current_hbm_load_expr} + 3 + ({lr_param} % 2) * 3) % 4"
                next_tcdm_expr = f"({current_hbm_load_expr} + 3 + (({lr_param} + 1) % 2) * 3) % 4"
                for edge in state.edges():
                    if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data in transients_to_modify.values():
                        edge.data.subset = self._replace_in_subset(edge.data.subset, "__db_local_stream_param", current_tcdm_expr)
                        edge.data.other_subset = self._replace_in_subset(edge.data.other_subset, "__db_local_array_param", next_tcdm_expr)
                    elif isinstance(edge.src, nodes.AccessNode) and edge.src.data in transients_to_modify.values():
                        edge.data.subset = self._replace_in_subset(edge.data.subset, "__db_local_array_param", current_tcdm_expr)
                        edge.data.other_subset = self._replace_in_subset(edge.data.other_subset, "__db_local_stream_param", current_tcdm_expr)
                
                
        
        return nsdfg_node

    @staticmethod
    def _modify_memlet(sdfg, subset, data_name, param_name:str):
        desc = sdfg.arrays[data_name]
        if isinstance(desc, data.Array):
            if len(subset) == len(desc.shape):
                # Already in the right shape, modify new dimension
                subset = list(subset)[1:]

            new_subset = subsets.Range([(param_name, param_name, 1)] + list(subset))
            return new_subset
        
        if isinstance(desc, data.Stream):
            if len(subset) == len(desc.shape):
                # Already in the right shape, modify new dimension
                subset = list(subset)[:2] + list(subset)[3:]

            new_subset = subsets.Range(list(subset)[:2] + [(param_name, param_name, 1)] + list(subset)[2:])
            return new_subset

    @staticmethod
    def _replace_in_subset(subset, string_or_symbol, new_string_or_symbol):
        new_subset = copy.deepcopy(subset)

        repldict = {symbolic.pystr_to_symbolic(string_or_symbol): symbolic.pystr_to_symbolic(new_string_or_symbol)}

        for i, dim in enumerate(new_subset):
            try:
                new_subset[i] = tuple(d.subs(repldict) for d in dim)
            except TypeError:
                new_subset[i] = (dim.subs(repldict) if symbolic.issymbolic(dim) else dim)

        return new_subset
