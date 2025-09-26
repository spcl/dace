# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy

from dace import data, sdfg as sd, subsets, symbolic, InterstateEdge, SDFGState, Memlet, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, Property, SymbolicProperty, CodeBlock
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion


@make_properties
class SystolicTransformer(transformation.SingleStateTransformation):

    map_entry = transformation.PatternNode(nodes.MapEntry)
    transient = transformation.PatternNode(nodes.AccessNode)

    # Properties
    npe = Property(default=None, allow_none=True, desc="Number of processing elements")
    tM = Property(default=None, allow_none=True, desc="tM")
    tN = Property(default=None, allow_none=True, desc="tN")
    M = SymbolicProperty(default=None, allow_none=True, desc="M")
    N = SymbolicProperty(default=None, allow_none=True, desc="N")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")
    i = SymbolicProperty(default=None, allow_none=True, desc="i")
    j = SymbolicProperty(default=None, allow_none=True, desc="j")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.transient)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        transient = self.transient

        # Only one dimensional maps are allowed
        if len(map_entry.map.params) != 1:
            return False

        # Verify that all directly-connected internal access nodes point to
        # transient arrays
        first = True
        for edge in graph.out_edges(map_entry):
            if isinstance(edge.dst, nodes.AccessNode):
                desc = sdfg.arrays[edge.dst.data]
                if not isinstance(desc, data.Array) or not desc.transient:
                    return False
                else:
                    # To avoid duplicate matches, only match the first transient
                    if first and edge.dst != transient:
                        return False
                    first = False

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        NPE = self.npe
        tM = self.tM
        tN = self.tN
        M = self.M
        N = self.N
        gi = self.gi
        gj = self.gj
        i = self.i
        j = self.j

        ##############################
        # Gather transients to modify
        transients_to_modify = set(edge.dst.data for edge in graph.out_edges(map_entry)
                                   if isinstance(edge.dst, nodes.AccessNode))

        global_arrays = set()
        for edge in graph.out_edges(map_entry):
            if isinstance(edge.dst, nodes.AccessNode):
                path = graph.memlet_path(edge)
                src_node = path[0].src
                global_arrays.add(src_node.data)

        map_param = map_entry.map.params[0]  # Assuming one dimensional
        ##############################
        # Change condition of loop to one fewer iteration (so that the
        # final one reads from the last buffer)
        map_rstart, map_rend, map_rstride = map_entry.map.range[0]
        new_map_rstride = (f"({map_rend + 1})")
        # map_entry.map.range = subsets.Range([(map_rstart, map_rend, new_map_rstride)])

        # Change out edges of map_entry to use the new map parameter
        for edge in graph.out_edges(map_entry):
            path = graph.memlet_path(edge)
            src_node = path[0].src
            if isinstance(src_node, nodes.AccessNode) and src_node.data in global_arrays:
                memlet = edge.data
                ranges = memlet.subset.ranges
                for idx, range in enumerate(ranges):
                    (r_start, r_end, r_stride) = range
                    if f"{r_start}" == f"{map_param}":
                        new_r_start = symbolic.pystr_to_symbolic(map_param)
                        new_r_end = symbolic.pystr_to_symbolic(f"{map_param} + {new_map_rstride} - 1")
                        new_r_stride = symbolic.pystr_to_symbolic(f"{new_map_rstride}")
                        edge.data.subset.ranges[idx] = (new_r_start, new_r_end, new_r_stride)

        map_entry.map.range = subsets.Range([(map_rstart, map_rend, new_map_rstride)])

        # Add dimension to transients and modify memlets
        for transient in transients_to_modify:
            desc: data.Array = sdfg.arrays[transient]
            # Using non-python syntax to ensure properties change
            desc.strides = [desc.total_size] + list(desc.strides)
            desc.shape = [2] + list(desc.shape)
            desc.offset = [0] + list(desc.offset)
            desc.total_size = desc.total_size * 2

        ##############################
        # Modify memlets to use map parameter as buffer index
        modified_subsets = []  # Store modified memlets for final state
        for edge in graph.scope_subgraph(map_entry).edges():
            if edge.data.data in transients_to_modify:
                edge.data.subset = self._modify_memlet(sdfg, edge.data.subset, edge.data.data)
                modified_subsets.append(edge.data.subset)
            else:  # Could be other_subset
                path = graph.memlet_path(edge)
                src_node = path[0].src
                dst_node = path[-1].dst

                # other_subset could be None. In that case, recreate from array
                dataname = None
                if (isinstance(src_node, nodes.AccessNode) and src_node.data in transients_to_modify):
                    dataname = src_node.data
                elif (isinstance(dst_node, nodes.AccessNode) and dst_node.data in transients_to_modify):
                    dataname = dst_node.data
                if dataname is not None:
                    subset = (edge.data.other_subset or subsets.Range.from_array(sdfg.arrays[dataname]))
                    edge.data.other_subset = self._modify_memlet(sdfg, subset, dataname)
                    modified_subsets.append(edge.data.other_subset)

        from dace.transformation.helpers import nest_state_subgraph
        from dace.sdfg import SDFG, SDFGState
        ##############################
        node = nest_state_subgraph(sdfg, graph, graph.scope_subgraph(map_entry, include_entry=False,
                                                                     include_exit=False))
        # node.schedule = map_entry.map.schedule
        nsdfg: SDFG = node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]
        ##############################
        # Add symbols to the nested state
        nsdfg.add_symbol("gi", stype=gi.dtype)
        nsdfg.add_symbol("gj", stype=gj.dtype)
        nsdfg.add_symbol("i", stype=i.dtype)
        nsdfg.add_symbol("j", stype=j.dtype)
        node.symbol_mapping.update({"gi": gi, "gj": gj, "i": i, "j": j})
        ##############################
        for array in nsdfg.arrays.keys():
            if array in global_arrays:
                # copy the shape of the global arrays into the nested state
                desc_nsdfg: data.Array = nsdfg.arrays[array]
                desc_graph: data.Array = sdfg.arrays[array]
                desc_nsdfg.shape = copy.deepcopy(desc_graph.shape)
        ##############################
        # Change the input edge of the nested state
        for edge in graph.in_edges(node):
            path = graph.memlet_path(edge)
            src_node = path[0].src
            if isinstance(src_node, nodes.AccessNode) and src_node.data in global_arrays:
                memlet = edge.data
                ranges = memlet.subset.ranges
                for idx, range in enumerate(ranges):
                    (r_start, r_end, r_stride) = range
                    if f"{r_start}" == f"{map_param}":
                        if idx == 0:
                            # memlet.subset = self._replace_in_subset(memlet.subset, map_param, f"{map_param} + (({gi}+{gj})%{NPE})*{map_rstride}")
                            memlet.other_subset = None
                        if idx == 1:
                            # memlet.subset = self._replace_in_subset(memlet.subset, map_param, f"{map_param} + (({gi}+{gj})%{NPE})*{map_rstride}")
                            memlet.other_subset = None

        ##############################
        new_streams = {}
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            trans_name = transient
            trans_dtype = desc.dtype
            trans_storage = desc.storage
            trans_shape = desc.shape
            sn, s = nsdfg.add_stream(f"s_{trans_name}",
                                     dtype=trans_dtype,
                                     storage=trans_storage,
                                     buffer_size=1,
                                     shape=(NPE, NPE) + trans_shape,
                                     transient=True)
            new_streams[f"s_{trans_name}"] = s

        ##############################
        # init state
        init_state = nsdfg.add_state("init", is_start_block=True)
        # init_state.add_tasklet(name="init",
        #                     inputs=None,
        #                     outputs=None,
        #                     code=f'''
        #                     if (flex_is_dm_core()) {{
        #                         flex_dma_async_wait_all();
        #                     }}
        #                     flex_intra_cluster_sync();
        #                     ''',
        #                     language=dtypes.Language.CPP)
        ##############################
        # condition_expr=f"_c < ((({gi}+{gj}+1) if (({i} < {M} - {NPE}*{tM}) or ({j} < {N} - {NPE}*{tN})) else ({2*NPE-1})) + ({new_map_rstride}/{map_rstride}))",
        # condition_expr=f"_c < (({gi}+{gj}+1) + ({new_map_rstride}/{map_rstride}))",
        ##############################
        lr = LoopRegion(
            label="systolic_loop",
            condition_expr=
            f"_c < ((({gi}+{gj}+1) if (({i} < {M} - {NPE}*{tM}) or ({j} < {N} - {NPE}*{tN})) else ({2*NPE-1})) + ({new_map_rstride}/{map_rstride}))",
            loop_var="_c",
            initialize_expr=f"_c = (({i} + {j}) or 0) * ({gi}+{gj})",
            update_expr="_c = _c + 1",
            sdfg=nsdfg,
        )

        nsdfg.add_edge(init_state, lr, InterstateEdge(None, None))
        lr_param = lr.loop_variable

        lr_s0: SDFGState = lr.add_state("systolic_start", is_start_block=True)

        cb_compute = ConditionalBlock(
            label="compute",
            sdfg=nsdfg,
            parent=lr,
        )

        lr.add_edge(lr_s0, cb_compute, InterstateEdge(None, None))

        cb_compute_cfg1 = ControlFlowRegion(label="cb_compute_s1")

        compute_cfg1_cond = CodeBlock(
            code=f"(({lr_param} > {gi} + {gj}) and ({lr_param} <= {gi} + {gj} + {new_map_rstride}/{map_rstride}))",
            language=dtypes.Language.Python)

        cb_compute.add_branch(condition=compute_cfg1_cond, branch=cb_compute_cfg1)

        lr_s1: SDFGState = cb_compute_cfg1.add_state("systolic_compute")
        lr_s1.add_nodes_from(nstate.nodes())
        for e in nstate.edges():
            lr_s1.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))

        # change the input edge of the compute state
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            for edge in lr_s1.edges():
                if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == transient:
                    compute_stream = f"s_{transient}"
                    # s = lr_s1.add_access(compute_stream)
                    # remove src node of the edge
                    src_node = edge.src
                    lr_s1.remove_edge(edge)
                    if lr_s1.out_degree(src_node) == 0:
                        lr_s1.remove_node(src_node)
                    # new_subset = copy.deepcopy(edge.data.other_subset)
                    # new_subset = subsets.Range([(gi, gi, 1)] + [(gj, gj, 1)] + list(new_subset))
                    # new_edge = copy.deepcopy(edge)
                    # new_edge.data.data = compute_stream
                    # new_edge.data.subset = new_subset
                    # new_edge.data.other_subset = None
                    # lr_s1.add_edge(s, None, edge.dst, None, new_edge.data)

        compute_expr = symbolic.pystr_to_symbolic('(%s) %% 2' % (lr_param))
        sd.replace(lr_s1, '__dace_db_param', compute_expr)

        # systolic communication conditional block
        cb_communication = ConditionalBlock(
            label="communication",
            sdfg=nsdfg,
            parent=lr,
        )

        lr.add_edge(cb_compute, cb_communication, InterstateEdge(None, None))

        cb_comm_cfg1 = ControlFlowRegion(label="cb_comm_s1")

        # condition when to communicate
        comm_cfg1_cond = CodeBlock(
            code=f"(({lr_param} >= {gi} + {gj}) and ({lr_param} < {gi} + {gj} + {new_map_rstride}/{map_rstride}))",
            language=dtypes.Language.Python)

        cb_communication.add_branch(condition=comm_cfg1_cond, branch=cb_comm_cfg1)
        _empty_state_comm = cb_comm_cfg1.add_state(f"empty_comm", is_start_block=True)
        local_cb_list = []
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            for edge in nstate.edges():
                if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == transient:
                    hbm_src = copy.deepcopy(edge.src)
                    hbm_edge = copy.deepcopy(edge)

                    cb_communication_local = ConditionalBlock(label=f"communication_{transient}",
                                                              sdfg=nsdfg,
                                                              parent=cb_comm_cfg1)

                    local_cb_list.append(cb_communication_local)

                    if transient == "local_A":
                        # Load from HBM
                        comm_cfg1_local_cond_hbm = CodeBlock(code=f"({gj} == 0)", language=dtypes.Language.Python)

                        # Load from neighbor TCDM
                        comm_cfg1_local_cond_tcdm = CodeBlock(code=f"(({gj} > 0) and ({gj} < {NPE} - 1))",
                                                              language=dtypes.Language.Python)

                        # Load from neighbor TCDM
                        comm_cfg1_local_cond_tcdm_last = CodeBlock(code=f"({gj} == {NPE} - 1)",
                                                                   language=dtypes.Language.Python)

                    if transient == "local_B":
                        # Load from HBM
                        comm_cfg1_local_cond_hbm = CodeBlock(code=f"({gi} == 0)", language=dtypes.Language.Python)

                        # Load from neighbor TCDM
                        comm_cfg1_local_cond_tcdm = CodeBlock(code=f"(({gi} > 0) and ({gi} < {NPE} - 1))",
                                                              language=dtypes.Language.Python)

                        # Load from neighbor TCDM
                        comm_cfg1_local_cond_tcdm_last = CodeBlock(code=f"({gi} == {NPE} - 1)",
                                                                   language=dtypes.Language.Python)

                    # Load from HBM
                    cb_comm_local_hbm_cfg = ControlFlowRegion(label=f"cb_comm_{transient}_hbm")

                    cb_communication_local.add_branch(condition=comm_cfg1_local_cond_hbm, branch=cb_comm_local_hbm_cfg)

                    local_hbm_state = cb_comm_local_hbm_cfg.add_state(f"{transient}_hbm")
                    hbm_an = local_hbm_state.add_access(hbm_src.data)
                    local_an = local_hbm_state.add_access(f"{transient}")

                    if transient == "local_A":
                        range_expr = symbolic.pystr_to_symbolic('(%s) - %s - %s' % (lr_param, gi, gj))
                        hbm_edge.data.subset.ranges[1] = (range_expr * map_rstride,
                                                          range_expr * map_rstride + map_rstride - 1, 1)
                    elif transient == "local_B":
                        range_expr = symbolic.pystr_to_symbolic('(%s) - %s - %s' % (lr_param, gi, gj))
                        hbm_edge.data.subset.ranges[0] = (range_expr * map_rstride,
                                                          range_expr * map_rstride + map_rstride - 1, 1)
                    # hbm_edge.data.other_subset = subsets.Range([(gi, gi, 1)] + [(gj, gj, 1)] + list(hbm_edge.data.other_subset))
                    local_hbm_state.add_edge(hbm_an, None, local_an, None, hbm_edge.data)
                    sd.replace(local_hbm_state, '__dace_db_param', '(%s+1) %% 2' % lr_param)

                    # Load from neighbor TCDM
                    cb_comm_local_tcdm_cfg = ControlFlowRegion(label=f"cb_comm_{transient}_tcdm")

                    cb_communication_local.add_branch(condition=comm_cfg1_local_cond_tcdm,
                                                      branch=cb_comm_local_tcdm_cfg)

                    local_tcdm_state = cb_comm_local_tcdm_cfg.add_state(f"{transient}_tcdm")

                    for comm_state in [local_hbm_state, local_tcdm_state]:
                        local_an = comm_state.add_access(f"{transient}")
                        remote_san = comm_state.add_access(f"s_{transient}")
                        curr_buffer_range = subsets.Range([(f"{lr_param} % 2", f"{lr_param} % 2", 1)])
                        next_buffer_range = subsets.Range([(f"({lr_param}+1) % 2", f"({lr_param}+1) % 2", 1)])
                        # if transient == "local_A":
                        next_x_pos_range = subsets.Range([(gi, gi, 1)])
                        #     next_y_pos_range = subsets.Range([(f"({gj}+1) % {NPE}", f"({gj}+1) % {NPE}", 1)])
                        # if transient == "local_B":
                        #     next_x_pos_range = subsets.Range([(f"({gi}+1) % {NPE}", f"({gi}+1) % {NPE}", 1)])
                        next_y_pos_range = subsets.Range([(gj, gj, 1)])
                        local_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
                        local_subset = subsets.Range(list(curr_buffer_range) + list(local_subset)[1:])
                        stream_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
                        stream_subset = subsets.Range(list(curr_buffer_range) + list(stream_subset)[1:])
                        stream_subset = subsets.Range(
                            list(next_x_pos_range) + list(next_y_pos_range) + list(stream_subset))
                        comm_state.add_edge(
                            local_an,
                            None,
                            remote_san,
                            None,
                            memlet=Memlet(data=transient, subset=local_subset, other_subset=stream_subset),
                        )

                    # Last TCDM
                    cb_comm_local_tcdm_last_cfg = ControlFlowRegion(label=f"cb_comm_{transient}_tcdm_last")

                    cb_communication_local.add_branch(condition=comm_cfg1_local_cond_tcdm_last,
                                                      branch=cb_comm_local_tcdm_last_cfg)

                    local_tcdm_last_state = cb_comm_local_tcdm_last_cfg.add_state(f"{transient}_tcdm_last")

                    for comm_state in [local_tcdm_state, local_tcdm_last_state]:
                        local_an = comm_state.add_access(f"{transient}")
                        remote_san = comm_state.add_access(f"s_{transient}")
                        curr_buffer_range = subsets.Range([(f"{lr_param} % 2", f"{lr_param} % 2", 1)])
                        next_buffer_range = subsets.Range([(f"({lr_param}+1) % 2", f"({lr_param}+1) % 2", 1)])
                        if transient == "local_A":
                            next_x_pos_range = subsets.Range([(gi, gi, 1)])
                            next_y_pos_range = subsets.Range([(f"({gj}+{NPE-1}) % {NPE}", f"({gj}+{NPE-1}) % {NPE}", 1)
                                                              ])
                        if transient == "local_B":
                            next_x_pos_range = subsets.Range([(f"({gi}+{NPE-1}) % {NPE}", f"({gi}+{NPE-1}) % {NPE}", 1)
                                                              ])
                            next_y_pos_range = subsets.Range([(gj, gj, 1)])
                        local_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
                        local_subset = subsets.Range(list(next_buffer_range) + list(local_subset)[1:])
                        stream_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
                        stream_subset = subsets.Range(list(curr_buffer_range) + list(stream_subset)[1:])
                        stream_subset = subsets.Range(
                            list(next_x_pos_range) + list(next_y_pos_range) + list(stream_subset))
                        comm_state.add_edge(
                            remote_san,
                            None,
                            local_an,
                            None,
                            memlet=Memlet(data=f"s_{transient}", subset=stream_subset, other_subset=local_subset),
                        )
        # Add edge from emtpy_comm one by one in the order of the local_cb_list
        for idx, cb in enumerate(local_cb_list):
            if idx == 0:
                cb_comm_cfg1.add_edge(_empty_state_comm, cb, InterstateEdge(None, None))
            else:
                cb_comm_cfg1.add_edge(local_cb_list[idx - 1], cb, InterstateEdge(None, None))

        lr_sync: SDFGState = lr.add_state("canon_sync")
        lr.add_edge(cb_communication, lr_sync, InterstateEdge(None, None))
        lr_sync.add_tasklet(name="SoftHier_sync",
                            inputs=None,
                            outputs=None,
                            code=f'''
                            if (flex_is_dm_core()) {{
                                flex_dma_async_wait_all();
                            }}
                            flex_intra_cluster_sync();
                            flex_global_barrier_xy();
                            ''',
                            language=dtypes.Language.CPP)

        nsdfg.remove_node(nstate)

    @staticmethod
    def _modify_memlet(sdfg, subset, data_name):
        desc = sdfg.arrays[data_name]
        if len(subset) == len(desc.shape):
            # Already in the right shape, modify new dimension
            subset = list(subset)[1:]

        new_subset = subsets.Range([('__dace_db_param', '__dace_db_param', 1)] + list(subset))
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
