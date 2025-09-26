# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy

from dace import data, sdfg as sd, subsets, symbolic, InterstateEdge, SDFGState, Memlet, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, Property, SymbolicProperty
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion


@make_properties
class CanonTransformer(transformation.SingleStateTransformation):

    map_entry = transformation.PatternNode(nodes.MapEntry)
    transient = transformation.PatternNode(nodes.AccessNode)

    # Properties
    npe = Property(default=None, allow_none=True, desc="Number of processing elements")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")

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
        gi = self.gi
        gj = self.gj

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
        new_map_rstride = (f"({map_rstride})*({NPE})")
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
                        new_r_end = symbolic.pystr_to_symbolic(f"{map_param} + {map_rstride}*{NPE} - 1")
                        new_r_stride = symbolic.pystr_to_symbolic(f"{map_rstride}*{NPE}")
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
        node.schedule = map_entry.map.schedule
        nsdfg: SDFG = node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]
        nsdfg.add_symbol("gi", stype=gi.dtype)
        nsdfg.add_symbol("gj", stype=gj.dtype)
        node.symbol_mapping.update({"gi": gi, "gj": gj})
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

        # Add the canon_init state
        init_state = nsdfg.add_state("canon_init")

        ##############################
        # init state
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            for edge in nstate.edges():
                if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == transient:
                    init_src = edge.src.data
                    init_stream = f"s_{transient}"
                    init_edge = copy.deepcopy(edge)
                    if transient == "local_A":
                        init_edge.data.subset.ranges[1] = (((gi + gj) % NPE) * map_rstride,
                                                           ((gi + gj) % NPE) * map_rstride + map_rstride - 1,
                                                           map_rstride)
                    elif transient == "local_B":
                        init_edge.data.subset.ranges[0] = (((gi + gj) % NPE) * map_rstride,
                                                           ((gi + gj) % NPE) * map_rstride + map_rstride - 1,
                                                           map_rstride)

                    init_edge.data.other_subset = subsets.Range([(gi, gi, 1)] + [(gj, gj, 1)] +
                                                                list(init_edge.data.other_subset))
                    init_src_node = init_state.add_access(init_src)
                    init_dst_node = init_state.add_access(init_stream)
                    init_state.add_edge(init_src_node, None, init_dst_node, None, memlet=init_edge.data)
        sd.replace(init_state, '__dace_db_param', 0)

        ##############################
        # Define the loop region (compute + communicate steps)
        lr = LoopRegion(
            label="canon",
            condition_expr=f"_c < {NPE}",
            loop_var="_c",
            initialize_expr="_c = 0",
            update_expr="_c = _c + 1",
            sdfg=nsdfg,
        )

        lr_param = lr.loop_variable

        ##############################
        nsdfg.add_edge(init_state, lr, InterstateEdge(None, None))

        lr_s1: SDFGState = lr.add_state("canon_compute")

        ##############################
        # conon compute state

        # copy all the nodes and edges from the nested state to the compute state
        for node in nstate.nodes():
            lr_s1.add_node(node)
        for edge in nstate.edges():
            lr_s1.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        # change the input edge of the compute state
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            for edge in lr_s1.edges():
                if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == transient:
                    compute_stream = f"s_{transient}"
                    s = lr_s1.add_access(compute_stream)
                    # remove src node of the edge
                    src_node = edge.src
                    lr_s1.remove_edge(edge)
                    if lr_s1.out_degree(src_node) == 0:
                        lr_s1.remove_node(src_node)
                    new_subset = copy.deepcopy(edge.data.other_subset)
                    new_subset = subsets.Range([(gi, gi, 1)] + [(gj, gj, 1)] + list(new_subset))
                    new_edge = copy.deepcopy(edge)
                    new_edge.data.data = compute_stream
                    new_edge.data.subset = new_subset
                    new_edge.data.other_subset = None
                    lr_s1.add_edge(s, None, edge.dst, None, new_edge.data)

        compute_expr = symbolic.pystr_to_symbolic('(%s) %% 2' % (lr_param))
        sd.replace(lr_s1, '__dace_db_param', compute_expr)

        ##############################
        # Add the canon_communication
        for transient in transients_to_modify:
            desc: data.Array = nsdfg.arrays[transient]
            local_an = lr_s1.add_access(transient)
            s_an = lr_s1.add_access(f"s_{transient}")
            curr_buffer_range = subsets.Range([(f"{lr_param} % 2", f"{lr_param} % 2", 1)])
            next_buffer_range = subsets.Range([(f"({lr_param}+1) % 2", f"({lr_param}+1) % 2", 1)])
            if transient == "local_A":  # pass the localA to the right PE
                next_x_pos_range = subsets.Range([(gi, gi, 1)])
                next_y_pos_range = subsets.Range([(f"({gj}+1) % {NPE}", f"({gj}+1) % {NPE}", 1)])
            elif transient == "local_B":  # pass the localB to the bottom PE
                next_x_pos_range = subsets.Range([(f"({gi}+1) % {NPE}", f"({gi}+1) % {NPE}", 1)])
                next_y_pos_range = subsets.Range([(gj, gj, 1)])

            local_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
            local_subset = subsets.Range(list(curr_buffer_range) + list(local_subset)[1:])
            stream_subset = subsets.Range([(0, s - 1, 1) for s in desc.shape])
            stream_subset = subsets.Range(list(next_buffer_range) + list(stream_subset)[1:])
            stream_subset = subsets.Range(list(next_x_pos_range) + list(next_y_pos_range) + list(stream_subset))
            lr_s1.add_edge(
                local_an,
                None,
                s_an,
                None,
                memlet=Memlet(
                    data=transient,
                    subset=local_subset,
                    # other_subset=dace.subsets.Range([((gi, gi, 1), (gi, gi, 1), 1), (((gj + NPE - 1) % NPE), ((gj + NPE -1) % NPE), 1)])
                    other_subset=stream_subset),
            )

        lr_s2: SDFGState = lr.add_state("canon_sync")
        lr.add_edge(lr_s1, lr_s2, InterstateEdge(None, None))
        lr_s2.add_tasklet(name="SoftHier_sync",
                          inputs=None,
                          outputs=None,
                          code="flex_global_barrier_xy();",
                          language=dtypes.Language.CPP)

        ##############################
        # remove nstate from the nested state
        nsdfg.remove_node(nstate)

        return nsdfg

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
