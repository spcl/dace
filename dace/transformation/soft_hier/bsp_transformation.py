# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy
import ast
import random
from random import randint as rand
from dace import data, sdfg as sd, subsets, symbolic, InterstateEdge, SDFGState, Memlet, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation 
from dace.properties import make_properties, Property, SymbolicProperty, CodeBlock, CodeProperty
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion

@make_properties
class BSPTransformer(transformation.SingleStateTransformation):

    accumulator = transformation.PatternNode(nodes.AccessNode)
    map_entry = transformation.PatternNode(nodes.MapEntry)
    transient = transformation.PatternNode(nodes.AccessNode)
    
    # Properties
    npe_x = Property(default=None, allow_none=True, desc="Number of processing elements")
    npe_y = Property(default=None, allow_none=True, desc="Number of processing elements")
    tM = Property(default=None, allow_none=True, desc="tM")
    tN = Property(default=None, allow_none=True, desc="tN")
    tK = Property(default=None, allow_none=True, desc="tK")
    M  = SymbolicProperty(default=None, allow_none=True, desc="M")
    N  = SymbolicProperty(default=None, allow_none=True, desc="N")
    K  = SymbolicProperty(default=None, allow_none=True, desc="K")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")
    i = SymbolicProperty(default=None, allow_none=True, desc="i")
    j = SymbolicProperty(default=None, allow_none=True, desc="j")

    pre_shift = Property(default=None, allow_none=True, desc="Code to execute before the systolic loop")

    BSP_stride = Property(default=None, allow_none=True, desc="BSP stride")
    BSP_init = CodeProperty(default=None, allow_none=True, desc="Code to execute before the systolic loop")
    BSP_loop = CodeProperty(default=None, allow_none=True, desc="Code to execute before the systolic loop")
    BSP_compute = CodeProperty(default=None, allow_none=True, desc="Code to execute before the systolic loop")
    BSP_communication = CodeProperty(default=None, allow_none=True, desc="Code to execute before the systolic loop")
    BSP_sync = Property(default=True, allow_none=True, desc="is sync needed")

    post_shift = Property(default=None, allow_none=True, desc="Code to execute after the systolic loop")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.accumulator, cls.map_entry, cls.transient)]

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
        accumulator = self.accumulator
        NPE = self.npe_x
        npe_x = self.npe_x
        npe_y = self.npe_y
        tM = self.tM
        tN = self.tN
        tK = self.tK
        M = self.M
        N = self.N
        K = self.K
        gi = self.gi
        gj = self.gj
        i = self.i
        j = self.j

        pre_shift = self.pre_shift
        post_shift = self.post_shift

        BSP_stride = self.BSP_stride
        BSP_init = self.BSP_init
        BSP_loop = self.BSP_loop
        BSP_compute = self.BSP_compute
        BSP_communication = self.BSP_communication
        BSP_sync = self.BSP_sync

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
        new_map_rstride = (f"({BSP_stride})")
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
                        edge.data.subset.ranges[idx] = (new_r_start, new_r_end, 1)

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
        node = nest_state_subgraph(sdfg, graph, graph.scope_subgraph(map_entry, include_entry=False, include_exit=False))
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

        cfg_list = []
        ##############################
        for array in nsdfg.arrays.keys():
            if array in global_arrays:
                # copy the shape of the global arrays into the nested state
                desc_nsdfg: data.Array = nsdfg.arrays[array]
                desc_graph: data.Array = sdfg.arrays[array]
                # print(desc_nsdfg.shape)
                # desc_nsdfg.shape = copy.deepcopy(desc_graph.shape)
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
            sn, s = nsdfg.add_stream(f"s_{trans_name}", dtype=trans_dtype, storage=trans_storage, buffer_size=1, shape=(NPE, NPE)+trans_shape, transient=True)
            new_streams[f"s_{trans_name}"] = s

        ##############################
        # init state
        init_state = nsdfg.add_state("init", is_start_block=True)
        cfg_list.append(init_state)
        
        if BSP_init is not None:
            for ast_node in BSP_init.code:
                if isinstance(ast_node, ast.Assign):
                    self._generate_assign_sdfg(ast_node, init_state, transients_to_modify, global_arrays, new_streams, nsdfg)
            init_sync_state = nsdfg.add_state("init_sync")
            nsdfg.add_edge(init_state, init_sync_state, InterstateEdge(None, None))
            init_sync_state.add_tasklet(name="SoftHier_sync",
                                        inputs=None,
                                        outputs=None,
                                        code=f'''
                                        if (flex_is_dm_core()) {{
                                            flex_dma_async_wait_all();
                                        }}
                                        flex_intra_cluster_sync();
                                        ''',
                                        language=dtypes.Language.CPP)
            cfg_list.append(init_sync_state)


        ##############################
        # Create the LoopRegion using the extracted expressions and loop variable.
        # Extract the assignment and while nodes from BSP_loop.code
        assign_node = BSP_loop.code[0]
        while_node = BSP_loop.code[1]
        initialize_expr = ast.unparse(assign_node)  # e.g., "_c = gi + gj"
        condition_expr = ast.unparse(while_node.test)  # e.g., "_c < gi + gj + 512 / 64"
        update_expr = ast.unparse(while_node.body[0])  # e.g., "_c += 1"
        if isinstance(assign_node.targets[0], ast.Name):
            loop_var = assign_node.targets[0].id
        else:
            raise ValueError("Loop variable extraction failed; expected a Name node.")
        lr = LoopRegion(
            label="loop",
            condition_expr=condition_expr,
            loop_var=loop_var,
            initialize_expr=initialize_expr,
            update_expr=update_expr,
            sdfg=nsdfg,
        )
        
        nsdfg.add_edge(cfg_list[-1], lr, InterstateEdge(None, None))
        cfg_list.append(lr)
        lr_param = lr.loop_variable
        lr_cfg_list = []
        ##############################
        # start block for loop region
        lr_s0 : SDFGState = lr.add_state("start", is_start_block=True)
        lr_cfg_list.append(lr_s0)

        if BSP_compute is not None:
            if_node = BSP_compute.code[0]
            condition_expr = ast.unparse(if_node.test)

            cb_compute = ConditionalBlock(
                label="compute",
                sdfg=nsdfg,
                parent=lr,
            )

            cb_compute_cfg1 = ControlFlowRegion(
                label="cb_compute_s1"
            )

            compute_cfg1_cond = CodeBlock(
                code=condition_expr,
                language=dtypes.Language.Python
            )

            cb_compute.add_branch(
                condition=compute_cfg1_cond,
                branch=cb_compute_cfg1
            )
        if BSP_compute is not None:
            lr_s1 : SDFGState = cb_compute_cfg1.add_state("compute")
            lr.add_edge(lr_s0, cb_compute, InterstateEdge(None, None))
            lr_cfg_list.append(cb_compute)
        else:
            lr_s1 : SDFGState = lr.add_state("compute")
            lr.add_edge(lr_cfg_list[-1], lr_s1, InterstateEdge(None, None))
            lr_cfg_list.append(lr_s1)

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

        compute_expr = symbolic.pystr_to_symbolic('(%s) %% 2' % (lr_param)) 
        sd.replace(lr_s1, '__dace_db_param', compute_expr)

        if len(BSP_communication.code) == 1 and isinstance(BSP_communication.code[0], ast.If):
            cb_comm = self._process_if_node(lr, BSP_communication.code[0], transients_to_modify, global_arrays, new_streams, nsdfg)
            lr.add_edge(lr_cfg_list[-1], cb_comm, InterstateEdge(None, None))
            lr_cfg_list.append(cb_comm)
        elif len(BSP_communication.code) > 1 and isinstance(BSP_communication.code[0], ast.Assign):
            lr_s2 : SDFGState = lr.add_state("communication")
            lr.add_edge(lr_cfg_list[-1], lr_s2, InterstateEdge(None, None))
            lr_cfg_list.append(lr_s2)
            for ast_node in BSP_communication.code:
                if isinstance(ast_node, ast.Assign):
                    self._generate_assign_sdfg(ast_node, lr_s2, transients_to_modify, global_arrays, new_streams, nsdfg)


        if BSP_sync:
            lr_sync : SDFGState = lr.add_state("sync")
            lr.add_edge(lr_cfg_list[-1], lr_sync, InterstateEdge(None, None))
            lr_cfg_list.append(lr_sync)
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

        for edge in graph.in_edges(accumulator):
            entry_node = edge.src

        if pre_shift is not None and post_shift is not None:
            node = nest_state_subgraph(sdfg, graph, graph.scope_subgraph(entry_node, include_entry=False, include_exit=False))
            nsdfg: SDFG = node.sdfg
            nstate: SDFGState = nsdfg.nodes()[0]
            if pre_shift is not None:
                pre_shift_state = nsdfg.add_state("pre_shift", is_start_block=True)
                nsdfg.add_edge(pre_shift_state, nstate, InterstateEdge(None, None))
                pre_shift_state.add_tasklet(name="pre_shift",
                                            inputs=None,
                                            outputs=None,
                                            code=pre_shift,
                                            language=dtypes.Language.CPP)
            if post_shift is not None:
                post_shift_state = nsdfg.add_state("post_shift")
                nsdfg.add_edge(nstate, post_shift_state, InterstateEdge(None, None))
                post_shift_state.add_tasklet(name="post_shift",
                                            inputs=None,
                                            outputs=None,
                                            code=post_shift,
                                            language=dtypes.Language.CPP)


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
    
    def _process_assignment(self, assign_node):
        def _extract_name(node):
            """
            Recursively extract the base variable name from an AST node.
            If the node is a Name, return its id. If it's a Subscript, keep
            traversing its value.
            """
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Subscript):
                return _extract_name(node.value)
            else:
                # Fall back to using ast.unparse if the node type is unexpected.
                return ast.unparse(node).strip()
        dst = _extract_name(assign_node.targets[0])
        src = _extract_name(assign_node.value)
        return dst, src
    
    def _get_index_info(self, subscript_node):
        """
        Given a Subscript node, examine its .slice attribute and return a dictionary
        describing the index. If it is a slice (e.g. a:b or a:b:c), return the lower,
        upper, and step (if available). If it is a simple index expression, return that.
        """
        # For Python 3.9+, ast.Index is removed so the slice is either an ast.Slice or an expression.
        slice_node = subscript_node.slice

        if isinstance(slice_node, ast.Slice):
            lower = ast.unparse(slice_node.lower).strip() if slice_node.lower is not None else None
            upper = ast.unparse(slice_node.upper).strip() if slice_node.upper is not None else None
            step  = ast.unparse(slice_node.step).strip() if slice_node.step is not None else None
            return {"type": "slice", "lower": lower, "upper": upper, "step": step}
        else:
            # If not a slice, then it's a single index expression.
            value = ast.unparse(slice_node).strip()
            return {"type": "index", "value": value}

    def _extract_subscript_info(self, node):
        """
        Recursively traverse a chain of Subscript nodes to extract the base variable name
        and a list of index information for each dimension. The indexes are returned in the
        correct order (first dimension, second dimension, etc.).
        """
        indexes = []
        # Walk up the nested Subscript chain.
        while isinstance(node, ast.Subscript):
            indexes.append(self._get_index_info(node))
            node = node.value  # Move to the next outer node.
        # The base variable should now be a Name node.
        if isinstance(node, ast.Name):
            base_var = node.id
        else:
            base_var = ast.unparse(node).strip()
        indexes.reverse()  # Reverse to restore the original order.
        return base_var, indexes

    def _generate_assign_sdfg(self, assign_node, state, transients_to_modify, global_arrays, new_streams, nsdfg):
        dst, src = self._process_assignment(assign_node)
        # Add the access nodes to the state
        
        # Process left-hand side (target) and right-hand side (value).
        lhs_base, lhs_indexes = self._extract_subscript_info(assign_node.targets[0])
        rhs_base, rhs_indexes = self._extract_subscript_info(assign_node.value)
        # Add the edge between the access nodes
        new_memlet = Memlet()
        if dst in transients_to_modify and src in global_arrays:
            new_memlet.data = f"{src}"
        elif dst in transients_to_modify and src in new_streams.keys():
            new_memlet.data = f"{src}"                            
        elif dst in new_streams.keys() and src in transients_to_modify:
            new_memlet.data = f"{src}"

        # initialize the subset of the memlet, subset is the shape from src, other_subset is the shape from dst
        new_memlet.subset = subsets.Range([(0, s-1, 1) for s in nsdfg.arrays[src].shape])
        new_memlet.other_subset = subsets.Range([(0, s-1, 1) for s in nsdfg.arrays[dst].shape])

        # print("Left-hand side:")
        # print("  Base variable:", lhs_base)
        for dim, idx_info in enumerate(lhs_indexes):
            # print(f"  Dimension {dim}: {idx_info}") 
            if idx_info["type"] == "index":
                new_memlet.other_subset.ranges[dim] = (symbolic.pystr_to_symbolic(idx_info["value"]), 
                                                        symbolic.pystr_to_symbolic(idx_info["value"]), 
                                                        1)
            if idx_info["type"] == "slice":
                # if lower and upper are None, do nothing
                if idx_info["lower"] is None and idx_info["upper"] is None:
                    continue   
        # print("\nRight-hand side:")
        # print("  Base variable:", rhs_base)
        for dim, idx_info in enumerate(rhs_indexes):
            # print(f"  Dimension {dim}: {idx_info}")
            if idx_info["type"] == "index":
                new_memlet.subset.ranges[dim] = (symbolic.pystr_to_symbolic(idx_info["value"]), 
                                                    symbolic.pystr_to_symbolic(idx_info["value"]), 
                                                    1)
            if idx_info["type"] == "slice":
                # if lower and upper are None, do nothing
                if idx_info["lower"] is None and idx_info["upper"] is None:
                    continue
                else:
                    # if step is None, set it to 1
                    if idx_info["step"] is None:
                        idx_info["step"] = "1"
                    new_memlet.subset.ranges[dim] = (symbolic.pystr_to_symbolic(idx_info["lower"]), 
                                                        symbolic.pystr_to_symbolic(idx_info["upper"])-1, 
                                                        symbolic.pystr_to_symbolic(idx_info["step"]))   
        # check if there is dependency between the two access nodes
        # NOTE: this is a naive implementation, it only checks if they share same subset.
        src_an = None
        for e in state.edges():
            if isinstance(e.dst, nodes.AccessNode) and e.dst.data == src:
                # check the memlet of the edge
                if e.data.other_subset == new_memlet.subset:
                    src_an = e.dst
        if src_an is None:
            src_an = state.add_access(src)            
        dst_an = state.add_access(dst)
        state.add_edge(src_an, None, dst_an, None, new_memlet)   

    def _process_if_node(self, parent, if_node, transients_to_modify, global_arrays, new_streams, nsdfg, is_elif=False):
        local_cb_list = []
        if not is_elif:
            cb_communication_local = ConditionalBlock(
                        label=f"communication_{rand(a=0, b=1000)}",
                        sdfg=nsdfg,
                        parent=parent
                    )
        if is_elif:
            cb_communication_local = parent

        condition_expr = ast.unparse(if_node.test)
        comm_cfg1_local_cond = CodeBlock(
            code=condition_expr,
            language=dtypes.Language.Python
        )
                            
        cb_cfg = ControlFlowRegion(
            label=f"cb_comm_{rand(a=0, b=1000)}"
        )
        
        cb_communication_local.add_branch(
            condition=comm_cfg1_local_cond,
            branch=cb_cfg
        )
        
        local_state = cb_cfg.add_state(f"local_{rand(a=0, b=1000)}")
        # Process all nodes in the body of the current if node
        for node in if_node.body:
            if isinstance(node, ast.Assign):
                # Generate the assignment node from the assignment AST node.
                self._generate_assign_sdfg(node, local_state, transients_to_modify, global_arrays, new_streams, nsdfg)
            # You can add more clauses here if you need to support other node types.
            elif isinstance(node, ast.If):
                local_cb_list.append(self._process_if_node(cb_cfg, node, transients_to_modify, global_arrays, new_streams, nsdfg))
        # Process the orelse part if it exists (could be else or elif(s))
        for node in if_node.orelse:
            if isinstance(node, ast.If):
                self._process_if_node(cb_communication_local, node, transients_to_modify, global_arrays, new_streams, nsdfg, is_elif=True)

        for idx, cb in enumerate(local_cb_list):
            if idx == 0:
                cb_cfg.add_edge(local_state, cb, InterstateEdge(None, None))
            else:
                cb_cfg.add_edge(local_cb_list[idx-1], cb, InterstateEdge(None, None))
        return cb_communication_local