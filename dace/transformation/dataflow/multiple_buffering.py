import copy
import functools
import warnings
import dace
from dace.properties import make_properties
import dace.properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import CodeBlock, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import transformation
import typing

@make_properties
class MultipleBuffering(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(dace.nodes.MapEntry)

    device_map_type = dace.properties.Property(
        dtype=dace.dtypes.ScheduleType,
        default=dace.dtypes.ScheduleType.GPU_Device,
        desc="The schedule type of the map entry to which the transformation is applied. "
    )
    copy_src_type = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Global,
        desc="The storage type of the source arrays for the copy operations. "
    )
    copy_dst_type = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Shared,
        desc="The storage type of the destination arrays for the copy operations. "
             "This is typically GPU_Shared for GPU kernels.",
    )
    pipeline_depth = dace.properties.Property(
        dtype=int,
        default=2,
        desc="The depth of the pipeline for multiple buffering. "
             "This is the number of iterations that can be in flight at once.",
    )
    validate = dace.properties.Property(
        dtype=bool,
        default=True,
        desc="Whether to validate the SDFG after applying the transformation.",
    )
    prefill_cfg_id = dace.properties.Property(
        dtype=int,
        default=0,
        desc="Internal property to keep track of the prefill CFG ID. "
             "This is used to ensure unique names for the prefill states.",
    )
    prefetch_cfg_id = dace.properties.Property(
        dtype=int,
        default=0,
        desc="Internal property to keep track of the prefetch CFG ID. "
             "This is used to ensure unique names for the prefetch states.",
    )
    synchronous = dace.properties.Property(
        dtype=bool,
        default=True,
        desc="Whether to use synchronous or asynchronous copies. "
             "If True, the transformation will use synchronous copies, "
             "which means that the copy operations will use synchronous API (use registers on GPUs). "
             "If False, the transformation will use asynchronous copies (cuda::memcpy_async + pipeline), "
             "which means that the copy operations will not block.",
    )

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index, sdfg: dace.SDFG, permissive=False):
        # Is map entry
        if not isinstance(self.map_entry, dace.sdfg.nodes.MapEntry):
            warnings.warn("MultipleBuffering transformation can only be applied to MapEntry nodes.")
            return False

        # Ensure device map
        if self.map_entry.map.schedule != self.device_map_type:
            warnings.warn(f"MultipleBuffering transformation can only be applied to MapEntry nodes with schedule {self.device_map_type}. "
                  f"Current schedule: {self.map_entry.map.schedule}")
            return False

        # Kernel directly in a state
        if not isinstance(graph, dace.SDFGState):
            warnings.warn("MultipleBuffering transformation can only be applied to a state in an SDFG.")
            return False

        if self.device_map_type is None or self.copy_src_type is None or self.copy_dst_type is None:
            warnings.warn("Device Map Type, Copy Source Type, and Copy Destination Type must be set for MultipleBuffering transformation differently for None.")
            return False

        # For now only GPU codegen supports this
        if self.device_map_type != dace.dtypes.ScheduleType.GPU_Device:
            #    raise ValueError("The device_map_type must be set to GPU_Device for this transformation currently. TO-DO")
            warnings.warn("MultipleBuffering transformation can currently only be applied to GPU_Device maps.")
            return False
        if self.copy_src_type != dace.dtypes.StorageType.GPU_Global:
            #    raise ValueError("The copy_src_type must be set to GPU_Global for this transformation currently. TO-DO")
            warnings.warn("MultipleBuffering transformation can currently only be applied to GPU_Global source arrays.")
            return False
        if self.copy_dst_type != dace.dtypes.StorageType.GPU_Shared:
            #    raise ValueError("The copy_dst_type must be set to GPU_Shared for this transformation currently. TO-DO")
            warnings.warn("MultipleBuffering transformation can currently only be applied to GPU_Shared destination arrays.")
            return False

        # At least one copy from src to dst within the kernel
        copy_access_nodes = MultipleBuffering._get_copy_access_nodes(
            graph, sdfg, self.map_entry, self.copy_src_type, self.copy_dst_type
        )
        has_src_to_dst_copy = len(copy_access_nodes) > 0

        if not has_src_to_dst_copy:
            print("MultipleBuffering transformation requires at least one copy from source to destination within the kernel.")
            return False

        parent_scopes = MultipleBuffering._get_copy_nodes_parent_scopes(
            graph, sdfg, copy_access_nodes
        )

        if len(parent_scopes) > 1:
            print(f"MultipleBuffering transformation requires all copies to be all from {self.copy_src_type} to {self.copy_dst_type} in the same scope.")
            return False

        parent_scope = next(iter(parent_scopes))
        if (not isinstance(parent_scope, dace.nodes.MapEntry) and
           parent_scope.map.schedule != dace.dtypes.ScheduleType.Sequential):
            print("MultipleBuffering transformation requires all copy nodes to be in a sequential map scope currently (LoopCFG will be added later).")
            return False

        parent_scope_dims = len(parent_scope.map.range)
        if parent_scope_dims > 1:
            print("MultipleBuffering transformation requires all copy nodes to be in a single-dimensional map scope currently. (e.g. matmul K reduction loop, or phases)")
            return False

        if self.pipeline_depth < 2:
            print("MultipleBuffering transformation requires a pipeline depth of at least 2.")
            return False

        for copy_node in copy_access_nodes:
            array_name = copy_node.data
            array = sdfg.arrays[array_name]
            if array.storage != self.copy_dst_type:
                print(f"MultipleBuffering transformation requires all copy destination arrays to be of type {self.copy_dst_type}. "
                      f"Array {array_name} is of type {array.storage}.")
                return False
            if not (MultipleBuffering._is_fortran_style_strides(sdfg, array_name) or MultipleBuffering._is_c_style_strides(sdfg, array_name)):
                print(f"MultipleBuffering transformation requires all copy destination arrays to have Fortran or C-style strides. Indices ordered either from fastest changing to slowest changing (C-style) or vice versa (Fortran-style), and is allocated as a contiguous 1D block. "
                      f"Array {array_name} has strides {array.strides} and shape {array.shape}.")
                return False

        return True

    #@functools.lru_cache(maxsize=None)
    @staticmethod
    def _get_copy_access_nodes(graph: ControlFlowRegion, sdfg: dace.SDFG, map_entry: dace.nodes.MapEntry,
                               copy_src_type: dace.dtypes.StorageType,
                               copy_dst_type: dace.dtypes.StorageType):
        # At least one copy from src to dst within the kernel
        kernel_nodes = graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        kernel_edges = graph.all_edges(*kernel_nodes)
        copy_access_nodes = set()
        for edge in kernel_edges:
            if isinstance(edge.dst, dace.nodes.AccessNode):
                if not isinstance(edge.src, dace.nodes.Tasklet):
                    src_arr = sdfg.arrays[edge.data.data]
                    dst_arr = sdfg.arrays[edge.dst.data]
                    if (src_arr.storage == copy_src_type and
                        dst_arr.storage == copy_dst_type):
                        copy_access_nodes.add(edge.dst)
        return frozenset(copy_access_nodes)

    #@functools.lru_cache(maxsize=None)
    @staticmethod
    def _get_copy_nodes_parent_scopes(graph: ControlFlowRegion, sdfg: dace.SDFG, copy_access_nodes: typing.Set[dace.nodes.AccessNode]):
        parent_scopes = set()
        state = graph
        assert isinstance(state, dace.SDFGState), "Graph must be an SDFG state."
        sdict = state.scope_dict()
        for access_node in copy_access_nodes:
            parent_scopes.add(sdict[access_node])

        return frozenset(parent_scopes)

    #@functools.lru_cache(maxsize=None)
    @staticmethod
    def _is_fortran_style_strides(sdfg: dace.SDFG, array_name: str) -> bool:
        array = sdfg.arrays[array_name]
        # Let's default to C for 1D
        if len(array.shape) == 1:
            return False
        acc_shape = 1
        for i, (dim, stride) in enumerate(zip(array.shape, array.strides)):
            if i != 0:
                acc_shape *= dim
            if stride == acc_shape:
                continue
            else:
                return False
        return True

    #@functools.lru_cache(maxsize=None)
    @staticmethod
    def _is_c_style_strides(sdfg: dace.SDFG, array_name: str) -> bool:
        array = sdfg.arrays[array_name]
        # Let's default to C for 1D
        if len(array.shape) == 1:
            return True
        acc_shape = 1
        for i, (dim, stride) in enumerate(zip(reversed(array.shape), reversed(array.strides))):
            if i != 0:
                acc_shape *= dim
            if stride == acc_shape:
                continue
            else:
                return False
        return True

    @staticmethod
    def _extend_dim(sdfg: dace.SDFG, array_name: str, dim_to_append: int) -> None:
        array = sdfg.arrays[array_name]
        is_fortran_style = MultipleBuffering._is_fortran_style_strides(sdfg, array_name)
        is_c_style = MultipleBuffering._is_c_style_strides(sdfg, array_name)
        array_size = functools.reduce(
            lambda x, y: x * y, array.shape, 1)
        copy_desc = copy.deepcopy(array)
        if not (is_fortran_style or is_c_style):
            raise ValueError(f"Array {array_name} does not have Fortran or C-style strides, cannot extend shape.")
        elif is_fortran_style:
            sdfg.remove_data(name=array_name, validate=False)
            sdfg.add_array(
                name=array_name,
                shape=tuple(list(copy_desc.shape) + [dim_to_append]),
                dtype=copy_desc.dtype,
                strides=tuple(list(copy_desc.strides) + [array_size]),
                storage=copy_desc.storage,
                transient=copy_desc.transient,
            )
            pass
        else:
            assert is_c_style, "Array must be either Fortran or C-style strides."
            sdfg.remove_data(name=array_name, validate=False)
            sdfg.add_array(
                name=array_name,
                shape=tuple([dim_to_append] + list(copy_desc.shape)),
                dtype=copy_desc.dtype,
                strides=tuple([array_size] + list(copy_desc.strides)),
                storage=copy_desc.storage,
                transient=copy_desc.transient,
            )

    def apply(self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        # Setup dst access nodes of copies, and the get the parent scope aagain
        copy_access_nodes = MultipleBuffering._get_copy_access_nodes(
            graph, sdfg, self.map_entry, self.copy_src_type, self.copy_dst_type
        )
        parent_scopes = MultipleBuffering._get_copy_nodes_parent_scopes(
            graph, sdfg, copy_access_nodes
        )
        state : dace.SDFGState = graph
        assert isinstance(state, dace.SDFGState), "Graph must be an SDFG state."
        assert len(parent_scopes) == 1, "MultipleBuffering transformation requires all copy nodes to be in the same scope."
        parent_scope = next(iter(parent_scopes))
        assert len(parent_scope.map.range) == 1, "MultipleBuffering transformation requires all copy nodes to be in a single-dimensional map scope currently."

        previous_range = parent_scope.map.range[0]
        (b, e, s) = previous_range
        assert s == 1, "MultipleBuffering transformation requires the map range to have a step of 1."
        parent_scope_begin = b
        parent_scope_end = e
        parent_scope_map_size = (e + 1 - b) // s
        map_param = parent_scope.map.params[0]

        # Get the symbolic expressions for each copies
        # Copy expression is src_name, dst_name, and memlet tuples
        prefill_copy_expressions = []
        prefetch_copy_expressions = []
        prefill_loop_var_name = "pipe_stage"

        # Extend shared memory (dst copy stype storage) dimensions with the pipeline_depth
        for copy_node in copy_access_nodes:
            array_name = copy_node.data
            assert array_name in sdfg.arrays, f"Array {array_name} not found in SDFG."
            assert sdfg.arrays[array_name].storage == self.copy_dst_type, f"Array {array_name} must be of type {self.copy_dst_type}."
            MultipleBuffering._extend_dim(
                sdfg=sdfg,
                array_name=array_name,
                dim_to_append=self.pipeline_depth,
            )

        # Generate expressions for pipeline prefill
        for access_node in copy_access_nodes:
            in_edges = state.in_edges(access_node)
            assert len(in_edges) == 1, "MultipleBuffering transformation requires each copy access node to have exactly one incoming edge."
            in_edge = in_edges[0]
            src_name = in_edge.data.data
            new_memlet = copy.deepcopy(in_edge.data)
            dst_name = access_node.data
            new_memlet.replace(
                repl_dict = {map_param: prefill_loop_var_name},
            )
            out_edges = state.out_edges(access_node)
            assert len(out_edges) == 1, "MultipleBuffering transformation requires each copy access node to have exactly one outgoing edge."
            out_edge = out_edges[0]
            out_subset = copy.deepcopy(out_edge.data)

            new_other_subset = MultipleBuffering._map_src_subset_to_dst_subset(
                sdfg=sdfg,
                src_name=src_name,
                map_param=map_param,
                prefill_loop_var_name=prefill_loop_var_name,
                out_subset=out_subset,
                modulo_pipeline_depth=False,
                pipeline_depth=self.pipeline_depth,
            )

            prefill_copy_expressions.append(
                (src_name, dst_name, new_memlet, new_other_subset, access_node)
            )

        # Generate expressions for next iteration
        map_param_p1 = f"({map_param} + 1)"
        for access_node in copy_access_nodes:
            in_edges = state.in_edges(access_node)
            assert len(in_edges) == 1, "MultipleBuffering transformation requires each copy access node to have exactly one incoming edge."
            in_edge = in_edges[0]
            # Using sympy.subs would be better
            src_name = in_edge.data.data
            new_memlet = copy.deepcopy(in_edge.data)
            dst_name = access_node.data
            new_memlet.replace(
                repl_dict = {map_param: map_param_p1}
            )
            out_edges = state.out_edges(access_node)
            assert len(out_edges) == 1, "MultipleBuffering transformation requires each copy access node to have exactly one outgoing edge."
            out_edge = out_edges[0]
            out_subset = copy.deepcopy(out_edge.data)

            new_other_subset = MultipleBuffering._map_src_subset_to_dst_subset(
                sdfg=sdfg,
                src_name=src_name,
                map_param=map_param,
                prefill_loop_var_name=map_param_p1,
                out_subset=out_subset,
                modulo_pipeline_depth=True,
                pipeline_depth=self.pipeline_depth,
            )

            prefetch_copy_expressions.append(
                (src_name, dst_name, new_memlet, new_other_subset, access_node)
            )


        # Create pipeline prefill CFG
        # Prefilling the pipeline looks as follows:
        # For i = 0; i < pipeline_depth - 1; i++
        #   If i < map_end
        #     // Insert copy for i

        # Create SDFG for the prefill loop, and the prefill loop CFG
        pipeline_names = dict()
        ret_tupl = self._add_prefill_state(state, prefill_loop_var_name,
                                           e, prefill_copy_expressions, sdfg,
                                           pipeline_names)
        prefill_state: dace.SDFGState = ret_tupl[0]
        prefill_inputs: typing.Set[str] = ret_tupl[1]
        prefill_outputs: typing.Set[str] = ret_tupl[2]
        prefill_nsdfg: dace.nodes.NestedSDFG = ret_tupl[3]
        self.prefill_cfg_id += 1

        # Add in flow in is the whole global memory accessed for this kernel (in edges to parent scope)
        # And out flow is the whole dst_storage accessed repeatedly within this kernel (whole pipeline width)
        for (src_name, dst_name, memlet, other_subset, _) in prefill_copy_expressions:
            assert src_name in prefill_inputs
            assert dst_name in prefill_outputs
            parent_parent = state.entry_node(parent_scope)
            out_edges = list(state.out_edges_by_connector(node=parent_parent, connector="OUT_" + src_name))
            assert len(out_edges) == 1
            out_edge = out_edges[0]
            assert out_edge.dst == parent_scope

            nsdfg_in_memlet = copy.deepcopy(out_edge.data)

            state.add_edge(
                u=parent_parent,
                u_connector=out_edge.src_conn,
                v=prefill_nsdfg,
                v_connector=src_name,
                memlet=dace.Memlet.from_array(
                    dataname=src_name,
                    datadesc=sdfg.arrays[src_name],
                    wcr=None,
                )
            )

            # Need an intermediate out access node for the SDFG to be correct
            intermediate_out_access = state.add_access(
                array_or_stream_name=dst_name,
            )
            state.add_edge(
                u=prefill_nsdfg,
                u_connector=dst_name,
                v=intermediate_out_access,
                v_connector=None,
                memlet=dace.Memlet.from_array(
                    dataname=dst_name,
                    datadesc=sdfg.arrays[dst_name],
                    wcr=None,
                )
            )
            state.add_edge(
                u=intermediate_out_access,
                u_connector=None,
                v=parent_scope,
                v_connector=out_edge.dst_conn,
                memlet=dace.Memlet.from_array(
                    dataname=dst_name,
                    datadesc=sdfg.arrays[dst_name],
                    wcr=None,
                )
            )
            # We can remove the out_edge from the parent scope
            state.remove_edge(out_edge)


        ret_tupl = self._add_prefetch_state(state, map_param_p1,
                                            e, prefetch_copy_expressions, sdfg, pipeline_names)
        prefetch_state: dace.SDFGState = ret_tupl[0]
        prefetch_inputs: typing.Set[str] = ret_tupl[1]
        prefetch_outputs: typing.Set[str] = ret_tupl[2]
        prefetch_nsdfg: dace.nodes.NestedSDFG = ret_tupl[3]
        self.prefetch_cfg_id += 1

        # Add in flow for the prefetch is is the global for next iteration
        # And out flow is the shared memory for the next iteration
        for (src_name, dst_name, memlet, other_subset, access_node) in prefetch_copy_expressions:
            out_edges = state.out_edges(access_node)
            assert len(out_edges) == 1
            out_edge = out_edges[0]
            in_edges = state.in_edges(access_node)
            assert len(in_edges) == 1
            in_edge = in_edges[0]

            # I find it easier to let the whole array flow in to the Nested SDFG
            state.add_edge(
                u=in_edge.src,
                u_connector="OUT_prefetch_" + src_name,
                v=prefetch_nsdfg,
                v_connector=src_name,
                memlet=dace.Memlet.from_array(
                    dataname=src_name,
                    datadesc=sdfg.arrays[src_name],
                )
            )

            in_edge.src.add_out_connector("OUT_prefetch_" + src_name)
            in_edge.src.add_in_connector("IN_prefetch_" + src_name)

            # Need to add data flow until device map entry of the src_name
            parent_parent = state.entry_node(parent_scope)
            prop_up_expr = str(parent_scope_map_size)
            propagated_subset_as_list = [(b,e,s) for (b,e,s) in copy.deepcopy(in_edge.data.subset)]
            map_param_as_sym = dace.symbolic.symbol(map_param)
            propagated_updated_subset = [(b.subs(map_param_as_sym, parent_scope_begin),e.subs(map_param_as_sym, parent_scope_end),s) for (b,e,s) in propagated_subset_as_list ]
            state.add_edge(
                u=parent_parent,
                u_connector="OUT_" + src_name,
                v=parent_scope,
                v_connector="IN_prefetch_" + src_name,
                memlet=dace.Memlet.from_array(
                    dataname=src_name,
                    datadesc=sdfg.arrays[src_name],
                    wcr=None,
                ),
            )
            if not "OUT_" + src_name in parent_parent.out_connectors:
                parent_parent.add_out_connector("OUT_" + src_name)
                parent_parent.add_in_connector("IN_" + src_name)
                data_access = state.add_access(
                    array_or_stream_name=src_name,
                )
                state.add_edge(
                    u=data_access,
                    u_connector=None,
                    v=parent_parent,
                    v_connector="IN_" + src_name,
                    memlet=dace.Memlet.from_array(
                        dataname=src_name,
                        datadesc=sdfg.arrays[src_name],
                        wcr=None,
                    )
                )

            # Need an intermediate out access node for the SDFG to be correct
            intermediate_out_access = state.add_access(
                array_or_stream_name=dst_name,
            )

            state.add_edge(
                u=prefetch_nsdfg,
                u_connector=dst_name,
                v=intermediate_out_access,
                v_connector=None,
                memlet=dace.Memlet.from_array(
                    dataname=dst_name,
                    datadesc=sdfg.arrays[dst_name],
                    wcr=None,
                )
            )
            state.add_edge(
                u=intermediate_out_access,
                u_connector=None,
                v=out_edge.dst,
                v_connector=None,
                memlet=dace.Memlet()
            )


            # Before we had A[...1] -> shrA[...2]
            # It needs to be changed to shrA[...2] -> SyncTasklet -> A[...2]
            base_subset = copy.deepcopy(out_edge.data.subset)
            in_edge.data.data = dst_name
            in_edge.data.subset = base_subset
            in_edge.data.other_subset = None

            # Extend all subsets from shrA[...2] to shrA[...2][pipeline_stage]
            pipe_stage_expr_str = f"({map_param} % {self.pipeline_depth})"
            MultipleBuffering._expand_subset(
                sdfg=sdfg,
                state=state,
                array_name=dst_name,
                expand_expr=pipe_stage_expr_str,
                entry_scope=parent_scope,
            )

            # Rm access node if sync variant, forward the in edge
            if self.synchronous:
                new_in_edge_tuple = (in_edge.src, in_edge.src_conn, out_edge.dst, out_edge.dst_conn, copy.deepcopy(in_edge.data))
                state.remove_edge(out_edge)
                state.remove_edge(in_edge)
                state.remove_node(access_node)
                state.add_edge(*new_in_edge_tuple)
            else:
                # Replace access node with a SyncTasklet
                data_name = access_node.data
                pipeline_name = pipeline_names[data_name]
                sync_tasklet = state.add_tasklet(
                    name=f"sync_{pipeline_name}",
                    inputs={"_in1"},
                    outputs={"_out1"},
                    code=f"{pipeline_name}.consumer_wait();",
                    language=dace.dtypes.Language.CPP,
                )
                assert in_edge.data.subset == out_edge.data.subset
                new_in_edge_tuple = (in_edge.src, in_edge.src_conn, sync_tasklet, "_in1", copy.deepcopy(in_edge.data))
                new_out_edge_tuple = (sync_tasklet, "_out1", out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
                state.remove_edge(out_edge)
                state.remove_edge(in_edge)
                state.remove_node(access_node)
                state.add_edge(*new_in_edge_tuple)
                state.add_edge(*new_out_edge_tuple)

        exit_node = state.exit_node(parent_scope)
        next_entry_node = [n for n in state.nodes() if isinstance(n, dace.nodes.EntryNode) and state.entry_node(n) == parent_scope][0]
        prev_exit_node = state.exit_node(next_entry_node)
        if not self.synchronous:
            # Release the consumer pipeline
            t2 = state.add_tasklet(
                name=f"release_pipelines",
                inputs={},
                outputs={},
                code=f"\n".join([f"{pipeline_name}.consumer_release();" for pipeline_name in pipeline_names.values()]),
                language= dace.dtypes.Language.CPP,
                side_effects=True,
            )
            state.add_edge(prev_exit_node, None, t2, None, dace.Memlet())
            state.add_edge(t2, None, exit_node, None, dace.Memlet())
        else:
            # Add a sync threads to the map exit and map entry before all other edges
            t2 = state.add_tasklet(
                name=f"snyc_threads",
                inputs={},
                outputs={},
                code=f"__syncthreads();",
                language= dace.dtypes.Language.CPP,
                side_effects=True,
            )
            state.add_edge(prev_exit_node, None, t2, None, dace.Memlet())
            state.add_edge(t2, None, exit_node, None, dace.Memlet())

            t3 = state.add_tasklet(
                name=f"snyc_threads",
                inputs={},
                outputs={},
                code=f"__syncthreads();",
                language= dace.dtypes.Language.CPP,
                side_effects=True,
            )
            sources = set([ie.src for ie in state.in_edges(next_entry_node)])
            for src in sources:
                if not isinstance(src, dace.nodes.Tasklet):
                    state.add_edge(src, None, t3, None, dace.Memlet())

            state.add_edge(t3, None, next_entry_node, None, dace.Memlet())



        MultipleBuffering._add_missing_symbols(
            parent_sdfg=sdfg,
            nsdfg=prefetch_nsdfg,
            state=state,
        )
        MultipleBuffering._add_missing_symbols(
            parent_sdfg=sdfg,
            nsdfg=prefill_nsdfg,
            state=state,
        )

        if self.validate:
            sdfg.validate()

    @staticmethod
    def _add_missing_symbols(parent_sdfg: dace.SDFG, nsdfg: dace.nodes.NestedSDFG, state: dace.SDFGState):
        connectors = set(nsdfg.in_connectors.keys()).union(nsdfg.out_connectors.keys())
        symbols = set(k for k in nsdfg.sdfg.free_symbols if k not in connectors)
        missing_symbols = [s for s in symbols if s not in nsdfg.symbol_mapping]

        for missing_symbol in missing_symbols:
            smybols_defined_at_scope = state.symbols_defined_at(nsdfg)
            if missing_symbol not in smybols_defined_at_scope:
                raise Exception(f"Missing Symbol {missing_symbol} of Nested SDFG {nsdfg} ({nsdfg.sdfg.label}) not found in parent SDFG ({parent_sdfg.label}).")

            assert smybols_defined_at_scope[missing_symbol] is not None
            if missing_symbol not in nsdfg.sdfg.symbols:
                nsdfg.sdfg.add_symbol(
                    name=missing_symbol,
                    stype=smybols_defined_at_scope[missing_symbol],
                    find_new_name=False,
                )

            # Identity for the symbol mapping
            assert missing_symbol not in nsdfg.symbol_mapping
            nsdfg.symbol_mapping[missing_symbol] = missing_symbol


    @staticmethod
    def _expand_subset(sdfg: dace.SDFG, state: dace.SDFGState, array_name: str, expand_expr: str, entry_scope: dace.nodes.MapEntry):
        end_dimension_count = len(sdfg.arrays[array_name].shape)
        start_dimension_count = len(sdfg.arrays[array_name].shape) - 1
        all_nodes = state.all_nodes_between(entry_scope, state.exit_node(entry_scope))
        for node in [entry_scope] + list(all_nodes) + [state.exit_node(entry_scope)]:
            for edge in state.out_edges(node):
                if (edge.data.data == array_name
                ):
                    assert (len(edge.data.subset) == start_dimension_count or
                     len(edge.data.subset) == end_dimension_count), \
                        f"MultipleBuffering transformation requires the subset of the edge {edge} to have either {start_dimension_count} or {end_dimension_count} dimensions, but got {len(edge.data.subset)} dimensions."
                    assert edge.data.other_subset is None

                    if len(edge.data.subset) == start_dimension_count:
                        subset_as_list = [(b,e,s) for (b,e,s) in copy.deepcopy(edge.data.subset)]
                        if MultipleBuffering._is_fortran_style_strides(sdfg, array_name):
                            new_subset = dace.subsets.Range(
                                ranges=subset_as_list + [(dace.symbolic.SymExpr(expand_expr), dace.symbolic.SymExpr(expand_expr), 1)],
                            )
                        else:
                            new_subset = dace.subsets.Range(
                                ranges=[(dace.symbolic.SymExpr(expand_expr), dace.symbolic.SymExpr(expand_expr), 1)] + subset_as_list,
                            )

                        edge.data = dace.Memlet(
                            data=edge.data.data,
                            subset=new_subset,
                        )

    @staticmethod
    def _map_src_subset_to_dst_subset(sdfg: dace.SDFG, src_name: str, map_param:str,     prefill_loop_var_name: str, out_subset: dace.subsets.Range, modulo_pipeline_depth:bool,
                                      pipeline_depth: int) -> dace.subsets.Range:
        other_subset_base = copy.deepcopy(out_subset)
        other_subset_base2 = copy.deepcopy(out_subset)
        other_subset_base2.replace(
            repl_dict = {map_param: "0"},
        )
        if other_subset_base2 != other_subset_base:
            raise ValueError(
                f"In MultipleBuffering transformation the volume read from the dst arrays should not depend on the map that is split into multiple buffers. (E.g. write to shared memory should cover the whole shared memory array in a GPU kernel to guarantee this)"
                f"Got {other_subset_base2} != {other_subset_base} for array {src_name}."
            )

        new_other_subset_list = [(b, e, s) for (b, e, s) in other_subset_base.subset]
        # Other subset now has a new dimension [expr(i, loop_var_name)] -> [expr(i)][prefoll_loop_var_name]
        # For this replace prefill loop variable with 0 and add it to the next dimension
        if modulo_pipeline_depth:
            expr = dace.symbolic.SymExpr(f"({prefill_loop_var_name} % {pipeline_depth})")
            access_expr = [(expr, expr, 1)]
        else:
            prefill_loop_var = dace.symbol(name=prefill_loop_var_name)
            access_expr = [(prefill_loop_var, prefill_loop_var, 1)]
        if MultipleBuffering._is_fortran_style_strides(sdfg, src_name):
            new_other_subset = dace.subsets.Range(
                ranges=new_other_subset_list + access_expr,
            )
        else:
            new_other_subset = dace.subsets.Range(
                ranges=access_expr + new_other_subset_list,
            )
        return new_other_subset

    def _add_prefill_state(self, state: dace.SDFGState, prefill_loop_var_name: str, map_range_end: typing.Any,
                           prefill_copy_expressions: typing.List[typing.Tuple[str, str, dace.Memlet]], parent_sdfg: dace.SDFG,
                           pipeline_names: dict[str, str] ):
        prefill_main_sdfg = dace.SDFG(
            name=f"pipeline_prefill_main_sdfg_{self.prefill_cfg_id}",
            parent=state,
        )
        prefill_loop = LoopRegion(
            label=f"pipeline_prefill_loop_{self.prefill_cfg_id}",
            condition_expr=f"{prefill_loop_var_name} < ({self.pipeline_depth - 1})",
            loop_var=f"{prefill_loop_var_name}",
            initialize_expr=f"{prefill_loop_var_name} = 0",
            update_expr=f"{prefill_loop_var_name} = {prefill_loop_var_name} + 1", # += not allowed
            inverted=False,
            sdfg=prefill_main_sdfg,
            update_before_condition=True
        )
        prefill_main_sdfg.add_node(
            node=prefill_loop,
            is_start_block=True,
        )

        # Add condition as a node to the Loop CFG Body
        prefill_cond = ConditionalBlock(
            label=f"pipeline_prefill_cond_{self.prefill_cfg_id}",
            sdfg=prefill_loop.sdfg,
            parent=prefill_loop,
        )
        prefill_loop.add_node(
            node=prefill_cond,
            is_start_block=True,
        )


        test_cfg = ControlFlowRegion(
            label=f"pipeline_prefill_test_{self.prefill_cfg_id}",
            sdfg=prefill_cond.sdfg,
            parent=prefill_cond,
        )
        prefill_cond.add_branch(
            condition=CodeBlock(f"{prefill_loop_var_name} <= {map_range_end}"),
            branch=test_cfg,
        )
        prefill_state = test_cfg.add_state(
            label=f"pipeline_prefill_state_{self.prefill_cfg_id}",
            is_start_block=True,
        )

        inputs = {src for (src, _, _, _, _) in prefill_copy_expressions}
        outputs = {dst for (_, dst, _, _, _) in prefill_copy_expressions}
        nsdfg = state.add_nested_sdfg(
            sdfg = prefill_main_sdfg,
            name=f"pipeline_prefill_nsdfg_{self.prefill_cfg_id}",
            parent=state,
            inputs=inputs,
            outputs=outputs,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        for (src_name, dst_name, memlet, other_subset, _) in prefill_copy_expressions:
            src_access_node = prefill_state.add_access(array_or_stream_name=src_name)
            dst_access_node = prefill_state.add_access(array_or_stream_name=dst_name)

            for name in [src_name, dst_name]:
                if name not in prefill_main_sdfg.arrays:
                    array = copy.deepcopy(parent_sdfg.arrays[name])
                    array.transient = False
                    prefill_main_sdfg.add_datadesc(
                        name=name,
                        datadesc=array,
                        find_new_name=False,
                    )

            prefill_state.add_edge(
                u=src_access_node,
                u_connector=None,
                v=dst_access_node,
                v_connector=None,
                memlet=dace.Memlet(data=src_name, subset=memlet.subset, other_subset=other_subset),
            )
            dst_access_node.async_copy = True
            dst_access_node.async_pipeline = f"pipeline_{dst_name}"
            pipeline_names[dst_access_node.data] = dst_access_node.async_pipeline

        return prefill_state, inputs, outputs, nsdfg

    def _add_prefetch_state(self, state: dace.SDFGState, prefetch_cond_var_name: str, map_range_end: typing.Any,
                           prefetch_copy_expressions: typing.List[typing.Tuple[str, str, dace.Memlet]], parent_sdfg: dace.SDFG,
                           pipeline_names: dict[str, str]):
        prefetch_main_sdfg = dace.SDFG(
            name=f"pipeline_prefetch_main_sdfg_{self.prefetch_cfg_id}",
            parent=state,
        )

        # Add condition as a node to the Loop CFG Body
        prefetch_cond = ConditionalBlock(
            label=f"pipeline_prefetch_cond_{self.prefetch_cfg_id}",
            sdfg=prefetch_main_sdfg,
            parent=prefetch_main_sdfg,
        )
        prefetch_main_sdfg.add_node(
            node=prefetch_cond,
            is_start_block=True,
        )

        test_cfg = ControlFlowRegion(
            label=f"pipeline_prefetch_test_{self.prefetch_cfg_id}",
            sdfg=prefetch_cond.sdfg,
            parent=prefetch_cond,
        )
        prefetch_cond.add_branch(
            condition=CodeBlock(f"{prefetch_cond_var_name} <= {map_range_end}"),
            branch=test_cfg,
        )
        prefetch_state = test_cfg.add_state(
            label=f"pipeline_prefetch_state_{self.prefetch_cfg_id}",
            is_start_block=True,
        )

        inputs = set([src for (src, _, _, _, _) in prefetch_copy_expressions])
        outputs = set([dst for (_, dst, _, _, _) in prefetch_copy_expressions])
        nsdfg = state.add_nested_sdfg(
            sdfg = prefetch_main_sdfg,
            name=f"pipeline_prefetch_nsdfg_{self.prefetch_cfg_id}",
            parent=state,
            inputs=inputs,
            outputs=outputs,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        for (src_name, dst_name, memlet, other_subset, _) in prefetch_copy_expressions:
            src_access_node = prefetch_state.add_access(array_or_stream_name=src_name)
            dst_access_node = prefetch_state.add_access(array_or_stream_name=dst_name)

            for name in [src_name, dst_name]:
                if name not in prefetch_main_sdfg.arrays:
                    array = copy.deepcopy(parent_sdfg.arrays[name])
                    array.transient = False
                    prefetch_main_sdfg.add_datadesc(
                        name=name,
                        datadesc=array,
                        find_new_name=False,
                    )

            prefetch_state.add_edge(
                u=src_access_node,
                u_connector=None,
                v=dst_access_node,
                v_connector=None,
                memlet=dace.Memlet(data=src_name, subset=memlet.subset,
                                   other_subset=other_subset),
            )
            dst_access_node.async_copy = True
            dst_access_node.async_pipeline = f"pipeline_{dst_name}"
            pipeline_names[dst_access_node.data] = dst_access_node.async_pipeline

        return prefetch_state, inputs, outputs, nsdfg

    def annotates_memlets(self) -> bool:
        return True
        #return False