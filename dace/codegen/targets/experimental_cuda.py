# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
# Standard library imports
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import networkx as nx
import sympy

# DaCe core imports
import dace
from dace import data as dt, Memlet
from dace import dtypes, registry, symbolic
from dace.config import Config
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

# DaCe codegen imports
from dace.codegen import common
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (codeblock_to_cpp, memlet_copy_to_absolute_strides, mangle_dace_state_struct_name,
                                      ptr, sym2cpp)
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute

# DaCe transformation imports
from dace.transformation.passes import analysis as ap
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpustream.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpustream.insert_gpu_streams_to_kernels import InsertGPUStreamsToKernels
from dace.transformation.passes.gpustream.insert_gpu_streams_to_tasklets import InsertGPUStreamsToTasklets
from dace.transformation.passes.insert_gpu_copy_tasklets import InsertGPUCopyTasklets
from dace.transformation.passes.gpustream.gpu_stream_topology_simplification import GPUStreamTopologySimplification
from dace.transformation.passes.gpustream.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
#from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync
from dace.transformation.passes.shared_memory_synchronization2 import DefaultSharedMemorySync
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.passes.analysis.infer_gpu_grid_and_block_size import InferGPUGridAndBlockSize

# Experimental CUDA helper imports
from dace.codegen.targets.experimental_cuda_helpers.gpu_stream_manager import GPUStreamManager
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import symbolic_to_cpp, emit_sync_debug_checks, get_defined_type


from dace.codegen.targets import cpp

# Type checking imports (conditional)
if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen

# add symbolic_to_cpp !


@registry.autoregister_params(name='experimental_cuda')
class ExperimentalCUDACodeGen(TargetCodeGenerator):
    """ Experimental CUDA code generator."""
    target_name = 'experimental_cuda'
    title = 'CUDA'

    ###########################################################################
    # Initialization & Preprocessing

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):

        self._frame: DaCeCodeGenerator = frame_codegen  # creates the frame code, orchestrates the code generation for targets
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher  # responsible for dispatching code generation to the appropriate target

        self._in_device_code = False
        self._cpu_codegen: Optional['CPUCodeGen'] = None

        # NOTE: Moved from preprossessing to here
        self.backend: str = common.get_gpu_backend()
        self.language = 'cu' if self.backend == 'cuda' else 'cpp'
        target_type = '' if self.backend == 'cuda' else self.backend
        self._codeobject = CodeObject(sdfg.name + '_' + 'cuda',
                                      '',
                                      self.language,
                                      ExperimentalCUDACodeGen,
                                      'CUDA',
                                      target_type=target_type)

        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()

        # TODO: init and exitcode seem to serve no purpose actually.
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()

        self._global_sdfg: SDFG = sdfg
        self._toplevel_schedule = None

        # Positions at which to deallocate memory pool arrays
        self.pool_release: Dict[Tuple[SDFG, str], Tuple[SDFGState, Set[nodes.Node]]] = {}
        self.has_pool = False

        # INFO:
        # Register GPU schedules and storage types for ExperimentalCUDACodeGen.
        # The dispatcher maps GPU-related schedules and storage types to the
        # appropriate code generation functions in this code generator.

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN, self)
        self._dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        # TODO: Add this to dtypes as well
        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]

        self._dispatcher.register_array_dispatcher(gpu_storage, self)
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)
        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                self._dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self._dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        # NOTE:
        # "Register illegal copies" code NOT copied from cuda.py
        # Behavior unclear for me yet.

        ################## New variables ##########################

        self._current_kernel_spec: Optional[KernelSpec] = None
        self._gpu_stream_manager: Optional[GPUStreamManager] = None
        self._kernel_dimensions_map: Set[nodes.MapEntry] = set()

    def preprocess(self, sdfg: SDFG) -> None:
        """
        Preprocess the SDFG to prepare it for GPU code generation. This includes:
        - Handling GPU<->GPU strided copies.
        - Adding explicit ThreadBlock Maps where missing and infer Grid and Block dimensions for
          every Kernel in the SDFG
        - Runs a pipeline for making GPU stream explicit at the SDFG level and handles other
          GPU stream related initialization.
        - TODO
        - Handling memory pool management

        Note that the order of the steps matters, e.g. TODO
        """

        #------------------------- Hanlde GPU<->GPU strided copies --------------------------

        # Find GPU<->GPU strided copies that cannot be represented by a single copy command
        from dace.transformation.dataflow import CopyToMap
        for e, state in list(sdfg.all_edges_recursive()):
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
                nsdfg = state.parent
                if (e.src.desc(nsdfg).storage == dtypes.StorageType.GPU_Global
                        and e.dst.desc(nsdfg).storage == dtypes.StorageType.GPU_Global):
                    copy_shape, src_strides, dst_strides, _, _ = memlet_copy_to_absolute_strides(
                        None, nsdfg, state, e, e.src, e.dst)
                    dims = len(copy_shape)

                    # Skip supported copy types
                    if dims == 1:
                        continue
                    elif dims == 2:
                        if src_strides[-1] != 1 or dst_strides[-1] != 1:
                            # NOTE: Special case of continuous copy
                            # Example: dcol[0:I, 0:J, k] -> datacol[0:I, 0:J]
                            # with copy shape [I, J] and strides [J*K, K], [J, 1]
                            try:
                                is_src_cont = src_strides[0] / src_strides[1] == copy_shape[1]
                                is_dst_cont = dst_strides[0] / dst_strides[1] == copy_shape[1]
                            except (TypeError, ValueError):
                                is_src_cont = False
                                is_dst_cont = False
                            if is_src_cont and is_dst_cont:
                                continue
                        else:
                            continue
                    elif dims > 2:
                        if not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                            continue

                    # Turn unsupported copy to a map
                    try:
                        CopyToMap.apply_to(nsdfg, save=False, annotate=False, a=e.src, b=e.dst)
                    except ValueError:  # If transformation doesn't match, continue normally
                        continue


        #----------------- Add ThreadBlock Maps & Infer Kernel Grid & Block Sizes --------------------

        # new_nodes - old_nodes gives us all Kernel Entry nodes that were created during the insertion
        # of ThreadBlock maps. Note: the original Kernel Entry was transformed into a ThreadBlock map,
        # and a new GPU_Device (i.e., Kernel) map was inserted on top of it.
        old_nodes = set(node for node, _ in sdfg.all_nodes_recursive())

        # Insert default explicit GPU_ThreadBlock maps where they are missing
        sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)

        new_nodes = set(node for node, _ in sdfg.all_nodes_recursive()) - old_nodes
        kernels_with_added_tb_maps = {
            n
            for n in new_nodes if isinstance(n, nodes.MapEntry) and n.schedule == dtypes.ScheduleType.GPU_Device
        }

        # Infer GPU Grid and Block dimensions
        self._kernel_dimensions_map = InferGPUGridAndBlockSize().apply_pass(sdfg, kernels_with_added_tb_maps)

        #------------------------- GPU Stream related Logic --------------------------

        # Register GPU context in state struct
        self._frame.statestruct.append('dace::cuda::Context *gpu_context;')

        # Define backend stream access expression (e.g., CUDA stream handle)
        gpu_stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"

        # Prepare the Pipeline to make GPU streams explicit: Add and connect SDFG nodes
        # with GPU stream AccessNodes where used
        stream_pipeline = Pipeline(
            [
                NaiveGPUStreamScheduler(),
                InsertGPUStreamsToKernels(),
                InsertGPUStreamsToTasklets(),
                InsertGPUStreamSyncTasklets(),
                InsertGPUCopyTasklets(),
                GPUStreamTopologySimplification(),
            ]
        )
        
        # TODO: Missed copies due to InsertGPUCopyTasklet -> maybe check wheter copies were 
        # handled above than just adding this codegen to used_targets by default
        self._dispatcher._used_targets.add(self)
        gpustream_assignments = stream_pipeline.apply_pass(sdfg, {})['NaiveGPUStreamScheduler']

        # TODO: probably to be deleted
        # Define backend stream access expression (e.g., CUDA stream handle)        
        gpu_stream_access_template = "__state->gpu_context->streams[{gpu_stream}]"

        # Initialize runtime GPU stream manager
        self._gpu_stream_manager = GPUStreamManager(sdfg, gpustream_assignments, gpu_stream_access_template)

        #----------------- Shared Memory Synchronization related Logic -----------------

        auto_sync = Config.get('compiler', 'cuda', 'auto_syncthreads_insertion')
        if auto_sync:
            DefaultSharedMemorySync().apply_pass(sdfg, None)

        #------------------------- Memory Pool related Logic --------------------------

        # Find points where memory should be released to the memory pool
        self._compute_pool_release(sdfg)

    def _compute_pool_release(self, top_sdfg: SDFG):
        """
        Computes positions in the code generator where a memory pool array is no longer used and
        ``backendFreeAsync`` should be called to release it.

        :param top_sdfg: The top-level SDFG to traverse.
        :raises ValueError: If the backend does not support memory pools.
        """
        # Find release points for every array in every SDFG
        reachability = access_nodes = None
        for sdfg in top_sdfg.all_sdfgs_recursive():
            # Skip SDFGs without memory pool hints
            pooled = set(aname for aname, arr in sdfg.arrays.items()
                         if getattr(arr, 'pool', False) is True and arr.transient)
            if not pooled:
                continue
            self.has_pool = True
            if self.backend != 'cuda':
                raise ValueError(f'Backend "{self.backend}" does not support the memory pool allocation hint')

            # Lazily compute reachability and access nodes
            if reachability is None:
                reachability = ap.StateReachability().apply_pass(top_sdfg, {})
                access_nodes = ap.FindAccessStates().apply_pass(top_sdfg, {})

            reachable = reachability[sdfg.cfg_id]
            access_sets = access_nodes[sdfg.cfg_id]
            for state in sdfg.nodes():
                # Find all data descriptors that will no longer be used after this state
                last_state_arrays: Set[str] = set(
                    s for s in access_sets
                    if s in pooled and state in access_sets[s] and not (access_sets[s] & reachable[state]) - {state})

                anodes = list(state.data_nodes())
                for aname in last_state_arrays:
                    # Find out if there is a common descendant access node.
                    # If not, release at end of state
                    ans = [an for an in anodes if an.data == aname]
                    terminator = None
                    for an1 in ans:
                        if all(nx.has_path(state.nx, an2, an1) for an2 in ans if an2 is not an1):
                            terminator = an1
                            break

                    # Old logic below, now we use the gpu_stream manager which returns nullptr automatically
                    # to all nodes thatdid not got assigned a cuda stream
                    """
                    # Enforce a cuda_stream field so that the state-wide deallocation would work
                    if not hasattr(an1, '_cuda_stream'):
                        an1._cuda_stream = 'nullptr'
                    """

                    # If access node was found, find the point where all its reads are complete
                    terminators = set()
                    if terminator is not None:
                        parent = state.entry_node(terminator)
                        # If within a scope, once all memlet paths going out of that scope are complete,
                        # it is time to release the memory
                        if parent is not None:
                            # Just to be safe, release at end of state (e.g., if misused in Sequential map)
                            terminators = set()
                        else:
                            # Otherwise, find common descendant (or end of state) following the ends of
                            # all memlet paths (e.g., (a)->...->[tasklet]-->...->(b))
                            for e in state.out_edges(terminator):
                                if isinstance(e.dst, nodes.EntryNode):
                                    terminators.add(state.exit_node(e.dst))
                                else:
                                    terminators.add(e.dst)
                            # After all outgoing memlets of all the terminators have been processed, memory
                            # will be released

                    self.pool_release[(sdfg, aname)] = (state, terminators)

            # If there is unfreed pooled memory, free at the end of the SDFG
            unfreed = set(arr for arr in pooled if (sdfg, arr) not in self.pool_release)
            if unfreed:
                # Find or make single sink node
                sinks = sdfg.sink_nodes()
                if len(sinks) == 1:
                    sink = sinks[0]
                elif len(sinks) > 1:
                    sink = sdfg.add_state()
                    for s in sinks:
                        sdfg.add_edge(s, sink)
                else:  # len(sinks) == 0:
                    raise ValueError('End state not found when trying to free pooled memory')

                # Add sink as terminator state
                for arr in unfreed:
                    self.pool_release[(sdfg, arr)] = (sink, set())

    ###########################################################################
    # Determine wheter initializer and finalizer should be called

    @property
    def has_initializer(self) -> bool:
        return True

    @property
    def has_finalizer(self) -> bool:
        return True

    ###########################################################################
    # Scope generation

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

        # Import strategies here to avoid circular dependencies
        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import (ScopeGenerationStrategy,
                                                                                     KernelScopeGenerator,
                                                                                     ThreadBlockScopeGenerator,
                                                                                     WarpScopeGenerator)
        # Entry Node of the scope
        scope_entry = dfg_scope.source_nodes()[0]

        #--------------- Start of Kernel Function Code Generation --------------------

        if not self._in_device_code:

            # Enter kernel context and recursively generate device code

            # New scope for defined variables (kernel functions scope)
            self._dispatcher.defined_vars.enter_scope(scope_entry)

            # Store kernel metadata (name, dimensions, arguments, etc.) in a KernelSpec object 
            # and save it as an attribute
            kernel_spec = KernelSpec(cudaCodeGen=self,
                                     sdfg=sdfg,
                                     cfg=cfg,
                                     dfg_scope=dfg_scope,
                                     state_id=state_id)
            
            self._current_kernel_spec = kernel_spec

            # (Re)define variables for the new scope
            self._define_variables_in_kernel_scope(sdfg, self._dispatcher)

            # declare and call kernel wrapper function (in the CPU-side code)
            self._declare_and_invoke_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            # Recursively generate GPU code into the kernel_stream (will be in a .cu file)
            kernel_stream = CodeIOStream()
            kernel_function_stream = self._globalcode

            self._in_device_code = True

            kernel_scope_generator = KernelScopeGenerator(codegen=self)
            if kernel_scope_generator.applicable(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream):
                kernel_scope_generator.generate(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream)
            else:
                raise ValueError("Invalid kernel configuration: This strategy is only applicable if the "
                                 "outermost GPU schedule is of type GPU_Device (most likely cause).")

            # Append generated kernel code to localcode
            self._localcode.write(kernel_stream.getvalue() + '\n')

            # Exit kernel context
            self._in_device_code = False

            # Generate kernel wrapper, i.e. function which will launch the kernel
            self._generate_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            # Exit scope for defined variables 
            self._dispatcher.defined_vars.exit_scope(scope_entry)

            return

        #--------------- Nested GPU Scope --------------------
        supported_strategies: List[ScopeGenerationStrategy] = [
            ThreadBlockScopeGenerator(codegen=self),
            WarpScopeGenerator(codegen=self)
        ]

        for strategy in supported_strategies:
            if strategy.applicable(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
                strategy.generate(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
                return

        #--------------- Unsupported Cases --------------------
        # Note: We are inside a nested GPU scope at this point.

        schedule_type = scope_entry.map.schedule

        if schedule_type == dace.ScheduleType.GPU_Device:
            raise NotImplementedError("Dynamic parallelism (nested GPU_Device schedules) is not supported.")

        raise NotImplementedError(
            f"Scope generation for schedule type '{schedule_type}' is not implemented in ExperimentalCUDACodeGen. "
            "Please check for supported schedule types or implement the corresponding strategy.")

    def _define_variables_in_kernel_scope(self, sdfg: SDFG, dispatcher: TargetDispatcher):
        """
        Define kernel-visible variables in the dispatcher's scope.

        - Certain variables stored in the host-side ``__state`` struct (e.g., persistent or external
          data) cannot be accessed directly in kernel code. They are passed as arguments instead, with 
          pointer names resolved via ``cpp.ptr(..)``. These must be registered in the dispatcher for use 
          in kernel context.

        - KernelSpec may also mark certain variables/arguments as constants, which must be registered with 
          the appropriate ``const`` qualifier in their ctype.
        """
        # Extract argument and constant definitions from the KernelSpec
        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_constants: Set[str] = kernel_spec.kernel_constants
        kernel_arglist: Dict[str, dt.Data] = kernel_spec.arglist

        # Save current in_device_code value for restoration later
        restore_in_device_code = self._in_device_code 
        for name, data_desc in kernel_arglist.items():
    
            # Only arrays relevant
            if not name in sdfg.arrays:
                continue

            data_desc = sdfg.arrays[name]
            # Get the outer/host pointer name
            self._in_device_code = False
            host_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            # Get defined type and ctype for the data (use host pointer name)
            is_global: bool = data_desc.lifetime in (dtypes.AllocationLifetime.Global, 
                                                        dtypes.AllocationLifetime.Persistent,
                                                        dtypes.AllocationLifetime.External)
            defined_type, ctype = dispatcher.defined_vars.get(host_ptrname, is_global=is_global)

            # Get the inner/device pointer name
            self._in_device_code = True
            device_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            # Add the const qualifier if it is a constant AND is not marked as such yet
            if name in kernel_constants:
                if not "const " in ctype:
                    ctype = f"const {ctype}"

            # Register variable with the device pointer name for the kernel context
            dispatcher.defined_vars.add(device_ptrname, defined_type, ctype, allow_shadowing=True)
        
        # Restore in_device_code field
        self._in_device_code = restore_in_device_code

    def _declare_and_invoke_kernel_wrapper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                                            function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

        scope_entry = dfg_scope.source_nodes()[0]

        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_wrapper_args_as_input = kernel_spec.kernel_wrapper_args_as_input
        kernel_wrapper_args_typed = kernel_spec.kernel_wrapper_args_typed

        # Declaration of the kernel wrapper function (in the CPU-side code)
        function_stream.write(
            'DACE_EXPORTED void __dace_runkernel_%s(%s);\n' % (kernel_name, ', '.join(kernel_wrapper_args_typed)), cfg,
            state_id, scope_entry)

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        state = cfg.state(state_id)
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('{', cfg, state_id, scope_entry)

        # Synchronize all events leading to dynamic map range connectors
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg, state_id, scope_entry)
            
        # Calling the kernel wrapper function (in the CPU-side code)
        callsite_stream.write('__dace_runkernel_%s(%s);\n' % (kernel_name, ', '.join(kernel_wrapper_args_as_input)),
                              cfg, state_id, scope_entry)

            
        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('}', cfg, state_id, scope_entry)

    def _generate_kernel_wrapper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

        scope_entry = dfg_scope.source_nodes()[0]

        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_args_as_input = kernel_spec.args_as_input
        kernel_launch_args_typed = kernel_spec.kernel_wrapper_args_typed

        # get kernel dimensions and transform into a c++ string
        grid_dims = kernel_spec.grid_dims
        block_dims = kernel_spec.block_dims
        gdims = ', '.join(symbolic_to_cpp(grid_dims))
        bdims = ', '.join(symbolic_to_cpp(block_dims))

        # ----------------- Kernel Launch Function Declaration -----------------------

        self._localcode.write(
            f"""
            DACE_EXPORTED void __dace_runkernel_{kernel_name}({', '.join(kernel_launch_args_typed)});
            void __dace_runkernel_{kernel_name}({', '.join(kernel_launch_args_typed)})
            """,
            cfg, state_id, scope_entry
        )

        # Open bracket
        self._localcode.write('{', cfg, state_id, scope_entry)

        # ----------------- Guard Checks handling -----------------------

        # Ensure that iteration space is neither empty nor negative sized
        single_dimchecks = []
        for gdim in grid_dims:
            # Only emit a guard if we can't statically prove gdim > 0
            if (gdim > 0) != True:
                single_dimchecks.append(f'(({symbolic_to_cpp(gdim)}) <= 0)')

        dimcheck = ' || '.join(single_dimchecks)

        if dimcheck:
            emptygrid_warning = ''
            if Config.get('debugprint') == 'verbose' or Config.get_bool('compiler', 'cuda', 'syncdebug'):
                emptygrid_warning = (f'printf("Warning: Skipping launching kernel \\"{kernel_name}\\" '
                                     'due to an empty grid.\\n");')

            self._localcode.write(
                f'''
                    if ({dimcheck}) {{
                        {emptygrid_warning}
                        return;
                    }}''', cfg, state_id, scope_entry)

        # ----------------- Kernel Launch Invocation -----------------------
        stream_var_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[1]
        kargs = ', '.join(['(void *)&' + arg for arg in kernel_args_as_input])
        self._localcode.write(
            f'''
            void  *{kernel_name}_args[] = {{ {kargs} }};
            gpuError_t __err = {self.backend}LaunchKernel(
                (void*){kernel_name}, dim3({gdims}), dim3({bdims}), {kernel_name}_args, {0}, {stream_var_name}
            );
            ''', cfg, state_id, scope_entry)

        self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});')
        emit_sync_debug_checks(self.backend, self._localcode)

        # Close bracket
        self._localcode.write('}', cfg, state_id, scope_entry)

    ###########################################################################
    # Generation of Memory Copy Logic

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    edge: Tuple[nodes.Node, str, nodes.Node, str,
                                Memlet], function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

        from dace.codegen.targets.experimental_cuda_helpers.copy_strategies import (CopyContext, CopyStrategy,
                                                                                    OutOfKernelCopyStrategy,
                                                                                    SyncCollaboritveGPUCopyStrategy,
                                                                                    AsyncCollaboritveGPUCopyStrategy,
                                                                                    FallBackGPUCopyStrategy)

        context = CopyContext(self, self._gpu_stream_manager, state_id, src_node, dst_node, edge, sdfg, cfg, dfg,
                              callsite_stream)

        # Order matters: fallback must come last
        strategies: List[CopyStrategy] = [
            OutOfKernelCopyStrategy(),
            SyncCollaboritveGPUCopyStrategy(),
            AsyncCollaboritveGPUCopyStrategy(),
            FallBackGPUCopyStrategy()
        ]

        for strategy in strategies:
            if strategy.applicable(context):
                strategy.generate_copy(context)
                return

        raise RuntimeError("No applicable GPU memory copy strategy found (this should not happen).")

    #############################################################################
    # Predicates for Dispatcher

    def state_dispatch_predicate(self, sdfg, state):
        """
        Determines whether a state should be handled by this
        code generator (`ExperimentalCUDACodeGen`).

        Returns True if the generator is currently generating kernel code.
        """
        return self._in_device_code

    def node_dispatch_predicate(self, sdfg, state, node):
        """
        Determines whether a node should be handled by this
        code generator (`ExperimentalCUDACodeGen`).

        Returns True if:
        - The node has a GPU schedule handled by this backend, or
        - The generator is currently generating kernel code.
        """
        schedule = getattr(node, 'schedule', None)

        if schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            return True

        if self._in_device_code:
            return True

        return False

    #############################################################################
    # Nested SDFG related, testing phase

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = False) -> None:

        # User frame code  to generate state
        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

        # Special: Release of pooled memory if not in device code that need to be released her
        if not self._in_device_code:

            handled_keys = set()
            backend = self.backend
            for (pool_sdfg, name), (pool_state, _) in self.pool_release.items():

                if (pool_sdfg is not sdfg) or (pool_state is not state):
                    continue

                data_descriptor = pool_sdfg.arrays[name]
                ptrname = ptr(name, data_descriptor, pool_sdfg, self._frame)

                # Adjust if there is an offset
                if isinstance(data_descriptor, dt.Array) and data_descriptor.start_offset != 0:
                    ptrname = f'({ptrname} - {sym2cpp(data_descriptor.start_offset)})'

                # Free the memory
                callsite_stream.write(f'DACE_GPU_CHECK({backend}Free({ptrname}));\n', pool_sdfg)

                emit_sync_debug_checks(self.backend, callsite_stream)

                # We handled the key (pool_sdfg, name) and can remove it later
                handled_keys.add((pool_sdfg, name))

            # Delete the handled keys here (not in the for loop, which would cause issues)
            for key in handled_keys:
                del self.pool_release[key]

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

        # get the generating function's name
        gen = getattr(self, '_generate_' + type(node).__name__, False)

        # if it is not implemented, use generate node of cpu impl
        if gen is not False:
            gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node).__name__ == 'MapExit' and node.schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            # Special case: It is a MapExit but from a GPU_schedule- the MapExit is already
            # handled by a KernelScopeManager instance. Otherwise cpu_codegen will close it
            return
        else:
            self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label):
        return 'DACE_DFI ' + self._cpu_codegen.generate_nsdfg_header(
            sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=False)

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(sdfg,
                                                     cfg,
                                                     state,
                                                     node,
                                                     memlet_references,
                                                     sdfg_label,
                                                     state_struct=False)

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        args = self._cpu_codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state, node)
        return args
    
    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: CodeIOStream,
                             callsite_stream: CodeIOStream) -> None:
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self
        
        # Determine and update ctype of new constant data and symbols within the NSDFG
        parent_state: SDFGState = cfg.state(state_id)
        nsdfg = node.sdfg

        # New scope for defined variables
        dispatcher: TargetDispatcher = self._dispatcher
        dispatcher.defined_vars.enter_scope(node)

        # Add the const qualifier to any constants not marked as such

        # update const data
        new_const_data = sdutil.get_constant_data(node, parent_state) - self._current_kernel_spec.kernel_constants
        for name in new_const_data:
            desc = nsdfg.arrays[name]    
            ptr_name = ptr(name, desc, nsdfg, self._frame)
            try: 
                defined_type, ctype = dispatcher.defined_vars.get(ptr_name, is_global=True)
                if not "const " in desc.ctype:
                    ctype = f"const {desc.ctype}"
            except:
                defined_type = get_defined_type(desc)
                if not "const " in desc.ctype:
                    ctype = f"const {desc.ctype}" 
            dispatcher.defined_vars.add(ptr_name, defined_type, ctype, allow_shadowing=True)

        # update const symbols
        new_const_symbols = sdutil.get_constant_symbols(node, parent_state) - self._current_kernel_spec.kernel_constants
        for name in new_const_symbols:
            defined_type = DefinedType.Scalar
            if not "const" in nsdfg.symbols[name].ctype:
                ctype = f"const {nsdfg.symbols[name].ctype}"



        # Redirect rest to CPU codegen
        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        # Exit scope
        dispatcher.defined_vars.exit_scope(node)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule

    #######################################################################
    # Array Declaration, Allocation and Deallocation

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                      declaration_stream: CodeIOStream) -> None:

        ptrname = ptr(node.data, nodedesc, sdfg, self._frame)
        fsymbols = self._frame.symbols_and_constants(sdfg)

        # ----------------- Guard checks --------------------

        # NOTE: `dfg` is None iff `nodedesc` is non-free symbol dependent (see DaCeCodeGenerator.determine_allocation_lifetime).
        # We avoid `is_nonfree_sym_dependent` when dfg is None and `nodedesc` is a View.
        if dfg and not sdutil.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError(
                "declare_array is only for variables that require separate declaration and allocation.")

        if nodedesc.storage == dtypes.StorageType.GPU_Shared:
            raise NotImplementedError("Dynamic shared memory unsupported")

        if nodedesc.storage == dtypes.StorageType.Register:
            raise ValueError("Dynamic allocation of registers is not allowed")

        if nodedesc.storage not in {dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned}:
            raise NotImplementedError(f"CUDA: Unimplemented storage type {nodedesc.storage.name}.")

        if self._dispatcher.declared_arrays.has(ptrname):
            return  # Already declared

        # ----------------- Declaration --------------------
        dataname = node.data
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
        self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, array_ctype)

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        """
        Maybe document here that this also does declaration and that declare_array only declares specific
        kind of data
        """

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------- Guard checks & Redirect to CPU CodeGen -------------

        # Skip if variable is already defined
        if self._dispatcher.defined_vars.has(dataname):
            return

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError("allocate_stream not implemented in ExperimentalCUDACodeGen")
        
        elif isinstance(nodedesc, dace.data.View):
            return self._cpu_codegen.allocate_view(sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream,
                                                   allocation_stream)
        elif isinstance(nodedesc, dace.data.Reference):
            return self._cpu_codegen.allocate_reference(sdfg, cfg, dfg, state_id, node, function_stream,
                                                        declaration_stream, allocation_stream)

        # No clue what is happening here
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # NOTE: Experimental for GPU stream
        if nodedesc.dtype == dtypes.gpuStream_t:
            return

        # ------------------- Allocation/Declaration -------------------

        # Call the appropriate handler based on storage type
        gen = getattr(self, f'_prepare_{nodedesc.storage.name}_array', None)
        if gen:
            gen(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
        else:
            raise NotImplementedError(f'CUDA: Unimplemented storage type {nodedesc.storage}')

    def _prepare_GPU_Global_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Declaration -------------------
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            array_ctype = f'{nodedesc.dtype.ctype} *'
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        if nodedesc.pool:
            gpu_stream_manager = self._gpu_stream_manager
            gpu_stream = gpu_stream_manager.get_stream_node(node)
            if gpu_stream != 'nullptr':
                gpu_stream = f'__state->gpu_context->streams[{gpu_stream}]'
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {gpu_stream}));\n',
                cfg, state_id, node)
            emit_sync_debug_checks(self.backend, allocation_stream)
        else:
            # Strides are left to the user's discretion
            allocation_stream.write(f'DACE_GPU_CHECK({self.backend}Malloc((void**)&{dataname}, {arrsize_malloc}));\n',
                                    cfg, state_id, node)

        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(f'DACE_GPU_CHECK({self.backend}Memset({dataname}, 0, {arrsize_malloc}));\n', cfg,
                                    state_id, node)

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            allocation_stream.write(f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

    def _prepare_CPU_Pinned_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Declaration -------------------
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            array_ctype = f'{nodedesc.dtype.ctype} *'
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        # Strides are left to the user's discretion
        allocation_stream.write(f'DACE_GPU_CHECK({self.backend}MallocHost(&{dataname}, {arrsize_malloc}));\n', cfg,
                                state_id, node)
        if node.setzero:
            allocation_stream.write(f'memset({dataname}, 0, {arrsize_malloc});\n', cfg, state_id, node)

        if nodedesc.start_offset != 0:
            allocation_stream.write(f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

    def _prepare_GPU_Shared_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        arrsize = nodedesc.total_size

        # ------------------- Guard checks -------------------
        if symbolic.issymbolic(arrsize, sdfg.constants):
            raise NotImplementedError('Dynamic shared memory unsupported')
        if nodedesc.start_offset != 0:
            raise NotImplementedError('Start offset unsupported for shared memory')

        # ------------------- Declaration -------------------
        array_ctype = f'{nodedesc.dtype.ctype} *'

        declaration_stream.write(f'__shared__ {nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}];\n', cfg,
                                 state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(
                f'dace::ResetShared<{nodedesc.dtype.ctype}, {", ".join(symbolic_to_cpp(self._current_kernel_spec.block_dims))}, {symbolic_to_cpp(arrsize)}, '
                f'1, false>::Reset({dataname});\n', cfg, state_id, node)

    def _prepare_Register_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # ------------------- Guard checks -------------------
        if symbolic.issymbolic(arrsize, sdfg.constants):
            raise ValueError('Dynamic allocation of registers not allowed')
        if nodedesc.start_offset != 0:
            raise NotImplementedError('Start offset unsupported for registers')

        # ------------------- Declaration & Initialization -------------------
        arrsize = nodedesc.total_size
        array_ctype = '{nodedesc.dtype.ctype} *'
        init_clause = ' = {0}' if node.setzero else ''

        declaration_stream.write(f'{nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}]{init_clause};\n', cfg,
                                 state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        # Adjust offset if needed
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f'({dataname} - {symbolic_to_cpp(nodedesc.start_offset)})'

        # Remove declaration info
        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (
                dtypes.AllocationLifetime.Global,
                dtypes.AllocationLifetime.Persistent,
                dtypes.AllocationLifetime.External,
            )
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        # Special case: Stream
        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError('stream code is not implemented in ExperimentalCUDACodeGen (yet)')

        # Special case: View - no deallocation
        if isinstance(nodedesc, dace.data.View):
            return

        # Main deallocation logic by storage type
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if not nodedesc.pool:  # If pooled, will be freed somewhere else
                callsite_stream.write(f'DACE_GPU_CHECK({self.backend}Free({dataname}));\n', cfg, state_id, node)

        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            if nodedesc.dtype == dtypes.gpuStream_t:
                return
            callsite_stream.write(f'DACE_GPU_CHECK({self.backend}FreeHost({dataname}));\n', cfg, state_id, node)

        elif nodedesc.storage in {dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register}:
            # No deallocation needed
            return

        else:
            raise NotImplementedError(f'Deallocation not implemented for storage type: {nodedesc.storage.name}')

    def get_generated_codeobjects(self):

        # My comment: first part creates the header and stores it in a object property
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'cuda')

        # The GPU stream array is set to have a persistent allocation lifetime (see preprocess GPU stream pipeline).
        # Thus the definition of the GPU stream array in the state struct and the access to it is handled elsewhere and
        # in several different files (e.g., framecode.py, cpu.py, cpp.py). For the sake of consistency, we initialize it 
        # as it is expected in the other modules. I.e. prepend with an ID for all SDFGs it is defined.
        # Note that all the different variable names point to the same GPU stream array.
        init_gpu_stream_vars = ""
        gpu_stream_array_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(",")[0]
        for csdfg, name, desc in self._global_sdfg.arrays_recursive(include_nested_data=True):
            if name == gpu_stream_array_name and desc.lifetime == dtypes.AllocationLifetime.Persistent:
                gpu_stream_field_name = f'__{csdfg.cfg_id}_{name}'
                init_gpu_stream_vars += f"__state->{gpu_stream_field_name} = __state->gpu_context->streams;\n"
                init_gpu_stream_vars += f"    "

        # My comment: takes codeblocks and transforms it nicely to code
        initcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if 'cuda' in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code['cuda']), sd)
        initcode.write(self._initcode.getvalue())

        # My comment: takes codeblocks and transforms it nicely to code- probably same as before now for exit code
        exitcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if 'cuda' in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code['cuda']), sd)
        exitcode.write(self._exitcode.getvalue())

        # My comment: Uses GPU backend (NVIDIA or AMD) to get correct header files
        if self.backend == 'cuda':
            backend_header = 'cuda_runtime.h'
        elif self.backend == 'hip':
            backend_header = 'hip/hip_runtime.h'
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

        # My comment: Seems to get all function params, needed for later
        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        #My comment looks life Memory information
        pool_header = ''
        if self.has_pool:
            poolcfg = Config.get('compiler', 'cuda', 'mempool_release_threshold')
            pool_header = f'''
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = {poolcfg if poolcfg != -1 else 'UINT64_MAX'};
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
'''

        # My comment: Looks like a "base" template, where more details will probably be added later
        self._codeobject.code = """
#include <{backend_header}>
#include <dace/dace.h>

// New, cooperative groups and asnyc copy
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

{file_header}

DACE_EXPORTED int __dace_init_experimental_cuda({sdfg_state_name} *__state{params});
DACE_EXPORTED int __dace_exit_experimental_cuda({sdfg_state_name} *__state);

{other_globalcode}

int __dace_init_experimental_cuda({sdfg_state_name} *__state{params}) {{
    int count;

    // Check that we are able to run {backend} code
    if ({backend}GetDeviceCount(&count) != {backend}Success)
    {{
        printf("ERROR: GPU drivers are not configured or {backend}-capable device "
               "not found\\n");
        return 1;
    }}
    if (count == 0)
    {{
        printf("ERROR: No {backend}-capable devices found\\n");
        return 2;
    }}

    // Initialize {backend} before we run the application
    float *dev_X;
    DACE_GPU_CHECK({backend}Malloc((void **) &dev_X, 1));
    DACE_GPU_CHECK({backend}Free(dev_X));

    {pool_header}

    __state->gpu_context = new dace::cuda::Context({nstreams}, {nevents});

    // Create {backend} streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_GPU_CHECK({backend}StreamCreateWithFlags(&__state->gpu_context->internal_streams[i], {backend}StreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        DACE_GPU_CHECK({backend}EventCreateWithFlags(&__state->gpu_context->events[i], {backend}EventDisableTiming));
    }}

    {other_gpustream_init}

    {initcode}

    return 0;
}}

int __dace_exit_experimental_cuda({sdfg_state_name} *__state) {{
    {exitcode}

    // Synchronize and check for CUDA errors
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    if (__err == 0)
        __err = static_cast<int>({backend}DeviceSynchronize());

    // Destroy {backend} streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        DACE_GPU_CHECK({backend}StreamDestroy(__state->gpu_context->internal_streams[i]));
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        DACE_GPU_CHECK({backend}EventDestroy(__state->gpu_context->events[i]));
    }}

    delete __state->gpu_context;
    return __err;
}}

DACE_EXPORTED bool __dace_gpu_set_stream({sdfg_state_name} *__state, int streamid, gpuStream_t stream)
{{
    if (streamid < 0 || streamid >= {nstreams})
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}}

DACE_EXPORTED void __dace_gpu_set_all_streams({sdfg_state_name} *__state, gpuStream_t stream)
{{
    for (int i = 0; i < {nstreams}; ++i)
        __state->gpu_context->streams[i] = stream;
}}

{localcode}
""".format(params=params_comma,
           sdfg_state_name=mangle_dace_state_struct_name(self._global_sdfg),
           initcode=initcode.getvalue(),
           exitcode=exitcode.getvalue(),
           other_globalcode=self._globalcode.getvalue(),
           localcode=self._localcode.getvalue(),
           file_header=fileheader.getvalue(),
           nstreams=self._gpu_stream_manager.num_gpu_streams,
           nevents=self._gpu_stream_manager.num_gpu_events,
           other_gpustream_init=init_gpu_stream_vars,
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg)

        return [self._codeobject]

    #######################################################################
    # Compilation Related

    @staticmethod
    def cmake_options():
        options = []

        # Override CUDA toolkit
        if Config.get('compiler', 'cuda', 'path'):
            options.append("-DCUDA_TOOLKIT_ROOT_DIR=\"{}\"".format(
                Config.get('compiler', 'cuda', 'path').replace('\\', '/')))

        # Get CUDA architectures from configuration
        backend = common.get_gpu_backend()
        if backend == 'cuda':
            cuda_arch = Config.get('compiler', 'cuda', 'cuda_arch').split(',')
            cuda_arch = [ca for ca in cuda_arch if ca is not None and len(ca) > 0]

            cuda_arch = ';'.join(cuda_arch)
            options.append(f'-DDACE_CUDA_ARCHITECTURES_DEFAULT="{cuda_arch}"')

            flags = Config.get("compiler", "cuda", "args")
            options.append("-DCMAKE_CUDA_FLAGS=\"{}\"".format(flags))

        if backend == 'hip':
            hip_arch = Config.get('compiler', 'cuda', 'hip_arch').split(',')
            hip_arch = [ha for ha in hip_arch if ha is not None and len(ha) > 0]

            flags = Config.get("compiler", "cuda", "hip_args")
            flags += ' ' + ' '.join(
                '--offload-arch={arch}'.format(arch=arch if arch.startswith("gfx") else "gfx" + arch)
                for arch in hip_arch)
            options.append("-DEXTRA_HIP_FLAGS=\"{}\"".format(flags))

        if Config.get('compiler', 'cpu', 'executable'):
            host_compiler = make_absolute(Config.get("compiler", "cpu", "executable"))
            options.append("-DCUDA_HOST_COMPILER=\"{}\"".format(host_compiler))

        return options

    #######################################################################
    # Callback to CPU codegen

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        self._cpu_codegen.define_out_memlet(sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
                                            callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)


#########################################################################
# helper class
# This one is closely linked to the ExperimentalCUDACodeGen. In fact,
# it only exists to not have to much attributes and methods in the ExperimentalCUDACodeGen
# and to group Kernel specific methods & information. Thus, KernelSpec should remain in this file
class KernelSpec:
    """
    A helper class to encapsulate information required for working with kernels.
    This class provides a structured way to store and retrieve kernel parameters.
    """

    def __init__(self, cudaCodeGen: ExperimentalCUDACodeGen, sdfg: SDFG, cfg: ControlFlowRegion,
                 dfg_scope: ScopeSubgraphView, state_id: int):

        # Get kernel entry/exit nodes and current state
        kernel_map_entry: nodes.MapEntry  = dfg_scope.source_nodes()[0]
        kernel_parent_state: SDFGState = cfg.state(state_id)

        self._kernel_map_entry: nodes.MapEntry = kernel_map_entry
        self._kernels_state: SDFGState = kernel_parent_state

        # Kernel name
        self._kernel_name: str = f'{kernel_map_entry.map.label}_{cfg.cfg_id}_{kernel_parent_state.block_id}_{kernel_parent_state.node_id(kernel_map_entry)}'

        # Get and store kernel constants — needed for applying 'const' and updating defined 
        # constant variable types in the dispatcher (handled at GPU codegen)
        kernel_const_data = sdutil.get_constant_data(kernel_map_entry, kernel_parent_state)
        kernel_const_symbols = sdutil.get_constant_symbols(kernel_map_entry, kernel_parent_state)
        kernel_constants = kernel_const_data | kernel_const_symbols
        self._kernel_constants: Set[str] = kernel_constants

        # Retrieve arguments required for the kernels subgraph
        arglist: Dict[str, dt.Data] = kernel_parent_state.scope_subgraph(kernel_map_entry).arglist()
        self._arglist = arglist

        # save _in_device_code value for restoring later
        restore_in_device_code = cudaCodeGen._in_device_code

        # Certain args are called in the CUDA/HIP file or kernel funcion, in which the pointer name of the args are different
        cudaCodeGen._in_device_code = True
        self._args_as_input = [ptr(name, data, sdfg, cudaCodeGen._frame) for name, data in arglist.items()]
        self._args_typed = [('const ' if name in kernel_constants else '') + data.as_arg(name=name) for name, data in arglist.items()]

        # Args for the kernel wrapper function
        cudaCodeGen._in_device_code = False

        # Gather GPU stream information:
        # - Use the connector name when passing the stream to the kernel
        # - Use the configured variable name (from Config) in the wrapper’s function signature
        #   (this same name is also used when invoking {backend}LaunchKernel inside the wrapper)
        gpustream_var_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[1]
        gpustream_input = [e for e in dace.sdfg.dynamic_map_inputs(kernel_parent_state, kernel_map_entry) if e.src.desc(sdfg).dtype == dtypes.gpuStream_t]
        if len(gpustream_input) > 1:
            raise ValueError(f"There can not be more than one GPU stream assigned to a kernel, but {len(gpustream_input)} were assigned.")
        
        # Final wrapper arguments:
        # - State struct (__state)
        # - Original kernel args
        # - GPU stream
        self._kernel_wrapper_args_as_input = (['__state']
                                              + [ptr(name, data, sdfg, cudaCodeGen._frame) for name, data in arglist.items()] 
                                              + [str(gpustream_input[0].dst_conn)])
        self._kernel_wrapper_args_typed = ([f'{mangle_dace_state_struct_name(cudaCodeGen._global_sdfg)} *__state'] 
                                           + [('const ' if name in kernel_constants else '') + data.as_arg(name=name) for name, data in arglist.items()] 
                                           + [f"gpuStream_t {gpustream_var_name}"])

        cudaCodeGen._in_device_code = restore_in_device_code

        # The kernel's grid and block dimensions
        self._grid_dims, self._block_dims = cudaCodeGen._kernel_dimensions_map[kernel_map_entry]

        # C type of block, thread, and warp indices (as a string)
        self._gpu_index_ctype: str = self.get_gpu_index_ctype()

        # Warp size (backend-dependent)
        if cudaCodeGen.backend not in ['cuda', 'hip']:
            raise ValueError(f"Unsupported backend '{cudaCodeGen.backend}' in ExperimentalCUDACodeGen. "
                             "Only 'cuda' and 'hip' are supported.")

        warp_size_key = 'cuda_warp_size' if cudaCodeGen.backend == 'cuda' else 'hip_warp_size'
        self._warpSize = Config.get('compiler', 'cuda', warp_size_key)

    def get_gpu_index_ctype(self, config_key='gpu_index_type') -> str:
        """
        Retrieves the GPU index data type as a C type string (for thread, block, warp indices)
        from the configuration and if it matches a DaCe data type.

        Raises:
            ValueError: If the configured type does not match a DaCe data type.

        Returns:
            str:
                The C type string corresponding to the configured GPU index type.
                Used for defining thread, block, and warp indices in the generated code.
        """
        type_name = Config.get('compiler', 'cuda', config_key)
        dtype = getattr(dtypes, type_name, None)
        if not isinstance(dtype, dtypes.typeclass):
            raise ValueError(
                f'Invalid {config_key} "{type_name}" configured (used for thread, block, and warp indices): '
                'no matching DaCe data type found.\n'
                'Please use a valid type from dace.dtypes (e.g., "int32", "uint64").')
        return dtype.ctype

    @property
    def kernel_constants(self) -> Set[str]:
        """Returns the kernel's constant data and symbols."""
        return self._kernel_constants
    
    @property
    def kernel_name(self) -> list[str]:
        """Returns the kernel (function's) name."""
        return self._kernel_name

    @property
    def kernel_map_entry(self) -> nodes.MapEntry:
        """
        Returns the entry node of the kernel, which is a MapEntry node
        scheduled with dace.dtypes.ScheduleType.GPU_Device.
        """
        return self._kernel_map_entry

    @property
    def kernel_map(self) -> nodes.Map:
        """Returns the kernel's map node."""
        return self._kernel_map_entry.map
    
    @property
    def arglist(self) -> Dict[str, dt.Data]:
        """
        Returns a dictionary of arguments for the kernel's subgraph,  
        mapping each data name to its corresponding data descriptor.
        """
        return self._arglist

    @property
    def args_as_input(self) -> list[str]:
        """
        Returns the kernel function arguments formatted for use as inputs
        when calling/launching the kernel function.
        """
        return self._args_as_input

    @property
    def args_typed(self) -> list[str]:
        """
        Returns the typed kernel function arguments suitable for declaring
        the kernel function. Each argument includes its corresponding data type.
        """
        return self._args_typed

    @property
    def kernel_wrapper_args_as_input(self) -> list[str]:
        """
        Returns the argument names passed to the kernel wrapper function.

        The kernel wrapper is a function defined in the CUDA/HIP code that is called
        from the CPU code and is responsible for launching the kernel function.
        """
        return self._kernel_wrapper_args_as_input

    @property
    def kernel_wrapper_args_typed(self) -> list[str]:
        """
        Returns the typed arguments used to declare the kernel wrapper function.

        The kernel wrapper is defined in the CUDA/HIP code, called from the CPU side,
        and is responsible for launching the actual kernel function.
        """
        return self._kernel_wrapper_args_typed

    @property
    def grid_dims(self) -> list:
        """Returns the grid dimensions of the kernel."""
        return self._grid_dims

    @property
    def block_dims(self) -> list:
        """Returns the block dimensions of the kernel."""
        return self._block_dims

    @property
    def warpSize(self) -> int:
        """
        Returns the warp size used in this kernel.
        This value depends on the selected backend (CUDA or HIP)
        and is retrieved from the configuration.
        """
        return self._warpSize

    @property
    def gpu_index_ctype(self) -> str:
        """
        Returns the C data type used for GPU indices (thread, block, warp)
        in generated code. This type is determined by the 'gpu_index_type'
        setting in the configuration and matches with a DaCe typeclass.
        """
        return self._gpu_index_ctype
