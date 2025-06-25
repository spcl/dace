# Standard library imports
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Type

# DaCe core imports
import dace
from dace import dtypes, subsets, symbolic

from dace.config import Config

# DaCe SDFG imports
from dace.sdfg import SDFG, ScopeSubgraphView, nodes, SDFGState
from dace.sdfg.state import ControlFlowRegion

# DaCe codegen imports
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.dispatcher import DefinedType, TargetDispatcher

# DaCe transformation imports
from dace.transformation import helpers

# Experimental CUDA imports
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen, KernelSpec
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import (
    symbolic_to_cpp, 
    get_cuda_dim, 
    product
)


#----------------------------------------------------------------------------------
# GPU Scope Generation Strategies
#----------------------------------------------------------------------------------

class ScopeGenerationStrategy(ABC):
    """Base strategy for generating GPU scope code"""
    
    def __init__(self, codegen: ExperimentalCUDACodeGen):
        self.codegen: ExperimentalCUDACodeGen = codegen
        self._dispatcher: TargetDispatcher = codegen._dispatcher
        self._current_kernel_spec: KernelSpec = codegen._current_kernel_spec
        
    @abstractmethod
    def applicable(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> bool:
        raise NotImplementedError('Abstract class')
    
    @abstractmethod
    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        raise NotImplementedError('Abstract class')


class KernelScopeGenerator(ScopeGenerationStrategy):
   
    def __init__(self, codegen: ExperimentalCUDACodeGen):
        super().__init__(codegen)

    def applicable(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> bool:
        
        node = dfg_scope.source_nodes()[0]
        schedule_type = node.map.schedule

        # This strategy starts kernel code generation and is only valid if
        # the outermost (first) GPU schedule is of type GPU_Device.
        applicable = schedule_type == dtypes.ScheduleType.GPU_Device
        return applicable
    
    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                 state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        

        # Generate kernel function signature
        self._generate_kernel_signature(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

        # Generate kernel body
        with ScopeManager(frame_codegen=self.codegen._frame, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id,
                          function_stream=function_stream, callsite_stream=callsite_stream, comment="Kernel scope") as scope_manager:
            

            # ----------------- Initialize Kernel Scope Constructs -----------------------

            self._generate_kernel_initialization(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
        
            # ----------------- Retrieve kernel configuration -----------------------

            kernel_spec = self._current_kernel_spec
            kernel_entry_node = kernel_spec._kernel_entry_node # = dfg_scope.source_nodes()[0]
            kernel_map = kernel_spec.kernel_map
            has_tbmap = kernel_spec.has_tbmap
            kernel_block_dims = self._current_kernel_spec.block_dims


            # ----------------- Kernel/Map Range Preprocessing -----------------------

            reversed_kernel_range = kernel_map.range[::-1] # also reverse it 
            kernel_range = subsets.Range(reversed_kernel_range)
            kernel_dimensions = len(kernel_range)
            kernel_dim_sizes = kernel_range.size()


            # ----------------- Set up symbolic index expressions -----------------------

            symbolic_indices = [ symbolic.symbol(f'__SYM_IDX{dim}', nonnegative=True, integer=True) for dim in range(kernel_dimensions)]
            symbolic_index_bounds = [ idx + block_dim - 1 for idx, block_dim in zip(symbolic_indices, kernel_block_dims)]
            symbolic_coordinates = kernel_range.coord_at(symbolic_indices)


            # ----------------- Generate Thread or Block index Definitions -----------------------


            thread_id_ctype = kernel_spec.gpu_index_ctype # Data type of CUDA thread/block indices


            # In case there is no ThreadBlock map used in a submap, the map variables will
            # be mapped to thread IDs instead of block IDs
            for dim in range(kernel_dimensions):

                var_name = kernel_map.params[-dim - 1] # also reverse it here!

                # Compute index expressions for up to 3 dimensions (x, y, z)
                if dim < 3:
                    if has_tbmap:
                        index_expr = f'blockIdx.{get_cuda_dim(dim)}'
                    else:
                        index_expr = f'(blockIdx.{get_cuda_dim(dim)} * {symbolic_to_cpp(kernel_block_dims[dim])} + threadIdx.{get_cuda_dim(dim)})'

                    # Delinearize third dimension if more than 3D (used in 3D+ mapping)
                    if dim == 2 and kernel_dimensions > 3:
                        tail_prod = product(kernel_dim_sizes[3:])
                        index_expr = f"({index_expr} / ({symbolic_to_cpp(tail_prod)}))"

                else:  # Handle dimensions beyond the third (delinearize and modulo)
                    if has_tbmap:
                        index_expr = f'blockIdx.z'
                    else:
                        index_expr = f'(blockIdx.z * {symbolic_to_cpp(kernel_block_dims[2])} + threadIdx.z)'

                    tail_prod = product(kernel_dim_sizes[dim + 1:])
                    index_expr = (f"({index_expr} / ({symbolic_to_cpp(tail_prod)})) % ({symbolic_to_cpp(kernel_dim_sizes[dim])})")


                # Define thread/Block index
                var_def = symbolic_to_cpp(symbolic_coordinates[dim]).replace(f'__SYM_IDX{dim}', index_expr)
                callsite_stream.write(f'{thread_id_ctype} {var_name} = {var_def};', cfg, state_id, kernel_entry_node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, thread_id_ctype) 


            # ----------------- Guard Conditions for Block Execution -----------------------

            if not has_tbmap:
                minels = kernel_range.min_element()
                maxels = kernel_range.max_element()

                for dim, (var_name, start, end) in enumerate(zip(kernel_map.params[::-1], minels, maxels)):
                    condition = ''

                    # Optimize conditions if they are always true
                    if dim >= 3 or (symbolic_indices[dim] >= start) != True:
                        condition += f'{var_name} >= {symbolic_to_cpp(start)}' 

                    if (dim >= 3 or ((symbolic_index_bounds[dim] < end) != False 
                            and ((symbolic_index_bounds[dim] % kernel_block_dims[dim]) != 0) == True) or (kernel_block_dims[dim] > end) == True):

                        if len(condition) > 0:
                            condition += ' && '
                        condition += f'{var_name} < {symbolic_to_cpp(end + 1)}'

                    if len(condition) > 0:
                        scope_manager.open(condition=condition)


            # ----------------- Dispatch Subgraph code generation -----------------------

            self._dispatcher.dispatch_subgraph(sdfg, cfg, dfg_scope, state_id, function_stream, 
                                               callsite_stream, skip_entry_node=True)

    def _generate_kernel_signature(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
            
        kernel_name = self._current_kernel_spec.kernel_name
        kernel_args = self._current_kernel_spec.args_typed
        block_dims = self._current_kernel_spec.block_dims
        node = dfg_scope.source_nodes()[0]

        # Conditionally add __launch_bounds__ for block size optimization.
        launch_bounds = ''
        if node.gpu_launch_bounds != '-1':
            if node.gpu_launch_bounds == "0":
                if not any(symbolic.issymbolic(b) for b in block_dims):
                    launch_bounds = f'__launch_bounds__({product(block_dims)})'
            else:
                launch_bounds = f'__launch_bounds__({node.gpu_launch_bounds})'


        # Emit kernel function signature
        callsite_stream.write(
            f'__global__ void {launch_bounds} {kernel_name}({", ".join(kernel_args)}) ',
            cfg, state_id, node
        )     

    def _generate_kernel_initialization(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                                        state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        
        """
        NOTE: Under construction
        Tell yakup:
        1. This is as far as I know really cuda specific- maybe I should raise an error if wrong backend (HIP) is used
        2. What about the shared state allocation? Is it correct to tell about this allocation? generally, did I
           tell the dispatcher everything correctly?
        """

        # Skip this if there are no metada, nothing to initialize
        metadata = sdfg.metadata
        if metadata == None:
            return
        
        node = dfg_scope.source_nodes()[0]

        callsite_stream.write(f"\n", cfg, state_id, node)
        # initialize block group using coopertive groups
        tblock_obj_name = Config.get('compiler', 'cuda', 'current_thread_block_name')
        tblock_obj_ctype = "auto"
        callsite_stream.write(f"{tblock_obj_ctype} {tblock_obj_name} = cg::this_thread_block();\n", cfg, state_id, node)
        self._dispatcher.defined_vars.add(tblock_obj_name, DefinedType.Object, tblock_obj_ctype)

        # initialize pipeline 
        pipelines = dict()
        for node_guid, node_meta in metadata.items():
            pipelines = node_meta.get("pipelines", {})
            for pipeline_name, pipeline_info in pipelines.items():
                pipelines[pipeline_name] = pipeline_info["pipeline_depth"]
            

    
        for pipeline_name, pipeline_depth in pipelines.items():
            callsite_stream.write(f"\n", cfg, state_id, node)
            # initialize pipeline depth scalar
            depth_name = f"pipeline_depth_{pipeline_name}"
            depth_ctype = "const uint"
            callsite_stream.write(f"{depth_ctype} {depth_name} = {pipeline_depth};\n", cfg, state_id, node)
            self._dispatcher.defined_vars.add(depth_name, DefinedType.Scalar, depth_ctype)

            # allocate shared pipeline state 
            shared_state_name = f"shared_state_{pipeline_name}"
            shared_state_ctype = f"cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, {depth_name}>"
            callsite_stream.write(f" __shared__ {shared_state_ctype} {shared_state_name};\n")
            self._dispatcher.declared_arrays.add(shared_state_name, DefinedType.Pointer, shared_state_ctype)

            # intialize the pipeline
            pipeline_ctype = "auto"
            callsite_stream.write(f"{pipeline_ctype} {pipeline_name} = cuda::make_pipeline({tblock_obj_name}, &{shared_state_name});\n", cfg, state_id, node)
            self._dispatcher.defined_vars.add(pipeline_name, DefinedType.Object, pipeline_ctype)
    
        callsite_stream.write(f"\n", cfg, state_id, node)


class ThreadBlockScopeGenerator(ScopeGenerationStrategy):
    
    def __init__(self, codegen: ExperimentalCUDACodeGen):
        super().__init__(codegen)
        
    def applicable(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> bool:
        
        node = dfg_scope.source_nodes()[0]
        applicable = node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock

        return applicable
    
    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                 state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        # NOTE: not my code, but my insights. Approval for commenting this needed
        with ScopeManager(frame_codegen=self.codegen._frame, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id,
                          function_stream=function_stream, callsite_stream=callsite_stream, comment="ThreadBlock Scope") as scope_manager:
            
            node = dfg_scope.source_nodes()[0]
            scope_map = node.map


            # ----------------- Map Range Preprocessing -----------------------

            # Reverse range for better performance (e.g. memory coalescing)
            reversed_scope_range = scope_map.range[::-1]
            map_range = subsets.Range(reversed_scope_range)
            map_dimensions = len(map_range)
            map_dim_sizes = map_range.size()

            kernel_block_dims = self._current_kernel_spec.block_dims


            # ----------------- Symbolic Index Expressions -----------------------

            symbolic_indices = [ symbolic.symbol(f'__SYM_IDX{dim}', nonnegative=True, integer=True) for dim in range(map_dimensions)]
            symbolic_index_bounds = [idx + (block_dim * rng[2]) - 1 for idx, block_dim, rng in zip(symbolic_indices, kernel_block_dims, map_range)]
            symbolic_coordinates = map_range.coord_at(symbolic_indices)


            # ----------------- Generate Index Variable Definitions -----------------------

            # Get the block's index dace data type
            block_id_ctype = self._current_kernel_spec.gpu_index_ctype

            for dim in range(map_dimensions):
                var_name = scope_map.params[-dim - 1] # also reverse it here!

                if dim < 3:
                    # First three dimensions: direct mapping or partial delinearization
                    if dim == 2 and map_dimensions > 3:
                        tail_prod = product(map_dim_sizes[3:])
                        base_expr = f"(threadIdx.z / ({symbolic_to_cpp(tail_prod)}))"
                    else:
                        base_expr = f"threadIdx.{get_cuda_dim(dim)}"
                else:
                    # Dimensions beyond the third: full delinearization
                    tail_prod = product(map_dim_sizes[dim + 1:])
                    base_expr = (f"(threadIdx.z / ({symbolic_to_cpp(tail_prod)})) % "f"({symbolic_to_cpp(map_dim_sizes[dim])})")


                var_def = symbolic_to_cpp(symbolic_coordinates[dim]).replace(f'__SYM_IDX{dim}', base_expr)
                callsite_stream.write(f'{block_id_ctype} {var_name} = {var_def};', cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, block_id_ctype) 


            # ----------------- Guard Conditions for Block Execution -----------------------

            # Generate conditions for this block's execution using min and max
            # element, e.g. skipping out-of-bounds threads in trailing block
            minels = map_range.min_element()
            maxels = map_range.max_element()
            for dim, (var_name, start, end) in enumerate(zip(scope_map.params[::-1], minels, maxels)):

                # Optimize conditions if they are always true
                #############################################

                condition = ''

                # Block range start
                if dim >= 3 or (symbolic_indices[dim] >= start) != True:
                    condition += f'{var_name} >= {symbolic_to_cpp(start)}' 

                # Special case: block size is exactly the range of the map (0:b)
                if dim >= 3:
                    skipcond = False
                else:
                    skipcond = symbolic_index_bounds[dim].subs({symbolic_indices[dim]: start}) == end

                # Block range end
                if dim >= 3 or (not skipcond and (symbolic_index_bounds[dim] < end) != True):
                    if len(condition) > 0:
                        condition += ' && '
                    condition += f'{var_name} < {symbolic_to_cpp(end + 1)}'

                # Emit condition in code if any
                if len(condition) > 0:
                    scope_manager.open(condition=condition)


            # ----------------- Dispatch Subgraph code generation -----------------------

            self._dispatcher.dispatch_subgraph(sdfg, cfg, dfg_scope, state_id, function_stream,
                                               callsite_stream, skip_entry_node=True)


class WarpScopeGenerator(ScopeGenerationStrategy):
    
    def __init__(self, codegen: ExperimentalCUDACodeGen):
        super().__init__(codegen)
        
    def applicable(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> bool:
        
        node = dfg_scope.source_nodes()[0]
        applicable = node.map.schedule == dtypes.ScheduleType.GPU_Warp

        return applicable
    
    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                 state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        with ScopeManager(frame_codegen=self.codegen._frame, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id, 
                          function_stream=function_stream, callsite_stream=callsite_stream, comment="WarpLevel Scope") as scope_manager:


            # Get kernel specifications
            kernel_spec = self._current_kernel_spec
            block_dims = kernel_spec.block_dims
            warpSize = kernel_spec.warpSize

            state_dfg = cfg.state(state_id)
            node = dfg_scope.source_nodes()[0]
            scope_map = node.map

            map_range = subsets.Range(scope_map.range[::-1])  # Reversed for potential better performance
            warp_dim = len(map_range)
            
            # The following sizes and bounds are be symbolic
            num_threads_in_block = product(block_dims) 
            warp_dim_bounds = [max_elem + 1 for max_elem in map_range.max_element()]
            num_warps = product(warp_dim_bounds)


            # The C type used to define the (flat) threadId and warpId variables
            ids_ctype = kernel_spec.gpu_index_ctype

            # ----------------- Guard checks -----------------------

            
            # handles checks either at compile time or runtime (i.e. checks in the generated code)
            self._handle_GPU_Warp_scope_guards(state_dfg, node, map_range, warp_dim, num_threads_in_block, num_warps,
                                               callsite_stream, scope_manager)
                        


            # ----------------- Define (flat) Thread ID within Block -----------------------

            flattened_terms = []

            for i, dim_size in enumerate(block_dims):

                if dim_size == 1:
                    continue

                dim = get_cuda_dim(i)
                stride = [f"{block_dims[j]}" for j in range(i) if block_dims[j] > 1]
                idx_expr = " * ".join(stride + [f"threadIdx.{get_cuda_dim(i)}"]) if stride else f"threadIdx.{dim}"
                flattened_terms.append(idx_expr)


            joined_terms = " + ".join(flattened_terms)
            flat_thread_idx_expr = f"({joined_terms})" if len(flattened_terms) > 1 else joined_terms

            threadID_name = 'ThreadId_%s_%d_%d_%d' % (scope_map.label, cfg.cfg_id, state_dfg.block_id, state_dfg.node_id(node))

            callsite_stream.write(f"{ids_ctype} {threadID_name} = ({flat_thread_idx_expr}) / {warpSize};", cfg, state_id, node)
            self._dispatcher.defined_vars.add(threadID_name, DefinedType.Scalar, ids_ctype)


            
            # ----------------- Compute Map indices (= Warp indices) -----------------------

            for i in range(warp_dim):
                var_name = scope_map.params[-i - 1]  # reverse order
                previous_sizes = warp_dim_bounds[:i]

                if len(previous_sizes) > 0:
                    divisor = product(previous_sizes)
                    expr = f"({threadID_name} / {divisor}) % {warp_dim_bounds[i]}"
                else:
                    expr = f"{threadID_name} % {warp_dim_bounds[i]}"

                callsite_stream.write(f"{ids_ctype} {var_name} = {expr};", cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ids_ctype)



            # ----------------- Guard Conditions for Warp Execution -----------------------


            if num_warps * warpSize != num_threads_in_block:
                condition = f'{threadID_name} < {num_warps}'
                scope_manager.open(condition)

            warp_range = [(start, end + 1, stride) for start, end, stride in map_range.ranges]

            for dim, (var_name, (start, _, stride)) in enumerate(zip(scope_map.params[::-1], warp_range)):
                
                condition_terms = []
                
                if start != 0:
                    condition_terms.append(f"{var_name} >= {start}")
                
                if stride != 1:
                    expr = var_name if start == 0 else f"({var_name} - {start})"
                    condition_terms.append(f'{expr} % {stride} == 0' )
                
                if condition_terms:
                    condition = " && ".join(condition_terms)
                    scope_manager.open(condition)


            # ----------------- Dispatch Subgraph code generation -----------------------


            self._dispatcher.dispatch_subgraph(
                sdfg, cfg, dfg_scope, state_id, function_stream,
                callsite_stream, skip_entry_node=True
            )

    def _handle_GPU_Warp_scope_guards(self, state_dfg: SDFGState, node: nodes.MapEntry, map_range: subsets.Range,
                                      warp_dim: int, num_threads_in_block, num_warps, kernel_stream: CodeIOStream,
                                      scope_manager: 'ScopeManager'):
        
        #TODO: Move them to sdfg validation as well if possible

        # Get warpSize from the kernel specification
        warpSize = self._current_kernel_spec.warpSize
        
        parent_map, _ = helpers.get_parent_map(state_dfg, node)
        if parent_map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            raise ValueError("GPU_Warp map must be nested within a GPU_ThreadBlock map.")

        if warp_dim > 3:
            raise NotImplementedError("GPU_Warp maps are limited to 3 dimensions.")


        # Guard against invalid thread/block configurations.
        # - For concrete (compile-time) values, raise Python errors early.
        # - For symbolic values, insert runtime CUDA checks (guards) into the generated kernel.
        #   These will emit meaningful error messages and abort execution if violated.
        if isinstance(num_threads_in_block, symbolic.symbol):
            condition = (
                f"{num_threads_in_block} % {warpSize} != 0 || "
                f"{num_threads_in_block} > 1024 || "
                f"{num_warps} * {warpSize} > {num_threads_in_block}"
            )
            kernel_stream.write(f"""\
            if ({condition}) {{
                printf("CUDA error:\\n"
                    "1. Block must be a multiple of {warpSize} threads (DaCe requirement for GPU_Warp scheduling).\\n"
                    "2. Block size must not exceed 1024 threads (CUDA hardware limit).\\n"
                    "3. Number of warps x {warpSize} must fit in the block (otherwise logic is unclear).\\n");
                asm("trap;");
            }}
            """)

        else:
            if isinstance(num_warps, symbolic.symbol):
                condition = f"{num_warps} * {warpSize} > {num_threads_in_block}"
                scope_manager.open(condition=condition)

            elif num_warps * warpSize > num_threads_in_block:
                raise ValueError(f"Invalid configuration: {num_warps} warps x {warpSize} threads exceed "
                                f"{num_threads_in_block} threads in the block.")

            if num_threads_in_block % warpSize != 0:
                raise ValueError(f"Block must be a multiple of {warpSize} threads for GPU_Warp scheduling "
                                    f"(got {num_threads_in_block}).")

            if num_threads_in_block > 1024:
                raise ValueError("CUDA does not support more than 1024 threads per block (hardware limit).")
            
        
        for min_element in map_range.min_element():
            if isinstance(min_element, symbolic.symbol):
                kernel_stream.write(f'if ({min_element} < 0) {{\n'
                                    f'    printf("Runtime error: Warp ID symbol {min_element} must be non-negative.\\n");\n'
                                    f'    asm("trap;");\n'
                                    f'}}\n')
            elif min_element < 0:
                raise ValueError(f"Warp ID value {min_element} must be non-negative.")


#----------------------------------------------------------------------------------
# Scope Manager, handling brackets and allocation/deallocation of arrays in Scopes
#----------------------------------------------------------------------------------

class ScopeManager:
    """
    A helper class to manage opening and closing brackets in a structured way using the 'with' statement.
    This class simplifies the process of correctly opening and closing brackets. It also supports an optional
    debug mode to include comments in the generated code, which can help with debugging and understanding
    the code structure.
    """

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: SDFG,
                 cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int, 
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream, comment: str = None,
                 debug: bool = False): 
        """
        Initializes the KernelScopeManager.

        :param frame_codegen: The frame codegenerator used for allocation and deallocation of arrays in scopes
        :param sdfg: The SDFG instance for context.
        :param cfg: The ControlFlowRegion instance for context.
        :param dfg_scope: The ScopeSubgraphView instance for context.
        :param state_id: The ID of the current state for context.
        :param function_stream: The CodeIOStream for function-level code.
        :param callsite_stream: The CodeIOStream for callsite-level code.
        :param comment: A descriptive comment explaining the purpose of the code block being opened. Default is None.
        :param debug: Whether to include debug comments in the output. Defaults to False.
        """
        self.frame_codegen = frame_codegen
        self.sdfg = sdfg
        self.cfg = cfg
        self.dfg_scope = dfg_scope
        self.state_id = state_id
        self.function_stream = function_stream
        self.callsite_stream = callsite_stream
        self.comment = comment
        self.debug = debug
        self._opened = 0

        self.entry_node = self.dfg_scope.source_nodes()[0]
        self.exit_node = self.dfg_scope.sink_nodes()[0]

    def __enter__(self):
        """
        Writes the opening bracket to the stream and allocates arrays in scope.
        """
        self.open()
        self.frame_codegen.allocate_arrays_in_scope(
            self.sdfg, self.cfg, self.entry_node, self.function_stream, self.callsite_stream
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deallocates arrays in scope and writes the closing brackets to the stream.
        """
        self.frame_codegen.deallocate_arrays_in_scope(
            self.sdfg, self.cfg, self.entry_node, self.function_stream, self.callsite_stream
        )
        for i in range(self._opened):
            line = "}"
            if self.debug:
                line += f" // {self.comment} (close {i + 1})"
            self.callsite_stream.write(line, self.cfg, self.state_id, self.exit_node)

    def open(self, condition: str = None):
        """
        Opens a bracket. If a condition is given, emits 'if (condition) {', otherwise just '{'.
        Tracks the number of open brackets for closing later.

        :param condition: Optional condition for the opening bracket.
        """
        line = f"if ({condition}) {{" if condition else "{"
        if self.debug:
            line += f" // {self.comment} (open {self._opened + 1})"
        self.callsite_stream.write(line, self.cfg, self.state_id, self.entry_node)
        self._opened += 1


