# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from abc import ABC, abstractmethod

from dace import dtypes, subsets, symbolic
from dace.sdfg import SDFG, ScopeSubgraphView, nodes, SDFGState
from dace.sdfg.state import ControlFlowRegion
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.transformation import helpers
from dace.codegen.targets.cpp import sym2cpp
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen, KernelSpec
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import get_cuda_dim
from dace.transformation.dataflow.add_threadblock_map import product

#----------------------------------------------------------------------------------
# GPU Scope Generation Strategies
#----------------------------------------------------------------------------------


def _emit_dim_index_definitions(scope_map, axis: str, ctype: str, callsite_stream: CodeIOStream, cfg: ControlFlowRegion,
                                state_id: int, anchor_node, dispatcher: TargetDispatcher):
    """Emit ``{ctype} {var_name} = {expr};`` per map dim using the symbolic-
    coordinate substitution. ``axis`` is ``'blockIdx'`` (kernel scope) or
    ``'threadIdx'`` (thread-block scope). First three dims map directly to
    ``axis.{x|y|z}``; further dims delinearize off ``axis.z``. Returns
    ``(map_range, sym_indices, sym_coords)`` for callers that need the
    symbolic forms downstream (e.g. for guard conditions)."""
    map_range = subsets.Range(scope_map.range[::-1])  # reversed for memory coalescing
    dimensions = len(map_range)
    dim_sizes = map_range.size()
    sym_indices = [symbolic.symbol(f'__SYM_IDX{i}', nonnegative=True, integer=True) for i in range(dimensions)]
    sym_coords = map_range.coord_at(sym_indices)

    for dim in range(dimensions):
        var_name = scope_map.params[-dim - 1]  # reversed
        if dim < 3:
            expr = f"{axis}.{get_cuda_dim(dim)}"
            if dim == 2 and dimensions > 3:
                tail = product(dim_sizes[3:])
                expr = f"({expr} / ({sym2cpp(tail)}))"
        else:
            tail = product(dim_sizes[dim + 1:])
            expr = f"(({axis}.z / ({sym2cpp(tail)})) % ({sym2cpp(dim_sizes[dim])}))"
        var_def = sym2cpp(sym_coords[dim]).replace(f'__SYM_IDX{dim}', expr)
        callsite_stream.write(f'{ctype} {var_name} = {var_def};', cfg, state_id, anchor_node)
        dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ctype)

    return map_range, sym_indices, sym_coords


class ScopeGenerationStrategy(ABC):
    """Base strategy for generating GPU scope code.

    Subclasses set ``SCHEDULE`` to the schedule type they handle and
    ``SCOPE_COMMENT`` to the human-readable label used by ``ScopeManager``.
    The base ``applicable()`` matches ``SCHEDULE`` against the source
    MapEntry's schedule; subclasses implement ``generate()`` and reuse the
    ``_dispatch_and_deallocate`` tail.
    """

    SCHEDULE: dtypes.ScheduleType = None
    SCOPE_COMMENT: str = ""

    def __init__(self, codegen: ExperimentalCUDACodeGen):
        self.codegen: ExperimentalCUDACodeGen = codegen
        self._dispatcher: TargetDispatcher = codegen._dispatcher
        self._current_kernel_spec: KernelSpec = codegen._current_kernel_spec

    def applicable(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                   function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> bool:
        return dfg_scope.source_nodes()[0].map.schedule == self.SCHEDULE

    @abstractmethod
    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        raise NotImplementedError('Abstract class')

    def _dispatch_and_deallocate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                                 entry_node: nodes.MapEntry, function_stream: CodeIOStream,
                                 callsite_stream: CodeIOStream):
        """Common tail of every ``generate``: dispatch the inner subgraph,
        then deallocate scope-local arrays."""
        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)
        self.codegen._frame.deallocate_arrays_in_scope(sdfg, cfg, entry_node, function_stream, callsite_stream)


class KernelScopeGenerator(ScopeGenerationStrategy):

    SCHEDULE = dtypes.ScheduleType.GPU_Device
    SCOPE_COMMENT = "Kernel scope"

    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        self._generate_kernel_signature(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

        with ScopeManager(frame_codegen=self.codegen._frame,
                          sdfg=sdfg,
                          cfg=cfg,
                          dfg_scope=dfg_scope,
                          state_id=state_id,
                          function_stream=function_stream,
                          callsite_stream=callsite_stream,
                          comment=self.SCOPE_COMMENT) as scope_manager:

            # ----------------- Retrieve kernel configuration -----------------------

            kernel_spec = self._current_kernel_spec
            kernel_entry_node = kernel_spec.kernel_map_entry  # == dfg_scope.source_nodes()[0]

            # Without an inner ThreadBlock map the kernel-map variables bind
            # to thread indices instead — same blockIdx-based formulas.
            _emit_dim_index_definitions(kernel_spec.kernel_map, 'blockIdx', kernel_spec.gpu_index_ctype,
                                        callsite_stream, cfg, state_id, kernel_entry_node, self._dispatcher)

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, kernel_entry_node, function_stream, callsite_stream)

            self._dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, kernel_entry_node, function_stream,
                                          callsite_stream)

    def _generate_kernel_signature(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView,
                                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        kernel_name = self._current_kernel_spec.kernel_name
        kernel_args = self._current_kernel_spec.args_typed
        block_dims = self._current_kernel_spec.block_dims
        node = dfg_scope.source_nodes()[0]

        # Conditionally add __launch_bounds__ for block size optimization.
        min_warps_per_eu = ''
        if node.gpu_min_warps_per_eu is not None and node.gpu_min_warps_per_eu > 0:
            min_warps_per_eu = f',{node.gpu_min_warps_per_eu}'
        launch_bounds = ''
        if node.gpu_launch_bounds != '-1':
            if node.gpu_launch_bounds == "0":
                if not any(symbolic.issymbolic(b) for b in block_dims):
                    launch_bounds = f'__launch_bounds__({product(block_dims)}{min_warps_per_eu})'
            else:
                launch_bounds = f'__launch_bounds__({node.gpu_launch_bounds}{min_warps_per_eu})'

        # Emit kernel function signature
        callsite_stream.write(f'__global__ void {launch_bounds} {kernel_name}({", ".join(kernel_args)}) ', cfg,
                              state_id, node)


class ThreadBlockScopeGenerator(ScopeGenerationStrategy):

    SCHEDULE = dtypes.ScheduleType.GPU_ThreadBlock
    SCOPE_COMMENT = "ThreadBlock Scope"

    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        with ScopeManager(frame_codegen=self.codegen._frame,
                          sdfg=sdfg,
                          cfg=cfg,
                          dfg_scope=dfg_scope,
                          state_id=state_id,
                          function_stream=function_stream,
                          callsite_stream=callsite_stream,
                          comment=self.SCOPE_COMMENT) as scope_manager:

            node = dfg_scope.source_nodes()[0]
            scope_map = node.map
            kernel_block_dims = self._current_kernel_spec.block_dims

            map_range, symbolic_indices, _sym_coords = _emit_dim_index_definitions(
                scope_map, 'threadIdx', self._current_kernel_spec.gpu_index_ctype, callsite_stream, cfg, state_id, node,
                self._dispatcher)

            symbolic_index_bounds = [
                idx + (block_dim * rng[2]) - 1
                for idx, block_dim, rng in zip(symbolic_indices, kernel_block_dims, map_range)
            ]

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, callsite_stream)

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
                    condition += f'{var_name} >= {sym2cpp(start)}'

                # Special case: block size is exactly the range of the map (0:b)
                if dim >= 3:
                    skipcond = False
                else:
                    skipcond = symbolic_index_bounds[dim].subs({symbolic_indices[dim]: start}) == end

                # Block range end
                if dim >= 3 or (not skipcond and (symbolic_index_bounds[dim] < end) != True):
                    if len(condition) > 0:
                        condition += ' && '
                    condition += f'{var_name} < {sym2cpp(end + 1)}'

                # Emit condition in code if any
                if len(condition) > 0:
                    scope_manager.open(condition=condition)

            self._dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, node, function_stream, callsite_stream)


class WarpScopeGenerator(ScopeGenerationStrategy):

    SCHEDULE = dtypes.ScheduleType.GPU_Warp
    SCOPE_COMMENT = "WarpLevel Scope"

    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        with ScopeManager(frame_codegen=self.codegen._frame,
                          sdfg=sdfg,
                          cfg=cfg,
                          dfg_scope=dfg_scope,
                          state_id=state_id,
                          function_stream=function_stream,
                          callsite_stream=callsite_stream,
                          comment=self.SCOPE_COMMENT) as scope_manager:

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

            threadID_name = 'ThreadId_%s_%d_%d_%d' % (scope_map.label, cfg.cfg_id, state_dfg.block_id,
                                                      state_dfg.node_id(node))

            callsite_stream.write(f"{ids_ctype} {threadID_name} = ({flat_thread_idx_expr}) / {warpSize};", cfg,
                                  state_id, node)
            self._dispatcher.defined_vars.add(threadID_name, DefinedType.Scalar, ids_ctype)

            # ----------------- Compute Map indices (= Warp indices) -----------------------

            for i in range(warp_dim):
                var_name = scope_map.params[-i - 1]  # reverse order
                previous_sizes = warp_dim_bounds[:i]

                if len(previous_sizes) > 0:
                    divisor = product(previous_sizes)
                    expr = f"(({threadID_name} / {divisor}) % ({warp_dim_bounds[i]}))"
                else:
                    expr = f"({threadID_name} % ({warp_dim_bounds[i]}))"

                callsite_stream.write(f"{ids_ctype} {var_name} = {expr};", cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ids_ctype)

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, callsite_stream)

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
                    condition_terms.append(f'{expr} % {stride} == 0')

                if condition_terms:
                    condition = " && ".join(condition_terms)
                    scope_manager.open(condition)

            self._dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, node, function_stream, callsite_stream)

    def _handle_GPU_Warp_scope_guards(self, state_dfg: SDFGState, node: nodes.MapEntry, map_range: subsets.Range,
                                      warp_dim: int, num_threads_in_block, num_warps, kernel_stream: CodeIOStream,
                                      scope_manager: 'ScopeManager'):

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
            condition = (f"{num_threads_in_block} % {warpSize} != 0 || "
                         f"{num_threads_in_block} > 1024 || "
                         f"{num_warps} * {warpSize} > {num_threads_in_block}")
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
                kernel_stream.write(
                    f'if ({min_element} < 0) {{\n'
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

    def __init__(self,
                 frame_codegen: DaCeCodeGenerator,
                 sdfg: SDFG,
                 cfg: ControlFlowRegion,
                 dfg_scope: ScopeSubgraphView,
                 state_id: int,
                 function_stream: CodeIOStream,
                 callsite_stream: CodeIOStream,
                 comment: str = None,
                 brackets_on_enter: bool = True,
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
        :param brackets_on_enter: Whether on entering (i.e. when using "with", there should be a bracket opened). Default is True.
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
        self.brackets_on_enter = brackets_on_enter
        self.debug = debug
        self._opened = 0

        self.entry_node = self.dfg_scope.source_nodes()[0]
        self.exit_node = self.dfg_scope.sink_nodes()[0]

    def __enter__(self):
        """
        Writes the opening bracket in case self.brackets_on_enter
        is set to true, which it is by default.
        """
        if self.brackets_on_enter:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Writes the closing brackets to the stream.
        """
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
