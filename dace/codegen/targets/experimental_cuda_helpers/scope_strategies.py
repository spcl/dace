# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scope-emission strategies (RAII bracket managers) for the experimental CUDA codegen."""
from abc import ABC, abstractmethod

from dace import dtypes, subsets, symbolic
from dace.codegen import common
from dace.sdfg import SDFG, ScopeSubgraphView, nodes, SDFGState
from dace.sdfg.state import ControlFlowRegion
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.transformation import helpers
from dace.codegen.targets.cpp import sym2cpp
from dace.codegen.targets.cpu import (collect_gpu_block_reductions, drain_gpu_block_reduction,
                                      register_gpu_block_reduction)
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen, KernelSpec
from dace.codegen.targets.cuda import (_named_idx, chiplet_padding_condition, kernel_grid_conditions,
                                       kernel_index_definitions, kernel_launch_qualifiers)
from dace.transformation.dataflow.add_threadblock_map import product


def emit_dim_index_definitions(scope_map, axis: str, index_types, callsite_stream: CodeIOStream, cfg: ControlFlowRegion,
                               state_id: int, anchor_node, dispatcher: TargetDispatcher):
    """Emit ``{type} {var_name} = {expr};`` per map dim from the symbolic map coordinates.

    ``axis`` is ``'blockIdx'`` (kernel scope) or ``'threadIdx'`` (thread-block scope). The first
    three dims map directly to ``axis.{x|y|z}``; further dims delinearize off ``axis.z``.

    :returns: ``(map_range, sym_indices, sym_coords)`` for callers building downstream guards.
    """
    map_range = subsets.Range(scope_map.range[::-1])  # reversed for memory coalescing
    dimensions = len(map_range)
    dim_sizes = map_range.size()
    sym_indices = [symbolic.symbol(f'__SYM_IDX{i}', nonnegative=True, integer=True) for i in range(dimensions)]
    sym_coords = map_range.coord_at(sym_indices)

    for dim in range(dimensions):
        var_name = scope_map.params[-dim - 1]  # reversed
        if dim < 3:
            expr = f"{axis}.{_named_idx(dim)}"
            if dim == 2 and dimensions > 3:
                tail = product(dim_sizes[3:])
                expr = f"({expr} / ({sym2cpp(tail)}))"
        else:
            tail = product(dim_sizes[dim + 1:])
            expr = f"(({axis}.z / ({sym2cpp(tail)})) % ({sym2cpp(dim_sizes[dim])}))"
        var_def = sym2cpp(sym_coords[dim]).replace(f'__SYM_IDX{dim}', expr)
        ctype = index_types[var_name].ctype
        callsite_stream.write(f'{ctype} {var_name} = {var_def};', cfg, state_id, anchor_node)
        dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ctype)

    return map_range, sym_indices, sym_coords


class ScopeGenerationStrategy(ABC):
    """Base strategy for generating GPU scope code.

    Subclasses set ``SCHEDULE`` (matched by ``applicable()`` against the source MapEntry's
    schedule) and ``SCOPE_COMMENT``, implement ``generate()``, and reuse the
    ``dispatch_and_deallocate`` tail.
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
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        raise NotImplementedError('Abstract class')

    def dispatch_and_deallocate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
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


def open_block_reductions(strategy: 'ScopeGenerationStrategy', sdfg: SDFG, cfg: ControlFlowRegion, state_id: int,
                          scope_entry: nodes.MapEntry, block_dims, stream: CodeIOStream) -> list:
    """Declare and identity-initialize the register partial of every map-exit WCR accumulator under
    ``scope_entry`` that folds by ``gpucub::BlockReduce`` plus one atomic per block. Emit before the
    bounds guard, so out-of-range threads still join the barrier-using fold."""
    reductions = collect_gpu_block_reductions(sdfg, cfg.state(state_id), scope_entry, block_dims,
                                              strategy.codegen._frame)
    covered = strategy.codegen._cpu_codegen._gpu_block_reduction_covered
    for red in reductions:
        stream.write(register_gpu_block_reduction(red, covered), cfg, state_id, scope_entry)
    return reductions


def drain_block_reductions(strategy: 'ScopeGenerationStrategy', reductions: list, label: str, cfg: ControlFlowRegion,
                           state_id: int, scope_entry: nodes.MapEntry, stream: CodeIOStream):
    """Fold the partials :func:`open_block_reductions` declared; emit after the bounds guard closes."""
    covered = strategy.codegen._cpu_codegen._gpu_block_reduction_covered
    for i, red in enumerate(reductions):
        stream.write(drain_gpu_block_reduction(red, f'{label}_{i}', covered), cfg, state_id, scope_entry)


class KernelScopeGenerator(ScopeGenerationStrategy):

    SCHEDULE = dtypes.ScheduleType.GPU_Device
    SCOPE_COMMENT = "Kernel scope"

    def generate(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        with ScopeManager(frame_codegen=self.codegen._frame,
                          sdfg=sdfg,
                          cfg=cfg,
                          dfg_scope=dfg_scope,
                          state_id=state_id,
                          function_stream=function_stream,
                          callsite_stream=callsite_stream,
                          comment=self.SCOPE_COMMENT,
                          brackets_on_enter=False) as scope_manager:
            scope_manager.open(prefix=self.kernel_signature(dfg_scope))

            kernel_spec = self._current_kernel_spec
            kernel_entry_node = kernel_spec.kernel_map_entry  # == dfg_scope.source_nodes()[0]

            for var_name, expr in kernel_index_definitions(kernel_spec.kernel_map, kernel_spec.block_dims,
                                                           kernel_spec.per_thread, kernel_spec.chiplets,
                                                           kernel_spec.chiplet_chunk, kernel_spec.index_types):
                ctype = kernel_spec.index_types[var_name].ctype
                callsite_stream.write(f'{ctype} {var_name} = {expr};', cfg, state_id, kernel_entry_node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ctype)
            # Without a thread-block map every thread handles one iteration and masks the trailing blocks,
            # and the kernel map's own WCR accumulators fold across the block (``emit_tree_reductions``
            # gates the legacy codegen only).
            reductions = []
            if kernel_spec.per_thread:
                reductions = open_block_reductions(self, sdfg, cfg, state_id, kernel_entry_node, kernel_spec.block_dims,
                                                   callsite_stream)
            unguarded = scope_manager.opened
            if kernel_spec.per_thread:
                conditions = kernel_grid_conditions(kernel_spec.kernel_map, kernel_spec.block_dims,
                                                    kernel_spec.chiplets)
            else:
                conditions = [chiplet_padding_condition(kernel_spec.kernel_map)] if kernel_spec.chiplets > 1 else []
            for condition in filter(None, conditions):
                scope_manager.open(condition=condition)

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, kernel_entry_node, function_stream, callsite_stream)

            self.dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, kernel_entry_node, function_stream,
                                         callsite_stream)
            scope_manager.close_to(unguarded)
            drain_block_reductions(self, reductions, kernel_spec.kernel_name, cfg, state_id, kernel_entry_node,
                                   callsite_stream)

    def kernel_signature(self, dfg_scope: ScopeSubgraphView) -> str:
        kernel_name = self._current_kernel_spec.kernel_name
        kernel_args = self._current_kernel_spec.args_typed
        block_dims = self._current_kernel_spec.block_dims
        node = dfg_scope.source_nodes()[0]

        maxnreg, launch_bounds = kernel_launch_qualifiers(node, block_dims)

        qualifiers = ' '.join(q for q in ('__global__ void', maxnreg, launch_bounds, kernel_name) if q)
        return f'{qualifiers}({", ".join(kernel_args)}) '


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

            state = cfg.state(state_id)
            index_types = common.gpu_map_index_types(sdfg, state, node,
                                                     self.codegen._frame.symbols_defined_at(state, node))
            map_range, symbolic_indices, _sym_coords = emit_dim_index_definitions(scope_map, 'threadIdx', index_types,
                                                                                  callsite_stream, cfg, state_id, node,
                                                                                  self._dispatcher)

            symbolic_index_bounds = [
                idx + (block_dim * rng[2]) - 1
                for idx, block_dim, rng in zip(symbolic_indices, kernel_block_dims, map_range)
            ]

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, callsite_stream)

            # Map-exit WCR accumulators fold by gpucub::BlockReduce and one atomic per block, always
            # (``emit_tree_reductions`` gates the legacy codegen only).
            reductions = open_block_reductions(self, sdfg, cfg, state_id, node, kernel_block_dims, callsite_stream)
            unguarded = scope_manager.opened

            # Guard each dim so out-of-bounds threads in a trailing block are skipped.
            minels = map_range.min_element()
            maxels = map_range.max_element()
            for dim, (var_name, start, end) in enumerate(zip(scope_map.params[::-1], minels, maxels)):

                # Emit only the bounds that are not provably always-true.
                condition = ''

                if dim >= 3 or (symbolic_indices[dim] >= start) != True:
                    condition += f'{var_name} >= {sym2cpp(start)}'

                # Special case: block size is exactly the range of the map (0:b)
                if dim >= 3:
                    skipcond = False
                else:
                    skipcond = symbolic_index_bounds[dim].subs({symbolic_indices[dim]: start}) == end

                if dim >= 3 or (not skipcond and (symbolic_index_bounds[dim] < end) != True):
                    if len(condition) > 0:
                        condition += ' && '
                    condition += f'{var_name} < {sym2cpp(end + 1)}'

                if len(condition) > 0:
                    scope_manager.open(condition=condition)

            self.dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, node, function_stream, callsite_stream)
            scope_manager.close_to(unguarded)
            drain_block_reductions(self, reductions, scope_map.label, cfg, state_id, node, callsite_stream)


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

            kernel_spec = self._current_kernel_spec
            block_dims = kernel_spec.block_dims
            warpSize = common.gpu_warp_size()

            state_dfg = cfg.state(state_id)
            node = dfg_scope.source_nodes()[0]
            scope_map = node.map

            map_range = subsets.Range(scope_map.range[::-1])  # Reversed for potential better performance
            warp_dim = len(map_range)

            # These sizes and bounds may be symbolic.
            num_threads_in_block = product(block_dims)
            warp_dim_bounds = [max_elem + 1 for max_elem in map_range.max_element()]
            num_warps = product(warp_dim_bounds)

            ids_ctype = common.gpu_thread_id_type().ctype

            self.handle_GPU_Warp_scope_guards(state_dfg, node, map_range, warp_dim, num_threads_in_block, num_warps,
                                              callsite_stream, scope_manager)

            flat_thread_idx_expr = flat_thread_index_expr(block_dims)
            threadID_name = 'ThreadId_%s_%d_%d_%d' % (scope_map.label, cfg.cfg_id, state_dfg.block_id,
                                                      state_dfg.node_id(node))

            callsite_stream.write(f"{ids_ctype} {threadID_name} = ({flat_thread_idx_expr}) / {warpSize};", cfg,
                                  state_id, node)
            self._dispatcher.defined_vars.add(threadID_name, DefinedType.Scalar, ids_ctype)

            # Compute the map indices (the warp indices), in reverse parameter order.
            for i in range(warp_dim):
                var_name = scope_map.params[-i - 1]
                expr = warp_index_expr(threadID_name, warp_dim_bounds, i)
                callsite_stream.write(f"{ids_ctype} {var_name} = {expr};", cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, ids_ctype)

            self.codegen._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, callsite_stream)

            # Guard conditions for warp execution.
            if num_warps * warpSize != num_threads_in_block:
                condition = f'{threadID_name} < {num_warps}'
                scope_manager.open(condition)

            warp_range = [(start, end + 1, stride) for start, end, stride in map_range.ranges]

            for var_name, (start, _, stride) in zip(scope_map.params[::-1], warp_range):
                condition = strided_range_guard(var_name, start, stride)
                if condition:
                    scope_manager.open(condition)

            self.dispatch_and_deallocate(sdfg, cfg, dfg_scope, state_id, node, function_stream, callsite_stream)

    def handle_GPU_Warp_scope_guards(self, state_dfg: SDFGState, node: nodes.MapEntry, map_range: subsets.Range,
                                     warp_dim: int, num_threads_in_block, num_warps, kernel_stream: CodeIOStream,
                                     scope_manager: 'ScopeManager'):

        warpSize = common.gpu_warp_size()

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


class ScopeManager:
    """RAII context manager that balances ``{`` / ``}`` for a generated scope.

    Optional ``debug`` mode annotates each bracket with ``comment`` for readability.
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
        """Initialize the scope manager.

        :param frame_codegen: frame codegen used for in-scope array (de)allocation.
        :param comment: block label surfaced in ``debug`` mode.
        :param brackets_on_enter: open a bracket on ``__enter__`` (default).
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
        """Open a bracket when ``brackets_on_enter`` is set (the default)."""
        if self.brackets_on_enter:
            self.open()
        return self

    @property
    def opened(self) -> int:
        """Brackets currently open."""
        return self._opened

    def close_to(self, depth: int):
        """Close brackets until ``depth`` remain open."""
        while self._opened > depth:
            self._opened -= 1
            line = "}"
            if self.debug:
                line += f" // {self.comment} (close to {depth})"
            self.callsite_stream.write(line, self.cfg, self.state_id, self.exit_node)

    def __exit__(self, exc_type, exc_value, traceback):
        """Write the closing bracket for every bracket opened by this manager."""
        for i in range(self._opened):
            line = "}"
            if self.debug:
                line += f" // {self.comment} (close {i + 1})"
            self.callsite_stream.write(line, self.cfg, self.state_id, self.exit_node)

    def open(self, condition: str = None, prefix: str = ''):
        """Open a bracket, emitting ``if (condition) {`` when ``condition`` is given else ``{prefix}{``."""
        line = f"if ({condition}) {{" if condition else f"{prefix}{{"
        if self.debug:
            line += f" // {self.comment} (open {self._opened + 1})"
        self.callsite_stream.write(line, self.cfg, self.state_id, self.entry_node)
        self._opened += 1


def flat_thread_index_expr(block_dims) -> str:
    """The thread's flat index within its block, skipping unit block dimensions."""
    terms = []
    for i, dim_size in enumerate(block_dims):
        if dim_size == 1:
            continue
        stride = [f"{block_dims[j]}" for j in range(i) if block_dims[j] > 1]
        terms.append(" * ".join(stride + [f"threadIdx.{_named_idx(i)}"]))
    joined = " + ".join(terms)
    return f"({joined})" if len(terms) > 1 else joined


def warp_index_expr(warp_id: str, warp_dim_bounds, i: int) -> str:
    """Index of warp dimension ``i`` within the flat warp id."""
    if i == 0:
        return f"({warp_id} % ({warp_dim_bounds[0]}))"
    return f"(({warp_id} / {product(warp_dim_bounds[:i])}) % ({warp_dim_bounds[i]}))"


def strided_range_guard(var_name: str, start, stride) -> str:
    """Condition selecting the iterations of ``start::stride`` among ``start, start + 1, ...``; empty if all."""
    terms = []
    if start != 0:
        terms.append(f"{var_name} >= {start}")
    if stride != 1:
        expr = var_name if start == 0 else f"({var_name} - {start})"
        terms.append(f'{expr} % {stride} == 0')
    return " && ".join(terms)
