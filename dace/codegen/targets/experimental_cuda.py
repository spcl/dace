# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Experimental CUDA code generator: emits kernels, streams, and host glue for GPU SDFGs."""

from typing import TYPE_CHECKING, Any

import dace
from dace import Memlet, cpf_lowering, dtypes, registry
from dace import data as dt
from dace.codegen import common
from dace.codegen.codeobject import CodeObject
from dace.codegen.common import update_persistent_desc
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.codegen.exceptions import CodegenError
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.targets import cpp
from dace.codegen.targets.cpp import mangle_dace_state_struct_name, ptr, sym2cpp
from dace.codegen.targets.cpu import CPUCodeGen
from dace.codegen.targets.cuda import (
    _DYNAMIC_SHARED_MEMORY_SYMBOL,
    chiplet_count,
    compute_pool_release,
    distribute_over_chiplets,
    dynamic_map_input_args,
    dynamic_shared_memory_request,
    gpu_cmake_options,
    gpu_runtime_code,
    gpu_scope_maps_recursive,
    kernel_read_only_data,
    location_condition,
    location_index_exprs,
    plan_shared_memory,
    reset_shared_code,
)
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import (
    assigned_stream_expr,
    generate_sync_debug_call,
    host_read_device_copies,
    num_gpu_streams,
)
from dace.config import Config
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_access, as_range, config_int, config_str
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.transformation.passes import gpu_shared_memory
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator

#: Lifetimes that allocate an array for the whole program rather than inside a state or scope.
GLOBAL_LIFETIMES = (
    dtypes.AllocationLifetime.Global,
    dtypes.AllocationLifetime.Persistent,
    dtypes.AllocationLifetime.External,
)


def scope_map_entry(dfg_scope: ScopeSubgraphView) -> nodes.MapEntry:
    """The map entry that opens a GPU scope subgraph."""
    entry = dfg_scope.source_nodes()[0]
    assert isinstance(entry, nodes.MapEntry)
    return entry


@registry.autoregister_params(name="experimental_cuda")
class ExperimentalCUDACodeGen(TargetCodeGenerator):
    """Experimental CUDA code generator."""

    target_name = "experimental_cuda"
    title = "CUDA"

    def __init__(self, frame_codegen: "DaCeCodeGenerator", sdfg: SDFG):

        self._frame: DaCeCodeGenerator = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher

        self._in_device_code = False

        self.backend: str = common.get_gpu_backend()
        self.language = "cu" if self.backend == "cuda" else "cpp"
        target_type = "" if self.backend == "cuda" else self.backend
        self._codeobject = CodeObject(
            sdfg.name + "_" + "cuda", "", self.language, ExperimentalCUDACodeGen, "CUDA", target_type=target_type
        )

        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()

        self._global_sdfg: SDFG = sdfg

        self.pool_release: dict[tuple[SDFG, str], tuple[SDFGState, set[nodes.Node]]] = {}
        # Every pooled array released early, which the end of its lifetime must not free again
        self.pool_released_early: OrderedSet[tuple[SDFG, str]] = OrderedSet()
        self.has_pool = False

        cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        assert isinstance(cpu_codegen, CPUCodeGen)  # the generic node dispatcher is always the CPU target
        self._cpu_codegen: CPUCodeGen = cpu_codegen
        self._dispatcher.register_map_dispatcher(dtypes.EXPERIMENTAL_GPU_SCHEDULES, self)
        self._dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        gpu_storage = dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES
        self._dispatcher.register_array_dispatcher(gpu_storage, self)
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)
        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                self._dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self._dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        self._current_kernel_spec: KernelSpec | None = None
        self._num_gpu_streams: int = 0
        self._kernel_dimensions_map: dict[nodes.MapEntry, tuple[list, list]] = {}
        self._tb_inserted_kernels: OrderedSet[nodes.MapEntry] = OrderedSet()
        self._kernel_arglists: dict[nodes.MapEntry, dict[str, dt.Data]] = {}

        # Device-to-host copies already synchronized, keyed by (cfg id, state id, destination node).
        self._synchronized_d2h: OrderedSet[tuple[int, int, nodes.AccessNode]] = OrderedSet()

    @property
    def current_kernel_spec(self) -> "KernelSpec":
        """The spec of the kernel being generated; only valid between the start and end of its scope."""
        assert self._current_kernel_spec is not None
        return self._current_kernel_spec

    def preprocess(self, sdfg: SDFG):
        """Prepare the SDFG for GPU code generation.

        All SDFG-level transformation lives in :class:`GPUCodegenPreprocessPipeline`;
        this method only does framecode-target bookkeeping (statestruct entry, cache
        rebuild, stream manager, pool-release, per-kernel arglists).
        """
        # CPF supplies its own context (cpf_gpu_context) with the same shape, because the field is what the generated
        # body reaches the stream through; the runtime type would drag in the header the rendering exists to do without.
        context_type = "cpf_gpu_context" if cpf_lowering.device() else "dace::cuda::Context"
        self._frame.statestruct.append(f"{context_type} *gpu_context;")
        self._dispatcher._used_targets.add(self)

        pipeline_results: dict[str, Any] = {}
        GPUCodegenPreprocessPipeline().apply_pass(sdfg, pipeline_results)
        self._globalcode.write(plan_shared_memory(sdfg))

        # AddThreadBlockMaps returns the kernel-dimension map and the set of kernels it
        # tiled; both are consulted when emitting kernel launches.
        atb_results = pipeline_results.get("AddThreadBlockMaps", {}) or {}
        self._kernel_dimensions_map = atb_results.get("kernel_dimensions_map", {})
        self._tb_inserted_kernels = atb_results.get("tb_inserted_kernels", OrderedSet())

        # Library-node expansion adds new nested SDFGs with new cfg_ids; re-seed
        # the framecode's symbol/constant cache so lookups succeed for them.
        self._frame.resolve_symbols_and_constants(sdfg)

        # Streams are read off ``Node.gpu_stream_id``, so a deserialized SDFG needs no re-scheduling.
        self._num_gpu_streams = num_gpu_streams(sdfg)

        if Config.get("compiler", "cuda", "auto_syncthreads_insertion"):
            DefaultSharedMemorySync().apply_pass(sdfg, None)

        if compute_pool_release(sdfg, self.pool_release):
            self.has_pool = True
        self.pool_released_early = OrderedSet(self.pool_release)

        shared_transients = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(sdfg, recursive=True):
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                if state.parent not in shared_transients:
                    shared_transients[state.parent] = state.parent.shared_transients()
                self._kernel_arglists[node] = state.scope_subgraph(node).arglist(
                    defined_syms, shared_transients[state.parent]
                )
                self._kernel_arglists[node].update(dynamic_map_input_args(state, node))

    @property
    def has_initializer(self) -> bool:
        return True

    @property
    def has_finalizer(self) -> bool:
        return True

    def generate_scope(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):

        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import (
            KernelScopeGenerator,
            ScopeGenerationStrategy,
            ThreadBlockScopeGenerator,
            WarpScopeGenerator,
        )

        scope_entry = scope_map_entry(dfg_scope)

        if not self._in_device_code:
            state = cfg.state(state_id)
            scope_exit = dfg_scope.sink_nodes()[0]
            assert isinstance(scope_exit, nodes.MapExit)
            scope_entry_stream = CodeIOStream()
            scope_exit_stream = CodeIOStream()

            instr = self._dispatcher.instrumentation[scope_entry.map.instrument]
            if instr is not None:
                instr.on_scope_entry(
                    sdfg, cfg, state, scope_entry, callsite_stream, scope_entry_stream, self._globalcode
                )
                outer_stream = CodeIOStream()
                instr.on_scope_exit(sdfg, cfg, state, scope_exit, outer_stream, scope_exit_stream, self._globalcode)

            self._dispatcher.defined_vars.enter_scope(scope_entry)

            kernel_spec = KernelSpec(cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id)
            self._current_kernel_spec = kernel_spec

            self.define_variables_in_kernel_scope(sdfg, self._dispatcher)
            self.synchronize_host_reads(cfg, state_id, scope_entry, callsite_stream, launch_arguments=True)
            self.declare_and_invoke_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            kernel_stream = CodeIOStream()
            kernel_function_stream = self._globalcode

            self._in_device_code = True
            # Everything the CPU codegen emits for this kernel (allocations included, which it is
            # dispatched for directly) goes into the device file, so it keys its per-file helpers here.
            host_calling_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self

            kernel_scope_generator = KernelScopeGenerator(codegen=self)
            try:
                if not kernel_scope_generator.applicable(
                    sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream
                ):
                    raise ValueError(
                        "Invalid kernel configuration: This strategy is only applicable if the "
                        "outermost GPU schedule is of type GPU_Device (most likely cause)."
                    )
                kernel_scope_generator.generate(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream)
            finally:
                self._cpu_codegen.calling_codegen = host_calling_codegen

            self._localcode.write(scope_entry_stream.getvalue())
            self._localcode.write(kernel_stream.getvalue() + "\n")
            self._localcode.write(scope_exit_stream.getvalue())

            self._in_device_code = False

            self.generate_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            self._dispatcher.defined_vars.exit_scope(scope_entry)

            if instr is not None:
                callsite_stream.write(outer_stream.getvalue())

            return

        # Nested GPU scope.
        supported_strategies: list[ScopeGenerationStrategy] = [
            ThreadBlockScopeGenerator(codegen=self),
            WarpScopeGenerator(codegen=self),
        ]

        for strategy in supported_strategies:
            if strategy.applicable(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
                strategy.generate(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
                return

        schedule_type = scope_entry.map.schedule

        if schedule_type == dace.ScheduleType.GPU_Device:
            raise NotImplementedError("Dynamic parallelism (nested GPU_Device schedules) is not supported.")

        raise NotImplementedError(
            f"Scope generation for schedule type '{schedule_type}' is not implemented in ExperimentalCUDACodeGen. "
            "Please check for supported schedule types or implement the corresponding strategy."
        )

    def define_variables_in_kernel_scope(self, sdfg: SDFG, dispatcher: TargetDispatcher):
        """Register every kernel argument in the dispatcher under its device-side pointer name.

        Persistent/external data that lives in ``__state`` cannot be referenced directly from
        device code -- it is passed as a kernel argument, and the dispatcher needs to resolve
        accesses through the device pointer.  Constants pick up a ``const`` ctype qualifier.
        """
        kernel_spec = self.current_kernel_spec
        kernel_constants: set[str] = kernel_spec.kernel_constants
        kernel_arglist: dict[str, dt.Data] = kernel_spec.arglist

        restore_in_device_code = self._in_device_code
        for name, data_desc in kernel_arglist.items():
            if not name in sdfg.arrays:
                continue

            data_desc = sdfg.arrays[name]
            self._in_device_code = False
            host_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            is_global: bool = data_desc.lifetime in GLOBAL_LIFETIMES
            defined_type, ctype = dispatcher.defined_vars.get(host_ptrname, is_global=is_global)

            self._in_device_code = True
            device_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            if name in kernel_constants and "const " not in ctype:
                ctype = f"const {ctype}"

            dispatcher.defined_vars.add(device_ptrname, defined_type, ctype, allow_shadowing=True)

        self._in_device_code = restore_in_device_code

    def declare_and_invoke_kernel_wrapper(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):

        scope_entry = scope_map_entry(dfg_scope)

        kernel_spec = self.current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_wrapper_args_as_input = kernel_spec.kernel_wrapper_args_as_input
        kernel_wrapper_args_typed = kernel_spec.kernel_wrapper_args_typed

        function_stream.write(
            "DACE_EXPORTED void __dace_runkernel_%s(%s);\n" % (kernel_name, ", ".join(kernel_wrapper_args_typed)),
            cfg,
            state_id,
            scope_entry,
        )

        # Wrap the invocation in a block so dynamic-input local declarations don't leak.
        state = cfg.state(state_id)
        dyn_inputs = list(dace.sdfg.dynamic_map_inputs(state, scope_entry))
        has_dyn_inputs = len(dyn_inputs) > 0
        if has_dyn_inputs:
            callsite_stream.write("{", cfg, state_id, scope_entry)

        for e in dyn_inputs:
            if e.data.data == e.dst_conn:
                continue  # Already in scope under that name; redefining it would self-initialize.
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg,
                state_id,
                scope_entry,
            )

        callsite_stream.write(
            "__dace_runkernel_%s(%s);\n" % (kernel_name, ", ".join(kernel_wrapper_args_as_input)),
            cfg,
            state_id,
            scope_entry,
        )

        if has_dyn_inputs:
            callsite_stream.write("}", cfg, state_id, scope_entry)

    def generate_kernel_wrapper(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):

        scope_entry = scope_map_entry(dfg_scope)

        kernel_spec = self.current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_args_as_input = kernel_spec.args_as_input
        kernel_launch_args_typed = kernel_spec.kernel_wrapper_args_typed

        grid_dims = kernel_spec.grid_dims
        block_dims = kernel_spec.block_dims
        gdims = ", ".join(sym2cpp(grid_dims))
        bdims = ", ".join(sym2cpp(block_dims))

        self._localcode.write(
            f"""
            DACE_EXPORTED void __dace_runkernel_{kernel_name}({", ".join(kernel_launch_args_typed)});
            void __dace_runkernel_{kernel_name}({", ".join(kernel_launch_args_typed)})
            """,
            cfg,
            state_id,
            scope_entry,
        )

        self._localcode.write("{", cfg, state_id, scope_entry)

        # Skip launches on empty or negative-sized grids that we can't prove non-empty statically.
        single_dimchecks = []
        for gdim in grid_dims:
            if (gdim > 0) != True:
                single_dimchecks.append(f"(({sym2cpp(gdim)}) <= 0)")

        dimcheck = " || ".join(single_dimchecks)

        if dimcheck:
            emptygrid_warning = ""
            if Config.get("debugprint") == "verbose" or Config.get_bool("compiler", "cuda", "syncdebug"):
                emptygrid_warning = (
                    f'printf("Warning: Skipping launching kernel \\"{kernel_name}\\" due to an empty grid.\\n");'
                )

            self._localcode.write(
                f"""
                    if ({dimcheck}) {{
                        {emptygrid_warning}
                        return;
                    }}""",
                cfg,
                state_id,
                scope_entry,
            )

        # The bytes of dynamic shared memory the kernel uses (see ``gpu_shared_memory.PlanSharedMemory``)
        dynsmem_size = vars(scope_entry)["_cuda_dynamic_shared_memory"]  # set on the entry by gpu_shared_memory
        self._localcode.write(dynamic_shared_memory_request(kernel_name, scope_entry), cfg, state_id, scope_entry)
        stream_var_name = config_str("compiler", "cuda", "gpu_stream_name").split(",")[1]
        kargs = ", ".join(["(void *)&" + arg for arg in kernel_args_as_input])
        self._localcode.write(
            f"""
            void  *{kernel_name}_args[] = {{ {kargs} }};
            gpuError_t __err = {self.backend}LaunchKernel((void*){kernel_name}, dim3({gdims}), dim3({bdims}), {kernel_name}_args, {sym2cpp(dynsmem_size)}, {stream_var_name});
            """,
            cfg,
            state_id,
            scope_entry,
        )

        self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});\n')
        self._localcode.write(generate_sync_debug_call())

        self._localcode.write("}", cfg, state_id, scope_entry)

    def copy_memory(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge[Memlet],
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        # One container handed on through the scope exits with a single subset moves nothing; the
        # CPU copy would take the exit's slice as the source and the whole container as the target.
        if (
            isinstance(src_node, nodes.AccessNode)
            and isinstance(dst_node, nodes.AccessNode)
            and src_node.data == dst_node.data
            and all(e.data.data == src_node.data and e.data.other_subset is None for e in dfg.memlet_path(edge))
        ):
            return
        # All CPU<->GPU and GPU<->GPU AccessNode->AccessNode edges (host-issued
        # and in-kernel collaborative) are lifted to ``CopyLibraryNode`` by
        # ``InsertExplicitCopies`` during ``preprocess()`` and
        # lowered through their expansions. Anything reaching this dispatch
        # is a register / scope-local CPU copy -- delegate to CPU codegen.
        # A host-side copy touching device memory has no CPU lowering: CopyND would dereference
        # device pointers on the host, so refuse it rather than emit a segfault.
        if (
            not self._in_device_code
            and isinstance(src_node, nodes.AccessNode)
            and isinstance(dst_node, nodes.AccessNode)
            and GPU_RESIDENT_STORAGES & {sdfg.arrays[src_node.data].storage, sdfg.arrays[dst_node.data].storage}
        ):
            raise CodegenError(
                f"Copy {src_node} -> {dst_node} involves GPU memory but was not lowered to a "
                "CopyLibraryNode by InsertExplicitCopies; the CPU fallback would access device "
                "memory from the host."
            )
        if not isinstance(src_node, (nodes.Tasklet, nodes.AccessNode)) or not isinstance(
            dst_node, (nodes.Tasklet, nodes.AccessNode)
        ):
            raise CodegenError(f"Copy {src_node} -> {dst_node} is neither between tasklets nor access nodes")
        self._cpu_codegen.copy_memory(
            sdfg, cfg, dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream
        )

    def synchronize_host_reads(
        self,
        cfg: ControlFlowRegion,
        state_id: int,
        consumer: nodes.Node,
        callsite_stream: CodeIOStream,
        launch_arguments: bool = False,
    ) -> None:
        """Block the host on the copy stream before ``consumer`` reads a device-to-host copy.

        Emitted lazily at the first host consumer instead of at the copy, and only once per
        destination: the copy is asynchronous, so the first host read of its destination has to wait
        for it, and everything after that read is ordered by the host's own program order.
        ``launch_arguments`` marks a kernel launch, which reads scalars on the host to pack them into
        the argument list by value; its pointer arguments stay on the stream and need no sync.
        """
        state = cfg.state(state_id)
        for destination, producer in host_read_device_copies(state, consumer):
            if launch_arguments:
                if not isinstance(state.sdfg.arrays[destination.data], dt.Scalar):
                    continue
            elif consumer.gpu_stream_id is not None and consumer.gpu_stream_id == producer.gpu_stream_id:
                continue  # Consumer issues on the copy's stream, which already orders the two.
            key = (cfg.cfg_id, state_id, destination)
            if key in self._synchronized_d2h:
                continue
            self._synchronized_d2h.add(key)
            gpu_stream = self.issued_stream_expression(state, producer)
            callsite_stream.write(
                f"DACE_GPU_CHECK({self.backend}StreamSynchronize({gpu_stream}));\n", cfg, state_id, consumer
            )

    def issued_stream_expression(self, state: SDFGState, producer: nodes.Node) -> str:
        """The stream ``producer`` issued its work on, spelled exactly as at the issue site.

        A lifted copy names its stream through its ``__dace_current_stream`` connector, so the wait
        renders that connector's memlet: the stream manager's context-array expression is a different
        stream object than the ``gpu_streams`` element the copy read, and waiting on it orders nothing.
        A producer without a wired stream connector falls back to the assigned stream.
        """
        from dace.codegen.targets.experimental_cpu import ExperimentalCPUCodeGen, format_index_access
        from dace.libraries.standard.helper import CURRENT_STREAM_NAME

        for edge in state.in_edges(producer):
            if edge.dst_conn != CURRENT_STREAM_NAME or edge.data is None or edge.data.data is None:
                continue
            parts = None
            if isinstance(self._cpu_codegen, ExperimentalCPUCodeGen):
                parts = self._cpu_codegen.array_index_access(
                    state.sdfg, state.sdfg.arrays[edge.data.data], edge.data.data
                )
            if parts is None:
                return cpp.cpp_array_expr(state.sdfg, edge.data, framecode=self._frame)
            ptrname, fnname, extra_syms = parts[0], parts[1], parts[3]
            return format_index_access(
                ptrname, fnname, [str(index) for index in as_range(edge.data.subset).min_element()], extra_syms
            )
        return assigned_stream_expr(producer)

    def reads_unsynchronized_device_copy(self, state: SDFGState, node: nodes.Node) -> bool:
        """Whether ``node`` is a host node whose first read of a device-to-host copy is still unsynced."""
        if not isinstance(node, (nodes.Tasklet, nodes.NestedSDFG)):
            return False
        cfg_id = state.parent_graph.cfg_id
        state_id = state.parent_graph.node_id(state)
        for destination, producer in host_read_device_copies(state, node):
            if node.gpu_stream_id is not None and node.gpu_stream_id == producer.gpu_stream_id:
                continue
            if (cfg_id, state_id, destination) not in self._synchronized_d2h:
                return True
        return False

    def state_dispatch_predicate(self, sdfg, state):
        """Return True iff this codegen should drive code emission for ``state``.

        A state is claimed when it holds a pooled allocation that still needs to be released,
        or when code generation is already inside a device-side kernel.
        """
        return any(s is state for s, _ in self.pool_release.values()) or self._in_device_code

    def node_dispatch_predicate(self, sdfg, state, node):
        """Return True iff ``node`` should be emitted by this codegen.

        Claimed nodes are those carrying a GPU schedule served by this backend, plus every
        node encountered while already emitting device code.
        """
        schedule = None
        if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
            schedule = node.map.schedule
        elif isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
            schedule = node.consume.schedule
        elif isinstance(node, nodes.LibraryNode):
            schedule = node.schedule
        if schedule in dtypes.EXPERIMENTAL_GPU_SCHEDULES:
            return True
        if self._in_device_code:
            return True
        # Host node reading a device-to-host copy: claimed to place the copy's synchronization
        # right before it, then generated by the CPU codegen as usual.
        return self.reads_unsynchronized_device_copy(state, node)

    def generate_state(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state: SDFGState,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
        generate_state_footer: bool = False,
    ):

        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

        # Emit cudaFree for pooled transients whose lifetime ends in this state.
        if not self._in_device_code:
            handled_keys: OrderedSet[tuple[SDFG, str]] = OrderedSet()
            backend = self.backend
            for (pool_sdfg, name), (pool_state, _) in self.pool_release.items():
                if (pool_sdfg is not sdfg) or (pool_state is not state):
                    continue

                data_descriptor = pool_sdfg.arrays[name]
                ptrname = ptr(name, data_descriptor, pool_sdfg, self._frame)

                if isinstance(data_descriptor, dt.Array) and data_descriptor.start_offset != 0:
                    ptrname = f"({ptrname} - {sym2cpp(data_descriptor.start_offset)})"

                callsite_stream.write(f"DACE_GPU_CHECK({backend}Free({ptrname}));\n", pool_sdfg)
                callsite_stream.write(generate_sync_debug_call())

                handled_keys.add((pool_sdfg, name))

            # Deferred so we don't mutate the dict while iterating.
            for key in handled_keys:
                del self.pool_release[key]

        for instr in self._frame._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_end(sdfg, cfg, state, callsite_stream, function_stream)

    def generate_node(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Node,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):

        if not self._in_device_code:
            self.synchronize_host_reads(cfg, state_id, node, callsite_stream)

        # Exact type, not isinstance: subclasses (e.g. RTLTasklet) belong to the CPU codegen. Host
        # nodes reach this dispatch only for their synchronization and are generated by the CPU one.
        if type(node) is nodes.NestedSDFG and self._in_device_code:
            self._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node) is nodes.Tasklet and self._in_device_code:
            self._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node) is nodes.MapExit and node.schedule in dtypes.EXPERIMENTAL_GPU_SCHEDULES:
            # A GPU MapExit is closed by the kernel's scope manager; suppress the CPU fallback.
            return
        else:
            self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label):
        return "DACE_DFI " + self._cpu_codegen.generate_nsdfg_header(
            sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=False
        )

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(
            sdfg, cfg, state, node, memlet_references, sdfg_label, state_struct=False
        )

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        args = self._cpu_codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state, node)
        return args

    def _generate_NestedSDFG(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        dispatcher: TargetDispatcher = self._dispatcher
        dispatcher.defined_vars.enter_scope(node)

        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        dispatcher.defined_vars.exit_scope(node)

        self._cpu_codegen.calling_codegen = old_codegen

    def _generate_Tasklet(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Tasklet,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import ScopeManager

        tasklet: nodes.Tasklet = node
        with ScopeManager(
            sdfg, cfg, dfg, state_id, function_stream, callsite_stream, brackets_on_enter=False
        ) as scope_manager:
            # ``location`` guards run the tasklet on a specific slice of threads/warps/blocks.
            for name, index_expr in location_index_exprs(self.current_kernel_spec.block_dims):
                if name in tasklet.location:
                    scope_manager.open(condition=location_condition(name, index_expr, tasklet.location[name]))

            # Tag this as device (.cu) generation so the delegate's generated-function dedup keys on
            # the .cu owner, not the host TU -- otherwise a ``<name>_idx`` helper flushed here lands in
            # the .cu under the host key and is re-emitted under the device key = a C++ redefinition.
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self
            try:
                self._cpu_codegen._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            finally:
                self._cpu_codegen.calling_codegen = old_codegen

    def declare_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Node,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
    ) -> None:
        node = as_access(node)
        ptrname = ptr(node.data, nodedesc, sdfg, self._frame)
        fsymbols = self._frame.symbols_and_constants(sdfg)

        # ``dfg`` is None iff ``nodedesc`` is non-free-symbol dependent (see
        # DaCeCodeGenerator.determine_allocation_lifetime); skip the
        # ``is_nonfree_sym_dependent`` check when dfg is None and ``nodedesc`` is a View.
        if dfg and not sdutil.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError(
                "declare_array is only for variables that require separate declaration and allocation."
            )

        dynamic_shared = gpu_shared_memory.is_dynamic_shared_memory_buffer(nodedesc)
        if nodedesc.storage == dtypes.StorageType.GPU_Shared and not dynamic_shared:
            raise NotImplementedError(
                f'Shared memory container "{node.data}" is placed in static shared memory, '
                "which requires a size known before the kernel starts"
            )

        if nodedesc.storage == dtypes.StorageType.Register:
            raise ValueError("Dynamic allocation of registers is not allowed")

        if (
            nodedesc.storage not in {dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned}
            and not dynamic_shared
        ):
            raise NotImplementedError(f"CUDA: Unimplemented storage type {nodedesc.storage.name}.")

        if self._dispatcher.declared_arrays.has(ptrname):
            return

        dataname = node.data
        array_ctype = f"{nodedesc.dtype.ctype} *"
        declaration_stream.write(f"{array_ctype} {dataname};\n", cfg, state_id, node)
        self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, array_ctype)

    def allocate_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Node,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ) -> None:
        """Declare and allocate a data container, dispatching on its storage type.

        Views and references fall through to the CPU codegen.  The actual allocation for
        GPU/CPU-pinned/shared arrays is delegated to ``_prepare_<storage>_array``.
        """
        node = as_access(node)
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        if self._dispatcher.defined_vars.has(dataname):
            return

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError("allocate_stream not implemented in ExperimentalCUDACodeGen")

        elif isinstance(nodedesc, dace.data.View):
            self._cpu_codegen.allocate_view(
                sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream, allocation_stream
            )
            if node.setzero and nodedesc.storage == dtypes.StorageType.GPU_Shared:
                # A container placed in dynamic shared memory is a view, zeroed where it is allocated
                allocation_stream.write(
                    reset_shared_code(dataname, nodedesc, self.current_kernel_spec.block_dims), cfg, state_id, node
                )
            return
        elif isinstance(nodedesc, dace.data.Reference):
            return self._cpu_codegen.allocate_reference(
                sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream, allocation_stream
            )

        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # gpuStream_t handles are materialised by the GPU stream manager, not here.
        if nodedesc.dtype == dtypes.gpuStream_t:
            return

        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            self.prepare_GPU_Global_array(
                sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream
            )
        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            self.prepare_CPU_Pinned_array(
                sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream
            )
        elif nodedesc.storage == dtypes.StorageType.GPU_Shared:
            self.prepare_GPU_Shared_array(
                sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream
            )
        else:
            raise NotImplementedError(f"CUDA: Unimplemented storage type {nodedesc.storage}")

    def declare_pointer_if_needed(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        declaration_stream: CodeIOStream,
    ) -> str:
        """Emit ``T* {name};`` once and register the host pointer in ``defined_vars``.

        Hoist the binding above ``SDFGState`` scopes (which are popped between
        states) so a Scope-lifetime transient declared at SDFG scope and
        allocated at first-state scope stays visible to the consuming state.
        Stay at the current scope when it is already an ``SDFG`` (nested SDFG
        codegen) -- its ``can_access_parent=False`` blocks the outer frame.
        """
        from dace.sdfg.state import SDFGState

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        array_ctype = f"{nodedesc.dtype.ctype} *"
        if not self._dispatcher.declared_arrays.has(dataname):
            declaration_stream.write(f"{array_ctype} {dataname};\n", cfg, state_id, node)
        if not self._dispatcher.defined_vars.has(dataname):
            topmost_parent, _, _ = self._dispatcher.defined_vars._scopes[-1]
            ancestor = 1 if isinstance(topmost_parent, SDFGState) else 0
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype, ancestor=ancestor)
        return dataname

    def prepare_GPU_Global_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ):
        dataname = self.declare_pointer_if_needed(sdfg, cfg, state_id, node, nodedesc, declaration_stream)
        arrsize_malloc = f"{sym2cpp(nodedesc.total_size)} * sizeof({nodedesc.dtype.ctype})"
        assert isinstance(nodedesc, (dt.Array, dt.Scalar))  # the only GPU_Global descriptors with ``pool``

        if nodedesc.pool:
            gpu_stream = assigned_stream_expr(node)
            allocation_stream.write(
                f"DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {gpu_stream}));\n",
                cfg,
                state_id,
                node,
            )
            allocation_stream.write(generate_sync_debug_call())
        else:
            allocation_stream.write(
                f"DACE_GPU_CHECK({self.backend}Malloc((void**)&{dataname}, {arrsize_malloc}));\n", cfg, state_id, node
            )

        if node.setzero:
            allocation_stream.write(
                f"DACE_GPU_CHECK({self.backend}Memset({dataname}, 0, {arrsize_malloc}));\n", cfg, state_id, node
            )
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            allocation_stream.write(f"{dataname} += {sym2cpp(nodedesc.start_offset)};\n", cfg, state_id, node)

    def prepare_CPU_Pinned_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ):
        dataname = self.declare_pointer_if_needed(sdfg, cfg, state_id, node, nodedesc, declaration_stream)
        arrsize_malloc = f"{sym2cpp(nodedesc.total_size)} * sizeof({nodedesc.dtype.ctype})"

        allocation_stream.write(
            f"DACE_GPU_CHECK({self.backend}MallocHost(&{dataname}, {arrsize_malloc}));\n", cfg, state_id, node
        )
        if node.setzero:
            allocation_stream.write(f"memset({dataname}, 0, {arrsize_malloc});\n", cfg, state_id, node)
        assert isinstance(nodedesc, (dt.Array, dt.Scalar, dt.Stream))
        if nodedesc.start_offset != 0:
            allocation_stream.write(f"{dataname} += {sym2cpp(nodedesc.start_offset)};\n", cfg, state_id, node)

    def prepare_GPU_Shared_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ):

        if gpu_shared_memory.is_dynamic_shared_memory_buffer(nodedesc):
            # The flat buffer of dynamic shared memory that the kernel's dynamic containers view
            dataname = self.declare_pointer_if_needed(sdfg, cfg, state_id, node, nodedesc, declaration_stream)
            allocation_stream.write(f"{dataname} = {_DYNAMIC_SHARED_MEMORY_SYMBOL};\n", cfg, state_id, node)
            return
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        arrsize = nodedesc.total_size
        assert isinstance(nodedesc, (dt.Array, dt.Scalar, dt.Stream))
        if nodedesc.start_offset != 0:
            raise NotImplementedError("Start offset unsupported for shared memory")

        array_ctype = f"{nodedesc.dtype.ctype} *"

        declaration_stream.write(
            f"__shared__ {nodedesc.dtype.ctype} {dataname}[{sym2cpp(arrsize)}];\n", cfg, state_id, node
        )

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        if node.setzero:
            allocation_stream.write(
                reset_shared_code(dataname, nodedesc, self.current_kernel_spec.block_dims), cfg, state_id, node
            )

    def deallocate_array(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.Node,
        nodedesc: dt.Data,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        node = as_access(node)
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f"({dataname} - {sym2cpp(nodedesc.start_offset)})"

        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in GLOBAL_LIFETIMES
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError("stream code is not implemented in ExperimentalCUDACodeGen (yet)")

        if isinstance(nodedesc, dace.data.View):
            return

        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            assert isinstance(nodedesc, (dt.Array, dt.Scalar))
            if nodedesc.pool:
                # Pooled arrays whose release point was picked up by compute_pool_release are
                # freed in generate_state; everything else is freed here.
                if (sdfg, node.data) not in self.pool_released_early:
                    gpu_stream = assigned_stream_expr(node)
                    callsite_stream.write(
                        f"DACE_GPU_CHECK({self.backend}FreeAsync({dataname}, {gpu_stream}));\n", cfg, state_id, node
                    )
            else:
                callsite_stream.write(f"DACE_GPU_CHECK({self.backend}Free({dataname}));\n", cfg, state_id, node)

        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            if nodedesc.dtype == dtypes.gpuStream_t:
                return
            callsite_stream.write(f"DACE_GPU_CHECK({self.backend}FreeHost({dataname}));\n", cfg, state_id, node)

        elif nodedesc.storage in {dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register}:
            return

        else:
            raise NotImplementedError(f"Deallocation not implemented for storage type: {nodedesc.storage.name}")

    def get_generated_codeobjects(self):
        if cpf_lowering.device():
            # CPF renders one unit: only globals, kernels and launch wrappers; CPF supplies the rest
            # (see :func:`~dace.codegen.cpf.device_prologue`).
            fileheader = CodeIOStream()
            self._frame.generate_fileheader(self._global_sdfg, fileheader, "cuda")
            self._codeobject.code = (
                f"{fileheader.getvalue()}\n{self._globalcode.getvalue()}\n{self._localcode.getvalue()}"
            )
            return [self._codeobject]
        stream_create = stream_destroy = None
        if config_int("compiler", "cuda", "max_concurrent_streams") == -1:
            # Every stream is the default (null) stream.
            stream_create = "__state->gpu_context->internal_streams[i] = nullptr"
            stream_destroy = "{ /* no action needed */ }"
        self._codeobject.code = gpu_runtime_code(
            self._frame,
            self._global_sdfg,
            "experimental_cuda",
            self.backend,
            self.has_pool,
            self._initcode,
            self._exitcode,
            self._globalcode.getvalue(),
            self._localcode.getvalue(),
            self._num_gpu_streams,
            0,
            stream_create,
            stream_destroy,
        )
        return [self._codeobject]

    @staticmethod
    def cmake_options():
        return gpu_cmake_options()

    def define_out_memlet(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state_dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge[Memlet],
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        self._cpu_codegen.define_out_memlet(
            sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream
        )

    def process_out_memlets(self, *args, **kwargs):
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)


class KernelSpec:
    """Kernel metadata (name, grid/block dims, argument forms, warp size) used by
    ``ExperimentalCUDACodeGen`` to emit the ``__global__`` and its host launch wrapper.
    """

    def __init__(
        self,
        cudaCodeGen: ExperimentalCUDACodeGen,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
    ):

        kernel_map_entry = scope_map_entry(dfg_scope)
        kernel_parent_state: SDFGState = cfg.state(state_id)

        self.kernel_map_entry: nodes.MapEntry = kernel_map_entry
        self.kernel_map: nodes.Map = kernel_map_entry.map
        # Label and ids are unique only within one SDFG; the top-level SDFG name keeps two programs
        # linked into one process (e.g. torch forward + backward) from binding each other's kernels.
        self.kernel_name: str = (
            f"{cudaCodeGen._global_sdfg.name}_{kernel_map_entry.map.label}_{cfg.cfg_id}"
            f"_{kernel_parent_state.block_id}_{kernel_parent_state.node_id(kernel_map_entry)}"
        )

        self.arglist: dict[str, dt.Data] = cudaCodeGen._kernel_arglists[kernel_map_entry]

        kernel_const_data = kernel_read_only_data(kernel_map_entry, kernel_parent_state)
        kernel_const_symbols = sdutil.get_constant_symbols(kernel_map_entry, kernel_parent_state)
        # A pointer (Array/View) arg may be ``const`` ONLY when it is read-only in this kernel, i.e. in
        # ``kernel_const_data`` (read-set minus write-set). ``get_constant_symbols`` can surface a WRITTEN
        # data container's name as a "constant symbol" -- its MapEntry branch returns
        # ``used_symbols_within_scope`` (which includes data names used in subset/offset expressions) and,
        # unlike the CFG branches, never subtracts writes. Letting such a name into ``kernel_constants``
        # const-qualifies a written output pointer -> ``expression must be a modifiable lvalue``. Drop any
        # pointer arg that is not genuinely read-only; scalar-symbol args are unaffected.
        written_pointers = {
            name
            for name, data in self.arglist.items()
            if isinstance(data, (dt.Array, dt.View)) and name not in kernel_const_data
        }
        self.kernel_constants: set[str] = (kernel_const_data | kernel_const_symbols) - written_pointers

        restore_in_device_code = cudaCodeGen._in_device_code

        # ptr() resolves a different name on the device side (persistent arrays live in __state);
        # toggle the flag so we capture the device-side pointer name here.
        cudaCodeGen._in_device_code = True
        self.args_as_input: list[str] = [
            ptr(name, data, sdfg, cudaCodeGen._frame) for name, data in self.arglist.items()
        ]

        args_typed = []
        for name, data in self.arglist.items():
            if data.lifetime == dtypes.AllocationLifetime.Persistent:
                arg_name = ptr(name, data, sdfg, cudaCodeGen._frame)
            else:
                arg_name = name
            args_typed.append(("const " if name in self.kernel_constants else "") + data.as_arg(name=arg_name))
        self.args_typed: list[str] = args_typed

        cudaCodeGen._in_device_code = False

        # The kernel wrapper function runs on the host; its signature receives __state,
        # every kernel argument, and exactly one gpuStream_t handle.
        gpustream_var_name = config_str("compiler", "cuda", "gpu_stream_name").split(",")[1]
        # Resolve the descriptor from the memlet, not from ``e.src``: when the kernel map sits
        # inside a host-scheduled map the stream edge is routed through the enclosing MapEntry,
        # so ``e.src`` is that MapEntry rather than the gpu_streams AccessNode.
        gpustream_input = [
            e
            for e in dace.sdfg.dynamic_map_inputs(kernel_parent_state, kernel_map_entry)
            if e.data.data is not None and sdfg.arrays[e.data.data].dtype == dtypes.gpuStream_t
        ]
        if len(gpustream_input) > 1:
            raise ValueError(
                f"There can not be more than one GPU stream assigned to a kernel, but {len(gpustream_input)} were assigned."
            )

        # If no stream edge was wired to this kernel (e.g. the kernel sits inside a
        # libnode-expanded NestedSDFG whose stream chain hasn't been propagated past
        # expansion), launch on the default stream (CUDA stream 0 / ``nullptr``).
        stream_arg = str(gpustream_input[0].dst_conn) if gpustream_input else "nullptr"

        self.kernel_wrapper_args_as_input: list[str] = (
            ["__state"]
            + [ptr(name, data, sdfg, cudaCodeGen._frame) for name, data in self.arglist.items()]
            + [stream_arg]
        )

        self.kernel_wrapper_args_typed: list[str] = (
            [f"{mangle_dace_state_struct_name(cudaCodeGen._global_sdfg)} *__state"]
            + args_typed
            + [f"gpuStream_t {gpustream_var_name}"]
        )

        cudaCodeGen._in_device_code = restore_in_device_code

        # A kernel created after AddThreadBlockMaps (GridStrideKernels' device-sized map) is inferred here; entries
        # computed in preprocess, which know the inserted thread-block maps, take precedence.
        if kernel_map_entry not in cudaCodeGen._kernel_dimensions_map:
            from dace.transformation.passes.analysis.infer_gpu_grid_and_block_size import InferGPUGridAndBlockSize

            inferred = InferGPUGridAndBlockSize().infer(kernel_parent_state.sdfg, set()) or {}
            if kernel_map_entry in inferred:
                cudaCodeGen._kernel_dimensions_map[kernel_map_entry] = inferred[kernel_map_entry]
        self.grid_dims, self.block_dims = cudaCodeGen._kernel_dimensions_map[kernel_map_entry]
        # Without a thread-block map, a kernel's own map spans the threads rather than the blocks
        self.per_thread: bool = not any(
            m.schedule == dtypes.ScheduleType.GPU_ThreadBlock
            for m, _ in gpu_scope_maps_recursive(kernel_parent_state.scope_subgraph(kernel_map_entry))
        )
        self.chiplets: int = chiplet_count(kernel_map_entry, cudaCodeGen.backend, False, False, [])
        self.grid_dims, self.chiplet_chunk = distribute_over_chiplets(
            kernel_map_entry, list(self.grid_dims), self.chiplets
        )
        self.index_types: dict[str, dtypes.typeclass] = common.gpu_map_index_types(
            sdfg,
            kernel_parent_state,
            kernel_map_entry,
            cudaCodeGen._frame.symbols_defined_at(kernel_parent_state, kernel_map_entry),
        )

        if cudaCodeGen.backend not in ["cuda", "hip"]:
            raise ValueError(
                f"Unsupported backend '{cudaCodeGen.backend}' in ExperimentalCUDACodeGen. "
                "Only 'cuda' and 'hip' are supported."
            )
