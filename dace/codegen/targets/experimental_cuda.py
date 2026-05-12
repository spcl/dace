# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

import dace
from dace import data as dt, Memlet
from dace import dtypes, registry, symbolic, subsets
from dace.config import Config
from dace.sdfg import SDFG, ScopeSubgraphView, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.scope import get_node_schedule
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

from dace.codegen import common
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (codeblock_to_cpp, mangle_dace_state_struct_name, ptr, sym2cpp)
from dace.codegen.target import TargetCodeGenerator, make_absolute

from dace.transformation.passes import analysis as ap
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import read_stream_assignments_from_wired_sdfg
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync

from dace.codegen.targets.experimental_cuda_helpers.gpu_stream_manager import GPUStreamManager
from dace.codegen.targets.experimental_cuda_helpers.gpu_utils import generate_sync_debug_call

from dace.codegen.targets import cpp

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen


@registry.autoregister_params(name='experimental_cuda')
class ExperimentalCUDACodeGen(TargetCodeGenerator):
    """Experimental CUDA code generator."""
    target_name = 'experimental_cuda'
    title = 'CUDA'

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):

        self._frame: DaCeCodeGenerator = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher

        self._in_device_code = False
        self._cpu_codegen: Optional['CPUCodeGen'] = None

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
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()

        self._global_sdfg: SDFG = sdfg
        self._toplevel_schedule = None

        self.pool_release: Dict[Tuple[SDFG, str], Tuple[SDFGState, Set[nodes.Node]]] = {}
        self.has_pool = False

        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        self._dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN, self)
        self._dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]
        self._dispatcher.register_array_dispatcher(gpu_storage, self)
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)
        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                self._dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self._dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        self._current_kernel_spec: Optional[KernelSpec] = None
        self._gpu_stream_manager: Optional[GPUStreamManager] = None
        self._kernel_dimensions_map: Dict[nodes.MapEntry, Tuple[List, List]] = {}
        self._tb_inserted_kernels: Set[nodes.MapEntry] = set()
        self._kernel_arglists: Dict[nodes.MapEntry, Dict[str, dt.Data]] = {}

    def preprocess(self, sdfg: SDFG):
        """Prepare the SDFG for GPU code generation.

        All SDFG-level transformation lives in
        :class:`GPUCodegenPreprocessPipeline`. This method only does
        framecode-target bookkeeping: the ``gpu_context`` statestruct
        entry, kernel-dimension cache hand-off, frame symbol cache rebuild,
        ``GPUStreamManager`` construction, pool-release computation, and
        the per-kernel arglist build.
        """
        self._frame.statestruct.append('dace::cuda::Context *gpu_context;')
        self._dispatcher._used_targets.add(self)

        pipeline_results: Dict[str, Any] = {}
        GPUCodegenPreprocessPipeline().apply_pass(sdfg, pipeline_results)

        # The ``AddThreadBlockMaps`` Pass returns the kernel-dimension
        # map and the set of kernels it tiled; the codegen consults both
        # when emitting kernel launches.
        atb_results = pipeline_results.get('AddThreadBlockMaps', {}) or {}
        self._kernel_dimensions_map = atb_results.get('kernel_dimensions_map', {})
        self._tb_inserted_kernels = atb_results.get('tb_inserted_kernels', set())

        # Library-node expansion adds new nested SDFGs with new cfg_ids; re-seed
        # the framecode's symbol/constant cache so lookups succeed for them.
        self._rebuild_frame_symbol_cache(sdfg)

        # Strategy stamps the WCC assignment dict on the SDFG; codegen
        # consumers (memory-pool path needs AccessNode stream ids, not
        # just wired-consumer ids) read it from there. Pre-lowered
        # fixtures fall back to reading consumers from wired connectors.
        gpustream_assignments = (getattr(sdfg, '_gpu_stream_assignments', None)
                                 or read_stream_assignments_from_wired_sdfg(sdfg))
        self._gpu_stream_manager = GPUStreamManager(sdfg, gpustream_assignments)

        if Config.get('compiler', 'cuda', 'auto_syncthreads_insertion'):
            DefaultSharedMemorySync().apply_pass(sdfg, None)

        self._compute_pool_release(sdfg)

        shared_transients = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(sdfg, recursive=True):
            if (isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device):
                if state.parent not in shared_transients:
                    shared_transients[state.parent] = state.parent.shared_transients()
                self._kernel_arglists[node] = state.scope_subgraph(node).arglist(defined_syms,
                                                                                 shared_transients[state.parent])

    def _rebuild_frame_symbol_cache(self, sdfg: SDFG):
        """Re-seed the framecode's symbol/constant cache for the current SDFG hierarchy.

        Needed whenever ``preprocess`` adds new nested SDFGs -- the cache is keyed
        by ``cfg_id`` and populated once in the framecode's constructor.
        """
        frame = self._frame
        frame._symbols_and_constants = {}
        sdfg.reset_cfg_list()
        frame._symbols_and_constants[sdfg.cfg_id] = sdfg.free_symbols.union(sdfg.constants_prop.keys())
        for nested, state in sdfg.all_nodes_recursive():
            if isinstance(nested, nodes.NestedSDFG):
                nsdfg = nested.sdfg
                result = nsdfg.free_symbols.union(nsdfg.constants_prop.keys())
                parent_constants = frame._symbols_and_constants[nsdfg.parent_sdfg.cfg_id]
                result |= parent_constants
                for edge in state.in_edges(nested):
                    if edge.data.data in parent_constants:
                        result.add(edge.dst_conn)
                frame._symbols_and_constants[nsdfg.cfg_id] = result

    def _compute_pool_release(self, top_sdfg: SDFG):
        """Find the point at which each pooled array should be released (``cudaFreeAsync``).

        :raises ValueError: if the backend does not support memory pools.
        """
        reachability = access_nodes = None
        for sdfg in top_sdfg.all_sdfgs_recursive():
            pooled = set(aname for aname, arr in sdfg.arrays.items()
                         if getattr(arr, 'pool', False) is True and arr.transient)
            if not pooled:
                continue
            self.has_pool = True
            if self.backend != 'cuda':
                raise ValueError(f'Backend "{self.backend}" does not support the memory pool allocation hint')

            pooled = filter(
                lambda aname: sdfg.arrays[aname].lifetime in
                (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.
                 External), pooled)

            if reachability is None:
                reachability = ap.StateReachability().apply_pass(top_sdfg, {})
                access_nodes = ap.FindAccessStates().apply_pass(top_sdfg, {})

            reachable = reachability[sdfg.cfg_id]
            access_sets = access_nodes[sdfg.cfg_id]
            for state in sdfg.states():
                last_state_arrays: Set[str] = set(
                    s for s in access_sets
                    if s in pooled and state in access_sets[s] and not (access_sets[s] & reachable[state]) - {state})

                anodes = list(state.data_nodes())
                for aname in last_state_arrays:
                    ans = [an for an in anodes if an.data == aname]
                    terminator = None
                    for an1 in ans:
                        if all(nx.has_path(state.nx, an2, an1) for an2 in ans if an2 is not an1):
                            terminator = an1
                            break

                    # Release at end of the last memlet path out of the terminator access node;
                    # if the terminator sits inside a scope, defer release to the end of state.
                    terminators = set()
                    if terminator is not None:
                        parent = state.entry_node(terminator)
                        if parent is not None:
                            terminators = set()
                        else:
                            # Otherwise, find common descendant (or end of state) following the ends of
                            # all memlet paths (e.g., (a)->...->[tasklet]-->...->(b))
                            for e in state.out_edges(terminator):
                                if isinstance(e.dst, nodes.EntryNode):
                                    terminators.add(state.exit_node(e.dst))
                                else:
                                    terminators.add(e.dst)

                    self.pool_release[(sdfg, aname)] = (state, terminators)

            # Release anything still live at SDFG sink.
            unfreed = set(arr for arr in pooled if (sdfg, arr) not in self.pool_release)
            if unfreed:
                sinks = sdfg.sink_nodes()
                if len(sinks) == 1:
                    sink = sinks[0]
                elif len(sinks) > 1:
                    sink = sdfg.add_state()
                    for s in sinks:
                        sdfg.add_edge(s, sink)
                else:
                    raise ValueError('End state not found when trying to free pooled memory')

                for arr in unfreed:
                    self.pool_release[(sdfg, arr)] = (sink, set())

    @property
    def has_initializer(self) -> bool:
        return True

    @property
    def has_finalizer(self) -> bool:
        return True

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import (ScopeGenerationStrategy,
                                                                                     KernelScopeGenerator,
                                                                                     ThreadBlockScopeGenerator,
                                                                                     WarpScopeGenerator)
        scope_entry = dfg_scope.source_nodes()[0]

        if not self._in_device_code:

            state = cfg.state(state_id)
            scope_entry = dfg_scope.source_nodes()[0]
            scope_exit = dfg_scope.sink_nodes()[0]
            scope_entry_stream = CodeIOStream()
            scope_exit_stream = CodeIOStream()

            instr = self._dispatcher.instrumentation[scope_entry.map.instrument]
            if instr is not None:
                instr.on_scope_entry(sdfg, cfg, state, scope_entry, callsite_stream, scope_entry_stream,
                                     self._globalcode)
                outer_stream = CodeIOStream()
                instr.on_scope_exit(sdfg, cfg, state, scope_exit, outer_stream, scope_exit_stream, self._globalcode)

            self._dispatcher.defined_vars.enter_scope(scope_entry)

            kernel_spec = KernelSpec(cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id)
            self._current_kernel_spec = kernel_spec

            self._define_variables_in_kernel_scope(sdfg, self._dispatcher)
            self._declare_and_invoke_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            kernel_stream = CodeIOStream()
            kernel_function_stream = self._globalcode

            self._in_device_code = True

            kernel_scope_generator = KernelScopeGenerator(codegen=self)
            if kernel_scope_generator.applicable(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream):
                kernel_scope_generator.generate(sdfg, cfg, dfg_scope, state_id, kernel_function_stream, kernel_stream)
            else:
                raise ValueError("Invalid kernel configuration: This strategy is only applicable if the "
                                 "outermost GPU schedule is of type GPU_Device (most likely cause).")

            self._localcode.write(scope_entry_stream.getvalue())
            self._localcode.write(kernel_stream.getvalue() + '\n')
            self._localcode.write(scope_exit_stream.getvalue())

            self._in_device_code = False

            self._generate_kernel_wrapper(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            self._dispatcher.defined_vars.exit_scope(scope_entry)

            if instr is not None:
                callsite_stream.write(outer_stream.getvalue())

            return

        # Nested GPU scope.
        supported_strategies: List[ScopeGenerationStrategy] = [
            ThreadBlockScopeGenerator(codegen=self),
            WarpScopeGenerator(codegen=self)
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
            "Please check for supported schedule types or implement the corresponding strategy.")

    def _define_variables_in_kernel_scope(self, sdfg: SDFG, dispatcher: TargetDispatcher):
        """Register every kernel argument in the dispatcher under its device-side pointer name.

        Persistent/external data that lives in ``__state`` cannot be referenced directly from
        device code -- it is passed as a kernel argument, and the dispatcher needs to resolve
        accesses through the device pointer.  Constants pick up a ``const`` ctype qualifier.
        """
        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_constants: Set[str] = kernel_spec.kernel_constants
        kernel_arglist: Dict[str, dt.Data] = kernel_spec.arglist

        restore_in_device_code = self._in_device_code
        for name, data_desc in kernel_arglist.items():
            if not name in sdfg.arrays:
                continue

            data_desc = sdfg.arrays[name]
            self._in_device_code = False
            host_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            is_global: bool = data_desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                     dtypes.AllocationLifetime.Persistent,
                                                     dtypes.AllocationLifetime.External)
            defined_type, ctype = dispatcher.defined_vars.get(host_ptrname, is_global=is_global)

            self._in_device_code = True
            device_ptrname = cpp.ptr(name, data_desc, sdfg, self._frame)

            if name in kernel_constants and "const " not in ctype:
                ctype = f"const {ctype}"

            dispatcher.defined_vars.add(device_ptrname, defined_type, ctype, allow_shadowing=True)

        self._in_device_code = restore_in_device_code

    def _declare_and_invoke_kernel_wrapper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView,
                                           state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        scope_entry = dfg_scope.source_nodes()[0]

        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_wrapper_args_as_input = kernel_spec.kernel_wrapper_args_as_input
        kernel_wrapper_args_typed = kernel_spec.kernel_wrapper_args_typed

        function_stream.write(
            'DACE_EXPORTED void __dace_runkernel_%s(%s);\n' % (kernel_name, ', '.join(kernel_wrapper_args_typed)), cfg,
            state_id, scope_entry)

        # Wrap the invocation in a block so dynamic-input local declarations don't leak.
        state = cfg.state(state_id)
        dyn_inputs = list(dace.sdfg.dynamic_map_inputs(state, scope_entry))
        has_dyn_inputs = len(dyn_inputs) > 0
        if has_dyn_inputs:
            callsite_stream.write('{', cfg, state_id, scope_entry)

        for e in dyn_inputs:
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg, state_id, scope_entry)

        callsite_stream.write('__dace_runkernel_%s(%s);\n' % (kernel_name, ', '.join(kernel_wrapper_args_as_input)),
                              cfg, state_id, scope_entry)

        if has_dyn_inputs:
            callsite_stream.write('}', cfg, state_id, scope_entry)

    def _generate_kernel_wrapper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        scope_entry = dfg_scope.source_nodes()[0]

        kernel_spec: KernelSpec = self._current_kernel_spec
        kernel_name = kernel_spec.kernel_name
        kernel_args_as_input = kernel_spec.args_as_input
        kernel_launch_args_typed = kernel_spec.kernel_wrapper_args_typed

        grid_dims = kernel_spec.grid_dims
        block_dims = kernel_spec.block_dims
        gdims = ', '.join(sym2cpp(grid_dims))
        bdims = ', '.join(sym2cpp(block_dims))

        self._localcode.write(
            f"""
            DACE_EXPORTED void __dace_runkernel_{kernel_name}({', '.join(kernel_launch_args_typed)});
            void __dace_runkernel_{kernel_name}({', '.join(kernel_launch_args_typed)})
            """, cfg, state_id, scope_entry)

        self._localcode.write('{', cfg, state_id, scope_entry)

        # Skip launches on empty or negative-sized grids that we can't prove non-empty statically.
        single_dimchecks = []
        for gdim in grid_dims:
            if (gdim > 0) != True:
                single_dimchecks.append(f'(({sym2cpp(gdim)}) <= 0)')

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

        stream_var_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[1]
        kargs = ', '.join(['(void *)&' + arg for arg in kernel_args_as_input])
        self._localcode.write(
            f'''
            void  *{kernel_name}_args[] = {{ {kargs} }};
            gpuError_t __err = {self.backend}LaunchKernel(
                (void*){kernel_name}, dim3({gdims}), dim3({bdims}), {kernel_name}_args, {0}, {stream_var_name}
            );
            ''', cfg, state_id, scope_entry)

        self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});\n')
        self._localcode.write(generate_sync_debug_call())

        self._localcode.write('}', cfg, state_id, scope_entry)

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    edge: Tuple[nodes.Node, str, nodes.Node, str,
                                Memlet], function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        # All CPU↔GPU and GPU↔GPU AccessNode→AccessNode edges (host-issued
        # and in-kernel collaborative) are lifted to ``CopyLibraryNode`` by
        # ``InsertExplicitGPUGlobalMemoryCopies`` during ``preprocess()`` and
        # lowered through their expansions. Anything reaching this dispatch
        # is a register / scope-local CPU copy — delegate to CPU codegen.
        self._cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)

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
        schedule = getattr(node, 'schedule', None)
        if schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            return True
        if self._in_device_code:
            return True
        return False

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = False):

        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

        # Emit cudaFree for pooled transients whose lifetime ends in this state.
        if not self._in_device_code:

            handled_keys = set()
            backend = self.backend
            for (pool_sdfg, name), (pool_state, _) in self.pool_release.items():

                if (pool_sdfg is not sdfg) or (pool_state is not state):
                    continue

                data_descriptor = pool_sdfg.arrays[name]
                ptrname = ptr(name, data_descriptor, pool_sdfg, self._frame)

                if isinstance(data_descriptor, dt.Array) and data_descriptor.start_offset != 0:
                    ptrname = f'({ptrname} - {sym2cpp(data_descriptor.start_offset)})'

                callsite_stream.write(f'DACE_GPU_CHECK({backend}Free({ptrname}));\n', pool_sdfg)
                callsite_stream.write(generate_sync_debug_call())

                handled_keys.add((pool_sdfg, name))

            # Deferred so we don't mutate the dict while iterating.
            for key in handled_keys:
                del self.pool_release[key]

        # Invoke all instrumentation providers
        for instr in self._frame._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_end(sdfg, cfg, state, callsite_stream, function_stream)

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream):

        gen = getattr(self, '_generate_' + type(node).__name__, False)

        if gen is not False:
            gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node).__name__ == 'MapExit' and node.schedule in dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN:
            # A GPU MapExit is closed by the kernel's scope manager; suppress the CPU fallback.
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
                             node: nodes.NestedSDFG, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        old_schedule = self._toplevel_schedule
        nested_schedule = get_node_schedule(sdfg, dfg, node)
        if nested_schedule != dtypes.ScheduleType.Default:
            self._toplevel_schedule = nested_schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        dispatcher: TargetDispatcher = self._dispatcher
        dispatcher.defined_vars.enter_scope(node)

        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        dispatcher.defined_vars.exit_scope(node)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule

    def _generate_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.Tasklet, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import ScopeManager

        tasklet: nodes.Tasklet = node
        with ScopeManager(self, sdfg, cfg, dfg, state_id, function_stream, callsite_stream,
                          brackets_on_enter=False) as scope_manager:

            # ``location`` guards run the tasklet on a specific slice of threads/warps/blocks.
            for name, index_fn in (('gpu_thread', self._get_thread_id), ('gpu_warp', self._get_warp_id),
                                   ('gpu_block', self._get_block_id)):
                if name in tasklet.location:
                    cond = self._generate_condition_from_location(name, index_fn(), tasklet.location[name])
                    scope_manager.open(condition=cond)

            self._cpu_codegen._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def _generate_condition_from_location(self, name: str, index_expr: str, location: Union[int, str,
                                                                                            subsets.Range]) -> str:
        if isinstance(location, str) and ':' in location:
            location = subsets.Range.from_string(location)
            if len(location) != 1:
                raise ValueError(f'Only one-dimensional ranges are allowed for {name} specialization, {location} given')
        elif symbolic.issymbolic(location):
            location = sym2cpp(location)

        if isinstance(location, subsets.Range):
            begin, end, stride = location[0]
            rb, re, rs = sym2cpp(begin), sym2cpp(end), sym2cpp(stride)
            cond = f'(({index_expr}) >= {rb}) && (({index_expr}) <= {re})'
            if stride != 1:
                cond += f' && ((({index_expr}) - {rb}) % {rs} == 0)'
        else:
            cond = f'({index_expr}) == {location}'

        return cond

    def _get_thread_id(self) -> str:
        kernel_block_dims: List = self._current_kernel_spec.block_dims
        result = 'threadIdx.x'
        if kernel_block_dims[1] != 1:
            result += f' + ({sym2cpp(kernel_block_dims[0])}) * threadIdx.y'
        if kernel_block_dims[2] != 1:
            result += f' + ({sym2cpp(kernel_block_dims[0] * kernel_block_dims[1])}) * threadIdx.z'
        return result

    def _get_warp_id(self) -> str:
        return f'(({self._get_thread_id()}) / warpSize)'

    def _get_block_id(self) -> str:
        kernel_block_dims: List = self._current_kernel_spec.block_dims
        result = 'blockIdx.x'
        if kernel_block_dims[1] != 1:
            result += f' + gridDim.x * blockIdx.y'
        if kernel_block_dims[2] != 1:
            result += f' + gridDim.x * gridDim.y * blockIdx.z'
        return result

    #######################################################################
    # Array Declaration, Allocation and Deallocation

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                      declaration_stream: CodeIOStream):

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
            return

        dataname = node.data
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
        self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, array_ctype)

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        """Declare and allocate a data container, dispatching on its storage type.

        Views and references fall through to the CPU codegen.  The actual allocation for
        GPU/CPU-pinned/shared arrays is delegated to ``_prepare_<storage>_array``.
        """
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

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

        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # gpuStream_t handles are materialised by the GPU stream manager, not here.
        if nodedesc.dtype == dtypes.gpuStream_t:
            return

        gen = getattr(self, f'_prepare_{nodedesc.storage.name}_array', None)
        if gen:
            gen(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
        else:
            raise NotImplementedError(f'CUDA: Unimplemented storage type {nodedesc.storage}')

    def _declare_pointer_if_needed(self, sdfg: SDFG, cfg: ControlFlowRegion, state_id: int, node: nodes.AccessNode,
                                   nodedesc: dt.Data, declaration_stream: CodeIOStream) -> str:
        """Emit ``T* {name};`` once and register it in defined_vars; return the
        resolved data name. No-op if a prior pass already declared it."""
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        if not self._dispatcher.declared_arrays.has(dataname):
            array_ctype = f'{nodedesc.dtype.ctype} *'
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)
        return dataname

    def _prepare_GPU_Global_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        dataname = self._declare_pointer_if_needed(sdfg, cfg, state_id, node, nodedesc, declaration_stream)
        arrsize_malloc = f'{sym2cpp(nodedesc.total_size)} * sizeof({nodedesc.dtype.ctype})'

        if nodedesc.pool:
            gpu_stream = self._gpu_stream_manager.get_stream_node(node)
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {gpu_stream}));\n',
                cfg, state_id, node)
            allocation_stream.write(generate_sync_debug_call())
        else:
            allocation_stream.write(f'DACE_GPU_CHECK({self.backend}Malloc((void**)&{dataname}, {arrsize_malloc}));\n',
                                    cfg, state_id, node)

        if node.setzero:
            allocation_stream.write(f'DACE_GPU_CHECK({self.backend}Memset({dataname}, 0, {arrsize_malloc}));\n', cfg,
                                    state_id, node)
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            allocation_stream.write(f'{dataname} += {sym2cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

    def _prepare_CPU_Pinned_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        dataname = self._declare_pointer_if_needed(sdfg, cfg, state_id, node, nodedesc, declaration_stream)
        arrsize_malloc = f'{sym2cpp(nodedesc.total_size)} * sizeof({nodedesc.dtype.ctype})'

        allocation_stream.write(f'DACE_GPU_CHECK({self.backend}MallocHost(&{dataname}, {arrsize_malloc}));\n', cfg,
                                state_id, node)
        if node.setzero:
            allocation_stream.write(f'memset({dataname}, 0, {arrsize_malloc});\n', cfg, state_id, node)
        if nodedesc.start_offset != 0:
            allocation_stream.write(f'{dataname} += {sym2cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

    def _prepare_GPU_Shared_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                  node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                  declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
        arrsize = nodedesc.total_size

        if symbolic.issymbolic(arrsize, sdfg.constants):
            raise NotImplementedError('Dynamic shared memory unsupported')
        if nodedesc.start_offset != 0:
            raise NotImplementedError('Start offset unsupported for shared memory')

        array_ctype = f'{nodedesc.dtype.ctype} *'

        declaration_stream.write(f'__shared__ {nodedesc.dtype.ctype} {dataname}[{sym2cpp(arrsize)}];\n', cfg, state_id,
                                 node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        if node.setzero:
            allocation_stream.write(
                f'dace::ResetShared<{nodedesc.dtype.ctype}, {", ".join(sym2cpp(self._current_kernel_spec.block_dims))}, {sym2cpp(arrsize)}, '
                f'1, false>::Reset({dataname});\n', cfg, state_id, node)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream):

        dataname = ptr(node.data, nodedesc, sdfg, self._frame)

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f'({dataname} - {sym2cpp(nodedesc.start_offset)})'

        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (
                dtypes.AllocationLifetime.Global,
                dtypes.AllocationLifetime.Persistent,
                dtypes.AllocationLifetime.External,
            )
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError('stream code is not implemented in ExperimentalCUDACodeGen (yet)')

        if isinstance(nodedesc, dace.data.View):
            return

        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if nodedesc.pool:
                # Pooled arrays whose release point was picked up by _compute_pool_release are
                # freed in generate_state; everything else is freed here.
                if (sdfg, dataname) not in self.pool_release:
                    gpu_stream = self._gpu_stream_manager.get_stream_node(node)
                    callsite_stream.write(f'DACE_GPU_CHECK({self.backend}FreeAsync({dataname}, {gpu_stream}));\n', cfg,
                                          state_id, node)
            else:
                callsite_stream.write(f'DACE_GPU_CHECK({self.backend}Free({dataname}));\n', cfg, state_id, node)

        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            if nodedesc.dtype == dtypes.gpuStream_t:
                return
            callsite_stream.write(f'DACE_GPU_CHECK({self.backend}FreeHost({dataname}));\n', cfg, state_id, node)

        elif nodedesc.storage in {dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register}:
            return

        else:
            raise NotImplementedError(f'Deallocation not implemented for storage type: {nodedesc.storage.name}')

    def get_generated_codeobjects(self):
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'cuda')

        # The GPU stream array has a persistent allocation lifetime and is declared in the state
        # struct under an SDFG-id-prefixed name by the frame codegen; resolve the prefixed name so
        # our backend initialization can refer to the same storage.
        cnt = 0
        init_gpu_stream_vars = ""
        gpu_stream_array_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(",")[0]
        for csdfg, name, desc in self._global_sdfg.arrays_recursive(include_nested_data=True):
            if name == gpu_stream_array_name and desc.lifetime == dtypes.AllocationLifetime.Persistent:
                init_gpu_stream_vars = f"__state->__{csdfg.cfg_id}_{name}"
                break

        initcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if 'cuda' in sd.init_code:
                initcode.write(codeblock_to_cpp(sd.init_code['cuda']), sd)
        initcode.write(self._initcode.getvalue())

        exitcode = CodeIOStream()
        for sd in self._global_sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if 'cuda' in sd.exit_code:
                exitcode.write(codeblock_to_cpp(sd.exit_code['cuda']), sd)
        exitcode.write(self._exitcode.getvalue())

        if self.backend == 'cuda':
            backend_header = 'cuda_runtime.h'
        elif self.backend == 'hip':
            backend_header = 'hip/hip_runtime.h'
        else:
            raise NameError('GPU backend "%s" not recognized' % self.backend)

        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        pool_header = ''
        if self.has_pool:
            poolcfg = Config.get('compiler', 'cuda', 'mempool_release_threshold')
            pool_header = f'''
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = {poolcfg if poolcfg != -1 else 'UINT64_MAX'};
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
'''

        self._codeobject.code = """
#include <{backend_header}>
#include <dace/dace.h>

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
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg)

        return [self._codeobject]

    @staticmethod
    def cmake_options():
        options = []

        if Config.get('compiler', 'cuda', 'path'):
            options.append("-DCUDA_TOOLKIT_ROOT_DIR=\"{}\"".format(
                Config.get('compiler', 'cuda', 'path').replace('\\', '/')))

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
            flags += " -G -g"
            flags += ' ' + ' '.join(
                '--offload-arch={arch}'.format(arch=arch if arch.startswith("gfx") else "gfx" + arch)
                for arch in hip_arch)
            options.append("-DEXTRA_HIP_FLAGS=\"{}\"".format(flags))

        if Config.get('compiler', 'cpu', 'executable'):
            host_compiler = make_absolute(Config.get("compiler", "cpu", "executable"))
            options.append("-DCUDA_HOST_COMPILER=\"{}\"".format(host_compiler))

        return options

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        self._cpu_codegen.define_out_memlet(sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
                                            callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)


class KernelSpec:
    """Kernel metadata (name, grid/block dims, arguments) used by ``ExperimentalCUDACodeGen``.

    Public attributes:
      - ``kernel_map_entry``: the ``GPU_Device`` MapEntry that is the kernel's root scope.
      - ``kernel_map``: shorthand for ``kernel_map_entry.map``.
      - ``kernel_name``: function name of the generated ``__global__``.
      - ``kernel_constants``: data + symbols that take a ``const`` qualifier in the kernel signature.
      - ``arglist``: ``{name: descriptor}`` for every kernel argument.
      - ``args_as_input`` / ``args_typed``: kernel-side argument forms (call site / declaration).
      - ``kernel_wrapper_args_as_input`` / ``kernel_wrapper_args_typed``: host-wrapper forms.
      - ``grid_dims`` / ``block_dims``: launch geometry.
      - ``warpSize``: backend warp size from config.
      - ``gpu_index_ctype``: C type for thread/block/warp indices (``compiler.cuda.gpu_index_type``).
    """

    def __init__(self, cudaCodeGen: ExperimentalCUDACodeGen, sdfg: SDFG, cfg: ControlFlowRegion,
                 dfg_scope: ScopeSubgraphView, state_id: int):

        kernel_map_entry: nodes.MapEntry = dfg_scope.source_nodes()[0]
        kernel_parent_state: SDFGState = cfg.state(state_id)

        self.kernel_map_entry: nodes.MapEntry = kernel_map_entry
        self.kernel_map: nodes.Map = kernel_map_entry.map
        self.kernel_name: str = f'{kernel_map_entry.map.label}_{cfg.cfg_id}_{kernel_parent_state.block_id}_{kernel_parent_state.node_id(kernel_map_entry)}'

        kernel_const_data = sdutil.get_constant_data(kernel_map_entry, kernel_parent_state)
        kernel_const_symbols = sdutil.get_constant_symbols(kernel_map_entry, kernel_parent_state)
        self.kernel_constants: Set[str] = kernel_const_data | kernel_const_symbols

        self.arglist: Dict[str, dt.Data] = cudaCodeGen._kernel_arglists[kernel_map_entry]

        restore_in_device_code = cudaCodeGen._in_device_code

        # ptr() resolves a different name on the device side (persistent arrays live in __state);
        # toggle the flag so we capture the device-side pointer name here.
        cudaCodeGen._in_device_code = True
        self.args_as_input: List[str] = [
            ptr(name, data, sdfg, cudaCodeGen._frame) for name, data in self.arglist.items()
        ]

        args_typed = []
        for name, data in self.arglist.items():
            if data.lifetime == dtypes.AllocationLifetime.Persistent:
                arg_name = ptr(name, data, sdfg, cudaCodeGen._frame)
            else:
                arg_name = name
            args_typed.append(('const ' if name in self.kernel_constants else '') + data.as_arg(name=arg_name))
        self.args_typed: List[str] = args_typed

        cudaCodeGen._in_device_code = False

        # The kernel wrapper function runs on the host; its signature receives __state,
        # every kernel argument, and exactly one gpuStream_t handle.
        gpustream_var_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[1]
        gpustream_input = [
            e for e in dace.sdfg.dynamic_map_inputs(kernel_parent_state, kernel_map_entry)
            if e.src.desc(sdfg).dtype == dtypes.gpuStream_t
        ]
        if len(gpustream_input) > 1:
            raise ValueError(
                f"There can not be more than one GPU stream assigned to a kernel, but {len(gpustream_input)} were assigned."
            )

        self.kernel_wrapper_args_as_input: List[str] = (
            ['__state'] + [ptr(name, data, sdfg, cudaCodeGen._frame)
                           for name, data in self.arglist.items()] + [str(gpustream_input[0].dst_conn)])

        self.kernel_wrapper_args_typed: List[str] = (
            [f'{mangle_dace_state_struct_name(cudaCodeGen._global_sdfg)} *__state'] + args_typed +
            [f"gpuStream_t {gpustream_var_name}"])

        cudaCodeGen._in_device_code = restore_in_device_code

        self.grid_dims, self.block_dims = cudaCodeGen._kernel_dimensions_map[kernel_map_entry]
        self.gpu_index_ctype: str = self.get_gpu_index_ctype()

        if cudaCodeGen.backend not in ['cuda', 'hip']:
            raise ValueError(f"Unsupported backend '{cudaCodeGen.backend}' in ExperimentalCUDACodeGen. "
                             "Only 'cuda' and 'hip' are supported.")

        warp_size_key = 'cuda_warp_size' if cudaCodeGen.backend == 'cuda' else 'hip_warp_size'
        self.warpSize: int = Config.get('compiler', 'cuda', warp_size_key)

    def get_gpu_index_ctype(self, config_key='gpu_index_type') -> str:
        """Return the C type string for the configured DaCe dtype under
        ``compiler.cuda.<config_key>``. Raises if the name does not resolve
        to a DaCe ``typeclass``."""
        type_name = Config.get('compiler', 'cuda', config_key)
        dtype = getattr(dtypes, type_name, None)
        if not isinstance(dtype, dtypes.typeclass):
            raise ValueError(
                f'Invalid {config_key} "{type_name}" configured (used for thread, block, and warp indices): '
                'no matching DaCe data type found.\n'
                'Please use a valid type from dace.dtypes (e.g., "int32", "uint64").')
        return dtype.ctype
