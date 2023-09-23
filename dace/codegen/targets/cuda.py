# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import ctypes
import functools
import os
import warnings
from typing import Any, Dict, List, Set, Tuple, Union

import networkx as nx
import sympy
from six import StringIO

import dace
from dace import data as dt
from dace import dtypes, registry
from dace import sdfg as sd
from dace import subsets, symbolic
from dace.codegen import common, cppunparse
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (codeblock_to_cpp, cpp_array_expr, memlet_copy_to_absolute_strides, sym2cpp,
                                      synchronize_streams, unparse_cr, unparse_cr_split)
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import (SDFG, ScopeSubgraphView, SDFGState, dynamic_map_inputs, has_dynamic_map_inputs,
                       is_array_stream_view, is_devicelevel_gpu, nodes, scope_contains_scope)
from dace.sdfg import utils as sdutil
from dace.transformation import helpers as xfh
from dace.transformation.passes import analysis as ap


def prod(iterable):
    return functools.reduce(sympy.Mul, iterable, 1)


def _expr(val):
    if isinstance(val, symbolic.SymExpr):
        return val.expr
    return val


def cpu_to_gpu_cpred(sdfg, state, src_node, dst_node):
    """ Copy predicate from CPU to GPU that determines when a copy is illegal.
        Returns True if copy is illegal, False otherwise.
    """
    if isinstance(sdfg.arrays[src_node.data], dt.Scalar):
        return False
    return True


@registry.autoregister_params(name='cuda')
class CUDACodeGen(TargetCodeGenerator):
    """ GPU (CUDA/HIP) code generator. """
    target_name = 'cuda'
    title = 'CUDA'
    _in_device_code = False

    def __init__(self, frame_codegen, sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self.create_grid_barrier = False
        self.extra_nsdfg_args = []
        CUDACodeGen._in_device_code = False
        self._cpu_codegen = None
        self._block_dims = None
        self._grid_dims = None
        self._kernel_map = None
        self._kernel_state = None
        self._kernel_grid_conditions: List[str] = []
        self._scope_has_collaborative_copy = False
        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self._global_sdfg: SDFG = sdfg
        self._toplevel_schedule = None
        self._arglists: Dict[nodes.MapEntry, Dict[str, dt.Data]] = {}

        # Keep track of current "scope entry/exit" code streams for extra
        # code generation
        self.scope_entry_stream = self._initcode
        self.scope_exit_stream = self._exitcode

        self._cuda_streams, self._cuda_events = 0, 0

        # Positions at which to deallocate memory pool arrays
        self.pool_release: Dict[Tuple[SDFG, str], Tuple[SDFGState, Set[nodes.Node]]] = {}
        self.has_pool = False

        # Register dispatchers
        self._cpu_codegen = dispatcher.get_generic_node_dispatcher()

        # Register additional CUDA dispatchers
        dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES, self)

        dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)

        dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]
        dispatcher.register_array_dispatcher(gpu_storage, self)
        dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)

        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        # Register illegal copies
        cpu_unpinned_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal]
        gpu_private_storage = [dtypes.StorageType.GPU_Shared]
        illegal_copy = IllegalCopy()
        for st in cpu_unpinned_storage:
            for gst in gpu_private_storage:
                dispatcher.register_copy_dispatcher(st, gst, None, illegal_copy)
                dispatcher.register_copy_dispatcher(gst, st, None, illegal_copy)
        for st in cpu_unpinned_storage:
            for sched_type in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock]:
                # NOTE: Only reading to GPU has an exception (for Scalar inputs)
                dispatcher.register_copy_dispatcher(st,
                                                    dtypes.StorageType.Register,
                                                    sched_type,
                                                    illegal_copy,
                                                    predicate=cpu_to_gpu_cpred)
                dispatcher.register_copy_dispatcher(dtypes.StorageType.Register, st, sched_type, illegal_copy)
        # End of illegal copies
        # End of dispatcher registration
        ######################################

    def _emit_sync(self, codestream: CodeIOStream):
        if Config.get_bool('compiler', 'cuda', 'syncdebug'):
            codestream.write('''DACE_GPU_CHECK({backend}GetLastError());
            DACE_GPU_CHECK({backend}DeviceSynchronize());'''.format(backend=self.backend))

    def preprocess(self, sdfg: SDFG) -> None:
        # Determine GPU backend
        self.backend = common.get_gpu_backend()
        self.language = 'cu' if self.backend == 'cuda' else 'cpp'
        target_type = "" if self.backend == 'cuda' else self.backend
        self._codeobject = CodeObject(sdfg.name + '_' + 'cuda',
                                      '',
                                      self.language,
                                      CUDACodeGen,
                                      'CUDA',
                                      target_type=target_type)

        # Find GPU<->GPU strided copies that cannot be represented by a single copy command
        from dace.transformation.dataflow import CopyToMap
        for e, state in list(sdfg.all_edges_recursive()):
            nsdfg = state.parent
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
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

        # Annotate CUDA streams and events
        self._cuda_streams, self._cuda_events = self._compute_cudastreams(sdfg)

        # Find points where memory should be released to the memory pool
        self._compute_pool_release(sdfg)

        # Write GPU context to state structure
        self._frame.statestruct.append('dace::cuda::Context *gpu_context;')

        # Collect all defined symbols and argument lists with one traversal
        shared_transients = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(sdfg, recursive=True):
            if (isinstance(node, nodes.MapEntry)
                    and node.map.schedule in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)):
                if state.parent not in shared_transients:
                    shared_transients[state.parent] = state.parent.shared_transients()
                self._arglists[node] = state.scope_subgraph(node).arglist(defined_syms, shared_transients[state.parent])

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

            reachable = reachability[sdfg.sdfg_id]
            access_sets = access_nodes[sdfg.sdfg_id]
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

                    # Enforce a cuda_stream field so that the state-wide deallocation would work
                    if not hasattr(an1, '_cuda_stream'):
                        an1._cuda_stream = 'nullptr'

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

    # Generate final code
    def get_generated_codeobjects(self):
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'cuda')

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

DACE_EXPORTED int __dace_init_cuda({sdfg.name}_t *__state{params});
DACE_EXPORTED int __dace_exit_cuda({sdfg.name}_t *__state);

{other_globalcode}

int __dace_init_cuda({sdfg.name}_t *__state{params}) {{
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

int __dace_exit_cuda({sdfg.name}_t *__state) {{
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

DACE_EXPORTED bool __dace_gpu_set_stream({sdfg.name}_t *__state, int streamid, gpuStream_t stream)
{{
    if (streamid < 0 || streamid >= {nstreams})
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}}

DACE_EXPORTED void __dace_gpu_set_all_streams({sdfg.name}_t *__state, gpuStream_t stream)
{{
    for (int i = 0; i < {nstreams}; ++i)
        __state->gpu_context->streams[i] = stream;
}}

{localcode}
""".format(params=params_comma,
           initcode=initcode.getvalue(),
           exitcode=exitcode.getvalue(),
           other_globalcode=self._globalcode.getvalue(),
           localcode=self._localcode.getvalue(),
           file_header=fileheader.getvalue(),
           nstreams=max(1, self._cuda_streams),
           nevents=max(1, self._cuda_events),
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg)

        return [self._codeobject]

    def node_dispatch_predicate(self, sdfg, state, node):
        if hasattr(node, 'schedule'):  # NOTE: Works on nodes and scopes
            if node.schedule in dtypes.GPU_SCHEDULES:
                return True
        if isinstance(node, nodes.NestedSDFG) and CUDACodeGen._in_device_code:
            return True
        return False

    def state_dispatch_predicate(self, sdfg, state):
        if self._toplevel_schedule in dtypes.GPU_SCHEDULES:
            return True
        for node in state.sink_nodes():
            if hasattr(node, '_cuda_stream'):
                return True
            else:
                for e in state.in_edges(node):
                    if hasattr(e.src, '_cuda_stream'):
                        return True
        for s, _ in self.pool_release.values():
            if s is state:
                return True
        return False

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return True

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

            flags = Config.get("compiler", "cuda", "args")
            flags += ' ' + ' '.join('-gencode arch=compute_{arch},code=sm_{arch}'.format(arch=arch)
                                    for arch in cuda_arch)

            options.append("-DCUDA_NVCC_FLAGS=\"{}\"".format(flags))

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

    def declare_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream):

        fsymbols = self._frame.symbols_and_constants(sdfg)
        # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
        # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
        # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
        # `nodedesc` is a View and `dfg` is None.
        if dfg and not sdutil.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError("The declare_array method should only be used for variables "
                                      "that must have their declaration and allocation separate.")

        ptrname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)

        # Check if array is already declared
        if self._dispatcher.declared_arrays.has(ptrname):
            return

        result_decl = StringIO()
        ctypedef = '%s *' % nodedesc.dtype.ctype
        dataname = node.data

        # Different types of GPU arrays
        if (nodedesc.storage == dtypes.StorageType.GPU_Global or nodedesc.storage == dtypes.StorageType.CPU_Pinned):
            result_decl.write('%s %s;\n' % (ctypedef, dataname))
            self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, ctypedef)
        elif nodedesc.storage == dtypes.StorageType.GPU_Shared:
            raise NotImplementedError('Dynamic shared memory unsupported')
        elif nodedesc.storage == dtypes.StorageType.Register:
            raise ValueError('Dynamic allocation of registers not allowed')
        else:
            raise NotImplementedError("CUDA: Unimplemented storage type " + str(nodedesc.storage))

        declaration_stream.write(result_decl.getvalue(), sdfg, state_id, node)

    def allocate_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                       allocation_stream):
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)

        try:
            self._dispatcher.defined_vars.get(dataname)
            return
        except KeyError:
            pass  # The variable was not defined, we can continue

        # Check if array is already declared
        declared = False
        try:
            self._dispatcher.declared_arrays.get(dataname)
            declared = True  # Array was already declared in this or upper scopes
        except KeyError:  # Array not declared yet
            pass

        if isinstance(nodedesc, dace.data.Stream):
            return self.allocate_stream(sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                                        allocation_stream)
        elif isinstance(nodedesc, dace.data.View):
            return self._cpu_codegen.allocate_view(sdfg, dfg, state_id, node, function_stream, declaration_stream,
                                                   allocation_stream)
        elif isinstance(nodedesc, dace.data.Reference):
            return self._cpu_codegen.allocate_reference(sdfg, dfg, state_id, node, function_stream, declaration_stream,
                                                        allocation_stream)

        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        arrsize_malloc = '%s * sizeof(%s)' % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = '%s *' % nodedesc.dtype.ctype

        # Different types of GPU arrays
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if not declared:
                result_decl.write('%s %s;\n' % (ctypedef, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)

            if nodedesc.pool:
                cudastream = getattr(node, '_cuda_stream', 'nullptr')
                if cudastream != 'nullptr':
                    cudastream = f'__state->gpu_context->streams[{cudastream}]'
                result_alloc.write(
                    f'DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {cudastream}));\n'
                )
                self._emit_sync(result_alloc)
            else:
                # Strides are left to the user's discretion
                result_alloc.write('DACE_GPU_CHECK(%sMalloc((void**)&%s, %s));\n' %
                                   (self.backend, dataname, arrsize_malloc))

            if node.setzero:
                result_alloc.write('DACE_GPU_CHECK(%sMemset(%s, 0, %s));\n' % (self.backend, dataname, arrsize_malloc))
            if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
                result_alloc.write(f'{dataname} += {cpp.sym2cpp(nodedesc.start_offset)};\n')
        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            if not declared:
                result_decl.write('%s %s;\n' % (ctypedef, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)

            # Strides are left to the user's discretion
            result_alloc.write('DACE_GPU_CHECK(%sMallocHost(&%s, %s));\n' % (self.backend, dataname, arrsize_malloc))
            if node.setzero:
                result_alloc.write('memset(%s, 0, %s);\n' % (dataname, arrsize_malloc))
            if nodedesc.start_offset != 0:
                result_alloc.write(f'{dataname} += {cpp.sym2cpp(nodedesc.start_offset)};\n')
        elif nodedesc.storage == dtypes.StorageType.GPU_Shared:
            if is_dynamically_sized:
                raise NotImplementedError('Dynamic shared memory unsupported')
            if nodedesc.start_offset != 0:
                raise NotImplementedError('Start offset unsupported for shared memory')
            result_decl.write("__shared__ %s %s[%s];\n" % (nodedesc.dtype.ctype, dataname, sym2cpp(arrsize)))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
            if node.setzero:
                result_alloc.write('dace::ResetShared<{type}, {block_size}, {elements}, '
                                   '1, false>::Reset({ptr});\n'.format(type=nodedesc.dtype.ctype,
                                                                       block_size=', '.join(_topy(self._block_dims)),
                                                                       ptr=dataname,
                                                                       elements=sym2cpp(arrsize)))
        elif nodedesc.storage == dtypes.StorageType.Register:
            if is_dynamically_sized:
                raise ValueError('Dynamic allocation of registers not allowed')
            if nodedesc.start_offset != 0:
                raise NotImplementedError('Start offset unsupported for registers')
            szstr = ' = {0}' if node.setzero else ''
            result_decl.write("%s %s[%s]%s;\n" % (nodedesc.dtype.ctype, dataname, sym2cpp(arrsize), szstr))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
        else:
            raise NotImplementedError("CUDA: Unimplemented storage type " + str(nodedesc.storage))

        declaration_stream.write(result_decl.getvalue(), sdfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), sdfg, state_id, node)

    def allocate_stream(self, sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                        allocation_stream):
        dataname = node.data
        allocname = cpp.ptr(dataname, nodedesc, sdfg, self._frame)
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            fmtargs = {
                'name': allocname,  # TODO: Handle persistent streams
                'allocname': allocname,
                'type': nodedesc.dtype.ctype,
                'is_pow2': sym2cpp(sympy.log(nodedesc.buffer_size, 2).is_Integer),
                'location': '%s_%s_%s' % (sdfg.sdfg_id, state_id, dfg.node_id(node))
            }

            ctypedef = 'dace::GPUStream<{type}, {is_pow2}>'.format(**fmtargs)
            self._dispatcher.defined_vars.add(allocname, DefinedType.Stream, ctypedef)

            if is_array_stream_view(sdfg, dfg, node):
                edges = dfg.out_edges(node)
                if len(edges) > 1:
                    raise NotImplementedError("Cannot handle streams writing to multiple arrays.")

                fmtargs['ptr'] = nodedesc.sink + ' + ' + cpp_array_expr(
                    sdfg, edges[0].data, with_brackets=False, codegen=self._frame)

                # Assuming 1D subset of sink/src
                # sym2cpp(edges[0].data.subset[-1])
                fmtargs['size'] = sym2cpp(nodedesc.buffer_size)

                # (important) Ensure GPU array is allocated before the stream
                datanode = dfg.out_edges(node)[0].dst
                sinkdesc = sdfg.arrays[datanode.data]
                self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, datanode, sinkdesc, function_stream,
                                                   allocation_stream)

                function_stream.write(
                    'DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'
                    .format(**fmtargs), sdfg, state_id, node)
                self._globalcode.write(
                    """
DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);
void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result) {{
    result = dace::AllocGPUArrayStreamView<{type}, {is_pow2}>(ptr, size);
}}""".format(**fmtargs), sdfg, state_id, node)
                declaration_stream.write('dace::GPUStream<{type}, {is_pow2}> {name};'.format(**fmtargs), sdfg, state_id,
                                         node)
                allocation_stream.write('__dace_alloc_{location}({ptr}, {size}, {allocname});'.format(**fmtargs), sdfg,
                                        state_id, node)
            else:
                fmtargs['size'] = sym2cpp(nodedesc.buffer_size)

                function_stream.write(
                    'DACE_EXPORTED void __dace_alloc_{location}(uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'
                    .format(**fmtargs), sdfg, state_id, node)
                self._globalcode.write(
                    """
DACE_EXPORTED void __dace_alloc_{location}(uint32_t {size}, dace::GPUStream<{type}, {is_pow2}>& result);
void __dace_alloc_{location}(uint32_t {size}, dace::GPUStream<{type}, {is_pow2}>& result) {{
    result = dace::AllocGPUStream<{type}, {is_pow2}>({size});
}}""".format(**fmtargs), sdfg, state_id, node)
                declaration_stream.write('dace::GPUStream<{type}, {is_pow2}> {name};'.format(**fmtargs), sdfg, state_id,
                                         node)
                allocation_stream.write('__dace_alloc_{location}({size}, {allocname});'.format(**fmtargs), sdfg,
                                        state_id, node)

    def deallocate_stream(self, sdfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if is_array_stream_view(sdfg, dfg, node):
                callsite_stream.write('dace::FreeGPUArrayStreamView(%s);' % dataname, sdfg, state_id, node)
            else:
                callsite_stream.write('dace::FreeGPUStream(%s);' % dataname, sdfg, state_id, node)

    def deallocate_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f'({dataname} - {cpp.sym2cpp(nodedesc.start_offset)})'

        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        if isinstance(nodedesc, dace.data.Stream):
            return self.deallocate_stream(sdfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream)
        elif isinstance(nodedesc, dace.data.View):
            return

        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if not nodedesc.pool:  # If pooled, will be freed somewhere else
                callsite_stream.write('DACE_GPU_CHECK(%sFree(%s));\n' % (self.backend, dataname), sdfg, state_id, node)
        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            callsite_stream.write('DACE_GPU_CHECK(%sFreeHost(%s));\n' % (self.backend, dataname), sdfg, state_id, node)
        elif nodedesc.storage == dtypes.StorageType.GPU_Shared or \
             nodedesc.storage == dtypes.StorageType.Register:
            pass  # Do nothing
        else:
            raise NotImplementedError

    def _compute_cudastreams(self, sdfg: SDFG, default_stream=0, default_event=0):
        """ Annotates an SDFG (and all nested ones) to include a `_cuda_stream`
            field. This field is applied to all GPU maps, tasklets, and copies
            that can be executed in parallel.
            
            :param sdfg: The sdfg to modify.
            :param default_stream: The stream ID to start counting from (used
                                   in recursion to nested SDFGs).
            :param default_event: The event ID to start counting from (used
                                  in recursion to nested SDFGs).
            :return: 2-tuple of the number of streams, events to create.
        """
        concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))
        if concurrent_streams < 0:
            return 0, 0

        def increment(streams):
            if concurrent_streams > 0:
                return (streams + 1) % concurrent_streams
            return streams + 1

        state_streams = []
        state_subsdfg_events = []

        for state in sdfg.nodes():
            # Start by annotating source nodes
            source_nodes = state.source_nodes()

            # Concurrency can only be found in each state
            max_streams = default_stream
            max_events = default_event

            for i, node in enumerate(source_nodes):
                if isinstance(node, nodes.AccessNode):
                    continue
                if isinstance(node, nodes.NestedSDFG):
                    if node.schedule == dtypes.ScheduleType.GPU_Device:
                        continue
                    if node.schedule not in dtypes.GPU_SCHEDULES:
                        max_streams, max_events = self._compute_cudastreams(node.sdfg, max_streams, max_events + 1)
                node._cuda_stream = max_streams
                node._cs_childpath = False
                max_streams = increment(max_streams)

            # Maintain the same CUDA stream in DFS order, add more when
            # possible.
            for e in state.dfs_edges(source_nodes):
                if hasattr(e.dst, '_cuda_stream'):
                    continue
                if hasattr(e.src, '_cuda_stream'):
                    c = e.src._cuda_stream

                    if (isinstance(e.dst, nodes.AccessNode) and isinstance(sdfg.arrays[e.dst.data], dt.View)):
                        # Skip views
                        e.dst._cuda_stream = c
                        e.dst._cs_childpath = False
                        continue

                    if e.src._cs_childpath == True:
                        c = max_streams
                        max_streams = increment(max_streams)
                    e.src._cs_childpath = True

                    # Do not create multiple streams within GPU scopes
                    if (isinstance(e.src, nodes.EntryNode) and e.src.schedule in dtypes.GPU_SCHEDULES):
                        e.src._cs_childpath = False
                    elif state.entry_node(e.src) is not None:
                        parent = state.entry_node(e.src)
                        if parent.schedule in dtypes.GPU_SCHEDULES:
                            e.src._cs_childpath = False
                else:
                    c = max_streams
                    if (isinstance(e.dst, nodes.AccessNode) and isinstance(sdfg.arrays[e.dst.data], dt.View)):
                        # Skip views
                        pass
                    else:
                        max_streams = increment(max_streams)
                e.dst._cuda_stream = c
                if not hasattr(e.dst, '_cs_childpath'):
                    e.dst._cs_childpath = False
                if isinstance(e.dst, nodes.NestedSDFG):
                    if e.dst.schedule not in dtypes.GPU_SCHEDULES:
                        max_streams, max_events = self._compute_cudastreams(e.dst.sdfg, e.dst._cuda_stream,
                                                                            max_events + 1)

            state_streams.append(max_streams if concurrent_streams == 0 else concurrent_streams)
            state_subsdfg_events.append(max_events)

        # Remove CUDA streams from paths of non-gpu copies and CPU tasklets
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(graph, SDFGState):
                cur_sdfg = graph.parent

                if (isinstance(node, (nodes.EntryNode, nodes.ExitNode)) and node.schedule in dtypes.GPU_SCHEDULES):
                    # Node must have GPU stream, remove childpath and continue
                    if hasattr(node, '_cs_childpath'):
                        delattr(node, '_cs_childpath')
                    continue

                for e in graph.all_edges(node):
                    path = graph.memlet_path(e)
                    # If leading from/to a GPU memory node, keep stream
                    if ((isinstance(path[0].src, nodes.AccessNode)
                         and path[0].src.desc(cur_sdfg).storage == dtypes.StorageType.GPU_Global)
                            or (isinstance(path[-1].dst, nodes.AccessNode)
                                and path[-1].dst.desc(cur_sdfg).storage == dtypes.StorageType.GPU_Global)):
                        break
                    # If leading from/to a GPU tasklet, keep stream
                    if ((isinstance(path[0].src, nodes.CodeNode) and is_devicelevel_gpu(cur_sdfg, graph, path[0].src))
                            or (isinstance(path[-1].dst, nodes.CodeNode)
                                and is_devicelevel_gpu(cur_sdfg, graph, path[-1].dst))):
                        break
                else:  # If we did not break, we do not need a CUDA stream
                    if hasattr(node, '_cuda_stream'):
                        delattr(node, '_cuda_stream')
                # In any case, remove childpath
                if hasattr(node, '_cs_childpath'):
                    delattr(node, '_cs_childpath')

        # Compute maximal number of events by counting edges (within the same
        # state) that point from one stream to another
        state_events = []
        for i, state in enumerate(sdfg.nodes()):
            events = state_subsdfg_events[i]

            for e in state.edges():
                if hasattr(e.src, '_cuda_stream'):
                    # If there are two or more CUDA streams involved in this
                    # edge, or the destination is unrelated to CUDA
                    if (not hasattr(e.dst, '_cuda_stream') or e.src._cuda_stream != e.dst._cuda_stream):
                        for mpe in state.memlet_path(e):
                            mpe._cuda_event = events
                        events += 1

            state_events.append(events)

        # Maximum over all states
        max_streams = max(state_streams)
        max_events = max(state_events)

        return max_streams, max_events

    def _emit_copy(self, state_id, src_node, src_storage, dst_node, dst_storage, dst_schedule, edge, sdfg, dfg,
                   callsite_stream):
        u, uconn, v, vconn, memlet = edge
        state_dfg = sdfg.nodes()[state_id]

        cpu_storage_types = [
            dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.CPU_Pinned
        ]
        gpu_storage_types = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared]

        copy_shape = memlet.subset.bounding_box_size()
        copy_shape = [symbolic.overapproximate(s) for s in copy_shape]
        # Determine directionality
        if (isinstance(src_node, nodes.AccessNode) and memlet.data == src_node.data):
            outgoing_memlet = True
        elif (isinstance(dst_node, nodes.AccessNode) and memlet.data == dst_node.data):
            outgoing_memlet = False
        else:
            raise LookupError('Memlet does not point to any of the nodes')

        if (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)
                and not CUDACodeGen._in_device_code
                and (src_storage in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned]
                     or dst_storage in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned])
                and not (src_storage in cpu_storage_types and dst_storage in cpu_storage_types)):
            src_location = 'Device' if src_storage == dtypes.StorageType.GPU_Global else 'Host'
            dst_location = 'Device' if dst_storage == dtypes.StorageType.GPU_Global else 'Host'

            # Corner case: A stream is writing to an array
            if (isinstance(sdfg.arrays[src_node.data], dt.Stream) and isinstance(sdfg.arrays[dst_node.data],
                                                                                 (dt.Scalar, dt.Array))):
                return  # Do nothing (handled by ArrayStreamView)

            syncwith = {}  # Dictionary of {stream: event}
            is_sync = False
            max_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

            if hasattr(src_node, '_cuda_stream'):
                cudastream = src_node._cuda_stream
                if not hasattr(dst_node, '_cuda_stream'):
                    # Copy after which data is needed by the host
                    is_sync = True
                elif dst_node._cuda_stream != src_node._cuda_stream:
                    syncwith[dst_node._cuda_stream] = getattr(edge, '_cuda_event', None)
                else:
                    pass  # Otherwise, no need to synchronize
            elif hasattr(dst_node, '_cuda_stream'):
                cudastream = dst_node._cuda_stream
            else:
                if max_streams >= 0:
                    print('WARNING: Undefined stream, reverting to default')
                if dst_location == 'Host':
                    is_sync = True
                cudastream = 'nullptr'

            # Handle case of impending kernel/tasklet on another stream
            if max_streams >= 0:
                for e in state_dfg.out_edges(dst_node):
                    if isinstance(e.dst, nodes.AccessNode):
                        continue
                    if not hasattr(e.dst, '_cuda_stream'):
                        is_sync = True
                    elif not hasattr(e, '_cuda_event'):
                        is_sync = True
                    elif e.dst._cuda_stream != cudastream:
                        syncwith[e.dst._cuda_stream] = e._cuda_event

                if cudastream != 'nullptr':
                    cudastream = '__state->gpu_context->streams[%d]' % cudastream

            if memlet.wcr is not None:
                raise NotImplementedError('Accumulate %s to %s not implemented' % (src_location, dst_location))
            #############################

            # Obtain copy information
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (memlet_copy_to_absolute_strides(
                self._dispatcher, sdfg, state_dfg, edge, src_node, dst_node, self._cpu_codegen._packed_types))
            dims = len(copy_shape)

            dtype = dst_node.desc(sdfg).dtype

            # Handle unsupported copy types
            if dims == 2 and (src_strides[-1] != 1 or dst_strides[-1] != 1):
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
                    dims = 1
                    copy_shape = [copy_shape[0] * copy_shape[1]]
                    src_strides = [src_strides[1]]
                    dst_strides = [dst_strides[1]]
                else:
                    raise NotImplementedError('2D copy only supported with one stride')

            # Currently we only support ND copies when they can be represented
            # as a 1D copy or as a 2D strided copy
            if dims > 2:
                if src_strides[-1] != 1 or dst_strides[-1] != 1:
                    raise NotImplementedError(
                        'GPU copies are not supported for N-dimensions if they cannot be represented by a strided copy\n'
                        f'  Nodes: src {src_node} ({src_storage}), dst {dst_node}({dst_storage})\n'
                        f'  Strides: src {src_strides}, dst {dst_strides}')
                else:
                    # Write for-loop headers
                    for d in range(dims - 2):
                        callsite_stream.write(f"for (int __copyidx{d} = 0; "
                                              f"__copyidx{d} < {copy_shape[d]};"
                                              f"++__copyidx{d}) {{")
                    # Write Memcopy2DAsync
                    current_src_expr = src_expr + " + " + " + ".join(
                        ["(__copyidx{} * ({}))".format(d, sym2cpp(s)) for d, s in enumerate(src_strides[:-2])])
                    current_dst_expr = dst_expr + " + " + "+ ".join(
                        ["(__copyidx{} * ({}))".format(d, sym2cpp(s)) for d, s in enumerate(dst_strides[:-2])])
                    callsite_stream.write(
                        'DACE_GPU_CHECK(%sMemcpy2DAsync(%s, %s, %s, %s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                        (self.backend, current_dst_expr,
                         _topy(dst_strides[-2]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, current_src_expr,
                         sym2cpp(src_strides[-2]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                         sym2cpp(copy_shape[-1]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                         sym2cpp(copy_shape[-2]), self.backend, src_location, dst_location, cudastream), sdfg, state_id,
                        [src_node, dst_node])
                    # Write for-loop footers
                    for d in range(dims - 2):
                        callsite_stream.write("}")

            if dims == 1 and not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                copysize = ' * '.join(_topy(copy_shape))
                array_length = copysize
                copysize += ' * sizeof(%s)' % dtype.ctype

                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpyAsync(%s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, src_expr, copysize, self.backend, src_location, dst_location, cudastream),
                    sdfg, state_id, [src_node, dst_node])
                node_dtype = dst_node.desc(sdfg).dtype
                if issubclass(node_dtype.type, ctypes.Structure):
                    callsite_stream.write('for (size_t __idx = 0; __idx < {arrlen}; ++__idx) '
                                          '{{'.format(arrlen=array_length))
                    for field_name, field_type in node_dtype._data.items():
                        if isinstance(field_type, dtypes.pointer):
                            tclass = field_type.type
                            length = node_dtype._length[field_name]
                            size = 'sizeof({})*{}[__idx].{}'.format(dtypes._CTYPES[tclass], str(src_node), length)
                            callsite_stream.write('DACE_GPU_CHECK({backend}Malloc(&{dst}[__idx].{fname}, '
                                                  '{sz}));'.format(dst=str(dst_node),
                                                                   fname=field_name,
                                                                   sz=size,
                                                                   backend=self.backend))
                            callsite_stream.write(
                                'DACE_GPU_CHECK({backend}MemcpyAsync({dst}[__idx].{fname}, '
                                '{src}[__idx].{fname}, {sz}, '
                                '{backend}Memcpy{sloc}To{dloc}, {stream}));'.format(dst=str(dst_node),
                                                                                    src=str(src_node),
                                                                                    fname=field_name,
                                                                                    sz=size,
                                                                                    sloc=src_location,
                                                                                    dloc=dst_location,
                                                                                    stream=cudastream,
                                                                                    backend=self.backend), sdfg,
                                state_id, [src_node, dst_node])
                    callsite_stream.write('}')
            elif dims == 1 and ((src_strides[-1] != 1 or dst_strides[-1] != 1)):
                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpy2DAsync(%s, %s, %s, %s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, _topy(dst_strides[0]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                     src_expr, sym2cpp(src_strides[0]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                     'sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, sym2cpp(
                         copy_shape[0]), self.backend, src_location, dst_location, cudastream), sdfg, state_id,
                    [src_node, dst_node])
            elif dims == 2:
                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpy2DAsync(%s, %s, %s, %s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, _topy(dst_strides[0]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                     src_expr, sym2cpp(src_strides[0]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                     sym2cpp(copy_shape[1]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, sym2cpp(
                         copy_shape[0]), self.backend, src_location, dst_location, cudastream), sdfg, state_id,
                    [src_node, dst_node])

            # Post-copy synchronization
            if is_sync:
                # Synchronize with host (done at destination)
                pass
            else:
                # Synchronize with other streams as necessary
                for streamid, event in syncwith.items():
                    syncstream = '__state->gpu_context->streams[%d]' % streamid
                    callsite_stream.write(
                        '''
    DACE_GPU_CHECK({backend}EventRecord(__state->gpu_context->events[{ev}], {src_stream}));
    DACE_GPU_CHECK({backend}StreamWaitEvent({dst_stream}, __state->gpu_context->events[{ev}], 0));
                    '''.format(ev=event, src_stream=cudastream, dst_stream=syncstream, backend=self.backend), sdfg,
                        state_id, [src_node, dst_node])

            self._emit_sync(callsite_stream)

        # Copy within the GPU
        elif (src_storage in gpu_storage_types and dst_storage in gpu_storage_types):

            state_dfg = sdfg.nodes()[state_id]
            sdict = state_dfg.scope_dict()
            schedule_node = src_node
            if scope_contains_scope(sdict, src_node, dst_node):
                schedule_node = dst_node

            state = state_dfg
            while (schedule_node is None or not isinstance(schedule_node, nodes.MapEntry)
                   or schedule_node.map.schedule == dtypes.ScheduleType.Sequential):
                ret = xfh.get_parent_map(state, schedule_node)
                if ret is None:
                    schedule_node = None
                    break
                schedule_node, state = ret

            if schedule_node is None:
                inner_schedule = dtypes.SCOPEDEFAULT_SCHEDULE[None]
            else:
                inner_schedule = schedule_node.map.schedule

            # Collaborative load
            if inner_schedule == dtypes.ScheduleType.GPU_Device:
                # Obtain copy information
                copy_shape, src_strides, dst_strides, src_expr, dst_expr = (memlet_copy_to_absolute_strides(
                    self._dispatcher, sdfg, state, edge, src_node, dst_node, self._cpu_codegen._packed_types))

                dims = len(copy_shape)

                funcname = 'dace::%sTo%s%dD' % (_get_storagename(src_storage), _get_storagename(dst_storage), dims)
                self._scope_has_collaborative_copy = True
                accum = ''
                custom_reduction = []
                if memlet.wcr is not None:
                    redtype = operations.detect_reduction_type(memlet.wcr)
                    reduction_tmpl = ''
                    # Special call for detected reduction types
                    if redtype != dtypes.ReductionType.Custom:
                        credtype = ('dace::ReductionType::' + str(redtype)[str(redtype).find('.') + 1:])
                        reduction_tmpl = '<%s>' % credtype
                    else:
                        dtype = dst_node.desc(sdfg).dtype
                        custom_reduction = [unparse_cr(sdfg, memlet.wcr, dtype)]
                    accum = '::template Accum%s' % reduction_tmpl

                if any(symbolic.issymbolic(s, sdfg.constants) for s in copy_shape):
                    callsite_stream.write(('    {func}Dynamic<{type}, {bdims}, {is_async}>{accum}({args});').format(
                        func=funcname,
                        type=dst_node.desc(sdfg).dtype.ctype,
                        bdims=', '.join(_topy(self._block_dims)),
                        is_async='true' if state_dfg.out_degree(dst_node) > 0 else 'true',
                        accum=accum,
                        args=', '.join([src_expr] + _topy(src_strides) + [dst_expr] + custom_reduction +
                                       _topy(dst_strides) + _topy(copy_shape))), sdfg, state_id, [src_node, dst_node])
                else:
                    callsite_stream.write(
                        ('    {func}<{type}, {bdims}, {copysize}, ' +
                         '{dststrides}, {is_async}>{accum}({args});').format(
                             func=funcname,
                             type=dst_node.desc(sdfg).dtype.ctype,
                             bdims=', '.join(_topy(self._block_dims)),
                             copysize=', '.join(_topy(copy_shape)),
                             dststrides=', '.join(_topy(dst_strides)),
                             is_async='true' if state_dfg.out_degree(dst_node) > 0 else 'true',
                             accum=accum,
                             args=', '.join([src_expr] + _topy(src_strides) + [dst_expr] + custom_reduction)), sdfg,
                        state_id, [src_node, dst_node])
            # Per-thread load (same as CPU copies)
            else:
                self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)
        else:
            self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, memlet, function_stream, callsite_stream):
        state = sdfg.node(state_id)
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            src_parent = state.entry_node(src_node)
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        dst_parent = state.entry_node(dst_node)
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        # Emit actual copy
        self._emit_copy(state_id, src_node, src_storage, dst_node, dst_storage, dst_schedule, memlet, sdfg, dfg,
                        callsite_stream)

    def define_out_memlet(self, sdfg, state_dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream):
        self._cpu_codegen.define_out_memlet(sdfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
                                            callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)

    def _begin_streams(self, sdfg, state):
        result = set()
        for node in state.source_nodes():
            if hasattr(node, '_cuda_stream'):
                if (isinstance(node, nodes.AccessNode) and isinstance(sdfg.arrays[node.data], dt.View)):
                    continue
                result.add(node._cuda_stream)
            else:
                # Collect other streams in state start
                for e in state.out_edges(node):
                    if hasattr(e.dst, '_cuda_stream'):
                        if (isinstance(node, nodes.AccessNode) and isinstance(sdfg.arrays[node.data], dt.View)):
                            continue
                        result.add(e.dst._cuda_stream)
        return result

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        # Two modes: device-level state and if this state has active streams
        if CUDACodeGen._in_device_code:
            self.generate_devicelevel_state(sdfg, state, function_stream, callsite_stream)
        else:
            # Active streams found. Generate state normally and sync with the
            # streams in the end
            self._frame.generate_state(sdfg, state, function_stream, callsite_stream, generate_state_footer=False)

            # Reset thread-block-level information
            self._scope_has_collaborative_copy = False

            # Free pooled memory that needs to be released here
            to_remove = set()
            backend = self.backend
            for (sd, name), (pstate, terminators) in self.pool_release.items():
                if sd is not sdfg or state is not pstate:
                    continue

                desc = sd.arrays[name]
                ptrname = cpp.ptr(name, desc, sd, self._frame)
                if isinstance(desc, dt.Array) and desc.start_offset != 0:
                    ptrname = f'({ptrname} - {cpp.sym2cpp(desc.start_offset)})'

                callsite_stream.write(f'DACE_GPU_CHECK({backend}Free({ptrname}));\n', sd)
                self._emit_sync(callsite_stream)
                to_remove.add((sd, name))
            for sd, name in to_remove:
                del self.pool_release[sd, name]

            if state.nosync == False:
                streams_to_sync = set()
                for node in state.sink_nodes():
                    if hasattr(node, '_cuda_stream') and node._cuda_stream != 'nullptr':
                        streams_to_sync.add(node._cuda_stream)
                    else:
                        # Synchronize sink-node copies at the end of the state
                        for e in state.in_edges(node):
                            if hasattr(e.src, '_cuda_stream') and e.src._cuda_stream != 'nullptr':
                                streams_to_sync.add(e.src._cuda_stream)

                # Relaxed condition for skipping synchronization:
                # if ALL the immediately reachable non-empty states (i.e.,
                # ignoring guard states) use ONLY the same streams as the
                # current state does, and there is only one such stream,
                # then we can skip synchronization.
                next_states = sdutil.get_next_nonempty_states(sdfg, state)
                if next_states and len(streams_to_sync) == 1:
                    if all(self._begin_streams(sdfg, ns) == streams_to_sync for ns in next_states):
                        # Relax synchronization
                        streams_to_sync = set()

                for stream in streams_to_sync:
                    callsite_stream.write(
                        'DACE_GPU_CHECK(%sStreamSynchronize(__state->gpu_context->streams[%d]));' %
                        (self.backend, stream), sdfg, sdfg.node_id(state))

            # After synchronizing streams, generate state footer normally
            callsite_stream.write('\n')

            # Emit internal transient array deallocation
            self._frame.deallocate_arrays_in_scope(sdfg, state, function_stream, callsite_stream)

            # Invoke all instrumentation providers
            for instr in self._frame._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream, function_stream)

    def generate_devicelevel_state(self, sdfg, state, function_stream, callsite_stream):

        # Special case: if this is a GPU grid state and something is reading
        # from a possible result of a collaborative write, sync first
        if self._toplevel_schedule == dtypes.ScheduleType.GPU_Device:
            state_id = next(i for i, s in enumerate(sdfg.nodes()) if s == state)
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage == dtypes.StorageType.GPU_Shared
                        and state.in_degree(node) == 0 and state.out_degree(node) > 0):
                    if not self._scope_has_collaborative_copy:
                        callsite_stream.write('__syncthreads();', sdfg, state_id)
                    break

        # In GPU_Persistent scopes, states need global barriers between them,
        # the DFGs inside of a state are independent, so they don't need
        # synchronization. DFGs in a GPU_Persistent scope are per se executed
        # by a single thread only. (Device) Maps however can be distributed
        # across multiple threads
        elif self._toplevel_schedule == dtypes.ScheduleType.GPU_Persistent:

            # reset streams in GPU persistent maps if the lifetime is scope,
            # otherwise streams do not behave as expected becasue they are
            # allocated on host side
            streams_to_reset = [
                node for node in state.data_nodes() if isinstance(node.desc(sdfg), dace.nodes.data.Stream)
                and node.desc(sdfg).lifetime == dtypes.AllocationLifetime.Scope
            ]
            for stream in streams_to_reset:
                ptrname = cpp.ptr(stream.data, stream.desc(sdfg), sdfg, self._frame)
                callsite_stream.write("{}.reset();".format(ptrname), sdfg, state.node_id)

            components = dace.sdfg.concurrent_subgraphs(state)
            for c in components:

                has_map = any(isinstance(node, dace.nodes.MapEntry) for node in c.nodes())
                # If a global is modified, execute once per global state,
                # if a shared memory element is modified, execute once per block,
                # if a local scalar is modified, execute in every thread.
                if not has_map:
                    written_nodes = [n for n in c if state.in_degree(n) > 0 and isinstance(n, dace.nodes.AccessNode)]

                    # The order of the branching below matters - it reduces the scope with every detected write
                    write_scope = 'thread'  # General case acts in every thread
                    if any(sdfg.arrays[n.data].storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned)
                           for n in written_nodes):
                        write_scope = 'grid'
                    if any(sdfg.arrays[n.data].storage == dtypes.StorageType.GPU_Shared for n in written_nodes):
                        write_scope = 'block'
                    if any(sdfg.arrays[n.data].storage == dtypes.StorageType.Register for n in written_nodes):
                        write_scope = 'thread'

                    if write_scope == 'grid':
                        callsite_stream.write("if (blockIdx.x == 0 "
                                            "&& threadIdx.x == 0) "
                                            "{  // sub-graph begin", sdfg, state.node_id)
                    elif write_scope == 'block':
                        callsite_stream.write("if (threadIdx.x == 0) "
                                            "{  // sub-graph begin", sdfg, state.node_id)
                    else:
                        callsite_stream.write("{  // subgraph begin", sdfg, state.node_id)
                else:
                    callsite_stream.write("{  // subgraph begin", sdfg, state.node_id)

                # Need to skip certain entry nodes to make sure that they are
                # not processed twice
                # TODO this is not robust, replace by better solution
                #  (or wait for new codegen)
                entry_nodes = list(v for v in c.nodes() if len(list(c.predecessors(v))) == 0)
                comp_same_entry = [comp for comp in components if comp != c and entry_nodes[0] in comp.nodes()]
                skip_entry = len(comp_same_entry) > 0 and has_map

                self._dispatcher.dispatch_subgraph(sdfg,
                                                   c,
                                                   sdfg.node_id(state),
                                                   function_stream,
                                                   callsite_stream,
                                                   skip_entry_node=skip_entry)

                callsite_stream.write("}  // subgraph end", sdfg, state.node_id)

            callsite_stream.write('__gbar.Sync();', sdfg, state.node_id)

            # done here, code is generated
            return

        self._frame.generate_state(sdfg, state, function_stream, callsite_stream)

    # NOTE: This function is ONLY called from the CPU side. Therefore, any
    # schedule that is out of the ordinary will raise an exception
    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream, callsite_stream):
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]

        state = sdfg.nodes()[state_id]

        # If in device-level code, call appropriate function
        if (self._kernel_map is not None and self._kernel_map.map.schedule in dtypes.GPU_SCHEDULES):
            self.generate_devicelevel_scope(sdfg, dfg_scope, state_id, function_stream, callsite_stream)
            return

        # If not device-level code, ensure the schedule is correct
        if scope_entry.map.schedule not in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent):
            raise TypeError('Cannot schedule %s directly from non-GPU code' % str(scope_entry.map.schedule))

        # Modify thread-blocks if dynamic ranges are detected
        for node, graph in dfg_scope.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                smap = node.map
                if (smap.schedule == dtypes.ScheduleType.GPU_ThreadBlock and has_dynamic_map_inputs(graph, node)):
                    warnings.warn('Thread-block map cannot be used with '
                                  'dynamic ranges, switching map "%s" to '
                                  'sequential schedule' % smap.label)
                    smap.schedule = dtypes.ScheduleType.Sequential

        # Determine whether to create a global (grid) barrier object
        create_grid_barrier = False
        if scope_entry.map.schedule == dtypes.ScheduleType.GPU_Persistent:
            create_grid_barrier = True
        for node in dfg_scope.nodes():
            if scope_entry == node:
                continue
            if (isinstance(node, nodes.EntryNode) and node.map.schedule == dtypes.ScheduleType.GPU_Device):
                # Create grid barrier only if there is a synchronization requirement on nested GPU_Device maps
                if any(p is not scope_entry for p in dfg_scope.predecessors(node)):
                    create_grid_barrier = True

        self.create_grid_barrier = create_grid_barrier
        kernel_name = '%s_%d_%d_%d' % (scope_entry.map.label, sdfg.sdfg_id, sdfg.node_id(state),
                                       state.node_id(scope_entry))

        # Comprehend grid/block dimensions from scopes
        grid_dims, block_dims, tbmap, dtbmap, _ = self.get_kernel_dimensions(dfg_scope)
        is_persistent = (dfg_scope.source_nodes()[0].map.schedule == dtypes.ScheduleType.GPU_Persistent)

        # Get parameters of subgraph
        kernel_args = self._arglists[scope_entry]

        # Handle dynamic map inputs
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            kernel_args[str(e.src)] = e.src.desc(sdfg)

        # Add data from nested SDFGs to kernel arguments
        extra_call_args = []
        extra_call_args_typed = []
        extra_kernel_args = []
        extra_kernel_args_typed = []
        self.extra_nsdfg_args = []
        visited = set()
        for node, parent in dfg_scope.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode):
                nsdfg: SDFG = parent.parent
                desc = node.desc(nsdfg)
                if (nsdfg, node.data) in visited:
                    continue
                visited.add((nsdfg, node.data))
                if desc.transient and self._frame.where_allocated[(nsdfg, node.data)] is not nsdfg:
                    outer_name = cpp.ptr(node.data, desc, nsdfg, self._frame)

                    # Create name from within kernel
                    oldval = CUDACodeGen._in_device_code
                    CUDACodeGen._in_device_code = True
                    inner_name = cpp.ptr(node.data, desc, nsdfg, self._frame)
                    CUDACodeGen._in_device_code = oldval

                    self.extra_nsdfg_args.append((desc.as_arg(name=''), inner_name, outer_name))
                    self._dispatcher.defined_vars.add(inner_name,
                                                      DefinedType.Pointer,
                                                      desc.dtype.ctype,
                                                      allow_shadowing=True)
                    extra_call_args.append(outer_name)
                    extra_call_args_typed.append(desc.as_arg(name=inner_name))
                    extra_kernel_args.append(f'(void *)&{inner_name}')
                    extra_kernel_args_typed.append(desc.as_arg(name=inner_name))

        const_params = _get_const_params(dfg_scope)
        # make dynamic map inputs constant
        # TODO move this into _get_const_params(dfg_scope)
        const_params |= set((str(e.src)) for e in dace.sdfg.dynamic_map_inputs(state, scope_entry))

        # Store init/exit code streams
        old_entry_stream = self.scope_entry_stream
        old_exit_stream = self.scope_exit_stream
        self.scope_entry_stream = CodeIOStream()
        self.scope_exit_stream = CodeIOStream()

        # Instrumentation for kernel scope
        instr = self._dispatcher.instrumentation[scope_entry.map.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state, scope_entry, callsite_stream, self.scope_entry_stream, self._globalcode)
            outer_stream = CodeIOStream()
            instr.on_scope_exit(sdfg, state, scope_exit, outer_stream, self.scope_exit_stream, self._globalcode)

        # Redefine constant arguments and rename arguments to device counterparts
        # TODO: This (const behavior and code below) is all a hack.
        #       Refactor and fix when nested SDFGs are separate functions.
        self._dispatcher.defined_vars.enter_scope(scope_entry)
        prototype_kernel_args = {}
        for aname, arg in kernel_args.items():  # `list` wrapper is used to modify kernel_args within the loop
            if aname in const_params:
                defined_type, ctype = None, None
                if aname in sdfg.arrays:
                    data_desc = sdfg.arrays[aname]
                    is_global = data_desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                       dtypes.AllocationLifetime.Persistent,
                                                       dtypes.AllocationLifetime.External)
                    # Non-free symbol dependent Arrays due to their shape
                    dependent_shape = (isinstance(data_desc, dt.Array) and not isinstance(data_desc, dt.View) and any(
                        str(s) not in self._frame.symbols_and_constants(sdfg)
                        for s in self._frame.free_symbols(data_desc)))
                    try:
                        # NOTE: It is hard to get access to the view-edge here,
                        # so always check the declared-arrays dictionary for
                        # Views.
                        if dependent_shape or isinstance(data_desc, dt.View):
                            defined_type, ctype = (self._dispatcher.declared_arrays.get(aname, is_global=is_global))
                    except KeyError:
                        pass
                    ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    if not defined_type:
                        defined_type, ctype = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)

                    CUDACodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    CUDACodeGen._in_device_code = False

                    self._dispatcher.defined_vars.add(inner_ptrname,
                                                      defined_type,
                                                      'const %s' % ctype,
                                                      allow_shadowing=True)

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname
            else:
                if aname in sdfg.arrays:
                    data_desc = sdfg.arrays[aname]
                    ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    is_global = data_desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                       dtypes.AllocationLifetime.Persistent,
                                                       dtypes.AllocationLifetime.External)
                    defined_type, ctype = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)
                    CUDACodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    CUDACodeGen._in_device_code = False
                    self._dispatcher.defined_vars.add(inner_ptrname, defined_type, ctype, allow_shadowing=True)

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname

            prototype_kernel_args[aname] = arg

        kernel_args_typed = [('const ' if k in const_params else '') + v.as_arg(name=k)
                             for k, v in prototype_kernel_args.items()]

        kernel_stream = CodeIOStream()
        self.generate_kernel_scope(sdfg, dfg_scope, state_id, scope_entry.map, kernel_name, grid_dims, block_dims,
                                   tbmap, dtbmap, kernel_args_typed, self._globalcode, kernel_stream)

        self._dispatcher.defined_vars.exit_scope(scope_entry)

        # Add extra kernel arguments for a grid barrier object
        if create_grid_barrier:
            extra_kernel_args_typed.append('cub::GridBarrier __gbar')

        node = dfg_scope.source_nodes()[0]

        # Set kernel launch bounds
        if node.gpu_launch_bounds == "-1":
            launch_bounds = ''
        elif node.gpu_launch_bounds == "0":
            if any(symbolic.issymbolic(b) for b in block_dims):
                launch_bounds = ''
            else:
                launch_bounds = f'__launch_bounds__({_topy(prod(block_dims))})'
        else:
            launch_bounds = f'__launch_bounds__({node.gpu_launch_bounds})'

        # Write kernel prototype
        self._localcode.write(
            '__global__ void %s %s(%s) {\n' %
            (launch_bounds, kernel_name, ', '.join(kernel_args_typed + extra_kernel_args_typed)), sdfg, state_id, node)

        # Write constant expressions in GPU code
        self._frame.generate_constants(sdfg, self._localcode)

        self._localcode.write(self.scope_entry_stream.getvalue())

        # Assuming kernel can write to global scope (function_stream), we
        # output the kernel last
        self._localcode.write(kernel_stream.getvalue() + '\n')

        self._localcode.write(self.scope_exit_stream.getvalue())

        # Restore init/exit code streams
        self.scope_entry_stream = old_entry_stream
        self.scope_exit_stream = old_exit_stream

        state_param = [f'{self._global_sdfg.name}_t *__state']

        # Write callback function definition
        self._localcode.write(
            """
DACE_EXPORTED void __dace_runkernel_{fname}({fargs});
void __dace_runkernel_{fname}({fargs})
{{
""".format(fname=kernel_name, fargs=', '.join(state_param + kernel_args_typed + extra_call_args_typed)), sdfg, state_id,
            node)

        if is_persistent:
            self._localcode.write('''
int dace_number_SMs;
DACE_GPU_CHECK({backend}DeviceGetAttribute(&dace_number_SMs, {backend}DevAttrMultiProcessorCount, 0));
int dace_number_blocks = ((int) ceil({fraction} * dace_number_SMs)) * {occupancy};
                '''.format(fraction=Config.get('compiler', 'cuda', 'persistent_map_SM_fraction'),
                           occupancy=Config.get('compiler', 'cuda', 'persistent_map_occupancy'),
                           backend=self.backend))

        if create_grid_barrier:
            gbar = '__gbar_' + kernel_name
            self._localcode.write('    cub::GridBarrierLifetime %s;\n' % gbar, sdfg, state_id, node)
            self._localcode.write(
                '{}.Setup({});'.format(gbar,
                                       ' * '.join(_topy(grid_dims)) if not is_persistent else 'dace_number_blocks'),
                sdfg, state_id, node)
            extra_kernel_args.append('(void *)((cub::GridBarrier *)&%s)' % gbar)

        # Compute dynamic shared memory
        dynsmem_size = 0
        # For all access nodes, if array storage == GPU_Shared and size is
        # symbolic, add it. If nested SDFG, check all internal arrays
        for node in dfg_scope.nodes():
            if isinstance(node, nodes.AccessNode):
                arr = sdfg.arrays[node.data]
                if (arr.storage == dtypes.StorageType.GPU_Shared and arr.transient):
                    numel = functools.reduce(lambda a, b: a * b, arr.shape)
                    if symbolic.issymbolic(numel, sdfg.constants):
                        dynsmem_size += numel
            elif isinstance(node, nodes.NestedSDFG):
                for sdfg_internal, _, arr in node.sdfg.arrays_recursive():
                    if (arr is not None and arr.storage == dtypes.StorageType.GPU_Shared and arr.transient):
                        numel = functools.reduce(lambda a, b: a * b, arr.shape)
                        if symbolic.issymbolic(numel, sdfg_internal.constants):
                            dynsmem_size += numel

        max_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))
        if max_streams >= 0:
            cudastream = '__state->gpu_context->streams[%d]' % scope_entry._cuda_stream
        else:
            cudastream = 'nullptr'

        # make sure dynamic map inputs are properly handled
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            self._localcode.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                sdfg, state_id, scope_entry)

        gdims = 'dace_number_blocks, 1, 1' if is_persistent else ', '.join(_topy(grid_dims))
        bdims = ', '.join(_topy(block_dims))

        # Prepare an empty-grid check for runtime grids
        dimcheck = ''
        if is_persistent:
            dimcheck = 'dace_number_blocks == 0'
        else:
            for gdim in grid_dims:
                if symbolic.issymbolic(gdim) and (gdim > 0) != True:
                    if not dimcheck:
                        dimcheck = f'({_topy(gdim)}) == 0'
                    else:
                        dimcheck += f' || ({_topy(gdim)}) == 0'

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
                }}''', sdfg, state_id, scope_entry)

        self._localcode.write(
            '''
void  *{kname}_args[] = {{ {kargs} }};
gpuError_t __err = {backend}LaunchKernel((void*){kname}, dim3({gdims}), dim3({bdims}), {kname}_args, {dynsmem}, {stream});'''
            .format(kname=kernel_name,
                    kargs=', '.join(['(void *)&' + arg for arg in prototype_kernel_args] + extra_kernel_args),
                    gdims=gdims,
                    bdims=bdims,
                    dynsmem=_topy(dynsmem_size),
                    stream=cudastream,
                    backend=self.backend), sdfg, state_id, scope_entry)

        # Check kernel launch for errors
        self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});')

        self._emit_sync(self._localcode)

        # Close the runkernel function
        self._localcode.write('}')
        #######################
        # Add invocation to calling code (in another file)
        function_stream.write(
            'DACE_EXPORTED void __dace_runkernel_%s(%s);\n' %
            (kernel_name, ', '.join(state_param + kernel_args_typed + extra_call_args_typed)), sdfg, state_id,
            scope_entry)

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('{', sdfg, state_id, scope_entry)

        # Synchronize all events leading to dynamic map range connectors
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            if hasattr(e, '_cuda_event'):
                ev = e._cuda_event
                callsite_stream.write(
                    'DACE_GPU_CHECK({backend}EventSynchronize(__state->gpu_context->events[{ev}]));'.format(
                        ev=ev, backend=self.backend), sdfg, state_id, [e.src, e.dst])
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                sdfg, state_id, node)

        # Invoke kernel call
        callsite_stream.write(
            '__dace_runkernel_%s(%s);\n' %
            (kernel_name,
             ', '.join(['__state'] + [cpp.ptr(aname, arg, sdfg, self._frame)
                                      for aname, arg in kernel_args.items()] + extra_call_args)), sdfg, state_id,
            scope_entry)

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('}', sdfg, state_id, scope_entry)

        synchronize_streams(sdfg, state, state_id, scope_entry, scope_exit, callsite_stream, self)

        # Instrumentation (post-kernel)
        if instr is not None:
            callsite_stream.write(outer_stream.getvalue())

    def get_tb_maps_recursive(self, subgraph):
        res = []
        for node in subgraph.nodes():
            if isinstance(node, nodes.NestedSDFG):
                for state in node.sdfg.states():
                    tbmaps = self.get_tb_maps_recursive(state)
                    for map, sym_map in tbmaps:
                        for k in sym_map.values():
                            for kk, vv in node.symbol_mapping.items():
                                sym_map[k] = sym_map[k].subs(dace.symbol(kk), vv)
                        res.append((map, sym_map))
            elif isinstance(node, nodes.MapEntry) and node.schedule in (
                    dtypes.ScheduleType.GPU_Device,
                    dtypes.ScheduleType.GPU_ThreadBlock,
                    dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
            ):
                res.append((node.map, {dace.symbol(k): dace.symbol(k) for k in node.map.range.free_symbols}))
        return res

    def get_kernel_dimensions(self, dfg_scope):
        """
        Determines a GPU kernel's grid/block dimensions from map scopes.

        Ruleset for kernel dimensions:

            1. If only one map (device-level) exists, of an integer set ``S``,
                the block size is ``32x1x1`` and grid size is ``ceil(|S|/32)`` in
                1st dimension.
            2. If nested thread-block maps exist ``(T_1,...,T_n)``, grid
                size is ``|S|`` and block size is ``max(|T_1|,...,|T_n|)`` with
                block specialization.
            3. If block size can be overapproximated, it is (for
                dynamically-sized blocks that are bounded by a
                predefined size).
            4. If nested device maps exist, they generate extra grid dimensions (block size 1)
                as the sum of all their sizes ``(|T_1| + ... + |T_n|)``

        :note: Kernel dimensions are separate from the map
                variables, and they should be treated as such.
        :note: To make use of the grid/block 3D registers, we use multi-
                dimensional kernels up to 3 dimensions, and flatten the
                rest into the third dimension.
        """

        kernelmap_entry: nodes.MapEntry = dfg_scope.source_nodes()[0]
        grid_size = kernelmap_entry.map.range.size(True)[::-1]
        block_size = None
        is_persistent = (kernelmap_entry.map.schedule == dtypes.ScheduleType.GPU_Persistent)
        int_ceil = symbolic.int_ceil

        # Obtain thread-block maps from nested SDFGs
        subgraph = dfg_scope.scope_subgraph(kernelmap_entry)
        sub_maps = self.get_tb_maps_recursive(subgraph)

        # Introduce extra grid dimensions based on device sub-maps
        extra_dim_offsets: Dict[nodes.Map, symbolic.SymbolicType] = {}
        extra_grid_dims: List[symbolic.SymbolicType] = None
        for submap, sym_map in sub_maps:
            submap: nodes.Map
            if submap.schedule != dtypes.ScheduleType.GPU_Device or submap is kernelmap_entry.map:
                continue
            if extra_grid_dims is not None and len(submap.params) != len(extra_grid_dims):
                raise NotImplementedError(
                    'Multiple GPU_Device sub-ranges with different dimensionality not yet implemented (found: '
                    f'{len(submap.params)}, existing: {len(extra_grid_dims)}, map: {kernelmap_entry})')

            # Add and overapproximate sizes
            gsize = [s.subs(list(sym_map.items())) for s in submap.range.size()[::-1]]
            gsize = [symbolic.overapproximate(s) for s in gsize]
            if extra_grid_dims is None:
                extra_grid_dims = gsize
                extra_dim_offsets[submap] = [0] * len(submap.params)
            else:
                extra_dim_offsets[submap] = extra_grid_dims
                extra_grid_dims = [(sz + gsz) for sz, gsz in zip(extra_grid_dims, gsize)]
        if extra_grid_dims is None:
            extra_grid_dims = []
        grid_size.extend(extra_grid_dims)

        # Linearize (flatten) rest of dimensions to third
        if len(grid_size) > 3:
            grid_size[2] = functools.reduce(sympy.Mul, grid_size[2:], 1)
            del grid_size[3:]

        # Extend to 3 dimensions if necessary
        grid_size = grid_size + [1] * (3 - len(grid_size))

        # Thread-block map cases
        has_dtbmap = len(
            [tbmap for tbmap, _ in sub_maps if tbmap.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic]) > 0

        # keep only thread-block maps
        tb_maps_sym_map = [(tbmap, sym_map) for tbmap, sym_map in sub_maps
                           if tbmap.schedule == dtypes.ScheduleType.GPU_ThreadBlock]

        # Map thread-block size override
        block_size = kernelmap_entry.map.gpu_block_size
        if block_size is not None:
            # Complement to three dimensions
            block_size += [1] * (3 - len(block_size))
            # Linearize (flatten) rest of dimensions to third
            if len(block_size) > 3:
                block_size[2] = functools.reduce(sympy.Mul, block_size[2:], 1)
                del block_size[3:]

        # No thread-block maps
        if len(tb_maps_sym_map) == 0:
            if block_size is None:
                if has_dtbmap:
                    if (Config.get('compiler', 'cuda', 'dynamic_map_block_size') == 'max'):
                        raise NotImplementedError('max dynamic block size unimplemented')
                    else:
                        block_size = [
                            int(b) for b in Config.get('compiler', 'cuda', 'dynamic_map_block_size').split(',')
                        ]
                else:
                    def_bsize = Config.get('compiler', 'cuda', 'default_block_size')
                    warnings.warn(
                        f'No `gpu_block_size` property specified on map "{kernelmap_entry.map.label}". '
                        f'Falling back to the configuration entry `compiler.cuda.default_block_size`: {def_bsize}. '
                        'You can either specify the block size to use with the gpu_block_size property, '
                        'or by adding nested `GPU_ThreadBlock` maps, which map work to individual threads. '
                        'For more information, see https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

                    if (Config.get('compiler', 'cuda', 'default_block_size') == 'max'):
                        raise NotImplementedError('max dynamic block size unimplemented')
                    else:
                        block_size = [int(b) for b in Config.get('compiler', 'cuda', 'default_block_size').split(',')]

                    block_ndim = max(1, sum(1 if b != 1 else 0 for b in block_size))
                    grid_ndim = max(1, sum(1 if g != 1 else 0 for g in grid_size))
                    if block_ndim > grid_ndim:
                        linearized_remainder = prod(block_size[grid_ndim:])
                        block_size = block_size[:grid_ndim] + [1] * (3 - grid_ndim)
                        block_size[grid_ndim - 1] *= linearized_remainder
                        warnings.warn(f'Default block size has more dimensions ({block_ndim}) than kernel dimensions '
                                      f'({grid_ndim}) in map "{kernelmap_entry.map.label}". Linearizing block '
                                      f'size to {block_size}. Consider setting the ``gpu_block_size`` property.')

            assert (len(block_size) >= 1 and len(block_size) <= 3)

            # Grid size = ceil(|S|/32) for first dimension, rest = |S|
            grid_size = [int_ceil(gs, bs) for gs, bs in zip(grid_size, block_size)]

        else:
            # Find all thread-block maps to determine overall block size
            detected_block_sizes = [block_size] if block_size is not None else []
            for tbmap, sym_map in tb_maps_sym_map:
                tbsize = [s.subs(list(sym_map.items())) for s in tbmap.range.size()[::-1]]

                # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
                # The partial trailing thread-block is emitted as an if-condition
                # that returns on some of the participating threads
                tbsize = [symbolic.overapproximate(s) for s in tbsize]

                # Linearize (flatten) rest of dimensions to third
                if len(tbsize) > 3:
                    tbsize[2] = functools.reduce(sympy.Mul, tbsize[2:], 1)
                    del tbsize[3:]

                # Extend to 3 dimensions if necessary
                tbsize = tbsize + [1] * (3 - len(tbsize))

                if len(detected_block_sizes) == 0:
                    block_size = tbsize
                else:
                    block_size = [sympy.Max(sz, bbsz) for sz, bbsz in zip(block_size, tbsize)]

                if block_size != tbsize or len(detected_block_sizes) == 0:
                    detected_block_sizes.append(tbsize)

            # TODO: If grid/block sizes contain elements only defined within the
            #       kernel, raise an invalid SDFG exception and recommend
            #       overapproximation.

            if len(detected_block_sizes) > 1:

                # Error when both gpu_block_size and thread-block maps were defined and conflict
                if kernelmap_entry.map.gpu_block_size is not None:
                    raise ValueError('Both the `gpu_block_size` property and internal thread-block '
                                     'maps were defined with conflicting sizes for kernel '
                                     f'"{kernelmap_entry.map.label}" (sizes detected: {detected_block_sizes}). '
                                     'Use `gpu_block_size` only if you do not need access to individual '
                                     'thread-block threads, or explicit block-level synchronization (e.g., '
                                     '`__syncthreads`). Otherwise, use internal maps with the `GPU_Threadblock` or '
                                     '`GPU_ThreadBlock_Dynamic` schedules. For more information, see '
                                     'https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

                warnings.warn('Multiple thread-block maps with different sizes detected for '
                              f'kernel "{kernelmap_entry.map.label}": {detected_block_sizes}. '
                              f'Over-approximating to block size {block_size}.\n'
                              'If this was not the intent, try tiling one of the thread-block maps to match.')

            # both thread-block map and dynamic thread-block map exist at the same
            # time
            if has_dtbmap:
                raise NotImplementedError("GPU_ThreadBlock and GPU_ThreadBlock_Dynamic are currently "
                                          "not supported in the same scope")

        if is_persistent:
            grid_size = ['gridDim.x', '1', '1']

        # Check block size against configured maximum values, if those can be determined
        total_bsize = prod(block_size)
        total_limit = Config.get('compiler', 'cuda', 'block_size_limit')
        lastdim_limit = Config.get('compiler', 'cuda', 'block_size_lastdim_limit')
        if (total_bsize > total_limit) == True:
            raise ValueError(f'Block size for kernel "{kernelmap_entry.map.label}" ({block_size}) '
                             f'is larger than the possible number of threads per block ({total_limit}). '
                             'The kernel will potentially not run, please reduce the thread-block size. '
                             'To increase this limit, modify the `compiler.cuda.block_size_limit` '
                             'configuration entry.')
        if (block_size[-1] > lastdim_limit) == True:
            raise ValueError(f'Last block size dimension for kernel "{kernelmap_entry.map.label}" ({block_size}) '
                             'is larger than the possible number of threads in the last block dimension '
                             f'({lastdim_limit}). The kernel will potentially not run, please reduce the '
                             'thread-block size. To increase this limit, modify the '
                             '`compiler.cuda.block_size_lastdim_limit` configuration entry.')

        return grid_size, block_size, len(tb_maps_sym_map) > 0, has_dtbmap, extra_dim_offsets

    def generate_kernel_scope(self, sdfg: SDFG, dfg_scope: ScopeSubgraphView, state_id: int, kernel_map: nodes.Map,
                              kernel_name: str, grid_dims: list, block_dims: list, has_tbmap: bool, has_dtbmap: bool,
                              kernel_params: list, function_stream: CodeIOStream, kernel_stream: CodeIOStream):
        node = dfg_scope.source_nodes()[0]

        # Get the thread/block index type
        ttype = Config.get('compiler', 'cuda', 'thread_id_type')
        tidtype = getattr(dtypes, ttype, False)
        if not isinstance(tidtype, dtypes.typeclass):
            raise ValueError(f'Configured type "{ttype}" for ``thread_id_type`` does not match any DaCe data type. '
                             'See ``dace.dtypes`` for available types (for example ``int32``).')

        # allocating shared memory for dynamic threadblock maps
        if has_dtbmap:
            kernel_stream.write(
                '__shared__ dace::'
                'DynamicMap<{fine_grained}, {block_size}>'
                '::shared_type dace_dyn_map_shared;'.format(
                    fine_grained=('true'
                                  if Config.get_bool('compiler', 'cuda', 'dynamic_map_fine_grained') else 'false'),
                    block_size=functools.reduce(
                        (lambda x, y: x * y),
                        [int(x) for x in Config.get('compiler', 'cuda', 'dynamic_map_block_size').split(',')])), sdfg,
                state_id, node)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        kernel_stream.write('{', sdfg, state_id, node)

        # Add more opening braces for scope exit to close
        for dim in range(len(node.map.range) - 1):
            kernel_stream.write('{', sdfg, state_id, node)

        # Generate all index arguments for kernel grid
        krange = subsets.Range(kernel_map.range[::-1])
        kdims = krange.size()
        dsym = [symbolic.symbol('__DAPB%d' % i, nonnegative=True, integer=True) for i in range(len(krange))]
        bidx = krange.coord_at(dsym)

        # handle dynamic map inputs
        for e in dace.sdfg.dynamic_map_inputs(sdfg.states()[state_id], dfg_scope.source_nodes()[0]):
            kernel_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                sdfg, state_id,
                dfg_scope.source_nodes()[0])

        # do not generate an index if the kernel map is persistent
        if node.map.schedule != dtypes.ScheduleType.GPU_Persistent:
            # First three dimensions are evaluated directly
            for i in range(min(len(krange), 3)):
                varname = kernel_map.params[-i - 1]

                # If we defaulted to a fixed number of threads per block, offset by thread ID
                block_expr = 'blockIdx.%s' % _named_idx(min(i, 2))
                if not has_tbmap or has_dtbmap:
                    block_expr = '(%s * %s + threadIdx.%s)' % (block_expr, _topy(block_dims[i]), _named_idx(i))

                # Delinearize third dimension if necessary
                if i == 2 and len(krange) > 3:
                    block_expr = f'({block_expr} / ({_topy(functools.reduce(sympy.Mul, kdims[3:], 1))}))'

                expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)

                kernel_stream.write(f'{tidtype.ctype} {varname} = {expr};', sdfg, state_id, node)
                self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, tidtype.ctype)

            # Delinearize beyond the third dimension
            if len(krange) > 3:
                for i in range(3, len(krange)):
                    varname = kernel_map.params[-i - 1]

                    block_expr = 'blockIdx.z'
                    if not has_tbmap or has_dtbmap:
                        block_expr = '(%s * %s + threadIdx.z)' % (block_expr, _topy(block_dims[2]))

                    # true dim i = z / ('*'.join(kdims[i+1:])) % kdims[i]
                    block_expr = '((%s / (%s)) %% (%s))' % (
                        block_expr,
                        _topy(functools.reduce(sympy.Mul, kdims[i + 1:], 1)),
                        _topy(kdims[i]),
                    )

                    expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)
                    kernel_stream.write(f'{tidtype.ctype} {varname} = {expr};', sdfg, state_id, node)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, tidtype.ctype)

        # Dispatch internal code
        assert CUDACodeGen._in_device_code is False
        CUDACodeGen._in_device_code = True
        self._kernel_map = node
        self._kernel_state = sdfg.node(state_id)
        self._block_dims = block_dims
        self._grid_dims = grid_dims

        # Emit internal array allocation (deallocation handled at MapExit)
        self._frame.allocate_arrays_in_scope(sdfg, node, function_stream, kernel_stream)

        scope_entry = dfg_scope.source_nodes()[0]

        # Generate conditions for this block's execution using min and max
        # element, e.g., skipping out-of-bounds threads in trailing block
        # unless thsi is handled by another map down the line
        if (not has_tbmap and not has_dtbmap and node.map.schedule != dtypes.ScheduleType.GPU_Persistent):
            dsym_end = [d + bs - 1 for d, bs in zip(dsym, self._block_dims)]
            minels = krange.min_element()
            maxels = krange.max_element()
            for i, (v, minel, maxel) in enumerate(zip(kernel_map.params[::-1], minels, maxels)):
                condition = ''

                # Optimize conditions if they are always true
                if i >= 3 or (dsym[i] >= minel) != True:
                    condition += '%s >= %s' % (v, _topy(minel))
                if (i >= 3 or ((dsym_end[i] < maxel) != False and ((dsym_end[i] % self._block_dims[i]) != 0) == True)
                        or (self._block_dims[i] > maxel) == True):
                    if len(condition) > 0:
                        condition += ' && '
                    condition += '%s < %s' % (v, _topy(maxel + 1))
                if len(condition) > 0:
                    self._kernel_grid_conditions.append(f'if ({condition}) {{')
                    kernel_stream.write('if (%s) {' % condition, sdfg, state_id, scope_entry)
                else:
                    self._kernel_grid_conditions.append('{')
                    kernel_stream.write('{', sdfg, state_id, scope_entry)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           kernel_stream,
                                           skip_entry_node=True)

        if (not has_tbmap and not has_dtbmap and node.map.schedule != dtypes.ScheduleType.GPU_Persistent):
            for _ in kernel_map.params:
                kernel_stream.write('}', sdfg, state_id, node)

        self._block_dims = None
        self._kernel_map = None
        self._kernel_state = None
        CUDACodeGen._in_device_code = False
        self._grid_dims = None

    def get_next_scope_entries(self, dfg, scope_entry):
        parent_scope_entry = dfg.entry_node(scope_entry)
        # We're in a nested SDFG, use full graph
        if parent_scope_entry is None:
            parent_scope = dfg
        else:
            parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all non-sequential scopes from the same level
        all_scopes = [
            node for node in parent_scope.topological_sort(scope_entry)
            if isinstance(node, nodes.EntryNode) and node.map.schedule != dtypes.ScheduleType.Sequential
        ]

        # TODO: Fix to include *next* scopes, without concurrent scopes

        return all_scopes[all_scopes.index(scope_entry) + 1:]

    def generate_devicelevel_scope(self, sdfg, dfg_scope, state_id, function_stream, callsite_stream):
        # Sanity check
        assert CUDACodeGen._in_device_code == True

        dfg = sdfg.nodes()[state_id]
        sdict = dfg.scope_dict()
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]
        scope_map = scope_entry.map
        next_scopes = self.get_next_scope_entries(dfg, scope_entry)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        callsite_stream.write('{', sdfg, state_id, scope_entry)

        if scope_map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
            if self.backend == 'hip':
                raise NotImplementedError('Dynamic thread-block maps on HIP are currently unsupported')
            if len(scope_map.params) > 1:
                raise ValueError('Only one-dimensional maps are supported for dynamic block map schedule (got %d)' %
                                 len(scope_map.params))
            total_block_size = 1
            for bdim in self._block_dims:
                if symbolic.issymbolic(bdim, sdfg.constants):
                    raise ValueError('Block size has to be constant for block-wide dynamic map schedule (got %s)' %
                                     str(bdim))
                total_block_size *= bdim

            ##### TODO (later): Generalize
            # Find thread-block param map and its name
            if self._block_dims[1] != 1 or self._block_dims[2] != 1:
                raise NotImplementedError('Dynamic block map schedule only implemented for 1D blocks currently')

            # Define all input connectors of this map entry
            # Note: no need for a C scope around these, as there will not be
            #       more than one dynamic thread-block map in a GPU device map
            callsite_stream.write('unsigned int __dace_dynmap_begin = 0, __dace_dynmap_end = 0;', sdfg, state_id,
                                  scope_entry)

            outer_scope = sdfg.nodes()[state_id].entry_node(scope_entry)
            current_sdfg = sdfg
            while not outer_scope and current_sdfg:
                current_state = current_sdfg.parent
                nsdfg_node = current_sdfg.parent_nsdfg_node
                outer_scope = current_state.entry_node(nsdfg_node)
                current_sdfg = current_state.parent
            if not outer_scope:
                raise ValueError(f'Failed to find the outer scope of {scope_entry}')
            callsite_stream.write(
                'if ({} < {}) {{'.format(outer_scope.map.params[0],
                                         _topy(subsets.Range(outer_scope.map.range[::-1]).max_element()[0] + 1)), sdfg,
                state_id, scope_entry)

            # NOTE: Dynamic map inputs must be defined both outside and inside the dynamic Map schedule.
            # They define inside the schedule the bounds of the any nested Maps.
            # They define outside the schedule the bounds of the dynamic Map's for-loop invocation.
            # NOTE: The value of the dynamic Map's variable may differ inside and outside the schedule.
            for e in dace.sdfg.dynamic_map_inputs(dfg, scope_entry):
                callsite_stream.write(
                    self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn,
                                                        e.dst.in_connectors[e.dst_conn]), sdfg, state_id, scope_entry)

            dynmap_var = scope_map.params[0]
            dynmap_begin = scope_map.range[0][0]
            dynmap_end = scope_map.range[0][1] + 1
            dynmap_step = scope_map.range[0][2]
            if dynmap_step != 1:
                dynmap_var = f'{dynmap_var}_idx'
                dynmap_begin = 0
                dynmap_end = f'int_ceil({dynmap_end - dynmap_begin}, {dynmap_step})'
            callsite_stream.write(
                '__dace_dynmap_begin = {begin};\n'
                '__dace_dynmap_end = {end};'.format(begin=dynmap_begin, end=dynmap_end), sdfg, state_id, scope_entry)

            # close if
            callsite_stream.write('}', sdfg, state_id, scope_entry)

            callsite_stream.write(
                'dace::DynamicMap<{fine_grained}, {bsize}>::'
                'schedule(dace_dyn_map_shared, __dace_dynmap_begin, '
                '__dace_dynmap_end, {kmapIdx}, [&](auto {kmapIdx}, '
                'auto {param}) {{'.format(fine_grained=('true' if Config.get_bool(
                    'compiler', 'cuda', 'dynamic_map_fine_grained') else 'false'),
                                          bsize=total_block_size,
                                          kmapIdx=outer_scope.map.params[0],
                                          param=dynmap_var), sdfg, state_id, scope_entry)

            for e in dace.sdfg.dynamic_map_inputs(dfg, scope_entry):
                callsite_stream.write(
                    self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn,
                                                        e.dst.in_connectors[e.dst_conn]), sdfg, state_id, scope_entry)

            if dynmap_step != 1:
                callsite_stream.write(
                    f'auto {scope_map.params[0]} = {scope_map.range[0][0]} + {dynmap_step} * {dynmap_var};', sdfg,
                    state_id, scope_entry)

        elif scope_map.schedule == dtypes.ScheduleType.GPU_Device:
            dfg_kernel = self._kernel_state.scope_subgraph(self._kernel_map)
            grid_dims, block_dims, has_tbmap, has_dtbmap, extra_gdim_offsets = self.get_kernel_dimensions(dfg_kernel)

            if (self._kernel_map.map == dtypes.ScheduleType.GPU_Persistent and len(scope_map.params) > 1):
                raise ValueError('Only one-dimensional device maps are currently supported '
                                 'for persistent kernel maps (got %d)'.format(len(scope_map.params)))

            if self._kernel_map.schedule == dtypes.ScheduleType.GPU_Persistent:
                is_persistent = True
                block_dims = self._block_dims
                node = dfg_scope.source_nodes()[0]

                device_map_range = subsets.Range(scope_map.range[::-1])
                device_map_dims = device_map_range.size()
                dsym = [
                    symbolic.symbol('__DAPB%d' % i, nonnegative=True, integer=True)
                    for i in range(len(device_map_range))
                ]
                bidx = device_map_range.coord_at(dsym)

                # handle dynamic map inputs
                for e in dace.sdfg.dynamic_map_inputs(dfg, scope_entry):
                    callsite_stream.write(
                        self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn,
                                                            e.dst.in_connectors[e.dst_conn]), sdfg, state_id,
                        scope_entry)

                # variables that need to be declared + the value they need to be initialized with
                declarations = []

                for i in range(min(len(device_map_range), 3)):
                    varname = scope_map.params[-i - 1]

                    # Delinearize third dimension if necessary
                    if i == 2 and len(device_map_range) > 3:
                        block_expr = '(blockIdx.z / (%s))' % _topy(functools.reduce(sympy.Mul, device_map_dims[3:], 1))
                    else:
                        block_expr = 'blockIdx.%s' % _named_idx(i)
                        # If we defaulted to 32 threads per block, offset by thread ID
                        if not has_tbmap or has_dtbmap:
                            block_expr = '(%s * %s + threadIdx.%s)' % (block_expr, _topy(block_dims[i]), _named_idx(i))

                    expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)

                    declarations.append((varname, expr))

                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

                # Delinearize beyond the third dimension
                if len(device_map_range) > 3:
                    for i in range(3, len(device_map_range)):
                        varname = scope_map.params[-i - 1]
                        # true dim i = z / ('*'.join(kdims[i+1:])) % kdims[i]
                        block_expr = '(blockIdx.z / (%s)) %% (%s)' % (
                            _topy(functools.reduce(sympy.Mul, device_map_dims[i + 1:], 1)),
                            _topy(device_map_dims[i]),
                        )

                        expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)

                        declarations.append((varname, expr))

                        self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

                kmap_min = subsets.Range(self._kernel_map.range[::-1]).min_element()
                kmap_max = subsets.Range(self._kernel_map.range[::-1]).max_element()

                # if has_tbmap == False and has_dtbmap == False:
                dsym_end = [d + bs - 1 for d, bs in zip(dsym, self._block_dims)]
                minels = device_map_range.min_element()
                maxels = device_map_range.max_element()
                for i, (v, minel, maxel) in enumerate(zip(scope_map.params[::-1], minels, maxels)):
                    condition = ''

                    # Optimize conditions if they are always true
                    if i >= 3 or (dsym[i] >= minel) != True:
                        condition += '%s >= %s' % (v, _topy(minel))
                    if (i >= 3
                            or ((dsym_end[i] < maxel) != False and ((dsym_end[i] % self._block_dims[i]) != 0) == True)
                            or (self._block_dims[i] > maxel) == True):
                        if len(condition) > 0:
                            condition += ' && '
                        if has_dtbmap:
                            condition += '{mapIdx} < int_ceil({max}, {bs}) * {bs}'.format(
                                mapIdx=v,
                                max=_topy(maxel + 1),
                                bs=_topy(block_dims[i]),
                            )
                        else:
                            condition += '%s < %s' % (v, _topy(maxel + 1))

                    if is_persistent and not has_tbmap:
                        stride = 'gridDim.x * {}'.format(_topy(block_dims[i]))
                    elif is_persistent and has_tbmap:
                        stride = 'gridDim.x'
                    else:
                        stride = self._grid_dims[i] if has_tbmap \
                            else (kmap_max[i] + 1 - kmap_min[i])

                    if len(condition) > 0:
                        varname, expr = declarations.pop(0)
                        callsite_stream.write(
                            'for (int {varname} = {expr}; {cond}; {varname} += '
                            '{stride}) {{'.format(
                                varname=varname,
                                expr=expr,
                                cond=condition,
                                stride=stride,
                                pers=is_persistent,
                            ), sdfg, state_id, node)
                    else:
                        # will only be entered once
                        varname, expr = declarations.pop(0)
                        callsite_stream.write('int {varname} = {expr};\n'
                                              '{{'.format(
                                                  varname=varname,
                                                  expr=expr,
                                              ), sdfg, state_id, node)
            else:  # Device map in Device map
                brange = subsets.Range(scope_map.range[::-1])
                kdims = brange.size()
                dsym = [
                    symbolic.symbol('__DAPT%d' % i, nonnegative=True, integer=True) - off
                    for i, off in zip(range(len(brange)), extra_gdim_offsets[scope_map])
                ]
                gdims = len(self._kernel_map.params)
                relevant_block_dims = self._block_dims[gdims:] + [1] * len(scope_map.params)
                dsym_end = [d + (bs * rng[2]) - 1 for d, bs, rng in zip(dsym, relevant_block_dims, brange)]
                tidx = brange.coord_at(dsym)
                if len(brange) + gdims > 3:
                    raise NotImplementedError('Delinearization with nested scope maps not yet implemented')

                # First three dimensions are evaluated directly
                for i in range(len(brange)):
                    varname = scope_map.params[-i - 1]
                    idx = _named_idx(i + gdims)
                    block_expr = f'blockIdx.{idx}'
                    if relevant_block_dims[i] != 1:
                        block_expr = f'(blockIdx.{idx} * {_topy(relevant_block_dims[i])} + threadIdx.{idx})'

                    expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                    callsite_stream.write('int %s = %s;' % (varname, expr), sdfg, state_id, scope_entry)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

                # Generate conditions for this subgrid's execution using min and max
                # element, e.g. skipping out-of-bounds threads
                minels = brange.min_element()
                maxels = brange.max_element()
                for i, (v, minel, maxel) in enumerate(zip(scope_map.params[::-1], minels, maxels)):
                    condition = ''

                    # Optimize conditions if they are always true
                    #############################################

                    # Block range start
                    if i >= 3 or (dsym[i] >= minel) != True:
                        condition += '%s >= %s' % (v, _topy(minel))

                    # Special case: block size is exactly the range of the map (0:b)
                    if i >= 3:
                        skipcond = False
                    else:
                        skipcond = dsym_end[i].subs({dsym[i]: minel}) == maxel

                    # Block range end
                    if i >= 3 or (not skipcond and (dsym_end[i] < maxel) != True):
                        if len(condition) > 0:
                            condition += ' && '
                        condition += '%s < %s' % (v, _topy(maxel + 1))

                    # Emit condition in code
                    if len(condition) > 0:
                        self._kernel_grid_conditions.append(f'if ({condition}) {{')
                        callsite_stream.write('if (%s) {' % condition, sdfg, state_id, scope_entry)
                    else:
                        self._kernel_grid_conditions.append('{')
                        callsite_stream.write('{', sdfg, state_id, scope_entry)

        else:
            for dim in range(len(scope_map.range)):
                callsite_stream.write('{', sdfg, state_id, scope_entry)

        # Emit internal array allocation (deallocation handled at MapExit)
        self._frame.allocate_arrays_in_scope(sdfg, scope_entry, function_stream, callsite_stream)

        # Generate all index arguments for block
        if scope_map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            if self._scope_has_collaborative_copy:
                # Emit post-copy synchronization
                callsite_stream.write('__syncthreads();', sdfg, state_id, scope_entry)
                # Reset thread-block-level information
                self._scope_has_collaborative_copy = False

            brange = subsets.Range(scope_map.range[::-1])
            kdims = brange.size()
            dsym = [symbolic.symbol('__DAPT%d' % i, nonnegative=True, integer=True) for i in range(len(brange))]
            dsym_end = [d + (bs * rng[2]) - 1 for d, bs, rng in zip(dsym, self._block_dims, brange)]
            tidx = brange.coord_at(dsym)

            # First three dimensions are evaluated directly
            for i in range(min(len(brange), 3)):
                varname = scope_map.params[-i - 1]

                # Delinearize third dimension if necessary
                if i == 2 and len(brange) > 3:
                    block_expr = '(threadIdx.z / (%s))' % _topy(functools.reduce(sympy.Mul, kdims[3:], 1))
                else:
                    block_expr = 'threadIdx.%s' % _named_idx(i)

                expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                callsite_stream.write('int %s = %s;' % (varname, expr), sdfg, state_id, scope_entry)
                self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

            # Delinearize beyond the third dimension
            if len(brange) > 3:
                for i in range(3, len(brange)):
                    varname = scope_map.params[-i - 1]
                    # true dim i = z / ('*'.join(kdims[i+1:])) % kdims[i]
                    block_expr = '(threadIdx.z / (%s)) %% (%s)' % (
                        _topy(functools.reduce(sympy.Mul, kdims[i + 1:], 1)),
                        _topy(kdims[i]),
                    )

                    expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                    callsite_stream.write('int %s = %s;' % (varname, expr), sdfg, state_id, scope_entry)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

            # Generate conditions for this block's execution using min and max
            # element, e.g. skipping out-of-bounds threads in trailing block
            minels = brange.min_element()
            maxels = brange.max_element()
            for i, (v, minel, maxel) in enumerate(zip(scope_map.params[::-1], minels, maxels)):
                condition = ''

                # Optimize conditions if they are always true
                #############################################

                # Block range start
                if i >= 3 or (dsym[i] >= minel) != True:
                    condition += '%s >= %s' % (v, _topy(minel))

                # Special case: block size is exactly the range of the map (0:b)
                if i >= 3:
                    skipcond = False
                else:
                    skipcond = dsym_end[i].subs({dsym[i]: minel}) == maxel

                # Block range end
                if i >= 3 or (not skipcond and (dsym_end[i] < maxel) != True):
                    if len(condition) > 0:
                        condition += ' && '
                    condition += '%s < %s' % (v, _topy(maxel + 1))

                # Emit condition in code
                if len(condition) > 0:
                    callsite_stream.write('if (%s) {' % condition, sdfg, state_id, scope_entry)
                else:
                    callsite_stream.write('{', sdfg, state_id, scope_entry)

        ##########################################################

        # need to handle subgraphs appropriately if they contain
        # dynamic thread block maps
        if any((isinstance(node, dace.nodes.MapEntry) and node != scope_entry
                and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic) for node in dfg_scope.nodes()):

            subgraphs = dace.sdfg.concurrent_subgraphs(dfg_scope)
            for subdfg in subgraphs:
                components = dace.sdfg.utils.separate_maps(
                    sdfg.nodes()[state_id],
                    subdfg,
                    dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
                )

                for c in components:
                    if not isinstance(c, dace.sdfg.scope.ScopeSubgraphView):
                        callsite_stream.write(
                            'if ({} < {}) {{'.format(scope_map.params[0],
                                                     _topy(subsets.Range(scope_map.range[::-1]).max_element()[0] + 1)),
                            sdfg, state_id, scope_entry)

                    self._dispatcher.dispatch_subgraph(sdfg,
                                                       c,
                                                       state_id,
                                                       function_stream,
                                                       callsite_stream,
                                                       skip_entry_node=False)

                    if not isinstance(c, dace.sdfg.scope.ScopeSubgraphView):
                        callsite_stream.write('}')

            # exit node gets lost in the process, thus needs to be
            # dispatched manually
            self._dispatcher.dispatch_node(sdfg, dfg_scope, state_id, scope_exit, function_stream, callsite_stream)

        else:
            # Generate contents normally
            self._dispatcher.dispatch_subgraph(sdfg,
                                               dfg_scope,
                                               state_id,
                                               function_stream,
                                               callsite_stream,
                                               skip_entry_node=True)

        # If there are any other threadblock maps down the road,
        # synchronize the thread-block / grid
        parent_scope, _ = xfh.get_parent_map(dfg, scope_entry)
        if (len(next_scopes) > 0 or parent_scope.schedule == dtypes.ScheduleType.Sequential):
            # Thread-block synchronization
            if scope_entry.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                callsite_stream.write('__syncthreads();', sdfg, state_id, scope_entry)
            # Grid synchronization (kernel fusion)
            elif scope_entry.map.schedule == dtypes.ScheduleType.GPU_Device \
                    and self._kernel_map.schedule == dtypes.ScheduleType.GPU_Device:
                # Escape grid conditions
                for _ in self._kernel_grid_conditions:
                    callsite_stream.write('}', sdfg, state_id, scope_entry)

                # Synchronize entire grid
                callsite_stream.write('__gbar.Sync();', sdfg, state_id, scope_entry)

                # Rewrite grid conditions
                for cond in self._kernel_grid_conditions:
                    callsite_stream.write(cond, sdfg, state_id, scope_entry)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        if self.node_dispatch_predicate(sdfg, dfg, node):
            # Dynamically obtain node generator according to class name
            gen = getattr(self, '_generate_' + type(node).__name__)
            gen(sdfg, dfg, state_id, node, function_stream, callsite_stream)
            return

        if not CUDACodeGen._in_device_code:
            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node, function_stream, callsite_stream)
            return

        self._locals.clear_scope(self._code_state.indentation + 1)

        if CUDACodeGen._in_device_code and isinstance(node, nodes.MapExit):
            return  # skip

        self._cpu_codegen.generate_node(sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_nsdfg_header(self, sdfg, state, state_id, node, memlet_references, sdfg_label):
        return 'DACE_DFI ' + self._cpu_codegen.generate_nsdfg_header(
            sdfg, state, state_id, node, memlet_references, sdfg_label, state_struct=False)

    def generate_nsdfg_call(self, sdfg, state, node, memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(sdfg,
                                                     state,
                                                     node,
                                                     memlet_references,
                                                     sdfg_label,
                                                     state_struct=False)

    def generate_nsdfg_arguments(self, sdfg, dfg, state, node):
        result = self._cpu_codegen.generate_nsdfg_arguments(sdfg, dfg, state, node)
        if self.create_grid_barrier:
            result.append(('cub::GridBarrier&', '__gbar', '__gbar'))

        # Add data from nested SDFGs to kernel arguments
        result.extend([(atype, aname, aname) for atype, aname, _ in self.extra_nsdfg_args])
        for arg in self.extra_nsdfg_args:
            defined_type, ctype = self._dispatcher.defined_vars.get(arg[1], 1)
            self._dispatcher.defined_vars.add(arg[1], defined_type, ctype)

        return result

    def _generate_NestedSDFG(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        self._cpu_codegen._generate_NestedSDFG(sdfg, dfg, state_id, node, function_stream, callsite_stream)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        if node.map.schedule == dtypes.ScheduleType.GPU_Device:
            # Remove grid invocation conditions
            for i in range(len(node.map.params)):
                if self._kernel_grid_conditions:
                    self._kernel_grid_conditions.pop()

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            # Close block invocation conditions
            for i in range(len(node.map.params)):
                callsite_stream.write('}', sdfg, state_id, node)

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
            # Close lambda function
            callsite_stream.write('});', sdfg, state_id, node)
            # Close block invocation
            callsite_stream.write('}', sdfg, state_id, node)
            return

        self._cpu_codegen._generate_MapExit(sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def make_ptr_vector_cast(self, *args, **kwargs):
        return cpp.make_ptr_vector_cast(*args, **kwargs)


########################################################################
########################################################################
########################################################################
########################################################################
# Helper functions and classes


def _topy(arr):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr, cpp_mode=True))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d, cpp_mode=True)) for d in arr]


def _named_idx(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx > 2:
        raise ValueError('idx must be between 0 and 2, got %d' % idx)
    return ('x', 'y', 'z')[idx]


def _get_storagename(storage):
    """ Returns a string containing the name of the storage location.
        Example: dtypes.StorageType.GPU_Shared will return "Shared". """
    sname = str(storage)
    return sname[sname.rindex('_') + 1:]


def _get_const_params(dfg_scope):
    state = dfg_scope.graph
    sdfg = dfg_scope.parent
    scope_entry = dfg_scope.source_nodes()[0]
    scope_exit = dfg_scope.sink_nodes()[0]
    input_params = set(e.data.data for e in state.in_edges(scope_entry))
    output_params = set(e.data.data for e in state.out_edges(scope_exit))
    toplevel_params = set(node.data for node in dfg_scope.nodes()
                          if isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].toplevel)
    dynamic_inputs = set(e.data.data for e in dace.sdfg.dynamic_map_inputs(state, scope_entry))
    return input_params - (output_params | toplevel_params | dynamic_inputs)
