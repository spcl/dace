# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import functools
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import sympy
from six import StringIO

import dace
from dace import data as dt, Memlet
from dace import dtypes, registry
from dace import subsets, symbolic
from dace.codegen import common, cppunparse
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import update_persistent_desc
from dace.codegen.targets.cpp import (codeblock_to_cpp, cpp_array_expr, memlet_copy_to_absolute_strides, sym2cpp,
                                      synchronize_streams, unparse_cr, mangle_dace_state_struct_name)
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import (SDFG, ScopeSubgraphView, SDFGState, has_dynamic_map_inputs, is_array_stream_view,
                       is_devicelevel_gpu, nodes, scope_contains_scope)
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.transformation import helpers as xfh
from dace.transformation.passes import analysis as ap

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen


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


@registry.autoregister_params(name='soft_hier')
class SoftHierCodeGen(TargetCodeGenerator):
    """ GPU (SoftHier/HIP) code generator. """
    target_name = 'soft_hier'
    title = 'SoftHier'
    _in_device_code = False

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._locals = cppunparse.CPPLocals()
        self._ldepth = 0
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self.create_grid_barrier = False
        self.dynamic_tbmap_type = None
        self.extra_nsdfg_args = []
        SoftHierCodeGen._in_device_code = False
        self._cpu_codegen: Optional['CPUCodeGen'] = None
        self._block_dims = None
        self._grid_dims = None
        self._kernel_map = None
        self._kernel_state = None
        self._kernel_grid_conditions: List[str] = []
        self._scope_has_collaborative_copy = False
        self._has_async_dma = False
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

        # Stream name mapping
        self._stream_name_map: Dict[str, str] = {}

        # dma core map dictionary
        self._dma_core_map = {}

        # tcdm array hbm map dic
        self._tcdm_hbm_map = {}


        # Register additional SoftHier dispatchers
        dispatcher.register_map_dispatcher(dtypes.SOFTHIER_SCHEDULES, self)

        dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)

        dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        soft_hier_storage = [dtypes.StorageType.SoftHier_HBM, dtypes.StorageType.SoftHier_TCDM]
        dispatcher.register_array_dispatcher(soft_hier_storage, self)
        dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)

        for storage in soft_hier_storage:
            for other_storage in dtypes.StorageType:
                dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        # Note down all allocated SoftHier_TCDM arrays with their sizes
        self.tcdm_offset = 0
        self._soft_hier_dims = []
        # Register illegal copies
        # cpu_unpinned_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal]
        # gpu_private_storage = [dtypes.StorageType.GPU_Shared]
        # illegal_copy = IllegalCopy()
        # for st in cpu_unpinned_storage:
        #     for gst in gpu_private_storage:
        #         dispatcher.register_copy_dispatcher(st, gst, None, illegal_copy)
        #         dispatcher.register_copy_dispatcher(gst, st, None, illegal_copy)
        # for st in cpu_unpinned_storage:
        #     for sched_type in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock]:
        #         # NOTE: Only reading to GPU has an exception (for Scalar inputs)
        #         dispatcher.register_copy_dispatcher(st,
        #                                             dtypes.StorageType.Register,
        #                                             sched_type,
        #                                             illegal_copy,
        #                                             predicate=cpu_to_gpu_cpred)
        #         dispatcher.register_copy_dispatcher(dtypes.StorageType.Register, st, sched_type, illegal_copy)
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
        self.language = 'shcc' if self.backend == 'cuda' else 'cpp'
        target_type = "" if self.backend == 'cuda' else self.backend
        self._codeobject = CodeObject(sdfg.name + '_' + 'soft_hier',
                                      '',
                                      self.language,
                                      SoftHierCodeGen,
                                      'SoftHier',
                                      target_type=target_type)

        # Find GPU<->GPU strided copies that cannot be represented by a single copy command
        from dace.transformation.dataflow import CopyToMap
        for e, state in list(sdfg.all_edges_recursive()):
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode):
                nsdfg = state.parent
                if (e.src.desc(nsdfg).storage == dtypes.StorageType.SoftHier_HBM
                        and e.dst.desc(nsdfg).storage == dtypes.StorageType.SoftHier_HBM):
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
        
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.AccessNode):
                desc_stream = node.desc(parent)
                if isinstance(desc_stream, dt.Stream) and desc_stream.storage == dtypes.StorageType.SoftHier_TCDM:
                    for edge in parent.out_edges(node):
                        if isinstance(edge.dst, nodes.AccessNode):
                            desc_dst = edge.dst.desc(parent)
                            if isinstance(desc_dst, dt.Array) and desc_dst.storage == dtypes.StorageType.SoftHier_TCDM:
                                self._stream_name_map[node.data] = f"{edge.dst.data}"
                                # print(f"SoftHier: Stream {node.data} mapped to { self._stream_name_map[node.data]}")
                                break

                    for edge in parent.in_edges(node):
                        if isinstance(edge.src, nodes.AccessNode):
                            desc_src = edge.src.desc(parent)
                            if isinstance(desc_src, dt.Array) and desc_src.storage == dtypes.StorageType.SoftHier_TCDM:
                                if node.data in self._stream_name_map:
                                    if self._stream_name_map[node.data] != edge.src.data:
                                        raise ValueError(f"Stream {node.data} already mapped to {self._stream_name_map[node.data]}")
                                else:
                                    self._stream_name_map[node.data] = f"{edge.src.data}"
                                    # print(f"SoftHier: Stream {node.data} mapped to { self._stream_name_map[node.data]}")
                                    break

        for edge, parent in sdfg.all_edges_recursive():
            # get the source of the edge
            src_node = edge.src
            dst_node = edge.dst
            path = parent.memlet_path(edge)
            if len(path) > 0:
                src_node = path[0].src
                dst_node = path[-1].dst
                if isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode):
                    src_desc = src_node.desc(parent)
                    if src_desc.storage == dtypes.StorageType.SoftHier_HBM:    
                        if edge.data.data not in self._dma_core_map:
                            self._dma_core_map[edge.data.data] = len(self._dma_core_map)
                        if isinstance(dst_node, nodes.AccessNode):
                            dst_desc = dst_node.desc(parent)
                            if dst_desc.storage == dtypes.StorageType.SoftHier_TCDM and isinstance(dst_desc, dt.Array):
                                if dst_node.data not in self._tcdm_hbm_map:
                                    self._tcdm_hbm_map[dst_node.data] = edge.data.data
                                else:
                                    if self._tcdm_hbm_map[dst_node.data] != edge.data.data:
                                        raise ValueError(f"Array {dst_node.data} already mapped to {self._tcdm_hbm_map[dst_node.data]}")

                if isinstance(dst_node, nodes.AccessNode):   
                    dst_desc = dst_node.desc(parent)
                    if dst_desc.storage == dtypes.StorageType.SoftHier_HBM:
                        if edge.data.data not in self._dma_core_map:
                            self._dma_core_map[edge.data.data] = len(self._dma_core_map)
                        if isinstance(src_node, nodes.AccessNode):
                            src_desc = src_node.desc(parent)
                            if src_desc.storage == dtypes.StorageType.SoftHier_TCDM and isinstance(src_desc, dt.Array):
                                if src_node.data not in self._tcdm_hbm_map:
                                    self._tcdm_hbm_map[src_node.data] = edge.data.data
                                else:
                                    if self._tcdm_hbm_map[src_node.data] != edge.data.data:
                                        raise ValueError(f"Array {src_node.data} already mapped to {self._tcdm_hbm_map[src_node.data]}")


        print(f"SoftHier: DMA core map: {self._dma_core_map}")
        print(f"SoftHier: TCDM HBM map: {self._tcdm_hbm_map}")

        # Annotate SoftHier streams and events
        self._cuda_streams, self._cuda_events = self._compute_cudastreams(sdfg)

        # Find points where memory should be released to the memory pool
        self._compute_pool_release(sdfg)

        # Write GPU context to state structure
        self._frame.statestruct.append('int filler;')

        # Collect all defined symbols and argument lists with one traversal
        shared_transients = {}
        for state, node, defined_syms in sdutil.traverse_sdfg_with_defined_symbols(sdfg, recursive=True):
            if (isinstance(node, nodes.MapEntry)
                    and node.map.schedule in (dtypes.ScheduleType.SoftHier_Device, )):
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
    uint32_t threshold = {poolcfg if poolcfg != -1 else 'UINT64_MAX'};
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);            
'''

        self._codeobject.code = """
// #include <{backend_header}>
// #include <dace/dace.h>
#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#define floor(x) ((x))
#define Mod(x, y) ((x) % (y))
{file_header}

static uint64_t HBM_ADDRESS_SPACE = {hbm_address_space};
static uint64_t HBM_ADDRESS_BASE = {hbm_address_base};

int __dace_init_cuda(struct {sdfg_state_name} *__state{params});
int __dace_exit_cuda(struct {sdfg_state_name} *__state);

typedef struct DacePlacementInfo
{{
    uint32_t channel_id;
    uint32_t tile_offset;
}} DacePlacementInfo;

{other_globalcode}

int __dace_init_cuda(struct {sdfg_state_name} *__state{params}) {{
    
    {pool_header}

    // __state->gpu_context = new dace::cuda::Context({nstreams}, {nevents});

    {initcode}

    return 0;
}}

int __dace_exit_cuda(struct {sdfg_state_name} *__state) {{
    {exitcode}
    int __err = 0;
    // delete __state->gpu_context;
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
           nstreams=max(1, self._cuda_streams),
           nevents=max(1, self._cuda_events),
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg,
           hbm_address_space=dace.config.Config.get("backend", "softhier", "HBM_ADDRESS_SPACE"),
           hbm_address_base=dace.config.Config.get("backend", "softhier", "HBM_ADDRESS_BASE"))

        return [self._codeobject]

    def node_dispatch_predicate(self, sdfg, state, node):
        if hasattr(node, 'schedule'):  # NOTE: Works on nodes and scopes
            if node.schedule in dtypes.SOFTHIER_SCHEDULES:
                return True
        if SoftHierCodeGen._in_device_code:
            return True
        return False

    def state_dispatch_predicate(self, sdfg, state):
        # print(f"SoftHier: State dispatch predicate for {state.label}")
        if self._toplevel_schedule in dtypes.SOFTHIER_SCHEDULES:
            return True
        # for node in state.sink_nodes():
        #     if hasattr(node, '_cuda_stream'):
        #         return True
        #     else:
        #         for e in state.in_edges(node):
        #             if hasattr(e.src, '_cuda_stream'):
        #                 return True
        # for s, _ in self.pool_release.values():
        #     if s is state:
        #         return True
        return False

    @property
    def has_initializer(self):
        return False

    @property
    def has_finalizer(self):
        return False

    @staticmethod
    def cmake_options():
        options = []

        # Override SoftHier toolkit
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

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                      declaration_stream: CodeIOStream) -> None:
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
        if (nodedesc.storage == dtypes.StorageType.SoftHier_HBM):
            result_decl.write('%s %s;\n' % (ctypedef, dataname))
            self._dispatcher.declared_arrays.add(dataname, DefinedType.Pointer, ctypedef)
        else:
            raise NotImplementedError("SoftHier: Unimplemented storage type " + str(nodedesc.storage))

        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        # print(f"SoftHier: Allocating {dataname}")
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
            return self.allocate_stream(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                                        allocation_stream)
        elif isinstance(nodedesc, dace.data.View):
            return self._cpu_codegen.allocate_view(sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream,
                                                   allocation_stream)
        elif isinstance(nodedesc, dace.data.Reference):
            return self._cpu_codegen.allocate_reference(sdfg, cfg, dfg, state_id, node, function_stream,
                                                        declaration_stream, allocation_stream)

        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        arrsize_malloc = '%s * sizeof(%s)' % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = '%s *' % nodedesc.dtype.ctype

        # Different types of SoftHier arrays
        if nodedesc.storage == dtypes.StorageType.SoftHier_HBM:
            if not declared:
                result_decl.write('%s %s;\n' % (ctypedef, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)

            # Strides are left to the user's discretion
            result_alloc.write('DACE_ACL_CHECK(aclrtMalloc((void**)&%s, %s));\n' %
                                ( dataname, arrsize_malloc))

            # if node.setzero:
            #     result_alloc.write('''
            #         if(flex_is_dm_core())
            #         {
            #             flex_dma_async_1d(local(accumulator), zomem(0), 2048);
            #             flex_dma_async_wait_all();
            #         }\n''')
            if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
                result_alloc.write(f'{dataname} += {cpp.sym2cpp(nodedesc.start_offset)};\n')
        elif nodedesc.storage == dtypes.StorageType.SoftHier_TCDM:
            write_type = 'uint32_t'
            if not declared:
                result_decl.write('%s %s;\n' % (write_type, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
            data_size = nodedesc.dtype.bytes  # Number of bytes per element
            total_size = arrsize * data_size  # Total size in bytes
            # Strides are left to the user's discretion
            # result_alloc.write('DACE_ACL_CHECK(aclrtMalloc((void**)&%s, %s));\n' %
            #                     ( dataname, arrsize_malloc))
            result_alloc.write(f"{dataname} = {self.tcdm_offset};\n")
            # if node.setzero:
            #     result_alloc.write('// DACE_ACL_CHECK(aclrtMemset(%s, 0, %s));\n' % ( dataname, arrsize_malloc))
            #     result_alloc.write(f'''
            #         if(flex_is_dm_core())
            #         {{
            #             flex_dma_async_1d(local({dataname}), zomem(0), {total_size});
            #             flex_dma_async_wait_all();
            #         }}
            #     ''')
            if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
                result_alloc.write(f'{dataname} += {cpp.sym2cpp(nodedesc.start_offset)};\n')
            self.tcdm_offset += total_size
        else:
            raise NotImplementedError("SoftHier: Unimplemented storage type " + str(nodedesc.storage))

        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)

    def allocate_stream(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                        declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        dataname = node.data
        allocname = cpp.ptr(dataname, nodedesc, sdfg, self._frame)
        result_decl = StringIO()
        result_alloc = StringIO()   
        if nodedesc.storage == dtypes.StorageType.SoftHier_TCDM:
            ctypedef = 'dace::SoftHierTCDMStream<%s>' % nodedesc.dtype.ctype
            self._dispatcher.defined_vars.add(allocname, DefinedType.Stream, ctypedef)
            array_name = self._stream_name_map[dataname]
            node_to_allocate = None
            node_desc_to_allocate = None
            for n, parent in sdfg.all_nodes_recursive():
                if isinstance(n, nodes.AccessNode) and n.data == array_name:
                    node_to_allocate = n
                    node_to_allocate_desc = n.desc(parent)
                    break
            if node_to_allocate is None:
                raise ValueError(f"Stream {dataname} mapped to array {array_name} not found")
            try:
                self._dispatcher.defined_vars.get(array_name)
                return
            except KeyError:
                pass  # The variable was not defined, we can continue

            # Check if array is already declared
            declared = False
            try:
                self._dispatcher.declared_arrays.get(array_name)
                declared = True  # Array was already declared in this or upper scopes
            except KeyError:  # Array not declared yet
                pass

            
            write_type = 'uint32_t'
            shape = node_to_allocate_desc.shape
            arrsize = prod(shape)
            if not declared:
                result_decl.write('%s %s;\n' % (write_type, array_name))
            ctypedef = '%s *' % node_to_allocate_desc.dtype.ctype
            self._dispatcher.defined_vars.add(array_name, DefinedType.Pointer, ctypedef)
            data_size = node_to_allocate_desc.dtype.bytes  # Number of bytes per element
            total_size = arrsize * data_size  # Total size in bytes
            # Strides are left to the user's discretion
            # result_alloc.write('DACE_ACL_CHECK(aclrtMalloc((void**)&%s, %s));\n' %
            #                     ( dataname, arrsize_malloc))
            result_alloc.write(f"{array_name} = {self.tcdm_offset};\n")
            self.tcdm_offset += total_size
            # if node_to_allocate.setzero:
            #     result_alloc.write(f'''
            #         if(flex_is_dm_core())
            #         {{
            #             flex_dma_async_1d(local({dataname}), zomem(0), {total_size});
            #             flex_dma_async_wait_all();
            #         }}
            #     ''')
        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)


    def deallocate_stream(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                          callsite_stream: CodeIOStream) -> None:
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        if nodedesc.storage == dtypes.StorageType.GPU_Global:
            if is_array_stream_view(sdfg, dfg, node):
                callsite_stream.write('dace::FreeGPUArrayStreamView(%s);' % dataname, cfg, state_id, node)
            else:
                callsite_stream.write('dace::FreeGPUStream(%s);' % dataname, cfg, state_id, node)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            dataname = f'({dataname} - {cpp.sym2cpp(nodedesc.start_offset)})'

        if self._dispatcher.declared_arrays.has(dataname):
            is_global = nodedesc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
            self._dispatcher.declared_arrays.remove(dataname, is_global=is_global)

        if isinstance(nodedesc, dace.data.Stream):
            return self.deallocate_stream(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream)
        elif isinstance(nodedesc, dace.data.View):
            return

        result_alloc = StringIO()
        if nodedesc.storage == dtypes.StorageType.SoftHier_HBM:
            result_alloc.write("DACE_ACL_CHECK(aclrtFree((void**)&%s));\n" % (dataname))
        elif nodedesc.storage in [
            dtypes.StorageType.SoftHier_TCDM
        ]:
            result_alloc.write(f"// Free {dataname}\n")
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

            # Maintain the same SoftHier stream in DFS order, add more when
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

        # Remove SoftHier streams from paths of non-gpu copies and CPU tasklets
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
                else:  # If we did not break, we do not need a SoftHier stream
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
                    # If there are two or more SoftHier streams involved in this
                    # edge, or the destination is unrelated to SoftHier
                    if (not hasattr(e.dst, '_cuda_stream') or e.src._cuda_stream != e.dst._cuda_stream):
                        for mpe in state.memlet_path(e):
                            mpe._cuda_event = events
                        events += 1

            state_events.append(events)

        # Maximum over all states
        max_streams = max(state_streams)
        max_events = max(state_events)

        return max_streams, max_events

    def _emit_copy(self, state_id: int, src_node: nodes.Node, src_storage: dtypes.StorageType, dst_node: nodes.Node,
                   dst_storage: dtypes.StorageType, dst_schedule: dtypes.ScheduleType,
                   edge: Tuple[nodes.Node, str, nodes.Node, str, Memlet], sdfg: SDFG, cfg: ControlFlowRegion,
                   dfg: StateSubgraphView, callsite_stream: CodeIOStream) -> None:
        u, uconn, v, vconn, memlet = edge
        state_dfg = cfg.state(state_id)
        # print('SoftHier: Emitting copy from', src_node, 'to', dst_node)
        cpu_storage_types = [
            dtypes.StorageType.CPU_Heap
        ]
        soft_hier_storage_types = [dtypes.StorageType.SoftHier_HBM, dtypes.StorageType.SoftHier_TCDM]

        copy_shape = memlet.subset.bounding_box_size()
        copy_shape = [symbolic.overapproximate(s) for s in copy_shape]
        # Determine directionality
        if (isinstance(src_node, nodes.AccessNode) and memlet.data == src_node.data):
            outgoing_memlet = True
        elif (isinstance(dst_node, nodes.AccessNode) and memlet.data == dst_node.data):
            outgoing_memlet = False
        else:
            raise LookupError('Memlet does not point to any of the nodes')

        if (isinstance(src_node, nodes.AccessNode)):
            src_node_desc = sdfg.arrays[src_node.data]
        if (isinstance(dst_node, nodes.AccessNode)):
            dst_node_desc = sdfg.arrays[dst_node.data]

        def _get_dma_core_expr(data_name):
            if data_name in self._dma_core_map:
                dma_id = self._dma_core_map[data_name]
            elif data_name in self._stream_name_map:
                data_name = self._stream_name_map[data_name]
                data_name = self._tcdm_hbm_map[data_name]
                dma_id = self._dma_core_map[data_name]
            elif data_name in self._tcdm_hbm_map:
                data_name = self._tcdm_hbm_map[data_name]
                dma_id = self._dma_core_map[data_name]
            expr = f"flex_get_core_id() == ({dma_id} % (ARCH_NUM_CORE_PER_CLUSTER - 1)) + 1"
            return expr
        
        def _emit_hbm_interleaved_code(nodedesc, src_name, s:subsets.Indices, callsite_stream, cfg, state_id, src_node, dst_node, is_load):   
            hbm_width = nodedesc.shape[1]
            hbm_height = nodedesc.shape[0]
            row_start = sym2cpp(s[0][0])
            col_start = sym2cpp(s[1][0])
            height_split = nodedesc.hbm_split_scheme[0]
            width_split = nodedesc.hbm_split_scheme[1]
            block_width = length_1
            block_height = length_0
            callsite_stream.write(f"const uint32_t tile_width = {src_name}_tile_width;")
            callsite_stream.write(f"const uint32_t tile_height = {src_name}_tile_height;")
            callsite_stream.write(f"const uint32_t row_start_offset = (({src_name} - {src_name}_base) / {data_size}) / {src_name}_width;")
            callsite_stream.write(f"const uint32_t col_start_offset = (({src_name} - {src_name}_base) / {data_size}) % {src_name}_width;")
            callsite_stream.write(f"const uint32_t col_start_temp = {col_start} + col_start_offset;")
            callsite_stream.write(f"const uint32_t col_start = col_start_temp % {src_name}_width;")
            callsite_stream.write(f"const uint32_t row_start = {row_start} + row_start_offset + col_start_temp / {src_name}_width;")
            callsite_stream.write(f"const uint32_t tile_row_index = row_start/tile_height;")
            callsite_stream.write(f"const uint32_t tile_col_index = col_start/tile_width;")
            callsite_stream.write(f"const uint32_t tile_row_offset = row_start%tile_height;")
            callsite_stream.write(f"const uint32_t tile_col_offset = col_start%tile_width;")
            callsite_stream.write(f"const uint32_t tile_index = tile_row_index*{width_split} + tile_col_index;")
            callsite_stream.write(f"const uint32_t channel_id = {src_name}_placement_info[tile_index].channel_id;")
            callsite_stream.write(f"const uint32_t num_blocks_per_tile = (tile_height/{block_height}) * (tile_width/{block_width});")
            callsite_stream.write(f"const uint32_t num_blocks_in_previous_tiles_in_channel = {src_name}_placement_info[tile_index].tile_offset * num_blocks_per_tile;")
            callsite_stream.write(f"const uint32_t block_row_index = tile_row_offset/{block_height};")
            callsite_stream.write(f"const uint32_t block_col_index = tile_col_offset/{block_width};")
            callsite_stream.write(f"const uint32_t block_index = block_row_index * (tile_width/{block_width}) + block_col_index;")
            callsite_stream.write(f"const uint32_t total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;")
            callsite_stream.write(f"const uint64_t block_addr = {src_name}_base + channel_id * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE + total_block_index * {block_height} * {block_width} * {data_size};")
            funcname = "flex_dma_async_1d"
            if is_load:
                callsite_stream.write(('    {func}({args});').format(
                        func=funcname,
                        args=', '.join([f'local({dst_expr})']+ 
                                    [f'hbm_addr(block_addr)'] + 
                                    [f'{block_height}*{block_width}*{data_size}'])), cfg, state_id, [src_node, dst_node]
                )
            else:
                callsite_stream.write(('    {func}({args});').format(
                        func=funcname,
                        args=', '.join([f'hbm_addr(block_addr)'] + 
                                    [f'local({src_expr})'] + 
                                    [f'{block_height}*{block_width}*{data_size}'])), cfg, state_id, [src_node, dst_node]
                )

        if (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode)
                and not SoftHierCodeGen._in_device_code
                and (src_storage in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned]
                     or dst_storage in [dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned])
                and not (src_storage in cpu_storage_types and dst_storage in cpu_storage_types)):
            pass
        # Copy within the SoftHier storage
        elif (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode) 
              and src_storage in soft_hier_storage_types and dst_storage in soft_hier_storage_types
              and isinstance(src_node_desc, dt.Array) and isinstance(dst_node_desc, dt.Array)):
            state = state_dfg
            # Obtain copy information
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (memlet_copy_to_absolute_strides(
                self._dispatcher, sdfg, state, edge, src_node, dst_node, self._cpu_codegen._packed_types))
            name = memlet.data
            # print memlet information
            # print(f"memlet: {memlet} {memlet.src_subset} {memlet.dst_subset}")
            nodedesc = sdfg.arrays[name]
            data_size = nodedesc.dtype.bytes
            dims = len(copy_shape)
            # print(f'copy_memory: {src_node} -> {dst_node}, {copy_shape}, {src_strides}, {dst_strides}, {src_expr}, {dst_expr}')
            callsite_stream.write(f'// copy_memory: {src_node} -> {dst_node}, {copy_shape}, {src_strides}, {dst_strides}, {src_expr}, {dst_expr}')
            # print(f'dims = {dims}')
            if vconn:
                # Rm. IN_
                dst_name = vconn[3:]
            elif isinstance(v, nodes.AccessNode):
                dst_name = v.data
            subset: subsets.Range = memlet.subset
            if uconn:
                # Rm. OUT_
                src_name = uconn[4:]
            elif isinstance(u, nodes.AccessNode):
                src_name = u.data
            #src_name = name
            # assert len(subset.string_list()) == 1
            
            if src_expr != src_name:
                src_expr = f"{src_expr} * {data_size}"
            if dst_expr != dst_name:
                dst_expr = f"{dst_expr} * {data_size}"
            is_sync = False
            # check whether the dst node has a out edge
            if len(state.out_edges(dst_node)) > 0:
                is_sync = True
            if len(state.in_edges(src_node)) > 0:
                is_sync = True
            callsite_stream.write(f'// is_sync = {is_sync}')
            self._has_async_dma = (not is_sync) or self._has_async_dma
            if dims == 1:
                beg, end, step = subset.ranges[0]
                length = (end + 1) - beg
                data_size = nodedesc.dtype.bytes  # Number of bytes per element
                if (
                    src_storage == dtypes.StorageType.SoftHier_HBM
                    and dst_storage == dtypes.StorageType.SoftHier_TCDM
                ):
                    callsite_stream.write(
                        "// SoftHier_HBM -> SoftHier_TCDM"
                    )
                    callsite_stream.write(
                        "if(flex_is_dm_core())"
                    )
                    callsite_stream.write("{", cfg, state_id, src_node)
                    funcname = "flex_dma_async_1d"
                    callsite_stream.write(('    {func}({args});').format(
                            func=funcname,
                            args=', '.join([f'local({dst_expr})'] + [f'hbm_addr({src_expr})'] + [f'{length}*{data_size}'])), cfg, state_id, [src_node, dst_node]
                    )
                    callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}", cfg, state_id, src_node)
                    # if is_sync:
                    #     callsite_stream.write("flex_intra_cluster_sync();")
                elif (
                    src_storage == dtypes.StorageType.SoftHier_TCDM
                    and dst_storage == dtypes.StorageType.SoftHier_TCDM
                ):
                    callsite_stream.write("// SoftHier_TCDM -> SoftHier_TCDM")
                    # when src_node name is same as dst_node name then no need to copy
                    if src_name == dst_name:
                        # do nothing
                        # print("src_expr == dst_expr")
                        # callsite_stream.write("/* src_expr == dst_expr */")
                        pass
                    else:
                        callsite_stream.write(
                            "if(flex_is_dm_core())"
                        )
                        callsite_stream.write("{", cfg, state_id, src_node)
                        funcname = "flex_dma_async_1d"
                        callsite_stream.write(('    {func}({args});').format(
                                func=funcname,
                                args=', '.join([f'local({dst_expr})'] + [f'local({src_expr})'] + [f'{length}*{data_size}'])), cfg, state_id, [src_node, dst_node]
                        )
                        callsite_stream.write("flex_dma_async_wait_all();")
                        callsite_stream.write("}", cfg, state_id, src_node)
                        # if is_sync:
                        #     callsite_stream.write("flex_intra_cluster_sync();")
                elif (
                    src_storage == dtypes.StorageType.SoftHier_TCDM
                    and dst_storage == dtypes.StorageType.SoftHier_HBM
                ):
                    callsite_stream.write(
                        "// SoftHier_TCDM -> SoftHier_HBM"
                    )
                    callsite_stream.write(
                        "if(flex_is_dm_core())"
                    )
                    callsite_stream.write("{", cfg, state_id, src_node)
                    funcname = "flex_dma_async_1d"
                    callsite_stream.write(('    {func}({args});').format(
                            func=funcname,
                            args=', '.join([f'hbm_addr({dst_expr})'] + 
                                           [f'local({src_expr})'] + 
                                           [f'{length}*{data_size}'])), cfg, state_id, [src_node, dst_node]
                    )
                    callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}", cfg, state_id, src_node)
                    # if is_sync:
                    #     callsite_stream.write("flex_intra_cluster_sync();")
                else:
                    if src_name == dst_name:
                        return
                    raise NotImplementedError(
                        f"Unimplemented copy type: {src_storage} -> {dst_storage}"
                    )
            elif dims == 2:
                # print(f'subset = {subset}; subset.string_list() = {subset.string_list()}')
                
                beg, end, step = subset.ranges[0]
                # print(f'index_0: {src_name} -> {dst_name}, {beg}, {end}, {step}')
                length_0 = (end + 1) - beg
                
                beg, end, step = subset.ranges[1]
                length_1 = (end + 1) - beg
                
                # print(f'index_1: {src_name} -> {dst_name}, {beg}, {end}, {step}')
                data_size = nodedesc.dtype.bytes
                if (
                    src_storage == dtypes.StorageType.SoftHier_HBM
                    and dst_storage == dtypes.StorageType.SoftHier_TCDM
                ):
                    s = subsets.Indices(memlet.src_subset)
                    # print(f"src_subset = {s[0][0]} {s[0][1]} {s[1][0]} {s[1][1]}")
                    # print(f"src data shape = {nodedesc.shape[0]} {nodedesc.shape[1]}")
                    

                    callsite_stream.write(
                        "// SoftHier_HBM -> SoftHier_TCDM 2D"
                    )
                    # def _get_dma_core_expr(data_name):
                    #     dma_id = self._dma_core_map[data_name]
                    #     expr = f"flex_get_core_id() == (ARCH_NUM_CORE_PER_CLUSTER - 1 - {dma_id})"
                    
                    cond_expr = _get_dma_core_expr(src_name)
                    callsite_stream.write(
                        f"if({cond_expr})"
                    )
                    callsite_stream.write("{", cfg, state_id, src_node)

                    if(nodedesc.is_hbm_interleaved):
                        _emit_hbm_interleaved_code(nodedesc, src_name, s, callsite_stream, cfg, state_id, src_node, dst_node, True)

                    else:
                        funcname = "flex_dma_sync_2d"
                        callsite_stream.write(('    {func}({args});').format(
                                func=funcname,
                                args=', '.join([f'local({dst_expr})'] + 
                                            [f'hbm_addr({src_expr})'] + 
                                            [f'{length_1}*{data_size}'] +
                                            [f'{dst_strides[0]}*{data_size}'] +
                                            [f'{src_strides[0]}*{data_size}'] +
                                            [f'{length_0}'])), cfg, state_id, [src_node, dst_node]
                        )
                    # if is_sync:
                    callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}", cfg, state_id, src_node)
                    # if is_sync:
                    #     callsite_stream.write("flex_intra_cluster_sync();")
                elif (
                    src_storage == dtypes.StorageType.SoftHier_TCDM
                    and dst_storage == dtypes.StorageType.SoftHier_TCDM
                ):
                    callsite_stream.write("// SoftHier_TCDM -> SoftHier_TCDM")
                    if src_name == dst_name:
                        # do nothing
                        pass
                    else:
                        callsite_stream.write(
                            "if(flex_is_dm_core())"
                        )
                        callsite_stream.write("{", cfg, state_id, src_node)
                        funcname = "flex_dma_sync_2d"
                        callsite_stream.write(('    {func}({args});').format(
                                func=funcname,
                                args=', '.join([f'local({dst_expr})'] + 
                                            [f'local({src_expr})'] + 
                                            [f'{length_1}*{data_size}'] +
                                            [f'{dst_strides[0]}*{data_size}'] +
                                            [f'{src_strides[0]}*{data_size}'] +
                                            [f'{length_0}'])), cfg, state_id, [src_node, dst_node]
                        )
                        # if is_sync:
                        callsite_stream.write("flex_dma_async_wait_all();")
                        callsite_stream.write("}", cfg, state_id, src_node)
                        # if is_sync:
                        #     callsite_stream.write("flex_intra_cluster_sync();")
                elif (
                    src_storage == dtypes.StorageType.SoftHier_TCDM
                    and dst_storage == dtypes.StorageType.SoftHier_HBM
                ):
                    s = subsets.Indices(memlet.dst_subset)
                    callsite_stream.write(
                        "// SoftHier_TCDM -> SoftHier_HBM"
                    )
                    
                    cond_expr = _get_dma_core_expr(dst_name)
                    callsite_stream.write(
                        f"if({cond_expr})"
                    )
                    callsite_stream.write("{", cfg, state_id, src_node)
                    if(nodedesc.is_hbm_interleaved):
                        _emit_hbm_interleaved_code(nodedesc, dst_name, s, callsite_stream, cfg, state_id, src_node, dst_node, False)
                        # if is_sync:
                        callsite_stream.write("flex_dma_async_wait_all();")
                    else:
                        funcname = "flex_dma_sync_2d"
                        callsite_stream.write(('    {func}({args});').format(
                                func=funcname,
                                args=', '.join([f'hbm_addr({dst_expr})'] + 
                                            [f'local({src_expr})'] + 
                                            [f'{length_1}*{data_size}'] +
                                            [f'{dst_strides[0]}*{data_size}'] +
                                            [f'{src_strides[0]}*{data_size}'] +
                                            [f'{length_0}'])), cfg, state_id, [src_node, dst_node]
                        )
                    # if is_sync:
                    #     callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}", cfg, state_id, src_node)
                elif (
                    src_storage == dtypes.StorageType.SoftHier_HBM
                    and dst_storage == dtypes.StorageType.SoftHier_HBM
                ):       
                    if src_expr == dst_expr:
                        # do nothing
                        pass
                    else:
                        raise NotImplementedError(
                            f"Unimplemented copy type: {src_storage} -> {dst_storage}"
                        )
                else:
                    raise NotImplementedError(
                        f"Unimplemented copy type: {src_storage} -> {dst_storage}"
                    )
        elif (isinstance(src_node, nodes.AccessNode) and isinstance(dst_node, nodes.AccessNode) 
              and src_storage in soft_hier_storage_types and dst_storage in soft_hier_storage_types
              and (isinstance(src_node_desc, dt.Stream) or isinstance(dst_node_desc, dt.Stream))):
            
            state = state_dfg
            is_sync = False
            # check whether the dst node has a out edge
            if len(state.out_edges(dst_node)) > 0:
                is_sync = True
            if len(state.in_edges(src_node)) > 0:
                is_sync = True
            copy_shape = edge.data.subset.size_exact()
            dims = len(copy_shape)
            
            self._has_async_dma = (not is_sync) or self._has_async_dma
            if vconn:
            # Rm. IN_
                dst_name = vconn[3:]
            elif isinstance(v, nodes.AccessNode):
                dst_name = v.data
            subset: subsets.Range = memlet.subset
            if uconn:
                # Rm. OUT_
                src_name = uconn[4:]
            elif isinstance(u, nodes.AccessNode):
                src_name = u.data

            if isinstance(src_node_desc, dt.Stream):
                callsite_stream.write(f'// copy_memory: {src_node} -> {dst_node}')
                callsite_stream.write(f'// is_sync = {is_sync}')
                data_size = dst_node_desc.dtype.bytes
                dst_subset = memlet.get_dst_subset(edge, state)        
                is_src_write = not memlet._is_data_src
                dst_expr = cpp.copy_expr(self._dispatcher,
                             sdfg,
                             dst_node.data,
                             memlet,
                             is_write=is_src_write,
                             offset=dst_subset,
                             relative_offset=False)
                if dst_expr != dst_name:
                    dst_expr = dst_expr + "*" + f"{data_size}"
                src_subset = memlet.get_src_subset(edge, state)
                pos_x_range = subsets.Range(src_subset[0:1])
                pos_x = pos_x_range.at([0],[1])
                pos_y_range = subsets.Range(src_subset[1:2])
                pos_y = pos_y_range.at([0],[1])
                pos_x = sym2cpp(pos_x)
                pos_y = sym2cpp(pos_y)
                # print(f"pos_x = {pos_x}, pos_y = {pos_y}")
                src_subset = subsets.Range(src_subset[2:])
                src_expr = sym2cpp(src_subset.at([0,0,0],[1,1,1]))
                src_size = src_subset.num_elements() * data_size
                src_expr = self._stream_name_map[src_node.data] + "+" + src_expr + " * " + str(src_size)
                if src_subset[-3] == dst_subset[-3]:
                    return
                if src_storage == dtypes.StorageType.SoftHier_TCDM and dst_storage == dtypes.StorageType.SoftHier_TCDM:
                    expr = _get_dma_core_expr(src_name)
                    callsite_stream.write(f"if ({expr})")
                    # callsite_stream.write("if (flex_is_dm_core())")
                    callsite_stream.write("{")
                    callsite_stream.write(f"bare_dma_start_1d(local({dst_expr}), dace_remote_xy({pos_x},{pos_y},{src_expr},{self._soft_hier_dims[0]}), {src_size});")
                    # if is_sync:
                    callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}")

            else:
                
                data_size = src_node_desc.dtype.bytes
                src_subset = memlet.get_src_subset(edge, state)        
                is_src_write = not memlet._is_data_src
                src_expr = cpp.copy_expr(self._dispatcher,
                             sdfg,
                             src_node.data,
                             memlet,
                             is_write=is_src_write,
                             offset=src_subset,
                             relative_offset=False)
                if src_expr != src_name:
                    src_expr = src_expr + "*" + f"{data_size}"
                dst_subset = memlet.get_dst_subset(edge, state)
                pos_x_range = subsets.Range(dst_subset[0:1])
                pos_x = pos_x_range.at([0],[1])
                pos_y_range = subsets.Range(dst_subset[1:2])
                pos_y = pos_y_range.at([0],[1])
                pos_x = sym2cpp(pos_x)
                pos_y = sym2cpp(pos_y)
                # print(f"pos_x = {pos_x}, pos_y = {pos_y}")
                dst_subset = subsets.Range(dst_subset[2:])
                dst_expr = sym2cpp(dst_subset.at([0,0,0],[1,1,1]))
                dst_size = dst_subset.num_elements() * data_size
                dst_expr = self._stream_name_map[dst_node.data] + "+" + dst_expr + " * " + str(dst_size)
                # check whether it is a broadcast
                (pos_x_start, pos_x_end, pos_x_step) = pos_x_range[0]
                (pos_y_start, pos_y_end, pos_y_step) = pos_y_range[0]
                is_broadcast = False
                if src_subset[-3] == dst_subset[-3] and pos_x_range.num_elements() == 1 and pos_y_range.num_elements() == 1:
                    return
                else:
                    is_broadcast = True
                    
                if src_storage == dtypes.StorageType.SoftHier_TCDM and dst_storage == dtypes.StorageType.SoftHier_TCDM:
                    expr = _get_dma_core_expr(dst_name)
                    callsite_stream.write(f"if ({expr})")
                    # callsite_stream.write("if (flex_is_dm_core())")
                    callsite_stream.write("{")
                    if is_broadcast:
                        callsite_stream.write(f"flex_dma_async_1d_broadcast(dace_remote_xy({pos_x_end-pos_x_step+1},{pos_y_end-pos_y_step+1},{dst_expr},{self._soft_hier_dims[0]}), local({src_expr}), {dst_size});")
                    else:
                        callsite_stream.write(f"bare_dma_start_1d(dace_remote_xy({pos_x},{pos_y},{dst_expr},{self._soft_hier_dims[0]}), local({src_expr}), {dst_size});")
                    # if is_sync:
                    callsite_stream.write("flex_dma_async_wait_all();")
                    callsite_stream.write("}")

                elif src_storage == dtypes.StorageType.SoftHier_HBM and dst_storage == dtypes.StorageType.SoftHier_TCDM:
                    name = memlet.data
                    nodedesc = sdfg.arrays[name]
                    raise NotImplementedError(
                        f"Unimplemented copy type: {src_storage} -> {dst_storage}"
                    )
                    if dims == 1:
                        beg, end, step = subset.ranges[0]
                        length = (end + 1) - beg
                        callsite_stream.write("if (flex_is_dm_core())")
                        callsite_stream.write("{")
                        callsite_stream.write(f"bare_dma_start_1d(dace_remote_xy({pos_x},{pos_y},{dst_expr}), hbm({src_expr}), {length * data_size});")
                        # if is_sync:
                        callsite_stream.write("flex_dma_async_wait_all();")
                        callsite_stream.write("}")
                    elif dims == 2:
                        beg, end, step = subset.ranges[0]
                        length_0 = (end + 1) - beg
                        beg, end, step = subset.ranges[1]
                        length_1 = (end + 1) - beg
                        if nodedesc.is_hbm_interleaved:
                            callsite_stream.write("if (flex_is_dm_core())")
                            callsite_stream.write("{")
                            s = subsets.Indices(memlet.src_subset)
                            _emit_hbm_interleaved_code(nodedesc, src_name, s, callsite_stream, cfg, state_id, src_node, dst_node)
                            # if is_sync:
                            callsite_stream.write("flex_dma_async_wait_all();")
                            callsite_stream.write("}")
                        else: # not interleaved
                            src_strides = src_node_desc.strides[-dims:]
                            dst_strides = dst_node_desc.strides[-dims:]
                            callsite_stream.write("if (flex_is_dm_core())")
                            callsite_stream.write("{")
                            funcname = "flex_dma_sync_2d"
                            callsite_stream.write(('    {func}({args});').format(
                                    func=funcname,
                                    args=', '.join([f'dace_remote_xy({pos_x},{pos_y},{dst_expr})'] + 
                                                [f'hbm_addr({src_expr})'] + 
                                                [f'{length_1}*{data_size}'] +
                                                [f'{dst_strides[0]}*{data_size}'] +
                                                [f'{src_strides[0]}*{data_size}'] +
                                                [f'{length_0}'])), cfg, state_id, [src_node, dst_node]
                            )
                            # if is_sync:
                            callsite_stream.write("flex_dma_async_wait_all();")
                            callsite_stream.write("}")
                    else:
                        raise NotImplementedError(
                            f"Not Implemented Dim > 2: {src_storage} -> {dst_storage}"
                        )
                # if is_sync:
                #     callsite_stream.write("flex_intra_cluster_sync();")
        else:
            print(
                sdfg,
                cfg,
                dfg,
                state_id,
                src_node,
                dst_node,
                edge,
                None,
                callsite_stream,
            )
            self._cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    memlet: Memlet, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        state = cfg.state(state_id)
        # print(f'copy_memory: {src_node} -> {dst_node}')
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
        self._emit_copy(state_id, src_node, src_storage, dst_node, dst_storage, dst_schedule, memlet, sdfg, cfg, dfg,
                        callsite_stream)

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        self._cpu_codegen.define_out_memlet(sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
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

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = False) -> None:
        # Two modes: device-level state and if this state has active streams
        # print('SoftHier: Generating state', state.label)
        if SoftHierCodeGen._in_device_code:
            self._has_async_dma = False
            self.generate_devicelevel_state(sdfg, cfg, state, function_stream, callsite_stream)
        else:
            # Active streams found. Generate state normally and sync with the
            # streams in the end
            self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream, generate_state_footer=False)

            # Reset thread-block-level information
            self._scope_has_collaborative_copy = False
            self._has_async_dma = False

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
                        (self.backend, stream), cfg, state.block_id)

            # After synchronizing streams, generate state footer normally
            callsite_stream.write('\n')

            # Emit internal transient array deallocation
            self._frame.deallocate_arrays_in_scope(sdfg, cfg, state, function_stream, callsite_stream)

            # Invoke all instrumentation providers
            for instr in self._frame._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, state, callsite_stream, function_stream)
        
    def generate_devicelevel_state(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState,
                                   function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # Special case: if this is a GPU grid state and something is reading
        # from a possible result of a collaborative write, sync first
        # print('SoftHier: Generating device-level state', state.label)
        # if self._toplevel_schedule == dtypes.ScheduleType.SoftHier_Device:
        #     for node in state.nodes():
        #         if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage == dtypes.StorageType.SoftHier_TCDM
        #                 and state.in_degree(node) == 0 and state.out_degree(node) > 0):
        #             if not self._scope_has_collaborative_copy:
        #                 callsite_stream.write('__syncthreads();', cfg, state.block_id)
        #             break

        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)
        if len(state.nodes()) == 0:
            return
        if self._has_async_dma:
            callsite_stream.write('if (flex_is_dm_core())')
            callsite_stream.write('{')
            callsite_stream.write('flex_dma_async_wait_all();')
            callsite_stream.write('}')
        callsite_stream.write('flex_intra_cluster_sync();')
    # NOTE: This function is ONLY called from the CPU side. Therefore, any
    # schedule that is out of the ordinary will raise an exception
    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: StateSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]

        state = cfg.state(state_id)
        # print("################################Using SoftHierCodeGen Scope######################################")
        # If in device-level code, call appropriate function
        if (self._kernel_map is not None and self._kernel_map.map.schedule in dtypes.SOFTHIER_SCHEDULES):
            # print("Generating device-level scope")
            self.generate_devicelevel_scope(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
            # print("Generated device-level scope")
            return

        # If not device-level code, ensure the schedule is correct
        if scope_entry.map.schedule not in (dtypes.ScheduleType.SoftHier_Device,):
            # print("Cannot schedule %s directly from non-GPU code" % scope_entry.map.schedule)
            raise TypeError('Cannot schedule %s directly from non-GPU code' % str(scope_entry.map.schedule))


        # self.create_grid_barrier = create_grid_barrier
        kernel_name = '%s_%d_%d_%d' % (scope_entry.map.label, sdfg.cfg_id, sdfg.node_id(state),
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
                    oldval = SoftHierCodeGen._in_device_code
                    SoftHierCodeGen._in_device_code = True
                    inner_name = cpp.ptr(node.data, desc, nsdfg, self._frame)
                    SoftHierCodeGen._in_device_code = oldval

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
        hbm_interleaved_args = {}
        for aname, arg in kernel_args.items():  # `list` wrapper is used to modify kernel_args within the loop
            def _generate_placement_info(aname, placement_scheme, split_scheme):
                            num_tiles = split_scheme[0] * split_scheme[1]
                            define_str = f"const DacePlacementInfo {aname}_placement_info[{num_tiles}] = \n"
                            define_str += "{\n"
                            assign_str = ""
                            tiling_offset_dic = {}
                            for i in range(num_tiles):
                                if placement_scheme[i] not in tiling_offset_dic.keys():
                                    tiling_offset_dic[placement_scheme[i]] = 0
                                assign_str += "{"
                                assign_str += f" .channel_id = {placement_scheme[i]}," # .channel_id = (no need for this)
                                assign_str += f" .tile_offset = {tiling_offset_dic[placement_scheme[i]]} " # .tile_offset =  (no need for this)
                                assign_str += "},\n"
                                tiling_offset_dic[placement_scheme[i]] += 1
                            # [:-2] to avoid last comma
                            return define_str + assign_str[:-2] + "\n" + "};\n"
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

                    SoftHierCodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    SoftHierCodeGen._in_device_code = False

                    self._dispatcher.defined_vars.add(inner_ptrname,
                                                      defined_type,
                                                      'const %s' % ctype,
                                                      allow_shadowing=True)

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname
                    if data_desc.is_hbm_interleaved:
                        self._globalcode.write(f"uint32_t {aname}_base;")
                        self._globalcode.write(f"uint32_t {aname}_height;")
                        self._globalcode.write(f"uint32_t {aname}_width;")
                        self._globalcode.write(f"uint32_t {aname}_tile_height;")
                        self._globalcode.write(f"uint32_t {aname}_tile_width;")
                        split_scheme = data_desc.hbm_split_scheme
                        placement_scheme = data_desc.hbm_placement_scheme
                        self._globalcode.write(_generate_placement_info(aname, placement_scheme, split_scheme))
                        self._dispatcher.defined_vars.add(f"{aname}_base", defined_type, ctype, allow_shadowing=True)         
                        hbm_interleaved_args[aname] = arg
            else:
                if aname in sdfg.arrays:
                    data_desc = sdfg.arrays[aname]
                    ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    is_global = data_desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                       dtypes.AllocationLifetime.Persistent,
                                                       dtypes.AllocationLifetime.External)
                    defined_type, ctype = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)
                    SoftHierCodeGen._in_device_code = True
                    inner_ptrname = cpp.ptr(aname, data_desc, sdfg, self._frame)
                    SoftHierCodeGen._in_device_code = False
                    self._dispatcher.defined_vars.add(inner_ptrname, defined_type, ctype, allow_shadowing=True)

                    # Rename argument in kernel prototype as necessary
                    aname = inner_ptrname
                    if data_desc.is_hbm_interleaved:
                        self._globalcode.write(f"uint32_t {aname}_base;")     
                        self._globalcode.write(f"uint32_t {aname}_height;")
                        self._globalcode.write(f"uint32_t {aname}_width;")
                        self._globalcode.write(f"uint32_t {aname}_tile_height;")
                        self._globalcode.write(f"uint32_t {aname}_tile_width;")
                        split_scheme = data_desc.hbm_split_scheme
                        placement_scheme = data_desc.hbm_placement_scheme
                        self._globalcode.write(_generate_placement_info(aname, placement_scheme, split_scheme))
                        self._dispatcher.defined_vars.add(f"{aname}_base", defined_type, ctype, allow_shadowing=True)
                        hbm_interleaved_args[aname] = arg     

            prototype_kernel_args[aname] = arg

        kernel_args_typed = [f'uint32_t {k}'
                             for k, v in prototype_kernel_args.items()]
        
        kernel_device_args_typed = []
        for k, v in prototype_kernel_args.items():
            kernel_device_args_typed.append(f'const uint32_t {k}')

                
        # print("Kernel Args Typed: ", kernel_args_typed)
        kernel_stream = CodeIOStream()
        self.generate_kernel_scope(sdfg, cfg, dfg_scope, state_id, scope_entry.map, kernel_name, grid_dims, block_dims,
                                   tbmap, dtbmap, kernel_args_typed, self._globalcode, kernel_stream)

        self._dispatcher.defined_vars.exit_scope(scope_entry)

        # Add extra kernel arguments for a grid barrier object
        # if create_grid_barrier:
        #     extra_kernel_args_typed.append('cub::GridBarrier __gbar')

        node = dfg_scope.source_nodes()[0]

        # Set kernel launch bounds
        # if node.gpu_launch_bounds == "-1":
        #     launch_bounds = ''
        # elif node.gpu_launch_bounds == "0":
        #     if any(symbolic.issymbolic(b) for b in block_dims):
        #         launch_bounds = ''
        #     else:
        #         launch_bounds = f'__launch_bounds__({_topy(prod(block_dims))})'
        # else:
        #     launch_bounds = f'__launch_bounds__({node.gpu_launch_bounds})'

        # Write kernel prototype
        self._localcode.write(
            'void %s(%s) {\n' %
            (kernel_name, ', '.join(kernel_device_args_typed)), sdfg, state_id, node)

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

        state_param = [f'{mangle_dace_state_struct_name(self._global_sdfg)} *__state']

        # Write callback function definition
        self._localcode.write(
            """
void main({fargs});
void main({fargs})
{{
""".format(fargs=', '.join(state_param + kernel_args_typed + extra_call_args_typed)), cfg, state_id,
            node)

        if is_persistent:
            self._localcode.write('''
int dace_number_SMs;
DACE_GPU_CHECK({backend}DeviceGetAttribute(&dace_number_SMs, {backend}DevAttrMultiProcessorCount, 0));
int dace_number_blocks = ((int) ceil({fraction} * dace_number_SMs)) * {occupancy};
                '''.format(fraction=Config.get('compiler', 'cuda', 'persistent_map_SM_fraction'),
                           occupancy=Config.get('compiler', 'cuda', 'persistent_map_occupancy'),
                           backend=self.backend))

        # if create_grid_barrier:
        #     gbar = '__gbar_' + kernel_name
        #     self._localcode.write('    cub::GridBarrierLifetime %s;\n' % gbar, cfg, state_id, node)
        #     self._localcode.write(
        #         '{}.Setup({});'.format(gbar,
        #                                ' * '.join(_topy(grid_dims)) if not is_persistent else 'dace_number_blocks'),
        #         cfg, state_id, node)
        #     extra_kernel_args.append('(void *)((cub::GridBarrier *)&%s)' % gbar)

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
        if max_streams >= 0 and hasattr(scope_entry, '_cuda_stream'):
            cudastream = '__state->gpu_context->streams[%d]' % scope_entry._cuda_stream
        else:
            cudastream = 'nullptr'

        # make sure dynamic map inputs are properly handled
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            self._localcode.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg, state_id, scope_entry)

        gdims = 'dace_number_blocks, 1, 1' if is_persistent else ', '.join(_topy(grid_dims))
        bdims = ', '.join(_topy(block_dims))

        # TODO: Read the arguments(_localcode)
        self._localcode.write(
            '''flex_barrier_xy_init();
                flex_global_barrier_xy();''')
        start_address = 0
        for k, v in prototype_kernel_args.items():
            self._localcode.write(f'{k} = ((uint32_t *)(hbm_addr({start_address})))[0];')
            start_address += 4
            #self._localcode.write(f'{k} = hbm_addr({k});')
            pass

        for k, v in hbm_interleaved_args.items():
            self._localcode.write(f'{k}_base = {k};')
        
        for aname, arg in hbm_interleaved_args.items():
            if aname in sdfg.arrays:
                nodedesc = sdfg.arrays[aname]
                if nodedesc.is_hbm_interleaved:
                    hbm_width = nodedesc.shape[1]
                    hbm_height = nodedesc.shape[0]
                    height_split = nodedesc.hbm_split_scheme[0]
                    width_split = nodedesc.hbm_split_scheme[1]
                    self._localcode.write(f'{aname}_height = {hbm_height};')
                    self._localcode.write(f'{aname}_width = {hbm_width};')
                    self._localcode.write(f'{aname}_tile_height = {hbm_height}/{height_split};')
                    self._localcode.write(f'{aname}_tile_width = {hbm_width}/{width_split};')


        # Just dump the whole HBM address space
        dump_str = "if (flex_is_dm_core() && (flex_get_cluster_id() == 0))\n{"
        for arr_name, arr in sdfg.arrays.items():
            if arr.transient is True:
                continue
            if arr.storage != dace.dtypes.StorageType.SoftHier_HBM:
                continue
            if arr_name != "B":
                continue
            dump_str += "flex_dump_open();\n"
            dump_str += f"flex_dump_hbm(A, A_tile_width * A_tile_height);\n"
            dump_str += "flex_dump_close();\n"
        dump_str += "}\n"
        dump_str += "flex_intra_cluster_sync();\n"
        # Prepare an empty-grid check for runtime grids
        dimcheck = True
        if dimcheck:
            emptygrid_warning = ''
            if Config.get('debugprint') == 'verbose' or Config.get_bool('compiler', 'cuda', 'syncdebug'):
                emptygrid_warning = (f'printf("Warning: Skipping launching kernel \\"{kernel_name}\\" '
                                     'due to an empty grid.\\n");')

            self._localcode.write(
            '''
            // if (flex_is_first_core() && (flex_get_cluster_id()==0))
            // {{
                // printf("A: %x\\n", A);
                // printf("B: %x\\n", B);
                // printf("C: %x\\n", C);
                // printf("K: %x\\n", K);
                // printf("M: %x\\n", M);
                // printf("N: %x\\n", N);
            // }}
            // if (flex_is_first_core() && (flex_get_cluster_id()==0))
            // {{
                // printf("%x\\n", ((uint32_t *)(hbm_addr(A)))[0]);
                // printf("%x\\n", ((uint32_t *)(hbm_addr(B)))[0]);
                // printf("%x\\n", ((uint32_t *)(hbm_addr(C)))[0]);
            // }}
            uint32_t eoc_val = 0;
            flex_global_barrier_xy();
            flex_timer_start();
            {kname}({kargs});
            flex_global_barrier_xy();
            flex_timer_end();
            flex_intra_cluster_sync();
            flex_global_barrier_xy();
            {dump_str}
            flex_global_barrier_xy();
            flex_eoc(eoc_val);
            return;'''
            .format(kname=kernel_name,
                    kargs=', '.join([arg for arg in prototype_kernel_args] + extra_kernel_args),
                    gdims=gdims,
                    bdims=bdims,
                    dynsmem=_topy(dynsmem_size),
                    stream=cudastream,
                    backend=self.backend,
                    dump_str=dump_str), cfg, state_id, scope_entry)
        # Check kernel launch for errors
        # self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});')
        
        self._emit_sync(self._localcode)

        # Close the runkernel function
        self._localcode.write('}')
        #######################
        # Add invocation to calling code (in another file)
        function_stream.write(
            '// DACE_EXPORTED void __dace_runkernel_%s(%s);\n' %
            (kernel_name, ', '.join(state_param + kernel_args_typed + extra_call_args_typed)), cfg, state_id,
            scope_entry)

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('{', cfg, state_id, scope_entry)

        # Synchronize all events leading to dynamic map range connectors
        for e in dace.sdfg.dynamic_map_inputs(state, scope_entry):
            if hasattr(e, '_cuda_event'):
                ev = e._cuda_event
                callsite_stream.write(
                    'DACE_GPU_CHECK({backend}EventSynchronize(__state->gpu_context->events[{ev}]));'.format(
                        ev=ev, backend=self.backend), cfg, state_id, [e.src, e.dst])
            callsite_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg, state_id, node)
        

        # TODO: Load the arguments(callsite_stream)
    #     callsite_stream.write('''
    #         size_t array_size = 8192;
    #         std::vector<uint32_t> array1 = {64, 64 + static_cast<uint32_t>(array_size * 2)};
    #         std::vector<uint32_t> array2;

    #         std::vector<float> array2_data(array_size, 0.5f);
    #         for (float val : array2_data) {
    #             uint32_t scaled_val = static_cast<uint32_t>(std::round(val * 65535));
    #             array2.push_back(scaled_val);
    #         }

    #         std::vector<std::vector<uint32_t>> arrays = {array1, array2};
    #         std::vector<std::string> dtypes = {"uint64", "uint64"};

    #         ELFGenerator generator;
    #         generator.generate("/scratch/dace4softhier/gvsoc/output.elf", arrays, dtypes);
    # ''', cfg, state_id, scope_entry)

        callsite_stream.write("printf(\"Start Running Kernel\");")
        # system("bash -c 'cd /scratch/dace4softhier/gvsoc && source sourceme.sh && ./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary ./sw_build/softhier.elf run --trace=/chip/cluster_0/redmule'");
        callsite_stream.write(
            'int result = system("cd ./.dacecache && ./dace.sh");'
            # 'int result = system("pwd");'
        )
        callsite_stream.write("printf(\"Result: %d\", result);")
        callsite_stream.write("printf(\"Finish Running Kernel\");")
        # Invoke kernel call
        callsite_stream.write(
            '// __dace_runkernel_%s(%s);\n' %
            (kernel_name,
             ', '.join(['__state'] + [cpp.ptr(aname, arg, sdfg, self._frame)
                                      for aname, arg in kernel_args.items()] + extra_call_args)), cfg, state_id,
            scope_entry)

        # If there are dynamic Map inputs, put the kernel invocation in its own scope to avoid redefinitions.
        if dace.sdfg.has_dynamic_map_inputs(state, scope_entry):
            callsite_stream.write('\\has_dynamic_map_inputs}', cfg, state_id, scope_entry)

        synchronize_streams(sdfg, cfg, state, state_id, scope_entry, scope_exit, callsite_stream, self)

        # Instrumentation (post-kernel)
        if instr is not None:
            callsite_stream.write(outer_stream.getvalue())
        
        self.tcdm_offset = 0
        # print("################################Finish Using SoftHierCodeGen Scope######################################")

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
                    print(f"Waring: No `gpu_block_size` property specified on map {kernelmap_entry.map.label}. ")
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

    def generate_kernel_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                              kernel_map: nodes.Map, kernel_name: str, grid_dims: list, block_dims: list,
                              has_tbmap: bool, has_dtbmap: bool, kernel_params: list, function_stream: CodeIOStream,
                              kernel_stream: CodeIOStream) -> None:
        node = dfg_scope.source_nodes()[0]

        # # Get the thread/block index type
        ttype = Config.get('compiler', 'cuda', 'thread_id_type')
        tidtype = getattr(dtypes, ttype, False)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        kernel_stream.write('{', cfg, state_id, node)
        kernel_stream.write("// TEST KERNEL SCOPE\n", cfg, state_id, node)
        # kernel_stream.write("flex_global_barrier_xy();\n", cfg, state_id, node)
        kernel_stream.write(f"const uint32_t cluster_id = flex_get_cluster_id();\n", cfg, state_id, node)
        # kernel_stream.write(f"uint32_t core_id = flex_get_core_id();\n", cfg, state_id, node)
        
        # for i in range(len(kernel_map.range)):
        #     varname = kernel_map.params[-i - 1]
        #     kernel_stream.write(f'int {varname} = 0;\n', cfg, state_id, node)
        #     self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')

        # Add more opening braces for scope exit to close
        for dim in range(len(node.map.range) - 1):
            kernel_stream.write('{', cfg, state_id, node)
        # Generate all index arguments for kernel grid
        krange = subsets.Range(kernel_map.range[::-1])
        kdims = krange.size()
        dsym = [symbolic.symbol('__DAPB%d' % i, nonnegative=True, integer=True) for i in range(len(krange))]
        bidx = krange.coord_at(dsym)
        entry_node = dfg_scope.source_nodes()[0]
        
        # print(f'krange: {krange}; kdims: {kdims}; dsym: {dsym}; bidx: {bidx}')
        for i, r in enumerate(node.map.range):
            var = kernel_map.params[i]
            begin, end, skip = r
            kernel_stream.write(
                "for (int %s = %s; %s < %s; %s += %s) {\n" %
                (var, cpp.sym2cpp(begin), var, cpp.sym2cpp(end + 1), var, cpp.sym2cpp(skip)),
                cfg,
                state_id,
                node,
            )

        # handle dynamic map inputs
        for e in dace.sdfg.dynamic_map_inputs(sdfg.states()[state_id], dfg_scope.source_nodes()[0]):
            kernel_stream.write(
                self._cpu_codegen.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]),
                cfg, state_id,
                dfg_scope.source_nodes()[0])

        # Dispatch internal code
        assert SoftHierCodeGen._in_device_code is False
        SoftHierCodeGen._in_device_code = True
        self._kernel_map = node
        self._kernel_state = sdfg.node(state_id)
        self._block_dims = block_dims
        self._grid_dims = grid_dims

        # Emit internal array allocation (deallocation handled at MapExit)
        self._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, kernel_stream)

        scope_entry = dfg_scope.source_nodes()[0]

        # kernel_stream.write("flex_global_barrier_xy();\n", cfg, state_id, node)
        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           kernel_stream,
                                           skip_entry_node=True)

        for i, r in enumerate(node.map.range):
            kernel_stream.write('}', cfg, state_id, node)
        self._block_dims = None
        self._kernel_map = None
        self._kernel_state = None
        SoftHierCodeGen._in_device_code = False
        self._grid_dims = None
        self.dynamic_tbmap_type = None

    def get_next_scope_entries(self, dfg, scope_entry):
        parent_scope_entry = dfg.entry_node(scope_entry)
        # We're in a nested SDFG, use full graph
        if parent_scope_entry is None:
            parent_scope = dfg
        else:
            parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all non-sequential scopes from the same level
        all_scopes = [
            node for node in parent_scope.bfs_nodes(scope_entry)
            if isinstance(node, nodes.EntryNode) and node.map.schedule != dtypes.ScheduleType.Sequential
        ]

        # TODO: Fix to include *next* scopes, without concurrent scopes

        return all_scopes[all_scopes.index(scope_entry) + 1:]

    def generate_devicelevel_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: StateSubgraphView,
                                   state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # Sanity check
        assert SoftHierCodeGen._in_device_code == True
        # print("################################Using SoftHierCodeGen Devicelevel Scope######################################")
        dfg = cfg.state(state_id)
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]
        scope_map = scope_entry.map
        next_scopes = self.get_next_scope_entries(dfg, scope_entry)
        if scope_map.schedule == dtypes.ScheduleType.SoftHier_Sequential:
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self
            self._cpu_codegen.is_soft_hier = True
            self._cpu_codegen.generate_scope(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
            self._cpu_codegen.is_soft_hier = False
            self._cpu_codegen.calling_codegen = old_codegen
            return
        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        callsite_stream.write('{', cfg, state_id, scope_entry)
        callsite_stream.write("// TEST DEVICE SCOPE\n")
        # Emit internal array allocation (deallocation handled at MapExit)

        self._frame.allocate_arrays_in_scope(sdfg, cfg, scope_entry, function_stream, callsite_stream)

        # Generate all index arguments for block
        if scope_map.schedule == dtypes.ScheduleType.SoftHier_Cluster:
            block_dims = self._block_dims
            brange = subsets.Range(scope_map.range[::-1])
            kdims = brange.size()
            minels = brange.min_element()
            maxels = brange.max_element()
            dsym = [
                symbolic.symbol('__DAPT%d' % i, nonnegative=True, integer=True) 
                for i in range(len(brange))
            ]
            dsym_end = [
                d + (bs * rng[2]) - 1 
                for d, bs, rng in zip(dsym, self._block_dims, brange)
            ]
            tidx = brange.coord_at(dsym)
            
            # First three dimensions are evaluated directly
            if len(brange) == 1:
                for i in range(min(len(brange), 3)):
                    varname = scope_map.params[-i - 1]
                    block_expr = 'cluster_id'
                    expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                    callsite_stream.write('const int %s = %s;' % (varname, expr), cfg, state_id, scope_entry)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')
                for i in range(min(len(brange), 3)):
                    varname = scope_map.params[-i - 1]
                    condition = ''
                    condition += '%s <= %s' % (varname, brange.max_element()[i])
                    if len(condition) > 0:
                        callsite_stream.write('if (%s) {' % condition, cfg, state_id, scope_entry)    
                    else:
                        callsite_stream.write('{', cfg, state_id, scope_entry)

            if len(brange) == 2:
                x_length = brange.max_element()[1] + 1
                y_length = brange.max_element()[0] + 1
                self._soft_hier_dims = [x_length, y_length]
                for i in range(min(len(brange), 3)):
                    varname = scope_map.params[i]
                    if i == 0:
                        block_expr = f"cluster_id % {x_length}"
                    else:
                        block_expr = f"cluster_id / {x_length}"
                    expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                    callsite_stream.write('int %s = %s;' % (varname, expr), cfg, state_id, scope_entry)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, 'int')
                for i in range(min(len(brange), 3)):
                    varname = scope_map.params[-i-1]
                    condition = ''
                    condition += '%s <= %s' % (varname, brange.max_element()[i])
                    if len(condition) > 0:
                        callsite_stream.write('if (%s) {' % condition, cfg, state_id, scope_entry)    
                    else:
                        callsite_stream.write('{', cfg, state_id, scope_entry)

            # Delinearize beyond the third dimension
            if len(brange) > 3:
                raise Exception("UWU, TODO, Ranges with more than 1 dimension")

            # Generate conditions for this block's execution using min and max
            # element, e.g. skipping out-of-bounds threads in trailing block
            
            callsite_stream.write(f'// Minels: {minels}, Maxels: {maxels}', cfg, state_id, scope_entry)
            # callsite_stream.write(f'// Configure RedMule Here\n')
            for node, parent in dfg_scope.all_nodes_recursive():
                if isinstance(node, nodes.Tasklet):
                    if node.name == 'mmad_redmule':
                        if node in parent.nodes():
                            # check every input memlets to get the shape
                            redmule_dims = []
                            for edge in parent.in_edges(node):
                                if edge.dst_conn == '_in_local_a':
                                    # add the shape of A
                                    redmule_dims.append(edge.data.subset.size())
                                elif edge.dst_conn == '_in_local_b':
                                    # add the shape of B
                                    redmule_dims.append(edge.data.subset.size())
                            if len(redmule_dims) == 2 and redmule_dims[0][-1] == redmule_dims[1][-2]:
                                # We have a matrix multiplication
                                print(f"RedMule Dims {redmule_dims}")
                                callsite_stream.write(f'// Configure RedMule Here\n')
                                callsite_stream.write(f'if(flex_is_first_core())')
                                callsite_stream.write('{', cfg, state_id, scope_entry)
                                callsite_stream.write(f'flex_redmule_config({redmule_dims[0][-2]}, {redmule_dims[0][-1]}, {redmule_dims[1][-1]});')
                                callsite_stream.write('}', cfg, state_id, scope_entry)
                                break
                            else:
                                raise Exception("RedMule only supports matrix multiplication")
                        else:
                            print(f"Tasklet {node} not found in subgraph")
            # callsite_stream.write(f"flex_intra_cluster_sync();\n", cfg, state_id, scope_entry)

        ##########################################################

        # Generate contents normally
        self._dispatcher.dispatch_subgraph(sdfg,
                                            cfg,
                                            dfg_scope,
                                            state_id,
                                            function_stream,
                                            callsite_stream,
                                            skip_entry_node=True)
        # callsite_stream.write("flex_global_barrier_xy();\n")
        callsite_stream.write("flex_intra_cluster_sync();\n")
        callsite_stream.write("// Finished deivelevel scope\n")
        self._soft_hier_dims = []


    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # print(f'SoftHier: Generating node {node} in state {state_id}')
        if self.node_dispatch_predicate(sdfg, dfg, node):
            # Dynamically obtain node generator according to class name
            gen = getattr(self, '_generate_' + type(node).__name__, False)
            if gen is not False:  # Not every node type has a code generator here
                gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
                return

        if not SoftHierCodeGen._in_device_code:
            self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            return
        if isinstance(node, nodes.ExitNode):
            self._locals.clear_scope(self._code_state.indentation + 1)

        if SoftHierCodeGen._in_device_code and isinstance(node, nodes.MapExit):
            return  # skip

        self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label):
        # return self._cpu_codegen.generate_nsdfg_header(
        #     sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=False)
        # TODO: Use a single method for GPU kernels, FPGA modules, and NSDFGs
        arguments = []
        state_struct=False
        if state_struct:
            toplevel_sdfg: SDFG = sdfg.cfg_list[0]
            arguments.append(f'{cpp.mangle_dace_state_struct_name(toplevel_sdfg)} *__state')

        # Add "__restrict__" keywords to arguments that do not alias with others in the context of this SDFG
        restrict_args = []
        write_type = 'const uint32_t'
        for atype, aname, _ in memlet_references:

            def make_restrict(expr: str) -> str:
                    return ''

            if aname in node.sdfg.arrays and not node.sdfg.arrays[aname].may_alias:
                restrict_args.append(make_restrict(atype))
            else:
                restrict_args.append('')

        arguments += [
            f'{write_type} {aname}' for (atype, aname, _), restrict in zip(memlet_references, restrict_args)
        ]
        fsyms = node.sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True)
        arguments += [
            f'{write_type} {aname}' for aname in sorted(node.symbol_mapping.keys())
            if aname in fsyms and aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        return f'inline int {sdfg_label}({arguments}) {{'

        

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(sdfg,
                                                     cfg,
                                                     state,
                                                     node,
                                                     memlet_references,
                                                     sdfg_label,
                                                     state_struct=False)

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
                # Connectors that are both input and output share the same name
        inout = set(node.in_connectors.keys() & node.out_connectors.keys())

        memlet_references = []
        for _, _, _, vconn, in_memlet in sorted(state.in_edges(node), key=lambda e: e.dst_conn or ''):
            if vconn in inout or in_memlet.data is None:
                continue
            memlet_references.append(
                cpp.emit_softhier_memlet_reference(self._dispatcher,
                                          sdfg,
                                          in_memlet,
                                          vconn,
                                          is_write=vconn in node.out_connectors,
                                          conntype=node.in_connectors[vconn],
                                          is_soft_hier=True))

        for _, uconn, _, _, out_memlet in sorted(state.out_edges(node), key=lambda e: e.src_conn or ''):
            if out_memlet.data is not None:
                memlet_references.append(
                    cpp.emit_softhier_memlet_reference(self._dispatcher,
                                              sdfg,
                                              out_memlet,
                                              uconn,
                                              conntype=node.out_connectors[uconn],
                                              is_soft_hier=True))

        result = memlet_references

        # Add data from nested SDFGs to kernel arguments
        result.extend([(atype, aname, aname) for atype, aname, _ in self.extra_nsdfg_args])
        for arg in self.extra_nsdfg_args:
            defined_type, ctype = self._dispatcher.defined_vars.get(arg[1], 1)
            self._dispatcher.defined_vars.add(arg[1], defined_type, ctype)

        return result

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: CodeIOStream,
                             callsite_stream: CodeIOStream) -> None:
        print("Generating NestedSDFG using SoftHierCodeGen")
        callsite_stream.write(f'// Nested SDFG {node.label} begin', cfg, state_id, node)
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule

    def _generate_MapExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.MapExit, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        if node.map.schedule == dtypes.ScheduleType.GPU_Device:
            # Remove grid invocation conditions
            for i in range(len(node.map.params)):
                if self._kernel_grid_conditions:
                    self._kernel_grid_conditions.pop()

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            # Close block invocation conditions
            for i in range(len(node.map.params)):
                callsite_stream.write('}', cfg, state_id, node)

        elif node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
            # Close lambda function
            callsite_stream.write('});', cfg, state_id, node)
            # Close block invocation
            callsite_stream.write('}', cfg, state_id, node)
            return

        self._cpu_codegen._generate_MapExit(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def _get_thread_id(self) -> str:
        result = 'threadIdx.x'
        if self._block_dims[1] != 1:
            result += f' + ({sym2cpp(self._block_dims[0])}) * threadIdx.y'
        if self._block_dims[2] != 1:
            result += f' + ({sym2cpp(self._block_dims[0] * self._block_dims[1])}) * threadIdx.z'
        return result

    def _get_warp_id(self) -> str:
        return f'(({self._get_thread_id()}) / warpSize)'

    def _get_block_id(self) -> str:
        result = 'blockIdx.x'
        if self._block_dims[1] != 1:
            result += f' + gridDim.x * blockIdx.y'
        if self._block_dims[2] != 1:
            result += f' + gridDim.x * gridDim.y * blockIdx.z'
        return result

    def _generate_condition_from_location(self, name: str, index_expr: str, node: nodes.Tasklet,
                                          callsite_stream: CodeIOStream) -> str:
        if name not in node.location:
            return 0

        location: Union[int, str, subsets.Range] = node.location[name]
        if isinstance(location, str) and ':' in location:
            location = subsets.Range.from_string(location)
        elif symbolic.issymbolic(location):
            location = sym2cpp(location)

        if isinstance(location, subsets.Range):
            # Range of indices
            if len(location) != 1:
                raise ValueError(f'Only one-dimensional ranges are allowed for {name} specialization, {location} given')
            begin, end, stride = location[0]
            rb, re, rs = sym2cpp(begin), sym2cpp(end), sym2cpp(stride)
            cond = ''
            cond += f'(({index_expr}) >= {rb}) && (({index_expr}) <= {re})'
            if stride != 1:
                cond += f' && ((({index_expr}) - {rb}) % {rs} == 0)'

            callsite_stream.write(f'if ({cond}) {{')
        else:
            # Single-element
            callsite_stream.write(f'if (({index_expr}) == {location}) {{')

        return 1

    def _generate_RedMule_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.Tasklet, function_stream: CodeIOStream, callsite_stream: CodeIOStream,
                        codegen=None):

        is_sync = False
        # Allow other code generators to call this with a callback
        codegen = codegen or self
        write_type = 'uint32_t'
        outer_stream_begin = CodeIOStream()
        outer_stream_end = CodeIOStream()
        inner_stream = CodeIOStream()

        # Add code to init and exit functions
        self._frame._initcode.write(codeblock_to_cpp(node.code_init), sdfg)
        self._frame._exitcode.write(codeblock_to_cpp(node.code_exit), sdfg)

        state_dfg: SDFGState = cfg.nodes()[state_id]

        # Free tasklets need to be presynchronized (e.g., CPU tasklet after
        # GPU->CPU copy)
        if state_dfg.entry_node(node) is None:
            cpp.presynchronize_streams(sdfg, cfg, state_dfg, state_id, node, callsite_stream)

        # Prepare preamble and code for after memlets
        after_memlets_stream = CodeIOStream()
        # codegen.generate_tasklet_preamble(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream,
        #                                 after_memlets_stream)

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in state_dfg.in_edges(node):
            # print("Edge: ", edge)
            u = edge.src
            memlet = edge.data
            src_node = state_dfg.memlet_path(edge)[0].src
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            # print(f"Src Node: {src_node}")
            # print(f"Dst Node: {dst_node}")
            # if there is a memlet path for src_node, then is_sync = True
            if len(state_dfg.in_edges(src_node)) > 0:
                is_sync = True
                for e in state_dfg.in_edges(src_node):
                    if isinstance(e.src, nodes.AccessNode):
                        desc = sdfg.arrays[e.src.data]
                        if isinstance(desc, dace.data.Stream):
                            is_sync = False
                            break
                        
                
            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError("Duplicates found in memlets")
                ctype = node.in_connectors[edge.dst_conn].ctype
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data
                    if not shared_data_name:
                        # Very unique name. TODO: Make more intuitive
                        shared_data_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(src_node),
                                                                    dfg.node_id(node), edge.src_conn)

                    # Read variable from shared storage
                    defined_type, _ = self._dispatcher.defined_vars.get(shared_data_name)
                    if defined_type in (DefinedType.Scalar, DefinedType.Pointer):
                        assign_str = (f"const {ctype} {edge.dst_conn} = {shared_data_name};")
                    else:
                        assign_str = (f"const {ctype} &{edge.dst_conn} = {shared_data_name};")
                    inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn, defined_type, f"const {ctype}")

                else:
                    u, uconn, v, vconn, memlet = edge
                    desc = sdfg.arrays[memlet.data]
                    # print(f"u: {u}, uconn: {uconn}, v: {v}, vconn: {vconn}, memlet: {memlet}")
                    # Obtain copy information
                    if isinstance(edge.src, nodes.AccessNode):
                        src_expr = cpp.copy_expr(self._dispatcher, sdfg, src_node.data, memlet, False)
                        if src_expr != edge.src.data:
                            src_expr = f'{src_expr} * {desc.dtype.bytes}'
                        assign_str = (f"{write_type} {edge.dst_conn} = {src_expr};")
                        inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                        self._dispatcher.defined_vars.add(edge.dst_conn, DefinedType.Pointer, ctype)
                    else:
                        # Find the source Access Node
                        desc = sdfg.arrays[memlet.data]  
                        src_subset = memlet.get_src_subset(edge, state_dfg)
                        is_src_write = not memlet._is_data_src
                        src_expr = cpp.copy_expr(self._dispatcher,
                             sdfg,
                             src_node.data,
                             memlet,
                             is_write=is_src_write,
                             offset=src_subset,
                             relative_offset=False,
                             packed_types=self._cpu_codegen._packed_types)
                        assign_str = (f"{write_type} {edge.dst_conn} = {src_expr};")
                        inner_stream.write(
                            assign_str,
                            cfg,
                            state_id,
                            [src_node, dst_node],
                        )
                        self._dispatcher.defined_vars.add(edge.dst_conn, DefinedType.Pointer, ctype)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1, ctype)
                arrays.add(edge.dst_conn)
        # print(f"Is Sync: {is_sync}")
        # Use outgoing edges to preallocate output local vars
        # in two stages: first we preallocate for data<->code cases,
        # followed by code<->code
        tasklet_out_connectors = set()
        for edge in state_dfg.out_edges(node):
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                # Handling this in a separate pass just below
                continue

            if edge.src_conn:
                if edge.src_conn in tasklet_out_connectors:  # Disallow duplicates
                    continue

                # self._dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, cfg, dfg, state_id,
                #                                             function_stream, inner_stream)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, node.out_connectors[edge.src_conn].ctype)
                tasklet_out_connectors.add(edge.src_conn)

        for edge in state_dfg.out_edges(node):
            # Special case: code->code
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if edge.src_conn is None:
                continue
            cdtype = node.out_connectors[edge.src_conn]
            ctype = cdtype.ctype
            # Convert dtype to data descriptor
            if isinstance(cdtype, dtypes.pointer):
                arg_type = dt.Array(cdtype._typeclass, [1])
            else:
                arg_type = dt.Scalar(cdtype)

            if (isinstance(dst_node, nodes.CodeNode) and edge.src_conn not in tasklet_out_connectors):
                memlet = edge.data

                # Generate register definitions for inter-tasklet memlets
                local_name = edge.data.data
                if not local_name:
                    # Very unique name. TODO: Make more intuitive
                    local_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(node),
                                                            dfg.node_id(dst_node), edge.src_conn)

                # Allocate variable type
                code = "%s %s;" % (write_type, local_name)
                outer_stream_begin.write(code, cfg, state_id, [edge.src, dst_node])
                if (isinstance(arg_type, dt.Scalar) or isinstance(arg_type, dtypes.typeclass)):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype, ancestor=1)
                elif isinstance(arg_type, dt.Array):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Pointer, ctype, ancestor=1)
                elif isinstance(arg_type, dt.Stream):
                    if arg_type.is_stream_array():
                        self._dispatcher.defined_vars.add(local_name, DefinedType.StreamArray, ctype, ancestor=1)
                    else:
                        self._dispatcher.defined_vars.add(local_name, DefinedType.Stream, ctype, ancestor=1)
                else:
                    raise TypeError("Unrecognized argument type: {}".format(type(arg_type).__name__))

                inner_stream.write("%s %s;" % (write_type, edge.src_conn), cfg, state_id, [edge.src, edge.dst])
                tasklet_out_connectors.add(edge.src_conn)
                self._dispatcher.defined_vars.add(edge.src_conn, DefinedType.Scalar, ctype)
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, ctype)
                locals_defined = True

        # Emit post-memlet tasklet preamble code
        callsite_stream.write(after_memlets_stream.getvalue())

        # Instrumentation: Pre-tasklet
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, state_dfg, node, outer_stream_begin, inner_stream, function_stream)

        inner_stream.write("\n    ///////////////////\n", cfg, state_id, node)
        self._cpu_codegen.unparse_tasklet(sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, self._locals,
                                self._ldepth, self._toplevel_schedule)
        inner_stream.write("flex_redmule_wait();\n", cfg, state_id, node)
        inner_stream.write("    ///////////////////\n\n", cfg, state_id, node)

        # Generate pre-memlet tasklet postamble
        after_memlets_stream = CodeIOStream()


        # Instrumentation: Post-tasklet
        if instr is not None:
            instr.on_node_end(sdfg, state_dfg, node, outer_stream_end, inner_stream, function_stream)
        if is_sync:
            callsite_stream.write("flex_intra_cluster_sync();", cfg, state_id, node)
        callsite_stream.write(outer_stream_begin.getvalue(), cfg, state_id, node)
        callsite_stream.write("if (flex_is_first_core())", cfg, state_id, node)
        callsite_stream.write('{', cfg, state_id, node)
        callsite_stream.write(inner_stream.getvalue(), cfg, state_id, node)
        callsite_stream.write(after_memlets_stream.getvalue())
        callsite_stream.write('}', cfg, state_id, node)
        if is_sync:
            callsite_stream.write("flex_intra_cluster_sync();", cfg, state_id, node)
        callsite_stream.write(outer_stream_end.getvalue(), cfg, state_id, node)

        # self._locals.clear_scope(self._ldepth + 1)
        self._dispatcher.defined_vars.exit_scope(node)


    
    
    def _generate_SoftHier_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.Tasklet, function_stream: CodeIOStream, callsite_stream: CodeIOStream,
                        codegen=None):

 # Allow other code generators to call this with a callback
        codegen = codegen or self
        write_type = 'uint64_t'
        outer_stream_begin = CodeIOStream()
        outer_stream_end = CodeIOStream()
        inner_stream = CodeIOStream()

        # Add code to init and exit functions
        self._frame._initcode.write(codeblock_to_cpp(node.code_init), sdfg)
        self._frame._exitcode.write(codeblock_to_cpp(node.code_exit), sdfg)

        state_dfg: SDFGState = cfg.nodes()[state_id]

        # Free tasklets need to be presynchronized (e.g., CPU tasklet after
        # GPU->CPU copy)
        if state_dfg.entry_node(node) is None:
            cpp.presynchronize_streams(sdfg, cfg, state_dfg, state_id, node, callsite_stream)

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in state_dfg.in_edges(node):
            u = edge.src
            memlet = edge.data
            src_node = state_dfg.memlet_path(edge)[0].src

            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError("Duplicates found in memlets")
                ctype = node.in_connectors[edge.dst_conn].ctype
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data
                    if not shared_data_name:
                        # Very unique name. TODO: Make more intuitive
                        shared_data_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(src_node),
                                                                      dfg.node_id(node), edge.src_conn)

                    # Read variable from shared storage
                    defined_type, _ = self._dispatcher.defined_vars.get(shared_data_name)
                    if defined_type in (DefinedType.Scalar, DefinedType.Pointer):
                        assign_str = (f"const {ctype} {edge.dst_conn} = {shared_data_name};")
                    else:
                        assign_str = (f"const {ctype} &{edge.dst_conn} = {shared_data_name};")
                    inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn, defined_type, f"const {ctype}")

                else:
                    u, uconn, v, vconn, memlet = edge
                    desc = sdfg.arrays[memlet.data]
                    # print(f"u: {u}, uconn: {uconn}, v: {v}, vconn: {vconn}, memlet: {memlet}")
                    # Obtain copy information
                    if isinstance(edge.src, nodes.AccessNode):
                        src_expr = cpp.copy_expr(self._dispatcher, sdfg, src_node.data, memlet, False)
                        if src_expr != edge.src.data:
                            src_expr = f'{src_expr} * {desc.dtype.bytes}'
                        assign_str = (f"{write_type} {edge.dst_conn} = {src_expr};")
                        inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                        self._dispatcher.defined_vars.add(edge.dst_conn, DefinedType.Pointer, ctype)
                    else:
                        # Find the source Access Node
                        desc = sdfg.arrays[memlet.data]  
                        src_subset = memlet.get_src_subset(edge, state_dfg)
                        is_src_write = not memlet._is_data_src
                        src_expr = cpp.copy_expr(self._dispatcher,
                             sdfg,
                             src_node.data,
                             memlet,
                             is_write=is_src_write,
                             offset=src_subset,
                             relative_offset=False,
                             packed_types=self._cpu_codegen._packed_types)
                        assign_str = (f"{write_type} {edge.dst_conn} = {src_expr};")
                        inner_stream.write(
                            assign_str,
                            cfg,
                            state_id,
                            [src_node, dst_node],
                        )
                        self._dispatcher.defined_vars.add(edge.dst_conn, DefinedType.Pointer, ctype)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1, ctype)
                arrays.add(edge.dst_conn)

        # Use outgoing edges to preallocate output local vars
        # in two stages: first we preallocate for data<->code cases,
        # followed by code<->code
        tasklet_out_connectors = set()
        for edge in state_dfg.out_edges(node):
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                # Handling this in a separate pass just below
                continue

            if edge.src_conn:
                if edge.src_conn in tasklet_out_connectors:  # Disallow duplicates
                    continue
                
                output_definition_stream = CodeIOStream()
                self._dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, cfg, dfg, state_id,
                                                            function_stream, output_definition_stream)
                # Replace strings in output_definition_stream start with dace:: with uint64_t
                import re

                s = output_definition_stream.getvalue()
                s = re.sub(r'\bdace::[A-Za-z0-9_]+(?:\s*\*+)?', 'uint64_t', s)
                output_definition_stream = s
                
                # inner_stream.write(output_definition_stream, cfg, state_id, node)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, node.out_connectors[edge.src_conn].ctype)
                tasklet_out_connectors.add(edge.src_conn)

        for edge in state_dfg.out_edges(node):
            # Special case: code->code
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if edge.src_conn is None:
                continue
            cdtype = node.out_connectors[edge.src_conn]
            ctype = cdtype.ctype
            # Convert dtype to data descriptor
            if isinstance(cdtype, dtypes.pointer):
                arg_type = dt.Array(cdtype._typeclass, [1])
            else:
                arg_type = dt.Scalar(cdtype)

            if (isinstance(dst_node, nodes.CodeNode) and edge.src_conn not in tasklet_out_connectors):
                memlet = edge.data

                # Generate register definitions for inter-tasklet memlets
                local_name = edge.data.data
                if not local_name:
                    # Very unique name. TODO: Make more intuitive
                    local_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(node),
                                                            dfg.node_id(dst_node), edge.src_conn)

                # Allocate variable type
                code = "%s %s;" % (ctype, local_name)
                outer_stream_begin.write(code, cfg, state_id, [edge.src, dst_node])
                if (isinstance(arg_type, dt.Scalar) or isinstance(arg_type, dtypes.typeclass)):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype, ancestor=1)
                elif isinstance(arg_type, dt.Array):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Pointer, ctype, ancestor=1)
                elif isinstance(arg_type, dt.Stream):
                    if arg_type.is_stream_array():
                        self._dispatcher.defined_vars.add(local_name, DefinedType.StreamArray, ctype, ancestor=1)
                    else:
                        self._dispatcher.defined_vars.add(local_name, DefinedType.Stream, ctype, ancestor=1)
                else:
                    raise TypeError("Unrecognized argument type: {}".format(type(arg_type).__name__))

                inner_stream.write("%s %s;" % ("uint64_t", edge.src_conn), cfg, state_id, [edge.src, edge.dst])
                tasklet_out_connectors.add(edge.src_conn)
                self._dispatcher.defined_vars.add(edge.src_conn, DefinedType.Scalar, ctype)
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, ctype)
                locals_defined = True

        # Instrumentation: Pre-tasklet
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, cfg, state_dfg, node, outer_stream_begin, inner_stream, function_stream)

        inner_stream.write("\n    ///////////////////\n", cfg, state_id, node)

        self._cpu_codegen.unparse_tasklet(sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, self._locals,
                                self._ldepth, self._toplevel_schedule)

        inner_stream.write("    ///////////////////\n\n", cfg, state_id, node)

        # Generate pre-memlet tasklet postamble
        after_memlets_stream = CodeIOStream()
        self._cpu_codegen.generate_tasklet_postamble(sdfg, cfg, dfg, state_id, node, function_stream, inner_stream,
                                           after_memlets_stream)

        # Process outgoing memlets
        self.process_out_memlets(
            sdfg,
            cfg,
            state_id,
            node,
            dfg,
            self._dispatcher,
            inner_stream,
            True,
            function_stream,
        )

        # Instrumentation: Post-tasklet
        if instr is not None:
            instr.on_node_end(sdfg, cfg, state_dfg, node, outer_stream_end, inner_stream, function_stream)

        callsite_stream.write(outer_stream_begin.getvalue(), cfg, state_id, node)
        callsite_stream.write('{', cfg, state_id, node)
        callsite_stream.write(inner_stream.getvalue(), cfg, state_id, node)
        callsite_stream.write(after_memlets_stream.getvalue())
        callsite_stream.write('}', cfg, state_id, node)
        callsite_stream.write(outer_stream_end.getvalue(), cfg, state_id, node)

        self._locals.clear_scope(self._ldepth + 1)
        self._dispatcher.defined_vars.exit_scope(node)

    
    def _generate_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.Tasklet, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        generated_preamble_scopes = 0
        if self._in_device_code:
            # If location dictionary prescribes that the code should run on a certain group of threads/blocks,
            # add condition
            generated_preamble_scopes += self._generate_condition_from_location('gpu_thread', self._get_thread_id(),
                                                                                node, callsite_stream)
            generated_preamble_scopes += self._generate_condition_from_location('gpu_warp', self._get_warp_id(), node,
                                                                                callsite_stream)
            generated_preamble_scopes += self._generate_condition_from_location('gpu_block', self._get_block_id(), node,
                                                                                callsite_stream)
        # print(f'generated_preamble_scopes: {generated_preamble_scopes}')
        # Call standard tasklet generation
        if node.name == "mmad_redmule":
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self
            self._generate_RedMule_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            self._cpu_codegen.calling_codegen = old_codegen
        elif node.name == "split_K_reduction":
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self
            self._generate_SoftHier_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            self._cpu_codegen.calling_codegen = old_codegen
        else:
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self
            self._cpu_codegen._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            self._cpu_codegen.calling_codegen = old_codegen

        if generated_preamble_scopes > 0:
            # Generate appropriate postamble
            for i in range(generated_preamble_scopes):
                callsite_stream.write('}', cfg, state_id, node)

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
