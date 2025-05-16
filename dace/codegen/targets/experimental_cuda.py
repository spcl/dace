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
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
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




# TODO: GENERAL, discuss with Yakup
# 1. Approval of dtypes


# TODO: I am not handling map with strided rights now,
# why? because This is handled somewhere else than in the scope



# My personal TODO's
# TODO: when tired
# include constant expressions + launch bounds logic
# insert warnings that gpu device must be first
# 4 dimensional example

# TODO: depending on what happens next
# change in_device_code to maybe in_kernel_code?




# TODO : I got rid of ScheduleType.GPU_Persistent (not supported anymore). If this codeBase 
# actually replaces the old one, this should be defined in dtypes.py and also accessed from 
# there. Also change GPU_SCHEDULES accesses to dtypes.GPU_SCHEDULES 
GPU_SCHEDULES = [
    dace.ScheduleType.GPU_Device,
    dace.ScheduleType.GPU_ThreadBlock,
    dace.ScheduleType.GPU_Warp
]


THREADS_PER_WARP = 32

@registry.autoregister_params(name='experimental_cuda')
class ExperimentalCUDACodeGen(TargetCodeGenerator):
    """ Experimental CUDA code generator."""
    target_name = 'experimental_cuda'
    title = 'CUDA'

    _in_device_code = False

    ######################## Initilization and Preprocessing related start #########################################################

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):

        self._frame: DaCeCodeGenerator = frame_codegen # creates the frame code, orchestrates the code generation for targets
        self._dispatcher: TargetDispatcher= frame_codegen.dispatcher # responsible for dispatching code generation to the appropriate target

        # dispatcher = self._dispatcher

        self.create_grid_barrier: bool = False # Used for grid level synchronization

        self.dynamic_tbmap_type = None
        self.extra_nsdfg_args = []
        ExperimentalCUDACodeGen._in_device_code = False  # TODO: Isn't this double?
        self._cpu_codegen: Optional['CPUCodeGen'] = None
        self._block_dims = None
        self._grid_dims = None

        # NOTE: Type may be wrong!
        self._kernel_map: Optional[nodes.MapEntry] = None  # Indicates whether the code generation is currently within a "kernel" map.

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

        self._kernel_state = None
        self._kernel_grid_conditions: List[str] = []
        self._scope_has_collaborative_copy = False

        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()

        # TODO: init and exitcode seem to serve no purpose actually.
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


        self._ignore_warnings = True

        # INFO: 
        # Register GPU schedules and storage types for ExperimentalCUDACodeGen.
        # The dispatcher maps GPU-related schedules and storage types to the
        # appropriate code generation functions in this code generator.

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_map_dispatcher(GPU_SCHEDULES, self)
        self._dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned]

        self._dispatcher.register_array_dispatcher(gpu_storage, self)
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CPU_Pinned, self)
        for storage in gpu_storage:
            for other_storage in dtypes.StorageType:
                self._dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self._dispatcher.register_copy_dispatcher(other_storage, storage, None, self)

        
        # NOTE: Moved it here from preprocessing, I think it fits better
        self._backend = common.get_gpu_backend() 
        self._language = 'cu' if self.backend == 'cuda' else 'cpp'
        target_type = "" if self.backend == 'cuda' else self.backend
        self._codeobject= CodeObject(sdfg.name + '_' + 'cuda',
                                      '',
                                      self._language,
                                      ExperimentalCUDACodeGen,
                                      'CUDA',
                                      target_type=target_type)


        # NOTE: 
        # "Register illegal copies" code NOT copied from cuda.py
        # Behavior unclear for me yet.


        ################## New variables ##########################
        self._current_kernel_spec: Optional[KernelSpec] = None


    # NOTE: I think this is good as is
    def preprocess(self, sdfg: SDFG) -> None:

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
                    and node.map.schedule == dtypes.ScheduleType.GPU_Device): # NOTE: Removed dtypes.ScheduleType.GPU_Persistent comparision
                if state.parent not in shared_transients:
                    shared_transients[state.parent] = state.parent.shared_transients()
                self._arglists[node] = state.scope_subgraph(node).arglist(defined_syms, shared_transients[state.parent])


    # NOTE: Used during preprocess. Seems good as is
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


    # NOTE: Used during preprocess. Seems good as is
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

        for state in sdfg.states():
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
        for i, state in enumerate(sdfg.states()):
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

    ######################## Initilization and Preprocessing related end #########################################################

    @property
    def has_initializer(self) -> bool:
        return True
    @property
    def has_finalizer(self) -> bool:
        return True






    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        
        # Are we generating host (launch) code or device (kernel) code?
        if not ExperimentalCUDACodeGen._in_device_code:

            # Prepare and cache kernel metadata (name, grid dims, arguments, etc.)
            self._current_kernel_spec = KernelSpec(
                cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id
            )
            
        
            self._generate_gpu_bridge(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)

            #--------------- Generate Kernel Function ----------------

            ExperimentalCUDACodeGen._in_device_code = True
            kernel_stream = CodeIOStream()

            kernel_name = self._current_kernel_spec.kernel_name
            kernel_args = self._current_kernel_spec.args_typed
            block_dims = self._current_kernel_spec.block_dims
            node = dfg_scope.source_nodes()[0]

            # Conditionally add __launch_bounds__ for block size optimization.
            launch_bounds = ''
            if node.gpu_launch_bounds != '-1':
                if node.gpu_launch_bounds == "0":
                    if not any(symbolic.issymbolic(b) for b in block_dims):
                        launch_bounds = f'__launch_bounds__({prod(block_dims)})'
                else:
                    launch_bounds = f'__launch_bounds__({node.gpu_launch_bounds})'


            # Emit kernel function signature
            kernel_stream.write(
                f'__global__ void {launch_bounds} {kernel_name}({", ".join(kernel_args)}) ',
                cfg, state_id, node
            )

            # generate kernel scope
            self._generate_kernel_scope(
                sdfg, cfg, dfg_scope, state_id, self._globalcode, kernel_stream
            )

            self._localcode.write(kernel_stream.getvalue() + '\n')
            ExperimentalCUDACodeGen._in_device_code = False 
            # --------------------------------------------------------------

            # Generate the actual launch call (host-side)
            self._generate_kernel_launch(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)


        else:
            # Nested scope: already inside a GPU kernel
            node = dfg_scope.source_nodes()[0]
            schedule_type = node.map.schedule.name

            if schedule_type == dace.ScheduleType.GPU_Device:
                raise NotImplementedError(
                    "Dynamic parallelism (nested GPU_Device schedules) is not supported."
                )

            gen = getattr(self, f'_generate_{schedule_type}_scope', None)
            if gen:
                gen(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream)
            else:
                raise NotImplementedError(
                    f"Scope generation for schedule type '{schedule_type}' is not implemented in ExperimentalCUDACodeGen. "
                    "Please check for supported schedule types or implement the corresponding generator."
                )

        
####################### helper functions to generate_scope ######################################


    def _generate_kernel_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                              function_stream: CodeIOStream, kernel_stream: CodeIOStream) -> None:
        
        
        # Get the Map Node (sould be a Map Node?)
        node = dfg_scope.source_nodes()[0]

        # Get kernel specifications
        kernel_spec = self._current_kernel_spec
        kernel_map = kernel_spec.kernel_map
        has_tbmap = kernel_spec.has_tbmap
        block_dims = kernel_spec.block_dims


        
        with KernelScopeManager(cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id,
                                function_stream=function_stream, callsite_stream=kernel_stream, comment="Kernel scope",) as scopeManager:

            # Get the thread/block index type
            ttype = Config.get('compiler', 'cuda', 'thread_id_type')
            tidtype = getattr(dtypes, ttype, False)
            if not isinstance(tidtype, dtypes.typeclass):
                raise ValueError(f'Configured type "{ttype}" for ``thread_id_type`` does not match any DaCe data type. '
                                'See ``dace.dtypes`` for available types (for example ``int32``).')
            

            # Generate all index arguments for kernel grid
            krange = subsets.Range(kernel_map.range[::-1])
            kdims = krange.size()
            dsym = [symbolic.symbol(f'__DAPB{i}', nonnegative=True, integer=True) for i in range(len(krange))]
            bidx = krange.coord_at(dsym)


            # First three dimensions are evaluated directly
            for i in range(min(len(krange), 3)):
                varname = kernel_map.params[-i - 1]

                # If we defaulted to a fixed number of threads per block, offset by thread ID
                block_expr = f'blockIdx.{_get_cuda_dim(min(i, 2))}'
                if not has_tbmap:
                    block_expr = f'({block_expr} * {symbolic_to_cpp(block_dims[i])} + threadIdx.{_get_cuda_dim(i)})'

                # Delinearize third dimension if necessary
                if i == 2 and len(krange) > 3:
                    block_expr = f'({block_expr} / ({symbolic_to_cpp(functools.reduce(sympy.Mul, kdims[3:], 1))}))'

                expr = symbolic_to_cpp(bidx[i]).replace(f'__DAPB{i}', block_expr)

                kernel_stream.write(f'{tidtype.ctype} {varname} = {expr};', cfg, state_id, node)
                self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, tidtype.ctype)


            # Delinearize beyond the third dimension
            if len(krange) > 3:
                for i in range(3, len(krange)):
                    varname = kernel_map.params[-i - 1]

                    block_expr = 'blockIdx.z'
                    if not has_tbmap:
                        block_expr = f'({block_expr} * {symbolic_to_cpp(block_dims[2])} + threadIdx.z)'

                    block_expr = '((%s / (%s)) %% (%s))' % (
                        block_expr,
                        symbolic_to_cpp(functools.reduce(sympy.Mul, kdims[i + 1:], 1)),
                        symbolic_to_cpp(kdims[i]),
                    )

                    expr = symbolic_to_cpp(bidx[i]).replace(f'__DAPB{i}', block_expr)
                    kernel_stream.write(f'{tidtype.ctype} {varname} = {expr};', cfg, state_id, node)
                    self._dispatcher.defined_vars.add(varname, DefinedType.Scalar, tidtype.ctype)


                
            # handle conditions 
            if not has_tbmap:
                dsym_end = [d + bs - 1 for d, bs in zip(dsym, block_dims)]
                minels = krange.min_element()
                maxels = krange.max_element()
                for i, (v, minel, maxel) in enumerate(zip(kernel_map.params[::-1], minels, maxels)):
                    condition = ''

                    # Optimize conditions if they are always true
                    if i >= 3 or (dsym[i] >= minel) != True:
                        condition += f'{v} >= {symbolic_to_cpp(minel)}' 

                    if (i >= 3 or ((dsym_end[i] < maxel) != False and ((dsym_end[i] % block_dims[i]) != 0) == True)
                        or (block_dims[i] > maxel) == True):

                        if len(condition) > 0:
                            condition += ' && '
                        condition += f'{v} < {symbolic_to_cpp(maxel + 1)}'

                    if len(condition) > 0:
                        scopeManager.open(condition= condition)




            self._dispatcher.dispatch_subgraph(sdfg, cfg, dfg_scope, state_id, function_stream, 
                                               kernel_stream, skip_entry_node=True)
        
            


    def _generate_GPU_ThreadBlock_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                                        state_id: int, function_stream: CodeIOStream, kernel_stream: CodeIOStream) -> None:


        # NOTE: not my code, but my insights. Approval for commenting this needed
        with KernelScopeManager(cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id,
                                function_stream=function_stream, callsite_stream=kernel_stream, comment="ThreadBlock Scope",) as scopeManager:
            
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

            for dim in range(map_dimensions):

                var_name = scope_map.params[-dim - 1] # also reverse it here!

                if dim < 3:
                    # First three dimensions: direct mapping or partial delinearization
                    if dim == 2 and map_dimensions > 3:
                        tail_prod = prod(map_dim_sizes[3:])
                        base_expr = f"(threadIdx.z / ({symbolic_to_cpp(tail_prod)}))"
                    else:
                        base_expr = f"threadIdx.{_get_cuda_dim(dim)}"
                else:
                    # Dimensions beyond the third: full delinearization
                    tail_prod = prod(map_dim_sizes[dim + 1:])
                    base_expr = (f"(threadIdx.z / ({symbolic_to_cpp(tail_prod)})) % "f"({symbolic_to_cpp(map_dim_sizes[dim])})")


                var_def = symbolic_to_cpp(symbolic_coordinates[dim]).replace(f'__SYM_IDX{dim}', base_expr)
                kernel_stream.write(f'int {var_name} = {var_def};', cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, 'int')

            

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
                    scopeManager.open(condition=condition)


            # ----------------- Dispatch Subgraph code generation -----------------------

            self._dispatcher.dispatch_subgraph(sdfg, cfg, dfg_scope, state_id, function_stream,
                            kernel_stream, skip_entry_node=True)




    def _generate_GPU_Warp_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                                function_stream: CodeIOStream, kernel_stream: CodeIOStream) -> None:


        with KernelScopeManager(cudaCodeGen=self, sdfg=sdfg, cfg=cfg, dfg_scope=dfg_scope, state_id=state_id, 
                                function_stream=function_stream, callsite_stream=kernel_stream, comment="WarpLevel Scope") as scopeManager:



            block_dims = self._current_kernel_spec.block_dims

            state_dfg = cfg.state(state_id)
            node = dfg_scope.source_nodes()[0]
            scope_map = node.map

            map_range = subsets.Range(scope_map.range[::-1])  # Reversed for potential better performance
            warp_dim = len(map_range)
            
            # The following sizes and bounds are be symbolic
            num_threads_in_block = prod(block_dims) 
            warp_dim_bounds = [max_elem + 1 for max_elem in map_range.max_element()]
            num_warps = prod(warp_dim_bounds)



            # ----------------- Guard checks -----------------------

            
            # handles checks either at compile time or runtime (i.e. checks in the generated code)
            self._hanlde_GPU_Warp_scope_guards(state_dfg, node, map_range, warp_dim, num_threads_in_block, num_warps,
                                               kernel_stream, scopeManager)
                        


            # ----------------- Define (flat) Thread ID within Block -----------------------

            flattened_terms = []

            for i, dim_size in enumerate(block_dims):

                if dim_size == 1:
                    continue

                dim = _get_cuda_dim(i)
                stride = [f"{block_dims[j]}" for j in range(i) if block_dims[j] > 1]
                idx_expr = " * ".join(stride + [f"threadIdx.{_get_cuda_dim(i)}"]) if stride else f"threadIdx.{dim}"
                flattened_terms.append(idx_expr)


            joined_terms = " + ".join(flattened_terms)
            flat_thread_idx_expr = f"({joined_terms})" if len(flattened_terms) > 1 else joined_terms
            # NOTE: name too ugly? How shorter but still unique ?
            threadID_name = 'ThreadId_%s_%d_%d_%d' % (scope_map.label, cfg.cfg_id, state_dfg.block_id, state_dfg.node_id(node))

            kernel_stream.write(f"int {threadID_name} = ({flat_thread_idx_expr}) / {THREADS_PER_WARP};", cfg, state_id, node)
            self._dispatcher.defined_vars.add(threadID_name, DefinedType.Scalar, 'int')


            
            # ----------------- Compute Map indices (= Warp indices) -----------------------

            for i in range(warp_dim):
                var_name = scope_map.params[-i - 1]  # reverse order
                previous_sizes = warp_dim_bounds[:i]

                if len(previous_sizes) > 0:
                    divisor = prod(previous_sizes)
                    expr = f"({threadID_name} / {divisor}) % {warp_dim_bounds[i]}"
                else:
                    expr = f"{threadID_name} % {warp_dim_bounds[i]}"

                kernel_stream.write(f"int {var_name} = {expr};", cfg, state_id, node)
                self._dispatcher.defined_vars.add(var_name, DefinedType.Scalar, 'int')



            # ----------------- Guard Conditions for Warp Execution -----------------------


            if num_warps * THREADS_PER_WARP != num_threads_in_block:
                condition = f'{threadID_name} < {num_warps}'
                scopeManager.open(condition)

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
                    scopeManager.open(condition)


            # ----------------- Dispatch Subgraph code generation -----------------------


            self._dispatcher.dispatch_subgraph(
                sdfg, cfg, dfg_scope, state_id, function_stream,
                kernel_stream, skip_entry_node=True
            )




    def _hanlde_GPU_Warp_scope_guards(self, state_dfg: SDFGState, node: nodes.MapEntry, map_range: subsets.Range,
                                       warp_dim: int, num_threads_in_block, num_warps, kernel_stream: CodeIOStream,
                                       scopeManager: 'KernelScopeManager'):
        
            #TODO: Move them to sdfg validation as well if possible
            
            #TODO: rename xfh, to cryptic
            parent_map, _ = xfh.get_parent_map(state_dfg, node)
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
                    f"{num_threads_in_block} % {THREADS_PER_WARP} != 0 || "
                    f"{num_threads_in_block} > 1024 || "
                    f"{num_warps} * {THREADS_PER_WARP} > {num_threads_in_block}"
                )
                kernel_stream.write(f"""\
                if ({condition}) {{
                    printf("CUDA error:\\n"
                        "1. Block must be a multiple of {THREADS_PER_WARP} threads (DaCe requirement for GPU_Warp scheduling).\\n"
                        "2. Block size must not exceed 1024 threads (CUDA hardware limit).\\n"
                        "3. Number of warps x {THREADS_PER_WARP} must fit in the block (otherwise logic is unclear).\\n");
                    asm("trap;");
                }}
                """)

            else:
                if isinstance(num_warps, symbolic.symbol):
                    condition = f"{num_warps} * {THREADS_PER_WARP} > {num_threads_in_block}"
                    scopeManager.open(condition=condition)

                elif num_warps * THREADS_PER_WARP > num_threads_in_block:
                    raise ValueError(f"Invalid configuration: {num_warps} warps x {THREADS_PER_WARP} threads exceed "
                                    f"{num_threads_in_block} threads in the block.")

                if num_threads_in_block % THREADS_PER_WARP != 0:
                    raise ValueError(f"Block must be a multiple of {THREADS_PER_WARP} threads for GPU_Warp scheduling "
                                     f"(got {num_threads_in_block}).")
    
                if num_threads_in_block > 1024:
                    raise ValueError("CUDA does not support more than 1024 threads per block (hardware limit).")
                
            
            for x in map_range.min_element():
                if isinstance(x, symbolic.symbol):
                    kernel_stream.write(f'if ({x} < 0) {{\n'
                                        f'    printf("Runtime error: Warp ID symbol {x} must be non-negative.\\n");\n'
                                        f'    asm("trap;");\n'
                                        f'}}\n')
                elif x < 0:
                    raise ValueError(f"Warp ID value {x} must be non-negative.")
                
    


    def _generate_gpu_bridge(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                             state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:


            scope_entry = dfg_scope.source_nodes()[0]

            kernel_spec: KernelSpec = self._current_kernel_spec
            kernel_name = kernel_spec.kernel_name
            kernel_bridge_args = kernel_spec.bridge_args
            kernel_bridge_args_typed = kernel_spec.bridge_args_typed

            # Declaration of the function which launches the kernel (C++ code)
            function_stream.write('DACE_EXPORTED void __dace_runkernel_%s(%s);\n' % 
                                (kernel_name, ', '.join(kernel_bridge_args_typed)), cfg, state_id, scope_entry)

            # Calling he function which launches the kernel (C++ code)
            callsite_stream.write( '__dace_runkernel_%s(%s);\n' %
                                (kernel_name, ', '.join(kernel_bridge_args)), cfg, state_id, scope_entry)
        



    def _generate_kernel_launch(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, 
                                    state_id: int, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
            
            # NOTE: This generates the function that launches the kernel.
            # Do not confuse it with CUDA's internal "LaunchKernel" API —
            # the generated function *calls* that API, but we also refer to it as a "launch function".

            scope_entry = dfg_scope.source_nodes()[0]

            kernel_spec: KernelSpec = self._current_kernel_spec
            kernel_name = kernel_spec.kernel_name
            kernel_args_as_input = kernel_spec.args_as_input
            kernel_launch_args_typed = kernel_spec.bridge_args_typed

            # get kernel dimensions and transform into a c++ string
            grid_dims = kernel_spec.grid_dims
            block_dims = kernel_spec.block_dims
            gdims = ', '.join(symbolic_to_cpp(grid_dims))
            bdims = ', '.join(symbolic_to_cpp(block_dims))



            # ----------------- Kernel Launch Function Declaration -----------------------
            self._localcode.write(
                """
                DACE_EXPORTED void __dace_runkernel_{fname}({fargs});
                void __dace_runkernel_{fname}({fargs})
                {{
                """.format(fname=kernel_name, fargs=', '.join(kernel_launch_args_typed)), 
                cfg, state_id, scope_entry
            )


            
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
            self._localcode.write(
                '''
                void  *{kname}_args[] = {{ {kargs} }};
                gpuError_t __err = {backend}LaunchKernel( (void*){kname}, dim3({gdims}), dim3({bdims}), {kname}_args, {dynsmem}, {stream}
                );
                '''.format(
                    kname=kernel_name,
                    kargs=', '.join(['(void *)&' + arg for arg in kernel_args_as_input]),
                    gdims=gdims,
                    bdims=bdims,
                    dynsmem='0',
                    stream='__state->gpu_context->streams[0]',
                    backend=self.backend
                ), 
                cfg, state_id, scope_entry
            )
            

            self._localcode.write(f'DACE_KERNEL_LAUNCH_CHECK(__err, "{kernel_name}", {gdims}, {bdims});')
            self._localcode.write('}')




    

################################# NESTED SDFG handling ############################################
# testing phase

    

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = False) -> None:
        
        if ExperimentalCUDACodeGen._in_device_code:
            self.generate_devicelevel_state(sdfg, cfg, state, function_stream, callsite_stream)
        else:
            self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream, generate_state_footer=False)

    def generate_devicelevel_state(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState,
                                   function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # Special case: if this is a GPU grid state and something is reading
        # from a possible result of a collaborative write, sync first
        if self._toplevel_schedule == dtypes.ScheduleType.GPU_Device:
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).storage == dtypes.StorageType.GPU_Shared
                        and state.in_degree(node) == 0 and state.out_degree(node) > 0):
                    break
            return

        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)

    def _emit_sync(self, codestream: CodeIOStream):
        if Config.get_bool('compiler', 'cuda', 'syncdebug'):
            codestream.write('''DACE_GPU_CHECK({backend}GetLastError());
            DACE_GPU_CHECK({backend}DeviceSynchronize());'''.format(backend=self.backend))

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

    def node_dispatch_predicate(self, sdfg, state, node):
        if hasattr(node, 'schedule'):  # NOTE: Works on nodes and scopes
            if node.schedule in dtypes.GPU_SCHEDULES:
                return True
        if ExperimentalCUDACodeGen._in_device_code:
            return True
        return False
    
    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        

        # get the generating function's name
        gen = getattr(self, '_generate_' + type(node).__name__, False)

        # if it is not implemented, use generate node of cpu impl
        if gen is not False: 
            gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        elif type(node).__name__ == 'MapExit' and node.schedule in GPU_SCHEDULES:
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
        result = self._cpu_codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state, node)
        return result

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: CodeIOStream,
                             callsite_stream: CodeIOStream) -> None:
        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule
        old_codegen = self._cpu_codegen.calling_codegen
        self._cpu_codegen.calling_codegen = self

        self._cpu_codegen._generate_NestedSDFG(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        self._cpu_codegen.calling_codegen = old_codegen
        self._toplevel_schedule = old_schedule
 



#######################################################################
    # Rather Minor "actual" changes, but much nicer to extend and maintain


    # For Yakup: I like it when we first "guard" and then implement the logic sorrow free
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
            raise NotImplementedError(
                f"CUDA: Unimplemented storage type {nodedesc.storage.name}.")

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

        # ------------------- Guard checks -------------------

        # Skip if variable is already defined
        if self._dispatcher.defined_vars.has(dataname):
            return

        if isinstance(nodedesc, (dace.data.View, dace.data.Reference)):
            return NotImplementedError("Pointers and References not implemented in ExperimentalCUDACodeGen")

        if isinstance(nodedesc, dace.data.Stream):
            raise NotImplementedError("allocate_stream not implemented in ExperimentalCUDACodeGen")

        # No clue what is happening here
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

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
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)

        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        if nodedesc.pool:
            cudastream = getattr(node, '_cuda_stream', 'nullptr')
            if cudastream != 'nullptr':
                cudastream = f'__state->gpu_context->streams[{cudastream}]'
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}MallocAsync((void**)&{dataname}, {arrsize_malloc}, {cudastream}));\n',
                cfg, state_id, node
            )
            self._emit_sync(allocation_stream)
        else:
            # Strides are left to the user's discretion
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}Malloc((void**)&{dataname}, {arrsize_malloc}));\n',
                cfg, state_id, node
            )

        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(
                f'DACE_GPU_CHECK({self.backend}Memset({dataname}, 0, {arrsize_malloc}));\n',
                cfg, state_id, node
            )

        if isinstance(nodedesc, dt.Array) and nodedesc.start_offset != 0:
            allocation_stream.write(
                f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n',
                cfg, state_id, node
            )


    def _prepare_CPU_Pinned_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                      node: nodes.AccessNode, nodedesc: dt.Data, function_stream: CodeIOStream,
                                      declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        
        dataname = ptr(node.data, nodedesc, sdfg, self._frame)
    
        # ------------------- Declaration -------------------
        array_ctype = f'{nodedesc.dtype.ctype} *'
        declared = self._dispatcher.declared_arrays.has(dataname)

        if not declared:
            declaration_stream.write(f'{array_ctype} {dataname};\n', cfg, state_id, node)

        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)


        # ------------------- Allocation -------------------
        arrsize = nodedesc.total_size
        arrsize_malloc = f'{symbolic_to_cpp(arrsize)} * sizeof({nodedesc.dtype.ctype})'

        # Strides are left to the user's discretion
        allocation_stream.write(
            f'DACE_GPU_CHECK({self.backend}MallocHost(&{dataname}, {arrsize_malloc}));\n',
            cfg, state_id, node
            )
        if node.setzero:
            allocation_stream.write(
                f'memset({dataname}, 0, {arrsize_malloc});\n',
                cfg, state_id, node
                )
            
        if nodedesc.start_offset != 0:
            allocation_stream.write(
                f'{dataname} += {symbolic_to_cpp(nodedesc.start_offset)};\n',
                cfg, state_id, node
                )


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

        declaration_stream.write(
            f'__shared__ {nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}];\n',
            cfg, state_id, node
            )
        
        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)


        # ------------------- Initialization -------------------
        if node.setzero:
            allocation_stream.write(
                f'dace::ResetShared<{nodedesc.dtype.ctype}, {", ".join(symbolic_to_cpp(self._block_dims))}, {symbolic_to_cpp(arrsize)}, '
                f'1, false>::Reset({dataname});\n',
                cfg, state_id, node
            )


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

        declaration_stream.write(
            f'{nodedesc.dtype.ctype} {dataname}[{symbolic_to_cpp(arrsize)}]{init_clause};\n',
            cfg, state_id, node
            )
        
        self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer, array_ctype)


    # I could also do deallocate based on type.. good for modularity, but may be an overkill here
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
                callsite_stream.write(
                    f'DACE_GPU_CHECK({self.backend}Free({dataname}));\n', 
                    cfg, state_id, node
                    )

        elif nodedesc.storage == dtypes.StorageType.CPU_Pinned:
            callsite_stream.write(
                'DACE_GPU_CHECK(%sFreeHost(%s));\n' % (self.backend, dataname), cfg, state_id, node)
            
        elif nodedesc.storage in {dtypes.StorageType.GPU_Shared, dtypes.StorageType.Register}:
            # No deallocation needed
            return
        
        else:
            raise NotImplementedError(f'Deallocation not implemented for storage type: {nodedesc.storage.name}')




    #######################################################################
    # Copy-pasted, might be changed in future


    def get_generated_codeobjects(self):

        # My comment: first part creates the header and stores it in a object property
        fileheader = CodeIOStream()

        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'cuda')

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
           nstreams=max(1, self._cuda_streams),
           nevents=max(1, self._cuda_events),
           backend=self.backend,
           backend_header=backend_header,
           pool_header=pool_header,
           sdfg=self._global_sdfg)

        return [self._codeobject]

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
                    if (not self._ignore_warnings): # NOTE: remove the ignoring of warnings later
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

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        self._cpu_codegen.define_out_memlet(sdfg, cfg, state_dfg, state_id, src_node, dst_node, edge, function_stream,
                                            callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)

    def _emit_copy(self, state_id: int, src_node: nodes.Node, src_storage: dtypes.StorageType, dst_node: nodes.Node,
                   dst_storage: dtypes.StorageType, dst_schedule: dtypes.ScheduleType,
                   edge: Tuple[nodes.Node, str, nodes.Node, str, Memlet], sdfg: SDFG, cfg: ControlFlowRegion,
                   dfg: StateSubgraphView, callsite_stream: CodeIOStream) -> None:
        u, uconn, v, vconn, memlet = edge
        state_dfg = cfg.state(state_id)

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
                and not ExperimentalCUDACodeGen._in_device_code
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
                         symbolic_to_cpp(dst_strides[-2]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, current_src_expr,
                         sym2cpp(src_strides[-2]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                         sym2cpp(copy_shape[-1]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                         sym2cpp(copy_shape[-2]), self.backend, src_location, dst_location, cudastream), cfg, state_id,
                        [src_node, dst_node])
                    # Write for-loop footers
                    for d in range(dims - 2):
                        callsite_stream.write("}")

            if dims == 1 and not (src_strides[-1] != 1 or dst_strides[-1] != 1):
                copysize = ' * '.join(symbolic_to_cpp(copy_shape))
                array_length = copysize
                copysize += ' * sizeof(%s)' % dtype.ctype

                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpyAsync(%s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, src_expr, copysize, self.backend, src_location, dst_location, cudastream),
                    cfg, state_id, [src_node, dst_node])
                node_dtype = dst_node.desc(sdfg).dtype
                if issubclass(node_dtype.type, ctypes.Structure):
                    callsite_stream.write('for (size_t __idx = 0; __idx < {arrlen}; ++__idx) '
                                          '{{'.format(arrlen=array_length))
                    # TODO: Study further when tackling Structures on GPU.
                    for field_name, field_type in node_dtype._typeclass.fields.items():
                        if isinstance(field_type, dtypes.pointer):
                            tclass = field_type.type

                            length = node_dtype._typeclass._length[field_name]
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
                                                                                    backend=self.backend), cfg,
                                state_id, [src_node, dst_node])
                    callsite_stream.write('}')
            elif dims == 1 and ((src_strides[-1] != 1 or dst_strides[-1] != 1)):
                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpy2DAsync(%s, %s, %s, %s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, symbolic_to_cpp(dst_strides[0]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                     src_expr, sym2cpp(src_strides[0]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                     'sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, sym2cpp(
                         copy_shape[0]), self.backend, src_location, dst_location, cudastream), cfg, state_id,
                    [src_node, dst_node])
            elif dims == 2:
                callsite_stream.write(
                    'DACE_GPU_CHECK(%sMemcpy2DAsync(%s, %s, %s, %s, %s, %s, %sMemcpy%sTo%s, %s));\n' %
                    (self.backend, dst_expr, symbolic_to_cpp(dst_strides[0]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                     src_expr, sym2cpp(src_strides[0]) + ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                     sym2cpp(copy_shape[1]) + ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype, sym2cpp(
                         copy_shape[0]), self.backend, src_location, dst_location, cudastream), cfg, state_id,
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
                    '''.format(ev=event, src_stream=cudastream, dst_stream=syncstream, backend=self.backend), cfg,
                        state_id, [src_node, dst_node])

            self._emit_sync(callsite_stream)

        # Copy within the GPU
        elif (src_storage in gpu_storage_types and dst_storage in gpu_storage_types):

            state_dfg = cfg.state(state_id)
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
                        bdims=', '.join(symbolic_to_cpp(self._block_dims)),
                        is_async='true' if state_dfg.out_degree(dst_node) == 0 else 'false',
                        accum=accum,
                        args=', '.join([src_expr] + symbolic_to_cpp(src_strides) + [dst_expr] + custom_reduction +
                                       symbolic_to_cpp(dst_strides) + symbolic_to_cpp(copy_shape))), cfg, state_id, [src_node, dst_node])
                elif funcname == 'dace::SharedToGlobal1D':
                    # special case: use a new template struct that provides functions for copy and reduction
                    callsite_stream.write(
                        ('    {func}<{type}, {bdims}, {copysize}, {is_async}>{accum}({args});').format(
                            func=funcname,
                            type=dst_node.desc(sdfg).dtype.ctype,
                            bdims=', '.join(symbolic_to_cpp(self._block_dims)),
                            copysize=', '.join(symbolic_to_cpp(copy_shape)),
                            is_async='true' if state_dfg.out_degree(dst_node) == 0 else 'false',
                            accum=accum or '::Copy',
                            args=', '.join([src_expr] + symbolic_to_cpp(src_strides) + [dst_expr] + symbolic_to_cpp(dst_strides) +
                                           custom_reduction)), cfg, state_id, [src_node, dst_node])
                else:
                    callsite_stream.write(
                        ('    {func}<{type}, {bdims}, {copysize}, ' +
                         '{dststrides}, {is_async}>{accum}({args});').format(
                             func=funcname,
                             type=dst_node.desc(sdfg).dtype.ctype,
                             bdims=', '.join(symbolic_to_cpp(self._block_dims)),
                             copysize=', '.join(symbolic_to_cpp(copy_shape)),
                             dststrides=', '.join(symbolic_to_cpp(dst_strides)),
                             is_async='true' if state_dfg.out_degree(dst_node) == 0 else 'false',
                             accum=accum,
                             args=', '.join([src_expr] + symbolic_to_cpp(src_strides) + [dst_expr] + custom_reduction)), cfg,
                        state_id, [src_node, dst_node])
            # Per-thread load (same as CPU copies)
            else:
                self._cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)
        else:
            self._cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    memlet: Memlet, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        state = cfg.state(state_id)
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



#########################################################################
# helper functions from old CUDACodeGen

def symbolic_to_cpp(arr):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr, cpp_mode=True))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d, cpp_mode=True)) for d in arr]


def _get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx > 2:
        raise ValueError('idx must be between 0 and 2, got %d' % idx)
    return ('x', 'y', 'z')[idx]


def _get_storagename(storage):
    """ Returns a string containing the name of the storage location.
        Example: dtypes.StorageType.GPU_Shared will return "Shared". """
    sname = str(storage)
    return sname[sname.rindex('_') + 1:]


# TODO: Just use product as name? 
def prod(iterable):
    return functools.reduce(sympy.Mul, iterable, 1)

#########################################################################
# Functions I had to redefine locally to not modify other files and ensure backwards compatibility


def ptr(name: str, desc: dace.data.Data, sdfg: SDFG = None, framecode=None) -> str:
    """
    Returns a string that points to the data based on its name and descriptor.

    This function should be in cpp.py, but for ExperimentalCUDACodeGen I defined 
    it here to not modify it there, s.t. we have backwards compatibility.

    :param name: Data name.
    :param desc: Data descriptor.
    :return: C-compatible name that can be used to access the data.
    """
    from dace.codegen.targets.framecode import DaCeCodeGenerator  # Avoid import loop
    framecode: DaCeCodeGenerator = framecode

    if '.' in name:
        root = name.split('.')[0]
        if root in sdfg.arrays and isinstance(sdfg.arrays[root], dace.data.Structure):
            name = name.replace('.', '->')

    # Special case: If memory is persistent and defined in this SDFG, add state
    # struct to name
    if (desc.transient and desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)):

        if desc.storage == dtypes.StorageType.CPU_ThreadLocal:  # Use unambiguous name for thread-local arrays
            return f'__{sdfg.cfg_id}_{name}'
        elif not ExperimentalCUDACodeGen._in_device_code:  # GPU kernels cannot access state
            return f'__state->__{sdfg.cfg_id}_{name}'
        elif (sdfg, name) in framecode.where_allocated and framecode.where_allocated[(sdfg, name)] is not sdfg:
            return f'__{sdfg.cfg_id}_{name}'
    elif (desc.transient and sdfg is not None and framecode is not None and (sdfg, name) in framecode.where_allocated
          and framecode.where_allocated[(sdfg, name)] is not sdfg):
        # Array allocated for another SDFG, use unambiguous name
        return f'__{sdfg.cfg_id}_{name}'

    return name




#########################################################################
# helper class


class KernelSpec:
    """
    A helper class to encapsulate information required for working with kernels.
    This class provides a structured way to store and retrieve kernel parameters.
    """

    def __init__(self, cudaCodeGen: ExperimentalCUDACodeGen, sdfg: SDFG, cfg: ControlFlowRegion,
                 dfg_scope: ScopeSubgraphView, state_id: int):
        # Entry and exit nodes of the scope
        scope_entry = dfg_scope.source_nodes()[0]
        state = cfg.state(state_id)

        self._kernel_map: nodes.Map = scope_entry.map

        # Kernel name
        self._kernel_name: str = '%s_%d_%d_%d' % (scope_entry.map.label, cfg.cfg_id, state.block_id, state.node_id(scope_entry))

        # Kernel arguments
        self._args: Dict = cudaCodeGen._arglists[scope_entry]
        self._args_typed: list[str] = [adata.as_arg(name=aname) for aname, adata in self._args.items()]
        self._args_as_input: list[str] = [ptr(aname, adata, sdfg, cudaCodeGen._frame) for aname, adata in self._args.items()]

        # Used for the bridging function, be careful: a change in the name __state will probably lead to compilation errors
        state_param: list[str] = [f'{mangle_dace_state_struct_name(cudaCodeGen._global_sdfg)} *__state']

        self._bridge_args: list[str] = ['__state'] + self._args_as_input
        self._bridge_args_typed: list[str] = state_param + self._args_typed

        # Kernel dimensions
        self._grid_dims, self._block_dims, self._has_tbmap, self._has_dtbmap, _ = cudaCodeGen.get_kernel_dimensions(dfg_scope)

    @property
    def kernel_name(self) -> list[str]:
        """Returns the kernel name."""
        return self._kernel_name
    
    @property
    def kernel_map(self) -> nodes.Map:
        """Returns the kernel map node"""
        return self._kernel_map
    
    
    @property
    def args_as_input(self) -> list[str]:
        """Returns the kernel function arguments
        that can be used as an input for calling the function.
        It is the __global__ kernel function, NOT the kernel launch function."""
        return self._args_as_input

    @property
    def args_typed(self) -> list[str]:
        """Returns the typed kernel function arguments
        that can be used for declaring the __global__ kernel function.
        These arguments include their respective data types."""
        return self._args_typed
    
    @property
    def bridge_args(self) -> list[str]:
        return self._bridge_args

    @property
    def bridge_args_typed(self) -> list[str]:
        return self._bridge_args_typed

    @property
    def grid_dims(self) -> list:
        """Returns the grid dimensions of the kernel."""
        return self._grid_dims

    @property
    def block_dims(self) -> list:
        """Returns the block dimensions of the kernel."""
        return self._block_dims

    @property
    def has_tbmap(self) -> bool:
        """Returns whether the kernel has a thread-block map."""
        return self._has_tbmap

    @property
    def has_dtbmap(self) -> bool:
        """Returns whether the kernel has a dynamic thread-block map."""
        return self._has_dtbmap



class KernelScopeManager:
    """
    A helper class to manage opening and closing brackets in a structured way using the 'with' statement.
    This class simplifies the process of correctly opening and closing brackets. It also supports an optional
    debug mode to include comments in the generated code, which can help with debugging and understanding
    the code structure.
    """

    def __init__(self, cudaCodeGen: ExperimentalCUDACodeGen, sdfg: SDFG,
                 cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int, 
                 function_stream: CodeIOStream, callsite_stream: CodeIOStream, comment: str = None,
                 debug: bool = True): 
        """
        Initializes the KernelScopeManager.

        :param cudaCodeGen: The ExperimentalCUDACodeGen instance for potential future use.
        :param sdfg: The SDFG instance for context.
        :param cfg: The ControlFlowRegion instance for context.
        :param dfg_scope: The ScopeSubgraphView instance for context.
        :param state_id: The ID of the current state for context.
        :param function_stream: The CodeIOStream for function-level code.
        :param callsite_stream: The CodeIOStream for callsite-level code.
        :param comment: A descriptive comment explaining the purpose of the code block being opened. Default is None.
        :param debug: Whether to include debug comments in the output. Defaults to False.
        """
        self.cudaCodeGen = cudaCodeGen
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
        self.cudaCodeGen._frame.allocate_arrays_in_scope(
            self.sdfg, self.cfg, self.entry_node, self.function_stream, self.callsite_stream
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deallocates arrays in scope and writes the closing brackets to the stream.
        """
        self.cudaCodeGen._frame.deallocate_arrays_in_scope(
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





