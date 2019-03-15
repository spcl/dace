from six import StringIO
import ast
import ctypes
import functools
import os
import sympy

import dace
from dace.frontend import operations
from dace import subsets, symbolic, types
from dace.config import Config
from dace.graph import nodes
from dace.sdfg import ScopeSubgraphView, SDFG, SDFGState, scope_contains_scope, is_devicelevel
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute, DefinedType)
from dace.codegen.targets.cpu import (sym2cpp, unparse_cr, cpp_array_expr,
                                      is_array_stream_view,
                                      synchronize_streams)
from dace.codegen.targets.framecode import _set_default_schedule_and_storage_types
from dace.properties import LambdaProperty

from dace.codegen import cppunparse

_SPECIAL_RTYPES = {
    types.ReductionType.Min_Location: 'ArgMin',
    types.ReductionType.Max_Location: 'ArgMax',
}


def prod(iterable):
    return functools.reduce(sympy.mul.Mul, iterable, 1)


def _expr(val):
    if isinstance(val, symbolic.SymExpr):
        return val.expr
    return val


class CUDACodeGen(TargetCodeGenerator):
    """ GPU (CUDA) code generator. """
    target_name = 'cuda'
    title = 'CUDA'
    language = 'cu'

    def __init__(self, frame_codegen, sdfg):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self._in_device_code = False
        self._cpu_codegen = None
        self._block_dims = None
        self._codeobject = CodeObject(sdfg.name + '_' + 'cuda', '', 'cu',
                                      CUDACodeGen, 'CUDA')
        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self._global_sdfg = sdfg
        self._toplevel_schedule = None

        # Keep track of current "scope entry/exit" code streams for extra
        # code generation
        self.scope_entry_stream = self._initcode
        self.scope_exit_stream = self._exitcode

        # Annotate CUDA streams and events
        self._cuda_streams, self._cuda_events = self._compute_cudastreams(sdfg)

        # Register dispatchers
        self._cpu_codegen = dispatcher.get_generic_node_dispatcher()

        # Register additional CUDA dispatchers
        dispatcher.register_map_dispatcher(types.GPU_SCHEDULES, self)

        dispatcher.register_node_dispatcher(
            self, CUDACodeGen.node_dispatch_predicate)

        dispatcher.register_state_dispatcher(self,
                                             self.state_dispatch_predicate)

        gpu_storage = [
            types.StorageType.GPU_Global, types.StorageType.GPU_Shared,
            types.StorageType.GPU_Stack
        ]
        dispatcher.register_array_dispatcher(gpu_storage, self)
        dispatcher.register_array_dispatcher(types.StorageType.CPU_Pinned,
                                             self)

        for storage in gpu_storage:
            for other_storage in types.StorageType:
                dispatcher.register_copy_dispatcher(storage, other_storage,
                                                    None, self)
                dispatcher.register_copy_dispatcher(other_storage, storage,
                                                    None, self)

        # Register illegal copies
        cpu_unpinned_storage = [
            types.StorageType.CPU_Heap, types.StorageType.CPU_Stack
        ]
        gpu_private_storage = [
            types.StorageType.GPU_Shared, types.StorageType.GPU_Stack
        ]
        illegal_copy = IllegalCopy()
        for st in cpu_unpinned_storage:
            for gst in gpu_private_storage:
                dispatcher.register_copy_dispatcher(st, gst, None,
                                                    illegal_copy)
                dispatcher.register_copy_dispatcher(gst, st, None,
                                                    illegal_copy)
        for st in cpu_unpinned_storage:
            for sched_type in [
                    types.ScheduleType.GPU_Device,
                    types.ScheduleType.GPU_ThreadBlock
            ]:
                dispatcher.register_copy_dispatcher(
                    st, types.StorageType.Register, sched_type, illegal_copy)
                dispatcher.register_copy_dispatcher(
                    types.StorageType.Register, st, sched_type, illegal_copy)
        # End of illegal copies
        # End of dispatcher registration
        ######################################

    # Generate final code
    def get_generated_codeobjects(self):
        fileheader = CodeIOStream()
        self._frame.generate_fileheader(self._global_sdfg, fileheader)

        self._codeobject.code = """
#include <cuda_runtime.h>
#include <dace/dace.h>

{file_header}

DACE_EXPORTED int __dace_init_cuda({params});
DACE_EXPORTED void __dace_exit_cuda({params});

{other_globalcode}

namespace dace {{ namespace cuda {{
    cudaStream_t __streams[{nstreams}];
    cudaEvent_t __events[{nevents}];
}} }}

int __dace_init_cuda({params}) {{
    int count;

    // Check that we are able to run CUDA code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {{
        printf("ERROR: CUDA drivers are not configured or CUDA-capable device "
               "not found\\n");
        return 1;
    }}
    if (count == 0)
    {{
        printf("ERROR: No CUDA-capable devices found\\n");
        return 2;
    }}

    // Initialize CUDA before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);

    // Create CUDA streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        cudaStreamCreateWithFlags(&dace::cuda::__streams[i], cudaStreamNonBlocking);
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        cudaEventCreateWithFlags(&dace::cuda::__events[i], cudaEventDisableTiming);
    }}

    {initcode}

    return 0;
}}

void __dace_exit_cuda({params}) {{
    {exitcode}

    // Destroy CUDA streams and events
    for(int i = 0; i < {nstreams}; ++i) {{
        cudaStreamDestroy(dace::cuda::__streams[i]);
    }}
    for(int i = 0; i < {nevents}; ++i) {{
        cudaEventDestroy(dace::cuda::__events[i]);
    }}
}}

{localcode}
""".format(params=self._global_sdfg.signature(),
           initcode=self._initcode.getvalue(),
           exitcode=self._exitcode.getvalue(),
           other_globalcode=self._globalcode.getvalue(),
           localcode=self._localcode.getvalue(),
           file_header=fileheader.getvalue(),
           nstreams=self._cuda_streams,
           nevents=self._cuda_events)

        return [self._codeobject]

    @staticmethod
    def node_dispatch_predicate(sdfg, node):
        if (getattr(node, 'schedule', False)
                and node.schedule in types.GPU_SCHEDULES):
            return True
        return False

    def state_dispatch_predicate(self, sdfg, state):
        if self._toplevel_schedule in types.GPU_SCHEDULES:
            return True
        for node in state.sink_nodes():
            if hasattr(node, '_cuda_stream'):
                return True
            else:
                for e in state.in_edges(node):
                    if hasattr(e.src, '_cuda_stream'):
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

        host_compiler = make_absolute(
            Config.get("compiler", "cpu", "executable"))
        compiler = make_absolute(Config.get("compiler", "cuda", "executable"))
        flags = Config.get("compiler", "cuda", "args")
        flags += Config.get("compiler", "cuda", "additional_args")

        # Get CUDA architectures from configuration
        cuda_arch = Config.get('compiler', 'cuda', 'cuda_arch').split(',')
        cuda_arch = [ca for ca in cuda_arch if ca is not None and len(ca) > 0]

        flags += ' ' + ' '.join(
            '-gencode arch=compute_{arch},code=sm_{arch}'.format(arch=arch)
            for arch in cuda_arch)

        options = [
            "-DCUDA_HOST_COMPILER=\"{}\"".format(host_compiler),
            "-DCUDA_NVCC_FLAGS=\"{}\"".format(flags),
            "-DCUDA_TOOLKIT_ROOT_DIR=\"{}\"".format(
                os.path.dirname(os.path.dirname(compiler).replace('\\', '/')))
        ]

        return options

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        nodedesc = node.desc(sdfg)
        if isinstance(nodedesc, dace.data.Stream):
            return self.allocate_stream(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream)

        result = StringIO()
        arrsize = ' * '.join([
            cppunparse.pyexpr2cpp(symbolic.symstr(s)) for s in nodedesc.strides
        ])
        is_dynamically_sized = any(
            symbolic.issymbolic(s, sdfg.constants) for s in nodedesc.strides)
        arrsize_malloc = arrsize + ' * sizeof(%s)' % nodedesc.dtype.ctype
        dataname = node.data

        # Different types of GPU arrays
        if nodedesc.storage == types.StorageType.GPU_Global:
            result.write(
                '%s *%s = nullptr;\n' % (nodedesc.dtype.ctype, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer)

            # Strides are left to the user's discretion
            result.write('cudaMalloc(&%s, %s);\n' % (dataname, arrsize_malloc))
            if node.setzero:
                result.write(
                    'cudaMemset(%s, 0, %s);\n' % (dataname, arrsize_malloc))

        elif nodedesc.storage == types.StorageType.CPU_Pinned:
            result.write(
                '%s *%s = nullptr;\n' % (nodedesc.dtype.ctype, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer)

            # Strides are left to the user's discretion
            result.write(
                'cudaMallocHost(&%s, %s);\n' % (dataname, arrsize_malloc))
            if node.setzero:
                result.write(
                    'memset(%s, 0, %s);\n' % (dataname, arrsize_malloc))
        elif nodedesc.storage == types.StorageType.GPU_Shared:
            if is_dynamically_sized:
                raise NotImplementedError('Dynamic shared memory unsupported')
            result.write("__shared__ %s %s[%s];\n" % (nodedesc.dtype.ctype,
                                                      dataname, arrsize))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer)
            if node.setzero:
                result.write(
                    'dace::ResetShared<{type}, {block_size}, {elements}, '
                    '1, false>::Reset({ptr});\n'.format(
                        type=nodedesc.dtype.ctype,
                        block_size=', '.join(_topy(self._block_dims)),
                        ptr=dataname,
                        elements=arrsize))
        elif nodedesc.storage == types.StorageType.GPU_Stack:
            if is_dynamically_sized:
                raise ValueError('Dynamic allocation of registers not allowed')
            szstr = ' = {0}' if node.setzero else ''
            result.write("%s %s[%s]%s;\n" % (nodedesc.dtype.ctype, dataname,
                                             arrsize, szstr))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Pointer)
        else:
            raise NotImplementedError("CUDA: Unimplemented storage type " +
                                      str(nodedesc.storage))

        callsite_stream.write(result.getvalue(), sdfg, state_id, node)

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        # No need (for now)
        pass

    def allocate_stream(self, sdfg, dfg, state_id, node, function_stream,
                        callsite_stream):
        nodedesc = node.desc(sdfg)
        dataname = node.data
        if nodedesc.storage == types.StorageType.GPU_Global:
            fmtargs = {
                'name': dataname,
                'type': nodedesc.dtype.ctype,
                'is_pow2': sym2cpp(
                    sympy.log(nodedesc.buffer_size, 2).is_Integer),
                'location':
                '%s_%s_%s' % (sdfg.name, state_id, dfg.node_id(node)),
            }

            self._dispatcher.defined_vars.add(dataname, DefinedType.Stream)

            if is_array_stream_view(sdfg, dfg, node):
                fmtargs['ptr'] = nodedesc.sink
                # Assuming 1D array sink/src
                fmtargs['size'] = sym2cpp(sdfg.arrays[nodedesc.sink].shape[0])

                function_stream.write(
                    'DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'.
                    format(**fmtargs), sdfg, state_id, node)
                self._globalcode.write(
                    """
DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);
void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result) {{
    result = dace::AllocGPUArrayStreamView<{type}, {is_pow2}>(ptr, size);
}}""".format(**fmtargs), sdfg, state_id, node)
                callsite_stream.write(
                    'dace::GPUStream<{type}, {is_pow2}> {name}; __dace_alloc_{location}({ptr}, {size}, {name});'.
                    format(**fmtargs), sdfg, state_id, node)
            else:
                fmtargs['size'] = sym2cpp(nodedesc.buffer_size)

                function_stream.write(
                    'DACE_EXPORTED void __dace_alloc_{location}(uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'.
                    format(**fmtargs), sdfg, state_id, node)
                self._globalcode.write(
                    """
DACE_EXPORTED void __dace_alloc_{location}(uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);
dace::GPUStream<{type}, {is_pow2}> __dace_alloc_{location}(uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result) {{
    result = dace::AllocGPUStream<{type}, {is_pow2}>({size});
}}""".format(**fmtargs), sdfg, state_id, node)
                callsite_stream.write(
                    'dace::GPUStream<{type}, {is_pow2}> {name}; __dace_alloc_{location}({size}, {name});'.
                    format(**fmtargs), sdfg, state_id, node)

    def deallocate_stream(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        nodedesc = node.desc(sdfg)
        dataname = node.data
        if nodedesc.storage == types.StorageType.GPU_Global:
            if is_array_stream_view(sdfg, dfg, node):
                callsite_stream.write(
                    'dace::FreeGPUArrayStreamView(%s);' % dataname, sdfg,
                    state_id, node)
            else:
                callsite_stream.write('dace::FreeGPUStream(%s);' % dataname,
                                      sdfg, state_id, node)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        nodedesc = node.desc(sdfg)
        dataname = node.data
        if isinstance(nodedesc, dace.data.Stream):
            return self.deallocate_stream(sdfg, dfg, state_id, node,
                                          function_stream, callsite_stream)

        if nodedesc.storage == types.StorageType.GPU_Global:
            callsite_stream.write('cudaFree(%s);\n' % dataname, sdfg, state_id,
                                  node)
        elif nodedesc.storage == types.StorageType.CPU_Pinned:
            callsite_stream.write('cudaFreeHost(%s);\n' % dataname, sdfg,
                                  state_id, node)
        elif nodedesc.storage == types.StorageType.GPU_Shared or \
             nodedesc.storage == types.StorageType.GPU_Stack:
            pass  # Do nothing
        else:
            raise NotImplementedError

    def _compute_cudastreams(self,
                             sdfg: SDFG,
                             default_stream=0,
                             default_event=0):
        """ Annotates an SDFG (and all nested ones) to include a `_cuda_stream`
            field. This field is applied to all GPU maps, tasklets, and copies
            that can be executed in parallel.
            @param sdfg: The sdfg to modify.
            @param default_stream: The stream ID to start counting from (used
                                   in recursion to nested SDFGs).
            @param default_event: The event ID to start counting from (used
                                  in recursion to nested SDFGs).
            @return: 2-tuple of the number of streams, events to create.
        """
        concurrent_streams = Config.get('compiler', 'cuda',
                                        'max_concurrent_streams')
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
                    if node.schedule == types.ScheduleType.GPU_Device:
                        continue
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
                    if e.src._cs_childpath == True:
                        c = max_streams
                        max_streams = increment(max_streams)
                    e.src._cs_childpath = True
                else:
                    c = max_streams
                    max_streams = increment(max_streams)
                e.dst._cuda_stream = c
                if not hasattr(e.dst, '_cs_childpath'):
                    e.dst._cs_childpath = False
                if isinstance(e.dst, nodes.NestedSDFG):
                    if e.dst.schedule not in types.GPU_SCHEDULES:
                        max_streams, max_events = self._compute_cudastreams(
                            e.dst.sdfg, e.dst._cuda_stream, max_events + 1)

            state_streams.append(max_streams if concurrent_streams == 0 else
                                 concurrent_streams)
            state_subsdfg_events.append(max_events)

        # Remove CUDA streams from paths of non-gpu copies and CPU tasklets
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(graph, SDFGState):
                cur_sdfg = graph.parent
                for e in graph.out_edges(node):
                    path = graph.memlet_path(e)
                    # If leading from/to a GPU memory node, keep stream
                    if ((isinstance(path[0].src, nodes.AccessNode)
                         and path[0].src.desc(
                             cur_sdfg).storage == types.StorageType.GPU_Global)
                            or (isinstance(path[-1].dst, nodes.AccessNode)
                                and path[-1].dst.desc(cur_sdfg).storage ==
                                types.StorageType.GPU_Global)):
                        break
                    # If leading from/to a GPU tasklet, keep stream
                    if ((isinstance(path[0].src, nodes.CodeNode)
                         and is_devicelevel(cur_sdfg, graph, path[0].src)) or
                        (isinstance(path[-1].dst, nodes.CodeNode)
                         and is_devicelevel(cur_sdfg, graph, path[-1].dst))):
                        break
                    # If leading from/to a GPU reduction, keep stream
                    if ((isinstance(path[0].src, nodes.Reduce) and
                         path[0].src.schedule == types.ScheduleType.GPU_Device)
                            or
                        (isinstance(path[-1].dst, nodes.Reduce) and path[-1]
                         .dst.schedule == types.ScheduleType.GPU_Device)):
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
                    if (not hasattr(e.dst, '_cuda_stream')
                            or e.src._cuda_stream != e.dst._cuda_stream):
                        for mpe in state.memlet_path(e):
                            mpe._cuda_event = events
                        events += 1

            state_events.append(events)

        # Maximum over all states
        max_streams = max(state_streams)
        max_events = max(state_events)

        return max_streams, max_events

    def _emit_copy(self, state_id, src_node, src_storage, dst_node,
                   dst_storage, dst_schedule, edge, sdfg, dfg,
                   callsite_stream):
        u, uconn, v, vconn, memlet = edge
        state_dfg = sdfg.nodes()[state_id]

        cpu_storage_types = [
            types.StorageType.CPU_Heap, types.StorageType.CPU_Stack,
            types.StorageType.CPU_Pinned
        ]
        gpu_storage_types = [
            types.StorageType.GPU_Global, types.StorageType.GPU_Shared,
            types.StorageType.GPU_Stack
        ]

        copy_shape = memlet.subset.bounding_box_size()
        copy_shape = [symbolic.overapproximate(s) for s in copy_shape]
        # Determine directionality
        if (isinstance(src_node, nodes.AccessNode)
                and memlet.data == src_node.data):
            outgoing_memlet = True
        elif (isinstance(dst_node, nodes.AccessNode)
              and memlet.data == dst_node.data):
            outgoing_memlet = False
        else:
            raise LookupError('Memlet does not point to any of the nodes')

        if (isinstance(src_node, nodes.AccessNode)
                and isinstance(dst_node, nodes.AccessNode)
                and not self._in_device_code
                and (src_storage == types.StorageType.GPU_Global
                     or dst_storage == types.StorageType.GPU_Global)):
            src_location = 'Device' if src_storage == types.StorageType.GPU_Global else 'Host'
            dst_location = 'Device' if dst_storage == types.StorageType.GPU_Global else 'Host'

            syncwith = {}  # Dictionary of {stream: event}
            is_sync = False
            max_streams = Config.get('compiler', 'cuda',
                                     'max_concurrent_streams')

            if hasattr(src_node, '_cuda_stream'):
                cudastream = src_node._cuda_stream
                if not hasattr(dst_node, '_cuda_stream'):
                    # Copy after which data is needed by the host
                    is_sync = True
                elif dst_node._cuda_stream != src_node._cuda_stream:
                    syncwith[dst_node._cuda_stream] = edge._cuda_event
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
                    elif e.dst._cuda_stream != cudastream:
                        syncwith[e.dst._cuda_stream] = e._cuda_event

                if cudastream != 'nullptr':
                    cudastream = 'dace::cuda::__streams[%d]' % cudastream

            if memlet.wcr is not None:
                raise NotImplementedError('Accumulate %s to %s not implemented'
                                          % (src_location, dst_location))
            #############################

            # Obtain copy information
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                self._cpu_codegen.memlet_copy_to_absolute_strides(
                    sdfg, memlet, src_node, dst_node))

            dims = len(copy_shape)

            # Handle unsupported copy types
            if dims == 2 and (src_strides[-1] != 1 or dst_strides[-1] != 1):
                raise NotImplementedError('2D copy only supported with one '
                                          'stride')

            # Currently we only support ND copies when they can be represented
            # as a 1D copy or as a 2D strided copy
            if dims > 2:
                raise NotImplementedError('Copies between CPU and GPU are not'
                                          ' supported for N-dimensions')

            if dims == 1:
                copysize = ' * '.join([
                    cppunparse.pyexpr2cpp(symbolic.symstr(s))
                    for s in copy_shape
                ])
                array_length = copysize
                copysize += ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype

                callsite_stream.write(
                    'cudaMemcpyAsync(%s, %s, %s, cudaMemcpy%sTo%s, %s);\n' %
                    (dst_expr, src_expr, copysize, src_location, dst_location,
                     cudastream), sdfg, state_id, [src_node, dst_node])
                node_dtype = dst_node.desc(sdfg).dtype
                if issubclass(node_dtype.type, ctypes.Structure):
                    callsite_stream.write(
                        'for (auto __idx = 0; __idx < {arrlen}; ++__idx) '
                        '{{'.format(arrlen=str(array_length)))
                    for field_name, field_type in node_dtype._data.items():
                        if isinstance(field_type, types.pointer):
                            tclass = field_type.type
                            length = node_dtype._length[field_name]
                            size = 'sizeof({})*{}[__idx].{}'.format(
                                types._CTYPES[tclass], str(src_node), length)
                            callsite_stream.write(
                                'cudaMalloc(&{dst}[__idx].{fname}, '
                                '{sz});'.format(
                                    dst=str(dst_node),
                                    fname=field_name,
                                    sz=size))
                            callsite_stream.write(
                                'cudaMemcpyAsync({dst}[__idx].{fname}, '
                                '{src}[__idx].{fname}, {sz}, '
                                'cudaMemcpy{sloc}To{dloc}, {stream});'.format(
                                    dst=str(dst_node),
                                    src=str(src_node),
                                    fname=field_name,
                                    sz=size,
                                    sloc=src_location,
                                    dloc=dst_location,
                                    stream=cudastream), sdfg, state_id,
                                [src_node, dst_node])
                    callsite_stream.write('}')
            elif dims == 2:
                callsite_stream.write(
                    'cudaMemcpy2DAsync(%s, %s, %s, %s, %s, %s, cudaMemcpy%sTo%s, %s);\n'
                    % (dst_expr, _topy(dst_strides[0]) +
                       ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                       src_expr, sym2cpp(src_strides[0]) +
                       ' * sizeof(%s)' % src_node.desc(sdfg).dtype.ctype,
                       sym2cpp(copy_shape[1]) +
                       ' * sizeof(%s)' % dst_node.desc(sdfg).dtype.ctype,
                       sym2cpp(copy_shape[0]), src_location, dst_location,
                       cudastream), sdfg, state_id, [src_node, dst_node])

            # Post-copy synchronization
            if is_sync:
                # Synchronize with host (done at destination)
                pass
            else:
                # Synchronize with other streams as necessary
                for streamid, event in syncwith.items():
                    syncstream = 'dace::cuda::__streams[%d]' % streamid
                    callsite_stream.write(
                        '''
    cudaEventRecord(dace::cuda::__events[{ev}], {src_stream});
    cudaStreamWaitEvent({dst_stream}, dace::cuda::__events[{ev}], 0);
                    '''.format(
                            ev=event,
                            src_stream=cudastream,
                            dst_stream=syncstream), sdfg, state_id,
                        [src_node, dst_node])

        # Copy within the GPU
        elif (src_storage in gpu_storage_types
              and dst_storage in gpu_storage_types):

            state_dfg = sdfg.nodes()[state_id]
            sdict = state_dfg.scope_dict()
            if scope_contains_scope(sdict, src_node, dst_node):
                inner_schedule = dst_schedule
            else:
                inner_schedule = sdict[src_node]
                if inner_schedule is not None:
                    inner_schedule = inner_schedule.map.schedule
            if inner_schedule is None:  # Top-level schedule
                inner_schedule = self._toplevel_schedule

            # Collaborative load
            if inner_schedule == types.ScheduleType.GPU_Device:
                # Obtain copy information
                copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                    self._cpu_codegen.memlet_copy_to_absolute_strides(
                        sdfg, memlet, src_node, dst_node))

                dims = len(copy_shape)

                funcname = 'dace::%sTo%s%dD' % (_get_storagename(src_storage),
                                                _get_storagename(dst_storage),
                                                dims)

                accum = ''
                custom_reduction = []
                if memlet.wcr is not None:
                    redtype = operations.detect_reduction_type(memlet.wcr)
                    reduction_tmpl = ''
                    # Special call for detected reduction types
                    if redtype != types.ReductionType.Custom:
                        credtype = ('dace::ReductionType::' +
                                    str(redtype)[str(redtype).find('.') + 1:])
                        reduction_tmpl = '<%s>' % credtype
                    else:
                        custom_reduction = [unparse_cr(memlet.wcr)]
                    accum = '::template Accum%s' % reduction_tmpl

                if any(
                        symbolic.issymbolic(s, sdfg.constants)
                        for s in copy_shape):
                    callsite_stream.write((
                        '    {func}Dynamic<dace::vec<{type}, {veclen}>, {bdims}, '
                        + '{dststrides}, {is_async}>{accum}({args});').format(
                            func=funcname,
                            type=dst_node.desc(sdfg).dtype.ctype,
                            veclen=memlet.veclen,
                            bdims=', '.join(_topy(self._block_dims)),
                            dststrides=', '.join(_topy(dst_strides)),
                            is_async='false'
                            if state_dfg.out_degree(dst_node) > 0 else 'true',
                            accum=accum,
                            args=', '.join([src_expr] + _topy(src_strides) +
                                           [dst_expr] + custom_reduction +
                                           _topy(copy_shape))), sdfg, state_id,
                                          [src_node, dst_node])
                else:
                    callsite_stream.write((
                        '    {func}<dace::vec<{type}, {veclen}>, {bdims}, {copysize}, '
                        + '{dststrides}, {is_async}>{accum}({args});').format(
                            func=funcname,
                            type=dst_node.desc(sdfg).dtype.ctype,
                            veclen=memlet.veclen,
                            bdims=', '.join(_topy(self._block_dims)),
                            copysize=', '.join(_topy(copy_shape)),
                            dststrides=', '.join(_topy(dst_strides)),
                            is_async='false'
                            if state_dfg.out_degree(dst_node) > 0 else 'true',
                            accum=accum,
                            args=', '.join([src_expr] + _topy(src_strides) +
                                           [dst_expr] + custom_reduction)),
                                          sdfg, state_id, [src_node, dst_node])
            # Per-thread load (same as CPU copies)
            else:
                self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node,
                                              dst_node, edge, None,
                                              callsite_stream)
        else:
            self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node,
                                          dst_node, edge, None,
                                          callsite_stream)

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, memlet,
                    function_stream, callsite_stream):
        if isinstance(src_node, nodes.Tasklet):
            src_storage = types.StorageType.Register
            src_parent = dfg.scope_dict()[src_node]
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = types.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        dst_parent = dfg.scope_dict()[dst_node]
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        # Emit actual copy
        self._emit_copy(state_id, src_node, src_storage, dst_node, dst_storage,
                        dst_schedule, memlet, sdfg, dfg, callsite_stream)

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        # Two modes: device-level state and if this state has active streams
        if self._toplevel_schedule in types.GPU_SCHEDULES:
            self.generate_devicelevel_state(sdfg, state, function_stream,
                                            callsite_stream)
        else:
            # Active streams found. Generate state normally and sync with the
            # streams in the end
            self._frame.generate_state(
                sdfg,
                state,
                function_stream,
                callsite_stream,
                generate_state_footer=False)
            if state.nosync == False:
                streams_to_sync = set()
                for node in state.sink_nodes():
                    if hasattr(node, '_cuda_stream'):
                        streams_to_sync.add(node._cuda_stream)
                    else:
                        # Synchronize sink-node copies at the end of the state
                        for e in state.in_edges(node):
                            if hasattr(e.src, '_cuda_stream'):
                                streams_to_sync.add(e.src._cuda_stream)
                for stream in streams_to_sync:
                    callsite_stream.write(
                        'cudaStreamSynchronize(dace::cuda::__streams[%d]);' %
                        stream, sdfg, sdfg.node_id(state))

            # After synchronizing streams, generate state footer normally

            # Emit internal transient array deallocation
            sid = sdfg.node_id(state)
            data_to_allocate = (set(state.top_level_transients()) - set(
                sdfg.shared_transients()))
            deallocated = set()
            for node in state.data_nodes():
                if node.data not in data_to_allocate or node.data in deallocated:
                    continue
                deallocated.add(node.data)
                self._frame._dispatcher.dispatch_deallocate(
                    sdfg, state, sid, node, function_stream, callsite_stream)

    def generate_devicelevel_state(self, sdfg, state, function_stream,
                                   callsite_stream):

        # Special case: if this is a GPU grid state and something is reading
        # from a possible result of a collaborative write, sync first
        if self._toplevel_schedule == types.ScheduleType.GPU_Device:
            state_id = next(
                i for i, s in enumerate(sdfg.nodes()) if s == state)
            for node in state.nodes():
                if (isinstance(node, nodes.AccessNode) and
                        node.desc(sdfg).storage == types.StorageType.GPU_Shared
                        and state.in_degree(node) == 0
                        and state.out_degree(node) > 0):
                    callsite_stream.write('__syncthreads();', sdfg, state_id)
                    break

        self._frame.generate_state(sdfg, state, function_stream,
                                   callsite_stream)

    # NOTE: This function is ONLY called from the CPU side. Therefore, any
    # schedule that is out of the ordinary will raise an exception
    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        scope_entry = dfg_scope.source_nodes()[0]
        scope_exit = dfg_scope.sink_nodes()[0]

        dfg = sdfg.nodes()[state_id]

        # If in device-level code, call appropriate function
        if (self._toplevel_schedule == types.ScheduleType.GPU_Device or
            (dfg.scope_dict()[scope_entry] is not None and dfg.scope_dict()
             [scope_entry].map.schedule in types.GPU_SCHEDULES)):
            self.generate_devicelevel_scope(sdfg, dfg_scope, state_id,
                                            function_stream, callsite_stream)
            return

        # If not device-level code, ensure the schedule is correct
        if scope_entry.map.schedule != types.ScheduleType.GPU_Device:
            raise TypeError('Cannot schedule %s directly from non-GPU code' %
                            str(scope_entry.map.schedule))

        # Determine whether to create a global (grid) barrier object
        create_grid_barrier = False
        for node in dfg_scope.nodes():
            if scope_entry == node: continue
            if (isinstance(node, nodes.EntryNode)
                    and node.map.schedule == types.ScheduleType.GPU_Device):
                create_grid_barrier = True

        kernel_name = '%s_%d_%d' % (
            scope_entry.map.label, dfg.node_id(scope_entry), sdfg.node_id(dfg))

        # Get parameters from input/output memlets to this map
        params = set(d.data for node in dfg_scope.source_nodes() for _,_,_,_,d in dfg.in_edges(node)) | \
                 set(d.data for node in dfg_scope.sink_nodes() for _,_,_,_,d in dfg.out_edges(node))

        # Get symbolic parameters (free symbols) for kernel
        syms = sdfg.symbols_defined_at(scope_entry)
        freesyms = {
            k: v
            for k, v in syms.items()
            if k not in sdfg.constants and k not in scope_entry.map.params
        }
        symbol_sigs = [
            v.dtype.ctype + ' ' + k for k, v in sorted(freesyms.items())
        ]
        symbol_names = [k for k in sorted(freesyms.keys())]

        # Hijack symbol_sigs to create a grid barrier object
        if create_grid_barrier:
            symbol_sigs.append('cub::GridBarrier __gbar')

        # Comprehend grid/block dimensions from scopes
        grid_dims, block_dims, tbmap = self.get_kernel_dimensions(dfg_scope)

        kernel_args = [
            sdfg.arrays[p].signature(False, name=p) for p in sorted(params)
        ] + symbol_names
        kernel_args_typed = [
            sdfg.arrays[p].signature(name=p) for p in sorted(params)
        ] + symbol_sigs

        # Store init/exit code streams
        old_entry_stream = self.scope_entry_stream
        old_exit_stream = self.scope_exit_stream
        self.scope_entry_stream = CodeIOStream()
        self.scope_exit_stream = CodeIOStream()

        kernel_stream = CodeIOStream()
        self.generate_kernel_scope(sdfg, dfg_scope, state_id, scope_entry.map,
                                   kernel_name, grid_dims, block_dims, tbmap,
                                   kernel_args_typed, self._globalcode,
                                   kernel_stream)

        # Write kernel prototype
        node = dfg_scope.source_nodes()[0]
        self._localcode.write(
            '__global__ void %s(%s) {\n' %
            (kernel_name, ', '.join(kernel_args_typed)), sdfg, state_id, node)

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

        # Write callback function definition
        self._localcode.write(
            """
DACE_EXPORTED void __dace_runkernel_{fname}({fargs});
void __dace_runkernel_{fname}({fargs})
{{
""".format(fname=kernel_name, fargs=', '.join(kernel_args_typed)), sdfg,
            state_id, node)

        if create_grid_barrier:
            gbar = '__gbar_' + kernel_name
            self._localcode.write('    cub::GridBarrierLifetime %s;\n' % gbar,
                                  sdfg, state_id, node)
            self._localcode.write(
                '    %s.Setup(%s);\n' % (gbar, ' * '.join(_topy(grid_dims))),
                sdfg, state_id, node)
            symbol_names.append(gbar)

        # Compute dynamic shared memory
        dynsmem_size = 0
        # For all access nodes, if array storage == GPU_Shared and size is
        # symbolic, add it. If nested SDFG, check all internal arrays
        for node in dfg_scope.nodes():
            if isinstance(node, nodes.AccessNode):
                arr = sdfg.arrays[node.data]
                if arr.storage == types.StorageType.GPU_Shared:
                    numel = functools.reduce(lambda a, b: a * b, arr.shape)
                    if symbolic.issymbolic(numel, sdfg.constants):
                        dynsmem_size += numel
            elif isinstance(node, nodes.NestedSDFG):
                for arr in node.sdfg.arrays_recursive():
                    if (arr is not None
                            and arr.storage == types.StorageType.GPU_Shared):
                        numel = functools.reduce(lambda a, b: a * b, arr.shape)
                        if symbolic.issymbolic(numel, sdfg.constants):
                            dynsmem_size += numel

        max_streams = Config.get('compiler', 'cuda', 'max_concurrent_streams')
        if max_streams >= 0:
            cudastream = 'dace::cuda::__streams[%d]' % scope_entry._cuda_stream
        else:
            cudastream = 'nullptr'

        self._localcode.write(
            '''
void  *{kname}_args[] = {{ {kargs} }};
cudaLaunchKernel((void*){kname}, dim3({gdims}), dim3({bdims}), {kname}_args, {dynsmem}, {stream});'''
            .format(
                kname=kernel_name,
                kargs=', '.join(['(void *)&' + arg for arg in kernel_args]),
                gdims=','.join(_topy(grid_dims)),
                bdims=','.join(_topy(block_dims)),
                dynsmem=_topy(dynsmem_size),
                stream=cudastream), sdfg, state_id, node)

        # Close the runkernel function
        self._localcode.write('}')
        #######################
        # Add invocation to calling code (in another file)
        function_stream.write(
            'DACE_EXPORTED void __dace_runkernel_%s(%s);\n' %
            (kernel_name, ', '.join(kernel_args_typed)), sdfg, state_id, node)
        callsite_stream.write(
            '__dace_runkernel_%s(%s);\n' %
            (kernel_name, ', '.join(kernel_args)), sdfg, state_id, node)

        synchronize_streams(sdfg, dfg, state_id, node, scope_exit,
                            callsite_stream)

    def get_kernel_dimensions(self, dfg_scope):
        """ Determines a CUDA kernel's grid/block dimensions from map
            scopes.

            Ruleset for kernel dimensions:
                1. If only one map (device-level) exists, of an integer set S,
                   the block size is 32x1x1 and grid size is ceil(|S|/32) in 
                   1st dimension.
                2. If nested thread-block maps exist (T_1,...,T_n), grid 
                   size is |S| and block size is max(|T_1|,...,|T_n|) with 
                   block specialization.
                3. If block size can be overapproximated, it is (for 
                    dynamically-sized blocks that are bounded by a 
                    predefined size).
    
            @note: Kernel dimensions are separate from the map
                   variables, and they should be treated as such.
            @note: To make use of the grid/block 3D registers, we use multi-
                   dimensional kernels up to 3 dimensions, and flatten the 
                   rest into the third dimension.
        """

        kernelmap_entry = dfg_scope.source_nodes()[0]
        grid_size = kernelmap_entry.map.range.size(True)[::-1]
        block_size = None

        # Linearize (flatten) rest of dimensions to third
        if len(grid_size) > 3:
            grid_size[2] = functools.reduce(sympy.mul.Mul, grid_size[2:], 1)
            del grid_size[3:]

        # Extend to 3 dimensions if necessary
        grid_size = grid_size + [1] * (3 - len(grid_size))

        # Obtain thread-block maps for case (2)
        tb_maps = [
            node.map for node, parent in dfg_scope.scope_dict().items()
            if parent == kernelmap_entry and isinstance(node, nodes.EntryNode)
            and node.schedule == types.ScheduleType.GPU_ThreadBlock
        ]
        # Append thread-block maps from nested SDFGs
        for node in dfg_scope.scope_subgraph(kernelmap_entry).nodes():
            if isinstance(node, nodes.NestedSDFG):
                _set_default_schedule_and_storage_types(
                    node.sdfg, node.schedule)

                tb_maps.extend([
                    n.map for state in node.sdfg.nodes()
                    for n in state.nodes() if isinstance(n, nodes.MapEntry)
                    and n.schedule == types.ScheduleType.GPU_ThreadBlock
                ])

        # Case (1): no thread-block maps
        if len(tb_maps) == 0:

            print('WARNING: Thread-block maps not found in kernel, assuming ' +
                  'block size of (%s)' %
                  Config.get('compiler', 'cuda', 'default_block_size'))
            block_size = [
                int(b) for b in Config.get('compiler', 'cuda',
                                           'default_block_size').split(',')
            ]
            assert (len(block_size) >= 1 and len(block_size) <= 3)

            int_ceil = sympy.Function('int_ceil')

            # Grid size = ceil(|S|/32) for first dimension, rest = |S|
            grid_size = [
                int_ceil(gs, bs) for gs, bs in zip(grid_size, block_size)
            ]

            return grid_size, block_size, False

        # Find all thread-block maps to determine overall block size
        block_size = [1, 1, 1]
        detected_block_sizes = [block_size]
        for tbmap in tb_maps:
            tbsize = tbmap.range.size()[::-1]

            # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
            # The partial trailing thread-block is emitted as an if-condition
            # that returns on some of the participating threads
            tbsize = [symbolic.overapproximate(s) for s in tbsize]

            # Linearize (flatten) rest of dimensions to third
            if len(tbsize) > 3:
                tbsize[2] = functools.reduce(sympy.mul.Mul, tbsize[2:], 1)
                del tbsize[3:]

            # Extend to 3 dimensions if necessary
            tbsize = tbsize + [1] * (len(block_size) - len(tbsize))

            block_size = [
                sympy.Max(sz, bbsz) for sz, bbsz in zip(block_size, tbsize)
            ]
            if block_size != tbsize:
                detected_block_sizes.append(tbsize)

        # TODO: If grid/block sizes contain elements only defined within the
        #       kernel, raise an invalid SDFG exception and recommend
        #       overapproximation.

        return grid_size, block_size, True

    def generate_kernel_scope(
            self, sdfg: SDFG, dfg_scope: ScopeSubgraphView, state_id: int,
            kernel_map: nodes.Map, kernel_name: str, grid_dims: list,
            block_dims: list, has_tbmap: bool, kernel_params: list,
            function_stream: CodeIOStream, kernel_stream: CodeIOStream):
        node = dfg_scope.source_nodes()[0]

        if not node.map.flatten:
            # Add more opening braces for scope exit to close
            for dim in range(len(node.map.range) - 1):
                kernel_stream.write('{\n', sdfg, state_id, node)

        # Generate all index arguments for kernel grid
        krange = subsets.Range(kernel_map.range[::-1])
        kdims = krange.size()
        dsym = [
            symbolic.symbol('__DAPB%d' % i, nonnegative=True, integer=True)
            for i in range(len(krange))
        ]
        bidx = krange.coord_at(dsym)

        # First three dimensions are evaluated directly
        for i in range(min(len(krange), 3)):
            varname = kernel_map.params[-i - 1]

            # Delinearize third dimension if necessary
            if i == 2 and len(krange) > 3:
                block_expr = '(blockIdx.z / (%s))' % _topy(
                    functools.reduce(sympy.mul.Mul, kdims[3:], 1))
            else:
                block_expr = 'blockIdx.%s' % _named_idx(i)
                # If we defaulted to 32 threads per block, offset by thread ID
                if not has_tbmap:
                    block_expr = '(%s * %s + threadIdx.%s)' % (
                        block_expr, _topy(block_dims[i]), _named_idx(i))

            expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)

            kernel_stream.write('int %s = %s;' % (varname, expr), sdfg,
                                state_id, node)
            self._dispatcher.defined_vars.add(varname, DefinedType.Scalar)

        # Delinearize beyond the third dimension
        if len(krange) > 3:
            for i in range(3, len(krange)):
                varname = kernel_map.params[-i - 1]
                # true dim i = z / ('*'.join(kdims[i+1:])) % kdims[i]
                block_expr = '(blockIdx.z / (%s)) %% (%s)' % (
                    _topy(functools.reduce(sympy.mul.Mul, kdims[i + 1:], 1)),
                    _topy(kdims[i]),
                )

                expr = _topy(bidx[i]).replace('__DAPB%d' % i, block_expr)
                kernel_stream.write('int %s = %s;' % (varname, expr), sdfg,
                                    state_id, node)
                self._dispatcher.defined_vars.add(varname, DefinedType.Scalar)

        # Dispatch internal code
        assert self._in_device_code == False
        self._in_device_code = True
        self._block_dims = block_dims

        # Emit internal array allocation (deallocation handled at MapExit)
        scope_entry = dfg_scope.source_nodes()[0]
        to_allocate = dace.sdfg.local_transients(sdfg, dfg_scope, scope_entry)
        allocated = set()
        for child in dfg_scope.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg_scope, state_id,
                                               child, function_stream,
                                               kernel_stream)
            self._dispatcher.dispatch_initialize(sdfg, dfg_scope, state_id,
                                                 child, function_stream,
                                                 kernel_stream)

        # Generate conditions for this block's execution using min and max
        # element, e.g., skipping out-of-bounds threads in trailing block
        if has_tbmap == False:
            dsym_end = [d + bs - 1 for d, bs in zip(dsym, self._block_dims)]
            minels = krange.min_element()
            maxels = krange.max_element()
            for i, (v, minel, maxel) in enumerate(
                    zip(kernel_map.params[::-1], minels, maxels)):
                condition = ''

                # Optimize conditions if they are always true
                if i >= 3 or (dsym[i] >= minel) != True:
                    condition += '%s >= %s' % (v, _topy(minel))
                if i >= 3 or (dsym_end[i] < maxel) != False:
                    if len(condition) > 0:
                        condition += ' && '
                    condition += '%s < %s' % (v, _topy(maxel + 1))
                if len(condition) > 0:
                    kernel_stream.write('if (%s) {' % condition, sdfg,
                                        state_id, scope_entry)
                else:
                    kernel_stream.write('{', sdfg, state_id, scope_entry)

        self._dispatcher.dispatch_subgraph(
            sdfg,
            dfg_scope,
            state_id,
            function_stream,
            kernel_stream,
            skip_entry_node=True)

        if has_tbmap == False:
            for _ in kernel_map.params:
                kernel_stream.write('}\n', sdfg, state_id, node)

        self._block_dims = None
        self._in_device_code = False

    def get_next_scope_entries(self, dfg, scope_entry):
        parent_scope_entry = dfg.scope_dict()[scope_entry]
        # We're in a nested SDFG, use full graph
        if parent_scope_entry is None:
            parent_scope = dfg
        else:
            parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all non-sequential scopes from the same level
        all_scopes = [
            node for node in parent_scope.topological_sort(scope_entry)
            if isinstance(node, nodes.EntryNode)
            and node.map.schedule != types.ScheduleType.Sequential
        ]

        # TODO: Fix to include *next* scopes, without concurrent scopes

        return all_scopes[all_scopes.index(scope_entry) + 1:]

    def generate_devicelevel_scope(self, sdfg, dfg_scope, state_id,
                                   function_stream, callsite_stream):
        # Sanity check
        assert self._in_device_code == True

        dfg = sdfg.nodes()[state_id]
        sdict = dfg.scope_dict()
        scope_entry = dfg_scope.source_nodes()[0]
        scope_map = scope_entry.map
        next_scopes = self.get_next_scope_entries(dfg, scope_entry)

        if scope_map.schedule == types.ScheduleType.GPU_ThreadBlock_Dynamic:
            if len(scope_map.params) > 1:
                raise ValueError('Only one-dimensional maps are supported for '
                                 'dynamic block map schedule (got %d)' % len(
                                     scope_map.params))
            total_block_size = 1
            for bdim in self._block_dims:
                if symbolic.issymbolic(bdim, sdfg.constants):
                    raise ValueError(
                        'Block size has to be constant for block-wide '
                        'dynamic map schedule (got %s)' % str(bdim))
                total_block_size *= bdim
            if _expr(scope_map.range[0][2]) != 1:
                raise NotImplementedError(
                    'Skip not implemented for dynamic thread-block map schedule'
                )

            ##### TODO (later): Generalize
            # Find thread-block param map and its name
            if self._block_dims[1] != 1 or self._block_dims[2] != 1:
                raise NotImplementedError(
                    'Dynamic block map schedule only '
                    'implemented for 1D blocks currently')
            pscope = sdict[scope_entry]
            while pscope is not None and pscope.map.schedule != types.ScheduleType.GPU_ThreadBlock:
                pscope = sdict[pscope]
            if pscope is None:
                raise NotImplementedError('Dynamic block map schedule '
                                          'currently requires block map')
            bname = pscope.map.params[0]

            callsite_stream.write(
                'dace::DynamicMap<{bsize}>::template '
                'schedule({begin}, {end}, {tid}, [&](auto {param}, '
                'auto {tid}) {{'.format(
                    bsize=total_block_size,
                    begin=scope_map.range[0][0],
                    end=scope_map.range[0][1] + 1,
                    param=scope_map.params[0],
                    tid=bname), sdfg, state_id, scope_entry)
        else:
            # If integer sets are used, only emit one opening curly brace
            if scope_map.flatten:
                callsite_stream.write('{', sdfg, state_id, scope_entry)
            else:
                for dim in range(len(scope_map.range)):
                    callsite_stream.write('{', sdfg, state_id, scope_entry)

        # Emit internal array allocation (deallocation handled at MapExit)
        to_allocate = dace.sdfg.local_transients(sdfg, dfg_scope, scope_entry)
        allocated = set()
        for child in dfg_scope.scope_dict(node_to_children=True)[scope_entry]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg_scope, state_id,
                                               child, function_stream,
                                               callsite_stream)
            self._dispatcher.dispatch_initialize(sdfg, dfg_scope, state_id,
                                                 child, function_stream,
                                                 callsite_stream)

        # Generate all index arguments for block
        if scope_map.schedule == types.ScheduleType.GPU_ThreadBlock:
            brange = subsets.Range(scope_map.range[::-1])
            kdims = brange.size()
            dsym = [
                symbolic.symbol(
                    '__DAPT%d' % i, nonnegative=True, integer=True)
                for i in range(len(brange))
            ]
            dsym_end = [d + bs - 1 for d, bs in zip(dsym, self._block_dims)]
            tidx = brange.coord_at(dsym)

            # First three dimensions are evaluated directly
            for i in range(min(len(brange), 3)):
                varname = scope_map.params[-i - 1]

                # Delinearize third dimension if necessary
                if i == 2 and len(brange) > 3:
                    block_expr = '(threadIdx.z / (%s))' % _topy(
                        functools.reduce(sympy.mul.Mul, kdims[3:], 1))
                else:
                    block_expr = 'threadIdx.%s' % _named_idx(i)

                expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                callsite_stream.write('int %s = %s;' % (varname, expr), sdfg,
                                      state_id, scope_entry)
                self._dispatcher.defined_vars.add(varname, DefinedType.Scalar)

            # Delinearize beyond the third dimension
            if len(brange) > 3:
                for i in range(3, len(brange)):
                    varname = scope_map.params[-i - 1]
                    # true dim i = z / ('*'.join(kdims[i+1:])) % kdims[i]
                    block_expr = '(threadIdx.z / (%s)) %% (%s)' % (
                        _topy(
                            functools.reduce(sympy.mul.Mul, kdims[i + 1:], 1)),
                        _topy(kdims[i]),
                    )

                    expr = _topy(tidx[i]).replace('__DAPT%d' % i, block_expr)
                    callsite_stream.write('int %s = %s;' % (varname, expr),
                                          sdfg, state_id, scope_entry)
                    self._dispatcher.defined_vars.add(varname,
                                                      DefinedType.Scalar)

            # Generate conditions for this block's execution using min and max
            # element, e.g. skipping out-of-bounds threads in trailing block
            minels = brange.min_element()
            maxels = brange.max_element()
            for i, (v, minel, maxel) in enumerate(
                    zip(scope_map.params[::-1], minels, maxels)):
                condition = ''

                # Optimize conditions if they are always true
                if i >= 3 or (dsym[i] >= minel) != True:
                    condition += '%s >= %s' % (v, _topy(minel))
                if i >= 3 or (dsym_end[i] < maxel) != False:
                    if len(condition) > 0:
                        condition += ' && '
                    condition += '%s < %s' % (v, _topy(maxel + 1))
                if len(condition) > 0:
                    callsite_stream.write('if (%s) {' % condition, sdfg,
                                          state_id, scope_entry)
                else:
                    callsite_stream.write('{', sdfg, state_id, scope_entry)
        ##########################################################

        # Generate contents normally
        self._dispatcher.dispatch_subgraph(
            sdfg,
            dfg_scope,
            state_id,
            function_stream,
            callsite_stream,
            skip_entry_node=True)

        # If there are any other threadblock maps down the road,
        # synchronize the thread-block / grid
        if len(next_scopes) > 0:
            # Thread-block synchronization
            if scope_entry.map.schedule == types.ScheduleType.GPU_ThreadBlock:
                callsite_stream.write('    __syncthreads();\n', sdfg, state_id,
                                      scope_entry)
            # Grid synchronization (kernel fusion)
            elif scope_entry.map.schedule == types.ScheduleType.GPU_Device:
                callsite_stream.write('    __gbar.Sync();\n', sdfg, state_id,
                                      scope_entry)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        if CUDACodeGen.node_dispatch_predicate(sdfg, node):
            # Dynamically obtain node generator according to class name
            gen = getattr(self, '_generate_' + type(node).__name__)
            gen(sdfg, dfg, state_id, node, function_stream, callsite_stream)
            return

        if not self._in_device_code:
            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)
            return

        self._locals.clear_scope(self._code_state.indentation + 1)

        if self._in_device_code and isinstance(node, nodes.MapExit):
            return  # skip

        self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream)

    def _generate_NestedSDFG(self, sdfg, dfg, state_id, node, function_stream,
                             callsite_stream):
        self._toplevel_schedule = node.schedule
        self._cpu_codegen._generate_NestedSDFG(
            sdfg, dfg, state_id, node, function_stream, callsite_stream)

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        if node.map.schedule == types.ScheduleType.GPU_ThreadBlock:
            # Close block invocation conditions
            for i in range(len(node.map.params)):
                callsite_stream.write('}', sdfg, state_id, node)
        elif node.map.schedule == types.ScheduleType.GPU_ThreadBlock_Dynamic:
            # Close lambda function
            callsite_stream.write('});', sdfg, state_id, node)
            return

        self._cpu_codegen._generate_MapExit(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

    def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        # Try to autodetect reduction type
        redtype = operations.detect_reduction_type(node.wcr)
        schedule = node.schedule
        node_id = dfg.node_id(node)
        idstr = '{sdfg}_{state}_{node}'.format(
            sdfg=sdfg.name, state=state_id, node=node_id)

        output_edge = dfg.out_edges(node)[0]
        output_memlet = output_edge.data
        output_type = 'dace::vec<%s, %s>' % (
            sdfg.arrays[output_memlet.data].dtype.ctype, output_memlet.veclen)

        if node.identity is None:
            raise ValueError('For GPU reduce nodes, initial value must be '
                             'defined')

        # Create a functor or use an existing one for reduction
        if redtype == types.ReductionType.Custom:
            body, arg1, arg2 = unparse_cr_split(node.wcr)
            self._globalcode.write(
                """
        struct __reduce_{id} {{
            template <typename T>
            DACE_HDFI T operator()(const T &{arg1}, const T &{arg2}) const {{
                {contents}
            }}
        }};""".format(id=idstr, arg1=arg1, arg2=arg2, contents=body), sdfg,
                state_id, node_id)
            reduce_op = ', __reduce_' + idstr + '(), ' + _topy(node.identity)
        elif redtype in _SPECIAL_RTYPES:
            reduce_op = ''
        else:
            credtype = 'dace::ReductionType::' + str(
                redtype)[str(redtype).find('.') + 1:]
            reduce_op = (
                (', dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)) +
                ', ' + _topy(node.identity))

        # Obtain some SDFG-related information
        input_data = dfg.memlet_path(dfg.in_edges(node)[0])[0].src
        output_data = dfg.memlet_path(dfg.out_edges(node)[0])[-1].dst
        input_memlet = dfg.in_edges(node)[0].data
        reduce_shape = input_memlet.subset.bounding_box_size()
        num_items = ' * '.join([_topy(s) for s in reduce_shape])
        input = (input_memlet.data + ' + ' + cpp_array_expr(
            sdfg, input_memlet, with_brackets=False))
        output = (output_memlet.data + ' + ' + cpp_array_expr(
            sdfg, output_memlet, with_brackets=False))

        # Options: Device-wide reduction (even from device code),
        #          block-wide reduction, sequential reduction (for loop)
        if node.schedule == types.ScheduleType.GPU_Device:
            # Verify that data is on the GPU
            if input_data.desc(sdfg).storage not in [
                    types.StorageType.GPU_Global, types.StorageType.CPU_Pinned
            ]:
                raise ValueError('Input of GPU reduction must either reside '
                                 ' in global GPU memory or pinned CPU memory')
            if output_data.desc(sdfg).storage not in [
                    types.StorageType.GPU_Global, types.StorageType.CPU_Pinned
            ]:
                raise ValueError('Output of GPU reduction must either reside '
                                 ' in global GPU memory or pinned CPU memory')

            # TODO(later): Enable device-wide reduction from device through
            # CUDA dynamic parallelism. It is disabled right now
            # due to temporary memory allocation (which needs to be done
            # on the host).
            if self._in_device_code:
                raise NotImplementedError('Device-wide reduction can only be'
                                          ' run on non-GPU code.')

            # Determine reduction type
            kname = (_SPECIAL_RTYPES[redtype]
                     if redtype in _SPECIAL_RTYPES else 'Reduce')

            # Create temp memory for this GPU
            self._globalcode.write(
                """
                void *__cub_storage_{sdfg}_{state}_{node} = NULL;
                size_t __cub_ssize_{sdfg}_{state}_{node} = 0;
            """.format(sdfg=sdfg.name, state=state_id, node=node_id), sdfg,
                state_id, node)

            # Call CUB to get the storage size, allocate and free it
            self.scope_entry_stream.write(
                """
                cub::DeviceReduce::{kname}(nullptr, __cub_ssize_{sdfg}_{state}_{node},
                                          ({intype}*)nullptr, ({outtype}*)nullptr, {num_items}{redop});
                cudaMalloc(&__cub_storage_{sdfg}_{state}_{node}, __cub_ssize_{sdfg}_{state}_{node});
""".format(sdfg=sdfg.name,
            state=state_id,
            node=node_id,
            num_items=num_items,
            redop=reduce_op,
            intype=input_data.desc(sdfg).dtype.ctype,
            outtype=output_data.desc(sdfg).dtype.ctype,
            kname=kname), sdfg, state_id, node)

            self.scope_exit_stream.write(
                'cudaFree(__cub_storage_{sdfg}_{state}_{node});'.format(
                    sdfg=sdfg.name, state=state_id, node=node_id), sdfg,
                state_id, node)

            max_streams = Config.get('compiler', 'cuda',
                                     'max_concurrent_streams')
            if max_streams >= 0:
                cudastream = 'dace::cuda::__streams[%d]' % node._cuda_stream
            else:
                cudastream = 'nullptr'

            # Write reduction function definition
            self._localcode.write(
                """
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output,
                                      size_t num_items);
void __dace_reduce_{id}({intype} *input, {outtype} *output, size_t num_items)
{{
    cub::DeviceReduce::{kname}(__cub_storage_{id}, __cub_ssize_{id},
                                input, output, num_items{redop}, {stream});
}}
            """.format(
                    id=idstr,
                    intype=input_data.desc(sdfg).dtype.ctype,
                    outtype=output_data.desc(sdfg).dtype.ctype,
                    kname=kname,
                    redop=reduce_op,
                    stream=cudastream), sdfg, state_id, node)

            # Write reduction function definition in caller file
            function_stream.write(
                """
DACE_EXPORTED void __dace_reduce_{id}({intype} *input, {outtype} *output,
                                      size_t num_items);
            """.format(
                    id=idstr,
                    intype=input_data.desc(sdfg).dtype.ctype,
                    outtype=output_data.desc(sdfg).dtype.ctype), sdfg,
                state_id, node)

            # Call reduction function where necessary
            input_dims = input_memlet.subset.dims()
            output_dims = output_memlet.subset.data_dims()
            if (node.axes is None or len(node.axes) == input_dims):
                callsite_stream.write(
                    '__dace_reduce_{id}({input}, {output}, {num_items});'
                    .format(
                        id=idstr,
                        input=input,
                        output=output,
                        num_items=num_items), sdfg, state_id, node)
            else:
                raise NotImplementedError(
                    'Multiple axis reductions not supported on GPUs. Please '
                    'apply ReduceExpansion')
                # Generate for loops around CUB calls and properly offset input
                # and output arrays
                #for axis in range(output_dims):
                #    if axis not in node.axes:
                #        callsite_stream.write(
                #            'for (int {var} = {begin}; {var} < {end}; {var} += {skip}) {{'.
                #            format(
                #                var='__o%d' % axis,
                #                begin=output_subset[axis][0],
                #                end=output_subset[axis][1] + 1,
                #                skip=output_subset[axis][2]), sdfg, state_id, node)
                #
                ### Obtain variable names per output and reduction axis
                #axis_vars = []
                #octr = 0
                #for d in range(input_dims):
                #    if d not in axes:
                #        axis_vars.append('__o%d' % octr)
                #        octr += 1
                #
                #input = (input_memlet.data.name + ' + ' + cpp_array_expr(
                #            sdfg, input_memlet, with_brackets=False))
                #output = (output_memlet.data.name + ' + ' + cpp_array_expr(
                #            sdfg, output_memlet, with_brackets=False))
                #num_items =
                #
                #callsite_stream.write(
                #    '__dace_reduce_{id}({input}, {output}, {num_items});'
                #    .format(
                #        id=idstr,
                #        input=input,
                #        output=output,
                #        num_items=num_items), sdfg, state_id, node)
                #
                ##cpp_array_expr(sdfg,
                ##            output_memlet,
                ##            offset=['__o%d' % i for i in range(output_dims)],
                ##            relative_offset=False))
                ##        invar = cpp_array_expr(sdfg,
                ##            input_memlet, offset=axis_vars, relative_offset=False)
                #for axis in range(output_dims):
                #    callsite_stream.write('}\n', sdfg, state_id, node)
            return

        # Block-wide reduction
        elif node.schedule == types.ScheduleType.GPU_ThreadBlock:
            # Checks
            if not self._in_device_code:
                raise ValueError('Block-wide GPU reduction must occur within'
                                 ' a GPU kernel')
            for bdim in self._block_dims:
                if symbolic.issymbolic(bdim, sdfg.constants):
                    raise ValueError(
                        'Block size has to be constant for block-wide '
                        'reduction (got %s)' % str(bdim))
            if (node.axes is not None and len(node.axes) < input_dims):
                raise ValueError(
                    'Only full reduction is supported for block-wide reduce,'
                    ' please use ReduceExpansion')
            if (input_data.desc(sdfg).storage != types.StorageType.GPU_Stack
                    or output_data.desc(sdfg).storage !=
                    types.StorageType.GPU_Stack):
                raise ValueError(
                    'Block-wise reduction only supports GPU register inputs '
                    'and outputs')
            if redtype in _SPECIAL_RTYPES:
                raise ValueError('%s block reduction not supported' % redtype)

            credtype = 'dace::ReductionType::' + str(
                redtype)[str(redtype).find('.') + 1:]
            if redtype == types.ReductionType.Custom:
                redop = '__reduce_%s()' % idstr
            else:
                redop = 'dace::_wcr_fixed<%s, %s>()' % (credtype, output_type)

            # Allocate shared memory for block reduce
            self.scope_entry_stream.write(
                """
            typedef cub::BlockReduce<{type}, {numthreads}> BlockReduce_{id};
            __shared__ typename BlockReduce_{id}::TempStorage temp_storage_{id};
                """.format(
                    id=idstr,
                    type=output_data.desc(sdfg).dtype.ctype,
                    numthreads=' * '.join(str(s) for s in self._block_dims)),
                sdfg, state_id, node)

            # TODO(later): If less than the whole block is participating,
            #              use special CUB function
            output = cpp_array_expr(sdfg, output_memlet)
            callsite_stream.write(
                """
                {output} = BlockReduce_{id}(temp_storage_{id}).Reduce({input}, {redop});
                """.format(
                    id=idstr,
                    redop=redop,
                    input=input_memlet.data,
                    output=output), sdfg, state_id, node)

            return
        # Sequential goes to CPU generator
        elif node.schedule == types.ScheduleType.Sequential:
            self._cpu_codegen._generate_Reduce(
                sdfg, dfg, state_id, node, function_stream, callsite_stream)
            return
        else:
            raise ValueError(
                'Unsupported reduction schedule %s' % str(node.schedule))


########################################################################
########################################################################
########################################################################
########################################################################
# Helper functions and classes


def unparse_cr_split(wcr_ast):
    """ Parses various types of WCR functions, returning a 3-tuple of body,
        first argument name and second argument name. """
    if isinstance(wcr_ast, ast.FunctionDef):
        return (cppunparse.cppunparse(wcr_ast.body, expr_semicolon=False),
                wcr_ast.args.args[0].arg, wcr_ast.args.args[1].arg)
    elif isinstance(wcr_ast, ast.Lambda):
        return (('return (' + cppunparse.cppunparse(
            wcr_ast.body, expr_semicolon=False) + ');'),
                wcr_ast.args.args[0].arg, wcr_ast.args.args[1].arg)
    elif isinstance(wcr_ast, ast.Module):
        return unparse_cr_split(wcr_ast.body[0].value)
    elif isinstance(wcr_ast, str):
        return unparse_cr_split(LambdaProperty.from_string(wcr_ast))
    else:
        raise NotImplementedError('INVALID TYPE OF WCR: ' +
                                  type(wcr_ast).__name__)


def _topy(arr):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d)) for d in arr]


def _named_idx(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx > 2:
        raise ValueError('idx must be between 0 and 2, got %d' % idx)
    return ('x', 'y', 'z')[idx]


def _get_storagename(storage):
    """ Returns a string containing the name of the storage location.
        Example: types.StorageType.GPU_Shared will return "Shared". """
    sname = str(storage)
    return sname[sname.rindex('_') + 1:]
