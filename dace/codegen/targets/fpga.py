from six import StringIO
import collections
import functools
import itertools
import re
import sympy as sp
import numpy as np

import dace
from dace import subsets
from dace.config import Config
from dace.frontend import operations
from dace.graph import nodes
from dace.sdfg import ScopeSubgraphView, find_input_arraynode, find_output_arraynode
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator, DefinedType
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute, DefinedType)
from dace.codegen.targets.cpu import cpp_offset_expr, cpp_array_expr
from dace.codegen.targets import cpu
from dace.codegen import cppunparse
from dace.properties import Property, make_properties, indirect_properties


class FPGACodeGen(TargetCodeGenerator):
    # Set by deriving class
    target_name = None
    title = None
    language = None

    def __init__(self, frame_codegen, sdfg):

        # The inheriting class must set target_name, title and language.

        self._in_device_code = False
        self._cpu_codegen = None
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher

        self._global_sdfg = sdfg
        self._program_name = sdfg.name

        # Verify that we did not miss the allocation of any global arrays, even
        # if they're nested deep in the SDFG
        self._allocated_global_arrays = set()
        self._unrolled_pes = set()

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        self._host_codes = []
        self._kernel_codes = []

        # Register additional FPGA dispatchers
        self._dispatcher.register_map_dispatcher(
            [dace.dtypes.ScheduleType.FPGA_Device], self)

        self._dispatcher.register_state_dispatcher(
            self,
            predicate=lambda sdfg, state: len(state.data_nodes()) > 0 and all([
                n.desc(sdfg).storage in [
                    dace.dtypes.StorageType.FPGA_Global, dace.dtypes.
                    StorageType.FPGA_Local, dace.dtypes.StorageType.
                    FPGA_Registers
                ] for n in state.data_nodes()
            ]))

        self._dispatcher.register_node_dispatcher(
            self, predicate=lambda *_: self._in_device_code)

        fpga_storage = [
            dace.dtypes.StorageType.FPGA_Global,
            dace.dtypes.StorageType.FPGA_Local,
            dace.dtypes.StorageType.FPGA_Registers,
        ]
        self._dispatcher.register_array_dispatcher(fpga_storage, self)

        # Register permitted copies
        for storage_from in itertools.chain(
                fpga_storage, [dace.dtypes.StorageType.Register]):
            for storage_to in itertools.chain(
                    fpga_storage, [dace.dtypes.StorageType.Register]):
                if (storage_from == dace.dtypes.StorageType.Register
                        and storage_to == dace.dtypes.StorageType.Register):
                    continue
                self._dispatcher.register_copy_dispatcher(
                    storage_from, storage_to, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.FPGA_Global,
            dace.dtypes.StorageType.CPU_Heap, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.FPGA_Global,
            dace.dtypes.StorageType.CPU_Stack, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.CPU_Heap,
            dace.dtypes.StorageType.FPGA_Global, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.CPU_Stack,
            dace.dtypes.StorageType.FPGA_Global, None, self)

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return False

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        """Generate a kernel that runs all connected components within a state
           as concurrent dataflow modules."""

        state_id = sdfg.node_id(state)

        # Determine independent components
        subgraphs = dace.sdfg.concurrent_subgraphs(state)

        # Generate kernel code
        shared_transients = set(sdfg.shared_transients())
        if not self._in_device_code:
            # Allocate global memory transients, unless they are shared with
            # other states
            all_transients = set(state.all_transients())
            allocated = set(shared_transients)
            for node in state.data_nodes():
                data = node.desc(sdfg)
                if node.data not in all_transients or node.data in allocated:
                    continue
                if data.storage != dace.dtypes.StorageType.FPGA_Global:
                    continue
                allocated.add(node.data)
                self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                                   function_stream,
                                                   callsite_stream)
                self._dispatcher.dispatch_initialize(sdfg, state, state_id,
                                                     node, function_stream,
                                                     callsite_stream)
            # Generate kernel code
            self.generate_kernel(sdfg, state, state.label, subgraphs,
                                 function_stream, callsite_stream)
        else:  # self._in_device_code == True
            to_allocate = dace.sdfg.local_transients(sdfg, state, None)
            allocated = set()
            for node in state.data_nodes():
                data = node.desc(sdfg)
                if node.data not in to_allocate or node.data in allocated:
                    continue
                # Make sure there are no global transients in the nested state
                # that are thus not gonna be allocated
                if data.storage == dace.dtypes.StorageType.FPGA_Global:
                    raise dace.codegen.codegen.CodegenError(
                        "Cannot allocate global memory from device code.")
                allocated.add(data)
                # Allocate transients
                self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                                   function_stream,
                                                   callsite_stream)
                self._dispatcher.dispatch_initialize(sdfg, state, state_id,
                                                     node, function_stream,
                                                     callsite_stream)
            self.generate_nested_state(sdfg, state, state.label, subgraphs,
                                       function_stream, callsite_stream)

    @staticmethod
    def shared_data(subgraphs):
        """Returns a set of data objects that are shared between two or more of
           the specified subgraphs."""
        shared = set()
        if len(subgraphs) >= 2:
            seen = {}
            for sg in subgraphs:
                for node in sg:
                    if isinstance(node, dace.graph.nodes.AccessNode):
                        if node.data in seen:
                            if seen[node.data] != sg:
                                shared.add(node.data)
                        else:
                            seen[node.data] = sg
        return shared

    @staticmethod
    def global_transient_nodes(subgraphs):
        """Generator that returns all transient global arrays nested in the
           passed subgraphs on the form (is_output, AccessNode)"""
        seen = set()
        for subgraph in subgraphs:
            for n, scope in subgraph.all_nodes_recursive():
                if (isinstance(n, dace.graph.nodes.AccessNode)
                        and n.desc(sdfg).transient and n.desc(sdfg).storage ==
                        dace.dtypes.StorageType.FPGA_Global):
                    if n.data in seen:
                        continue
                    seen.add(n.data)
                    if scope.out_degree(n) > 0:
                        yield (False, n)
                    if scope.in_degree(n) > 0:
                        yield (True, n)

    @classmethod
    def make_parameters(cls, sdfg, state, subgraphs):
        """Determines the parameters that must be passed to the passed list of
           subgraphs, as well as to the global kernel."""

        # Get a set of data nodes that are shared across subgraphs
        shared_data = cls.shared_data(subgraphs)

        # For some reason the array allocation dispatcher takes nodes, not
        # arrays. Build a dictionary of arrays to arbitrary data nodes
        # referring to them.
        data_to_node = {}

        global_data_parameters = []
        global_data_names = set()
        top_level_local_data = []
        subgraph_parameters = collections.OrderedDict()  # {subgraph: [params]}
        nested_global_transients = []
        nested_global_transients_seen = set()
        for subgraph in subgraphs:
            data_to_node.update({
                node.data: node
                for node in subgraph.nodes()
                if isinstance(node, dace.graph.nodes.AccessNode)
            })
            subsdfg = subgraph.parent
            candidates = []  # type: List[Tuple[bool,str,Data]]
            # [(is an output, dataname string, data object)]
            for n in subgraph.source_nodes():
                candidates += [(False, e.data.data,
                                subsdfg.arrays[e.data.data])
                               for e in state.in_edges(n)]
            for n in subgraph.sink_nodes():
                candidates += [(True, e.data.data, subsdfg.arrays[e.data.data])
                               for e in state.out_edges(n)]
            # Find other data nodes that are used internally
            for n, scope in subgraph.all_nodes_recursive():
                if isinstance(n, dace.graph.nodes.AccessNode):
                    # Add nodes if they are outer-level, or an inner-level
                    # transient (inner-level inputs/outputs are just connected
                    # to data in the outer layers, whereas transients can be
                    # independent).
                    if scope == subgraph or n.desc(scope).transient:
                        if scope.out_degree(n) > 0:
                            candidates.append((False, n.data, n.desc(scope)))
                        if scope.in_degree(n) > 0:
                            candidates.append((True, n.data, n.desc(scope)))
                        if scope != subgraph:
                            if (isinstance(n.desc(scope), dace.data.Array)
                                    and n.desc(scope).storage ==
                                    dace.dtypes.StorageType.FPGA_Global and
                                    n.data not in nested_global_transients_seen
                                ):
                                nested_global_transients.append(n)
                            nested_global_transients_seen.add(n.data)
            subgraph_parameters[subgraph] = []
            # Differentiate global and local arrays. The former are allocated
            # from the host and passed to the device code, while the latter are
            # (statically) allocated on the device side.
            for is_output, dataname, data in candidates:
                if (isinstance(data, dace.data.Array)
                        or isinstance(data, dace.data.Scalar)
                        or isinstance(data, dace.data.Stream)):
                    if data.storage == dace.dtypes.StorageType.FPGA_Global:
                        subgraph_parameters[subgraph].append((is_output,
                                                              dataname, data))
                        if is_output:
                            global_data_parameters.append((is_output, dataname,
                                                           data))
                        else:
                            global_data_parameters.append((is_output, dataname,
                                                           data))
                        global_data_names.add(dataname)
                    elif (data.storage == dace.dtypes.StorageType.FPGA_Local
                          or data.storage ==
                          dace.dtypes.StorageType.FPGA_Registers):
                        if dataname in shared_data:
                            # Only transients shared across multiple components
                            # need to be allocated outside and passed as
                            # parameters
                            subgraph_parameters[subgraph].append(
                                (is_output, dataname, data))
                            # Resolve the data to some corresponding node to be
                            # passed to the allocator
                            top_level_local_data.append(dataname)
                    else:
                        raise ValueError("Unsupported storage type: {}".format(
                            data.storage))
                else:
                    raise TypeError("Unsupported data type: {}".format(
                        type(data).__name__))
            subgraph_parameters[subgraph] = dace.dtypes.deduplicate(
                subgraph_parameters[subgraph])

        # Deduplicate
        global_data_parameters = dace.dtypes.deduplicate(
            global_data_parameters)
        top_level_local_data = dace.dtypes.deduplicate(top_level_local_data)
        top_level_local_data = [data_to_node[n] for n in top_level_local_data]

        scalar_parameters = sdfg.scalar_parameters(False)
        symbol_parameters = sdfg.undefined_symbols(False)

        return (global_data_parameters, top_level_local_data,
                subgraph_parameters, scalar_parameters, symbol_parameters,
                nested_global_transients)

    def generate_nested_state(self, sdfg, state, nest_name, subgraphs,
                              function_stream, callsite_stream):

        for sg in subgraphs:
            self._dispatcher.dispatch_subgraph(
                sdfg,
                sg,
                sdfg.node_id(state),
                function_stream,
                callsite_stream,
                skip_entry_node=False)

    @staticmethod
    def detect_memory_widths(subgraphs):
        # For each memory, checks that all the memlets are consistent (they have the same width).
        # This allow us to instantiate to generate data paths with a single data size throughout the subgraph.
        stack = []
        for sg in subgraphs:
            stack += [(n, sg) for n in sg.nodes()]
        memory_widths = {}
        seen = set()
        while len(stack) > 0:
            node, graph = stack.pop()
            if isinstance(node, dace.graph.nodes.NestedSDFG):
                for state in node.sdfg.states():
                    stack += [(n, state) for n in state.nodes()]
            elif isinstance(node, dace.graph.nodes.AccessNode):
                if node in seen:
                    continue
                seen.add(node)
                nodedesc = node.desc(graph)
                for edge in graph.all_edges(node):
                    if (isinstance(edge.data, dace.memlet.EmptyMemlet)
                            or edge.data.data is None):
                        continue
                    if node.data not in memory_widths:
                        if (isinstance(nodedesc, dace.data.Stream)
                                and nodedesc.veclen != edge.data.veclen):
                            raise ValueError(
                                "Vector length on memlet {} ({}) doesn't "
                                "match vector length of {} ({})".format(
                                    edge.data, edge.data.veclen, node.data,
                                    nodedesc.veclen))
                        memory_widths[node.data] = edge.data.veclen
                    else:
                        if memory_widths[node.data] != edge.data.veclen:
                            raise dace.codegen.codegen.CodegenError(
                                "Inconsistent vector length "
                                "on FPGA for \"{}\": got {}, had {}".format(
                                    node.data, edge.data.veclen,
                                    memory_widths[node.data]))
        return memory_widths

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):

        if not self._in_device_code:
            # If we're not already generating kernel code we need to set up the
            # kernel launch
            subgraphs = [dfg_scope]
            return self.generate_kernel(
                sdfg, sdfg.find_state(state_id),
                dfg_scope.source_nodes()[0].map.label.replace(" ", "_"),
                subgraphs, function_stream, callsite_stream)

        self.generate_node(sdfg, dfg_scope, state_id,
                           dfg_scope.source_nodes()[0], function_stream,
                           callsite_stream)

        self._dispatcher.dispatch_subgraph(
            sdfg,
            dfg_scope,
            state_id,
            function_stream,
            callsite_stream,
            skip_entry_node=True)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        result = StringIO()
        nodedesc = node.desc(sdfg)
        arrsize = nodedesc.total_size
        is_dynamically_sized = dace.symbolic.issymbolic(
            arrsize, sdfg.constants)

        dataname = node.data

        if isinstance(nodedesc, dace.data.Stream):

            if not self._in_device_code:
                raise dace.codegen.codegen.CodegenError(
                    "Cannot allocate FIFO from CPU code: {}".format(node.data))

            if is_dynamically_sized:
                raise dace.codegen.codegen.CodegenError(
                    "Arrays of streams cannot have dynamic size on FPGA")

            if nodedesc.buffer_size < 1:
                raise dace.codegen.codegen.CodegenError(
                    "Streams cannot be unbounded on FPGA")

            buffer_length_dynamically_sized = (dace.symbolic.issymbolic(
                nodedesc.buffer_size, sdfg.constants))

            if buffer_length_dynamically_sized:
                raise dace.codegen.codegen.CodegenError(
                    "Buffer length of stream cannot have dynamic size on FPGA")

            if cpu.sym2cpp(arrsize) != "1":
                # Is a stream array
                self._dispatcher.defined_vars.add(dataname,
                                                  DefinedType.StreamArray)
            else:
                # Single stream
                self._dispatcher.defined_vars.add(dataname, DefinedType.Stream)

            # Language-specific implementation
            self.define_stream(nodedesc.dtype, nodedesc.veclen,
                               nodedesc.buffer_size, dataname, arrsize,
                               function_stream, result)

        elif isinstance(nodedesc, dace.data.Array):

            if nodedesc.storage == dace.dtypes.StorageType.FPGA_Global:

                if self._in_device_code:

                    if nodedesc not in self._allocated_global_arrays:
                        raise RuntimeError("Cannot allocate global array "
                                           "from device code: {} in {}".format(
                                               node.label, sdfg.name))

                else:

                    devptr_name = dataname
                    if isinstance(nodedesc, dace.data.Array):
                        # TODO: Distinguish between read, write, and read+write
                        # TODO: Handle memory banks
                        self._allocated_global_arrays.add(node.data)
                        result.write(
                            "auto {} = hlslib::ocl::GlobalContext()."
                            "MakeBuffer<{}, hlslib::ocl::Access::readWrite>"
                            "({});".format(dataname, nodedesc.dtype.ctype,
                                           cpu.sym2cpp(arrsize)))
                        self._dispatcher.defined_vars.add(
                            dataname, DefinedType.Pointer)

            elif (nodedesc.storage == dace.dtypes.StorageType.FPGA_Local or
                  nodedesc.storage == dace.dtypes.StorageType.FPGA_Registers):

                if not self._in_device_code:
                    raise dace.codegen.codegen.CodegenError(
                        "Tried to allocate local FPGA memory "
                        "outside device code: {}".format(dataname))
                if is_dynamically_sized:
                    raise ValueError(
                        "Dynamic allocation of FPGA fast memory not allowed")

                # Absorb vector size into type and adjust array size
                # accordingly
                veclen = self._memory_widths[node.data]
                generate_scalar = False
                if veclen > 1:
                    arrsize_symbolic = nodedesc.total_size
                    arrsize_eval = dace.symbolic.eval(
                        arrsize_symbolic / veclen)
                    if cpu.sym2cpp(arrsize_eval) == "1":
                        generate_scalar = True
                    arrsize_vec = "({}) / {}".format(arrsize, veclen)
                else:
                    arrsize_vec = arrsize

                # If the array degenerates to a single element because of
                # vectorization, generate the variable as a scalar instead of
                # an array of size 1
                if generate_scalar:
                    # Language-specific
                    define_str = "{} {};".format(
                        self.make_vector_type(nodedesc.dtype, veclen, False),
                        dataname)
                    callsite_stream.write(define_str, sdfg, state_id, node)
                    self._dispatcher.defined_vars.add(dataname,
                                                      DefinedType.Scalar)
                else:
                    # Language-specific
                    self.define_local_array(nodedesc.dtype, veclen, dataname,
                                            arrsize_vec, nodedesc.storage,
                                            nodedesc.shape, function_stream,
                                            result, sdfg, state_id, node)
                    self._dispatcher.defined_vars.add(dataname,
                                                      DefinedType.Pointer)

            else:
                raise NotImplementedError("Unimplemented storage type " +
                                          str(nodedesc.storage))

        elif isinstance(nodedesc, dace.data.Scalar):

            result.write("{} {};\n".format(nodedesc.dtype.ctype, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Scalar)

        else:
            raise TypeError("Unhandled data type: {}".format(
                type(nodedesc).__name__))

        callsite_stream.write(result.getvalue(), sdfg, state_id, node)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass  # Handled by destructor

    def _emit_copy(self, sdfg, state_id, src_node, src_storage, dst_node,
                   dst_storage, dst_schedule, edge, dfg, callsite_stream):

        u, v, memlet = edge.src, edge.dst, edge.data

        cpu_storage_types = [
            dace.dtypes.StorageType.CPU_Heap,
            dace.dtypes.StorageType.CPU_Stack,
            dace.dtypes.StorageType.CPU_Pinned
        ]
        fpga_storage_types = [
            dace.dtypes.StorageType.FPGA_Global,
            dace.dtypes.StorageType.FPGA_Local,
            dace.dtypes.StorageType.FPGA_Registers,
        ]

        # Determine directionality
        if isinstance(
                src_node,
                dace.graph.nodes.AccessNode) and memlet.data == src_node.data:
            outgoing_memlet = True
        elif isinstance(
                dst_node,
                dace.graph.nodes.AccessNode) and memlet.data == dst_node.data:
            outgoing_memlet = False
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        data_to_data = (isinstance(src_node, dace.graph.nodes.AccessNode)
                        and isinstance(dst_node, dace.graph.nodes.AccessNode))

        host_to_device = (data_to_data and src_storage in cpu_storage_types and
                          dst_storage == dace.dtypes.StorageType.FPGA_Global)
        device_to_host = (data_to_data and
                          src_storage == dace.dtypes.StorageType.FPGA_Global
                          and dst_storage in cpu_storage_types)
        device_to_device = (
            data_to_data and src_storage == dace.dtypes.StorageType.FPGA_Global
            and dst_storage == dace.dtypes.StorageType.FPGA_Global)

        if (host_to_device or device_to_host) and self._in_device_code:
            raise RuntimeError(
                "Cannot copy between host and device from device")

        if (host_to_device or device_to_host
                or (device_to_device and not self._in_device_code)):

            dims = memlet.subset.dims()
            copy_shape = memlet.subset.bounding_box_size()
            copysize = ' * '.join([
                cppunparse.pyexpr2cpp(dace.symbolic.symstr(s))
                for s in copy_shape
            ])
            offset = cpp_array_expr(sdfg, memlet, with_brackets=False)

            if (not sum(copy_shape) == 1
                    and (not isinstance(memlet.subset, subsets.Range)
                         or any([step != 1 for _, _, step in memlet.subset]))):
                raise NotImplementedError("Only contiguous copies currently "
                                          "supported for FPGA codegen.")

            if host_to_device:

                callsite_stream.write(
                    "{}.CopyFromHost({}, {}, {});".format(
                        dst_node.data, (offset if not outgoing_memlet else 0),
                        copysize,
                        src_node.data + (" + {}".format(offset)
                                         if outgoing_memlet else "")), sdfg,
                    state_id, [src_node, dst_node])

            elif device_to_host:

                callsite_stream.write(
                    "{}.CopyToHost({}, {}, {});".format(
                        src_node.data, (offset
                                        if outgoing_memlet else 0), copysize,
                        dst_node.data + (" + {}".format(offset)
                                         if not outgoing_memlet else "")),
                    sdfg, state_id, [src_node, dst_node])

            elif device_to_device:

                callsite_stream.write(
                    "{}.CopyToDevice({}, {}, {}, {});".format(
                        src_node.data, (offset
                                        if outgoing_memlet else 0), copysize,
                        dst_node.data, (offset if not outgoing_memlet else 0)),
                    sdfg, state_id, [src_node, dst_node])

        # Reject copying to/from local memory from/to outside the FPGA
        elif (data_to_data
              and (((src_storage == dace.dtypes.StorageType.FPGA_Local
                     or src_storage == dace.dtypes.StorageType.FPGA_Registers)
                    and dst_storage not in fpga_storage_types) or
                   ((dst_storage == dace.dtypes.StorageType.FPGA_Local
                     or dst_storage == dace.dtypes.StorageType.FPGA_Registers)
                    and src_storage not in fpga_storage_types))):
            raise NotImplementedError(
                "Copies between host memory and FPGA "
                "local memory not supported: from {} to {}".format(
                    src_node, dst_node))

        elif data_to_data:

            if memlet.wcr is not None:
                raise NotImplementedError("WCR not implemented for copy edges")

            # Try to turn into degenerate/strided ND copies
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                self._cpu_codegen.memlet_copy_to_absolute_strides(
                    sdfg, memlet, src_node, dst_node, packed_types=True))

            ctype = src_node.desc(sdfg).dtype.ctype

            # TODO: detect in which cases we shouldn't unroll
            register_to_register = (src_node.desc(
                sdfg).storage == dace.dtypes.StorageType.FPGA_Registers
                                    or dst_node.desc(sdfg).storage ==
                                    dace.dtypes.StorageType.FPGA_Registers)

            num_loops = len([dim for dim in copy_shape if dim != 1])
            if num_loops > 0:
                if not register_to_register:
                    # Language-specific
                    self.generate_pipeline_loop_pre(callsite_stream, sdfg,
                                                    state_id, dst_node)
                if len(copy_shape) > 1:
                    # Language-specific
                    self.generate_flatten_loop_pre(callsite_stream, sdfg,
                                                   state_id, dst_node)
                for node in [src_node, dst_node]:
                    if (isinstance(node.desc(sdfg), dace.data.Array)
                            and node.desc(sdfg).storage in [
                                dace.dtypes.StorageType.FPGA_Local,
                                dace.StorageType.FPGA_Registers
                            ]):
                        # Language-specific
                        self.generate_no_dependence_pre(
                            node.data, callsite_stream, sdfg, state_id,
                            dst_node)

            # Loop intro
            for i, copy_dim in enumerate(copy_shape):
                if copy_dim != 1:
                    if register_to_register:
                        # Language-specific
                        self.generate_unroll_loop_pre(callsite_stream, None,
                                                      sdfg, state_id, dst_node)
                    callsite_stream.write(
                        "for (int __dace_copy{} = 0; __dace_copy{} < {}; "
                        "++__dace_copy{}) {{".format(i, i, copy_dim, i), sdfg,
                        state_id, dst_node)
                    if register_to_register:
                        # Language-specific
                        self.generate_unroll_loop_post(
                            callsite_stream, None, sdfg, state_id, dst_node)

            # Pragmas
            if num_loops > 0:
                if not register_to_register:
                    # Language-specific
                    self.generate_pipeline_loop_post(callsite_stream, sdfg,
                                                     state_id, dst_node)
                if len(copy_shape) > 1:
                    # Language-specific
                    self.generate_flatten_loop_post(callsite_stream, sdfg,
                                                    state_id, dst_node)

            # Construct indices (if the length of the stride array is zero,
            # resolves to an empty string)
            src_index = " + ".join([
                "__dace_copy{} * {}".format(i, cpu.sym2cpp(stride))
                for i, stride in enumerate(src_strides) if copy_shape[i] != 1
            ])
            dst_index = " + ".join([
                "__dace_copy{} * {}".format(i, cpu.sym2cpp(stride))
                for i, stride in enumerate(dst_strides) if copy_shape[i] != 1
            ])

            src_def_type = self._dispatcher.defined_vars.get(src_node.data)
            dst_def_type = self._dispatcher.defined_vars.get(dst_node.data)

            pattern = re.compile("([^\s]+)(\s*\+\s*)?(.*)")

            def sanitize_index(expr, index):
                var_name, _, expr_index = re.match(pattern, expr).groups()
                index = index.strip()
                expr_index = expr_index.strip()
                if index:
                    if expr_index:
                        return var_name, index + " + " + expr_index
                    return var_name, index
                else:
                    if expr_index:
                        return var_name, expr_index
                    return var_name, "0"

            # Pull out indices from expressions
            src_expr, src_index = sanitize_index(src_expr, src_index)
            dst_expr, dst_index = sanitize_index(dst_expr, dst_index)

            # Language specific
            read_expr = self.make_read(src_def_type, ctype, src_node.label,
                                       memlet.veclen, src_expr, src_index)

            # Language specific
            write_expr = self.make_write(dst_def_type, ctype, dst_node.label,
                                         memlet.veclen, dst_expr, dst_index,
                                         read_expr, memlet.wcr)

            callsite_stream.write(write_expr)

            # Inject dependence pragmas (DACE semantics implies no conflict)
            for node in [src_node, dst_node]:
                if (isinstance(node.desc(sdfg), dace.data.Array)
                        and node.desc(sdfg).storage in [
                            dace.dtypes.StorageType.FPGA_Local,
                            dace.StorageType.FPGA_Registers
                        ]):
                    # Language-specific
                    self.generate_no_dependence_post(
                        node.data, callsite_stream, sdfg, state_id, dst_node)

            # Loop outtro
            for _ in range(num_loops):
                callsite_stream.write("}")

        else:

            self.generate_memlet_definition(sdfg, dfg, state_id, src_node,
                                            dst_node, edge, callsite_stream)

    @staticmethod
    def opencl_parameters(sdfg, kernel_parameters):
        seen = set()
        out_parameters = []
        for is_output, pname, param in kernel_parameters:
            # Since we can have both input and output versions of the same
            # array, make sure we only pass it once from the host code
            if param in seen:
                continue
            seen.add(param)
            if isinstance(param, dace.data.Array):
                out_parameters.append(
                    "hlslib::ocl::Buffer<{}, "
                    "hlslib::ocl::Access::readWrite> &{}".format(
                        param.dtype.ctype, pname))
            else:
                out_parameters.append(
                    param.signature(with_types=True, name=pname))
        return out_parameters

    def get_next_scope_entries(self, sdfg, dfg, scope_entry):
        parent_scope_entry = dfg.scope_dict()[scope_entry]
        parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all scopes from the same level
        all_scopes = [
            node for node in parent_scope.topological_sort()
            if isinstance(node, dace.graph.nodes.EntryNode)
        ]

        return all_scopes[all_scopes.index(scope_entry) + 1:]

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        method_name = "_generate_" + type(node).__name__
        # Fake inheritance... use this class' method if it exists,
        # otherwise fall back on CPU codegen
        if hasattr(self, method_name):

            if hasattr(node, "schedule") and node.schedule not in [
                    dace.dtypes.ScheduleType.Default,
                    dace.dtypes.ScheduleType.FPGA_Device
            ]:
                # raise dace.codegen.codegen.CodegenError(
                #     "Cannot produce FPGA code for {} node with schedule {}: ".
                #     format(type(node).__name__, node.schedule, node))
                print("WARNING: found schedule {} on {} node in FPGA code. "
                      "Ignoring.".format(node.schedule,
                                         type(node).__name__))

            getattr(self, method_name)(sdfg, dfg, state_id, node,
                                       function_stream, callsite_stream)
        else:
            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

    def initialize_array(self, *args, **kwargs):
        pass

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):

        if isinstance(src_node, dace.graph.nodes.CodeNode):
            src_storage = dace.dtypes.StorageType.Register
            try:
                src_parent = dfg.scope_dict()[src_node]
            except KeyError:
                src_parent = None
            dst_schedule = (None
                            if src_parent is None else src_parent.map.schedule)
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, dace.graph.nodes.CodeNode):
            dst_storage = dace.dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.scope_dict()[dst_node]
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = sdfg.nodes()[state_id]

        # Emit actual copy
        self._emit_copy(sdfg, state_id, src_node, src_storage, dst_node,
                        dst_storage, dst_schedule, edge, state_dfg,
                        callsite_stream)

    def _generate_PipelineEntry(self, *args, **kwargs):
        self._generate_MapEntry(*args, **kwargs)

    def _generate_MapEntry(self, sdfg, dfg, state_id, node, function_stream,
                           callsite_stream):

        result = callsite_stream

        scope_dict = dfg.scope_dict()
        if node.map in self._unrolled_pes:

            # This is a top-level unrolled map, meaning it has been used to
            # replicate processing elements. Don't generate anything here.
            pass

        else:
            # Add extra opening brace (dynamic map ranges, closed in MapExit
            # generator)
            callsite_stream.write('{', sdfg, state_id, node)

            # Pipeline innermost loops
            scope = dfg.scope_dict(True)[node]

            # Generate custom iterators if this is a pipelined (and thus
            # flattened) loop
            if isinstance(node, PipelineEntry):
                for i in range(len(node.map.range)):
                    result.write("long {} = {};\n".format(
                        node.map.params[i], node.map.range[i][0]))

            if node.map.unroll:
                self.generate_unroll_loop_pre(result, None, sdfg, state_id,
                                              node)
            else:
                is_innermost = not any(
                    [isinstance(x, dace.graph.nodes.EntryNode) for x in scope])
                if is_innermost:
                    self.generate_pipeline_loop_pre(result, sdfg, state_id,
                                                    node)
                    self.generate_flatten_loop_pre(result, sdfg, state_id,
                                                   node)

            # Generate nested loops
            if not isinstance(node, PipelineEntry):
                for i, r in enumerate(node.map.range):
                    var = node.map.params[i]
                    begin, end, skip = r
                    # decide type of loop variable
                    loop_var_type = "int"
                    # try to decide type of loop variable
                    try:
                        if dace.symbolic.eval(
                                begin) >= 0 and dace.symbolic.eval(skip) > 0:
                            # it could be an unsigned (uint32) variable: we need to check to the type of 'end',
                            # if we are able to determine it
                            end_type = dace.symbolic.symbol.s_types.get(
                                cpu.sym2cpp(end + 1))
                            if end_type is not None:
                                if np.dtype(end_type.dtype.type) > np.dtype(
                                        'uint32'):
                                    loop_var_type = end.ctype
                                elif np.issubdtype(
                                        np.dtype(end_type.dtype.type),
                                        np.unsignedinteger):
                                    loop_var_type = "size_t"
                    except UnboundLocalError:
                        pass

                    result.write(
                        "for ({} {} = {}; {} < {}; {} += {}) {{\n".format(
                            loop_var_type, var, cpu.sym2cpp(begin), var,
                            cpu.sym2cpp(end + 1), var, cpu.sym2cpp(skip)),
                        sdfg, state_id, node)
            else:
                pipeline = node.pipeline
                flat_it = pipeline.iterator_str()
                bound = pipeline.loop_bound_str()
                result.write(
                    "for (long {it} = 0; {it} < {bound}; ++{it}) {{\n".format(
                        it=flat_it, bound=node.pipeline.loop_bound_str()))
                if pipeline.init_size > 0:
                    result.write("const bool {} = {} < {};\n".format(
                        node.pipeline.init_condition(), flat_it,
                        cpu.sym2cpp(pipeline.init_size)))
                if pipeline.drain_size > 0:
                    result.write("const bool {} = {} >= {};\n".format(
                        node.pipeline.drain_condition(), flat_it,
                        bound + (" - " + cpu.sym2cpp(pipeline.drain_size)
                                 if pipeline.drain_size != 0 else "")))

            if node.map.unroll:
                self.generate_unroll_loop_post(result, None, sdfg, state_id,
                                               node)
            else:
                is_innermost = not any(
                    [isinstance(x, dace.graph.nodes.EntryNode) for x in scope])
                if is_innermost:
                    self.generate_pipeline_loop_post(result, sdfg, state_id,
                                                     node)
                    self.generate_flatten_loop_post(result, sdfg, state_id,
                                                    node)

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(
            sdfg, sdfg.find_state(state_id), node)
        allocated = set()
        for child in dfg.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, dace.graph.nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child,
                                               None, result)
            self._dispatcher.dispatch_initialize(sdfg, dfg, state_id, child,
                                                 None, result)

    def _generate_PipelineExit(self, *args, **kwargs):
        self._generate_MapExit(*args, **kwargs)

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]
        if entry_node.map in self._unrolled_pes:
            # This was generated as unrolled processing elements, no need to
            # generate anything here
            return
        if isinstance(node, PipelineExit):
            flat_it = node.pipeline.iterator_str()
            bound = node.pipeline.loop_bound_str()
            pipeline = node.pipeline
            cond = []
            if pipeline.init_size > 0 and pipeline.init_overlap == False:
                cond.append("!" + pipeline.init_condition())
            if pipeline.drain_size > 0 and pipeline.drain_overlap == False:
                cond.append("!" + pipeline.drain_condition())
            if len(cond) > 0:
                callsite_stream.write("if ({}) {{".format(" && ".join(cond)))
            for it, r in reversed(list(zip(pipeline.params, pipeline.range))):
                callsite_stream.write(
                    "if ({it} >= {end}) {{\n{it} = {begin};\n".format(
                        it=it, begin=r[0], end=r[1]))
            for it, r in zip(pipeline.params, pipeline.range):
                callsite_stream.write(
                    "}} else {{\n{it} += {step};\n}}\n".format(
                        it=it, step=r[2]))
            if len(cond) > 0:
                callsite_stream.write("}\n")

        self._cpu_codegen._generate_MapExit(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

    def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):

        end_braces = 0

        axes = node.axes
        input_memlet = dfg.in_edges(node)[0].data
        src_data = sdfg.arrays[input_memlet.data]
        output_edge = dfg.out_edges(node)[0]
        output_memlet = output_edge.data
        dst_data = sdfg.arrays[output_memlet.data]

        output_type = self.make_vector_type(dst_data.dtype,
                                            output_memlet.veclen, False)

        # If axes were not defined, use all input dimensions
        input_dims = input_memlet.subset.dims()
        output_dims = output_memlet.subset.data_dims()
        if axes is None:
            axes = tuple(range(input_dims))
        output_axes = [a for a in range(input_dims) if a not in axes]

        # Obtain variable names per output and reduction axis
        axis_vars = []
        unroll_dim = []
        octr = 0
        for d in range(input_dims):
            if d in axes:
                axis_vars.append('__i%d' % d)
            else:
                axis_vars.append('__o%d' % octr)
                octr += 1
            if ((isinstance(src_data, dace.data.Stream)
                 and src_data.is_stream_array()) or
                (isinstance(src_data, dace.data.Array) and
                 src_data.storage == dace.dtypes.StorageType.FPGA_Registers)):
                # Unroll reads from registers and stream arrays
                unroll_dim.append(True)
            else:
                unroll_dim.append(False)

        # We want to pipeline the last non-unrolled dimension
        pipeline_dim = -1
        for i in itertools.chain(axes, output_axes):
            if not unroll_dim[i]:
                pipeline_dim = i

        if node.identity is not None:
            identity = cpu.sym2cpp(node.identity)
        else:
            identity = None

        # Determine reduction type
        reduction_type = operations.detect_reduction_type(node.wcr)
        if reduction_type == dace.dtypes.ReductionType.Custom:
            raise NotImplementedError("Custom reduction for FPGA is NYI")

        # Initialize accumulator variable if we're collapsing to a single value
        all_axes_collapsed = (len(axes) == input_dims)
        if all_axes_collapsed:
            accumulator = "_{}_accumulator".format(output_memlet.data)
            init_value = ""
            if identity is not None:
                init_value = " = " + identity
            elif reduction_type == dace.dtypes.ReductionType.Sum:
                # Set initial value to zero. Helpful for single cycle clock accumulator in Intel.
                init_value = " = 0"
            callsite_stream.write(
                "{} {}{};".format(output_type, accumulator, init_value), sdfg,
                state_id, node)

        # Generate inner loops (for each collapsed dimension)
        input_subset = input_memlet.subset
        iterators_inner = ["__i{}".format(axis) for axis in axes]
        for i, axis in enumerate(axes):
            if axis == pipeline_dim:
                self.generate_pipeline_loop_pre(callsite_stream, sdfg,
                                                state_id, node)
                self.generate_flatten_loop_pre(callsite_stream, sdfg, state_id,
                                               node)
            if unroll_dim[axis]:
                self.generate_unroll_loop_pre(callsite_stream, None, sdfg,
                                              state_id, node)
            callsite_stream.write(
                'for (size_t {var} = {begin}; {var} < {end}; {var} += {skip}) {{'
                .format(
                    var=iterators_inner[i],
                    begin=input_subset[axis][0],
                    end=input_subset[axis][1] + 1,
                    skip=input_subset[axis][2]), sdfg, state_id, node)
            if axis == pipeline_dim:
                self.generate_pipeline_loop_post(callsite_stream, sdfg,
                                                 state_id, node)
                self.generate_flatten_loop_post(callsite_stream, sdfg,
                                                state_id, node)
            if unroll_dim[axis]:
                self.generate_unroll_loop_post(callsite_stream, None, sdfg,
                                               state_id, node)
            end_braces += 1

        # Generate outer loops (over different output locations)
        output_subset = output_memlet.subset
        iterators_outer = ["__o{}".format(axis) for axis in range(output_dims)]
        for i, axis in enumerate(output_axes):
            if axis == pipeline_dim:
                self.generate_pipeline_loop_pre(callsite_stream, sdfg,
                                                state_id, node)
                self.generate_flatten_loop_pre(callsite_stream, sdfg, state_id,
                                               node)
            if unroll_dim[axis]:
                self.generate_unroll_loop_pre(callsite_stream, None, sdfg,
                                              state_id, node)
            callsite_stream.write(
                'for (size_t {var} = {begin}; {var} < {end}; {var} += {skip}) {{'
                .format(
                    var=iterators_outer[i],
                    begin=output_subset[i][0],
                    end=output_subset[i][1] + 1,
                    skip=output_subset[i][2]), sdfg, state_id, node)
            if axis == pipeline_dim:
                self.generate_pipeline_loop_post(callsite_stream, sdfg,
                                                 state_id, node)
                self.generate_flatten_loop_post(callsite_stream, sdfg,
                                                state_id, node)
            if unroll_dim[axis]:
                self.generate_unroll_loop_post(callsite_stream, None, sdfg,
                                               state_id, node)
            end_braces += 1

        # Input and output variables
        out_var = (accumulator
                   if all_axes_collapsed else cpp_array_expr(
                       sdfg,
                       output_memlet,
                       offset=iterators_outer,
                       relative_offset=False))
        in_var = cpp_array_expr(
            sdfg, input_memlet, offset=axis_vars, relative_offset=False)

        # generate reduction code

        self.make_reduction(sdfg, state_id, node, output_memlet,
                            dst_data.dtype, input_memlet.veclen,
                            output_memlet.veclen, output_type, reduction_type,
                            callsite_stream, iterators_inner, input_subset,
                            identity, out_var, in_var)

        # Generate closing braces
        for i in range(end_braces):
            callsite_stream.write('}', sdfg, state_id, node)

        if all_axes_collapsed:
            dst_expr = output_memlet.data
            offset = cpp_offset_expr(
                dst_data,
                output_memlet.subset,
                packed_veclen=output_memlet.veclen)
            if offset:
                dst_expr += " + " + offset
            def_type = self._dispatcher.defined_vars.get(output_memlet.data)
            callsite_stream.write(
                self.make_write(def_type, dst_data.dtype.ctype,
                                output_memlet.data, output_memlet.veclen,
                                dst_expr, "", out_var, output_memlet.wcr),
                sdfg, state_id, node)

    def generate_kernel(self, sdfg, state, kernel_name, subgraphs,
                        function_stream, callsite_stream):

        # Inspect the vector length of all memlets leading to each memory, to
        # make sure that they're consistent, and to allow us to instantiate the
        # memories as vector types to enable HLS to generate wider data paths.
        # Since we cannot pass this auxiliary data structure to the allocator,
        # which is called by the dispatcher, we temporarily store it in the
        # codegen object (naughty).
        self._memory_widths = type(self).detect_memory_widths(subgraphs)

        if self._in_device_code:
            raise CodegenError("Tried to generate kernel from device code")
        self._in_device_code = True
        self._cpu_codegen._packed_types = True

        kernel_stream = CodeIOStream()

        # Actual kernel code generation
        self.generate_kernel_internal(sdfg, state, kernel_name, subgraphs,
                                      kernel_stream, function_stream,
                                      callsite_stream)

        self._in_device_code = False
        self._cpu_codegen._packed_types = False

        # Store code strings to be passed to compilation phase
        self._kernel_codes.append((kernel_name, kernel_stream.getvalue()))

        # Delete the field we've abused to pass this dictionary to the memory
        # allocator
        del self._memory_widths
        self._allocated_global_arrays = set()

    def generate_modules(self, sdfg, state, kernel_name, subgraphs,
                         subgraph_parameters, scalar_parameters,
                         symbol_parameters, module_stream, entry_stream,
                         host_stream):
        """Main entry function for generating a Xilinx kernel."""

        # Module generation
        for subgraph in subgraphs:
            # Traverse to find first tasklets reachable in topological order
            to_traverse = subgraph.source_nodes()
            seen = set()
            tasklet_list = []
            access_nodes = []
            while len(to_traverse) > 0:
                n = to_traverse.pop()
                if n in seen:
                    continue
                seen.add(n)
                if (isinstance(n, dace.graph.nodes.Tasklet)
                        or isinstance(n, dace.graph.nodes.NestedSDFG)):
                    tasklet_list.append(n)
                else:
                    if isinstance(n, dace.graph.nodes.AccessNode):
                        access_nodes.append(n)
                    for e in subgraph.out_edges(n):
                        if e.dst not in seen:
                            to_traverse.append(e.dst)
            # Name module according to all reached tasklets (can be just one)
            labels = [n.label.replace(" ", "_") for n in tasklet_list]
            # If there are no tasklets, name it after access nodes in the
            # subgraph
            if len(labels) == 0:
                labels = [n.label.replace(" ", "_") for n in access_nodes]
            if len(labels) == 0:
                raise RuntimeError(
                    "Expected at least one tasklet or data node")
            module_name = "_".join(labels)
            self.generate_module(
                sdfg, state, module_name, subgraph,
                subgraph_parameters[subgraph] + scalar_parameters,
                symbol_parameters, module_stream, entry_stream, host_stream)

    def generate_host_function_boilerplate(
            self, sdfg, state, kernel_name, parameters, symbol_parameters,
            nested_global_transients, host_code_stream, header_stream,
            callsite_stream):

        # Generates:
        # - Definition of wrapper function in caller code
        # - Definition of kernel function in host code file
        # - Signature and opening brace of host code function in host code file

        # We exclude nested transients from the CPU code function call, as they
        # have not yet been allocated at this point
        nested_transient_set = {n.data for n in nested_global_transients}

        seen = set(nested_transient_set)
        kernel_args_call_host = []
        for is_output, argname, arg in parameters:
            # Only pass each array once from the host code
            if arg in seen:
                continue
            seen.add(arg)
            if not isinstance(arg, dace.data.Stream):
                kernel_args_call_host.append(
                    arg.signature(False, name=argname))

        # Treat scalars as symbols, assuming they can be input only
        symbol_sigs = [
            p.signature(name=name) for name, p in symbol_parameters.items()
        ]
        symbol_names = symbol_parameters.keys()

        kernel_args_call_host += symbol_names
        kernel_args_opencl = (self.opencl_parameters(
            sdfg, [p for p in parameters
                   if p[1] not in nested_transient_set]) + symbol_sigs)

        host_function_name = "__dace_runkernel_{}".format(kernel_name)

        # Write OpenCL host function
        host_code_stream.write(
            """\
DACE_EXPORTED void {host_function_name}({kernel_args_opencl}) {{
  hlslib::ocl::Program program = hlslib::ocl::GlobalContext().CurrentlyLoadedProgram();"""
            .format(
                host_function_name=host_function_name,
                kernel_args_opencl=", ".join(kernel_args_opencl)))

        header_stream.write("\n\nDACE_EXPORTED void {}({});\n\n".format(
            host_function_name, ", ".join(kernel_args_opencl)))

        callsite_stream.write("{}({});".format(
            host_function_name, ", ".join(kernel_args_call_host)))

        # Any extra transients stored in global memory on the FPGA must now be
        # allocated and passed to the kernel
        for arr_node in nested_global_transients:
            self._dispatcher.dispatch_allocate(sdfg, state, None, arr_node,
                                               None, host_code_stream)
            self._dispatcher.dispatch_initialize(sdfg, state, None, arr_node,
                                                 None, host_code_stream)


# ------------------------------------------------------------------------------


class PipelineEntry(dace.graph.nodes.MapEntry):
    @property
    def pipeline(self):
        return self._map

    @pipeline.setter
    def pipeline(self, val):
        self._map = val


class PipelineExit(dace.graph.nodes.MapExit):
    @property
    def pipeline(self):
        return self._map

    @pipeline.setter
    def pipeline(self, val):
        self._map = val


@make_properties
class Pipeline(dace.graph.nodes.Map):
    """ This a convenience-subclass of Map that allows easier implementation of
        loop nests (using regular Map indices) that need a constant-sized
        initialization and drain phase (e.g., N*M + c iterations), which would
        otherwise need a flattened one-dimensional map.
    """
    init_size = Property(
        dtype=int, desc="Number of initialization iterations.")
    init_overlap = Property(
        dtype=int,
        desc="Whether to increment regular map indices during initialization.")
    drain_size = Property(dtype=int, desc="Number of drain iterations.")
    drain_overlap = Property(
        dtype=int,
        desc="Whether to increment regular map indices during pipeline drain.")

    def __init__(self,
                 *args,
                 init_size=0,
                 init_overlap=False,
                 drain_size=0,
                 drain_overlap=False,
                 **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        self.init_size = init_size
        self.init_overlap = init_overlap
        self.drain_size = drain_size
        self.drain_overlap = drain_overlap
        self.flatten = True

    def iterator_str(self):
        return "__" + "".join(self.params)

    def loop_bound_str(self):
        bound = 1
        for begin, end, step in self.range:
            bound *= (step + end - begin) // step
        # Add init and drain phases when relevant
        add_str = (" + " + cpu.sym2cpp(self.init_size)
                   if self.init_size != 0 and not self.init_overlap else "")
        add_str += (" + " + cpu.sym2cpp(self.drain_size)
                    if self.drain_size != 0 and not self.drain_overlap else "")
        return cpu.sym2cpp(bound) + add_str

    def init_condition(self):
        """Variable that can be checked to see if pipeline is currently in
           initialization phase."""
        if self.init_size <= 0:
            raise ValueError("No init condition exists for " + self.label)
        return self.iterator_str() + "_init"

    def drain_condition(self):
        """Variable that can be checked to see if pipeline is currently in
           draining phase."""
        if self.drain_size <= 0:
            raise ValueError("No drain condition exists for " + self.label)
        return self.iterator_str() + "_drain"
