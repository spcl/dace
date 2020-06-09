from six import StringIO
import collections
import enum
import functools
import itertools
import re
import warnings
import sympy as sp
import numpy as np

import dace
from dace import subsets, data as dt
from dace.dtypes import deduplicate
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg import ScopeSubgraphView, find_input_arraynode, find_output_arraynode
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator, DefinedType
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute, DefinedType)
from dace.codegen.targets.cpp import (cpp_offset_expr, cpp_array_expr, sym2cpp,
                                      memlet_copy_to_absolute_strides)
from dace.codegen import cppunparse
from dace.properties import Property, make_properties, indirect_properties
from dace.symbolic import evaluate

_CPU_STORAGE_TYPES = {
    dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.CPU_ThreadLocal,
    dace.dtypes.StorageType.CPU_Pinned
}
_FPGA_STORAGE_TYPES = {
    dace.dtypes.StorageType.FPGA_Global, dace.dtypes.StorageType.FPGA_Local,
    dace.dtypes.StorageType.FPGA_Registers,
    dace.dtypes.StorageType.FPGA_ShiftRegister
}


class MemoryType(enum.Enum):
    DDR = enum.auto()
    HBM = enum.auto()


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
        self._bank_assignments = {}  # {(data name, sdfg): (type, id)}

        # Register additional FPGA dispatchers
        self._dispatcher.register_map_dispatcher(
            [dace.dtypes.ScheduleType.FPGA_Device], self)

        self._dispatcher.register_state_dispatcher(
            self,
            predicate=lambda sdfg, state: len(state.data_nodes()) > 0 and all([
                n.desc(sdfg).storage in [
                    dace.dtypes.StorageType.FPGA_Global, dace.dtypes.
                    StorageType.FPGA_Local, dace.dtypes.StorageType.
                    FPGA_Registers, dace.dtypes.StorageType.FPGA_ShiftRegister
                ] for n in state.data_nodes()
            ]))

        self._dispatcher.register_node_dispatcher(
            self, predicate=lambda *_: self._in_device_code)

        fpga_storage = [
            dace.dtypes.StorageType.FPGA_Global,
            dace.dtypes.StorageType.FPGA_Local,
            dace.dtypes.StorageType.FPGA_Registers,
            dace.dtypes.StorageType.FPGA_ShiftRegister,
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
            dace.dtypes.StorageType.CPU_ThreadLocal, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.CPU_Heap,
            dace.dtypes.StorageType.FPGA_Global, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.dtypes.StorageType.CPU_ThreadLocal,
            dace.dtypes.StorageType.FPGA_Global, None, self)

        # Inspect the vector length of all memlets leading to each memory, to
        # make sure that they're consistent, and to allow us to instantiate the
        # memories as vector types to enable HLS to generate wider data paths.
        self._memory_widths = self.detect_memory_widths(sdfg)

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
                allocated.add(node.data)
                # Allocate transients
                self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                                   function_stream,
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
                    if isinstance(node, dace.sdfg.nodes.AccessNode):
                        if node.data in seen:
                            if seen[node.data] != sg:
                                shared.add(node.data)
                        else:
                            seen[node.data] = sg
        return shared

    @classmethod
    def make_parameters(cls, sdfg, state, subgraphs):
        """Determines the parameters that must be passed to the passed list of
           subgraphs, as well as to the global kernel."""

        # Get a set of data nodes that are shared across subgraphs
        shared_data = cls.shared_data(subgraphs)

        # Find scalar parameters (to filter out from data parameters)
        scalar_parameters = [(k, v) for k, v in sdfg.arrays.items()
                             if isinstance(v, dt.Scalar) and not v.transient]
        scalar_set = set(p[0] for p in scalar_parameters)

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
                if isinstance(node, dace.sdfg.nodes.AccessNode)
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
                if isinstance(n, dace.sdfg.nodes.AccessNode):
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
                if dataname in scalar_set:
                    continue  # Skip already-parsed scalars
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
                    elif (data.storage in (
                            dace.dtypes.StorageType.FPGA_Local,
                            dace.dtypes.StorageType.FPGA_Registers,
                            dace.dtypes.StorageType.FPGA_ShiftRegister)):
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

        symbol_parameters = {
            k: dt.Scalar(v)
            for k, v in sdfg.symbols.items() if k not in sdfg.constants
        }

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
    def detect_memory_widths(parent_sdfg):
        """ For each memory, checks that all the memlets are consistent (they
            have the same width). This allow us to instantiate to generate data
            paths with a single data size throughout the subgraph. This is
            enforced for streams and global memories, but is optional for local
            memories.
        """
        memory_widths = {}
        seen = set()
        alias = {}  # Maps nested containers to their parent containers
        q = collections.deque()  # BFS traversal
        for state in parent_sdfg.nodes():
            q.extendleft((n, state, parent_sdfg) for n in state.nodes())
        while len(q) > 0:
            node, state, sdfg = q.pop()
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                nested_sdfg = node.sdfg
                for nested_state in nested_sdfg.nodes():
                    q.extendleft((n, nested_state, nested_sdfg)
                                 for n in nested_state.nodes())
                # Remember mapping of outer to inner containers given by the
                # connectors
                for edge in state.in_edges(node):
                    memlet_data = edge.data.data
                    if memlet_data is not None:
                        alias[(edge.dst_conn, nested_sdfg)] = (memlet_data,
                                                               sdfg)
                for edge in state.out_edges(node):
                    memlet_data = edge.data.data
                    if memlet_data is not None:
                        alias[(edge.src_conn, nested_sdfg)] = (memlet_data,
                                                               sdfg)
            elif isinstance(node, dace.sdfg.nodes.AccessNode):
                if node in seen:
                    continue
                seen.add(node)
                name = node.data
                desc = sdfg.data(name)
                # Trace to outermost version
                key = (name, sdfg)
                while key in alias:
                    key = alias[key]
                name = key[0]
                for edge in state.all_edges(node):
                    if (isinstance(edge.data, dace.memlet.EmptyMemlet)
                            or edge.data.data is None):
                        continue
                    if (isinstance(edge.src, dace.sdfg.nodes.AccessNode) and
                            isinstance(edge.dst, dace.sdfg.nodes.AccessNode)
                            and edge.data.veclen == 1):
                        # Consistent vectorization is not enforced for
                        # memcopies, but if this memory is not found anywhere
                        # else, we need to set it to 1 later. Or, if a
                        # vectorization width is set for this memcopy, we can
                        # still use it.
                        if key not in memory_widths:
                            memory_widths[key] = None
                        continue
                    if (key not in memory_widths
                            or memory_widths[key] is None):
                        if (isinstance(desc, dace.data.Stream)
                                and desc.veclen != edge.data.veclen):
                            raise ValueError(
                                "Vector length on memlet {} ({}) doesn't "
                                "match vector length of {} ({})".format(
                                    edge.data, edge.data.veclen, node.data,
                                    node.desc(sdfg).veclen))
                        memory_widths[key] = edge.data.veclen
                    else:
                        if (memory_widths[key] is not None
                                and memory_widths[key] != edge.data.veclen):
                            # If the vector length is inconsistent, set it to 1
                            memory_widths[key] = 1
        # Inherit the parent memory width for all aliased nested arrays
        for key in alias:
            if key in memory_widths:
                continue
            trace = key
            while trace in alias:
                trace = alias[trace]
            memory_widths[key] = memory_widths[trace]
        for key, width in memory_widths.items():
            if width is None:
                # If no non-memcopy accesses were found, default to one
                memory_widths[key] = 1
        return memory_widths

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):

        if not self._in_device_code:
            # If we're not already generating kernel code we need to set up the
            # kernel launch
            subgraphs = [dfg_scope]
            return self.generate_kernel(
                sdfg, sdfg.node(state_id),
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

            if sym2cpp(arrsize) != "1":
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
                        self._allocated_global_arrays.add(node.data)
                        memory_bank_arg = ""
                        if "bank" in nodedesc.location:
                            try:
                                bank = int(nodedesc.location["bank"])
                            except ValueError:
                                raise ValueError(
                                    "FPGA memory bank specifier "
                                    "must be an integer: {}".format(
                                        nodedesc.location["bank"]))
                            memory_bank_arg = (
                                "hlslib::ocl::MemoryBank::bank{}, ".format(
                                    bank))
                            # (memory type, bank id)
                            self._bank_assignments[(dataname,
                                                    sdfg)] = (MemoryType.DDR,
                                                              bank)
                        else:
                            self._bank_assignments[(dataname, sdfg)] = None
                        result.write(
                            "auto {} = dace::fpga::_context->Get()."
                            "MakeBuffer<{}, hlslib::ocl::Access::readWrite>"
                            "({}{});".format(dataname, nodedesc.dtype.ctype,
                                             memory_bank_arg,
                                             sym2cpp(arrsize)))
                        self._dispatcher.defined_vars.add(
                            dataname, DefinedType.Pointer)

            elif (nodedesc.storage in (dace.dtypes.StorageType.FPGA_Local,
                                       dace.dtypes.StorageType.FPGA_Registers,
                                       dace.dtypes.StorageType.FPGA_ShiftRegister)):

                if not self._in_device_code:
                    raise dace.codegen.codegen.CodegenError(
                        "Tried to allocate local FPGA memory "
                        "outside device code: {}".format(dataname))
                if is_dynamically_sized:
                    raise ValueError(
                        "Dynamic allocation of FPGA "
                        "fast memory not allowed: {}, size {}".format(
                            dataname, arrsize))

                # Absorb vector size into type and adjust array size
                # accordingly
                veclen = self._memory_widths[(node.data, sdfg)]
                generate_scalar = False
                if veclen > 1:
                    arrsize_symbolic = nodedesc.total_size
                    arrsize_eval = evaluate(arrsize_symbolic / veclen,
                                            sdfg.constants)
                    if sym2cpp(arrsize_eval) == "1":
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
                    if (nodedesc.storage ==
                            dace.dtypes.StorageType.FPGA_ShiftRegister):
                        self.define_shift_register(
                            dataname, nodedesc, arrsize_vec, veclen,
                            function_stream, result, sdfg, state_id, node)
                    else:
                        self.define_local_array(
                            dataname, nodedesc, arrsize_vec, veclen,
                            function_stream, result, sdfg, state_id, node)

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
                   dst_storage, dst_schedule, edge, dfg, function_stream,
                   callsite_stream):

        u, v, memlet = edge.src, edge.dst, edge.data

        # Determine directionality
        if isinstance(
                src_node,
                dace.sdfg.nodes.AccessNode) and memlet.data == src_node.data:
            outgoing_memlet = True
        elif isinstance(
                dst_node,
                dace.sdfg.nodes.AccessNode) and memlet.data == dst_node.data:
            outgoing_memlet = False
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        data_to_data = (isinstance(src_node, dace.sdfg.nodes.AccessNode)
                        and isinstance(dst_node, dace.sdfg.nodes.AccessNode))

        host_to_device = (data_to_data and src_storage in _CPU_STORAGE_TYPES
                          and
                          dst_storage == dace.dtypes.StorageType.FPGA_Global)
        device_to_host = (data_to_data and
                          src_storage == dace.dtypes.StorageType.FPGA_Global
                          and dst_storage in _CPU_STORAGE_TYPES)
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
              and (((src_storage in (dace.dtypes.StorageType.FPGA_Local,
                                     dace.dtypes.StorageType.FPGA_Registers,
                                     dace.dtypes.StorageType.FPGA_ShiftRegister))
                    and dst_storage not in _FPGA_STORAGE_TYPES) or
                   ((dst_storage in (dace.dtypes.StorageType.FPGA_Local,
                                     dace.dtypes.StorageType.FPGA_Registers,
                                     dace.dtypes.StorageType.FPGA_ShiftRegister))
                    and src_storage not in _FPGA_STORAGE_TYPES))):
            raise NotImplementedError(
                "Copies between host memory and FPGA "
                "local memory not supported: from {} to {}".format(
                    src_node, dst_node))

        elif data_to_data:

            if memlet.wcr is not None:
                raise NotImplementedError("WCR not implemented for copy edges")

            if src_storage == dace.dtypes.StorageType.FPGA_ShiftRegister:
                raise NotImplementedError(
                    "Reads from shift registers only supported from tasklets.")

            # Try to turn into degenerate/strided ND copies
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                memlet_copy_to_absolute_strides(
                    self._dispatcher,
                    sdfg,
                    memlet,
                    src_node,
                    dst_node,
                    packed_types=True))

            ctype = src_node.desc(sdfg).dtype.ctype

            # For a single vector access, tolerate not having set a range
            if copy_shape[-1] != 1:
                # Adjust for vectorization length
                copy_shape[-1] = copy_shape[-1] / memlet.veclen

            if dst_storage == dace.dtypes.StorageType.FPGA_ShiftRegister:
                if len(copy_shape) != 1:
                    raise ValueError(
                        "Only single-dimensional writes "
                        "to shift registers supported: {}{}".format(
                            dst_node.data, copy_shape))
                if copy_shape[-1] > memlet.veclen:
                    raise ValueError(
                        "Only a single (vector) element can be "
                        "written to a shift register: {}{}".format(
                            dst_node.data, copy_shape[-1]))

            # Check if we are copying between vectorized and non-vectorized
            # types
            memwidth_src = self._memory_widths[(src_node.data, sdfg)]
            memwidth_dst = self._memory_widths[(dst_node.data, sdfg)]
            if memwidth_src < memwidth_dst:
                is_pack = True
                is_unpack = False
                packing_factor = memwidth_dst // memwidth_src
                if memwidth_dst % memwidth_src != 0:
                    raise ValueError(
                        "Destination vectorization width {} "
                        "is not divisible by source vectorization width {}.".
                        format(memwidth_dst, memwidth_src))
                self.generate_converter(False, ctype, packing_factor, dst_node,
                                        state_id, sdfg, function_stream)
            elif memwidth_src > memwidth_dst:
                is_pack = False
                is_unpack = True
                packing_factor = memwidth_src // memwidth_dst
                if memwidth_src % memwidth_dst != 0:
                    raise ValueError(
                        "Source vectorization width {} is not divisible "
                        "by destination vectorization width {}.".format(
                            memwidth_dst, memwidth_src))
                self.generate_converter(True, ctype, packing_factor, dst_node,
                                        state_id, sdfg, function_stream)
            else:
                is_pack = False
                is_unpack = False
                packing_factor = 1

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
                "__dace_copy{}{}".format(
                    i, " * " + sym2cpp(stride) if stride != 1 else "")
                for i, stride in enumerate(src_strides) if copy_shape[i] != 1
            ])
            dst_index = " + ".join([
                "__dace_copy{}{}".format(
                    i, " * " + sym2cpp(stride) if stride != 1 else "")
                for i, stride in enumerate(dst_strides) if copy_shape[i] != 1
            ])

            src_def_type = self._dispatcher.defined_vars.get(src_node.data)
            dst_def_type = self._dispatcher.defined_vars.get(dst_node.data)

            pattern = re.compile(r"([^\s]+)(\s*\+\s*)?(.*)")

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
                                       memlet.veclen, src_expr, src_index,
                                       is_pack, packing_factor)

            # Language specific
            if dst_storage == dace.dtypes.StorageType.FPGA_ShiftRegister:
                write_expr = self.make_shift_register_write(
                    dst_def_type, ctype, dst_node.label, memlet.veclen,
                    dst_expr, dst_index, read_expr, None, is_unpack,
                    packing_factor)
            else:
                write_expr = self.make_write(dst_def_type, ctype,
                                             dst_node.label, memlet.veclen,
                                             dst_expr, dst_index, read_expr,
                                             None, is_unpack, packing_factor)

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
    def make_opencl_parameter(name, desc):
        if isinstance(desc, dace.data.Array):
            return ("hlslib::ocl::Buffer<{}, "
                    "hlslib::ocl::Access::readWrite> &{}".format(
                        desc.dtype.ctype, name))
        else:
            return (desc.signature(with_types=True, name=name))

    def get_next_scope_entries(self, sdfg, dfg, scope_entry):
        parent_scope_entry = dfg.scope_dict()[scope_entry]
        parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all scopes from the same level
        all_scopes = [
            node for node in parent_scope.topological_sort()
            if isinstance(node, dace.sdfg.nodes.EntryNode)
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
                warnings.warn("Found schedule {} on {} node in FPGA code. "
                              "Ignoring.".format(node.schedule,
                                                 type(node).__name__))

            getattr(self, method_name)(sdfg, dfg, state_id, node,
                                       function_stream, callsite_stream)
        else:
            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):

        if isinstance(src_node, dace.sdfg.nodes.CodeNode):
            src_storage = dace.dtypes.StorageType.Register
            try:
                src_parent = dfg.scope_dict()[src_node]
            except KeyError:
                src_parent = None
            dst_schedule = (None
                            if src_parent is None else src_parent.map.schedule)
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, dace.sdfg.nodes.CodeNode):
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
                        function_stream, callsite_stream)

    def _generate_PipelineEntry(self, *args, **kwargs):
        self._generate_MapEntry(*args, **kwargs)

    def _is_innermost(self, scope, scope_dict, sdfg):
        to_search = list(scope)
        while len(to_search) > 0:
            x = to_search.pop()
            if (isinstance(
                    x,
                (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.PipelineEntry))):
                # Degenerate loops should not be pipelined
                fully_degenerate = True
                for begin, end, skip in x.map.range:
                    if not self._is_degenerate(begin, end, skip, sdfg)[0]:
                        fully_degenerate = False
                        break
                # Non-unrolled, non-degenerate loops must be pipelined, so we
                # are not innermost
                if not x.unroll and not fully_degenerate:
                    return False
                to_search += scope_dict[x]
            elif isinstance(x, dace.sdfg.nodes.NestedSDFG):
                for state in x.sdfg:
                    if not self._is_innermost(state.nodes(),
                                              state.scope_dict(True), x.sdfg):
                        return False
        return True

    @staticmethod
    def _is_degenerate(begin, end, skip, sdfg):
        try:
            begin_val = evaluate(begin, sdfg.constants)
            skip_val = evaluate(skip, sdfg.constants)
            end_val = evaluate(end, sdfg.constants)
            is_degenerate = begin_val + skip_val > end_val
            return is_degenerate, begin_val
        except TypeError:  # Cannot statically evaluate expression
            return False, begin

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
            scope_dict = dfg.scope_dict(True)
            scope = scope_dict[node]
            is_innermost = self._is_innermost(scope, scope_dict, sdfg)

            # Generate custom iterators if this is a pipelined (and thus
            # flattened) loop
            if isinstance(node, dace.sdfg.nodes.PipelineEntry):
                for i in range(len(node.map.range)):
                    result.write("long {} = {};\n".format(
                        node.map.params[i], node.map.range[i][0]))

            is_degenerate = []
            degenerate_values = []
            for begin, end, skip in node.map.range:
                # If we know at compile-time that a loop will only have a
                # single iteration, we can replace it with a simple assignment
                b, val = self._is_degenerate(begin, end, skip, sdfg)
                is_degenerate.append(b)
                degenerate_values.append(val)
            fully_degenerate = all(is_degenerate)

            if not fully_degenerate:
                if node.map.unroll:
                    self.generate_unroll_loop_pre(result, None, sdfg, state_id,
                                                  node)
                elif is_innermost:
                    self.generate_pipeline_loop_pre(result, sdfg, state_id,
                                                    node)

            # Generate nested loops
            if not isinstance(node, dace.sdfg.nodes.PipelineEntry):

                if is_innermost and not fully_degenerate:
                    self.generate_flatten_loop_pre(result, sdfg, state_id,
                                                   node)

                for i, r in enumerate(node.map.range):
                    var = node.map.params[i]
                    begin, end, skip = r
                    # decide type of loop variable
                    loop_var_type = "int"
                    # try to decide type of loop variable
                    try:
                        if (evaluate(begin, sdfg.constants) >= 0
                                and evaluate(skip, sdfg.constants) > 0):
                            # it could be an unsigned (uint32) variable: we need
                            # to check to the type of 'end',
                            # if we are able to determine it
                            symbols = list(dace.symbolic.symlist(end).values())
                            if len(symbols) > 0:
                                sym = symbols[0]
                                if str(sym) in sdfg.symbols:
                                    end_type = sdfg.symbols[str(sym)].dtype
                                else:
                                    # Symbol not found, try to use symbol object
                                    # or use the default symbol type (int32)
                                    end_type = sym.dtype
                            else:
                                end_type = None
                            if end_type is not None:
                                if np.dtype(end_type.dtype.type) > np.dtype(
                                        'uint32'):
                                    loop_var_type = end_type.ctype
                                elif np.issubdtype(
                                        np.dtype(end_type.dtype.type),
                                        np.unsignedinteger):
                                    loop_var_type = "size_t"
                    except (UnboundLocalError):
                        raise UnboundLocalError('Pipeline scopes require '
                                                'specialized bound values')
                    except (TypeError):
                        # Raised when the evaluation of begin or skip fails.
                        # This could occur, for example, if they are defined in terms of other symbols, which
                        # is the case in a tiled map
                        pass

                    if is_degenerate[i]:
                        result.write(
                            "{{\nconst {} {} = {}; // Degenerate loop".format(
                                loop_var_type, var, degenerate_values[i]))
                    else:
                        result.write(
                            "for ({} {} = {}; {} < {}; {} += {}) {{\n".format(
                                loop_var_type, var, sym2cpp(begin), var,
                                sym2cpp(end + 1), var, sym2cpp(skip)), sdfg,
                            state_id, node)
            else:
                pipeline = node.pipeline
                flat_it = pipeline.iterator_str()
                bound = pipeline.loop_bound_str()
                result.write(
                    "for (long {it} = 0; {it} < {bound}; ++{it}) {{\n".format(
                        it=flat_it, bound=node.pipeline.loop_bound_str()))
                if pipeline.init_size != 0:
                    result.write("const bool {} = {} < {};\n".format(
                        node.pipeline.init_condition(), flat_it,
                        sym2cpp(pipeline.init_size)))
                if pipeline.drain_size != 0:
                    result.write("const bool {} = {} >= {};\n".format(
                        node.pipeline.drain_condition(), flat_it,
                        bound + (" - " + sym2cpp(pipeline.drain_size)
                                 if pipeline.drain_size != 0 else "")))

            if not fully_degenerate:
                if node.map.unroll:
                    self.generate_unroll_loop_post(result, None, sdfg,
                                                   state_id, node)
                elif is_innermost:
                    self.generate_pipeline_loop_post(result, sdfg, state_id,
                                                     node)
                    self.generate_flatten_loop_post(result, sdfg, state_id,
                                                    node)

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(sdfg, sdfg.node(state_id),
                                                 node)
        allocated = set()
        for child in dfg.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, dace.sdfg.nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child,
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
        if isinstance(node, dace.sdfg.nodes.PipelineExit):
            flat_it = node.pipeline.iterator_str()
            bound = node.pipeline.loop_bound_str()
            pipeline = node.pipeline
            cond = []
            if pipeline.init_size != 0 and pipeline.init_overlap == False:
                cond.append("!" + pipeline.init_condition())
            if pipeline.drain_size != 0 and pipeline.drain_overlap == False:
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
            callsite_stream.write("}\n}\n")
        else:
            self._cpu_codegen._generate_MapExit(sdfg, dfg, state_id, node,
                                                function_stream,
                                                callsite_stream)

    def generate_kernel(self, sdfg, state, kernel_name, subgraphs,
                        function_stream, callsite_stream):

        if self._in_device_code:
            from dace.codegen.codegen import CodegenError
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
                if (isinstance(n, dace.sdfg.nodes.Tasklet)
                        or isinstance(n, dace.sdfg.nodes.NestedSDFG)):
                    tasklet_list.append(n)
                else:
                    if isinstance(n, dace.sdfg.nodes.AccessNode):
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
        kernel_args_opencl = []
        # Split into arrays and scalars
        arrays = sorted(
            [t for t in parameters if not isinstance(t[2], dace.data.Scalar)],
            key=lambda t: t[1])
        scalars = [t for t in parameters if isinstance(t[2], dace.data.Scalar)]
        scalars += ((False, k, v) for k, v in symbol_parameters.items())
        scalars = list(sorted(scalars, key=lambda t: t[1]))
        for is_output, argname, arg in itertools.chain(arrays, scalars):
            # Only pass each array once from the host code
            if arg in seen:
                continue
            seen.add(arg)
            if not isinstance(arg, dace.data.Stream):
                kernel_args_call_host.append(
                    arg.signature(False, name=argname))
                kernel_args_opencl.append(
                    FPGACodeGen.make_opencl_parameter(argname, arg))

        kernel_args_call_host = dace.dtypes.deduplicate(kernel_args_call_host)
        kernel_args_opencl = dace.dtypes.deduplicate(kernel_args_opencl)

        host_function_name = "__dace_runkernel_{}".format(kernel_name)

        # Write OpenCL host function
        host_code_stream.write(
            """\
DACE_EXPORTED void {host_function_name}({kernel_args_opencl}) {{
  hlslib::ocl::Program program = dace::fpga::_context->Get().CurrentlyLoadedProgram();"""
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
