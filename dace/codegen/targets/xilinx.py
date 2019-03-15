from six import StringIO
import collections
import functools
import os
import itertools
import re
import sympy as sp

import dace
from dace import subsets
from dace.config import Config
from dace.frontend import operations
from dace.graph import nodes
from dace.sdfg import ScopeSubgraphView, find_input_arraynode, find_output_arraynode
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute, DefinedType)
from dace.codegen.targets.cpu import cpp_offset_expr, cpp_array_expr
from dace.codegen.targets import cpu, cuda

from dace.codegen import cppunparse

REDUCTION_TYPE_TO_HLSLIB = {
    dace.types.ReductionType.Min: "hlslib::op::Min",
    dace.types.ReductionType.Max: "hlslib::op::Max",
    dace.types.ReductionType.Sum: "hlslib::op::Sum",
    dace.types.ReductionType.Product: "hlslib::op::Product",
    dace.types.ReductionType.Logical_And: "hlslib::op::And",
}


class XilinxCodeGen(TargetCodeGenerator):
    """ Xilinx FPGA code generator. """
    target_name = 'xilinx'
    title = 'Xilinx'
    language = 'hls'

    def __init__(self, frame_codegen, sdfg):
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

        # Register additional Xilinx dispatchers
        self._dispatcher.register_map_dispatcher(
            [dace.types.ScheduleType.FPGA_Device], self)

        self._dispatcher.register_state_dispatcher(
            self,
            predicate=lambda sdfg, state: len(state.data_nodes()) > 0 and all([
                n.desc(sdfg).storage in [
                    dace.types.StorageType.FPGA_Global,
                    dace.types.StorageType.FPGA_Local,
                    dace.types.StorageType.FPGA_Registers]
                for n in state.data_nodes()]))

        self._dispatcher.register_node_dispatcher(
            self, predicate=lambda *_: self._in_device_code)

        xilinx_storage = [
            dace.types.StorageType.FPGA_Global,
            dace.types.StorageType.FPGA_Local,
            dace.types.StorageType.FPGA_Registers,
        ]
        self._dispatcher.register_array_dispatcher(xilinx_storage, self)

        # Register permitted copies
        for storage_from in itertools.chain(xilinx_storage,
                                            [dace.types.StorageType.Register]):
            for storage_to in itertools.chain(
                    xilinx_storage, [dace.types.StorageType.Register]):
                if (storage_from == dace.types.StorageType.Register
                        and storage_to == dace.types.StorageType.Register):
                    continue
                self._dispatcher.register_copy_dispatcher(
                    storage_from, storage_to, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.types.StorageType.FPGA_Global,
            dace.types.StorageType.CPU_Heap, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.types.StorageType.FPGA_Global,
            dace.types.StorageType.CPU_Stack, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.types.StorageType.CPU_Heap,
            dace.types.StorageType.FPGA_Global, None, self)
        self._dispatcher.register_copy_dispatcher(
            dace.types.StorageType.CPU_Stack,
            dace.types.StorageType.FPGA_Global, None, self)

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return False

    @staticmethod
    def cmake_options():
        compiler = make_absolute(
            Config.get("compiler", "xilinx", "executable"))
        host_flags = Config.get("compiler", "xilinx", "host_flags")
        synthesis_flags = Config.get("compiler", "xilinx", "synthesis_flags")
        build_flags = Config.get("compiler", "xilinx", "build_flags")
        mode = Config.get("compiler", "xilinx", "mode")
        target_platform = Config.get("compiler", "xilinx", "platform")
        enable_debugging = ("ON"
                            if Config.get_bool("compiler", "xilinx",
                                               "enable_debugging") else "OFF")
        options = [
            "-DSDACCEL_ROOT_DIR={}".format(
                os.path.dirname(os.path.dirname(compiler))),
            "-DDACE_XILINX_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_XILINX_SYNTHESIS_FLAGS=\"{}\"".format(synthesis_flags),
            "-DDACE_XILINX_BUILD_FLAGS=\"{}\"".format(build_flags),
            "-DDACE_XILINX_MODE={}".format(mode),
            "-DDACE_XILINX_TARGET_PLATFORM=\"{}\"".format(target_platform),
            "-DDACE_XILINX_ENABLE_DEBUGGING={}".format(enable_debugging),
        ]
        return options

    def generate_state(self, sdfg, state, function_stream, callsite_stream):
        """ Generate a kernel that runs all connected components within a state
            as concurrent dataflow modules. """

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
                if data.storage != dace.types.StorageType.FPGA_Global:
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
                if data.storage == dace.types.StorageType.FPGA_Global:
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
        """ Returns a set of data objects that are shared between two or more 
            of the specified subgraphs. """
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
        """ Generator that returns all transient global arrays nested in the
            passed subgraphs on the form (is_output, AccessNode). """
        seen = set()
        for subgraph in subgraphs:
            for n, scope in subgraph.all_nodes_recursive():
                if (isinstance(n, dace.graph.nodes.AccessNode)
                        and n.desc(sdfg).transient and n.desc(sdfg).storage ==
                        dace.types.StorageType.FPGA_Global):
                    if n.data in seen:
                        continue
                    seen.add(n.data)
                    if scope.out_degree(n) > 0:
                        yield (False, n)
                    if scope.in_degree(n) > 0:
                        yield (True, n)

    @staticmethod
    def make_parameters(sdfg, state, subgraphs):
        """ Determines the parameters that must be passed to the passed list of
            subgraphs, as well as to the global kernel. """

        # Get a set of data nodes that are shared across subgraphs
        shared_data = XilinxCodeGen.shared_data(subgraphs)

        # For some reason the array allocation dispatcher takes nodes, not
        # arrays. Build a dictionary of arrays to arbitrary data nodes
        # referring to them.
        data_to_node = {}

        global_data_params = []
        top_level_local_data = []
        subgraph_params = collections.OrderedDict()  # {subgraph: [params]}
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
                                    dace.types.StorageType.FPGA_Global and
                                    n.data not in nested_global_transients_seen
                                ):
                                nested_global_transients.append(n)
                            nested_global_transients_seen.add(n.data)
            subgraph_params[subgraph] = []
            # Differentiate global and local arrays. The former are allocated
            # from the host and passed to the device code, while the latter are
            # (statically) allocated on the device side.
            for is_output, dataname, data in candidates:
                if (isinstance(data, dace.data.Array)
                        or isinstance(data, dace.data.Scalar)
                        or isinstance(data, dace.data.Stream)):
                    if data.storage == dace.types.StorageType.FPGA_Global:
                        subgraph_params[subgraph].append((is_output, dataname,
                                                          data))
                        if is_output:
                            global_data_params.append((is_output, dataname,
                                                       data))
                        else:
                            global_data_params.append((is_output, dataname,
                                                       data))
                    elif (data.storage == dace.types.StorageType.FPGA_Local or
                          data.storage == dace.types.StorageType.FPGA_Registers
                          ):
                        if dataname in shared_data:
                            # Only transients shared across multiple components
                            # need to be allocated outside and passed as
                            # parameters
                            subgraph_params[subgraph].append((is_output,
                                                              dataname, data))
                            # Resolve the data to some corresponding node to be
                            # passed to the allocator
                            top_level_local_data.append(dataname)
                    else:
                        raise ValueError("Unsupported storage type: {}".format(
                            data.storage))
                else:
                    raise TypeError("Unsupported data type: {}".format(
                        type(data).__name__))
            subgraph_params[subgraph] = dace.types.deduplicate(
                subgraph_params[subgraph])

        # Deduplicate
        global_data_params = dace.types.deduplicate(global_data_params)
        top_level_local_data = dace.types.deduplicate(top_level_local_data)
        top_level_local_data = [data_to_node[n] for n in top_level_local_data]

        # Get scalar parameters
        scalar_parameters = sdfg.scalar_parameters(False)
        symbol_parameters = sdfg.undefined_symbols(False)

        return (global_data_params, top_level_local_data, subgraph_params,
                scalar_parameters, symbol_parameters, nested_global_transients)

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

    def generate_kernel(self, sdfg, state, kernel_name, subgraphs,
                        function_stream, callsite_stream):

        state_id = sdfg.node_id(state)

        (global_data_params, top_level_local_data, subgraph_params,
         scalar_parameters, symbol_parameters,
         nested_global_transients) = type(self).make_parameters(
             sdfg, state, subgraphs)

        # Scalar parameters are never output
        sc_parameters = [(False, pname, param)
                         for pname, param in scalar_parameters]

        symbol_params = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]

        # Inspect the vector length of all memlets leading to each memory, to
        # make sure that they're consistent, and to allow us to instantiate the
        # memories as vector types to enable HLS to generate wider data paths.
        # Since we cannot pass this auxiliary data structure to the allocator,
        # which is called by the dispatcher, we temporarily store it in the
        # codegen object.
        self._memory_widths = XilinxCodeGen.detect_memory_widths(subgraphs)

        # Write host code
        self.generate_host_code(sdfg, state, kernel_name,
                                global_data_params + sc_parameters,
                                symbol_parameters, nested_global_transients,
                                function_stream, callsite_stream)
        if self._in_device_code:
            raise CodegenError("Tried to generate kernel from device code")
        self._in_device_code = True
        self._cpu_codegen._packed_types = True

        # Now we write the device code
        module_stream = CodeIOStream()
        kernel_stream = CodeIOStream()

        # Write header
        module_stream.write("#include <dace/xilinx/device.h>\n\n", sdfg)
        self._frame.generate_fileheader(sdfg, module_stream)
        module_stream.write("\n", sdfg)

        # Build kernel signature
        kernel_args = []
        for is_output, dataname, data in global_data_params:
            if isinstance(data, dace.data.Array):
                kernel_args.append("dace::vec<{}, {}> *{}_{}".format(
                    data.dtype.ctype, self._memory_widths[dataname], dataname,
                    "out" if is_output else "in"))
            else:
                kernel_args.append(
                    data.signature(with_types=True, name=dataname))
        kernel_args += ([
            arg.signature(with_types=True, name=argname)
            for _, argname, arg in scalar_parameters
        ] + symbol_params)

        # Write kernel signature
        kernel_stream.write(
            "DACE_EXPORTED void {}({}) {{\n".format(
                kernel_name, ', '.join(kernel_args)), sdfg, state_id)

        # Insert interface pragmas
        mapped_args = 0
        for arg in kernel_args:
            var_name = re.findall("\w+", arg)[-1]
            if "*" in arg:
                kernel_stream.write(
                    "#pragma HLS INTERFACE m_axi port={} "
                    "offset=slave bundle=gmem{}".format(var_name, mapped_args),
                    sdfg, state_id)
                mapped_args += 1

        for arg in kernel_args + ["return"]:
            var_name = re.findall("\w+", arg)[-1]
            kernel_stream.write(
                "#pragma HLS INTERFACE s_axilite port={} bundle=control".
                format(var_name))

        # TODO: add special case if there's only one module for niceness
        kernel_stream.write("\n#pragma HLS DATAFLOW")
        kernel_stream.write("\nHLSLIB_DATAFLOW_INIT();")

        # Actual kernel code generation
        self.generate_modules(sdfg, state, kernel_name, subgraphs,
                              subgraph_params, sc_parameters,
                              symbol_parameters, top_level_local_data,
                              function_stream, module_stream, kernel_stream)

        kernel_stream.write("HLSLIB_DATAFLOW_FINALIZE();\n}\n")
        self._in_device_code = False
        self._cpu_codegen._packed_types = False

        concatenated_code = (
            module_stream.getvalue() + kernel_stream.getvalue())

        # Store code strings to be passed to compilation phase
        self._kernel_codes.append((kernel_name, concatenated_code))

        # Delete the field we've used to pass this dictionary to the memory
        # allocator
        del self._memory_widths
        self._allocated_global_arrays = set()

    def generate_modules(self, sdfg, state, kernel_name, subgraphs, params,
                         scalar_parameters, symbol_parameters,
                         top_level_local_data, function_stream, module_stream,
                         kernel_stream):

        # Emit allocations
        state_id = sdfg.node_id(state)
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                               module_stream, kernel_stream)
            self._dispatcher.dispatch_initialize(sdfg, state, state_id, node,
                                                 module_stream, kernel_stream)

        # Module generation
        for subgraph in subgraphs:
            # Traverse to find first tasklets reachable in topological order
            to_traverse = subgraph.source_nodes()
            seen = set()
            while len(to_traverse) > 0:
                n = to_traverse.pop()
                if n in seen:
                    continue
                seen.add(n)
                if (not isinstance(n, dace.graph.nodes.Tasklet)
                        and not isinstance(n, dace.graph.nodes.NestedSDFG)):
                    for e in subgraph.out_edges(n):
                        if e.dst not in seen:
                            to_traverse.append(e.dst)
            # Name module according to all reached tasklets (can be just one)
            labels = [
                n.label.replace(" ", "_") for n in seen
                if isinstance(n, dace.graph.nodes.Tasklet)
                or isinstance(n, dace.graph.nodes.NestedSDFG)
            ]
            if len(labels) == 0:
                labels = [
                    n.label.replace(" ", "_") for n in seen
                    if isinstance(n, dace.graph.nodes.AccessNode)
                ]
            if len(labels) == 0:
                raise RuntimeError(
                    "Expected at least one tasklet or data node")
            module_name = "_".join(labels)
            self.generate_module(sdfg, state, module_name, subgraph,
                                 params[subgraph] + scalar_parameters,
                                 symbol_parameters, function_stream,
                                 module_stream, kernel_stream)

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

    def generate_host_code(self, sdfg, state, kernel_name, params,
                           symbol_parameters, nested_global_transients,
                           function_stream, callsite_stream):

        state_id = sdfg.node_id(state)

        # We exclude nested transients from the CPU code function call, as they
        # have not yet been allocated at this point
        nested_transient_set = {n.data for n in nested_global_transients}

        symbol_sigs = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]
        symbol_names = list(symbol_parameters.keys())
        seen = set(nested_transient_set)
        kernel_args_call_wrapper = []
        kernel_args_call_host = []
        for is_output, pname, p in params:
            kernel_args_call_wrapper.append(p.signature(False, name=pname))
            # Only pass each array once from the host code
            if p in seen:
                continue
            seen.add(p)
            kernel_args_call_host.append(p.signature(False, name=pname))
        kernel_args_call_wrapper += symbol_names
        kernel_args_call_host += symbol_names
        kernel_args_opencl = (XilinxCodeGen.sdaccel_params(
            sdfg, [p for p in params
                   if p[1] not in nested_transient_set]) + symbol_sigs)
        kernel_args_hls = []
        kernel_args_hls_without_vectorization = []
        for is_output, argname, arg in params:
            if isinstance(arg, dace.data.Array):
                kernel_args_hls.append("dace::vec<{}, {}> *{}_{}".format(
                    arg.dtype.ctype, self._memory_widths[argname], argname,
                    "out" if is_output else "in"))
                kernel_args_hls_without_vectorization.append(
                    "{} *{}_{}".format(arg.dtype.ctype, argname, "out"
                                       if is_output else "in"))
            else:
                kernel_args_hls.append(
                    arg.signature(with_types=True, name=argname))
                kernel_args_hls_without_vectorization.append(
                    arg.signature(with_types=True, name=argname))
        kernel_args_hls += symbol_sigs
        kernel_args_hls_without_vectorization += symbol_sigs

        kernel_function_name = kernel_name

        #----------------------------------------------------------------------
        # Generate OpenCL host-code
        #----------------------------------------------------------------------

        kernel_file_name = "{}.xclbin".format(kernel_name)
        host_function_name = "__dace_runkernel_{}".format(kernel_name)

        # Write OpenCL host function
        code = CodeIOStream()
        code.write("""\
// Signature of kernel function (with raw pointers) for argument matching
DACE_EXPORTED void {kernel_function_name}({kernel_args_hls_novec});

DACE_EXPORTED void {host_function_name}({kernel_args_opencl}) {{""".format(
            kernel_function_name=kernel_function_name,
            kernel_args_hls_novec=", ".join(
                kernel_args_hls_without_vectorization),
            host_function_name=host_function_name,
            kernel_args_opencl=", ".join(kernel_args_opencl)))

        # Any extra transients stored in global memory on the FPGA must now be
        # allocated and passed to the kernel
        for arr_node in nested_global_transients:
            self._dispatcher.dispatch_allocate(sdfg, state, None, arr_node,
                                               None, code)
            self._dispatcher.dispatch_initialize(sdfg, state, None, arr_node,
                                                 None, code)

        code.write("""\
  hlslib::ocl::Program program =
      hlslib::ocl::GlobalContext().CurrentlyLoadedProgram();
  auto kernel = program.MakeKernel({kernel_function_name}, "{kernel_function_name}", {kernel_args});
  const std::pair<double, double> elapsed = kernel.ExecuteTask();
  std::cout << "Kernel executed in " << elapsed.second << " seconds.\\n" << std::flush;
}}""".format(
            kernel_function_name=kernel_function_name,
            kernel_args=", ".join(kernel_args_call_wrapper)))

        # Store code to be passed to compilation phase
        self._host_codes.append((kernel_name, code.getvalue()))

        #----------------------------------------------------------------------
        # Inject header for OpenCL host code in the calling code file
        #----------------------------------------------------------------------

        host_declaration = "\n\nDACE_EXPORTED void {}({});\n\n".format(
            host_function_name, ", ".join(kernel_args_opencl))
        function_stream.write(host_declaration, sdfg, state_id, None)

        #----------------------------------------------------------------------
        # Call the OpenCL host function from the callsite
        #----------------------------------------------------------------------

        callsite_stream.write(
            "{}({});".format(host_function_name,
                             ", ".join(kernel_args_call_host)), sdfg, state_id,
            None)


# Unused?
#    def generate_caller_code(self, sdfg, state, kernel_name, params,
#                             symbol_parameters, function_stream,
#                             callsite_stream):
#
#        state_id = sdfg.node_id(state)
#
#        symbol_sigs = [v.ctype + ' ' + k for k, v in symbol_parameters.items()]
#        symbol_names = symbol_parameters.keys()
#        kernel_args_call = [p.signature(False) for p in params] + symbol_names
#        kernel_args_plain = [i.signature() for i in params] + symbol_sigs
#
#        kernel_function_name = kernel_name
#
#        callsite_stream.write(
#            "{}({});".format(kernel_function_name,
#                             ", ".join(kernel_args_call)), sdfg, state_id,
#            None)

    def generate_module(self, sdfg, state, name, subgraph, params,
                        symbol_parameters, function_stream, module_stream,
                        kernel_stream):
        """Generates a module that will run as a dataflow function in the FPGA
           kernel."""

        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        symbol_sigs = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]
        symbol_names = list(symbol_parameters.keys())
        kernel_args_call = []
        kernel_args_module = []
        added = set()
        for is_output, pname, p in params:
            if isinstance(p, dace.data.Array):
                arr_name = "{}_{}".format(pname, "out" if is_output else "in")
                kernel_args_call.append(arr_name)
                kernel_args_module.append("dace::vec<{}, {}> {}*{}".format(
                    p.dtype.ctype, self._memory_widths[pname], "const "
                    if not is_output else "", arr_name))
            else:
                # Don't make duplicate arguments for other types than arrays
                if pname in added:
                    continue
                added.add(pname)
                if isinstance(p, dace.data.Stream):
                    kernel_args_call.append(
                        p.signature(with_types=False, name=pname))
                    if p.is_stream_array():
                        kernel_args_module.append(
                            "dace::FIFO<{}, {}, {}> {}[{}]".format(
                                p.dtype.ctype, p.veclen, p.buffer_size, pname,
                                p.size_string()))
                    else:
                        kernel_args_module.append(
                            "dace::FIFO<{}, {}, {}> &{}".format(
                                p.dtype.ctype, p.veclen, p.buffer_size, pname))
                else:
                    kernel_args_call.append(
                        p.signature(with_types=False, name=pname))
                    kernel_args_module.append(
                        p.signature(with_types=True, name=pname))
        kernel_args_call += symbol_names
        kernel_args_module += symbol_sigs

        module_function_name = "module_" + name

        # Unrolling processing elements: if the first scope of the subgraph
        # is an unrolled map, generate a processing element for each iteration
        scope_dict = subgraph.scope_dict(node_to_children=True)
        top_scopes = [
            n for n in scope_dict[None]
            if isinstance(n, dace.graph.nodes.EntryNode)
        ]
        unrolled_loops = 0
        if len(top_scopes) == 1:
            scope = top_scopes[0]
            if scope.unroll:
                self._unrolled_pes.add(scope.map)
                kernel_args_call += ", ".join(scope.map.params)
                kernel_args_module += ["int " + p for p in scope.params]
                for p, r in zip(scope.map.params, scope.map.range):
                    if len(r) > 3:
                        raise dace.codegen.codegen.CodegenError(
                            "Strided unroll not supported")
                    kernel_stream.write(
                        "for (int {param} = {begin}; {param} < {end}; "
                        "{param} += {increment}) {{\n#pragma HLS UNROLL".
                        format(
                            param=p, begin=r[0], end=r[1] + 1, increment=r[2]))
                    unrolled_loops += 1

        # Generate caller code in top-level function
        kernel_stream.write(
            "HLSLIB_DATAFLOW_FUNCTION({}, {});".format(
                module_function_name, ", ".join(kernel_args_call)), sdfg,
            state_id)

        for _ in range(unrolled_loops):
            kernel_stream.write("}")

        #----------------------------------------------------------------------
        # Generate kernel code
        #----------------------------------------------------------------------

        self._dispatcher.defined_vars.enter_scope(subgraph)

        module_body_stream = CodeIOStream()

        module_body_stream.write(
            "void {}({}) {{".format(module_function_name,
                                    ", ".join(kernel_args_module)), sdfg,
            state_id)

        # Construct ArrayInterface wrappers to pack input and output pointers
        # to the same global array
        in_args = {
            argname
            for out, argname, arg in params
            if isinstance(arg, dace.data.Array)
            and arg.storage == dace.types.StorageType.FPGA_Global and not out
        }
        out_args = {
            argname
            for out, argname, arg in params
            if isinstance(arg, dace.data.Array)
            and arg.storage == dace.types.StorageType.FPGA_Global and out
        }
        if len(in_args) > 0 or len(out_args) > 0:
            # Add ArrayInterface objects to wrap input and output pointers to
            # the same array
            module_body_stream.write("\n")
            interfaces_added = set()
            for _, argname, arg in params:
                if argname in interfaces_added:
                    continue
                interfaces_added.add(argname)
                has_in_ptr = argname in in_args
                has_out_ptr = argname in out_args
                if not has_in_ptr and not has_out_ptr:
                    continue
                in_ptr = ("{}_in".format(argname) if has_in_ptr else "nullptr")
                out_ptr = ("{}_out".format(argname)
                           if has_out_ptr else "nullptr")
                module_body_stream.write(
                    "dace::ArrayInterface<{}, {}> {}({}, {});".format(
                        arg.dtype.ctype, self._memory_widths[argname], argname,
                        in_ptr, out_ptr))
            module_body_stream.write("\n")

        # Allocate local transients
        data_to_allocate = (set(subgraph.top_level_transients()) - set(
            sdfg.shared_transients()) - set([p[1] for p in params]))
        allocated = set()
        for node in subgraph.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                               function_stream,
                                               module_body_stream)
            self._dispatcher.dispatch_initialize(sdfg, state, state_id, node,
                                                 function_stream,
                                                 module_body_stream)

        self._dispatcher.dispatch_subgraph(
            sdfg,
            subgraph,
            state_id,
            module_stream,
            module_body_stream,
            skip_entry_node=False)

        module_stream.write(module_body_stream.getvalue(), sdfg, state_id)
        module_stream.write("}\n\n")

        self._dispatcher.defined_vars.exit_scope(subgraph)

    def get_generated_codeobjects(self):

        execution_mode = Config.get("compiler", "xilinx", "mode")
        sdaccel_dir = os.path.dirname(
            os.path.dirname(
                make_absolute(Config.get("compiler", "xilinx", "executable"))))
        sdaccel_platform = Config.get("compiler", "xilinx", "platform")

        kernel_file_name = "DACE_BINARY_DIR \"{}".format(self._program_name)
        if execution_mode == "software_emulation":
            kernel_file_name += "_sw_emu.xclbin\""
            xcl_emulation_mode = "sw_emu"
            xilinx_sdx = sdaccel_dir
        elif execution_mode == "hardware_emulation":
            kernel_file_name += "_hw_emu.xclbin\""
            xcl_emulation_mode = "sw_emu"
            xilinx_sdx = sdaccel_dir
        elif execution_mode == "hardware" or execution_mode == "simulation":
            kernel_file_name += "_hw.xclbin\""
            xcl_emulation_mode = None
            xilinx_sdx = None
        else:
            raise dace.codegen.codegen.CodegenError(
                "Unknown Xilinx execution mode: {}".format(execution_mode))

        set_env_vars = ""
        set_str = "dace::set_environment_variable(\"{}\", \"{}\");\n"
        unset_str = "dace::unset_environment_variable(\"{}\");\n"
        set_env_vars += (set_str.format("XCL_EMULATION_MODE",
                                        xcl_emulation_mode)
                         if xcl_emulation_mode is not None else
                         unset_str.format("XCL_EMULATION_MODE"))
        set_env_vars += (set_str.format("XILINX_SDX", xilinx_sdx)
                         if xilinx_sdx is not None else
                         unset_str.format("XILINX_SDX"))

        host_code = CodeIOStream()
        host_code.write("""\
#include "dace/xilinx/host.h"
#include "dace/dace.h"
#include <iostream>\n\n""")

        self._frame.generate_fileheader(self._global_sdfg, host_code)

        host_code.write("""
DACE_EXPORTED int __dace_init_xilinx({signature}) {{
    {environment_variables}
    hlslib::ocl::GlobalContext().MakeProgram({kernel_file_name});
    return 0;
}}

{host_code}""".format(
            signature=self._global_sdfg.signature(),
            environment_variables=set_env_vars,
            kernel_file_name=kernel_file_name,
            host_code="".join([
                "{separator}\n// Kernel: {kernel_name}"
                "\n{separator}\n\n{code}\n\n".format(
                    separator="/" * 79, kernel_name=name, code=code)
                for (name, code) in self._host_codes
            ])))

        host_code_obj = CodeObject(self._program_name + "_host",
                                   host_code.getvalue(), "cpp", XilinxCodeGen,
                                   "Xilinx")

        kernel_code_objs = [
            CodeObject("kernel_" + kernel_name, code, "cpp", XilinxCodeGen,
                       "Xilinx") for (kernel_name, code) in self._kernel_codes
        ]

        return [host_code_obj] + kernel_code_objs

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        result = StringIO()
        nodedesc = node.desc(sdfg)
        arrsize = " * ".join([
            cppunparse.pyexpr2cpp(dace.symbolic.symstr(s))
            for s in nodedesc.strides
        ])
        is_dynamically_sized = any(
            dace.symbolic.issymbolic(s, sdfg.constants)
            for s in nodedesc.strides)

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

            buffer_length_dynamically_sized = (
                isinstance(nodedesc.buffer_size, sp.Expr)
                and len(nodedesc.free_symbols) > 0)

            if buffer_length_dynamically_sized:
                raise dace.codegen.codegen.CodegenError(
                    "Buffer length of stream cannot have dynamic size on FPGA")

            if arrsize != "1":
                is_stream_array = True
            else:
                is_stream_array = False

            if is_stream_array:
                result.write("dace::FIFO<{}, {}, {}> {}[{}];\n".format(
                    nodedesc.dtype.ctype, nodedesc.veclen,
                    nodedesc.buffer_size, dataname, arrsize))
                result.write("dace::SetNames({}, \"{}\", {});".format(
                    dataname, dataname, arrsize))
                self._dispatcher.defined_vars.add(dataname,
                                                  DefinedType.StreamArray)
            else:
                result.write("dace::FIFO<{}, {}, {}> {}(\"{}\");".format(
                    nodedesc.dtype.ctype, nodedesc.veclen,
                    nodedesc.buffer_size, dataname, dataname))
                self._dispatcher.defined_vars.add(dataname, DefinedType.Stream)

        elif isinstance(nodedesc, dace.data.Array):

            if nodedesc.storage == dace.types.StorageType.FPGA_Global:

                if self._in_device_code:

                    if nodedesc not in self._allocated_global_arrays:
                        raise RuntimeError("Cannot allocate global array "
                                           "from device code: {} in {}".format(
                                               node.label, sdfg.name))

                else:

                    devptr_name = dataname
                    if isinstance(nodedesc, dace.data.Array):
                        # TODO: Distinguish between read, write, and
                        #       read+write
                        # TODO: Handle memory banks (location?)
                        self._allocated_global_arrays.add(node.data)
                        result.write(
                            "auto {} = hlslib::ocl::GlobalContext()."
                            "MakeBuffer<{}, hlslib::ocl::Access::readWrite>"
                            "({});".format(dataname, nodedesc.dtype.ctype,
                                           arrsize))
                        self._dispatcher.defined_vars.add(
                            dataname, DefinedType.Pointer)

            elif (nodedesc.storage == dace.types.StorageType.FPGA_Local or
                  nodedesc.storage == dace.types.StorageType.FPGA_Registers):

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
                    arrsize_symbolic = functools.reduce(
                        sp.mul.Mul, nodedesc.strides)
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
                    result.write("dace::vec<{}, {}> {};\n".format(
                        nodedesc.dtype.ctype, veclen, dataname))
                    self._dispatcher.defined_vars.add(dataname,
                                                      DefinedType.Scalar)
                else:
                    result.write("dace::vec<{}, {}> {}[{}];\n".format(
                        nodedesc.dtype.ctype, veclen, dataname, arrsize_vec))
                    self._dispatcher.defined_vars.add(dataname,
                                                      DefinedType.Pointer)
                    if nodedesc.storage == dace.types.StorageType.FPGA_Registers:
                        result.write("#pragma HLS ARRAY_PARTITION variable={} "
                                     "complete\n".format(dataname))
                    elif len(nodedesc.shape) > 1:
                        result.write("#pragma HLS ARRAY_PARTITION variable={} "
                                     "block factor={}\n".format(
                                         dataname, nodedesc.shape[-2]))
                    # result.write(
                    #     "#pragma HLS DEPENDENCE variable={} false".format(
                    #         dataname))

            else:
                raise NotImplementedError("Xilinx: Unimplemented storage type "
                                          + str(nodedesc.storage))

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
            dace.types.StorageType.CPU_Heap, dace.types.StorageType.CPU_Stack,
            dace.types.StorageType.CPU_Pinned
        ]
        fpga_storage_types = [
            dace.types.StorageType.FPGA_Global,
            dace.types.StorageType.FPGA_Local,
            dace.types.StorageType.FPGA_Registers,
        ]

        # Determine directionality
        if isinstance(src_node,
                      nodes.AccessNode) and memlet.data == src_node.data:
            outgoing_memlet = True
        elif isinstance(dst_node,
                        nodes.AccessNode) and memlet.data == dst_node.data:
            outgoing_memlet = False
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        data_to_data = (isinstance(src_node, nodes.AccessNode)
                        and isinstance(dst_node, nodes.AccessNode))

        host_to_device = (data_to_data and src_storage in cpu_storage_types and
                          dst_storage == dace.types.StorageType.FPGA_Global)
        device_to_host = (data_to_data
                          and src_storage == dace.types.StorageType.FPGA_Global
                          and dst_storage in cpu_storage_types)
        device_to_device = (
            data_to_data and src_storage == dace.types.StorageType.FPGA_Global
            and dst_storage == dace.types.StorageType.FPGA_Global)

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
                                          "supported for Xilinx FPGA.")

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
              and (((src_storage == dace.types.StorageType.FPGA_Local
                     or src_storage == dace.types.StorageType.FPGA_Registers)
                    and dst_storage not in fpga_storage_types) or
                   ((dst_storage == dace.types.StorageType.FPGA_Local
                     or dst_storage == dace.types.StorageType.FPGA_Registers)
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
                sdfg).storage == dace.types.StorageType.FPGA_Registers
                                    or dst_node.desc(sdfg).storage ==
                                    dace.types.StorageType.FPGA_Registers)

            # Loop intro
            num_loops = 0
            for i, copy_dim in enumerate(copy_shape):
                if copy_dim != 1:
                    callsite_stream.write(
                        "for (auto __dace_copy{} = 0; __dace_copy{} < {}; "
                        "++__dace_copy{}) {{".format(i, i, copy_dim, i))
                    if register_to_register:
                        callsite_stream.write("#pragma HLS UNROLL")
                    num_loops += 1

            # Pragmas
            if num_loops > 0:
                if not register_to_register:
                    callsite_stream.write("#pragma HLS PIPELINE II=1")
                if len(copy_shape) > 1:
                    callsite_stream.write("#pragma HLS LOOP_FLATTEN")

            # Construct indices (if the length of the stride array is zero,
            # resolves to an empty string)
            src_index = " + ".join(([""] if len(dst_strides) > 0 else []) + [
                "__dace_copy{} * {}".format(i, cpu.sym2cpp(stride))
                for i, stride in enumerate(src_strides) if copy_shape[i] != 1
            ])
            dst_index = " + ".join(([""] if len(dst_strides) > 0 else []) + [
                "__dace_copy{} * {}".format(i, cpu.sym2cpp(stride))
                for i, stride in enumerate(dst_strides) if copy_shape[i] != 1
            ])

            src_def_type = self._dispatcher.defined_vars.get(src_node.data)
            dst_def_type = self._dispatcher.defined_vars.get(dst_node.data)

            if src_def_type == DefinedType.Stream:
                read_expr = src_expr
            elif src_def_type == DefinedType.Scalar:
                read_expr = src_node.label
            else:
                read_expr = "dace::Read<{}, {}>({}{})".format(
                    ctype, memlet.veclen, src_expr, src_index)

            if dst_def_type == DefinedType.Stream:
                callsite_stream.write("{}.push({});".format(
                    dst_expr, read_expr))
            else:
                if dst_def_type == DefinedType.Scalar:
                    write_expr = dst_node.label
                callsite_stream.write("dace::Write<{}, {}>({}{}, {});".format(
                    ctype, memlet.veclen, dst_expr, dst_index, read_expr))

            # Inject dependence pragmas (DaCe semantics implies no conflict)
            for node in [src_node, dst_node]:
                if (isinstance(node.desc(sdfg), dace.data.Array)
                        and node.desc(sdfg).storage in [
                            dace.types.StorageType.FPGA_Local,
                            dace.StorageType.FPGA_Registers
                        ]):
                    callsite_stream.write(
                        "#pragma HLS DEPENDENCE variable={} false".format(
                            node.data))

            # Loop outtro
            for _ in range(num_loops):
                callsite_stream.write("}")

        else:

            self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node,
                                          dst_node, edge, None,
                                          callsite_stream)

    @staticmethod
    def sdaccel_params(sdfg, kernel_params):
        seen = set()
        out_params = []
        for is_output, pname, param in kernel_params:
            # Since we can have both input and output versions of the same
            # array, make sure we only pass it once from the host code
            if param in seen:
                continue
            seen.add(param)
            if isinstance(param, dace.data.Array):
                out_params.append("hlslib::ocl::Buffer<{}, "
                                  "hlslib::ocl::Access::readWrite> &{}".format(
                                      param.dtype.ctype, pname))
            else:
                out_params.append(param.signature(with_types=True, name=pname))
        return out_params

    def get_next_scope_entries(self, sdfg, dfg, scope_entry):
        parent_scope_entry = dfg.scope_dict()[scope_entry]
        parent_scope = dfg.scope_subgraph(parent_scope_entry)

        # Get all scopes from the same level
        all_scopes = [
            node for node in parent_scope.topological_sort()
            if isinstance(node, nodes.EntryNode)
        ]

        return all_scopes[all_scopes.index(scope_entry) + 1:]

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        method_name = "_generate_" + type(node).__name__
        # Fake inheritance... use this class' method if it exists,
        # otherwise fall back on CPU codegen
        if hasattr(self, method_name):

            if hasattr(node, "schedule") and node.schedule not in [
                    dace.types.ScheduleType.Default,
                    dace.types.ScheduleType.FPGA_Device
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

        if isinstance(src_node, nodes.Tasklet):
            src_storage = dace.types.StorageType.Register
            try:
                src_parent = dfg.scope_dict()[src_node]
            except KeyError:
                src_parent = None
            dst_schedule = (None
                            if src_parent is None else src_parent.map.schedule)
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dace.types.StorageType.Register
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

    def _generate_MapEntry(self, sdfg, dfg, state_id, node, function_stream,
                           callsite_stream):

        result = callsite_stream

        scope_dict = dfg.scope_dict()

        if node.map in self._unrolled_pes:

            # This is a top-level unrolled map, meaning it has been used to
            # replicate processing elements. Don't generate anything here.
            pass

        else:

            # Generate nested loops
            for i, r in enumerate(node.map.range):
                var = node.map.params[i]
                begin, end, skip = r
                result.write(
                    "for (auto {} = {}; {} < {}; {} += {}) {{\n".format(
                        var, cpu.sym2cpp(begin), var, cpu.sym2cpp(end + 1),
                        var, cpu.sym2cpp(skip)), sdfg, state_id, node)

            # Pipeline innermost loops
            scope = dfg.scope_dict(True)[node]

            if node.map.unroll:
                result.write("#pragma HLS UNROLL\n", sdfg, state_id, node)
            else:
                is_innermost = not any(
                    [isinstance(x, nodes.EntryNode) for x in scope])
                if is_innermost:
                    result.write(
                        "#pragma HLS PIPELINE II=1\n#pragma HLS LOOP_FLATTEN",
                        sdfg, state_id, node)

            if node.map.flatten:
                result.write("#pragma HLS LOOP_FLATTEN\n", sdfg, state_id,
                             node)

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(
            sdfg, sdfg.find_state(state_id), node)
        allocated = set()
        for child in dfg.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child,
                                               None, result)
            self._dispatcher.dispatch_initialize(sdfg, dfg, state_id, child,
                                                 None, result)

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]
        if entry_node.map in self._unrolled_pes:
            # This was generated as unrolled processing elements, no need to
            # generate anything here
            return
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

        output_type = 'dace::vec<%s, %s>' % (dst_data.dtype.ctype,
                                             output_memlet.veclen)

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
                 src_data.storage == dace.types.StorageType.FPGA_Registers)):
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

        # Initialize accumulator variable if we're collapsing to a single value
        all_axes_collapsed = (len(axes) == input_dims)
        if all_axes_collapsed:
            accumulator = "_{}_accumulator".format(output_memlet.data)
            callsite_stream.write("{} {};".format(output_type, accumulator),
                                  sdfg, state_id, node)

        # Generate inner loops (for each collapsed dimension)
        input_subset = input_memlet.subset
        iterators_inner = ["__i{}".format(axis) for axis in axes]
        for i, axis in enumerate(axes):
            callsite_stream.write(
                'for (int {var} = {begin}; {var} < {end}; {var} += {skip}) {{'.
                format(
                    var=iterators_inner[i],
                    begin=input_subset[axis][0],
                    end=input_subset[axis][1] + 1,
                    skip=input_subset[axis][2]), sdfg, state_id, node)
            if unroll_dim[axis]:
                callsite_stream.write("#pragma HLS UNROLL\n")
            if axis == pipeline_dim:
                callsite_stream.write(
                    "#pragma HLS PIPELINE II=1\n#pragma HLS LOOP_FLATTEN")
            end_braces += 1

        # Generate outer loops (over different output locations)
        output_subset = output_memlet.subset
        iterators_outer = ["__o{}".format(axis) for axis in range(output_dims)]
        for i, axis in enumerate(output_axes):
            callsite_stream.write(
                'for (int {var} = {begin}; {var} < {end}; {var} += {skip}) {{'.
                format(
                    var=iterators_outer[i],
                    begin=output_subset[i][0],
                    end=output_subset[i][1] + 1,
                    skip=output_subset[i][2]), sdfg, state_id, node)
            if unroll_dim[axis]:
                callsite_stream.write("#pragma HLS UNROLL\n")
            if axis == pipeline_dim:
                callsite_stream.write(
                    "#pragma HLS PIPELINE II=1\n#pragma HLS LOOP_FLATTEN")
            end_braces += 1

        # Determine reduction type
        reduction_type = operations.detect_reduction_type(node.wcr)
        if reduction_type == dace.types.ReductionType.Custom:
            raise NotImplementedError("Custom reduction for FPGA is NYI")

        # Input and output variables
        out_var = (accumulator
                   if all_axes_collapsed else cpp_array_expr(
                       sdfg,
                       output_memlet,
                       offset=iterators_outer,
                       relative_offset=False))
        in_var = cpp_array_expr(
            sdfg, input_memlet, offset=axis_vars, relative_offset=False)

        # Call library function to perform reduction
        reduction_cpp = "dace::Reduce<{}, {}, {}, {}<{}>>".format(
            dst_data.dtype.ctype, input_memlet.veclen, output_memlet.veclen,
            REDUCTION_TYPE_TO_HLSLIB[reduction_type], dst_data.dtype.ctype)

        # Check if this is the first iteration of accumulating into this
        # location
        is_first_iteration = " && ".join([
            "{} == {}".format(iterators_inner[i], input_subset[axis][0])
            for i, axis in enumerate(axes)
        ])
        if identity is not None:
            # If this is the first iteration, set the previous value to be
            # identity, otherwise read the value from the output location
            prev_var = "{}_prev".format(output_memlet.data)
            callsite_stream.write(
                "{} {} = ({}) ? ({}) : ({});".format(
                    output_type, prev_var, is_first_iteration, identity,
                    out_var), sdfg, state_id, node)
            callsite_stream.write(
                "{} = {}({}, {});".format(out_var, reduction_cpp, prev_var,
                                          in_var), sdfg, state_id, node)
        else:
            # If this is the first iteration, assign the value read from the
            # input directly to the output
            callsite_stream.write(
                "{} = ({}) ? ({}) : {}({}, {});".format(
                    out_var, is_first_iteration, in_var, reduction_cpp,
                    out_var, in_var), sdfg, state_id, node)

        # Generate closing braces
        for i in range(end_braces):
            callsite_stream.write('}', sdfg, state_id, node)
            if i == end_braces - 1 and all_axes_collapsed:
                dst_expr = output_memlet.data
                offset = cpp_offset_expr(
                    dst_data,
                    output_memlet.subset,
                    packed_veclen=output_memlet.veclen)
                if offset:
                    dst_expr += " + " + offset
                callsite_stream.write(
                    "dace::Write({}, {});".format(dst_expr, out_var), sdfg,
                    state_id, node)

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):

        # TODO: this is copied from the CPU-codegen, necessary to inject
        # pragmas at the output memlets! Should consolidate.

        callsite_stream.write('{\n', sdfg, state_id, node)

        state_dfg = sdfg.nodes()[state_id]

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in dfg.in_edges(node):
            u = edge.src
            memlet = edge.data

            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError('Duplicates found in memlets')
                # Special case: code->code
                if isinstance(edge.src, nodes.CodeNode):
                    shared_data_name = 's%d_n%d%s_n%d%s' % (
                        state_id, dfg.node_id(edge.src), edge.src_conn,
                        dfg.node_id(edge.dst), edge.dst_conn)

                    # Read variable from shared storage
                    callsite_stream.write(
                        'const dace::vec<%s, %s>& %s = __%s;' %
                        (edge.data.data.dtype.ctype, sym2cpp(edge.data.veclen),
                         edge.dst_conn, shared_data_name), sdfg, state_id,
                        [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn,
                                                      DefinedType.Scalar)

                else:
                    src_node = find_input_arraynode(state_dfg, edge)

                    self._dispatcher.dispatch_copy(
                        src_node, node, edge, sdfg, state_dfg, state_id,
                        function_stream, callsite_stream)

                # Also define variables in the C++ unparser scope
                self._cpu_codegen._locals.define(edge.dst_conn, -1,
                                                 self._cpu_codegen._ldepth + 1)
                arrays.add(edge.dst_conn)

        callsite_stream.write('\n', sdfg, state_id, node)

        # Use outgoing edges to preallocate output local vars
        for edge in dfg.out_edges(node):
            v = edge.dst
            memlet = edge.data

            if edge.src_conn:
                if edge.src_conn in arrays:  # Disallow duplicates
                    continue
                # Special case: code->code
                if isinstance(edge.dst, nodes.CodeNode):
                    callsite_stream.write(
                        'dace::vec<%s, %s> %s;' %
                        (sdfg.arrays[memlet.data].dtype.ctype,
                         sym2cpp(memlet.veclen), edge.src_conn), sdfg,
                        state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.src_conn,
                                                      DefinedType.Scalar)
                else:
                    dst_node = find_output_arraynode(state_dfg, edge)

                    self._dispatcher.dispatch_copy(
                        node, dst_node, edge, sdfg, state_dfg, state_id,
                        function_stream, callsite_stream)

                # Also define variables in the C++ unparser scope
                self._cpu_codegen._locals.define(edge.src_conn, -1,
                                                 self._cpu_codegen._ldepth + 1)
                arrays.add(edge.src_conn)

        callsite_stream.write('\n    ///////////////////\n', sdfg, state_id,
                              node)

        cpu.unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                            callsite_stream, self._cpu_codegen._locals,
                            self._cpu_codegen._ldepth)

        callsite_stream.write('    ///////////////////\n\n', sdfg, state_id,
                              node)

        # Process outgoing memlets
        self._cpu_codegen.process_out_memlets(
            sdfg, state_id, node, state_dfg, self._dispatcher, callsite_stream,
            True, function_stream)

        for edge in state_dfg.out_edges(node):
            datadesc = sdfg.arrays[edge.data.data]
            if (isinstance(datadesc, dace.data.Array) and
                (datadesc.storage == dace.types.StorageType.FPGA_Local
                 or datadesc.storage == dace.types.StorageType.FPGA_Registers)
                    and edge.data.wcr is None):
                callsite_stream.write(
                    "#pragma HLS DEPENDENCE variable=__{} false".format(
                        edge.src_conn))

        callsite_stream.write('}\n', sdfg, state_id, node)

        self._dispatcher.defined_vars.exit_scope(node)
