# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from six import StringIO
import collections
import enum
import functools
import itertools
import re
import warnings
import sympy as sp
import numpy as np
from typing import Dict

import dace
from dace.codegen.targets import cpp
from dace import subsets, data as dt, dtypes
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import SDFG, nodes, utils, dynamic_map_inputs
from dace.sdfg import ScopeSubgraphView, find_input_arraynode, find_output_arraynode
from dace.codegen import exceptions as cgx
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute)
from dace.codegen import cppunparse
from dace.properties import Property, make_properties, indirect_properties
from dace.symbolic import evaluate

_CPU_STORAGE_TYPES = {
    dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal,
    dtypes.StorageType.CPU_Pinned
}
_FPGA_STORAGE_TYPES = {
    dtypes.StorageType.FPGA_Global, dtypes.StorageType.FPGA_Local,
    dtypes.StorageType.FPGA_Registers, dtypes.StorageType.FPGA_ShiftRegister
}


def vector_element_type_of(dtype):
    if isinstance(dtype, dace.pointer):
        # "Dereference" the pointer type and try again
        return vector_element_type_of(dtype.base_type)
    elif isinstance(dtype, dace.vector):
        return dtype.base_type
    return dtype


def is_fpga_kernel(sdfg, state):
    """
    Returns whether the given state is an FPGA kernel and should be dispatched
    to the FPGA code generator.
    :return: True if this is an FPGA kernel, False otherwise.
    """
    if ("is_FPGA_kernel" in state.location
            and state.location["is_FPGA_kernel"] == False):
        return False
    data_nodes = state.data_nodes()
    if len(data_nodes) == 0:
        return False
    for n in data_nodes:
        if n.desc(sdfg).storage not in (dtypes.StorageType.FPGA_Global,
                                        dtypes.StorageType.FPGA_Local,
                                        dtypes.StorageType.FPGA_Registers,
                                        dtypes.StorageType.FPGA_ShiftRegister):
            return False
    return True


class FPGACodeGen(TargetCodeGenerator):
    # Set by deriving class
    target_name = None
    title = None
    language = None

    def __init__(self, frame_codegen, sdfg: SDFG):

        # The inheriting class must set target_name, title and language.

        self._in_device_code = False
        self._cpu_codegen = None
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._kernel_count = 0
        self._global_sdfg = sdfg
        self._program_name = sdfg.name

        # Verify that we did not miss the allocation of any global arrays, even
        # if they're nested deep in the SDFG
        self._allocated_global_arrays = set()
        self._unrolled_pes = set()

        # Dependencies among kernels (if any)
        self._kernels_dependencies = dict()
        # TODO: is this ok, or do we want something else?
        self._kernels_names_to_id = dict()

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        self._host_codes = []
        self._kernel_codes = []
        # any other kind of generated file if any (name, code object)
        self._other_codes = {}
        self._bank_assignments = {}  # {(data name, sdfg): (type, id)}
        self._stream_connections = {}  # { name: [src, dst] }

        # Register additional FPGA dispatchers
        self._dispatcher.register_map_dispatcher(
            [dtypes.ScheduleType.FPGA_Device], self)

        self._dispatcher.register_state_dispatcher(self,
                                                   predicate=is_fpga_kernel)

        self._dispatcher.register_node_dispatcher(
            self,
            predicate=lambda sdfg, state, node: self._in_device_code and not (
                isinstance(node, nodes.Tasklet) and node.language == dtypes.
                Language.SystemVerilog))

        fpga_storage = [
            dtypes.StorageType.FPGA_Global,
            dtypes.StorageType.FPGA_Local,
            dtypes.StorageType.FPGA_Registers,
            dtypes.StorageType.FPGA_ShiftRegister,
        ]
        self._dispatcher.register_array_dispatcher(fpga_storage, self)

        # Register permitted copies
        for storage_from in itertools.chain(fpga_storage,
                                            [dtypes.StorageType.Register]):
            for storage_to in itertools.chain(fpga_storage,
                                              [dtypes.StorageType.Register]):
                if (storage_from == dtypes.StorageType.Register
                        and storage_to == dtypes.StorageType.Register):
                    continue
                self._dispatcher.register_copy_dispatcher(
                    storage_from, storage_to, None, self)
        self._dispatcher.register_copy_dispatcher(
            dtypes.StorageType.FPGA_Global, dtypes.StorageType.CPU_Heap, None,
            self)
        self._dispatcher.register_copy_dispatcher(
            dtypes.StorageType.FPGA_Global, dtypes.StorageType.CPU_ThreadLocal,
            None, self)
        self._dispatcher.register_copy_dispatcher(
            dtypes.StorageType.CPU_Heap, dtypes.StorageType.FPGA_Global, None,
            self)
        self._dispatcher.register_copy_dispatcher(
            dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.FPGA_Global,
            None, self)

        # Memory width converters (gearboxing) to generate globally
        self.converters_to_generate = set()

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return False

    def on_target_used(self) -> None:
        # Right before finalizing code, write FPGA context to state structure
        self._frame.statestruct.append('dace::fpga::Context *fpga_context;')

    def _kernels_subgraphs(self, graph, precedences, start_kernel=0):
        """ Finds subgraphs of an SDFGState or ScopeSubgraphView that identifies Kernels
        """
        from dace.sdfg.scope import ScopeSubgraphView

        if not isinstance(graph, (dace.sdfg.SDFGState, ScopeSubgraphView)):
            raise TypeError(
                "Expected SDFGState or ScopeSubgraphView, got: {}".format(
                    type(graph).__name__))
        # candidates = graph.source_nodes()
        # components = collections.OrderedDict()  # {start node: nodes in component}
        # for cand in candidates:
        #     if isinstance(cand, dace.nodes.AccessNode):
        #         # AccessNodes can be read from multiple concurrent components, so
        #         # check all out edges
        #         start_nodes = [e.dst for e in graph.out_edges(cand)]
        #         for n in start_nodes:
        #             if n not in components:
        #                 components[n] = {cand, n}
        #             else:
        #                 # Components can read from multiple start arrays
        #                 components[n].add(cand)
        #     else:
        #         # The source node == the first control or compute node
        #         components[cand] = {cand}

        subgraphs = collections.defaultdict(set)  # {pe_id: {nodes in subgraph}}

        # Go over the nodes and populate the kernels subgraphs
        print("--------------------")
        #TODO: Should we go recursive???
        for node, node_state in graph.all_nodes_recursive():
            # print(subgraphs.keys())
            # print("---", node, type(node))
            #TODO: access node may not have  Kernel ID
            # - cleanup and refactor, also considering how Kern ID are assigned

            assigned = False
            if hasattr(node, "_kernel"):
                print("* Node ", node, " belongs to kernel", node._kernel)
                if node._kernel in subgraphs:
                    subgraphs[node._kernel].add(node)
                else:
                    subgraphs[node._kernel] = {node}
                assigned = True
            # add this node to the corresponding subgraph
            if isinstance(node, dace.nodes.AccessNode):
                # AccessNodes can be read from multiple kernels, so
                # check all out edges
                # NOTE: we may recur on NSDFG
                start_nodes = [e.dst for e in node_state.out_edges(node)]
                for n in start_nodes:
                    if hasattr(n, "_kernel"):
                        print("* Node ", n, " belongs to kernel", n._kernel)
                        if n._kernel in subgraphs:
                            subgraphs[n._kernel].add(node)
                        else:
                            subgraphs[n._kernel] = {node}
                        assigned = True
            if not assigned:
                print("Node: ", node,
                      " is not assigned to any kernel---------------")

        # Check: if some subgraph overlaps with any other, then there is something wrong:
        # TODO:
        # - we can't check if Access node are in multiple subgraphs

        # Now stick each of the found components in a ScopeSubgraphView and return
        # them. Sort according kernel precedences order

        #TODO: find a nice way of doing this or completely remove this if we can use precedences in some other way

        #####################################################
        import networkx as nx
        kernels_graph = nx.DiGraph()
        for k in subgraphs.keys():
            # we could have no dependencies at all
            kernels_graph.add_node(k)
            if k in precedences:
                kernel_precedences = precedences[k]
                #TODO: this dependency can come from outside
                for p in kernel_precedences:
                    if p in kernels_graph:
                        kernels_graph.add_edge(p, k)

        subgraph_views = []
        all_nodes = graph.nodes()
        for kernel_id in nx.topological_sort(kernels_graph):

            print("Adding to the list of kernels: ",
                  kernel_id, " it depends on: ",
                  list(kernels_graph.predecessors(kernel_id)))
            # Return the subgraph and the kernel id
            subgraph_views.append((ScopeSubgraphView(
                graph, [n for n in all_nodes if n in subgraphs[kernel_id]],
                None), kernel_id))
        return subgraph_views
        # return [
        #     ScopeSubgraphView(graph, [n for n in all_nodes if n in sg], None)
        #     for sg in subgraphs.values()
        # ]

    def generate_state(self, sdfg: dace.SDFG, state: dace.SDFGState,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        """Generate a kernel that runs all Processing Elements within a state
           as concurrent dataflow modules."""
        # TODO: add proper docs and type hint

        # TODO: add something on top of this (or here) to detect all KERNELS first
        state_id = sdfg.node_id(state)

        # Determine kernels in state
        # num_kernels = 1
        # if state.parent.parent == None:
        #     # Top Level State: determine kernels in state
        #     # We do this by looking at
        #     subgraphs = dace.sdfg.concurrent_subgraphs(state)
        #     for sg in subgraphs:
        #         num_kernels, precedences = self.compute_kernels(sg)
        #
        #     num_kernels, precedences = self.compute_kernels(state)
        #
        # # Determine independent components
        # # subgraphs = dace.sdfg.concurrent_subgraphs(state)
        # print("Found ", num_kernels, "kernels")
        # # Get all the kernels inside this graph
        # num_kernels=1
        # if num_kernels == 1 or state.parent.parent != None:  # Top-Level State
        #     kernels = [state]
        # else:
        #     kernels = self._kernels_subgraphs(state, precedences)

        ################

        # Determine independent components: these are our starting kernels
        are_there_kernels = False  #TMP
        if state.parent.parent == None:
            subgraphs = dace.sdfg.concurrent_subgraphs(state)

            #Then, if we are on a top-level state, try to split these components further
            kernels = []
            start_kernel = 0
            for sg in subgraphs:
                # Top Level State: determine kernels in state
                # We do this by looking at
                if state.parent.parent == None:
                    # annotate nodes with kernel_ids
                    num_kernels, dependencies = self.compute_kernels(
                        sg, default_kernel=start_kernel)

                    if num_kernels > 1:
                        print(
                            "---- Found more than one kernel in a subgraph! ----"
                        )
                        #TODO find a better way of keeping track of deps
                        kernels.extend(
                            self._kernels_subgraphs(sg,
                                                    dependencies,
                                                    start_kernel=start_kernel))

                        self._kernels_dependencies.update(dependencies)
                    else:
                        kernels.append((sg, start_kernel))
                    start_kernel = start_kernel + num_kernels

            #if, at the end the number of kernels is equal to the number of components
            # use the previous version
            # TODO: cleanup this
            if len(subgraphs) == len(kernels):
                kernels = [(state, 0)]
            else:
                are_there_kernels = True

        else:
            kernels = [(state, 0)]

        # subgraphs = dace.sdfg.concurrent_subgraphs(state)
        # for i, sg in enumerate(subgraphs):
        #     print("Subgraph: ", i, " Nodes: ", sg.nodes())
        #
        #
        # # Go on and search for kernels
        # kernels = []
        # for sg in subgraphs:
        #     # Top Level State: determine kernels in state
        #     # We do this by looking at
        #     if state.parent.parent == None:
        #         num_kernels, precedences = self.compute_kernels(sg)
        #         if num_kernels > 1:
        #             print("---- Found more than one kernel in a subgraph! ----")
        #             kernels.extend(self._kernels_subgraphs(sg, precedences))
        #
        #             # TODO keep track of this precedences
        #         else:
        #             kernels.append(sg)
        #     else:
        #         kernels.append(sg)

        # Kernels are now sorted by topological order of their dependencies
        # Generate all kernels in this state
        if not self._in_device_code:

            state_parameters = []

            # As long as we generate the  kernels these two are populated
            kernel_host_header_stream = CodeIOStream()
            kernel_host_body_stream = CodeIOStream()

            for kern, kern_id in kernels:
                # TODO: this should return the same number of kernels
                subgraphs = dace.sdfg.concurrent_subgraphs(kern)
                if are_there_kernels:
                    assert len(subgraphs) == 1
                for i, sg in enumerate(subgraphs):
                    print("Kernel: ", kern_id, "Subgraph: ", i, " Nodes: ",
                          sg.nodes())
                shared_transients = set(sdfg.shared_transients())

                # Allocate global memory transients, unless they are shared with
                # other states
                all_transients = set(kern.all_transients())
                allocated = set(shared_transients)
                for node in kern.data_nodes():
                    data = node.desc(sdfg)
                    if node.data not in all_transients or node.data in allocated:
                        continue
                    if (data.storage == dtypes.StorageType.FPGA_Global
                            # and isinstance(data, dt.Array)
                            and not isinstance(data, dt.View)):
                        allocated.add(node.data)
                        self._dispatcher.dispatch_allocate(
                            sdfg, kern, state_id, node, function_stream,
                            callsite_stream)

                # Create a unique kernel name to avoid name clashes
                # If this kernels comes from a Nested SDFG, use that name also
                if sdfg.parent_nsdfg_node is not None:
                    kernel_name = f"{sdfg.parent_nsdfg_node.label}_{state.label}_{kern_id}_{sdfg.sdfg_id}"

                else:
                    kernel_name = f"{state.label}_{kern_id}_{sdfg.sdfg_id}"
                # Vitis HLS removes double underscores, which leads to a compilation
                # error down the road due to kernel name mismatch. Remove them here
                # to prevent this
                while True:
                    _kernel_name = kernel_name.replace("__", "_")
                    if kernel_name == _kernel_name:
                        break
                    else:
                        kernel_name = _kernel_name

                self._kernels_names_to_id[kernel_name] = kern_id

                # TODO: document all these type of streams
                # Call site is the cpu/program.cpp file

                # Generate kernel code
                self.generate_kernel(sdfg, state, kernel_name, subgraphs,
                                     function_stream, callsite_stream,
                                     kernel_host_header_stream,
                                     kernel_host_body_stream, state_parameters)

                # Emit the connections ini file
                if len(self._stream_connections) > 0:
                    ini_stream = CodeIOStream()
                    ini_stream.write('[connectivity]')
                    for _, (src, dst) in self._stream_connections.items():
                        ini_stream.write('stream_connect={}:{}'.format(
                            src, dst))
                    self._other_codes['link.ini'] = ini_stream

            kernel_args_call_host = []
            kernel_args_opencl = []

            # Include state in args
            kernel_args_opencl.append(f"{self._global_sdfg.name}_t *__state")
            kernel_args_call_host.append(f"__state")

            for is_output, argname, arg, _ in state_parameters:
                # Streams and Views are not passed as arguments
                if not isinstance(arg, dt.Stream) and not isinstance(
                        arg, dt.View):
                    kernel_args_call_host.append(
                        FPGACodeGen.make_host_parameter(argname, arg))
                    kernel_args_opencl.append(
                        FPGACodeGen.make_opencl_parameter(argname, arg))

            kernel_args_call_host = dtypes.deduplicate(kernel_args_call_host)
            kernel_args_opencl = dtypes.deduplicate(kernel_args_opencl)

            ## Generate the global function here

            # TODO: add profiling
            kernel_host_stream = CodeIOStream()
            host_function_name = f"__dace_runstate_{sdfg.sdfg_id}_{state.name}_{state_id}"
            function_stream.write("\n\nDACE_EXPORTED void {}({});\n\n".format(
                host_function_name, ", ".join(kernel_args_opencl)))
            # Write OpenCL host function
            kernel_host_stream.write(
                f"""\
DACE_EXPORTED void {host_function_name}({', '.join(kernel_args_opencl)}) {{
      hlslib::ocl::Program program = __state->fpga_context->Get().CurrentlyLoadedProgram();""")
            # Create a vector to collect all events that are being generate to facilitate
            # joining before exiting this state
            kernel_host_stream.write(f"std::vector<cl::Event> all_events;", )

            kernel_host_stream.write(kernel_host_header_stream.getvalue() +
                                     kernel_host_body_stream.getvalue())

            ## Wait for all the events
            kernel_host_stream.write(" cl::Event::waitForEvents(all_events);")

            kernel_host_stream.write("}\n")

            callsite_stream.write("{}({});".format(
                host_function_name, ", ".join(kernel_args_call_host)))

            # Store code strings to be passed to compilation phase
            # self._kernel_codes.append((kernel_name, kernel_stream.getvalue()))

            self._host_codes.append(
                (kernel_name, kernel_host_stream.getvalue()))



        else:  # self._in_device_code == True

            for kern, kern_id in kernels:
                to_allocate = dace.sdfg.local_transients(sdfg, kern, None)
                allocated = set()

                for node in kern.data_nodes():
                    data = node.desc(sdfg)
                    if node.data not in to_allocate or node.data in allocated:
                        continue
                    # Make sure there are no global transients in the nested state
                    # that are thus not gonna be allocated
                    if data.storage == dtypes.StorageType.FPGA_Global and not isinstance(
                            data, dt.View):
                        raise cgx.CodegenError(
                            "Cannot allocate global memory from device code.")
                    allocated.add(node.data)
                    # Allocate transients
                    self._dispatcher.dispatch_allocate(sdfg, state, state_id,
                                                       node, function_stream,
                                                       callsite_stream)

                self.generate_nested_state(sdfg, state, state.label,
                                           [k[0] for k in kernels],
                                           function_stream, callsite_stream)

        # if not self._in_device_code:
        #     for kn in self._kernels_names_to_id.keys():
        #         #TODO refactor: should I put it in a seperate function? Where?
        #         # ADD cleanup
        #         callsite_stream.write(f"thread___dace_runkernel_{kn}.join();")

        # for kern_id, kern in enumerate(kernels):
        #
        #     #TODO: use the real kernel_id
        #
        #
        #     if not self._in_device_code:
        #
        #         # Allocate global memory transients, unless they are shared with
        #         # other states
        #         all_transients = set(kern.all_transients())
        #         allocated = set(shared_transients)
        #         for node in kern.data_nodes():
        #             data = node.desc(sdfg)
        #             if node.data not in all_transients or node.data in allocated:
        #                 continue
        #             if (data.storage == dtypes.StorageType.FPGA_Global
        #                     # and isinstance(data, dt.Array)
        #                     and not isinstance(data, dt.View)):
        #                 allocated.add(node.data)
        #                 self._dispatcher.dispatch_allocate(
        #                     sdfg, kern, state_id, node, function_stream,
        #                     callsite_stream)
        #
        #         # Create a unique kernel name to avoid name clashes
        #         # If this kernels comes from a Nested SDFG, use that name also
        #         if sdfg.parent_nsdfg_node is not None:
        #             kernel_name = f"{sdfg.parent_nsdfg_node.label}_{state.label}_{kern_id}_{sdfg.sdfg_id}"
        #         else:
        #             kernel_name = f"{state.label}_{kern_id}_{sdfg.sdfg_id}"
        #         # Vitis HLS removes double underscores, which leads to a compilation
        #         # error down the road due to kernel name mismatch. Remove them here
        #         # to prevent this
        #         while True:
        #             _kernel_name = kernel_name.replace("__", "_")
        #             if kernel_name == _kernel_name:
        #                 break
        #             else:
        #                 kernel_name = _kernel_name
        #
        #         # Generate kernel code
        #         print("Generating kernel: ", kern_id)
        #         self.generate_kernel(sdfg, state, kern, kernel_name, subgraphs,
        #                              function_stream, callsite_stream)
        #         # Emit the connections ini file
        #         if len(self._stream_connections) > 0:
        #             ini_stream = CodeIOStream()
        #             ini_stream.write('[connectivity]')
        #             for _, (src, dst) in self._stream_connections.items():
        #                 ini_stream.write('stream_connect={}:{}'.format(
        #                     src, dst))
        #             self._other_codes['link.ini'] = ini_stream
        #
        #     else:  # self._in_device_code == True
        #         #TODO: not sure if this recognize kern. Previously it was state
        #         to_allocate = dace.sdfg.local_transients(sdfg, kern, None)
        #
        #         allocated = set()
        #         for node in kern.data_nodes():
        #             data = node.desc(sdfg)
        #             if node.data not in to_allocate or node.data in allocated:
        #                 continue
        #             # Make sure there are no global transients in the nested state
        #             # that are thus not gonna be allocated
        #             if data.storage == dtypes.StorageType.FPGA_Global and not isinstance(
        #                     data, dt.View):
        #                 raise cgx.CodegenError(
        #                     "Cannot allocate global memory from device code.")
        #             allocated.add(node.data)
        #             # Allocate transients
        #             self._dispatcher.dispatch_allocate(sdfg, state, state_id,
        #                                                node, function_stream,
        #                                                callsite_stream)
        #         #TODO complete
        #         self.generate_nested_state(sdfg, state, state.label, subgraphs,
        #                                    function_stream, callsite_stream)

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

    def make_parameters(self, sdfg, state, subgraphs):
        """
        Determines the parameters that must be passed to the passed list of
        subgraphs, as well as to the global kernel.
        :return: A tuple with the following six entries:
                 - Data container parameters that should be passed from the
                   host to the FPGA kernel.
                 - Data containers that are local to the kernel, but must be
                   allocated by the host prior to invoking the kernel.
                 - A dictionary mapping from each processing element subgraph
                   to which parameters it needs (from the total list of
                   parameters).
                 - Parameters that must be passed to the kernel from the host,
                   but that do not exist before the CPU calls the kernel
                   wrapper.
                 - A dictionary of which memory interfaces should be assigned to
                   which memory banks.
                 - External streams that connect different FPGA kernels, and
                   must be defined during the compilation flow.
        """

        # Get a set of data nodes that are shared across subgraphs
        shared_data = self.shared_data(subgraphs)
        # Transients that are accessed in other states in this SDFG
        used_outside = sdfg.shared_transients()

        # Build a dictionary of arrays to arbitrary data nodes referring to
        # them, needed to trace memory bank assignments and to pass to the array
        # allocator
        data_to_node: Dict[str:dace.nodes.Node] = {}

        global_data_parameters = set()
        # Count appearances of each global array to create multiple interfaces
        global_interfaces: Dict[str, int] = collections.defaultdict(int)

        top_level_local_data = set()
        subgraph_parameters = collections.OrderedDict()  # {subgraph: [params]}
        nested_global_transients = set()
        # [(Is an output, dataname string, data object, interface)]
        external_streams: Set[tuple[bool, str, dt, dict[str, int]]] = set()

        # Mapping from global arrays to memory interfaces
        bank_assignments: Dict[str, str] = {}

        # Mapping from symbol to a unique parameter tuple
        all_symbols = {
            k: (False, k, dt.Scalar(v), None)
            for k, v in sdfg.symbols.items() if k not in sdfg.constants
        }
        # Symbols that will be passed as parameters to the top-level kernel
        global_symbols = set()

        # Sorting by name, then by input/output, then by interface id
        sort_func = lambda t: f"{t[1]}{t[0]}{t[3]}"

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
                # Check if the node is connected to an RTL tasklet, in which
                # case it should be an external stream
                dsts = [e.dst for e in state.out_edges(n)]
                srcs = [e.src for e in state.in_edges(n)]
                tasks = [
                    t for t in dsts + srcs if isinstance(t, dace.nodes.Tasklet)
                ]
                external = any([
                    t.language == dtypes.Language.SystemVerilog for t in tasks
                ])
                if external:
                    external_streams |= {
                        (True, e.data.data, subsdfg.arrays[e.data.data], None)
                        for e in state.out_edges(n)
                        if isinstance(subsdfg.arrays[e.data.data], dt.Stream)
                    }
                else:
                    candidates += [(False, e.data.data,
                                    subsdfg.arrays[e.data.data])
                                   for e in state.in_edges(n)]
            for n in subgraph.sink_nodes():
                # Check if the node is connected to an RTL tasklet, in which
                # case it should be an external stream
                dsts = [e.dst for e in state.out_edges(n)]
                srcs = [e.src for e in state.in_edges(n)]
                tasks = [
                    t for t in dsts + srcs if isinstance(t, dace.nodes.Tasklet)
                ]
                external = any([
                    t.language == dtypes.Language.SystemVerilog for t in tasks
                ])
                if external:
                    external_streams |= {
                        (False, e.data.data, subsdfg.arrays[e.data.data], None)
                        for e in state.in_edges(n)
                        if isinstance(subsdfg.arrays[e.data.data], dt.Stream)
                    }
                else:
                    candidates += [(True, e.data.data,
                                    subsdfg.arrays[e.data.data])
                                   for e in state.out_edges(n)]
            # Find other data nodes that are used internally
            for n, scope in subgraph.all_nodes_recursive():
                if isinstance(n, dace.sdfg.nodes.AccessNode):
                    # Add nodes if they are outer-level, or an inner-level
                    # transient (inner-level inputs/outputs are just connected
                    # to data in the outer layers, whereas transients can be
                    # independent).
                    # Views are not nested global transients
                    if scope == subgraph or n.desc(scope).transient:
                        if scope.out_degree(n) > 0:
                            candidates.append((False, n.data, n.desc(scope)))
                        if scope.in_degree(n) > 0:
                            candidates.append((True, n.data, n.desc(scope)))
                        if scope != subgraph:
                            if (isinstance(n.desc(scope), dt.Array)
                                    and n.desc(scope).storage
                                    == dtypes.StorageType.FPGA_Global
                                    and not isinstance(n.desc(scope), dt.View)):
                                nested_global_transients.add(n)
            subgraph_parameters[subgraph] = set()
            # For each subgraph, keep a listing of array to current interface ID
            data_to_interface: Dict[str, int] = {}

            # Differentiate global and local arrays. The former are allocated
            # from the host and passed to the device code, while the latter are
            # (statically) allocated on the device side.
            for is_output, dataname, desc in candidates:
                # Ignore views, as these never need to be explicitly passed
                if isinstance(desc, dt.View):
                    continue
                # Only distinguish between inputs and outputs for arrays
                if not isinstance(desc, dt.Array):
                    is_output = None
                # If this is a global array, assign the correct interface ID and
                # memory interface (e.g., DDR or HBM bank)
                if (isinstance(desc, dt.Array)
                        and desc.storage == dtypes.StorageType.FPGA_Global):
                    if dataname in data_to_interface:
                        interface_id = data_to_interface[dataname]
                    else:
                        # Get and update global memory interface ID
                        interface_id = global_interfaces[dataname]
                        global_interfaces[dataname] += 1
                        data_to_interface[dataname] = interface_id
                    # Collect the memory bank specification, if present, by
                    # traversing outwards to where the data container is
                    # actually allocated
                    inner_node = data_to_node[dataname]
                    trace = utils.trace_nested_access(inner_node, subgraph,
                                                      sdfg)
                    bank = None
                    for (trace_in, trace_out), _, _, trace_sdfg in trace:
                        trace_node = trace_in or trace_out
                        trace_name = trace_node.data
                        trace_desc = trace_node.desc(trace_sdfg)
                        if "bank" in trace_desc.location:
                            trace_bank = trace_desc.location["bank"]
                            if (bank is not None and bank != trace_bank):
                                raise cgx.CodegenError(
                                    "Found inconsistent memory bank "
                                    f"specifier for {trace_name}.")
                            bank = trace_bank
                    # Make sure the array has been allocated on this bank in the
                    # outermost scope
                    if bank is not None:
                        outer_node = trace[0][0][0] or trace[0][0][1]
                        outer_desc = outer_node.desc(trace[0][2])
                        if ("bank" not in outer_desc.location or
                                str(outer_desc.location["bank"]) != str(bank)):
                            raise cgx.CodegenError(
                                "Memory bank allocation must be present on "
                                f"outermost data descriptor {outer_node.data} "
                                "to be allocated correctly.")
                    bank_assignments[dataname] = bank
                else:
                    interface_id = None
                if (not desc.transient
                        or desc.storage == dtypes.StorageType.FPGA_Global
                        or dataname in used_outside):
                    # Add the data as a parameter to this PE
                    subgraph_parameters[subgraph].add(
                        (is_output, dataname, desc, interface_id))
                    # Global data is passed from outside the kernel
                    global_data_parameters.add(
                        (is_output, dataname, desc, interface_id))
                elif dataname in shared_data:
                    # Add the data as a parameter to this PE
                    subgraph_parameters[subgraph].add(
                        (is_output, dataname, desc, interface_id))
                    # Must be allocated outside PEs and passed to them
                    top_level_local_data.add(dataname)
            # Order by name
            subgraph_parameters[subgraph] = list(
                sorted(subgraph_parameters[subgraph], key=sort_func))
            # Append symbols used in this subgraph
            for k in sorted(subgraph.free_symbols):
                if k not in sdfg.constants:
                    param = all_symbols[k]
                    subgraph_parameters[subgraph].append(param)
                    global_symbols.add(param)

        # Order by name
        global_data_parameters = list(
            sorted(global_data_parameters, key=sort_func))
        global_data_parameters += sorted(global_symbols, key=sort_func)
        external_streams = list(sorted(external_streams, key=sort_func))
        nested_global_transients = list(sorted(nested_global_transients))

        stream_names = {sname for _, sname, _, _ in external_streams}
        top_level_local_data = [
            data_to_node[name] for name in sorted(top_level_local_data)
            if name not in stream_names
        ]

        return (global_data_parameters, top_level_local_data,
                subgraph_parameters, nested_global_transients, bank_assignments,
                external_streams)

    def generate_nested_state(self, sdfg, state, nest_name, subgraphs,
                              function_stream, callsite_stream):

        for sg in subgraphs:
            self._dispatcher.dispatch_subgraph(sdfg,
                                               sg,
                                               sdfg.node_id(state),
                                               function_stream,
                                               callsite_stream,
                                               skip_entry_node=False)

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

        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       declaration_stream, allocation_stream):

        result_decl = StringIO()
        result_alloc = StringIO()
        nodedesc = node.desc(sdfg)
        arrsize = nodedesc.total_size
        is_dynamically_sized = dace.symbolic.issymbolic(arrsize, sdfg.constants)
        dataname = node.data

        if not isinstance(nodedesc, dt.Stream):
            # Unless this is a Stream, if the variable has been already defined we can return
            # For Streams, we still allocate them to keep track of their names across
            # nested SDFGs (needed by Intel FPGA backend for channel mangling)
            try:
                self._dispatcher.defined_vars.get(dataname)
                return
            except KeyError:
                pass  # The variable was not defined,  we can continue

        allocname = cpp.ptr(dataname, nodedesc)

        if isinstance(nodedesc, dt.View):
            return self.allocate_view(sdfg, dfg, state_id, node,
                                      function_stream, declaration_stream,
                                      allocation_stream)
        elif isinstance(nodedesc, dt.Stream):

            if not self._in_device_code:
                raise cgx.CodegenError(
                    "Cannot allocate FIFO from CPU code: {}".format(node.data))

            if is_dynamically_sized:
                raise cgx.CodegenError(
                    "Arrays of streams cannot have dynamic size on FPGA")

            try:
                buffer_size = dace.symbolic.evaluate(nodedesc.buffer_size,
                                                     sdfg.constants)
            except TypeError:
                raise cgx.CodegenError(
                    "Buffer length of stream cannot have dynamic size on FPGA")

            if buffer_size < 1:
                raise cgx.CodegenError("Streams cannot be unbounded on FPGA")

            # Language-specific implementation
            ctype, is_global = self.define_stream(nodedesc.dtype, buffer_size,
                                                  dataname, arrsize,
                                                  function_stream, result_decl)

            # defined type: decide whether this is a stream array or a single stream
            def_type = DefinedType.StreamArray if cpp.sym2cpp(
                arrsize) != "1" else DefinedType.Stream
            if is_global:
                self._dispatcher.defined_vars.add_global(
                    dataname, def_type, ctype)
            else:
                self._dispatcher.defined_vars.add(dataname, def_type, ctype)

        elif isinstance(nodedesc, dt.Array):

            if nodedesc.storage == dtypes.StorageType.FPGA_Global:

                if self._in_device_code:

                    if nodedesc not in self._allocated_global_arrays:
                        raise RuntimeError("Cannot allocate global array "
                                           "from device code: {} in {}".format(
                                               node.label, sdfg.name))

                else:
                    if isinstance(nodedesc, dt.Array):

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
                                f"hlslib::ocl::MemoryBank::bank{bank}, ")

                        # Define buffer, using proper type
                        result_decl.write(
                            "hlslib::ocl::Buffer <{}, hlslib::ocl::Access::readWrite> {};"
                            .format(nodedesc.dtype.ctype, dataname))
                        result_alloc.write(
                            "{} = __state->fpga_context->Get()."
                            "MakeBuffer<{}, hlslib::ocl::Access::readWrite>"
                            "({}{});".format(allocname, nodedesc.dtype.ctype,
                                             memory_bank_arg,
                                             cpp.sym2cpp(arrsize)))
                        self._dispatcher.defined_vars.add(
                            dataname, DefinedType.Pointer,
                            'hlslib::ocl::Buffer <{}, hlslib::ocl::Access::readWrite>'
                            .format(nodedesc.dtype.ctype))
            elif (nodedesc.storage in (dtypes.StorageType.FPGA_Local,
                                       dtypes.StorageType.FPGA_Registers,
                                       dtypes.StorageType.FPGA_ShiftRegister)):

                if not self._in_device_code:
                    raise cgx.CodegenError(
                        "Tried to allocate local FPGA memory "
                        "outside device code: {}".format(dataname))
                if is_dynamically_sized:
                    raise ValueError(
                        "Dynamic allocation of FPGA "
                        "fast memory not allowed: {}, size {}".format(
                            dataname, arrsize))

                generate_scalar = cpp.sym2cpp(arrsize) == "1"

                if generate_scalar:
                    # Language-specific
                    ctype = self.make_vector_type(nodedesc.dtype, False)
                    define_str = "{} {};".format(ctype, dataname)
                    result_decl.write(define_str)
                    self._dispatcher.defined_vars.add(dataname,
                                                      DefinedType.Scalar, ctype)
                else:
                    # Language-specific
                    if (nodedesc.storage ==
                            dtypes.StorageType.FPGA_ShiftRegister):
                        self.define_shift_register(dataname, nodedesc, arrsize,
                                                   function_stream, result_decl,
                                                   sdfg, state_id, node)
                    else:
                        self.define_local_array(dataname, nodedesc, arrsize,
                                                function_stream, result_decl,
                                                sdfg, state_id, node)

            else:
                raise NotImplementedError("Unimplemented storage type " +
                                          str(nodedesc.storage))

        elif isinstance(nodedesc, dt.Scalar):

            result_decl.write("{} {};\n".format(nodedesc.dtype.ctype, dataname))
            self._dispatcher.defined_vars.add(dataname, DefinedType.Scalar,
                                              nodedesc.dtype.ctype)

        else:
            raise TypeError("Unhandled data type: {}".format(
                type(nodedesc).__name__))

        declaration_stream.write(result_decl.getvalue(), sdfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), sdfg, state_id, node)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass  # Handled by destructor

    def compute_kernels(self,
                        state: dace.SDFGState,
                        default_kernel=0,
                        default_event=0):
        """ Annotates  state nodes (and all nested ones) to include a `_kernel_id`
            field. This field is applied to all FPGA maps, tasklets, and copies
            that can be executed in parallel in separate kernels.

            :param sdfg: The sdfg to analyze.
            :param default_kernel: The Kernel ID to start counting from (used
                                   in recursion to nested SDFGs).
            :param default_event: The event ID to start counting from (used
                                  in recursion to nested SDFGs).
            :return: 2-tuple of the number of streams, events to create.
        """

        # TODO:
        #  - control  this by some dace configuration flag?
        #  - Kernel numbering can be misleading (probably not solvable)
        #  - how to deal with AccessNode: Should they have a PE id? they can be accessed by multiple PEs?
        kernels = 0
        concurrent_kernels = 0
        sdfg = state.parent

        def increment(kernel_id):
            if concurrent_kernels > 0:
                return (kernel_id + 1) % concurrent_kernels
            return kernel_id + 1

        # Keep track of PEs per state
        state_kernels = []
        state_subsdfg_events = []
        max_events = 0

        # TODO: keep track of precedences among PEs

        precedences = dict()
        # if not is_fpga_kernel(state.parent, state):
        # if this is not an FPGA Kernel, we can skip PEs detection
        # TODO: FIX THIS OFC
        # state_kernels.append(0)
        # state_subsdfg_events.append(max_events)
        # return 1, precedences  # No kernel in here

        # Start by annotating source nodes

        # In the following:
        # node._kernel: Kernel ID associated to the node
        # node._kernel_childpath: if present indicates if a node is "parent" of more than one Kernel

        source_nodes = state.source_nodes()

        # Concurrency can only be found in each state
        max_kernels = default_kernel
        # max_events = default_event
        assigned = set()  # tmp, for check
        # First step: assign a diffeent Kernel ID
        # to each source node which is not an AccessNode
        for i, node in enumerate(source_nodes):
            if isinstance(node, nodes.AccessNode):
                continue

            # ATM we are not recurring into NestedSDFG. They are treated as a kernel
            # if isinstance(node, nodes.NestedSDFG):
            #     if node.schedule == dtypes.ScheduleType.FPGA_Device:
            #         continue
            node._kernel = max_kernels
            print("Source node ", node, " has associated PE: ", max_kernels)
            node._kernel_childpath = False
            max_kernels = increment(max_kernels)
            assigned.add(node)

        # Consecutive nodes that are not crossroads can be in
        # the same Kernel
        scopes = state.scope_dict()

        for e in state.dfs_edges(source_nodes):
            if hasattr(e.dst, '_kernel'):
                # Node has been already visited (THIS SHOULD NEVER OCCUR WITH DFS_EDGES, there is something strange)
                import pdb
                pdb.set_trace()
                assert (False)
                continue

            def trace_back_edge(edge, state, look_for_kernel_id=False):
                '''
                Given ad edge, this traverses the edges backwards.
                The goal is to understand if the backward path has some compute in it or not
                :return:
                '''
                # TODO: do reverse DFG till we reach another kernel boundaries
                # TODO: should we return this?
                backward_path = [edge]

                # If empty memlet, return False
                if (edge.src_conn is None and edge.dst_conn is None
                        and edge.data.is_empty()):
                    return False

                # Continue Until we find some compute part
                curedge = edge
                source_nodes = state.source_nodes()
                while not curedge.src in source_nodes:

                    if not look_for_kernel_id:
                        if isinstance(
                                curedge.src,
                            (nodes.EntryNode, nodes.ExitNode, nodes.CodeNode)):
                            # We can stop here, there is a Scope which will contain some compute, or a tasklet/libnode
                            return True
                    elif hasattr(curedge.src, "_kernel"):
                        # Found a node with a kernel id. Use that
                        return curedge.src._kernel
                    next_edge = next(e for e in state.in_edges(curedge.src))
                    curedge = next_edge

                # We didn't return before
                if not look_for_kernel_id:
                    return False
                else:
                    return curedge.src._kernel if hasattr(
                        curedge.src, "_kernel") else None

            if hasattr(e.src, '_kernel'):
                kernel = e.src._kernel

                if (isinstance(e.dst, nodes.AccessNode)
                        and isinstance(sdfg.arrays[e.dst.data], dt.View)):
                    # Skip views
                    e.dst._kernel = kernel
                    e.dst._kernel_childpath = False
                    assert e.dst not in assigned
                    assigned.add(e.dst)
                    continue

                # Does this node need to be in another Kernel?
                # TODO: find a more robust way of checking this
                # If this node is a crossroad node (has more than one predecessor)
                #  than it should be on a seperate Kernel
                # This is a crossroad node: A node with more than one precessor
                # and its predecessor contains some compute (so not just access nodes)

                # TODO: corner cases
                # - this is an exit node
                # - Systolic array? Not sure about this
                # - If i am in a Scope, then there is no need to check, we keep the same kernel
                # - TODO: we should not do this for Access Nodes? Maybe just code node and Entry Node
                if len(list(state.predecessors(e.dst))) > 1 and not isinstance(
                        e.dst, nodes.ExitNode) and scopes[e.dst] == None:
                    # Loop over all predecessor (except this edge) and see if there is some compute inside
                    crossroad_node = False
                    for pred_edge in state.in_edges(e.dst):
                        if pred_edge != e and trace_back_edge(pred_edge, state):
                            print(
                                f"{e.dst} is crossroad node because of edge {pred_edge}"
                            )
                            crossroad_node = True
                            break

                    if crossroad_node:
                        print(
                            "Node ", e.dst,
                            " has more than one predecessor. Put it in another Kernel"
                        )
                        if isinstance(e.dst, nodes.ExitNode):
                            print("!!!!!! But this was an Exit Node!!!!!")
                        kernel = max_kernels
                        max_kernels = increment(max_kernels)
                # elif e.src._kernel_childpath == True:
                #     import pdb
                #     pdb.set_trace()
                #     kernel = max_kernels
                #     max_kernels = increment(max_kernels)
                #
                # e.src._kernel_childpath = True

                # Do not create multiple kernels within FPGA scopes
                if (isinstance(e.src, nodes.EntryNode)
                        and e.src.schedule == dtypes.ScheduleType.FPGA_Device):
                    e.src._kernel_childpath = False
                elif state.entry_node(e.src) is not None:
                    parent = state.entry_node(e.src)
                    if parent.schedule == dtypes.ScheduleType.FPGA_Device:
                        e.src._kernel_childpath = False
            else:

                # From this edge we don't have any kernel id.
                # Look up for the other predecessors node, if any of them has a kernel
                # id, use that. Since we are in a DFS we visit this only once

                # TODO: guardare al nodo immediatamente successivo non va bene
                # dovremmo risalire e vedere se qualcuno c'e' l'ha
                # Onestamente e' un po' brutto fatto cosi. Forse dovremmo cambiare modo di visitare
                # il grafo
                # TODO refactor

                #TODO: controlla solo i compute, forse basta?

                for pred_edge in state.in_edges(e.dst):
                    if pred_edge != e:
                        kernel = trace_back_edge(pred_edge,
                                                 state,
                                                 look_for_kernel_id=True)
                        if kernel is not None:
                            print("Reuse kernel ID from a parent for ", e.dst)
                            break
                else:
                    # New PE: this should not have any predecessor
                    print("Use a new Kernel id for ", e.dst)
                    kernel = max_kernels
                    if (isinstance(e.dst, nodes.AccessNode)
                            and isinstance(sdfg.arrays[e.dst.data], dt.View)):
                        # Skip views
                        pass
                    else:
                        max_kernels = increment(max_kernels)
            assert (e.dst not in assigned)
            assigned.add(e.dst)
            e.dst._kernel = kernel
            print(e.dst, " has associated kernel ID ", kernel)

            if not hasattr(e.dst, '_kernel_childpath'):
                e.dst._kernel_childpath = False
            # if isinstance(e.dst, nodes.NestedSDFG):
            # We don't recur on NESTED SDFG
            # if e.dst.schedule == dtypes.ScheduleType.FPGA_Device:
            #     Recur into the nested SDFG
            # print("Recurring on nested SDFG", e.dst.sdfg.name)
            # max_kernels, _ = self._compute_kernels(e.dst.sdfg, e.dst._kernel, max_events +1)

        # do another pass and keep track of precedences among Kernels
        for node in state.nodes():
            if hasattr(node, '_kernel'):

                this_kernel = node._kernel
                # get all predecessors and see their associated kernel ID
                for pred in state.predecessors(node):
                    if hasattr(pred, '_kernel') and pred._kernel != this_kernel:
                        if this_kernel not in precedences:
                            precedences[this_kernel] = set()
                        precedences[this_kernel].add(pred._kernel)

        # TODO: are events useful?
        state_kernels.append(max_kernels if concurrent_kernels ==
                             0 else concurrent_kernels)
        state_subsdfg_events.append(max_events)

        # Cleanup Kernels in the following cases:
        # For the moment being we are assuming that everything here is for FPGA Kernels
        # TODO: this may change
        # DO we want to recur?
        for node, graph in state.all_nodes_recursive():
            if isinstance(graph, dace.sdfg.SDFGState):
                cur_sdfg = graph.parent

                if (isinstance(node, (nodes.EntryNode, nodes.ExitNode))
                        and node.schedule == dtypes.ScheduleType.FPGA_Device):
                    # Node must have an associated PE, remove childpath and continue
                    if hasattr(node, '_kernel_childpath'):
                        delattr(node, '_kernel_childpath')
                    continue

                if not is_fpga_kernel(cur_sdfg, graph):
                    # Not an FPGA kernel: skip
                    # TODO: relax this
                    continue

                for e in graph.all_edges(node):
                    path = graph.memlet_path(e)
                    # If leading from/to a FPGA memory node, keep the same Kernel IDs
                    if ((isinstance(path[0].src, nodes.AccessNode)
                         and path[0].src.desc(cur_sdfg).storage
                         == dtypes.StorageType.FPGA_Global)
                            or (isinstance(path[-1].dst, nodes.AccessNode)
                                and path[-1].dst.desc(cur_sdfg).storage
                                == dtypes.StorageType.FPGA_Global)):
                        # print("BREAK Leading to  FPGA Global Mem")
                        break
                    # If leading from/to a FPGA tasklet, keep stream
                    if ((isinstance(path[0].src, nodes.CodeNode)
                         and is_fpga_kernel(cur_sdfg, graph))
                            or (isinstance(path[-1].dst, nodes.CodeNode)
                                and is_fpga_kernel(cur_sdfg, graph))):
                        # print("BREAK FPGA KERNEL")
                        break
                else:  # If we did not break, we do not need another kernel
                    # print("Remove pe from", node)
                    if hasattr(node, '_kernel'):
                        delattr(node, '_kernel')
                # In any case, remove childpath
                if hasattr(node, '_kernel_childpath'):
                    delattr(node, '_kernel_childpath')

        # Compute maximal number of events by counting edges (within the same
        # state) that point from one stream to another
        # state_events = []
        # for i, state in enumerate(sdfg.nodes()):
        #     events = state_subsdfg_events[i]
        #
        #     for e in state.edges():
        #         if hasattr(e.src, '_kernel'):
        #             # If there are two or more Kernels involved in this
        #             # edge, or the destination is unrelated to CUDA
        #             if (not hasattr(e.dst, '_kernel')
        #                     or e.src._kernel != e.dst._kernel):
        #                 for mpe in state.memlet_path(e):
        #                     mpe._kernel_event = events
        #                 events += 1
        #
        #     state_events.append(events)

        # Maximum over all states
        max_kernels = max(state_kernels)
        # max_events = max(state_events)
        print("************ Kernel precedences: ", precedences)
        return max_kernels, precedences

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
                          and dst_storage == dtypes.StorageType.FPGA_Global)
        device_to_host = (data_to_data
                          and src_storage == dtypes.StorageType.FPGA_Global
                          and dst_storage in _CPU_STORAGE_TYPES)
        device_to_device = (data_to_data
                            and src_storage == dtypes.StorageType.FPGA_Global
                            and dst_storage == dtypes.StorageType.FPGA_Global)

        if (host_to_device or device_to_host) and self._in_device_code:
            raise RuntimeError(
                "Cannot copy between host and device from device")

        if (host_to_device or device_to_host
                or (device_to_device and not self._in_device_code)):

            dims = memlet.subset.dims()
            copy_shape = memlet.subset.bounding_box_size()
            offset = cpp.cpp_array_expr(sdfg, memlet, with_brackets=False)

            if (not sum(copy_shape) == 1
                    and (not isinstance(memlet.subset, subsets.Range)
                         or any([step != 1 for _, _, step in memlet.subset]))):
                raise NotImplementedError("Only contiguous copies currently "
                                          "supported for FPGA codegen.")

            if host_to_device or device_to_device:
                host_dtype = sdfg.data(src_node.data).dtype
                device_dtype = sdfg.data(dst_node.data).dtype
            elif device_to_host:
                device_dtype = sdfg.data(src_node.data).dtype
                host_dtype = sdfg.data(dst_node.data).dtype
            cast = False
            if not device_to_device and host_dtype != device_dtype:
                if ((isinstance(host_dtype, dace.vector)
                     or isinstance(device_dtype, dace.vector))
                        and host_dtype.base_type == device_dtype.base_type):
                    if ((host_to_device and memlet.data == src_node.data) or
                        (device_to_host and memlet.data == dst_node.data)):
                        if host_dtype.veclen > device_dtype.veclen:
                            copy_shape[-1] *= (host_dtype.veclen //
                                               device_dtype.veclen)
                        else:
                            copy_shape[-1] //= (device_dtype.veclen //
                                                host_dtype.veclen)
                    cast = True
                else:
                    raise TypeError(
                        "Memory copy type mismatch: {} vs {}".format(
                            host_dtype, device_dtype))

            copysize = " * ".join([
                cppunparse.pyexpr2cpp(dace.symbolic.symstr(s))
                for s in copy_shape
            ])

            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)

            if host_to_device:

                ptr_str = (cpp.ptr(src_node.data, src_nodedesc) +
                           (" + {}".format(offset)
                            if outgoing_memlet and str(offset) != "0" else ""))
                if cast:
                    ptr_str = "reinterpret_cast<{} const *>({})".format(
                        device_dtype.ctype, ptr_str)

                callsite_stream.write(
                    "{}.CopyFromHost({}, {}, {});".format(
                        cpp.ptr(dst_node.data, dst_nodedesc),
                        (offset if not outgoing_memlet else 0), copysize,
                        ptr_str), sdfg, state_id, [src_node, dst_node])

            elif device_to_host:

                ptr_str = (cpp.ptr(dst_node.data, dst_nodedesc) +
                           (" + {}".format(offset)
                            if outgoing_memlet and str(offset) != "0" else ""))
                if cast:
                    ptr_str = "reinterpret_cast<{} *>({})".format(
                        device_dtype.ctype, ptr_str)

                callsite_stream.write(
                    "{}.CopyToHost({}, {}, {});".format(
                        cpp.ptr(src_node.data, src_nodedesc),
                        (offset if outgoing_memlet else 0), copysize, ptr_str),
                    sdfg, state_id, [src_node, dst_node])

            elif device_to_device:

                callsite_stream.write(
                    "{}.CopyToDevice({}, {}, {}, {});".format(
                        cpp.ptr(src_node.data, src_nodedesc),
                        (offset if outgoing_memlet else 0), copysize,
                        cpp.ptr(dst_node.data, dst_nodedesc),
                        (offset if not outgoing_memlet else 0)), sdfg, state_id,
                    [src_node, dst_node])

        # Reject copying to/from local memory from/to outside the FPGA
        elif (data_to_data
              and (((src_storage in (dtypes.StorageType.FPGA_Local,
                                     dtypes.StorageType.FPGA_Registers,
                                     dtypes.StorageType.FPGA_ShiftRegister))
                    and dst_storage not in _FPGA_STORAGE_TYPES) or
                   ((dst_storage in (dtypes.StorageType.FPGA_Local,
                                     dtypes.StorageType.FPGA_Registers,
                                     dtypes.StorageType.FPGA_ShiftRegister))
                    and src_storage not in _FPGA_STORAGE_TYPES))):
            raise NotImplementedError(
                "Copies between host memory and FPGA "
                "local memory not supported: from {} to {}".format(
                    src_node, dst_node))

        elif data_to_data:

            if memlet.wcr is not None:
                raise NotImplementedError("WCR not implemented for copy edges")

            if src_storage == dtypes.StorageType.FPGA_ShiftRegister:
                raise NotImplementedError(
                    "Reads from shift registers only supported from tasklets.")

            # Try to turn into degenerate/strided ND copies
            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                cpp.memlet_copy_to_absolute_strides(self._dispatcher,
                                                    sdfg,
                                                    memlet,
                                                    src_node,
                                                    dst_node,
                                                    packed_types=True))

            dtype = src_node.desc(sdfg).dtype
            ctype = dtype.ctype

            if dst_storage == dtypes.StorageType.FPGA_ShiftRegister:
                if len(copy_shape) != 1:
                    raise ValueError(
                        "Only single-dimensional writes "
                        "to shift registers supported: {}{}".format(
                            dst_node.data, copy_shape))

            # Check if we are copying between vectorized and non-vectorized
            # types
            memwidth_src = src_node.desc(sdfg).veclen
            memwidth_dst = dst_node.desc(sdfg).veclen
            if memwidth_src < memwidth_dst:
                is_pack = True
                is_unpack = False
                packing_factor = memwidth_dst // memwidth_src
                if memwidth_dst % memwidth_src != 0:
                    raise ValueError(
                        "Destination vectorization width {} "
                        "is not divisible by source vectorization width {}.".
                        format(memwidth_dst, memwidth_src))
                self.converters_to_generate.add(
                    (False, vector_element_type_of(dtype).ctype,
                     packing_factor))
            elif memwidth_src > memwidth_dst:
                is_pack = False
                is_unpack = True
                packing_factor = memwidth_src // memwidth_dst
                if memwidth_src % memwidth_dst != 0:
                    raise ValueError(
                        "Source vectorization width {} is not divisible "
                        "by destination vectorization width {}.".format(
                            memwidth_dst, memwidth_src))
                self.converters_to_generate.add(
                    (True, vector_element_type_of(dtype).ctype, packing_factor))
            else:
                is_pack = False
                is_unpack = False
                packing_factor = 1

            # TODO: detect in which cases we shouldn't unroll
            register_to_register = (src_node.desc(sdfg).storage
                                    == dtypes.StorageType.FPGA_Registers
                                    or dst_node.desc(sdfg).storage
                                    == dtypes.StorageType.FPGA_Registers)

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
                    if (isinstance(node.desc(sdfg), dt.Array)
                            and node.desc(sdfg).storage in [
                                dtypes.StorageType.FPGA_Local,
                                dace.StorageType.FPGA_Registers
                            ]):
                        # Language-specific
                        self.generate_no_dependence_pre(callsite_stream, sdfg,
                                                        state_id, dst_node,
                                                        node.data)

            # Loop intro
            for i, copy_dim in enumerate(copy_shape):
                if copy_dim != 1:
                    if register_to_register:
                        # Language-specific
                        self.generate_unroll_loop_pre(callsite_stream, None,
                                                      sdfg, state_id, dst_node)
                    # If we are copying from a container to itself, and the memlet subsets do not intersect,
                    # then we can safely ignore loop carried dependencies

                    ignore_dependencies = src_node.data == dst_node.data and not dace.subsets.intersects(
                        memlet.src_subset, memlet.dst_subset)
                    if ignore_dependencies:
                        self.generate_no_dependence_pre(callsite_stream, sdfg,
                                                        state_id, dst_node)
                    callsite_stream.write(
                        "for (int __dace_copy{} = 0; __dace_copy{} < {}; "
                        "++__dace_copy{}) {{".format(i, i,
                                                     cpp.sym2cpp(copy_dim), i),
                        sdfg, state_id, dst_node)

                    if ignore_dependencies:
                        self.generate_no_dependence_post(
                            callsite_stream, sdfg, state_id, dst_node,
                            node.data)

                    if register_to_register:
                        # Language-specific
                        self.generate_unroll_loop_post(callsite_stream, None,
                                                       sdfg, state_id, dst_node)

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

            src_def_type, _ = self._dispatcher.defined_vars.get(src_node.data)
            dst_def_type, _ = self._dispatcher.defined_vars.get(dst_node.data)

            # Construct indices (if the length of the stride array is zero,
            # resolves to an empty string)
            src_index = " + ".join([
                "__dace_copy{}{}".format(
                    i, " * " + cpp.sym2cpp(stride) if stride != 1 else "")
                for i, stride in enumerate(src_strides) if copy_shape[i] != 1
            ])
            dst_index = " + ".join([
                "__dace_copy{}{}".format(
                    i, " * " + cpp.sym2cpp(stride) if stride != 1 else "")
                for i, stride in enumerate(dst_strides) if copy_shape[i] != 1
            ])
            if not src_index:
                src_index = "0"
            if not dst_index:
                dst_index = "0"

            # Language specific
            read_expr = self.make_read(src_def_type, dtype, src_node.label,
                                       src_expr, src_index, is_pack,
                                       packing_factor)

            # Language specific
            if dst_storage == dtypes.StorageType.FPGA_ShiftRegister:
                write_expr = self.make_shift_register_write(
                    dst_def_type, dtype, dst_node.label, dst_expr, dst_index,
                    read_expr, None, is_unpack, packing_factor, sdfg)
            else:
                write_expr = self.make_write(dst_def_type, dtype,
                                             dst_node.label, dst_expr,
                                             dst_index, read_expr, None,
                                             is_unpack, packing_factor)

            callsite_stream.write(write_expr)

            # Inject dependence pragmas (DACE semantics implies no conflict)
            for node in [src_node, dst_node]:
                if (isinstance(node.desc(sdfg), dt.Array)
                        and node.desc(sdfg).storage in [
                            dtypes.StorageType.FPGA_Local,
                            dace.StorageType.FPGA_Registers
                        ]):
                    # Language-specific
                    self.generate_no_dependence_post(callsite_stream, sdfg,
                                                     state_id, dst_node,
                                                     node.data)

            # Loop outtro
            for _ in range(num_loops):
                callsite_stream.write("}")

        else:

            self.generate_memlet_definition(sdfg, dfg, state_id, src_node,
                                            dst_node, edge, callsite_stream)

    @staticmethod
    def make_opencl_parameter(name, desc):
        if isinstance(desc, dt.Array):
            return (f"hlslib::ocl::Buffer<{desc.dtype.ctype}, "
                    f"hlslib::ocl::Access::readWrite> &{name}")
        else:
            return (desc.as_arg(with_types=True, name=name))

    @staticmethod
    def make_host_parameter(name, desc):
        # if isinstance(desc, dt.Array):
        #     # Buffer are passed by reference, but std::thread can not accept a reference
        #     # Use std::ref
        #     return (f"std::ref({name})")
        # else:
        #TODO remove, if we don't use thread is useless
        return (desc.as_arg(with_types=False, name=name))

    def get_next_scope_entries(self, sdfg, dfg, scope_entry):
        parent_scope_entry = dfg.entry_node(scope_entry)
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
                    dtypes.ScheduleType.Default, dtypes.ScheduleType.FPGA_Device
            ]:
                warnings.warn("Found schedule {} on {} node in FPGA code. "
                              "Ignoring.".format(node.schedule,
                                                 type(node).__name__))

            getattr(self, method_name)(sdfg, dfg, state_id, node,
                                       function_stream, callsite_stream)
        else:
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self

            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

            self._cpu_codegen.calling_codegen = old_codegen

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):

        if isinstance(src_node, dace.sdfg.nodes.CodeNode):
            src_storage = dtypes.StorageType.Register
            try:
                src_parent = dfg.entry_node(src_node)
            except KeyError:
                src_parent = None
            dst_schedule = (None
                            if src_parent is None else src_parent.map.schedule)
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, dace.sdfg.nodes.CodeNode):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.entry_node(dst_node)
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
                                              state.scope_children(), x.sdfg):
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

            # Define dynamic loop bounds variables (dynamic input memlets to
            # the MapEntry node)
            for e in dynamic_map_inputs(sdfg.node(state_id), node):
                if e.data.data != e.dst_conn:
                    callsite_stream.write(
                        self._cpu_codegen.memlet_definition(
                            sdfg, e.data, False, e.dst_conn,
                            e.dst.in_connectors[e.dst_conn]), sdfg, state_id,
                        node)

            # Pipeline innermost loops
            scope_children = dfg.scope_children()
            scope = scope_children[node]
            is_innermost = self._is_innermost(scope, scope_children, sdfg)

            # Generate custom iterators if this is a pipelined (and thus
            # flattened) loop
            if isinstance(node, dace.sdfg.nodes.PipelineEntry):
                for i in range(len(node.map.range)):
                    result.write("long {} = {};\n".format(
                        node.map.params[i], node.map.range[i][0]))
                for var, value in node.pipeline.additional_iterators.items():
                    result.write("long {} = {};\n".format(var, value))

            is_degenerate = []
            degenerate_values = []
            for begin, end, skip in node.map.range:
                # If we know at compile-time that a loop will only have a
                # single iteration, we can replace it with a simple assignment
                b, val = self._is_degenerate(begin, end, skip, sdfg)
                is_degenerate.append(b)
                degenerate_values.append(val)
            fully_degenerate = all(is_degenerate)

            # Being this a map (each iteration is independent), we can add pragmas to ignore dependencies on data
            # that is read/written inside this map, if there are no WCR. If there are no WCR at all, we can add
            # a more generic pragma to ignore all loop-carried dependencies.
            map_exit_node = dfg.exit_node(node)
            state = sdfg.nodes()[state_id]
            candidates_in = set()
            candidates_out = set()
            is_there_a_wcr = False
            # get data that is read/written
            for _, _, _, _, memlet in state.in_edges(node):
                if memlet.data is not None:
                    desc = sdfg.arrays[memlet.data]
                    if (isinstance(desc, dt.Array) and
                        (desc.storage == dtypes.StorageType.FPGA_Global
                         or desc.storage == dtypes.StorageType.FPGA_Local)
                            and memlet.wcr is None):
                        candidates_in.add(memlet.data)
                    elif memlet.wcr is not None:
                        is_there_a_wcr = True

            for _, _, _, _, memlet in state.out_edges(map_exit_node):
                if memlet.data is not None:
                    desc = sdfg.arrays[memlet.data]
                    if (isinstance(desc, dt.Array) and
                        (desc.storage == dtypes.StorageType.FPGA_Global
                         or desc.storage == dtypes.StorageType.FPGA_Local)
                            and memlet.wcr is None):
                        candidates_out.add(memlet.data)
                    elif memlet.wcr is not None:
                        is_there_a_wcr = True
            in_out_data = candidates_in.intersection(candidates_out)

            # add pragmas

            # Generate nested loops
            if not isinstance(node, dace.sdfg.nodes.PipelineEntry):

                for i, r in enumerate(node.map.range):

                    # Add pragmas
                    if not fully_degenerate and not is_degenerate[i]:
                        if node.map.unroll:
                            self.generate_unroll_loop_pre(
                                result, None, sdfg, state_id, node)
                        elif is_innermost:
                            self.generate_pipeline_loop_pre(
                                result, sdfg, state_id, node)
                            # Do not put pragma if this is degenerate (loop does not exist)
                            self.generate_flatten_loop_pre(
                                result, sdfg, state_id, node)
                        if not node.map.unroll:
                            if len(in_out_data) > 0 and is_there_a_wcr == False:
                                # add pragma to ignore all loop carried dependencies
                                self.generate_no_dependence_pre(
                                    result, sdfg, state_id, node)
                            else:
                                # add specific pragmas
                                for candidate in in_out_data:
                                    self.generate_no_dependence_pre(
                                        result, sdfg, state_id, node, candidate)

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
                                loop_var_type, var, cpp.sym2cpp(begin), var,
                                cpp.sym2cpp(end + 1), var, cpp.sym2cpp(skip)),
                            sdfg, state_id, node)

                    #Add unroll pragma
                    if not fully_degenerate and not is_degenerate[
                            i] and node.map.unroll:
                        self.generate_unroll_loop_post(result, None, sdfg,
                                                       state_id, node)

            else:
                pipeline = node.pipeline
                flat_it = pipeline.iterator_str()
                bound = pipeline.loop_bound_str()

                if len(in_out_data) > 0:
                    if is_there_a_wcr == False:
                        # add pragma to ignore all loop carried dependencies
                        self.generate_no_dependence_pre(result, sdfg, state_id,
                                                        node)
                    else:
                        # add specific pragmas
                        for candidate in in_out_data:
                            self.generate_no_dependence_pre(
                                result, sdfg, state_id, node, candidate)
                result.write(
                    "for (long {it} = 0; {it} < {bound}; ++{it}) {{\n".format(
                        it=flat_it, bound=node.pipeline.loop_bound_str()))
                if pipeline.init_size != 0:
                    result.write("const bool {} = {} < {};\n".format(
                        node.pipeline.init_condition(), flat_it,
                        cpp.sym2cpp(pipeline.init_size)))
                if pipeline.drain_size != 0:
                    result.write("const bool {} = {} >= {};\n".format(
                        node.pipeline.drain_condition(), flat_it,
                        bound + (" - " + cpp.sym2cpp(pipeline.drain_size)
                                 if pipeline.drain_size != 0 else "")))

            # Add pragmas
            if not fully_degenerate:
                if not node.map.unroll:
                    if is_innermost:
                        self.generate_pipeline_loop_post(
                            result, sdfg, state_id, node)
                        self.generate_flatten_loop_post(result, sdfg, state_id,
                                                        node)
                    # add pragmas for data read/written inside this map, but only for local arrays
                    for candidate in in_out_data:
                        if sdfg.arrays[
                                candidate].storage != dtypes.StorageType.FPGA_Global:
                            self.generate_no_dependence_post(
                                result, sdfg, state_id, node, candidate)

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(sdfg, sdfg.node(state_id),
                                                 node)
        allocated = set()
        for child in dfg.scope_children()[node]:
            if not isinstance(child, dace.sdfg.nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child, None,
                                               result)

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
            # ranges could have been defined in terms of floor/ceiling. Before printing the code
            # they are converted from a symbolic expression to a C++ compilable expression
            for it, r in reversed(list(zip(pipeline.params, pipeline.range))):
                callsite_stream.write(
                    "if ({it} >= {end}) {{\n{it} = {begin};\n".format(
                        it=it,
                        begin=dace.symbolic.symstr(r[0]),
                        end=dace.symbolic.symstr(r[1])))
            for it, r in zip(pipeline.params, pipeline.range):
                callsite_stream.write(
                    "}} else {{\n{it} += {step};\n}}\n".format(
                        it=it, step=dace.symbolic.symstr(r[2])))
            if len(cond) > 0:
                callsite_stream.write("}\n")
            callsite_stream.write("}\n}\n")
        else:
            self._cpu_codegen._generate_MapExit(sdfg, dfg, state_id, node,
                                                function_stream,
                                                callsite_stream)

    def generate_kernel(self, sdfg, state, kernel_name, subgraphs,
                        function_stream, callsite_stream,
                        kernel_host_header_stream, kernel_host_body_stream,
                        state_parameters):
        if self._in_device_code:
            raise cgx.CodegenError("Tried to generate kernel from device code")
        self._in_device_code = True
        self._cpu_codegen._packed_types = True
        kernel_stream = CodeIOStream()

        # Actual kernel code generation
        self.generate_kernel_internal(sdfg, state, kernel_name, subgraphs,
                                      kernel_stream, kernel_host_header_stream,
                                      kernel_host_body_stream, function_stream,
                                      callsite_stream, state_parameters)
        self._kernel_count = self._kernel_count + 1
        self._in_device_code = False
        self._cpu_codegen._packed_types = False

        # Store code strings to be passed to compilation phase
        self._kernel_codes.append((kernel_name, kernel_stream.getvalue()))

        self._allocated_global_arrays = set()

    def generate_modules(self, sdfg, state, kernel_name, subgraphs,
                         subgraph_parameters, module_stream, entry_stream,
                         host_stream):
        """Generate all PEs inside an FPGA Kernel"""

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
            labels = [
                n.label.replace(" ", "_") + f"_{state.node_id(n)}"
                for n in tasklet_list
            ]
            # If there are no tasklets, name it after access nodes in the
            # subgraph
            if len(labels) == 0:
                labels = [n.label.replace(" ", "_") for n in access_nodes]
            if len(labels) == 0:
                raise RuntimeError("Expected at least one tasklet or data node")
            module_name = "_".join(labels)

            self.generate_module(sdfg, state, kernel_name, module_name,
                                 subgraph, subgraph_parameters[subgraph],
                                 module_stream, entry_stream, host_stream)

    def generate_nsdfg_header(self, sdfg, state, state_id, node,
                              memlet_references, sdfg_label):
        return self._cpu_codegen.generate_nsdfg_header(sdfg,
                                                       state,
                                                       state_id,
                                                       node,
                                                       memlet_references,
                                                       sdfg_label,
                                                       state_struct=False)

    def generate_nsdfg_call(self, sdfg, state, node, memlet_references,
                            sdfg_label):
        return self._cpu_codegen.generate_nsdfg_call(sdfg,
                                                     state,
                                                     node,
                                                     memlet_references,
                                                     sdfg_label,
                                                     state_struct=False)

    def generate_nsdfg_arguments(self, sdfg, dfg, state, node):
        return self._cpu_codegen.generate_nsdfg_arguments(
            sdfg, state, dfg, node)

    def generate_host_function_boilerplate(self, sdfg, state, kernel_name,
                                           parameters, nested_global_transients,
                                           host_code_stream, header_stream,
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

        # Include state in args
        kernel_args_opencl.append(f"{self._global_sdfg.name}_t *__state")
        kernel_args_call_host.append(f"__state")

        for is_output, argname, arg, _ in parameters:
            # Streams and Views are not passed as arguments
            if not isinstance(arg, dt.Stream) and not isinstance(arg, dt.View):
                kernel_args_call_host.append(
                    FPGACodeGen.make_host_parameter(argname, arg))
                kernel_args_opencl.append(
                    FPGACodeGen.make_opencl_parameter(argname, arg))

        kernel_args_call_host = dtypes.deduplicate(kernel_args_call_host)
        kernel_args_opencl = dtypes.deduplicate(kernel_args_opencl)

        # add promise parameter to return to the caller the events associated to created kernels
        # kernel_args_opencl.append("std::promise<std::vector<cl::Event> *> promise")
        # kernel_args_opencl.append("std::vector<cl::Event> * wait_events")
        # promise_name = f"promise_{kernel_name}"
        # kernel_args_call_host.append(f"std::move({promise_name})")

        # host_function_name = "__dace_runkernel_{}".format(kernel_name)
        # Write OpenCL host function
        #         host_code_stream.write(
        #             """\
        # DACE_EXPORTED void {host_function_name}({kernel_args_opencl}) {{
        #   hlslib::ocl::Program program = __state->fpga_context->Get().CurrentlyLoadedProgram();"""
        #             .format(host_function_name=host_function_name,
        #                     kernel_args_opencl=", ".join(kernel_args_opencl)))
        #
        #         header_stream.write("\n\nDACE_EXPORTED void {}({});\n\n".format(
        #             host_function_name, ", ".join(kernel_args_opencl)))

        # callsite_stream.write("{}({});".format(
        #     host_function_name, ", ".join(kernel_args_call_host)))
        # Create a promise and a future to get from kernel launcing function the vector
        # of events associated to the kernel executed

        # kernel_id = self._kernels_names_to_id[kernel_name]
        # needs_synch = False
        # if kernel_id in self._kernels_dependencies:
        #     needs_synch = True
        #
        #
        # if needs_synch:
        #     #THIS MUST BE DONE IN THE VENDOR SPECIFIC FUCNTION
        #     callsite_stream.write(f"// {kernel_name} depends on {self._kernels_dependencies[kernel_id]}")
        #     # Build a vector containing all the events associated with the kernels from which this one depends
        #     def get_kernel_name(val):
        #         for key, value in self._kernels_names_to_id.items():
        #             if val == value:
        #                 return key
        #         raise RuntimeError("Error while generating kernel dependencies")
        #     kernel_deps_name = f"deps_{kernel_name}"
        #     callsite_stream.write(f"std::vector<cl::Event > *{kernel_deps_name} = new std::vector<cl::Event >();")
        #     for predecessor in self._kernels_dependencies[kernel_id]:
        #         pred_name = get_kernel_name(predecessor)
        #         # concatenate events from predecessor kernel
        #         callsite_stream.write(f"{kernel_deps_name}->insert({kernel_deps_name}->end(), events_{pred_name}->begin(), events_{pred_name}->end());")
        #
        #
        #     kernel_args_call_host.append(f"{kernel_deps_name}")
        # else:
        #     # must be always passed
        #     kernel_args_call_host.append(f"nullptr")
        # future_name = f"future_{kernel_name}"
        # events_name = f"events_{kernel_name}"
        # callsite_stream.write(f"std::promise<std::vector<cl::Event> *> {promise_name};")
        # callsite_stream.write(f"std::future<std::vector<cl::Event> *> {future_name} = {promise_name}.get_future();")
        # callsite_stream.write("std::thread thread_{}({}, {});".format(
        #     host_function_name, host_function_name, ", ".join(kernel_args_call_host)))

        # callsite_stream.write(f"std::vector<cl::Event> *  {events_name} = {future_name}.get();")
        # callsite_stream.write(f"std::cout << \"Kernel {kernel_name} has events: \" << {events_name}->size()<<std::endl;")
        # Any extra transients stored in global memory on the FPGA must now be
        # allocated and passed to the kernel
        for arr_node in nested_global_transients:
            self._dispatcher.dispatch_allocate(sdfg, state, None, arr_node,
                                               None, host_code_stream)

    def _generate_Tasklet(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen._generate_Tasklet(*args, codegen=self, **kwargs)

    def define_out_memlet(self, sdfg, state_dfg, state_id, src_node, dst_node,
                          edge, function_stream, callsite_stream):
        self._dispatcher.dispatch_copy(src_node, dst_node, edge, sdfg,
                                       state_dfg, state_id, function_stream,
                                       callsite_stream)

    def process_out_memlets(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(*args, codegen=self, **kwargs)

    def generate_tasklet_preamble(self, *args, **kwargs):
        # Fall back on CPU implementation
        self._cpu_codegen.generate_tasklet_preamble(*args, **kwargs)

    def generate_tasklet_postamble(self, sdfg, dfg, state_id, node,
                                   function_stream, before_memlets_stream,
                                   after_memlets_stream):
        # Inject dependency pragmas on memlets
        for edge in dfg.out_edges(node):
            datadesc = sdfg.arrays[edge.data.data]
            if (isinstance(datadesc, dt.Array)
                    and (datadesc.storage == dace.StorageType.FPGA_Local
                         or datadesc.storage == dace.StorageType.FPGA_Registers)
                    and not cpp.is_write_conflicted(dfg, edge)
                    and self._dispatcher.defined_vars.has(edge.src_conn)):

                self.generate_no_dependence_post(after_memlets_stream, sdfg,
                                                 state_id, node, edge.src_conn)

    def make_ptr_vector_cast(self, *args, **kwargs):
        return cpp.make_ptr_vector_cast(*args, **kwargs)

    def make_ptr_assignment(self, *args, **kwargs):
        return self._cpu_codegen.make_ptr_assignment(*args,
                                                     codegen=self,
                                                     **kwargs)
