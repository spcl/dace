# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import functools
import copy
import itertools
import os
import re
from six import StringIO
import numpy as np

import dace
from dace import registry, subsets, dtypes
from dace.codegen import cppunparse
from dace.config import Config
from dace.codegen import exceptions as cgx
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import make_absolute
from dace.codegen.targets import cpp, fpga
from dace.codegen.common import codeblock_to_cpp
from dace.codegen.tools.type_inference import infer_expr_type
from dace.frontend.python.astutils import rname, unparse
from dace.frontend import operations
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.sdfg import nodes, utils as sdutils
from dace.codegen.common import sym2cpp
from dace.sdfg import SDFGState
import dace.sdfg.utils as utils
from dace.symbolic import evaluate
from collections import defaultdict

REDUCTION_TYPE_TO_HLSLIB = {
    dace.dtypes.ReductionType.Min: "min",
    dace.dtypes.ReductionType.Max: "max",
    dace.dtypes.ReductionType.Sum: "+",
    dace.dtypes.ReductionType.Sub: "-",
    dace.dtypes.ReductionType.Product: "*",
    dace.dtypes.ReductionType.Div: "/",
    dace.dtypes.ReductionType.Logical_And: " && ",
    dace.dtypes.ReductionType.Bitwise_And: "&",
    dace.dtypes.ReductionType.Logical_Or: "||",
    dace.dtypes.ReductionType.Bitwise_Or: "|",
    dace.dtypes.ReductionType.Bitwise_Xor: "^"
}

REDUCTION_TYPE_TO_PYEXPR = {
    dace.dtypes.ReductionType.Min: "min({a}, {b})",
    dace.dtypes.ReductionType.Max: "max({a}, {b})",
    dace.dtypes.ReductionType.Sum: "{a} + {b}",
    dace.dtypes.ReductionType.Product: "*",
    dace.dtypes.ReductionType.Logical_And: " && ",
    dace.dtypes.ReductionType.Bitwise_And: "&",
    dace.dtypes.ReductionType.Logical_Or: "||",
    dace.dtypes.ReductionType.Bitwise_Or: "|",
    dace.dtypes.ReductionType.Bitwise_Xor: "^"
}


@registry.autoregister_params(name='intel_fpga')
class IntelFPGACodeGen(fpga.FPGACodeGen):
    target_name = 'intel_fpga'
    title = 'Intel FPGA'
    language = 'hls'

    def __init__(self, *args, **kwargs):
        fpga_vendor = Config.get("compiler", "fpga", "vendor")
        if fpga_vendor.lower() != "intel_fpga":
            # Don't register this code generator
            return
        # Keep track of generated converters to avoid multiple definition
        self.generated_converters = set()
        # constants
        self.generated_constants = set()
        # Channel mangles
        self.channel_mangle = defaultdict(dict)
        # Modules name mangles
        self.module_mange = defaultdict(dict)

        # Keep track of external streams
        self.external_streams = set()

        super().__init__(*args, **kwargs)

    @staticmethod
    def cmake_options():

        host_flags = Config.get("compiler", "intel_fpga", "host_flags")
        kernel_flags = Config.get("compiler", "intel_fpga", "kernel_flags")
        mode = Config.get("compiler", "intel_fpga", "mode")
        target_board = Config.get("compiler", "intel_fpga", "board")
        enable_debugging = ("ON" if Config.get_bool("compiler", "intel_fpga", "enable_debugging") else "OFF")
        autobuild = ("ON" if Config.get_bool("compiler", "fpga", "autobuild_bitstreams") else "OFF")
        options = [
            "-DDACE_INTELFPGA_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_INTELFPGA_KERNEL_FLAGS=\"{}\"".format(kernel_flags), "-DDACE_INTELFPGA_MODE={}".format(mode),
            "-DDACE_INTELFPGA_TARGET_BOARD=\"{}\"".format(target_board),
            "-DDACE_INTELFPGA_ENABLE_DEBUGGING={}".format(enable_debugging),
            "-DDACE_FPGA_AUTOBUILD_BITSTREAM={}".format(autobuild)
        ]
        # Override Intel FPGA OpenCL installation directory
        if Config.get("compiler", "intel_fpga", "path"):
            options.append("-DINTELFPGAOCL_ROOT_DIR=\"{}\"".format(
                Config.get("compiler", "intel_fpga", "path").replace("\\", "/")))
        return options

    def get_generated_codeobjects(self):

        execution_mode = Config.get("compiler", "intel_fpga", "mode")
        kernel_file_name = "DACE_BINARY_DIR \"/{}".format(self._program_name)
        emulation_flag = ""
        if execution_mode == "emulator":
            kernel_file_name += "_emulator.aocx\""
            emulation_flag = ("\n    dace::set_environment_variable"
                              "(\"CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA\", \"1\");")
        elif execution_mode == "simulator":
            kernel_file_name += "_simulator.aocx\""
        elif execution_mode == "hardware":
            kernel_file_name += "_hardware.aocx\""
        else:
            raise cgx.CodegenError("Unknown Intel FPGA execution mode: {}".format(execution_mode))

        host_code = CodeIOStream()
        host_code.write('#include "dace/intel_fpga/host.h"')
        if len(self._dispatcher.instrumentation) > 2:
            host_code.write("""\
#include "dace/perf/reporting.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
""")
        host_code.write("\n\n")

        self._frame.generate_fileheader(self._global_sdfg, host_code, 'intelfpga_host')

        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        host_code.write("""
DACE_EXPORTED int __dace_init_intel_fpga({sdfg.name}_t *__state{signature}) {{{emulation_flag}
    __state->fpga_context = new dace_fpga_context();
    __state->fpga_context->Get().MakeProgram({kernel_file_name});
    return 0;
}}

DACE_EXPORTED void __dace_exit_intel_fpga({sdfg.name}_t *__state) {{
    delete __state->fpga_context;
}}

{host_code}""".format(signature=params_comma,
                      sdfg=self._global_sdfg,
                      emulation_flag=emulation_flag,
                      kernel_file_name=kernel_file_name,
                      host_code="".join([
                          "{separator}\n// State: {kernel_name}"
                          "\n{separator}\n\n{code}\n\n".format(separator="/" * 79, kernel_name=name, code=code)
                          for (name, code) in self._host_codes
                      ])))

        host_code_obj = CodeObject(self._program_name,
                                   host_code.getvalue(),
                                   "cpp",
                                   IntelFPGACodeGen,
                                   "Intel FPGA",
                                   target_type="host")

        kernel_code_objs = [
            CodeObject(kernel_name, code, "cl", IntelFPGACodeGen, "Intel FPGA", target_type="device")
            for (kernel_name, code, _) in self._kernel_codes
        ]
        # add the util header if present
        other_code_objs = [
            CodeObject(file_name, code.getvalue(), "cl", IntelFPGACodeGen, "Intel FPGA", target_type="device")
            for (file_name, code) in self._other_codes.items()
        ]

        return [host_code_obj] + kernel_code_objs + other_code_objs

    def _internal_preprocess(self, sdfg: dace.SDFG):
        """
        Vendor-specific SDFG Preprocessing
        """
        pass

    def create_mangled_channel_name(self, var_name, kernel_id, external_stream):
        """
        Memorize and returns the mangled name of a global channel
        The dictionary is organized as ``(var_name) : {kernel_id: mangled_name}``

        :param external_stream: indicates whether this channel is an external stream
               (inter-FPGA Kernel) or not. If this is the case, it will not actually mangle
               the name by appending a suffix.
        """

        if kernel_id not in self.channel_mangle[var_name]:
            if not external_stream:
                existing_count = len(self.channel_mangle[var_name])
                suffix = f"_{existing_count}" if existing_count > 0 else ""
                mangled_name = f"{var_name}{suffix}"
            else:
                mangled_name = var_name
            self.channel_mangle[var_name][kernel_id] = mangled_name
        return self.channel_mangle[var_name][kernel_id]

    def get_mangled_channel_name(self, var_name, kernel_id):
        """
        Returns the mangled name of a channel if it is a global channel,
        or var_name if it is an alias (generated through #define)
        """
        if var_name in self.channel_mangle:
            return self.channel_mangle[var_name][kernel_id]
        else:
            return var_name

    def create_mangled_module_name(self, module_name, kernel_id):
        """
        Memorize and returns the mangled name of a module (OpenCL kernel)
        The dictionary is organized as {module_name: {kernel_id: mangled_name}}
        """

        if kernel_id not in self.module_mange[module_name]:
            existing_count = len(self.module_mange[module_name])
            suffix = f"_{existing_count}" if existing_count > 0 else ""
            mangled_name = f"{module_name}{suffix}"
            self.module_mange[module_name][kernel_id] = mangled_name
        return self.module_mange[module_name][kernel_id]

    def define_stream(self, dtype, buffer_size, var_name, array_size, function_stream, kernel_stream, sdfg):
        """
        Defines a stream

        :return: a tuple containing the  type of the created variable, and boolean indicating
            whether this is a global variable or not
        """
        vec_type = self.make_vector_type(dtype, False)
        minimum_depth = Config.get("compiler", "fpga", "minimum_fifo_depth")
        buffer_size = evaluate(buffer_size, sdfg.constants)
        if minimum_depth:
            minimum_depth = int(minimum_depth)
            if minimum_depth > buffer_size:
                buffer_size = minimum_depth
        if buffer_size != 1:
            depth_attribute = " __attribute__((depth({})))".format(cpp.sym2cpp(buffer_size))
        else:
            depth_attribute = ""
        if cpp.sym2cpp(array_size) != "1":
            size_str = "[" + cpp.sym2cpp(array_size) + "]"
        else:
            size_str = ""

        if var_name in self.external_streams:
            # This is an external streams: it connects two different FPGA Kernels
            # that will be code-generated as two separate files.
            # We need to declare the channel as global variable and it must have have
            # the same name in both the files.

            chan_name = self.create_mangled_channel_name(var_name, self._kernel_count, True)
            function_stream.write("channel {} {}{}{};".format(vec_type, chan_name, size_str, depth_attribute))
        else:
            # mangle name
            chan_name = self.create_mangled_channel_name(var_name, self._kernel_count, False)

            kernel_stream.write("channel {} {}{}{};".format(vec_type, chan_name, size_str, depth_attribute))

        # Return value is used for adding to defined_vars in fpga.py
        # In Intel FPGA, streams must be defined as global entity, so they will be added to the global variables
        return 'channel {}'.format(vec_type), True

    def define_local_array(self, var_name, desc, array_size, function_stream, kernel_stream, sdfg, state_id, node):
        vec_type = self.make_vector_type(desc.dtype, False)
        if desc.storage == dace.dtypes.StorageType.FPGA_Registers:
            attributes = " __attribute__((register))"
        else:
            attributes = ""
        kernel_stream.write("{}{} {}[{}];\n".format(vec_type, attributes, var_name, cpp.sym2cpp(array_size)))
        self._dispatcher.defined_vars.add(var_name, DefinedType.Pointer, vec_type)

    def define_shift_register(self, *args, **kwargs):
        # Shift registers are just arrays on Intel
        self.define_local_array(*args, **kwargs)

    @staticmethod
    def make_vector_type(dtype, is_const):
        return "{}{}".format("const " if is_const else "", dtype.ocltype)

    def make_kernel_argument(self, data, var_name, is_output, with_vectorization):
        if isinstance(data, dace.data.Array):
            if with_vectorization:
                vec_type = data.dtype.ocltype
            else:
                vec_type = fpga.vector_element_type_of(data.dtype).ocltype
            return "__global volatile  {}* restrict {}".format(vec_type, var_name)
        elif isinstance(data, dace.data.Stream):
            return None  # Streams are global objects
        else:
            return data.as_arg(with_types=True, name=var_name)

    @staticmethod
    def generate_unroll_loop_pre(kernel_stream, factor, sdfg, state_id, node):
        if factor is not None:
            factor_str = " " + factor
        else:
            factor_str = ""
        kernel_stream.write("#pragma unroll{}".format(factor_str), sdfg, state_id, node)

    @staticmethod
    def generate_unroll_loop_post(kernel_stream, factor, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_pipeline_loop_pre(kernel_stream, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_pipeline_loop_post(kernel_stream, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_flatten_loop_pre(kernel_stream, sdfg, state_id, node):
        kernel_stream.write("#pragma loop_coalesce")

    @staticmethod
    def generate_flatten_loop_post(kernel_stream, sdfg, state_id, node):
        pass

    def make_read(self, defined_type, dtype, var_name, expr, index, is_pack, packing_factor):
        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            # channel mangling: the expression could contain indexing
            expr.replace(var_name, self.get_mangled_channel_name(var_name, self._kernel_count))
            read_expr = "read_channel_intel({})".format(expr)
        elif defined_type == DefinedType.Pointer:
            if index and index != "0":
                read_expr = f"*({expr} + {index})"
            else:
                if " " in expr:
                    expr = f"({expr})"
                read_expr = f"*{expr}"
        elif defined_type == DefinedType.Scalar:
            read_expr = var_name
        else:
            raise NotImplementedError("Unimplemented read type: {}".format(defined_type))
        if is_pack:
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.converters_to_generate.add((True, ocltype, packing_factor))
            return "pack_{}{}(&({}))".format(ocltype, packing_factor, read_expr)
        else:
            return read_expr

    def make_write(self, defined_type, dtype, var_name, write_expr, index, read_expr, wcr, is_unpack, packing_factor):
        """
        Creates write expression, taking into account wcr if present
        """
        if wcr is not None:
            redtype = operations.detect_reduction_type(wcr, openmp=True)

        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            #mangle name
            chan_name = self.get_mangled_channel_name(write_expr, self._kernel_count)
            if defined_type == DefinedType.StreamArray:
                write_expr = "{}[{}]".format(chan_name, index)
            if is_unpack:
                return "\n".join("write_channel_intel({}, {}[{}]);".format(write_expr, read_expr, i)
                                 for i in range(packing_factor))
            else:
                return "write_channel_intel({}, {});".format(chan_name, read_expr)
        elif defined_type == DefinedType.Pointer:
            if wcr is not None:
                if (redtype != dace.dtypes.ReductionType.Min and redtype != dace.dtypes.ReductionType.Max):
                    return "{}[{}] = {}[{}] {} {};".format(write_expr, index, write_expr, index,
                                                           REDUCTION_TYPE_TO_HLSLIB[redtype], read_expr)
                else:
                    # use max/min opencl builtins
                    return "{}[{}] = {}{}({}[{}],{});".format(
                        write_expr, index, ("f" if dtype.ocltype == "float" or dtype.ocltype == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr, index, read_expr)
            else:
                if is_unpack:
                    ocltype = fpga.vector_element_type_of(dtype).ocltype
                    self.converters_to_generate.add((False, ocltype, packing_factor))
                    if not index or index == "0":
                        return "unpack_{}{}({}, {});".format(ocltype, packing_factor, read_expr, write_expr)
                    else:
                        return "unpack_{}{}({}, {} + {});".format(ocltype, packing_factor, read_expr, write_expr, index)
                else:
                    if " " in write_expr:
                        write_expr = f"({write_expr})"
                    if index and index != "0":
                        return f"{write_expr}[{index}] = {read_expr};"
                    else:
                        return f"*{write_expr} = {read_expr};"
        elif defined_type == DefinedType.Scalar:
            if wcr is not None:
                if redtype != dace.dtypes.ReductionType.Min and redtype != dace.dtypes.ReductionType.Max:
                    return "{} = {} {} {};".format(write_expr, write_expr, REDUCTION_TYPE_TO_HLSLIB[redtype], read_expr)
                else:
                    # use max/min opencl builtins
                    return "{} = {}{}({},{});".format(
                        write_expr, ("f" if dtype.ocltype == "float" or dtype.ocltype == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr, read_expr)
            else:
                if is_unpack:
                    ocltype = fpga.vector_element_type_of(dtype).ocltype
                    self.converters_to_generate.add((False, ocltype, packing_factor))
                    return "unpack_{}{}({}, {});".format(
                        vector_element_type_of(dtype).ocltype, packing_factor, read_expr, var_name)
                else:
                    return "{} = {};".format(var_name, read_expr)
        raise NotImplementedError("Unimplemented write type: {}".format(defined_type))

    def make_shift_register_write(self, defined_type, dtype, var_name, write_expr, index, read_expr, wcr, is_unpack,
                                  packing_factor, sdfg):
        if defined_type != DefinedType.Pointer:
            raise TypeError("Intel shift register must be an array: "
                            "{} is {}".format(var_name, defined_type))
        # Shift array
        arr_size = functools.reduce(lambda a, b: a * b, sdfg.data(var_name).shape, 1)
        res = """
#pragma unroll
for (int u_{name} = 0; u_{name} < {size} - {veclen}; ++u_{name}) {{
  {name}[u_{name}] = {name}[u_{name} + {veclen}];
}}\n""".format(name=var_name, size=arr_size, veclen=cpp.sym2cpp(dtype.veclen))
        # Then do write
        res += self.make_write(defined_type, dtype, var_name, write_expr, index, read_expr, wcr, is_unpack,
                               packing_factor)
        return res

    @staticmethod
    def generate_no_dependence_pre(kernel_stream, sdfg, state_id, node, var_name=None):
        """
            Adds pre-loop pragma for ignoring loop carried dependencies on a given variable
            (if var_name is provided) or all variables
        """
        if var_name is None:
            kernel_stream.write("#pragma ivdep", sdfg, state_id, node)
        else:
            kernel_stream.write("#pragma ivdep array({})".format(var_name), sdfg, state_id, node)

    @staticmethod
    def generate_no_dependence_post(kernel_stream, sdfg, state_id, node, var_name=None, accessed_subset=None):
        pass

    def generate_kernel_internal(self, sdfg: dace.SDFG, state: dace.SDFGState, kernel_name: str, predecessors: list,
                                 subgraphs: list, kernel_stream: CodeIOStream, state_host_header_stream: CodeIOStream,
                                 state_host_body_stream: CodeIOStream, instrumentation_stream: CodeIOStream,
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream, state_parameters: list):
        """
        Generates Kernel code, both device and host side.

        :param sdfg:
        :param state:
        :param kernel_name:
        :param predecessors: list containing all the name of kernels from which this one depends
        :param subgraphs:
        :param kernel_stream: Device code stream, contains the kernel code
        :param state_host_header_stream: Device-specific code stream: contains the host code
            for the state global declarations.
        :param state_host_body_stream: Device-specific code stream: contains all the code related to
            this state, for creating transient buffers, spawning kernels, and synchronizing them.
        :param instrumentation_stream: Code for profiling kernel execution time.
        :param function_stream: CPU code stream.
        :param callsite_stream: CPU code stream.
        :param state_parameters: list of state parameters. The kernel-specific parameters will be appended to it.
        """

        # In xilinx one of them is not used because part of the code goes in another place (entry_stream)
        state_id = sdfg.node_id(state)

        kernel_header_stream = CodeIOStream()
        kernel_body_stream = CodeIOStream()

        #reset list of needed converters
        self.converters_to_generate = set()

        kernel_header_stream.write("#include <dace/fpga_device.h>\n\n", sdfg)
        self.generate_constants(sdfg, kernel_header_stream)
        kernel_header_stream.write("\n", sdfg)

        (global_data_parameters, top_level_local_data, subgraph_parameters, nested_global_transients, bank_assignments,
         external_streams) = self.make_parameters(sdfg, state, subgraphs)

        # save the name of external streams
        self.external_streams = set([chan_name for _, chan_name, _, _ in external_streams])

        # Emit allocations of inter-kernel memories
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node, node.desc(sdfg), callsite_stream,
                                               kernel_body_stream)

        kernel_body_stream.write("\n")
        state_parameters.extend(global_data_parameters)
        # Generate host code (Global transients)
        self.generate_host_function_boilerplate(sdfg, state, nested_global_transients, state_host_body_stream)

        self.generate_host_function_prologue(sdfg, state, state_host_body_stream, kernel_name)

        # Generate PEs code
        self.generate_modules(sdfg, state, kernel_name, subgraphs, subgraph_parameters, kernel_body_stream,
                              state_host_header_stream, state_host_body_stream, instrumentation_stream)

        kernel_body_stream.write("\n")

        # Generate data width converters
        self.generate_converters(sdfg, kernel_header_stream)

        kernel_stream.write(kernel_header_stream.getvalue() + kernel_body_stream.getvalue())

        # Generate host kernel invocation
        self.generate_host_function_body(sdfg, state, state_host_body_stream, kernel_name, predecessors)

    def generate_host_function_prologue(self, sdfg, state, host_stream, kernel_name):
        seperator = "/" * 59
        host_stream.write(f"\n{seperator}\n// Kernel: {kernel_name}\n{seperator}\n\n")

        host_stream.write(f"std::vector<hlslib::ocl::Kernel> {kernel_name}_kernels;", sdfg, sdfg.node_id(state))

    def generate_host_function_body(self, sdfg: dace.SDFG, state: dace.SDFGState, host_stream: CodeIOStream,
                                    kernel_name: str, predecessors: list):
        """
        Generate the host-specific code for spawning and synchronizing the given kernel.

        :param sdfg:
        :param state:
        :param host_stream: Device-specific code stream
        :param kernel_name:
        :param predecessors: list containing all the name of kernels that must be finished before starting this one
        """
        state_id = sdfg.node_id(state)

        # Check if this kernel depends from other kernels
        needs_synch = len(predecessors) > 0

        if needs_synch:
            # Build a vector containing all the events associated with the kernels from which this one depends
            kernel_deps_name = f"deps_{kernel_name}"
            host_stream.write(f"std::vector<cl::Event> {kernel_deps_name};")
            for pred in predecessors:
                # concatenate events from predecessor kernel
                host_stream.write(
                    f"{kernel_deps_name}.insert({kernel_deps_name}.end(), {pred}_events.begin(), {pred}_events.end());")

        # While spawning the kernel, indicates the synchronization events (if any)
        host_stream.write(
            f"""\
  std::vector<cl::Event> {kernel_name}_events;
  for (auto &k : {kernel_name}_kernels) {{
    {kernel_name}_events.emplace_back(k.ExecuteTaskAsync({f'{kernel_deps_name}.begin(), {kernel_deps_name}.end()' if needs_synch else ''}));
  }}
  all_events.insert(all_events.end(), {kernel_name}_events.begin(), {kernel_name}_events.end());
""", sdfg, state_id)

    def generate_module(self, sdfg, state, kernel_name, module_name, subgraph, parameters, module_stream,
                        host_header_stream, host_body_stream, instrumentation_stream):
        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        kernel_args_opencl = []
        kernel_args_host = []
        kernel_args_call = []
        for is_output, pname, p, _ in parameters:
            if isinstance(p, dace.data.View):
                continue
            arg = self.make_kernel_argument(p, pname, is_output, True)
            if arg is not None:
                kernel_args_opencl.append(arg)
                kernel_args_host.append(p.as_arg(True, name=pname))
                kernel_args_call.append(pname)

        # If the kernel takes no arguments, we don't have to call it from the
        # host
        is_autorun = len(kernel_args_opencl) == 0

        # create a unique module name to prevent name clashes
        module_function_name = "mod_" + str(sdfg.sdfg_id) + "_" + module_name
        # The official limit suggested by Intel for module name is 61. However, the compiler
        # can also append text to the module. Longest seen so far is
        # "_cra_slave_inst", which is 15 characters, so we restrict to
        # 61 - 15 = 46, and round down to 36 to be conservative, since
        # internally could still fail while dealing with RTL.
        # However, in this way we could have name clashes (e.g., if we have two almost identical NestedSDFG).
        # Therefore we explicitly take care of this by mangling the name
        module_function_name = self.create_mangled_module_name(module_function_name[0:36], self._kernel_count)

        # Unrolling processing elements: if there first scope of the subgraph
        # is an unrolled map, generate a processing element for each iteration
        scope_children = subgraph.scope_children()
        top_scopes = [n for n in scope_children[None] if isinstance(n, dace.sdfg.nodes.EntryNode)]
        unrolled_loop = None
        if len(top_scopes) == 1:
            scope = top_scopes[0]
            if scope.unroll:
                # Unrolled processing elements
                self._unrolled_pes.add(scope.map)
                kernel_args_opencl += ["const int " + p for p in scope.params]  # PE id will be a macro defined constant
                kernel_args_call += [p for p in scope.params]
                unrolled_loop = scope.map

        # Ensure no duplicate parameters are used
        kernel_args_opencl = dtypes.deduplicate(kernel_args_opencl)
        kernel_args_call = dtypes.deduplicate(kernel_args_call)

        # Add kernel call host function
        if not is_autorun:
            if unrolled_loop is None:
                host_body_stream.write(
                    "{}_kernels.emplace_back(program.MakeKernel(\"{}\"{}));".format(
                        kernel_name, module_function_name,
                        ", ".join([""] + kernel_args_call) if len(kernel_args_call) > 0 else ""), sdfg, state_id)
                if state.instrument == dtypes.InstrumentationType.FPGA:
                    self.instrument_opencl_kernel(module_function_name, state_id, sdfg.sdfg_id, instrumentation_stream)
            else:
                # We will generate a separate kernel for each PE. Adds host call
                start, stop, skip = unrolled_loop.range.ranges[0]
                start_idx = evaluate(start, sdfg.constants)
                stop_idx = evaluate(stop, sdfg.constants)
                skip_idx = evaluate(skip, sdfg.constants)
                # Due to restrictions on channel indexing, PE IDs must start
                # from zero and skip index must be 1
                if start_idx != 0 or skip_idx != 1:
                    raise cgx.CodegenError(f"Unrolled Map in {sdfg.name} should start from 0 "
                                           "and have skip equal to 1")
                for p in range(start_idx, stop_idx + 1, skip_idx):
                    # Last element in list kernel_args_call is the PE ID, but
                    # this is already written in stone in the OpenCL generated
                    # code
                    unrolled_module_name = f"{module_function_name}_{p}"
                    host_body_stream.write(
                        "{}_kernels.emplace_back(program.MakeKernel(\"{}\"{}));".format(
                            kernel_name, unrolled_module_name,
                            ", ".join([""] + kernel_args_call[:-1]) if len(kernel_args_call) > 1 else ""), sdfg,
                        state_id)
                    if state.instrument == dtypes.InstrumentationType.FPGA:
                        self.instrument_opencl_kernel(unrolled_module_name, state_id, sdfg.sdfg_id,
                                                      instrumentation_stream)

        # ----------------------------------------------------------------------
        # Generate kernel code
        # ----------------------------------------------------------------------

        self._dispatcher.defined_vars.enter_scope(subgraph)

        module_body_stream = CodeIOStream()

        AUTORUN_STR = """\
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))\n"""

        if unrolled_loop is None:
            module_body_stream.write(
                "{}__kernel void {}({}) {{".format(AUTORUN_STR if is_autorun else "", module_function_name,
                                                   ", ".join(kernel_args_opencl)), sdfg, state_id)
        else:
            # Unrolled PEs: we have to generate a kernel for each PE. We will generate
            # a function that will be used create a kernel multiple times

            # generate a unique name for this function
            pe_function_name = "pe_" + str(sdfg.sdfg_id) + "_" + module_name + "_func"
            module_body_stream.write("inline void {}({}) {{".format(pe_function_name, ", ".join(kernel_args_opencl)),
                                     sdfg, state_id)

        # Allocate local transients
        data_to_allocate = (set(subgraph.top_level_transients()) - set(sdfg.shared_transients()) -
                            set([p[1] for p in parameters]))
        allocated = set()
        for node in subgraph.nodes():
            if not isinstance(node, dace.sdfg.nodes.AccessNode):
                continue
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node, node.desc(sdfg), module_stream,
                                               module_body_stream)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           subgraph,
                                           state_id,
                                           module_stream,
                                           module_body_stream,
                                           skip_entry_node=False)

        module_stream.write(module_body_stream.getvalue(), sdfg, state_id)
        module_stream.write("}\n\n")

        if unrolled_loop is not None:

            AUTORUN_STR_MACRO = """
__attribute__((max_global_work_dim(0))) \\
__attribute__((autorun)) \\"""

            # Unrolled PEs: create as many kernels as the number of PEs
            # To avoid long and duplicated code, do it with define (gosh)
            # Since OpenCL is "funny", it does not support variadic macros
            # One of the argument is for sure the PE_ID, which is also the last one in kernel_args lists:
            # it will be not passed by the host but code-generated
            module_stream.write("""\
#define _DACE_FPGA_KERNEL_{}(PE_ID{}{}) \\{}
__kernel void \\
{}_##PE_ID({}) \\
{{ \\
  {}({}{}PE_ID); \\
}}\\\n\n""".format(module_function_name, ", " if len(kernel_args_call) > 1 else "", ", ".join(kernel_args_call[:-1]),
                   AUTORUN_STR_MACRO if is_autorun else "", module_function_name, ", ".join(kernel_args_opencl[:-1]),
                   pe_function_name, ", ".join(kernel_args_call[:-1]), ", " if len(kernel_args_call) > 1 else ""))

            # create PE kernels by using the previously defined macro
            start, stop, skip = unrolled_loop.range.ranges[0]
            start_idx = evaluate(start, sdfg.constants)
            stop_idx = evaluate(stop, sdfg.constants)
            skip_idx = evaluate(skip, sdfg.constants)
            # First macro argument is the processing element id
            for p in range(start_idx, stop_idx + 1, skip_idx):
                module_stream.write("_DACE_FPGA_KERNEL_{}({}{}{})\n".format(module_function_name, p,
                                                                            ", " if len(kernel_args_call) > 1 else "",
                                                                            ", ".join(kernel_args_call[:-1])))
            module_stream.write("#undef _DACE_FPGA_KERNEL_{}\n".format(module_function_name))

        self._dispatcher.defined_vars.exit_scope(subgraph)

    def generate_nsdfg_header(self, sdfg, state, state_id, node, memlet_references, sdfg_label):
        # Intel FPGA needs to deal with streams
        arguments = [f'{atype} {aname}' for atype, aname, _ in memlet_references]
        arguments += [
            f'{node.sdfg.symbols[aname].as_arg(aname)}' for aname in sorted(node.symbol_mapping.keys())
            if aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        function_header = f'void {sdfg_label}({arguments}) {{'
        nested_stream = CodeIOStream()

        #generate Stream defines if needed
        for edge in state.in_edges(node):
            if edge.data.data is not None:  # skip empty memlets
                desc = sdfg.arrays[edge.data.data]
                if isinstance(desc, dace.data.Stream):
                    src_node = find_input_arraynode(state, edge)
                    self._dispatcher.dispatch_copy(src_node, node, edge, sdfg, state, state_id, None, nested_stream)
        for edge in state.out_edges(node):
            if edge.data.data is not None:  # skip empty memlets
                desc = sdfg.arrays[edge.data.data]
                if isinstance(desc, dace.data.Stream):
                    dst_node = find_output_arraynode(state, edge)
                    self._dispatcher.dispatch_copy(node, dst_node, edge, sdfg, state, state_id, None, nested_stream)
        return function_header + "\n" + nested_stream.getvalue()

    def generate_nsdfg_arguments(self, sdfg, dfg, state, node):
        # Connectors that are both input and output share the same name
        inout = set(node.in_connectors.keys() & node.out_connectors.keys())
        memlet_references = []

        for _, _, _, vconn, in_memlet in state.in_edges(node):
            if vconn in inout or in_memlet.data is None:
                continue
            desc = sdfg.arrays[in_memlet.data]
            ptrname = cpp.ptr(in_memlet.data, desc, sdfg, self._frame)
            defined_type, defined_ctype = self._dispatcher.defined_vars.get(ptrname, 1)

            if isinstance(desc, dace.data.Array) and (desc.storage == dtypes.StorageType.FPGA_Global
                                                      or desc.storage == dtypes.StorageType.FPGA_Local):
                # special case: in intel FPGA this must be handled properly to guarantee OpenCL compatibility
                # (no pass by reference)
                # The defined type can be a scalar, and therefore we get its address
                vec_type = desc.dtype.ocltype
                offset = cpp.cpp_offset_expr(desc, in_memlet.subset, None)
                offset_expr = '[' + offset + ']' if defined_type is not DefinedType.Scalar else ''

                expr = self.make_ptr_vector_cast(ptrname + offset_expr, desc.dtype, node.in_connectors[vconn], False,
                                                 defined_type)
                if desc.storage == dtypes.StorageType.FPGA_Global:
                    typedef = "__global volatile  {}* restrict".format(vec_type)
                else:
                    typedef = "{} *".format(vec_type)
                ref = '&' if defined_type is DefinedType.Scalar else ''
                memlet_references.append((typedef, vconn, ref + expr))
                # get the defined type (as defined in the parent)
                # Register defined variable
                self._dispatcher.defined_vars.add(vconn, DefinedType.Pointer, typedef, allow_shadowing=True)
            elif isinstance(desc, dace.data.Stream):
                # streams are defined as global variables
                continue
            elif isinstance(desc, dace.data.Scalar):
                # if this is a scalar and the argument passed is also a scalar
                # then we have to pass it by value, as references do not exist in C99
                typedef = defined_ctype
                if defined_type is not DefinedType.Pointer:
                    typedef = typedef + "*"

                memlet_references.append(
                    (typedef, vconn, cpp.cpp_ptr_expr(sdfg, in_memlet, defined_type, codegen=self._frame)))
                self._dispatcher.defined_vars.add(vconn, DefinedType.Pointer, typedef, allow_shadowing=True)
            else:
                # all the other cases
                memlet_references.append(
                    cpp.emit_memlet_reference(self._dispatcher,
                                              sdfg,
                                              in_memlet,
                                              vconn,
                                              conntype=node.in_connectors[vconn]))

        for _, uconn, _, _, out_memlet in state.out_edges(node):
            if out_memlet.data is not None:
                desc = sdfg.arrays[out_memlet.data]
                ptrname = cpp.ptr(out_memlet.data, desc, sdfg, self._frame)
                defined_type, defined_ctype = self._dispatcher.defined_vars.get(ptrname, 1)

                if isinstance(desc, dace.data.Array) and (desc.storage == dtypes.StorageType.FPGA_Global
                                                          or desc.storage == dtypes.StorageType.FPGA_Local):
                    # special case: in intel FPGA this must be handled properly.
                    # The defined type can be scalar, and therefore we get its address
                    vec_type = desc.dtype.ocltype
                    offset = cpp.cpp_offset_expr(desc, out_memlet.subset, None)
                    offset_expr = '[' + offset + ']' if defined_type is not DefinedType.Scalar else ''
                    if desc.storage == dtypes.StorageType.FPGA_Global:
                        typedef = "__global volatile  {}* restrict".format(vec_type)
                    else:
                        typedef = "{}*".format(vec_type)
                    ref = '&' if defined_type is DefinedType.Scalar else ''
                    expr = self.make_ptr_vector_cast(ptrname + offset_expr, desc.dtype, node.out_connectors[uconn],
                                                     False, defined_type)
                    memlet_references.append((typedef, uconn, ref + expr))
                    # Register defined variable
                    self._dispatcher.defined_vars.add(uconn, DefinedType.Pointer, typedef, allow_shadowing=True)
                elif isinstance(desc, dace.data.Stream):
                    # streams are defined as global variables
                    continue
                elif isinstance(desc, dace.data.Scalar):
                    # if this is a scalar and the argument passed is also a scalar
                    # then we have to pass it by reference, i.e., we should define it
                    # as a pointer since references do not exist in C99
                    typedef = defined_ctype
                    if defined_type is not DefinedType.Pointer:
                        typedef = typedef + "*"
                    memlet_references.append(
                        (typedef, uconn, cpp.cpp_ptr_expr(sdfg, out_memlet, defined_type, codegen=self._frame)))
                    self._dispatcher.defined_vars.add(uconn, DefinedType.Pointer, typedef, allow_shadowing=True)
                else:
                    memlet_references.append(
                        cpp.emit_memlet_reference(self._dispatcher,
                                                  sdfg,
                                                  out_memlet,
                                                  uconn,
                                                  conntype=node.out_connectors[uconn]))

        # Special case for Intel FPGA: this comes out from the unrolling processing elements:
        # if the first scope of the subgraph is an unrolled map, generates a processing element for each iteration
        # We need to pass to this function also the id of the PE (the top scope parameter)
        scope_children = dfg.scope_children()
        top_scopes = [n for n in scope_children[None] if isinstance(n, dace.sdfg.nodes.EntryNode)]
        if len(top_scopes) == 1:
            scope = top_scopes[0]
            if scope.unroll:
                # Unrolled processing elements
                typedef = "const int"
                for p in scope.params:
                    # if this is not already a mapped symbol, add it
                    if p not in node.symbol_mapping.keys():
                        memlet_references.append((typedef, p, p))
        return memlet_references

    def allocate_view(self, sdfg: dace.SDFG, dfg: SDFGState, state_id: int, node: dace.nodes.AccessNode,
                      global_stream: CodeIOStream, declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        """
        Allocates (creates pointer and refers to original) a view of an
        existing array, scalar, or view. Specifically tailored for Intel FPGA
        """
        name = node.data
        nodedesc = node.desc(sdfg)
        ptrname = cpp.ptr(name, nodedesc, sdfg, self._frame)
        if self._dispatcher.defined_vars.has(ptrname):
            return  # View was already allocated

        # Check directionality of view (referencing dst or src)
        edge = sdutils.get_view_edge(dfg, node)

        # Allocate the viewed data before the view, if necessary
        mpath = dfg.memlet_path(edge)
        viewed_dnode = mpath[0].src if edge.dst is node else mpath[-1].dst
        self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, viewed_dnode, viewed_dnode.desc(sdfg), global_stream,
                                           allocation_stream)

        # Emit memlet as a reference and register defined variable
        if nodedesc.storage == dace.dtypes.StorageType.FPGA_Global:
            # If the viewed (hence the view) node has global storage type, we need to specifically
            # derive the declaration/definition

            qualifier = "__global volatile "
            atype = dtypes.pointer(nodedesc.dtype).ctype + " restrict"
            aname = ptrname
            viewed_desc = sdfg.arrays[edge.data.data]
            eptr = cpp.ptr(edge.data.data, viewed_desc, sdfg, self._frame)
            defined_type, _ = self._dispatcher.defined_vars.get(eptr, 0)
            # Register defined variable
            self._dispatcher.defined_vars.add(aname, defined_type, atype, allow_shadowing=True)
            _, _, value = cpp.emit_memlet_reference(self._dispatcher,
                                                    sdfg,
                                                    edge.data,
                                                    name,
                                                    dtypes.pointer(nodedesc.dtype),
                                                    ancestor=0,
                                                    device_code=self._in_device_code)
        else:
            qualifier = ""
            atype, aname, value = cpp.emit_memlet_reference(self._dispatcher,
                                                            sdfg,
                                                            edge.data,
                                                            name,
                                                            dtypes.pointer(nodedesc.dtype),
                                                            ancestor=0)
        declaration_stream.write(f'{qualifier}{atype} {aname}  = {value};', sdfg, state_id, node)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node, dst_node, edge, callsite_stream):

        if isinstance(edge.dst, dace.sdfg.nodes.CodeNode):
            # Input memlet
            connector = edge.dst_conn
            is_output = False
            tasklet = edge.dst
            conntype = tasklet.in_connectors[connector]
        elif isinstance(edge.src, dace.sdfg.nodes.CodeNode):
            # Output memlet
            connector = edge.src_conn
            is_output = True
            tasklet = edge.src
            conntype = tasklet.out_connectors[connector]
        else:
            raise NotImplementedError("Not implemented for {} to {}".format(type(edge.src), type(edge.dst)))

        memlet = edge.data
        data_name = memlet.data
        data_desc = sdfg.arrays[data_name]
        data_dtype = data_desc.dtype

        is_scalar = not isinstance(conntype, dtypes.pointer)
        dtype = conntype if is_scalar else conntype._typeclass

        memlet_type = self.make_vector_type(dtype, False)
        offset = cpp.cpp_offset_expr(data_desc, memlet.subset, None)

        if dtype != data_dtype:
            if (isinstance(dtype, dace.vector) and dtype.base_type == data_dtype):
                cast = True
            else:
                raise TypeError("Type mismatch: {} vs. {}".format(dtype, data_dtype))
        else:
            cast = False

        result = ""

        # NOTE: FPGA Streams are defined at the top-level scope. We use the
        # following boolean to pass this informations to the `get` method of
        # the `defined_vars` object.
        is_global = False
        if isinstance(data_desc, dace.data.Stream):
            # Derive the name of the original stream, by tracing the memlet path through nested SDFGs
            outer_stream_node_trace = utils.trace_nested_access(dst_node if is_output else src_node,
                                                                sdfg.nodes()[state_id], sdfg)
            data_name = outer_stream_node_trace[0][0][1 if is_output else 0].label
            is_global = True

        data_name = cpp.ptr(data_name, data_desc, sdfg, self._frame)

        def_type, ctypedef = self._dispatcher.defined_vars.get(data_name, is_global=is_global)
        if def_type == DefinedType.Scalar:
            if cast:
                rhs = f"(*({memlet_type} const *)&{data_name})"
            else:
                rhs = data_name
            if not memlet.dynamic:
                if not is_output:
                    # We can pre-read the value
                    result += "{} {} = {};".format(memlet_type, connector, rhs)
                else:
                    # The value will be written during the tasklet, and will be
                    # automatically written out after
                    init = ""

                    result += "{} {}{};".format(memlet_type, connector, init)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar, memlet_type)
            else:
                # Variable number of reads or writes
                result += "{} *{} = &{};".format(memlet_type, connector, rhs)
                self._dispatcher.defined_vars.add(connector, DefinedType.Pointer, '%s *' % memlet_type)
        elif def_type == DefinedType.Pointer:
            if cast:
                rhs = f"(({memlet_type} const *){data_name})"
            else:
                rhs = data_name
            if is_scalar and not memlet.dynamic:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = {}[{}];".format(memlet_type, connector, rhs, offset)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar, memlet_type)
            else:
                if data_desc.storage == dace.dtypes.StorageType.FPGA_Global:
                    qualifiers = "__global "
                else:
                    qualifiers = ""
                ctype = '{}{} *'.format(qualifiers, memlet_type)
                result += "{}{} = &{}[{}];".format(ctype, connector, rhs, offset)
                self._dispatcher.defined_vars.add(connector, DefinedType.Pointer, ctype)
        elif def_type == DefinedType.Stream:
            if cast:
                raise TypeError("Cannot cast stream from {} to {}.".format(data_dtype, dtype))

            # In the define we refer to the stream defined in the outermost scope
            if not memlet.dynamic and memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = read_channel_intel({});".format(
                        memlet_type, connector, self.get_mangled_channel_name(data_name, self._kernel_count))
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar, memlet_type)
            else:
                # Desperate times call for desperate measures
                result += "#define {} {} // God save us".format(
                    connector, self.get_mangled_channel_name(data_name, self._kernel_count))
                self._dispatcher.defined_vars.add(connector, DefinedType.Stream, ctypedef)
        elif def_type == DefinedType.StreamArray:
            if cast:
                raise TypeError("Cannot cast stream array from {} to {}.".format(data_dtype, dtype))
            # We need to refer to the stream defined in the outermost scope
            # Since this is a Stream Array, we need also the offset, which is contained in the memlet that arrives/departs
            # from that stream
            outer_memlet = outer_stream_node_trace[0][1][1 if is_output else 0]
            outer_sdfg = outer_stream_node_trace[0][-1]

            if not memlet.dynamic and memlet.num_accesses == 1 and (is_output is True
                                                                    or isinstance(edge.dst, dace.sdfg.nodes.Tasklet)):
                # if this is an input memlet, generate the read only if this is a tasklet
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    global_node = utils.trace_nested_access(dst_node if is_output else src_node,
                                                            sdfg.nodes()[state_id], sdfg)
                    data_name = global_node[0][0][1 if is_output else 0].label

                    if outer_memlet is not None:
                        offset = cpp.cpp_offset_expr(outer_sdfg.arrays[data_name], outer_memlet.subset)

                    result += "{} {} = read_channel_intel({}[{}]);".format(
                        memlet_type, connector, self.get_mangled_channel_name(data_name, self._kernel_count), offset)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar, memlet_type)
            else:
                # Must happen directly in the code
                # Here we create a macro which take the proper channel
                if outer_memlet is not None:
                    channel_idx = cpp.cpp_offset_expr(outer_sdfg.arrays[data_name], outer_memlet.subset)
                else:
                    channel_idx = cpp.cpp_offset_expr(sdfg.arrays[data_name], memlet.subset)
                result += "#define {} {}[{}] // God save us".format(
                    connector, self.get_mangled_channel_name(data_name, self._kernel_count), channel_idx)
                self._dispatcher.defined_vars.add(connector, DefinedType.Stream, ctypedef)
        else:
            raise TypeError("Unknown variable type: {}".format(def_type))

        callsite_stream.write(result, sdfg, state_id, tasklet)

    def generate_channel_writes(self, sdfg, dfg, node, callsite_stream, state_id):
        for edge in dfg.out_edges(node):
            connector = edge.src_conn
            memlet = edge.data
            data_name = memlet.data
            if data_name is not None:
                data_desc = sdfg.arrays[data_name]
                if (isinstance(data_desc, dace.data.Stream) and memlet.volume == 1 and not memlet.dynamic):
                    # mangle channel
                    chan_name = self.get_mangled_channel_name(data_name, self._kernel_count)
                    if data_desc.is_stream_array():
                        offset = cpp.cpp_offset_expr(data_desc, memlet.subset)
                        target = f"{chan_name}[{offset}]"
                    else:
                        target = chan_name
                    callsite_stream.write(f"write_channel_intel({target}, {connector});", sdfg)

    def generate_undefines(self, sdfg, dfg, node, callsite_stream):
        for edge in itertools.chain(dfg.in_edges(node), dfg.out_edges(node)):
            memlet = edge.data
            data_name = memlet.data

            if edge.src == node:
                memlet_name = edge.src_conn
            elif edge.dst == node:
                memlet_name = edge.dst_conn

            if data_name is not None:
                data_desc = sdfg.arrays[data_name]
                if (isinstance(data_desc, dace.data.Stream) and (memlet.dynamic or memlet.num_accesses != 1)):
                    callsite_stream.write("#undef {}".format(memlet_name), sdfg)

    def _generate_converter(self, is_unpack, ctype, veclen, sdfg, function_stream):
        # Get the file stream
        if "converters" not in self._other_codes:
            self._other_codes["converters"] = CodeIOStream()
        converter_stream = self._other_codes["converters"]

        veclen = cpp.sym2cpp(veclen)

        if is_unpack:
            converter_name = "unpack_{dtype}{veclen}".format(dtype=ctype, veclen=veclen)
            signature = "void {name}(const {dtype}{veclen} value, {dtype} *const ptr)".format(name=converter_name,
                                                                                              dtype=ctype,
                                                                                              veclen=veclen)
            if converter_name not in self.generated_converters:
                self.generated_converters.add(converter_name)

                # create code for converter in appropriate header file
                converter_stream.write(
                    """\
{signature} {{
    #pragma unroll
    for (int u = 0; u < {veclen}; ++u) {{
        ptr[u] = value[u];
    }}
}}\n\n""".format(signature=signature, dtype=ctype, veclen=veclen), sdfg)

            # add forward declaration
            function_stream.write("extern {};".format(signature), sdfg)

        else:
            converter_name = "pack_{dtype}{veclen}".format(dtype=ctype, veclen=veclen)
            signature = "{dtype}{veclen} {name}({dtype} const *const ptr)".format(name=converter_name,
                                                                                  dtype=ctype,
                                                                                  veclen=veclen)
            if converter_name not in self.generated_converters:
                self.generated_converters.add(converter_name)
                # create code for converter in appropriate header file
                converter_stream.write(
                    """\
{signature} {{
    {dtype}{veclen} vec;
    #pragma unroll
    for (int u = 0; u < {veclen}; ++u) {{
        vec[u] = ptr[u];
    }}
    return vec;
}}\n\n""".format(signature=signature, dtype=ctype, veclen=veclen), sdfg)

            # add forward declaration
            function_stream.write("extern {};".format(signature), sdfg, self)

    def generate_converters(self, sdfg, function_stream):
        for unpack, ctype, veclen in self.converters_to_generate:
            self._generate_converter(unpack, ctype, veclen, sdfg, function_stream)

    def unparse_tasklet(self, sdfg, state_id, dfg, node, function_stream, callsite_stream, locals, ldepth,
                        toplevel_schedule):
        if node.label is None or node.label == "":
            return ''

        state_dfg: SDFGState = sdfg.nodes()[state_id]

        # Not [], "" or None
        if not node.code:
            return ''
        # Not [], "" or None
        if node.code_global and node.code_global.code:
            function_stream.write(
                codeblock_to_cpp(node.code_global),
                sdfg,
                state_id,
                node,
            )
            function_stream.write("\n", sdfg, state_id, node)

        # If raw C++ or OpenCL code, return the code directly
        if node.language != dtypes.Language.Python:
            if node.language != dtypes.Language.CPP and node.language != dtypes.Language.OpenCL:
                raise ValueError("Only Python, C++ and OpenCL code are supported in Intel FPGA codegen, got: {}".format(
                    node.language))
            callsite_stream.write(type(node).__properties__["code"].to_string(node.code), sdfg, state_id, node)
            return

        body = node.code.code

        callsite_stream.write('// Tasklet code (%s)\n' % node.label, sdfg, state_id, node)

        # Map local names to memlets (for WCR detection)
        memlets = {}
        for edge in state_dfg.all_edges(node):
            u, uconn, v, vconn, memlet = edge
            if u == node:
                if uconn in u.out_connectors:
                    conntype = u.out_connectors[uconn]
                else:
                    conntype = None

                # this could be a wcr
                memlets[uconn] = (memlet, not edge.data.wcr_nonatomic, edge.data.wcr, conntype)
            elif v == node:
                if vconn in v.in_connectors:
                    conntype = v.in_connectors[vconn]
                else:
                    conntype = None
                memlets[vconn] = (memlet, False, None, conntype)

        # Build dictionary with all the previously defined symbols
        # This is used for forward type inference
        defined_symbols = state_dfg.symbols_defined_at(node)

        # This could be problematic for numeric constants that have no dtype
        defined_symbols.update(
            {k: v.dtype if hasattr(v, 'dtype') else dtypes.typeclass(type(v))
             for k, v in sdfg.constants.items()})

        for connector, (memlet, _, _, conntype) in memlets.items():
            if connector is not None:
                defined_symbols.update({connector: conntype})

        for stmt in body:  # for each statement in tasklet body
            stmt = copy.deepcopy(stmt)
            ocl_visitor = OpenCLDaceKeywordRemover(sdfg, self._dispatcher.defined_vars, memlets, self)

            if isinstance(stmt, ast.Expr):
                rk = ocl_visitor.visit_TopLevelExpr(stmt)
            else:
                rk = ocl_visitor.visit(stmt)

            # Generate width converters
            self.converters_to_generate |= ocl_visitor.width_converters

            if rk is not None:
                result = StringIO()
                cppunparse.CPPUnparser(rk,
                                       ldepth + 1,
                                       locals,
                                       result,
                                       defined_symbols=defined_symbols,
                                       type_inference=True,
                                       language=dtypes.Language.OpenCL)
                callsite_stream.write(result.getvalue(), sdfg, state_id, node)

    def generate_constants(self, sdfg, callsite_stream):
        # To avoid a constant being multiple defined, define it once and
        # declare it as extern everywhere else.

        for cstname, (csttype, cstval) in sdfg.constants_prop.items():
            if isinstance(csttype, dace.data.Array):
                const_str = "__constant " + csttype.dtype.ctype + \
                            " " + cstname + "[" + str(cstval.size) + "]"

                if cstname not in self.generated_constants:
                    # First time, define it
                    self.generated_constants.add(cstname)
                    const_str += " = {"
                    it = np.nditer(cstval, order='C')
                    for i in range(cstval.size - 1):
                        const_str += str(it[0]) + ", "
                        it.iternext()
                    const_str += str(it[0]) + "};\n"
                else:
                    # only define
                    const_str = "extern " + const_str + ";\n"
                callsite_stream.write(const_str, sdfg)
            else:
                # This is a scalar: defining it as an extern variable has the drawback
                # that it is not resolved at compile time, preventing the compiler to
                # allocate fast memory. Therefore, we will use a #define
                callsite_stream.write(f"#define {cstname} {sym2cpp(cstval)}\n", sdfg)

    def generate_tasklet_postamble(self, sdfg, dfg, state_id, node, function_stream, callsite_stream,
                                   after_memlets_stream):
        super().generate_tasklet_postamble(sdfg, dfg, state_id, node, function_stream, callsite_stream,
                                           after_memlets_stream)
        self.generate_channel_writes(sdfg, dfg, node, after_memlets_stream, state_id)

    def write_and_resolve_expr(self, sdfg, memlet, nc, outname, inname, indices=None, dtype=None):
        desc = sdfg.arrays[memlet.data]
        offset = cpp.cpp_offset_expr(desc, memlet.subset, None)
        ptrname = cpp.ptr(memlet.data, desc, sdfg, self._frame)
        defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
        return self.make_write(defined_type, dtype, ptrname, ptrname, offset, inname, memlet.wcr, False, 1)

    def make_ptr_vector_cast(self, dst_expr, dst_dtype, src_dtype, is_scalar, defined_type):
        """
        Cast a destination pointer so the source expression can be written to it.

        :param dst_expr: Expression of the target pointer.
        :param dst_dtype: Type of the target pointer.
        :param src_dtype: Type of the variable that needs to be written.
        :param is_scalar: Whether the variable to be written is a scalar.
        :param defined_type: The code generated variable type of the
                             destination.
        """
        vtype = self.make_vector_type(src_dtype, False)
        expr = dst_expr
        if dst_dtype != src_dtype:
            if is_scalar:
                expr = f"*({vtype} *)(&{dst_expr})"
            elif src_dtype.base_type != dst_dtype:
                expr = f"({vtype})(&{expr})"
            elif defined_type == DefinedType.Pointer:
                expr = "&" + expr
        elif not is_scalar:
            expr = "&" + expr
        return expr

    def process_out_memlets(self, sdfg, state_id, node, dfg, dispatcher, result, locals_defined, function_stream,
                            **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen.process_out_memlets(sdfg,
                                              state_id,
                                              node,
                                              dfg,
                                              dispatcher,
                                              result,
                                              locals_defined,
                                              function_stream,
                                              codegen=self,
                                              **kwargs)
        # Inject undefines
        self.generate_undefines(sdfg, dfg, node, result)


class OpenCLDaceKeywordRemover(cpp.DaCeKeywordRemover):
    """
    Removes Dace Keywords and enforces OpenCL compliance
    """

    nptypes_to_ctypes = {'float64': 'double', 'float32': 'float', 'int32': 'int', 'int64': 'long'}
    nptypes = ['float64', 'float32', 'int32', 'int64']
    ctypes = [
        'bool', 'char', 'cl_char', 'unsigned char', 'uchar', 'cl_uchar', 'short', 'cl_short', 'unsigned short',
        'ushort', 'int', 'unsigned int', 'uint', 'long', 'unsigned long', 'ulong', 'float', 'half', 'size_t',
        'ptrdiff_t', 'intptr_t', 'uintptr_t', 'void', 'double'
    ]

    def __init__(self, sdfg, defined_vars, memlets, codegen):
        self.sdfg = sdfg
        self.defined_vars = defined_vars
        # Keep track of the different streams used in a tasklet
        self.used_streams = []
        self.width_converters = set()  # Pack and unpack vectors
        self.dtypes = {k: v[3] for k, v in memlets.items() if k is not None}  # Type inference
        # consider also constants: add them to known dtypes
        for k, v in sdfg.constants.items():
            if k is not None:
                self.dtypes[k] = v.dtype

        super().__init__(sdfg, memlets, sdfg.constants, codegen)

    def visit_Assign(self, node):
        target = rname(node.targets[0])
        if target not in self.memlets:
            # If we don't have a memlet for this target, it could be the case
            # that on the right hand side we have a constant (a Name or a subscript)
            # If this is the case, we try to infer the type, otherwise we fallback to generic visit
            if ((isinstance(node.value, ast.Name) and node.value.id in self.constants)
                    or (isinstance(node.value, ast.Subscript) and node.value.value.id in self.constants)):
                dtype = infer_expr_type(unparse(node.value), self.dtypes)
                value = cppunparse.cppunparse(self.visit(node.value), expr_semicolon=False)
                code_str = "{} {} = {};".format(dtype, target, value)
                updated = ast.Name(id=code_str)
                return updated
            else:
                return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[target]
        is_scalar = not isinstance(dtype, dtypes.pointer)

        value = cppunparse.cppunparse(self.visit(node.value), expr_semicolon=False)

        veclen_lhs = self.sdfg.data(memlet.data).veclen
        try:
            dtype_rhs = infer_expr_type(unparse(node.value), self.dtypes)
        except SyntaxError:
            # non-valid python
            dtype_rhs = None

        if dtype_rhs is None:
            # If we don't understand the vector length of the RHS, assume no
            # conversion is needed
            veclen_rhs = veclen_lhs
        else:
            veclen_rhs = dtype_rhs.veclen

        if ((veclen_lhs > veclen_rhs and veclen_rhs != 1) or (veclen_lhs < veclen_rhs and veclen_lhs != 1)):
            raise ValueError("Conflicting memory widths: {} and {}".format(veclen_lhs, veclen_rhs))

        if veclen_rhs > veclen_lhs:
            veclen = veclen_rhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((True, ocltype, veclen))
            unpack_str = "unpack_{}{}".format(ocltype, cpp.sym2cpp(veclen))

        if veclen_lhs > veclen_rhs and isinstance(dtype_rhs, dace.pointer):
            veclen = veclen_lhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((False, ocltype, veclen))
            pack_str = "pack_{}{}".format(ocltype, cpp.sym2cpp(veclen))
            # TODO: Horrible hack to not dereference pointers if we have to
            # unpack it
            if value[0] == "*":
                value = value[1:]
            value = "{}({})".format(pack_str, value)

        defined_type, _ = self.defined_vars.get(target)

        if defined_type == DefinedType.Pointer:
            # In case of wcr over an array, resolve access to pointer, replacing the code inside
            # the tasklet
            if isinstance(node.targets[0], ast.Subscript):

                if veclen_rhs > veclen_lhs:
                    code_str = unpack_str + "({src}, &{dst}[{idx}]);"
                else:
                    code_str = "{dst}[{idx}] = {src};"
                slice = self.visit(node.targets[0].slice)
                if (isinstance(slice, ast.Slice) and isinstance(slice.value, ast.Tuple)):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                if wcr is not None:
                    redtype = operations.detect_reduction_type(wcr)
                    red_str = REDUCTION_TYPE_TO_PYEXPR[redtype].format(a="{}[{}]".format(memlet.data, subscript),
                                                                       b=value)
                    code_str = code_str.format(dst=memlet.data, idx=subscript, src=red_str)
                else:
                    code_str = code_str.format(dst=target, idx=subscript, src=value)
            else:  # Target has no subscript
                if veclen_rhs > veclen_lhs:
                    code_str = unpack_str + "({}, {});".format(value, target)
                else:
                    if self.defined_vars.get(target)[0] == DefinedType.Pointer:
                        code_str = "*{} = {};".format(target, value)
                    else:
                        code_str = "{} = {};".format(target, value)
            updated = ast.Name(id=code_str)

        elif (defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray):
            if memlet.dynamic or memlet.num_accesses != 1:
                updated = ast.Name(id="write_channel_intel({}, {});".format(target, value))
                self.used_streams.append(target)
            else:
                # in this case for an output stream we have
                # previously defined an output local var: we use that one
                # instead of directly writing to channel
                updated = ast.Name(id="{} = {};".format(target, value))
        elif memlet is not None and (not is_scalar or memlet.dynamic):
            newnode = ast.Name(id="*{} = {}; ".format(target, value))
            return ast.copy_location(newnode, node)
        elif defined_type == DefinedType.Scalar:
            code_str = "{} = {};".format(target, value)
            updated = ast.Name(id=code_str)
        else:
            raise RuntimeError("Unhandled case: {}, type {}, veclen {}, "
                               "memory size {}, {} accesses".format(target, defined_type, veclen_lhs, veclen_lhs,
                                                                    memlet.num_accesses))

        return ast.copy_location(updated, node)

    def visit_BinOp(self, node):
        if node.op.__class__.__name__ == 'Pow':
            # Special case for integer power: do not generate dace namespaces (dace::math) but just call pow
            if not (isinstance(node.right,
                               (ast.Num, ast.Constant)) and int(node.right.n) == node.right.n and node.right.n >= 0):
                left_value = cppunparse.cppunparse(self.visit(node.left), expr_semicolon=False)
                right_value = cppunparse.cppunparse(self.visit(node.right), expr_semicolon=False)
                updated = ast.Name(id="pow({},{})".format(left_value, right_value))
                return ast.copy_location(updated, node)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[node.id]
        defined_type, _ = self.defined_vars.get(node.id)
        updated = node

        if ((defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray) and memlet.dynamic):
            # Input memlet, we read from channel
            # we should not need mangle here, since we are in a tasklet
            updated = ast.Call(func=ast.Name(id="read_channel_intel"), args=[ast.Name(id=node.id)], keywords=[])
            self.used_streams.append(node.id)
        elif defined_type == DefinedType.Pointer and memlet.dynamic:
            # if this has a variable number of access, it has been declared
            # as a pointer. We need to deference it
            if isinstance(node.id, ast.Subscript):
                slice = self.visit(node.id.slice)
                if isinstance(slice.value, ast.Tuple):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                updated = ast.Name(id="{}[{}]".format(node.id, subscript))
            else:  # no subscript
                updated = ast.Name(id="*{}".format(node.id))

        return ast.copy_location(updated, node)

    # Replace default modules (e.g., math) with OpenCL Compliant (e.g. "dace::math::"->"")
    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind(".")]
        func_name = attrname[attrname.rfind(".") + 1:]
        if module_name in dtypes._OPENCL_ALLOWED_MODULES:
            cppmodname = dtypes._OPENCL_ALLOWED_MODULES[module_name]
            return ast.copy_location(ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # enforce compliance to OpenCL
        # Type casting:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.ctypes:
                node.func.id = "({})".format(node.func.id)
            elif node.func.id in self.nptypes_to_ctypes:
                # if it as numpy type, convert to C type
                node.func.id = "({})".format(self.nptypes_to_ctypes[node.func.id])
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.ctypes:
                node.func.attr = "({})".format(node.func.attr)
            elif node.func.attr in self.nptypes_to_ctypes:
                # if it as numpy type, convert to C type
                node.func.attr = "({})".format(self.nptypes_to_ctypes[node.func.attr])
        elif (isinstance(node.func, (ast.Num, ast.Constant))
              and (node.func.n.to_string() in self.ctypes or node.func.n.to_string() in self.nptypes)):
            new_node = ast.Name(id="({})".format(node.func.n), ctx=ast.Load)
            new_node = ast.copy_location(new_node, node)
            node.func = new_node

        return self.generic_visit(node)
