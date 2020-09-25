# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
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
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import make_absolute, DefinedType
from dace.codegen.targets import cpp, fpga
from dace.codegen.targets.common import codeblock_to_cpp
from dace.codegen.tools.type_inference import infer_expr_type
from dace.frontend.python.astutils import rname, unparse
from dace.frontend import operations
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.sdfg import SDFGState
from dace.symbolic import evaluate

REDUCTION_TYPE_TO_HLSLIB = {
    dace.dtypes.ReductionType.Min: "min",
    dace.dtypes.ReductionType.Max: "max",
    dace.dtypes.ReductionType.Sum: "+",
    dace.dtypes.ReductionType.Product: "*",
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


class NameTooLongError(ValueError):
    pass


@registry.autoregister_params(name='intel_fpga')
class IntelFPGACodeGen(fpga.FPGACodeGen):
    target_name = 'intel_fpga'
    title = 'Intel FPGA'
    language = 'hls'

    def __init__(self, *args, **kwargs):
        fpga_vendor = Config.get("compiler", "fpga_vendor")
        if fpga_vendor.lower() != "intel_fpga":
            # Don't register this code generator
            return
        super().__init__(*args, **kwargs)

    @staticmethod
    def cmake_options():

        host_flags = Config.get("compiler", "intel_fpga", "host_flags")
        kernel_flags = Config.get("compiler", "intel_fpga", "kernel_flags")
        mode = Config.get("compiler", "intel_fpga", "mode")
        target_board = Config.get("compiler", "intel_fpga", "board")
        enable_debugging = ("ON" if Config.get_bool(
            "compiler", "intel_fpga", "enable_debugging") else "OFF")
        autobuild = ("ON" if Config.get_bool("compiler", "autobuild_bitstreams")
                     else "OFF")
        options = [
            "-DDACE_INTELFPGA_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_INTELFPGA_KERNEL_FLAGS=\"{}\"".format(kernel_flags),
            "-DDACE_INTELFPGA_MODE={}".format(mode),
            "-DDACE_INTELFPGA_TARGET_BOARD=\"{}\"".format(target_board),
            "-DDACE_INTELFPGA_ENABLE_DEBUGGING={}".format(enable_debugging),
            "-DDACE_FPGA_AUTOBUILD_BITSTREAM={}".format(autobuild)
        ]
        # Override Intel FPGA OpenCL installation directory
        if Config.get("compiler", "intel_fpga", "path"):
            options.append("-DINTELFPGAOCL_ROOT_DIR=\"{}\"".format(
                Config.get("compiler", "intel_fpga", "path").replace("\\",
                                                                     "/")))
        return options

    def get_generated_codeobjects(self):

        execution_mode = Config.get("compiler", "intel_fpga", "mode")
        kernel_file_name = "DACE_BINARY_DIR \"/{}".format(self._program_name)
        emulation_flag = ""
        if execution_mode == "emulator":
            kernel_file_name += "_emulator.aocx\""
            emulation_flag = (
                "\n    dace::set_environment_variable"
                "(\"CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA\", \"1\");")
        elif execution_mode == "simulator":
            kernel_file_name += "_simulator.aocx\""
        elif execution_mode == "hardware":
            kernel_file_name += "_hardware.aocx\""
        else:
            raise dace.codegen.codegen.CodegenError(
                "Unknown Intel FPGA execution mode: {}".format(execution_mode))

        host_code = CodeIOStream()
        host_code.write("""\
#include "dace/intel_fpga/host.h"
#include <iostream>\n\n""")

        self._frame.generate_fileheader(self._global_sdfg, host_code)

        host_code.write("""
dace::fpga::Context *dace::fpga::_context;

DACE_EXPORTED int __dace_init_intel_fpga({signature}) {{{emulation_flag}
    dace::fpga::_context = new dace::fpga::Context();
    dace::fpga::_context->Get().MakeProgram({kernel_file_name});
    return 0;
}}

DACE_EXPORTED void __dace_exit_intel_fpga({signature}) {{
    delete dace::fpga::_context;
}}

{host_code}""".format(signature=self._global_sdfg.signature(),
                      emulation_flag=emulation_flag,
                      kernel_file_name=kernel_file_name,
                      host_code="".join([
                          "{separator}\n// Kernel: {kernel_name}"
                          "\n{separator}\n\n{code}\n\n".format(separator="/" *
                                                               79,
                                                               kernel_name=name,
                                                               code=code)
                          for (name, code) in self._host_codes
                      ])))

        host_code_obj = CodeObject(self._program_name,
                                   host_code.getvalue(),
                                   "cpp",
                                   IntelFPGACodeGen,
                                   "Intel FPGA",
                                   target_type="host")

        kernel_code_objs = [
            CodeObject(kernel_name,
                       code,
                       "cl",
                       IntelFPGACodeGen,
                       "Intel FPGA",
                       target_type="device")
            for (kernel_name, code) in self._kernel_codes
        ]

        return [host_code_obj] + kernel_code_objs

    def define_stream(self, dtype, buffer_size, var_name, array_size,
                      function_stream, kernel_stream):
        vec_type = self.make_vector_type(dtype, False)
        if buffer_size > 1:
            depth_attribute = " __attribute__((depth({})))".format(buffer_size)
        else:
            depth_attribute = ""
        if cpp.sym2cpp(array_size) != "1":
            size_str = "[" + cpp.sym2cpp(array_size) + "]"
        else:
            size_str = ""
        kernel_stream.write("channel {} {}{}{};".format(vec_type, var_name,
                                                        size_str,
                                                        depth_attribute))

        # Return value is used for adding to defined_vars in fpga.py
        return 'channel {}'.format(vec_type)

    def define_local_array(self, var_name, desc, array_size, function_stream,
                           kernel_stream, sdfg, state_id, node):
        vec_type = self.make_vector_type(desc.dtype, False)
        if desc.storage == dace.dtypes.StorageType.FPGA_Registers:
            attributes = " __attribute__((register))"
        else:
            attributes = ""
        kernel_stream.write("{}{} {}[{}];\n".format(vec_type, attributes,
                                                    var_name,
                                                    cpp.sym2cpp(array_size)))
        self._dispatcher.defined_vars.add(var_name, DefinedType.Pointer,
                                          vec_type)

    def define_shift_register(self, *args, **kwargs):
        # Shift registers are just arrays on Intel
        self.define_local_array(*args, **kwargs)

    @staticmethod
    def make_vector_type(dtype, is_const):
        return "{}{}".format("const " if is_const else "", dtype.ocltype)

    def make_kernel_argument(self, data, var_name, is_output,
                             with_vectorization):
        if isinstance(data, dace.data.Array):
            if with_vectorization:
                vec_type = data.dtype.ocltype
            else:
                vec_type = fpga.vector_element_type_of(data.dtype).ocltype
            return "__global volatile  {}* restrict {}".format(
                vec_type, var_name)
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
        kernel_stream.write("#pragma unroll{}".format(factor_str), sdfg,
                            state_id, node)

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

    def make_read(self, defined_type, dtype, var_name, expr, index, is_pack,
                  packing_factor):
        if defined_type == DefinedType.Stream:
            read_expr = "read_channel_intel({})".format(expr)
        elif defined_type == DefinedType.StreamArray:
            # remove "[0]" index as this is not allowed if the subscripted value is not an array
            expr = expr.replace("[0]", "")
            read_expr = "read_channel_intel({})".format(expr)
        elif defined_type == DefinedType.Pointer:
            read_expr = "*({}{})".format(expr, " + " + index if index else "")
        elif defined_type == DefinedType.Scalar:
            read_expr = var_name
        else:
            raise NotImplementedError(
                "Unimplemented read type: {}".format(defined_type))
        if is_pack:
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.converters_to_generate.add((True, ocltype, packing_factor))
            return "pack_{}{}(&({}))".format(ocltype, packing_factor, read_expr)
        else:
            return read_expr

    def make_write(self, defined_type, dtype, var_name, write_expr, index,
                   read_expr, wcr, is_unpack, packing_factor):
        """
        Creates write expression, taking into account wcr if present
        """
        if wcr is not None:
            redtype = operations.detect_reduction_type(wcr)

        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if defined_type == DefinedType.StreamArray:
                if index == "0":
                    # remove "[0]" index as this is not allowed if the
                    # subscripted values is not an array
                    write_expr = write_expr.replace("[0]", "")
                else:
                    write_expr = "{}[{}]".format(write_expr, index)
            if is_unpack:
                return "\n".join("write_channel_intel({}, {}[{}]);".format(
                    write_expr, read_expr, i) for i in range(packing_factor))
            else:
                return "write_channel_intel({}, {});".format(
                    write_expr, read_expr)
        elif defined_type == DefinedType.Pointer:
            if wcr is not None:
                if (redtype != dace.dtypes.ReductionType.Min
                        and redtype != dace.dtypes.ReductionType.Max):
                    return "{}[{}] = {}[{}] {} {};".format(
                        write_expr, index, write_expr, index,
                        REDUCTION_TYPE_TO_HLSLIB[redtype], read_expr)
                else:
                    # use max/min opencl builtins
                    return "{}[{}] = {}{}({}[{}],{});".format(
                        write_expr, index,
                        ("f" if dtype.ocltype == "float"
                         or dtype.ocltype == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr, index,
                        read_expr)
            else:
                if is_unpack:
                    ocltype = fpga.vector_element_type_of(dtype).ocltype
                    self.converters_to_generate.add(
                        (False, ocltype, packing_factor))
                    return "unpack_{}{}({}, &{}[{}]);".format(
                        ocltype, packing_factor, read_expr, write_expr, index)
                else:
                    return "{}[{}] = {};".format(write_expr, index, read_expr)
        elif defined_type == DefinedType.Scalar:
            if wcr is not None:
                if redtype != dace.dtypes.ReductionType.Min and redtype != dace.dtypes.ReductionType.Max:
                    return "{} = {} {} {};".format(
                        write_expr, write_expr,
                        REDUCTION_TYPE_TO_HLSLIB[redtype], read_expr)
                else:
                    # use max/min opencl builtins
                    return "{} = {}{}({},{});".format(
                        write_expr, ("f" if dtype.ocltype == "float"
                                     or dtype.ocltype == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr,
                        read_expr)
            else:
                if is_unpack:
                    ocltype = fpga.vector_element_type_of(dtype).ocltype
                    self.converters_to_generate.add(
                        (False, ocltype, packing_factor))
                    return "unpack_{}{}({}, {});".format(
                        vector_element_type_of(dtype).ocltype, packing_factor,
                        read_expr, var_name)
                else:
                    return "{} = {};".format(var_name, read_expr)
        raise NotImplementedError(
            "Unimplemented write type: {}".format(defined_type))

    def make_shift_register_write(self, defined_type, dtype, var_name,
                                  write_expr, index, read_expr, wcr, is_unpack,
                                  packing_factor):
        if defined_type != DefinedType.Pointer:
            raise TypeError("Intel shift register must be an array: "
                            "{} is {}".format(var_name, defined_type))
        # Shift array
        arr_size = functools.reduce(lambda a, b: a * b,
                                    self._global_sdfg.data(var_name).shape, 1)
        res = """
#pragma unroll
for (int u_{name} = 0; u_{name} < {size} - {veclen}; ++u_{name}) {{
  {name}[u_{name}] = {name}[u_{name} + {veclen}];
}}\n""".format(name=var_name, size=arr_size, veclen=dtype.veclen)
        # Then do write
        res += self.make_write(defined_type, dtype, var_name, write_expr, index,
                               read_expr, wcr, is_unpack, packing_factor)
        return res

    @staticmethod
    def generate_no_dependence_pre(var_name, kernel_stream, sdfg, state_id,
                                   node):
        kernel_stream.write("#pragma ivdep array({})".format(var_name), sdfg,
                            state_id, node)

    @staticmethod
    def generate_no_dependence_post(var_name, kernel_stream, sdfg, state_id,
                                    node):
        pass

    def generate_kernel_internal(self, sdfg, state, kernel_name, subgraphs,
                                 kernel_stream, function_stream,
                                 callsite_stream):

        state_id = sdfg.node_id(state)

        kernel_header_stream = CodeIOStream()
        kernel_body_stream = CodeIOStream()

        kernel_header_stream.write("#include <dace/intel_fpga/device.h>\n\n",
                                   sdfg)
        self.generate_constants(sdfg, kernel_header_stream)
        kernel_header_stream.write("\n", sdfg)

        (global_data_parameters, top_level_local_data, subgraph_parameters,
         scalar_parameters, symbol_parameters,
         nested_global_transients) = self.make_parameters(
             sdfg, state, subgraphs)

        # Scalar parameters are never output
        sc_parameters = [(False, pname, param)
                         for pname, param in scalar_parameters]

        host_code_header_stream = CodeIOStream()
        host_code_body_stream = CodeIOStream()

        # Emit allocations of inter-kernel memories
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                               callsite_stream,
                                               kernel_body_stream)

        kernel_body_stream.write("\n")

        # Generate host code
        self.generate_host_function_boilerplate(
            sdfg, state, kernel_name, global_data_parameters + sc_parameters,
            symbol_parameters, nested_global_transients, host_code_body_stream,
            function_stream, callsite_stream)

        self.generate_host_function_prologue(sdfg, state, host_code_body_stream)

        self.generate_modules(sdfg, state, kernel_name, subgraphs,
                              subgraph_parameters, sc_parameters,
                              symbol_parameters, kernel_body_stream,
                              host_code_header_stream, host_code_body_stream)

        kernel_body_stream.write("\n")

        # Generate data width converters
        self.generate_converters(sdfg, kernel_header_stream)

        kernel_stream.write(kernel_header_stream.getvalue() +
                            kernel_body_stream.getvalue())

        self.generate_host_function_epilogue(sdfg, state, host_code_body_stream)

        # Store code to be passed to compilation phase
        self._host_codes.append(
            (kernel_name, host_code_header_stream.getvalue() +
             host_code_body_stream.getvalue()))

    @staticmethod
    def generate_host_function_prologue(sdfg, state, host_stream):
        host_stream.write("std::vector<hlslib::ocl::Kernel> kernels;", sdfg,
                          sdfg.node_id(state))

    @staticmethod
    def generate_host_function_epilogue(sdfg, state, host_stream):
        state_id = sdfg.node_id(state)
        host_stream.write(
            "const auto start = std::chrono::high_resolution_clock::now();",
            sdfg, state_id)
        launch_async = Config.get_bool("compiler", "intel_fpga", "launch_async")
        if launch_async:
            # hlslib uses std::async to launch each kernel launch as an
            # asynchronous task in a separate C++ thread. This seems to cause
            # problems with some versions of the Intel FPGA runtime, despite it
            # supposedly being thread-safe, so we allow disabling this.
            host_stream.write(
                """\
  std::vector<std::future<std::pair<double, double>>> futures;
  for (auto &k : kernels) {
    futures.emplace_back(k.ExecuteTaskAsync());
  }
  for (auto &f : futures) {
    f.wait();
  }""", sdfg, state_id)
        else:
            # Launch one-by-one and wait for the cl::Events
            host_stream.write(
                """\
  std::vector<cl::Event> events;
  for (auto &k : kernels) {
    events.emplace_back(k.ExecuteTaskFork());
  }
  cl::Event::waitForEvents(events);""", sdfg, state_id)
        host_stream.write(
            """\
  const auto end = std::chrono::high_resolution_clock::now();
  const double elapsedChrono = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Kernel executed in " << elapsedChrono << " seconds.\\n" << std::flush;
}""", sdfg, sdfg.node_id(state))

    def generate_module(self, sdfg, state, name, subgraph, parameters,
                        symbol_parameters, module_stream, host_header_stream,
                        host_body_stream):

        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        kernel_args_opencl = []
        kernel_args_host = []
        kernel_args_call = []
        added = set()
        # Split into arrays and scalars
        arrays = sorted(
            [t for t in parameters if not isinstance(t[2], dace.data.Scalar)],
            key=lambda t: t[1])
        scalars = [t for t in parameters if isinstance(t[2], dace.data.Scalar)]
        scalars += [(False, k, v) for k, v in symbol_parameters.items()]
        scalars = list(sorted(scalars, key=lambda t: t[1]))
        for is_output, pname, p in itertools.chain(arrays, scalars):
            if pname in added:
                continue
            added.add(pname)
            arg = self.make_kernel_argument(p, pname, is_output, True)
            if arg is not None:
                kernel_args_opencl.append(arg)
                kernel_args_host.append(p.as_arg(True, name=pname))
                kernel_args_call.append(pname)

        module_function_name = "module_" + name

        # The official limit suggested by Intel is 61. However, the compiler
        # can also append text to the module. Longest seen so far is
        # "_cra_slave_inst", which is 15 characters, so we restrict to
        # 61 - 15 = 46, and round down to 42 to be conservative.
        if len(module_function_name) > 42:
            raise NameTooLongError(
                "Due to a bug in the Intel FPGA OpenCL compiler, "
                "kernel names cannot be longer than 42 characters:\n\t{}".
                format(module_function_name))

        # Unrolling processing elements: if there first scope of the subgraph
        # is an unrolled map, generate a processing element for each iteration
        scope_dict = subgraph.scope_dict(node_to_children=True)
        top_scopes = [
            n for n in scope_dict[None]
            if isinstance(n, dace.sdfg.nodes.EntryNode)
        ]
        unrolled_loops = 0
        if len(top_scopes) == 1:
            scope = top_scopes[0]
            if scope.unroll:
                # Unrolled processing elements
                self._unrolled_pes.add(scope.map)
                kernel_args_opencl += [
                    "const int " + p for p in scope.params
                ]  # PE id will be a macro defined constant
                kernel_args_call += [p for p in scope.params]
                unrolled_loops += 1

        # Ensure no duplicate parameters are used
        kernel_args_opencl = dtypes.deduplicate(kernel_args_opencl)
        kernel_args_call = dtypes.deduplicate(kernel_args_call)

        # Add kernel call host function
        if unrolled_loops == 0:
            host_body_stream.write(
                "kernels.emplace_back(program.MakeKernel(\"{}\"{}));".format(
                    module_function_name, ", ".join([""] + kernel_args_call)
                    if len(kernel_args_call) > 0 else ""), sdfg, state_id)
        else:
            # We will generate a separate kernel for each PE. Adds host call
            for ul in self._unrolled_pes:
                start, stop, skip = ul.range.ranges[0]
                start_idx = evaluate(start, sdfg.constants)
                stop_idx = evaluate(stop, sdfg.constants)
                skip_idx = evaluate(skip, sdfg.constants)
                # Due to restrictions on channel indexing, PE IDs must start from zero
                # and skip index must be 1
                if start_idx != 0 or skip_idx != 1:
                    raise dace.codegen.codegen.CodegenError(
                        "Unrolled Map in {} should start from 0 and have skip equal to 1"
                        .format(sdfg.name))
                for p in range(start_idx, stop_idx + 1, skip_idx):
                    # last element in list kernel_args_call is the PE ID, but this is
                    # already written in stone in the OpenCL generated code
                    host_body_stream.write(
                        "kernels.emplace_back(program.MakeKernel(\"{}_{}\"{}));"
                        .format(
                            module_function_name, p,
                            ", ".join([""] + kernel_args_call[:-1]) if
                            len(kernel_args_call) > 1 else ""), sdfg, state_id)

        # ----------------------------------------------------------------------
        # Generate kernel code
        # ----------------------------------------------------------------------

        self._dispatcher.defined_vars.enter_scope(subgraph)

        module_body_stream = CodeIOStream()

        if unrolled_loops == 0:
            module_body_stream.write(
                "__kernel void {}({}) {{".format(module_function_name,
                                                 ", ".join(kernel_args_opencl)),
                sdfg, state_id)
        else:
            # Unrolled PEs: we have to generate a kernel for each PE. We will generate
            # a function that will be used create a kernel multiple times
            module_body_stream.write(
                "inline void {}_func({}) {{".format(
                    name, ", ".join(kernel_args_opencl)), sdfg, state_id)

        # Allocate local transients
        data_to_allocate = (set(subgraph.top_level_transients()) -
                            set(sdfg.shared_transients()) -
                            set([p[1] for p in parameters]))
        allocated = set()
        for node in subgraph.nodes():
            if not isinstance(node, dace.sdfg.nodes.AccessNode):
                continue
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                               module_stream,
                                               module_body_stream)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           subgraph,
                                           state_id,
                                           module_stream,
                                           module_body_stream,
                                           skip_entry_node=False)

        module_stream.write(module_body_stream.getvalue(), sdfg, state_id)
        module_stream.write("}\n\n")

        if unrolled_loops > 0:
            # Unrolled PEs: create as many kernels as the number of PEs
            # To avoid long and duplicated code, do it with define (gosh)
            # Since OpenCL is "funny", it does not support variadic macros
            # One of the argument is for sure the PE_ID, which is also the last one in kernel_args lists:
            # it will be not passed by the host but code-generated
            module_stream.write("""#define _DACE_FPGA_KERNEL_{}(PE_ID{}{}) \\
__kernel void \\
{}_##PE_ID({}) \\
{{ \\
  {}_func({}{}PE_ID); \\
}}\\\n\n""".format(module_function_name,
                   ", " if len(kernel_args_call) > 1 else "",
                   ",".join(kernel_args_call[:-1]), module_function_name,
                   ", ".join(kernel_args_opencl[:-1]), name,
                   ", ".join(kernel_args_call[:-1]),
                   ", " if len(kernel_args_call) > 1 else ""))

        for ul in self._unrolled_pes:
            # create PE kernels by using the previously defined macro
            start, stop, skip = ul.range.ranges[0]
            start_idx = evaluate(start, sdfg.constants)
            stop_idx = evaluate(stop, sdfg.constants)
            skip_idx = evaluate(skip, sdfg.constants)
            # First macro argument is the processing element id
            for p in range(start_idx, stop_idx + 1, skip_idx):
                module_stream.write("_DACE_FPGA_KERNEL_{}({}{}{})\n".format(
                    module_function_name, p,
                    ", " if len(kernel_args_call) > 1 else "",
                    ", ".join(kernel_args_call[:-1])))
            module_stream.write(
                "#undef _DACE_FPGA_KERNEL_{}\n".format(module_function_name))
        self._dispatcher.defined_vars.exit_scope(subgraph)

    def _generate_NestedSDFG(self, sdfg, dfg, state_id, node, function_stream,
                             callsite_stream):

        self._dispatcher.defined_vars.enter_scope(sdfg)

        state_dfg = sdfg.nodes()[state_id]

        # Take care of nested SDFG I/O
        for edge in state_dfg.in_edges(node):
            src_node = find_input_arraynode(state_dfg, edge)
            self._dispatcher.dispatch_copy(src_node, node, edge, sdfg,
                                           state_dfg, state_id, function_stream,
                                           callsite_stream)
        for edge in state_dfg.out_edges(node):
            dst_node = find_output_arraynode(state_dfg, edge)
            self._dispatcher.dispatch_copy(node, dst_node, edge, sdfg,
                                           state_dfg, state_id, function_stream,
                                           callsite_stream)

        callsite_stream.write('\n    ///////////////////\n', sdfg, state_id,
                              node)

        sdfg_label = '_%d_%d' % (state_id, dfg.node_id(node))

        # Generate code for internal SDFG
        global_code, local_code, used_targets, used_environments = \
            self._frame.generate_code(node.sdfg, node.schedule, sdfg_label)

        # Write generated code in the proper places (nested SDFG writes
        # location info)
        function_stream.write(global_code)
        callsite_stream.write(local_code)

        callsite_stream.write('    ///////////////////\n\n', sdfg, state_id,
                              node)

        # Process outgoing memlets with the internal SDFG
        self.process_out_memlets(sdfg, state_id, node, state_dfg,
                                 self._dispatcher, callsite_stream, True,
                                 function_stream)

        self._dispatcher.defined_vars.exit_scope(sdfg)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node,
                                   dst_node, edge, callsite_stream):

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
            raise NotImplementedError("Not implemented for {} to {}".format(
                type(edge.src), type(edge.dst)))

        memlet = edge.data
        data_name = memlet.data
        data_desc = sdfg.arrays[data_name]
        data_dtype = data_desc.dtype

        is_scalar = not isinstance(conntype, dtypes.pointer)
        dtype = conntype if is_scalar else conntype._typeclass

        memlet_type = self.make_vector_type(dtype, False)
        offset = cpp.cpp_offset_expr(data_desc, memlet.subset, None)

        if dtype != data_dtype:
            if (isinstance(dtype, dace.vector)
                    and dtype.base_type == data_dtype):
                cast = True
            else:
                raise TypeError("Type mismatch: {} vs. {}".format(
                    dtype, data_dtype))
        else:
            cast = False

        result = ""

        def_type, ctypedef = self._dispatcher.defined_vars.get(data_name)
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
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar,
                                                  memlet_type)
            else:
                # Variable number of reads or writes
                result += "{} *{} = &{};".format(memlet_type, connector, rhs)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Pointer,
                                                  '%s *' % memlet_type)
        elif def_type == DefinedType.Pointer:
            if cast:
                rhs = f"(({memlet_type} const *){data_name})"
            else:
                rhs = data_name
            if is_scalar and not memlet.dynamic:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = {}[{}];".format(memlet_type, connector,
                                                       rhs, offset)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar,
                                                  memlet_type)
            else:
                if data_desc.storage == dace.dtypes.StorageType.FPGA_Global:
                    qualifiers = "__global volatile "
                else:
                    qualifiers = ""
                ctype = '{}{} *'.format(qualifiers, memlet_type)
                result += "{}{} = &{}[{}];".format(ctype, connector, rhs,
                                                   offset)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Pointer, ctype)
        elif def_type == DefinedType.Stream:
            if cast:
                raise TypeError("Cannot cast stream from {} to {}.".format(
                    data_dtype, dtype))
            if not memlet.dynamic and memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = read_channel_intel({});".format(
                        memlet_type, connector, data_name)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar,
                                                  memlet_type)
            else:
                # Desperate times call for desperate measures
                result += "#define {} {} // God save us".format(
                    connector, data_name)
                self._dispatcher.defined_vars.add(connector, DefinedType.Stream,
                                                  ctypedef)
        elif def_type == DefinedType.StreamArray:
            if cast:
                raise TypeError(
                    "Cannot cast stream array from {} to {}.".format(
                        data_dtype, dtype))
            if not memlet.dynamic and memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    if offset == 0:
                        result += "{} {} = read_channel_intel({}[{}]);".format(
                            memlet_type, connector, data_name, offset)
                    else:
                        result += "{} {} = read_channel_intel({});".format(
                            memlet_type, connector, data_name)
                self._dispatcher.defined_vars.add(connector, DefinedType.Scalar,
                                                  memlet_type)
            else:
                # Must happen directly in the code
                # Here we create a macro which take the proper channel
                channel_idx = cpp.cpp_offset_expr(sdfg.arrays[data_name],
                                                  memlet.subset)

                if sdfg.parent is None:
                    result += "#define {} {}[{}] ".format(
                        connector, data_name, channel_idx)
                else:
                    # This is a nested SDFG: `data_name` channel has been already defined at the
                    # parent. Here we can not define the channel name with an array subscript
                    result += "#define {} {} ".format(connector, data_name)

                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.StreamArray,
                                                  ctypedef)
        else:
            raise TypeError("Unknown variable type: {}".format(def_type))

        callsite_stream.write(result, sdfg, state_id, tasklet)

    def generate_channel_writes(self, sdfg, dfg, node, callsite_stream):
        for edge in dfg.out_edges(node):
            connector = edge.src_conn
            memlet = edge.data
            data_name = memlet.data
            if data_name is not None:
                data_desc = sdfg.arrays[data_name]
                if (isinstance(data_desc, dace.data.Stream)
                        and memlet.volume == 1 and not memlet.dynamic):
                    callsite_stream.write(
                        f"write_channel_intel({data_name}, {connector});", sdfg)

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
                if (isinstance(data_desc, dace.data.Stream)
                        and memlet.num_accesses != 1):
                    callsite_stream.write("#undef {}".format(memlet_name), sdfg)

    def _generate_converter(self, is_unpack, ctype, veclen, sdfg,
                            function_stream):
        if is_unpack:
            function_stream.write(
                """\
void unpack_{dtype}{veclen}(const {dtype}{veclen} value, {dtype} *const ptr) {{
    #pragma unroll
    for (int u = 0; u < {veclen}; ++u) {{
        ptr[u] = value[u];
    }}
}}\n\n""".format(dtype=ctype, veclen=veclen), sdfg)
        else:
            function_stream.write(
                """\
{dtype}{veclen} pack_{dtype}{veclen}({dtype} const *const ptr) {{
    {dtype}{veclen} vec;
    #pragma unroll
    for (int u = 0; u < {veclen}; ++u) {{
        vec[u] = ptr[u];
    }}
    return vec;
}}\n\n""".format(dtype=ctype, veclen=veclen), sdfg)

    def generate_converters(self, sdfg, function_stream):
        for unpack, ctype, veclen in self.converters_to_generate:
            self._generate_converter(unpack, ctype, veclen, sdfg,
                                     function_stream)

    def unparse_tasklet(self, sdfg, state_id, dfg, node, function_stream,
                        callsite_stream, locals, ldepth, toplevel_schedule):
        if node.label is None or node.label == "":
            return ''

        state_dfg: SDFGState = sdfg.nodes()[state_id]

        # Not [], "" or None
        if not node.code:
            return ''

        # If raw C++ code, return the code directly
        if node.language != dtypes.Language.Python:
            if node.language != dtypes.Language.CPP:
                raise ValueError(
                    "Only Python or C++ code supported in CPU codegen, got: {}".
                    format(node.language))
            callsite_stream.write(
                type(node).__properties__["code"].to_string(node.code), sdfg,
                state_id, node)
            return

        body = node.code.code

        callsite_stream.write('// Tasklet code (%s)\n' % node.label, sdfg,
                              state_id, node)

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
                memlets[uconn] = (memlet, not edge.data.wcr_nonatomic,
                                  edge.data.wcr, conntype)
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
        defined_symbols.update({k: v.dtype for k, v in sdfg.constants.items()})

        for connector, (memlet, _, _, conntype) in memlets.items():
            if connector is not None:
                defined_symbols.update({connector: conntype})

        for stmt in body:  # for each statement in tasklet body
            stmt = copy.deepcopy(stmt)
            ocl_visitor = OpenCLDaceKeywordRemover(
                sdfg, self._dispatcher.defined_vars, memlets, self)
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
                                       type_inference=True)
                callsite_stream.write(result.getvalue(), sdfg, state_id, node)

    def generate_constants(self, sdfg, callsite_stream):
        # Use framecode's generate_constants, but substitute constexpr for
        # __constant
        constant_stream = CodeIOStream()
        self._frame.generate_constants(sdfg, constant_stream)
        constant_string = constant_stream.getvalue()
        constant_string = constant_string.replace("constexpr", "__constant")
        callsite_stream.write(constant_string, sdfg)

    def generate_tasklet_postamble(self, sdfg, dfg, state_id, node,
                                   function_stream, callsite_stream,
                                   after_memlets_stream):
        super().generate_tasklet_postamble(sdfg, dfg, state_id, node,
                                           function_stream, callsite_stream,
                                           after_memlets_stream)
        self.generate_channel_writes(sdfg, dfg, node, after_memlets_stream)
        self.generate_undefines(sdfg, dfg, node, after_memlets_stream)

    def write_and_resolve_expr(self,
                               sdfg,
                               memlet,
                               nc,
                               outname,
                               inname,
                               indices=None,
                               dtype=None):
        offset = cpp.cpp_offset_expr(sdfg.arrays[memlet.data], memlet.subset,
                                     None)
        defined_type, _ = self._dispatcher.defined_vars.get(memlet.data)
        return self.make_write(defined_type, dtype, memlet.data, memlet.data,
                               offset, inname, memlet.wcr, False, 1)

    def make_ptr_vector_cast(self, sdfg, expr, memlet, conntype, is_scalar,
                             var_type):
        vtype = self.make_vector_type(conntype, False)
        return f"{vtype}({expr})"


class OpenCLDaceKeywordRemover(cpp.DaCeKeywordRemover):
    """
    Removes Dace Keywords and enforces OpenCL compliance
    """

    nptypes_to_ctypes = {'float64': 'double'}
    nptypes = ['float64']
    ctypes = [
        'bool', 'char', 'cl_char', 'unsigned char', 'uchar', 'cl_uchar',
        'short', 'cl_short', 'unsigned short', 'ushort', 'int', 'unsigned int',
        'uint', 'long', 'unsigned long', 'ulong', 'float', 'half', 'size_t',
        'ptrdiff_t', 'intptr_t', 'uintptr_t', 'void', 'double'
    ]

    def __init__(self, sdfg, defined_vars, memlets, codegen):
        self.sdfg = sdfg
        self.defined_vars = defined_vars
        self.used_streams = [
        ]  # keep track of the different streams used in a tasklet
        self.width_converters = set()  # Pack and unpack vectors
        self.dtypes = {k: v[3] for k, v in memlets.items()}  # Type inference
        super().__init__(sdfg, memlets, sdfg.constants, codegen)

    def visit_Assign(self, node):
        target = rname(node.targets[0])
        if target not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[target]
        is_scalar = not isinstance(dtype, dtypes.pointer)

        value = cppunparse.cppunparse(self.visit(node.value),
                                      expr_semicolon=False)

        veclen_lhs = self.sdfg.data(memlet.data).veclen
        dtype_rhs = infer_expr_type(astunparse.unparse(node.value), self.dtypes)
        if dtype_rhs is None:
            # If we don't understand the vector length of the RHS, assume no
            # conversion is needed
            veclen_rhs = veclen_lhs
        else:
            veclen_rhs = dtype_rhs.veclen

        if ((veclen_lhs > veclen_rhs and veclen_rhs != 1)
                or (veclen_lhs < veclen_rhs and veclen_lhs != 1)):
            raise ValueError("Conflicting memory widths: {} and {}".format(
                veclen_lhs, veclen_rhs))

        if veclen_rhs > veclen_lhs:
            veclen = veclen_rhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((True, ocltype, veclen))
            unpack_str = "unpack_{}{}".format(ocltype, veclen)

        if veclen_lhs > veclen_rhs:
            veclen = veclen_lhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((False, ocltype, veclen))
            pack_str = "pack_{}{}".format(ocltype, veclen)
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
                if isinstance(slice.value, ast.Tuple):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                if wcr is not None:
                    redtype = operations.detect_reduction_type(wcr)
                    red_str = REDUCTION_TYPE_TO_PYEXPR[redtype].format(
                        a="{}[{}]".format(memlet.data, subscript), b=value)
                    code_str = code_str.format(dst=memlet.data,
                                               idx=subscript,
                                               src=red_str)
                else:
                    code_str = code_str.format(dst=target,
                                               idx=subscript,
                                               src=value)
            else:  # Target has no subscript
                if veclen_rhs > veclen_lhs:
                    code_str = unpack_str + "({}, {});".format(value, target)
                else:
                    if self.defined_vars.get(target)[0] == DefinedType.Pointer:
                        code_str = "*{} = {};".format(target, value)
                    else:
                        code_str = "{} = {};".format(target, value)
            updated = ast.Name(id=code_str)

        elif (defined_type == DefinedType.Stream
              or defined_type == DefinedType.StreamArray):
            if memlet.num_accesses != 1:
                updated = ast.Name(
                    id="write_channel_intel({}, {});".format(target, value))
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
                               "memory size {}, {} accesses".format(
                                   target, defined_type, veclen_lhs, veclen_lhs,
                                   memlet.num_accesses))

        return ast.copy_location(updated, node)

    def visit_Name(self, node):
        if node.id not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[node.id]
        defined_type, _ = self.defined_vars.get(node.id)
        updated = node

        if (defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray) \
                and memlet.num_accesses != 1:
            # Input memlet, we read from channel
            updated = ast.Name(id="read_channel_intel({})".format(node.id))
            self.used_streams.append(node.id)
        elif defined_type == DefinedType.Pointer and memlet.num_accesses != 1:
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
            return ast.copy_location(
                ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # enforce compliance to OpenCL
        # Type casting:
        if isinstance(node.func, ast.Name) and node.func.id in self.ctypes:
            node.func.id = "({})".format(node.func.id)
        elif isinstance(node.func,
                        ast.Name) and node.func.id in self.nptypes_to_ctypes:
            # if it as numpy type, convert to C type
            node.func.id = "({})".format(self.nptypes_to_ctypes(node.func.id))
        elif (isinstance(node.func, (ast.Num, ast.Constant))
              and (node.func.n.to_string() in self.ctypes
                   or node.func.n.to_string() in self.nptypes)):
            new_node = ast.Name(id="({})".format(node.func.n), ctx=ast.Load)
            new_node = ast.copy_location(new_node, node)
            node.func = new_node

        return self.generic_visit(node)
