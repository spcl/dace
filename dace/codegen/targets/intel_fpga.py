import ast
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

        compiler = make_absolute(
            Config.get("compiler", "intel_fpga", "executable"))
        host_flags = Config.get("compiler", "intel_fpga", "host_flags")
        kernel_flags = Config.get("compiler", "intel_fpga", "kernel_flags")
        mode = Config.get("compiler", "intel_fpga", "mode")
        target_board = Config.get("compiler", "intel_fpga", "board")
        enable_debugging = ("ON"
                            if Config.get_bool("compiler", "intel_fpga",
                                               "enable_debugging") else "OFF")
        options = [
            "-DINTELFPGAOCL_ROOT_DIR={}".format(
                os.path.dirname(os.path.dirname(compiler))),
            "-DDACE_INTELFPGA_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_INTELFPGA_KERNEL_FLAGS=\"{}\"".format(kernel_flags),
            "-DDACE_INTELFPGA_MODE={}".format(mode),
            "-DDACE_INTELFPGA_TARGET_BOARD=\"{}\"".format(target_board),
            "-DDACE_INTELFPGA_ENABLE_DEBUGGING={}".format(enable_debugging),
        ]
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
DACE_EXPORTED int __dace_init_intel_fpga({signature}) {{{emulation_flag}
    hlslib::ocl::GlobalContext().MakeProgram({kernel_file_name});
    return 0;
}}

{host_code}""".format(
            signature=self._global_sdfg.signature(),
            emulation_flag=emulation_flag,
            kernel_file_name=kernel_file_name,
            host_code="".join([
                "{separator}\n// Kernel: {kernel_name}"
                "\n{separator}\n\n{code}\n\n".format(
                    separator="/" * 79, kernel_name=name, code=code)
                for (name, code) in self._host_codes
            ])))

        host_code_obj = CodeObject(
            self._program_name,
            host_code.getvalue(),
            "cpp",
            IntelFPGACodeGen,
            "Intel FPGA",
            target_type="host")

        kernel_code_objs = [
            CodeObject(
                kernel_name,
                code,
                "cl",
                IntelFPGACodeGen,
                "Intel FPGA",
                target_type="device")
            for (kernel_name, code) in self._kernel_codes
        ]

        return [host_code_obj] + kernel_code_objs

    def define_stream(self, dtype, vector_length, buffer_size, var_name,
                      array_size, function_stream, kernel_stream):
        vec_type = self.make_vector_type(dtype, vector_length, False)
        if buffer_size > 1:
            depth_attribute = " __attribute__((depth({})))".format(buffer_size)
        else:
            depth_attribute = ""
        if cpp.sym2cpp(array_size) != "1":
            size_str = "[" + cpp.sym2cpp(array_size) + "]"
        else:
            size_str = ""
        kernel_stream.write("channel {} {}{}{};".format(
            vec_type, var_name, size_str, depth_attribute))

    def define_local_array(self, dtype, vector_length, var_name, array_size,
                           storage, shape, function_stream, kernel_stream,
                           sdfg, state_id, node):
        vec_type = self.make_vector_type(dtype, vector_length, False)
        if storage == dace.dtypes.StorageType.FPGA_Registers:
            attributes = " __attribute__((register))"
        else:
            attributes = ""
        kernel_stream.write("{}{} {}[{}];\n".format(vec_type, attributes,
                                                    var_name,
                                                    cpp.sym2cpp(array_size)))

    @staticmethod
    def make_vector_type(dtype, vector_length, is_const):
        return "{}{}{}".format(
            "const " if is_const else "", dtype.ctype, vector_length
            if str(vector_length) != "1" else "")

    def make_kernel_argument(self, data, var_name, vector_length, is_output,
                             with_vectorization):
        if isinstance(data, dace.data.Array):
            if with_vectorization:
                vec_type = self.make_vector_type(data.dtype, vector_length,
                                                 False)
            else:
                vec_type = data.dtype.ctype
            return "__global volatile  {}* restrict {}".format(
                vec_type, var_name)
        elif isinstance(data, dace.data.Stream):
            return None  # Streams are global objects
        else:
            return data.signature(with_types=True, name=var_name)

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

    @staticmethod
    def make_read(defined_type, type_str, var_name, vector_length, expr,
                  index):
        if defined_type in [DefinedType.Stream, DefinedType.StreamView]:
            return "read_channel_intel({})".format(expr)
        elif defined_type == DefinedType.StreamArray:
            # remove "[0]" index as this is not allowed if the subscripted value is not an array
            expr = expr.replace("[0]", "")
            return "read_channel_intel({})".format(expr)
        elif defined_type == DefinedType.Pointer:
            return "*({}{})".format(expr, " + " + index if index else "")
        elif defined_type == DefinedType.Scalar:
            return var_name
        raise NotImplementedError(
            "Unimplemented read type: {}".format(defined_type))

    @staticmethod
    def make_write(defined_type, type_str, var_name, vector_length, write_expr,
                   index, read_expr, wcr):
        """
        Creates write expression, taking into account wcr if present
        """
        if wcr is not None:
            redtype = operations.detect_reduction_type(wcr)

        if defined_type in [DefinedType.Stream, DefinedType.StreamView]:
            return "write_channel_intel({}, {});".format(write_expr, read_expr)
        elif defined_type == DefinedType.StreamArray:
            if index != "0":
                return "write_channel_intel({}[{}], {});".format(
                    write_expr, index, read_expr)
            else:
                # remove "[0]" index as this is not allowed if the subscripted values is not an array
                write_expr = write_expr.replace("[0]", "")
                return "write_channel_intel({}, {});".format(
                    write_expr, read_expr)
        elif defined_type == DefinedType.Pointer:
            if wcr is not None:
                if redtype != dace.dtypes.ReductionType.Min and redtype != dace.dtypes.ReductionType.Max:
                    return "{}[{}] = {}[{}] {} {};".format(
                        write_expr, index, write_expr, index,
                        REDUCTION_TYPE_TO_HLSLIB[redtype], read_expr)
                else:
                    # use max/min opencl builtins
                    return "{}[{}] = {}{}({}[{}],{});".format(
                        write_expr, index, ("f" if type_str == "float"
                                            or type_str == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr, index,
                        read_expr)
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
                        write_expr, ("f" if type_str == "float"
                                     or type_str == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[redtype], write_expr,
                        read_expr)
            else:
                return "{} = {};".format(var_name, read_expr)
        raise NotImplementedError(
            "Unimplemented write type: {}".format(defined_type))

    @staticmethod
    def make_reduction(sdfg, state_id, node, output_memlet, dtype,
                       vector_length_in, vector_length_out, output_type,
                       reduction_type, callsite_stream, iterators_inner,
                       input_subset, identity, out_var, in_var):
        """
        Generates reduction loop body
        """
        axes = node.axes

        # If axes were not defined, use all input dimensions
        if axes is None:
            axes = tuple(range(input_subset.dims()))

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
            if reduction_type != dace.dtypes.ReductionType.Min and reduction_type != dace.dtypes.ReductionType.Max:
                callsite_stream.write(
                    "{} = {} {} {};".format(
                        out_var, prev_var,
                        REDUCTION_TYPE_TO_HLSLIB[reduction_type], in_var),
                    sdfg, state_id, node)
            else:
                # use max/min opencl builtins
                callsite_stream.write(
                    "{} = {}{}({}, {});".format(
                        out_var, ("f" if output_type == "float"
                                  or output_type == "double" else ""),
                        REDUCTION_TYPE_TO_HLSLIB[reduction_type], prev_var,
                        in_var), sdfg, state_id, node)

        else:
            # If this is the first iteration, assign the value read from the
            # input directly to the output
            if reduction_type != dace.dtypes.ReductionType.Min and reduction_type != dace.dtypes.ReductionType.Max:
                callsite_stream.write(
                    "{} = ({}) ? ({}) : {} {} {};".format(
                        out_var, is_first_iteration, in_var, out_var,
                        REDUCTION_TYPE_TO_HLSLIB[reduction_type], in_var),
                    sdfg, state_id, node)
            else:
                callsite_stream.write(
                    "{} = ({}) ? ({}) : {}{}({}, {});".format(
                        out_var, is_first_iteration, in_var,
                        ("f"
                         if output_type == "float" or output_type == "double"
                         else ""), REDUCTION_TYPE_TO_HLSLIB[reduction_type],
                        out_var, in_var), sdfg, state_id, node)

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

        kernel_stream.write("#include <dace/intel_fpga/device.h>\n\n", sdfg)
        self.generate_constants(sdfg, kernel_stream)
        kernel_stream.write("\n", sdfg)

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
                                               callsite_stream, kernel_stream)
            self._dispatcher.dispatch_initialize(
                sdfg, state, state_id, node, callsite_stream, kernel_stream)

        kernel_stream.write("\n")

        # Generate host code
        self.generate_host_function_boilerplate(
            sdfg, state, kernel_name, global_data_parameters + sc_parameters,
            symbol_parameters, nested_global_transients, host_code_body_stream,
            function_stream, callsite_stream)

        self.generate_host_function_prologue(sdfg, state,
                                             host_code_body_stream)

        self.generate_modules(sdfg, state, kernel_name, subgraphs,
                              subgraph_parameters, sc_parameters,
                              symbol_parameters, kernel_stream,
                              host_code_header_stream, host_code_body_stream)

        kernel_stream.write("\n")

        self.generate_host_function_epilogue(sdfg, state,
                                             host_code_body_stream)

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
        host_stream.write(
            """\
  const auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::future<std::pair<double, double>>> futures;
  for (auto &k : kernels) {
    futures.emplace_back(k.ExecuteTaskAsync());
  }
  for (auto &f : futures) {
    f.wait();
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const double elapsedChrono = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Kernel executed in " << elapsedChrono << " seconds.\\n" << std::flush;
}""", sdfg, sdfg.node_id(state))

    def generate_module(self, sdfg, state, name, subgraph, parameters,
                        symbol_parameters, module_stream, host_header_stream,
                        host_body_stream):

        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        # Treat scalars and symbols the same, assuming there are no scalar
        # outputs
        symbol_sigs = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]
        symbol_names = symbol_parameters.keys()

        kernel_args_opencl = []
        kernel_args_host = []
        kernel_args_call = []
        added = set()
        for is_output, pname, p in parameters:
            # Don't make duplicate arguments for other types than arrays
            if pname in added:
                continue
            added.add(pname)
            if isinstance(p, dace.data.Array):
                arg = self.make_kernel_argument(
                    p, pname, self._memory_widths[pname], is_output, True)
            else:
                arg = self.make_kernel_argument(p, pname, 1, is_output, True)
            if arg is not None:
                kernel_args_opencl.append(arg)
                kernel_args_host.append(p.signature(True, name=pname))
                kernel_args_call.append(pname)

        kernel_args_opencl += symbol_sigs
        kernel_args_host += symbol_sigs
        kernel_args_call += symbol_names

        module_function_name = "module_" + name

        # Unrolling processing elements: if there first scope of the subgraph
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
                # Unrolled processing elements
                self._unrolled_pes.add(scope.map)
                kernel_args_opencl += [
                    "const int " + p for p in scope.params
                ]  # PE id will be a macro defined constant
                kernel_args_call += [p for p in scope.params]
                unrolled_loops += 1

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
                "__kernel void {}({}) {{".format(
                    module_function_name, ", ".join(kernel_args_opencl)), sdfg,
                state_id)
        else:
            # Unrolled PEs: we have to generate a kernel for each PE. We will generate
            # a function that will be used create a kernel multiple times
            module_body_stream.write(
                "inline void {}_func({}) {{".format(
                    name, ", ".join(kernel_args_opencl)), sdfg, state_id)

        # Allocate local transients
        data_to_allocate = (set(subgraph.top_level_transients()) - set(
            sdfg.shared_transients()) - set([p[1] for p in parameters]))
        allocated = set()
        for node in subgraph.nodes():
            if not isinstance(node, dace.graph.nodes.AccessNode):
                continue
            if node.data not in data_to_allocate or node.data in allocated:
                continue
            allocated.add(node.data)
            self._dispatcher.dispatch_allocate(
                sdfg, state, state_id, node, module_stream, module_body_stream)
            self._dispatcher.dispatch_initialize(
                sdfg, state, state_id, node, module_stream, module_body_stream)

        self._dispatcher.dispatch_subgraph(
            sdfg,
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
}}\\\n\n""".format(module_function_name, ", " if len(kernel_args_call) > 1 else
                   "", ",".join(kernel_args_call[:-1]), module_function_name,
                   ", ".join(kernel_args_opencl[:-1]), name, ", ".join(
                       kernel_args_call[:-1]), ", "
                   if len(kernel_args_call) > 1 else ""))

        for ul in self._unrolled_pes:
            # create PE kernels by using the previously defined macro
            start, stop, skip = ul.range.ranges[0]
            start_idx = evaluate(start, sdfg.constants)
            stop_idx = evaluate(stop, sdfg.constants)
            skip_idx = evaluate(skip, sdfg.constants)
            # First macro argument is the processing element id
            for p in range(start_idx, stop_idx + 1, skip_idx):
                module_stream.write("_DACE_FPGA_KERNEL_{}({}{}{})\n".format(
                    module_function_name, p, ", " if len(kernel_args_call) > 1
                    else "", ", ".join(kernel_args_call[:-1])))
            module_stream.write(
                "#undef _DACE_FPGA_KERNEL_{}\n".format(module_function_name))
        self._dispatcher.defined_vars.exit_scope(subgraph)

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):

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
                if isinstance(edge.src, dace.graph.nodes.CodeNode):
                    raise NotImplementedError(
                        "Tasklet to tasklet memlets not implemented")

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
                if isinstance(edge.dst, dace.graph.nodes.CodeNode):
                    raise NotImplementedError(
                        "Tasklet to tasklet memlets not implemented")

                else:
                    dst_node = find_output_arraynode(state_dfg, edge)
                    self._dispatcher.dispatch_copy(
                        node, dst_node, edge, sdfg, state_dfg, state_id,
                        function_stream, callsite_stream)

                # Also define variables in the C++ unparser scope
                self._cpu_codegen._locals.define(edge.src_conn, -1,
                                                 self._cpu_codegen._ldepth + 1)
                arrays.add(edge.src_conn)

        callsite_stream.write("\n////////////////////\n", sdfg, state_id, node)

        self.unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                             callsite_stream, self._cpu_codegen._locals,
                             self._cpu_codegen._ldepth,
                             self._cpu_codegen._toplevel_schedule)

        callsite_stream.write("////////////////////\n\n", sdfg, state_id, node)

        # Process outgoing memlets
        self.generate_memlet_outputs(sdfg, state_dfg, state_id, node,
                                     callsite_stream, function_stream)
        self.generate_undefines(sdfg, state_dfg, node, callsite_stream)

        for edge in state_dfg.out_edges(node):
            datadesc = sdfg.arrays[edge.data.data]
            if (isinstance(datadesc, dace.data.Array) and
                (datadesc.storage == dace.dtypes.StorageType.FPGA_Local
                 or datadesc.storage == dace.dtypes.StorageType.FPGA_Registers)
                    and not cpp.is_write_conflicted(dfg, edge)):
                self.generate_no_dependence_post(
                    edge.src_conn, callsite_stream, sdfg, state_id, node)

        callsite_stream.write('}\n', sdfg, state_id, node)
        self._dispatcher.defined_vars.exit_scope(node)

    def _generate_NestedSDFG(self, sdfg, dfg, state_id, node, function_stream,
                             callsite_stream):

        self._dispatcher.defined_vars.enter_scope(sdfg)

        # If SDFG parent is not set, set it
        state_dfg = sdfg.nodes()[state_id]
        node.sdfg.parent = state_dfg
        node.sdfg._parent_sdfg = sdfg

        # Take care of nested SDFG I/O
        for edge in state_dfg.in_edges(node):
            src_node = find_input_arraynode(state_dfg, edge)
            self._dispatcher.dispatch_copy(src_node, node, edge, sdfg,
                                           state_dfg, state_id,
                                           function_stream, callsite_stream)
        for edge in state_dfg.out_edges(node):
            dst_node = find_output_arraynode(state_dfg, edge)
            self._dispatcher.dispatch_copy(node, dst_node, edge, sdfg,
                                           state_dfg, state_id,
                                           function_stream, callsite_stream)

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
        self.generate_memlet_outputs(sdfg, state_dfg, state_id, node,
                                     callsite_stream, function_stream)

        self.generate_undefines(sdfg, state_dfg, node, callsite_stream)

        self._dispatcher.defined_vars.exit_scope(sdfg)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node,
                                   dst_node, edge, callsite_stream):

        if isinstance(edge.dst, dace.graph.nodes.CodeNode):
            # Input memlet
            connector = edge.dst_conn
            is_output = False
            tasklet = edge.dst
        elif isinstance(edge.src, dace.graph.nodes.CodeNode):
            # Output memlet
            connector = edge.src_conn
            is_output = True
            tasklet = edge.src
        else:
            raise NotImplementedError("Not implemented for {} to {}".format(
                type(edge.src), type(edge.dst)))

        memlet = edge.data
        data_name = memlet.data
        data_desc = sdfg.arrays[data_name]
        memlet_type = self.make_vector_type(
            data_desc.dtype, self._memory_widths[data_name], False)
        offset = cpp.cpp_offset_expr(data_desc, memlet.subset, None,
                                     memlet.veclen)

        result = ""

        def_type = self._dispatcher.defined_vars.get(data_name)
        if def_type == DefinedType.Scalar:
            if memlet.num_accesses == 1:
                if not is_output:
                    # We can pre-read the value
                    result += "{} {} = {};".format(memlet_type, connector,
                                                   data_name)
                else:
                    # The value will be written during the tasklet, and will be
                    # automatically written out after
                    init = ""

                    result += "{} {}{};".format(memlet_type, connector, init)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Scalar)
            else:
                # Variable number of reads or writes
                result += "{} *{} = &{};".format(memlet_type, connector,
                                                 data_name)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Pointer)
        elif def_type == DefinedType.Pointer:
            if memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = {}[{}];".format(memlet_type, connector,
                                                       data_name, offset)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Scalar)
            else:
                if data_desc.storage == dace.dtypes.StorageType.FPGA_Global:
                    qualifiers = "__global volatile "
                else:
                    qualifiers = ""
                result += "{}{} *{} = &{}[{}];".format(
                    qualifiers, memlet_type, connector, data_name, offset)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Pointer)
        elif def_type == DefinedType.Stream:
            if memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    result += "{} {} = read_channel_intel({});".format(
                        memlet_type, connector, data_name)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Scalar)
            else:
                # Desperate times call for desperate measures
                result += "#define {} {} // God save us".format(
                    connector, data_name)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Stream)
        elif def_type == DefinedType.StreamArray:
            if memlet.num_accesses == 1:
                if is_output:
                    result += "{} {};".format(memlet_type, connector)
                else:
                    if offset == 0:
                        result += "{} {} = read_channel_intel({}[{}]);".format(
                            memlet_type, connector, data_name, offset)
                    else:
                        result += "{} {} = read_channel_intel({});".format(
                            memlet_type, connector, data_name)
                self._dispatcher.defined_vars.add(connector,
                                                  DefinedType.Scalar)
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
                                                  DefinedType.StreamArray)
        else:
            raise TypeError("Unknown variable type: {}".format(def_type))

        callsite_stream.write(result, sdfg, state_id, tasklet)

    def generate_memlet_outputs(self, sdfg, dfg, state_id, node,
                                callsite_stream, function_stream):
        for edge in dfg.out_edges(node):
            connector = edge.src_conn
            memlet = edge.data
            data_name = memlet.data
            data_desc = sdfg.arrays[data_name]
            memlet_type = self.make_vector_type(
                data_desc.dtype, self._memory_widths[data_name], False)
            offset = cpp.cpp_offset_expr(data_desc, memlet.subset, None,
                                         memlet.veclen)

            result = ""

            src_def_type = self._dispatcher.defined_vars.get(connector)
            dst_def_type = self._dispatcher.defined_vars.get(data_name)

            read_expr = self.make_read(src_def_type, memlet_type, connector,
                                       self._memory_widths[data_name],
                                       connector, None)

            # create write expression
            write_expr = self.make_write(dst_def_type, memlet_type, data_name,
                                         self._memory_widths[data_name],
                                         data_name, offset, read_expr,
                                         memlet.wcr)

            if isinstance(data_desc, dace.data.Scalar):
                if memlet.num_accesses == 1:
                    # The value will be written during the tasklet, and will be
                    # automatically written out after
                    result += write_expr
                elif memlet.num_accesses == -1:
                    # Variable number of reads or writes
                    pass
                else:
                    raise dace.codegen.codegen.CodegenError(
                        "Unsupported number of accesses {} for scalar {}".
                        format(memlet.num_accesses, connector))
            elif isinstance(data_desc, dace.data.Array):
                if memlet.num_accesses == 1:
                    result += write_expr
                else:
                    pass
            elif isinstance(data_desc, dace.data.Stream):
                if not data_desc.is_stream_array():
                    if memlet.num_accesses == 1:
                        result += write_expr
                    else:
                        # Must happen directly in the code
                        pass
                else:  # is array of streams
                    if memlet.num_accesses == 1:
                        result += write_expr
                    else:
                        # Must happen directly in the code
                        pass
            else:
                raise TypeError("Unknown variable type: {}".format(data_desc))

            callsite_stream.write(result, sdfg, state_id, node)

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
                    callsite_stream.write("#undef {}".format(memlet_name),
                                          sdfg, sdfg.node_id(dfg), node)

    def unparse_tasklet(self, sdfg, state_id, dfg, node, function_stream,
                        callsite_stream, locals, ldepth, toplevel_schedule):
        if node.label is None or node.label == "":
            return ''

        state_dfg: SDFGState = sdfg.nodes()[state_id]

        # Not [], "" or None
        if not node.code:
            return ''
        # Not [], "" or None
        if node.code_global:
            if node.language is not dtypes.Language.CPP:
                raise ValueError(
                    "Global code only supported for C++ tasklets: got {}".
                    format(node.language))
            function_stream.write(
                type(node).__properties__["code_global"].to_string(
                    node.code_global), sdfg, state_id, node)
            function_stream.write("\n", sdfg, state_id, node)

        # If raw C++ code, return the code directly
        if node.language != dtypes.Language.Python:
            if node.language != dtypes.Language.CPP:
                raise ValueError(
                    "Only Python or C++ code supported in CPU codegen, got: {}"
                    .format(node.language))
            callsite_stream.write(
                type(node).__properties__["code"].to_string(node.code), sdfg,
                state_id, node)
            return

        body = node.code

        callsite_stream.write('// Tasklet code (%s)\n' % node.label, sdfg,
                              state_id, node)

        # Map local names to memlets (for WCR detection)
        memlets = {}
        for edge in state_dfg.all_edges(node):
            u, uconn, v, vconn, memlet = edge
            if u == node:
                # this could be a wcr
                memlets[uconn] = (memlet, edge.data.wcr_conflict,
                                  edge.data.wcr)
            elif v == node:
                memlets[vconn] = (memlet, False, None)

        # Build dictionary with all the previously defined symbols
        # This is used for forward type inference
        defined_symbols = state_dfg.scope_tree()[state_dfg.scope_dict()[
            node]].defined_vars

        # Dtypes is a dictionary containing associations name -> type (ctypes)
        # Add defined variables
        defined_symbols = {str(x): x.dtype for x in defined_symbols}
        # This could be problematic for numeric constants that have no dtype
        defined_symbols.update({k: v.dtype for k, v in sdfg.constants.items()})

        for connector, (memlet, _, _) in memlets.items():
            if connector is not None:
                defined_symbols.update({
                    connector:
                    sdfg.arrays[memlet.data].dtype
                })

        used_streams = []
        for stmt in body:  # for each statement in tasklet body
            if isinstance(stmt, ast.Expr):
                ocl_visitor = OpenCLDaceKeywordRemover(sdfg, memlets,
                                                       sdfg.constants)
                rk = ocl_visitor.visit_TopLevelExpr(stmt)
            else:
                ocl_visitor = OpenCLDaceKeywordRemover(
                    sdfg, self._dispatcher.defined_vars, memlets,
                    sdfg.constants)
                rk = ocl_visitor.visit(stmt)
            used_streams.extend(ocl_visitor.used_streams)
            if rk is not None:
                result = StringIO()
                cppunparse.CPPUnparser(
                    rk,
                    ldepth + 1,
                    locals,
                    result,
                    defined_symbols=defined_symbols,
                    type_inference=True)
                callsite_stream.write(result.getvalue(), sdfg, state_id, node)

        # In Intel OpenCL is not possible to have multiple access points to the same channel
        for s in used_streams:
            if used_streams.count(s) > 1:
                raise dace.codegen.codegen.CodegenError(
                    "Multiple access points for stream are forbidden in IntelFPGA (stream \"{}\" "
                    "in tasklet \"{}\")".format(s, node.name))

    def generate_constants(self, sdfg, callsite_stream):
        # Use framecode's generate_constants, but substitute constexpr for
        # __constant
        constant_stream = CodeIOStream()
        self._frame.generate_constants(sdfg, constant_stream)
        constant_string = constant_stream.getvalue()
        constant_string = constant_string.replace("constexpr", "__constant")
        callsite_stream.write(constant_string, sdfg)


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

    def __init__(self, sdfg, defined_vars, memlets, *args, **kwargs):
        self.sdfg = sdfg
        self.defined_vars = defined_vars
        self.used_streams = [
        ]  # keep track of the different streams used in a tasklet
        super().__init__(sdfg, memlets, constants=sdfg.constants)

    def visit_Subscript(self, node):
        target = rname(node)
        if target not in self.memlets and target not in self.constants:
            return self.generic_visit(node)

        slice = self.visit(node.slice)
        if not isinstance(slice, ast.Index):
            raise NotImplementedError('Range subscripting not implemented')

        if isinstance(slice.value, ast.Tuple):
            subscript = cpp.unparse(slice)[1:-1]
        else:
            subscript = cpp.unparse(slice)

        if target in self.constants:
            shape = self.constants[target].shape
        else:
            shape = self.sdfg.arrays[self.memlets[target][0].data].shape
        slice_str = cpp.DaCeKeywordRemover.ndslice_cpp(
            subscript.split(', '), shape)

        newnode = ast.parse('%s[%s]' % (target, slice_str)).body[0].value

        return ast.copy_location(newnode, node)

    def visit_Assign(self, node):
        target = rname(node.targets[0])
        if target not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr = self.memlets[target]

        value = self.visit(node.value)

        defined_type = self.defined_vars.get(memlet.data)
        updated = node

        if defined_type == DefinedType.Pointer:
            # In case of wcr over an array, resolve access to pointer, replacing the code inside
            # the tasklet
            if isinstance(node.targets[0], ast.Subscript):
                slice = self.visit(node.targets[0].slice)
                if isinstance(slice.value, ast.Tuple):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                if wcr is not None:
                    redtype = operations.detect_reduction_type(wcr)
                    target_str = "{}[{}]".format(memlet.data, subscript)
                    red_str = REDUCTION_TYPE_TO_PYEXPR[redtype].format(
                        a=target_str, b=unparse(value))
                    code_str = "{} = {};".format(target_str, red_str)
                else:
                    target_str = "{}[{}]".format(target, subscript)
                    code_str = "{} = {}; ".format(target_str, unparse(value))
                updated = ast.Name(id=code_str)
            else:  # target has no subscript
                updated = ast.Name(
                    id="{} = {};".format(target, unparse(value)))

        elif defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray:
            if memlet.num_accesses != 1:
                updated = ast.Name(id="write_channel_intel({}, {});".format(
                    target, cppunparse.cppunparse(value,
                                                  expr_semicolon=False)))
                self.used_streams.append(target)
            else:
                # in this case for an output stream we have
                # previously defined an output local var: we use that one instead of directly writing to channel
                updated = ast.Name(id="{} = {};".format(
                    target, cppunparse.cppunparse(value,
                                                  expr_semicolon=False)))
        elif memlet is not None and memlet.num_accesses != 1:
            newnode = ast.Name(id="*{} = {}; ".format(
                target, cppunparse.cppunparse(value, expr_semicolon=False)))
            return ast.copy_location(newnode, node)

        return ast.copy_location(updated, node)

    def visit_Name(self, node):
        if node.id not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr = self.memlets[node.id]
        defined_type = self.defined_vars.get(memlet.data)
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
        elif (isinstance(node.func, ast.Num)
              and (node.func.n.to_string() in self.ctypes
                   or node.func.n.to_string() in self.nptypes)):
            new_node = ast.Name(id="({})".format(node.func.n), ctx=ast.Load)
            new_node = ast.copy_location(new_node, node)
            node.func = new_node

        return self.generic_visit(node)
