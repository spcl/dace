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
from dace.codegen.targets import cpu, fpga

from dace.codegen import cppunparse

REDUCTION_TYPE_TO_HLSLIB = {
    dace.dtypes.ReductionType.Min: "hlslib::op::Min",
    dace.dtypes.ReductionType.Max: "hlslib::op::Max",
    dace.dtypes.ReductionType.Sum: "hlslib::op::Sum",
    dace.dtypes.ReductionType.Product: "hlslib::op::Product",
    dace.dtypes.ReductionType.Logical_And: "hlslib::op::And",
}


class XilinxCodeGen(fpga.FPGACodeGen):
    """ Xilinx FPGA code generator. """

    target_name = 'xilinx'
    title = 'Xilinx'
    language = 'hls'

    def __init__(self, *args, **kwargs):
        fpga_vendor = Config.get("compiler", "fpga_vendor")
        if fpga_vendor.lower() != "xilinx":
            # Don't register this code generator
            return
        super().__init__(*args, **kwargs)

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

    def get_generated_codeobjects(self):

        execution_mode = Config.get("compiler", "xilinx", "mode")
        try:
            sdaccel_dir = os.path.dirname(
                os.path.dirname(
                    make_absolute(
                        Config.get("compiler", "xilinx", "executable"))))
        except ValueError:
            sdaccel_dir = ''

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

        host_code_obj = CodeObject(
            self._program_name,
            host_code.getvalue(),
            "cpp",
            XilinxCodeGen,
            "Xilinx",
            target_type="host")

        kernel_code_objs = [
            CodeObject(
                kernel_name,
                code,
                "cpp",
                XilinxCodeGen,
                "Xilinx",
                target_type="device")
            for (kernel_name, code) in self._kernel_codes
        ]

        return [host_code_obj] + kernel_code_objs

    @staticmethod
    def define_stream(dtype, vector_length, buffer_size, var_name, array_size,
                      function_stream, kernel_stream):
        if cpu.sym2cpp(array_size) == "1":
            kernel_stream.write("dace::FIFO<{}, {}, {}> {}(\"{}\");".format(
                dtype.ctype, vector_length, buffer_size, var_name, var_name))
        else:
            kernel_stream.write("dace::FIFO<{}, {}, {}> {}[{}];\n".format(
                dtype.ctype, vector_length, buffer_size, var_name,
                cpu.sym2cpp(array_size)))
            kernel_stream.write("dace::SetNames({}, \"{}\", {});".format(
                var_name, var_name, cpu.sym2cpp(array_size)))

    @staticmethod
    def define_local_array(dtype, vector_length, var_name, array_size, storage,
                           shape, function_stream, kernel_stream, sdfg,
                           state_id, node):
        kernel_stream.write("dace::vec<{}, {}> {}[{}];\n".format(
            dtype.ctype, vector_length, var_name, cpu.sym2cpp(array_size)))
        if storage == dace.dtypes.StorageType.FPGA_Registers:
            kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} "
                                "complete\n".format(var_name))
        elif len(shape) > 1:
            kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} "
                                "block factor={}\n".format(
                                    var_name, shape[-2]))

    @staticmethod
    def make_vector_type(dtype, vector_length, is_const):
        return "{}dace::vec<{}, {}>".format("const " if is_const else "",
                                            dtype.ctype, vector_length)

    @staticmethod
    def make_kernel_argument(data, var_name, vector_length, is_output,
                             with_vectorization):
        if isinstance(data, dace.data.Array):
            var_name += "_" + ("out" if is_output else "in")
            if with_vectorization:
                return "dace::vec<{}, {}> *{}".format(data.dtype.ctype,
                                                      vector_length, var_name)
            else:
                return "{} *{}".format(data.dtype.ctype, var_name)
        else:
            return data.signature(with_types=True, name=var_name)

    @staticmethod
    def generate_unroll_loop_pre(kernel_stream, factor, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_unroll_loop_post(kernel_stream, factor, sdfg, state_id, node):
        if factor is None:
            kernel_stream.write("#pragma HLS UNROLL", sdfg, state_id, node)
        else:
            kernel_stream.write("#pragma HLS UNROLL factor={}".format(factor),
                                sdfg_state_id, node)

    @staticmethod
    def generate_pipeline_loop_pre(kernel_stream, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_pipeline_loop_post(kernel_stream, sdfg, state_id, node):
        kernel_stream.write("#pragma HLS PIPELINE II=1", sdfg, state_id, node)

    @staticmethod
    def generate_flatten_loop_pre(kernel_stream, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_flatten_loop_post(kernel_stream, sdfg, state_id, node):
        kernel_stream.write("#pragma HLS LOOP_FLATTEN")

    @staticmethod
    def make_read(defined_type, type_str, var_name, vector_length, expr,
                  index):
        if defined_type in [DefinedType.Stream, DefinedType.StreamView]:
            return "{}.pop()".format(expr)
        if defined_type == DefinedType.StreamArray:
            if " " in expr:
                expr = "(" + expr + ")"
            return "{}[{}].pop()".format(expr, index)
        elif defined_type == DefinedType.Scalar:
            return var_name
        else:
            expr = expr + " + " + index if index else expr
            return "dace::Read<{}, {}>({})".format(type_str, vector_length,
                                                   expr)

    @staticmethod
    def make_write(defined_type, type_str, var_name, vector_length, write_expr,
                   index, read_expr, wcr):
        if defined_type in [DefinedType.Stream, DefinedType.StreamView]:
            return "{}.push({});".format(write_expr, read_expr)
        elif defined_type == DefinedType.StreamArray:
            if not index:
                index = "0"
            return "{}[{}].push({}};".format(write_expr, index, read_expr)
        else:
            if defined_type == DefinedType.Scalar:
                write_expr = var_name
            else:
                write_expr = (write_expr + " + " + index
                              if index else write_expr)
            return "dace::Write<{}, {}>({}, {});".format(
                type_str, vector_length, write_expr, read_expr)

    @staticmethod
    def generate_no_dependence_pre(var_name, kernel_stream, sdfg, state_id,
                                   node):
        pass

    @staticmethod
    def generate_no_dependence_post(var_name, kernel_stream, sdfg, state_id,
                                    node):
        kernel_stream.write(
            "#pragma HLS DEPENDENCE variable={} false".format(var_name), sdfg,
            state_id, node)

    def generate_kernel_boilerplate_pre(self, sdfg, state_id, kernel_name,
                                        global_data_parameters,
                                        scalar_parameters, symbol_parameters,
                                        module_stream, kernel_stream):

        # Write header
        module_stream.write(
            """#include <dace/xilinx/device.h>
#include <dace/math.h>
#include <dace/complex.h>""", sdfg)
        self._frame.generate_fileheader(sdfg, module_stream)
        module_stream.write("\n", sdfg)

        symbol_params = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]

        # Build kernel signature
        kernel_args = []
        for is_output, dataname, data in global_data_parameters:
            kernel_arg = self.make_kernel_argument(
                data, dataname, self._memory_widths[dataname], is_output, True)
            if kernel_arg:
                kernel_args.append(kernel_arg)

        kernel_args += ([
            arg.signature(with_types=True, name=argname)
            for argname, arg in scalar_parameters
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

    @staticmethod
    def generate_kernel_boilerplate_post(kernel_stream, sdfg, state_id):
        kernel_stream.write("HLSLIB_DATAFLOW_FINALIZE();\n}\n", sdfg, state_id)

    def generate_host_function_body(self, sdfg, state, kernel_name, parameters,
                                    symbol_parameters, kernel_stream):

        # Just collect all variable names for calling the kernel function
        kernel_args = [
            p.signature(False, name=name) for is_output, name, p in parameters
        ]

        kernel_args += symbol_parameters.keys()

        kernel_function_name = kernel_name
        kernel_file_name = "{}.xclbin".format(kernel_name)
        host_function_name = "__dace_runkernel_{}".format(kernel_name)

        kernel_stream.write(
            """\
  auto kernel = program.MakeKernel({kernel_function_name}, "{kernel_function_name}", {kernel_args});
  const std::pair<double, double> elapsed = kernel.ExecuteTask();
  std::cout << "Kernel executed in " << elapsed.second << " seconds.\\n" << std::flush;
}}""".format(kernel_function_name=kernel_function_name,
             kernel_args=", ".join(kernel_args)), sdfg, sdfg.node_id(state))

    def generate_module(self, sdfg, state, name, subgraph, parameters,
                        symbol_parameters, module_stream, entry_stream,
                        host_stream):
        """Generates a module that will run as a dataflow function in the FPGA
           kernel."""

        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        # Treat scalars and symbols the same, assuming there are no scalar
        # outputs
        symbol_sigs = [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]
        symbol_names = symbol_parameters.keys()
        kernel_args_call = []
        kernel_args_module = []
        added = set()

        for is_output, pname, p in parameters:
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
                self._unrolled_pes.add(scope.map)
                kernel_args_call += ", ".join(scope.map.params)
                kernel_args_module += ["int " + p for p in scope.params]
                for p, r in zip(scope.map.params, scope.map.range):
                    if len(r) > 3:
                        raise dace.codegen.codegen.CodegenError(
                            "Strided unroll not supported")
                    entry_stream.write(
                        "for (size_t {param} = {begin}; {param} < {end}; "
                        "{param} += {increment}) {{\n#pragma HLS UNROLL".
                        format(
                            param=p, begin=r[0], end=r[1] + 1, increment=r[2]))
                    unrolled_loops += 1

        # Generate caller code in top-level function
        entry_stream.write(
            "HLSLIB_DATAFLOW_FUNCTION({}, {});".format(
                module_function_name, ", ".join(kernel_args_call)), sdfg,
            state_id)

        for _ in range(unrolled_loops):
            entry_stream.write("}")

        # ----------------------------------------------------------------------
        # Generate kernel code
        # ----------------------------------------------------------------------

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
            for out, argname, arg in parameters
            if isinstance(arg, dace.data.Array)
            and arg.storage == dace.dtypes.StorageType.FPGA_Global and not out
        }
        out_args = {
            argname
            for out, argname, arg in parameters
            if isinstance(arg, dace.data.Array)
            and arg.storage == dace.dtypes.StorageType.FPGA_Global and out
        }
        if len(in_args) > 0 or len(out_args) > 0:
            # Add ArrayInterface objects to wrap input and output pointers to
            # the same array
            module_body_stream.write("\n")
            interfaces_added = set()
            for _, argname, arg in parameters:
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

        self._dispatcher.defined_vars.exit_scope(subgraph)

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

        # generate library call
        reduction_cpp = "dace::Reduce<{}, {}, {}, {}<{}>>".format(
            dtype.ctype, vector_length_in, vector_length_out,
            REDUCTION_TYPE_TO_HLSLIB[reduction_type], dtype.ctype)

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

    def generate_kernel_internal(self, sdfg, state, kernel_name, subgraphs,
                                 kernel_stream, function_stream,
                                 callsite_stream):
        """Main entry function for generating a Xilinx kernel."""

        (global_data_parameters, top_level_local_data, subgraph_parameters,
         scalar_parameters, symbol_parameters,
         nested_global_transients) = self.make_parameters(
             sdfg, state, subgraphs)

        # Scalar parameters are never output
        sc_parameters = [(False, pname, param)
                         for pname, param in scalar_parameters]

        host_code_stream = CodeIOStream()

        # Generate host code
        self.generate_host_header(sdfg, kernel_name,
                                  global_data_parameters + sc_parameters,
                                  symbol_parameters, host_code_stream)
        self.generate_host_function_boilerplate(
            sdfg, state, kernel_name, global_data_parameters + sc_parameters,
            symbol_parameters, nested_global_transients, host_code_stream,
            function_stream, callsite_stream)
        self.generate_host_function_body(
            sdfg, state, kernel_name, global_data_parameters + sc_parameters,
            symbol_parameters, host_code_stream)
        # Store code to be passed to compilation phase
        self._host_codes.append((kernel_name, host_code_stream.getvalue()))

        # Now we write the device code
        module_stream = CodeIOStream()
        entry_stream = CodeIOStream()

        state_id = sdfg.node_id(state)

        self.generate_kernel_boilerplate_pre(
            sdfg, state_id, kernel_name, global_data_parameters,
            scalar_parameters, symbol_parameters, module_stream, entry_stream)

        # Emit allocations
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
                                               module_stream, entry_stream)
            self._dispatcher.dispatch_initialize(sdfg, state, state_id, node,
                                                 module_stream, entry_stream)

        self.generate_modules(sdfg, state, kernel_name, subgraphs,
                              subgraph_parameters, sc_parameters,
                              symbol_parameters, module_stream, entry_stream,
                              host_code_stream)

        kernel_stream.write(module_stream.getvalue())
        kernel_stream.write(entry_stream.getvalue())

        self.generate_kernel_boilerplate_post(kernel_stream, sdfg, state_id)

    def generate_host_header(self, sdfg, kernel_function_name, parameters,
                             symbol_parameters, host_code_stream):

        kernel_args = []

        seen = set()
        for is_output, name, arg in parameters:
            if isinstance(arg, dace.data.Array):
                kernel_args.append(
                    arg.signature(
                        with_types=True,
                        name=name + ("_out" if is_output else "_in")))
            else:
                if name in seen:
                    continue
                seen.add(name)
                kernel_args.append(arg.signature(with_types=True, name=name))

        kernel_args += [
            v.signature(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]

        host_code_stream.write(
            """\
// Signature of kernel function (with raw pointers) for argument matching
DACE_EXPORTED void {kernel_function_name}({kernel_args});\n\n""".format(
                kernel_function_name=kernel_function_name,
                kernel_args=", ".join(kernel_args)), sdfg)

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):

        # TODO: this is copy-pasta from the CPU-codegen, necessary to inject
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

        cpu.unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                            callsite_stream, self._cpu_codegen._locals,
                            self._cpu_codegen._ldepth,
                            self._cpu_codegen._toplevel_schedule)

        callsite_stream.write("////////////////////\n\n", sdfg, state_id, node)

        # Process outgoing memlets
        self._cpu_codegen.process_out_memlets(
            sdfg, state_id, node, state_dfg, self._dispatcher, callsite_stream,
            True, function_stream)

        for edge in state_dfg.out_edges(node):
            datadesc = sdfg.arrays[edge.data.data]
            if (isinstance(datadesc, dace.data.Array) and
                (datadesc.storage == dace.dtypes.StorageType.FPGA_Local
                 or datadesc.storage == dace.dtypes.StorageType.FPGA_Registers)
                    and edge.data.wcr is None):
                self.generate_no_dependence_post(
                    edge.src_conn, callsite_stream, sdfg, state_id, node)

        callsite_stream.write('}\n', sdfg, state_id, node)

        self._dispatcher.defined_vars.exit_scope(node)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node,
                                   dst_node, edge, callsite_stream):
        self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node,
                                      edge, None, callsite_stream)
