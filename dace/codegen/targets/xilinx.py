# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import itertools
import os
import re
import numpy as np

import dace
from dace import registry, dtypes
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import make_absolute, DefinedType
from dace.codegen.targets import cpp, fpga

REDUCTION_TYPE_TO_HLSLIB = {
    dace.dtypes.ReductionType.Min: "hlslib::op::Min",
    dace.dtypes.ReductionType.Max: "hlslib::op::Max",
    dace.dtypes.ReductionType.Sum: "hlslib::op::Sum",
    dace.dtypes.ReductionType.Product: "hlslib::op::Product",
    dace.dtypes.ReductionType.Logical_And: "hlslib::op::And",
}


@registry.autoregister_params(name='xilinx')
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
        # {(kernel name, interface name): (memory type, memory bank)}
        self._interface_assignments = {}

    @staticmethod
    def cmake_options():
        host_flags = Config.get("compiler", "xilinx", "host_flags")
        synthesis_flags = Config.get("compiler", "xilinx", "synthesis_flags")
        build_flags = Config.get("compiler", "xilinx", "build_flags")
        mode = Config.get("compiler", "xilinx", "mode")
        target_platform = Config.get("compiler", "xilinx", "platform")
        enable_debugging = ("ON" if Config.get_bool(
            "compiler", "xilinx", "enable_debugging") else "OFF")
        autobuild = ("ON" if Config.get_bool("compiler", "autobuild_bitstreams")
                     else "OFF")
        options = [
            "-DDACE_XILINX_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_XILINX_SYNTHESIS_FLAGS=\"{}\"".format(synthesis_flags),
            "-DDACE_XILINX_BUILD_FLAGS=\"{}\"".format(build_flags),
            "-DDACE_XILINX_MODE={}".format(mode),
            "-DDACE_XILINX_TARGET_PLATFORM=\"{}\"".format(target_platform),
            "-DDACE_XILINX_ENABLE_DEBUGGING={}".format(enable_debugging),
            "-DDACE_FPGA_AUTOBUILD_BITSTREAM={}".format(autobuild)
        ]
        # Override Vitis/SDx/SDAccel installation directory
        if Config.get("compiler", "xilinx", "path"):
            options.append("-DVITIS_ROOT_DIR=\"{}\"".format(
                Config.get("compiler", "xilinx", "path").replace("\\", "/")))
        return options

    def get_generated_codeobjects(self):

        execution_mode = Config.get("compiler", "xilinx", "mode")

        kernel_file_name = "DACE_BINARY_DIR \"/{}".format(self._program_name)
        if execution_mode == "software_emulation":
            kernel_file_name += "_sw_emu.xclbin\""
            xcl_emulation_mode = "\"sw_emu\""
            xilinx_sdx = "DACE_VITIS_DIR"
        elif execution_mode == "hardware_emulation":
            kernel_file_name += "_hw_emu.xclbin\""
            xcl_emulation_mode = "\"hw_emu\""
            xilinx_sdx = "DACE_VITIS_DIR"
        elif execution_mode == "hardware" or execution_mode == "simulation":
            kernel_file_name += "_hw.xclbin\""
            xcl_emulation_mode = None
            xilinx_sdx = None
        else:
            raise dace.codegen.codegen.CodegenError(
                "Unknown Xilinx execution mode: {}".format(execution_mode))

        set_env_vars = ""
        set_str = "dace::set_environment_variable(\"{}\", {});\n"
        unset_str = "dace::unset_environment_variable(\"{}\");\n"
        set_env_vars += (set_str.format("XCL_EMULATION_MODE",
                                        xcl_emulation_mode)
                         if xcl_emulation_mode is not None else
                         unset_str.format("XCL_EMULATION_MODE"))
        set_env_vars += (set_str.format("XILINX_SDX", xilinx_sdx) if xilinx_sdx
                         is not None else unset_str.format("XILINX_SDX"))

        host_code = CodeIOStream()
        host_code.write("""\
#include "dace/xilinx/host.h"
#include "dace/dace.h"
#include <iostream>\n\n""")

        self._frame.generate_fileheader(self._global_sdfg, host_code)

        host_code.write("""
dace::fpga::Context *dace::fpga::_context;

DACE_EXPORTED int __dace_init_xilinx({signature}) {{
    {environment_variables}
    dace::fpga::_context = new dace::fpga::Context();
    dace::fpga::_context->Get().MakeProgram({kernel_file_name});
    return 0;
}}

DACE_EXPORTED void __dace_exit_xilinx({signature}) {{
    delete dace::fpga::_context;
}}

{host_code}""".format(signature=self._global_sdfg.signature(),
                      environment_variables=set_env_vars,
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
                                   XilinxCodeGen,
                                   "Xilinx",
                                   target_type="host")

        kernel_code_objs = [
            CodeObject(kernel_name,
                       code,
                       "cpp",
                       XilinxCodeGen,
                       "Xilinx",
                       target_type="device")
            for (kernel_name, code) in self._kernel_codes
        ]

        # Configuration file with interface assignments
        are_assigned = [
            v is not None for v in self._interface_assignments.values()
        ]
        bank_assignment_code = []
        if any(are_assigned):
            if not all(are_assigned):
                raise RuntimeError("Some, but not all global memory arrays "
                                   "were assigned to memory banks: {}".format(
                                       self._interface_assignments))
            are_assigned = True
        else:
            are_assigned = False
        for name, _ in self._host_codes:
            # Only iterate over assignments if any exist
            if are_assigned:
                for (kernel_name, interface_name), (
                        memory_type,
                        memory_bank) in self._interface_assignments.items():
                    if kernel_name != name:
                        continue
                    bank_assignment_code.append("{},{},{}".format(
                        interface_name, memory_type.name, memory_bank))
            # Create file even if there are no assignments
            kernel_code_objs.append(
                CodeObject("{}_memory_interfaces".format(name),
                           "\n".join(bank_assignment_code),
                           "csv",
                           XilinxCodeGen,
                           "Xilinx",
                           target_type="device"))

        return [host_code_obj] + kernel_code_objs

    @staticmethod
    def define_stream(dtype, buffer_size, var_name, array_size, function_stream,
                      kernel_stream):
        ctype = "dace::FIFO<{}, {}, {}>".format(dtype.base_type.ctype,
                                                dtype.veclen, buffer_size)
        if cpp.sym2cpp(array_size) == "1":
            kernel_stream.write("{} {}(\"{}\");".format(ctype, var_name,
                                                        var_name))
        else:
            kernel_stream.write("{} {}[{}];\n".format(ctype, var_name,
                                                      cpp.sym2cpp(array_size)))
            kernel_stream.write("dace::SetNames({}, \"{}\", {});".format(
                var_name, var_name, cpp.sym2cpp(array_size)))

        # Return value is used for adding to defined_vars in fpga.py
        return ctype

    def define_local_array(self, var_name, desc, array_size, function_stream,
                           kernel_stream, sdfg, state_id, node):
        dtype = desc.dtype
        kernel_stream.write("{} {}[{}];\n".format(dtype.ctype, var_name,
                                                  cpp.sym2cpp(array_size)))
        if desc.storage == dace.dtypes.StorageType.FPGA_Registers:
            kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} "
                                "complete\n".format(var_name))
        elif desc.storage == dace.dtypes.StorageType.FPGA_Local:
            if len(desc.shape) > 1:
                kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} "
                                    "block factor={}\n".format(
                                        var_name, desc.shape[-2]))
        else:
            raise ValueError("Unsupported storage type: {}".format(
                desc.storage.name))
        self._dispatcher.defined_vars.add(var_name, DefinedType.Pointer,
                                          '%s *' % dtype.ctype)

    def define_shift_register(*args, **kwargs):
        raise NotImplementedError("Xilinx shift registers NYI")

    @staticmethod
    def make_vector_type(dtype, is_const):
        return "{}{}".format("const " if is_const else "",
                             dtype.base_type.ctype, dtype.veclen)

    @staticmethod
    def make_kernel_argument(data, var_name, is_output, with_vectorization):
        if isinstance(data, dace.data.Array):
            var_name += "_" + ("out" if is_output else "in")
            if with_vectorization:
                dtype = data.dtype
            else:
                dtype = data.dtype.base_type
            return "{} *{}".format(dtype.ctype, var_name)
        else:
            return data.as_arg(with_types=True, name=var_name)

    def generate_unroll_loop_pre(self, kernel_stream, factor, sdfg, state_id,
                                 node):
        pass

    @staticmethod
    def generate_unroll_loop_post(kernel_stream, factor, sdfg, state_id, node):
        if factor is None:
            kernel_stream.write("#pragma HLS UNROLL", sdfg, state_id, node)
        else:
            kernel_stream.write("#pragma HLS UNROLL factor={}".format(factor),
                                sdfg, state_id, node)

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

    def generate_nsdfg_header(self, sdfg, state, node, memlet_references,
                              sdfg_label):
        # TODO: Use a single method for GPU kernels, FPGA modules, and NSDFGs
        arguments = [
            f'{atype} {aname}' for atype, aname, _ in memlet_references
        ]
        arguments += [
            f'{node.sdfg.symbols[aname].as_arg(aname)}'
            for aname in sorted(node.symbol_mapping.keys())
            if aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        return f'void {sdfg_label}({arguments}) {{\n#pragma HLS INLINE'

    def write_and_resolve_expr(self,
                               sdfg,
                               memlet,
                               nc,
                               outname,
                               inname,
                               indices=None,
                               dtype=None):
        """
        Emits a conflict resolution call from a memlet.
        """
        redtype = operations.detect_reduction_type(memlet.wcr)
        if isinstance(indices, str):
            ptr = '%s + %s' % (cpp.cpp_ptr_expr(sdfg, memlet), indices)
        else:
            ptr = cpp.cpp_ptr_expr(sdfg, memlet, indices=indices)

        if isinstance(dtype, dtypes.pointer):
            dtype = dtype.base_type

        # Special call for detected reduction types
        if redtype != dtypes.ReductionType.Custom:
            credtype = "dace::ReductionType::" + str(
                redtype)[str(redtype).find(".") + 1:]
            if isinstance(dtype, dtypes.vector):
                return (f'dace::xilinx_wcr_fixed_vec<{credtype}, '
                        f'{dtype.vtype.ctype}, {dtype.veclen}>::reduce('
                        f'{ptr}, {inname})')
            return (
                f'dace::xilinx_wcr_fixed<{credtype}, {dtype.ctype}>::reduce('
                f'{ptr}, {inname})')

        # General reduction
        raise NotImplementedError('General reductions not yet implemented')

    @staticmethod
    def make_read(defined_type, dtype, var_name, expr, index, is_pack,
                  packing_factor):
        if defined_type == DefinedType.Stream:
            read_expr = "{}.pop()".format(expr)
        elif defined_type == DefinedType.StreamArray:
            if " " in expr:
                expr = "(" + expr + ")"
            read_expr = "{}[{}].pop()".format(expr, index)
        elif defined_type == DefinedType.Scalar:
            read_expr = var_name
        else:
            if index is not None and index != "0":
                read_expr = "{} + {}".format(expr, index)
            else:
                read_expr = expr
        if is_pack:
            return "dace::Pack<{}, {}>({})".format(dtype.base_type.ctype,
                                                   packing_factor, read_expr)
        else:
            return "dace::Read<{}, {}>({})".format(dtype.base_type.ctype,
                                                   dtype.veclen, read_expr)

    def generate_converter(*args, **kwargs):
        pass  # Handled in C++

    @staticmethod
    def make_write(defined_type, dtype, var_name, write_expr, index, read_expr,
                   wcr, is_unpack, packing_factor):
        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if defined_type == DefinedType.StreamArray:
                write_expr = "{}[{}]".format(write_expr,
                                             "0" if not index else index)
            if is_unpack:
                return "\n".join(
                    "{}.push({}[{}]);".format(write_expr, read_expr, i)
                    for i in range(packing_factor))
            else:
                return "{}.push({});".format(write_expr, read_expr)
        else:
            if defined_type == DefinedType.Scalar:
                write_expr = var_name
            elif index and index != "0":
                write_expr = "{} + {}".format(write_expr, index)
            if is_unpack:
                return "dace::Unpack<{}, {}>({}, {});".format(
                    dtype.base_type.ctype, packing_factor, read_expr,
                    write_expr)
            else:
                return "dace::Write<{}, {}>({}, {});".format(
                    dtype.base_type.ctype, dtype.veclen, write_expr, read_expr)

    def make_shift_register_write(self, defined_type, dtype, var_name,
                                  write_expr, index, read_expr, wcr, is_unpack,
                                  packing_factor):
        raise NotImplementedError("Xilinx shift registers NYI")

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
            v.as_arg(with_types=True, name=k)
            for k, v in symbol_parameters.items()
        ]
        arrays = list(sorted(global_data_parameters, key=lambda t: t[1]))
        scalars = scalar_parameters + list(symbol_parameters.items())
        scalars = list(sorted(scalars, key=lambda t: t[0]))

        # Build kernel signature
        array_args = []
        for is_output, dataname, data in arrays:
            kernel_arg = self.make_kernel_argument(data, dataname, is_output,
                                                   True)
            if kernel_arg:
                array_args.append(kernel_arg)
        kernel_args = array_args + [
            v.as_arg(with_types=True, name=k) for k, v in scalars
        ]

        kernel_args = dace.dtypes.deduplicate(kernel_args)

        # Write kernel signature
        kernel_stream.write(
            "DACE_EXPORTED void {}({}) {{\n".format(kernel_name,
                                                    ', '.join(kernel_args)),
            sdfg, state_id)

        # Insert interface pragmas
        num_mapped_args = 0
        for arg, (_, dataname, _) in zip(array_args, arrays):
            var_name = re.findall("\w+", arg)[-1]
            if "*" in arg:
                interface_name = "gmem{}".format(num_mapped_args)
                kernel_stream.write(
                    "#pragma HLS INTERFACE m_axi port={} "
                    "offset=slave bundle={}".format(var_name, interface_name),
                    sdfg, state_id)
                # Map this interface to the corresponding location
                # specification to be passed to the Xilinx compiler
                assignment = self._bank_assignments[(dataname, sdfg)]
                if assignment is not None:
                    mem_type, mem_bank = assignment
                    self._interface_assignments[(kernel_name,
                                                 interface_name)] = (mem_type,
                                                                     mem_bank)
                else:
                    self._interface_assignments[(kernel_name,
                                                 interface_name)] = None
                num_mapped_args += 1

        for arg in kernel_args + ["return"]:
            var_name = re.findall("\w+", arg)[-1]
            kernel_stream.write(
                "#pragma HLS INTERFACE s_axilite port={} bundle=control".format(
                    var_name))

        # TODO: add special case if there's only one module for niceness
        kernel_stream.write("\n#pragma HLS DATAFLOW")
        kernel_stream.write("\nHLSLIB_DATAFLOW_INIT();")

    @staticmethod
    def generate_kernel_boilerplate_post(kernel_stream, sdfg, state_id):
        kernel_stream.write("HLSLIB_DATAFLOW_FINALIZE();\n}\n", sdfg, state_id)

    def generate_host_function_body(self, sdfg, state, kernel_name, parameters,
                                    symbol_parameters, kernel_stream):

        # Just collect all variable names for calling the kernel function
        added = set()
        arrays = list(
            sorted([
                p for p in parameters if not isinstance(p[2], dace.data.Scalar)
            ],
                   key=lambda t: t[1]))
        scalars = [p for p in parameters if isinstance(p[2], dace.data.Scalar)]
        scalars += ((False, k, v) for k, v in symbol_parameters.items())
        scalars = dace.dtypes.deduplicate(sorted(scalars, key=lambda t: t[1]))
        kernel_args = []
        for _, name, p in itertools.chain(arrays, scalars):
            if not isinstance(p, dace.data.Array) and name in added:
                continue
            added.add(name)
            kernel_args.append(p.as_arg(False, name=name))

        kernel_function_name = kernel_name
        kernel_file_name = "{}.xclbin".format(kernel_name)

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

        kernel_args_call = []
        kernel_args_module = []
        added = set()

        parameters = list(sorted(parameters, key=lambda t: t[1]))
        arrays = [
            p for p in parameters if not isinstance(p[2], dace.data.Scalar)
        ]
        scalars = [p for p in parameters if isinstance(p[2], dace.data.Scalar)]
        scalars += ((False, k, v) for k, v in symbol_parameters.items())
        scalars = dace.dtypes.deduplicate(sorted(scalars, key=lambda t: t[1]))
        for is_output, pname, p in itertools.chain(parameters, scalars):
            if isinstance(p, dace.data.Array):
                arr_name = "{}_{}".format(pname, "out" if is_output else "in")
                kernel_args_call.append(arr_name)
                dtype = p.dtype
                kernel_args_module.append("{} {}*{}".format(
                    dtype.ctype, "const " if not is_output else "", arr_name))
            else:
                # Don't make duplicate arguments for other types than arrays
                if pname in added:
                    continue
                added.add(pname)
                if isinstance(p, dace.data.Stream):
                    kernel_args_call.append(
                        p.as_arg(with_types=False, name=pname))
                    if p.is_stream_array():
                        kernel_args_module.append(
                            "dace::FIFO<{}, {}, {}> {}[{}]".format(
                                p.dtype.base_type.ctype, p.veclen,
                                p.buffer_size, pname, p.size_string()))
                    else:
                        kernel_args_module.append(
                            "dace::FIFO<{}, {}, {}> &{}".format(
                                p.dtype.base_type.ctype, p.veclen,
                                p.buffer_size, pname))
                else:
                    kernel_args_call.append(
                        p.as_arg(with_types=False, name=pname))
                    kernel_args_module.append(
                        p.as_arg(with_types=True, name=pname))
        module_function_name = "module_" + name
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
                self._unrolled_pes.add(scope.map)
                kernel_args_call += ", ".join(scope.map.params)
                kernel_args_module += ["int " + p for p in scope.params]
                for p, r in zip(scope.map.params, scope.map.range):
                    if len(r) > 3:
                        raise dace.codegen.codegen.CodegenError(
                            "Strided unroll not supported")
                    entry_stream.write(
                        "for (size_t {param} = {begin}; {param} < {end}; "
                        "{param} += {increment}) {{\n#pragma HLS UNROLL".format(
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
                ctype = "dace::ArrayInterface<{}>".format(arg.dtype.ctype)
                module_body_stream.write("{} {}({}, {});".format(
                    ctype, argname, in_ptr, out_ptr))
                self._dispatcher.defined_vars.add(argname,
                                                  DefinedType.ArrayInterface,
                                                  ctype,
                                                  allow_shadowing=True)
            module_body_stream.write("\n")

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

        self._dispatcher.defined_vars.exit_scope(subgraph)

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
        self.generate_host_function_body(sdfg, state, kernel_name,
                                         global_data_parameters + sc_parameters,
                                         symbol_parameters, host_code_stream)
        # Store code to be passed to compilation phase
        self._host_codes.append((kernel_name, host_code_stream.getvalue()))

        # Now we write the device code
        module_stream = CodeIOStream()
        entry_stream = CodeIOStream()

        state_id = sdfg.node_id(state)

        self.generate_kernel_boilerplate_pre(sdfg, state_id, kernel_name,
                                             global_data_parameters,
                                             scalar_parameters,
                                             symbol_parameters, module_stream,
                                             entry_stream)

        # Emit allocations
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node,
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

        arrays = [
            p for p in parameters if not isinstance(p[2], dace.data.Scalar)
        ]
        arrays = list(sorted(arrays, key=lambda t: t[1]))
        scalars = [p for p in parameters if isinstance(p[2], dace.data.Scalar)]
        scalars += ((False, k, v) for k, v in symbol_parameters.items())
        scalars = list(sorted(scalars, key=lambda t: t[1]))

        kernel_args = []

        seen = set()
        for is_output, name, arg in itertools.chain(arrays, scalars):
            if isinstance(arg, dace.data.Array):
                kernel_args.append(
                    arg.as_arg(with_types=True,
                               name=name + ("_out" if is_output else "_in")))
            else:
                if name in seen:
                    continue
                seen.add(name)
                kernel_args.append(arg.as_arg(with_types=True, name=name))

        host_code_stream.write(
            """\
// Signature of kernel function (with raw pointers) for argument matching
DACE_EXPORTED void {kernel_function_name}({kernel_args});\n\n""".format(
                kernel_function_name=kernel_function_name,
                kernel_args=", ".join(kernel_args)), sdfg)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node,
                                   dst_node, edge, callsite_stream):
        memlet = edge.data
        if (self._dispatcher.defined_vars.get(
                memlet.data)[0] == DefinedType.FPGA_ShiftRegister):
            raise NotImplementedError("Shift register for Xilinx NYI")
        else:
            self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node,
                                          dst_node, edge, None, callsite_stream)

    def unparse_tasklet(self, *args, **kwargs):
        # Pass this object for callbacks into the Xilinx codegen
        cpp.unparse_tasklet(*args, codegen=self, **kwargs)
