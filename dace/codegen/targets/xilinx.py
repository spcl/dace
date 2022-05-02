# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
from dace.sdfg.sdfg import SDFG
import itertools
import os
import re
import numpy as np
import ast
import dace
from dace import data as dt, registry, dtypes, subsets
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import nodes, utils
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.codegen import exceptions as cgx
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import make_absolute
from dace.codegen.targets import cpp, fpga
from typing import List, Union

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
        fpga_vendor = Config.get("compiler", "fpga", "vendor")
        if fpga_vendor.lower() != "xilinx":
            # Don't register this code generator
            return
        super().__init__(*args, **kwargs)
        # Used to pass memory bank assignments from kernel generation code to
        # where they are written to file
        self._bank_assignments = {}

        # Keep track of external streams: original_name -> mangled_name
        self._external_streams = dict()
        self._defined_external_streams = set()
        self._execution_mode = Config.get("compiler", "xilinx", "mode")
        self._decouple_array_interfaces = Config.get_bool("compiler", "xilinx", "decouple_array_interfaces")

    @staticmethod
    def cmake_options():
        host_flags = Config.get("compiler", "xilinx", "host_flags")
        synthesis_flags = Config.get("compiler", "xilinx", "synthesis_flags")
        build_flags = Config.get("compiler", "xilinx", "build_flags")
        mode = Config.get("compiler", "xilinx", "mode")
        target_platform = Config.get("compiler", "xilinx", "platform")
        enable_debugging = ("ON" if Config.get_bool("compiler", "xilinx", "enable_debugging") else "OFF")
        autobuild = ("ON" if Config.get_bool("compiler", "fpga", "autobuild_bitstreams") else "OFF")
        frequency = Config.get("compiler", "xilinx", "frequency").strip().strip('"').split(":")[0]
        options = [
            "-DDACE_XILINX_HOST_FLAGS=\"{}\"".format(host_flags),
            "-DDACE_XILINX_SYNTHESIS_FLAGS=\"{}\"".format(synthesis_flags),
            "-DDACE_XILINX_BUILD_FLAGS=\"{}\"".format(build_flags), "-DDACE_XILINX_MODE={}".format(mode),
            "-DDACE_XILINX_TARGET_PLATFORM=\"{}\"".format(target_platform),
            "-DDACE_XILINX_ENABLE_DEBUGGING={}".format(enable_debugging),
            "-DDACE_FPGA_AUTOBUILD_BITSTREAM={}".format(autobuild), f"-DDACE_XILINX_TARGET_CLOCK={frequency}"
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
            raise cgx.CodegenError("Unknown Xilinx execution mode: {}".format(execution_mode))

        set_env_vars = ""
        set_str = "dace::set_environment_variable(\"{}\", {});\n"
        unset_str = "dace::unset_environment_variable(\"{}\");\n"
        set_env_vars += (set_str.format("XCL_EMULATION_MODE", xcl_emulation_mode)
                         if xcl_emulation_mode is not None else unset_str.format("XCL_EMULATION_MODE"))
        set_env_vars += (set_str.format("XILINX_SDX", xilinx_sdx)
                         if xilinx_sdx is not None else unset_str.format("XILINX_SDX"))
        set_env_vars += set_str.format(
            "EMCONFIG_PATH",
            "DACE_BINARY_DIR") if execution_mode == 'hardware_emulation' else unset_str.format("EMCONFIG_PATH")

        host_code = CodeIOStream()
        host_code.write("""\
#include "dace/xilinx/host.h"
#include "dace/dace.h"
#include "dace/xilinx/stream.h"
""")
        if len(self._dispatcher.instrumentation) > 2:
            host_code.write("""\
#include "dace/perf/reporting.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
""")
        host_code.write("\n\n")

        self._frame.generate_fileheader(self._global_sdfg, host_code, 'xilinx_host')

        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma

        host_code.write("""
DACE_EXPORTED int __dace_init_xilinx({sdfg.name}_t *__state{signature}) {{
    {environment_variables}

    __state->fpga_context = new dace_fpga_context();
    __state->fpga_context->Get().MakeProgram({kernel_file_name});
    return 0;
}}

DACE_EXPORTED void __dace_exit_xilinx({sdfg.name}_t *__state) {{
    delete __state->fpga_context;
}}

{host_code}""".format(signature=params_comma,
                      sdfg=self._global_sdfg,
                      environment_variables=set_env_vars,
                      kernel_file_name=kernel_file_name,
                      host_code="".join([
                          "{separator}\n// Kernel: {kernel_name}"
                          "\n{separator}\n\n{code}\n\n".format(separator="/" * 79, kernel_name=name, code=code)
                          for (name, code) in self._host_codes
                      ])))

        host_code_obj = CodeObject(self._program_name,
                                   host_code.getvalue(),
                                   "cpp",
                                   XilinxCodeGen,
                                   "Xilinx",
                                   target_type="host")

        kernel_code_objs = [
            CodeObject(kernel_name, code, "cpp", XilinxCodeGen, "Xilinx", target_type="device")
            for (kernel_name, code) in self._kernel_codes
        ]

        # Memory bank and streaming interfaces connectivity configuration file
        link_cfg = CodeIOStream()
        self._other_codes["link.cfg"] = link_cfg
        link_cfg.write("[connectivity]")
        are_assigned = [v is not None for v in self._bank_assignments.values()]
        if any(are_assigned):
            if not all(are_assigned):
                raise RuntimeError("Some, but not all global memory arrays "
                                   "were assigned to memory banks: {}".format(self._bank_assignments))
            # Emit mapping from kernel memory interfaces to DRAM banks
            for (kernel_name, interface_name), (memory_type, memory_bank) in self._bank_assignments.items():
                link_cfg.write(f"sp={kernel_name}_1.m_axi_{interface_name}:{memory_type}[{memory_bank}]")
        # Emit mapping between inter-kernel streaming interfaces
        for _, (src, dst) in self._stream_connections.items():
            link_cfg.write(f"stream_connect={src}:{dst}")

        other_objs = []
        for name, code in self._other_codes.items():
            name = name.split(".")
            other_objs.append(
                CodeObject(name[0], code.getvalue(), ".".join(name[1:]), XilinxCodeGen, "Xilinx", target_type="device"))

        return [host_code_obj] + kernel_code_objs + other_objs

    def _internal_preprocess(self, sdfg: dace.SDFG):
        '''
        Vendor-specific SDFG Preprocessing
        '''

        if self._decouple_array_interfaces:
            # If array accesses are decoupled, preprocess inter state edge assignments:
            # - look at every interstate edge
            # - if any of them accesses an ArrayInterface (Global FPGA memory), qualify its name and replace it
            #       in the assignment string

            for graph in sdfg.all_sdfgs_recursive():
                for state in graph.states():
                    out_edges = graph.out_edges(state)
                    for e in out_edges:
                        if len(e.data.assignments) > 0:
                            replace_dict = dict()

                            for variable, value in e.data.assignments.items():
                                expr = ast.parse(value)
                                # walk in the expression, get all array names and check whether we need to qualify them
                                for node in ast.walk(expr):
                                    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                                        arr_name = node.value.id

                                        if arr_name not in replace_dict and arr_name in graph.arrays and graph.arrays[
                                                arr_name].storage == dace.dtypes.StorageType.FPGA_Global:
                                            repl = fpga.fpga_ptr(
                                                arr_name,
                                                graph.arrays[node.value.id],
                                                sdfg,
                                                None,
                                                False,
                                                None,
                                                None,
                                                True,
                                                decouple_array_interfaces=self._decouple_array_interfaces)
                                            replace_dict[arr_name] = repl

                            # Perform replacement and update graph.arrays to allow type inference
                            # on interstate edges
                            for k, v in replace_dict.items():
                                e.data.replace(k, v)
                                if v not in graph.arrays:
                                    # Note: this redundancy occurs only during codegen
                                    graph.arrays[v] = graph.arrays[k]

    def define_stream(self, dtype, buffer_size, var_name, array_size, function_stream, kernel_stream, sdfg):
        """
           Defines a stream
           :return: a tuple containing the type of the created variable, and boolean indicating
               whether this is a global variable or not
           """

        ctype = "dace::FIFO<{}, {}, {}>".format(dtype.base_type.ctype, cpp.sym2cpp(dtype.veclen),
                                                cpp.sym2cpp(buffer_size))

        array_size_cpp = cpp.sym2cpp(array_size)
        if array_size_cpp == "1":
            kernel_stream.write("{} {}(\"{}\");".format(ctype, var_name, var_name))
        else:
            kernel_stream.write("{} {}[{}];\n".format(ctype, var_name, array_size_cpp))
            kernel_stream.write("dace::SetNames({}, \"{}\", {});".format(var_name, var_name, array_size_cpp))

        # In Xilinx, streams are defined as local variables
        # Return value is used for adding to defined_vars in fpga.py
        return ctype, False

    def define_local_array(self, var_name, desc, array_size, function_stream, kernel_stream, sdfg, state_id, node):
        dtype = desc.dtype
        kernel_stream.write("{} {}[{}];\n".format(dtype.ctype, var_name, cpp.sym2cpp(array_size)))
        if desc.storage == dace.dtypes.StorageType.FPGA_Registers:
            kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} " "complete\n".format(var_name))
        elif desc.storage == dace.dtypes.StorageType.FPGA_Local:
            pass
        else:
            raise ValueError("Unsupported storage type: {}".format(desc.storage.name))
        self._dispatcher.defined_vars.add(var_name, DefinedType.Pointer, '%s *' % dtype.ctype)

    def define_shift_register(*args, **kwargs):
        raise NotImplementedError("Xilinx shift registers NYI")

    @staticmethod
    def make_vector_type(dtype, is_const):
        return "{}{}".format("const " if is_const else "", dtype.ctype)

    @staticmethod
    def make_kernel_argument(data: dt.Data,
                             var_name: str,
                             subset_info: Union[int, subsets.Subset],
                             sdfg: SDFG,
                             is_output: bool,
                             with_vectorization: bool,
                             interface_id: Union[int, List[int]] = None,
                             decouple_array_interfaces=False):
        if isinstance(data, dt.Array):
            var_name = fpga.fpga_ptr(var_name,
                                     data,
                                     sdfg,
                                     subset_info,
                                     is_output,
                                     None,
                                     None,
                                     True,
                                     interface_id,
                                     decouple_array_interfaces=decouple_array_interfaces)
            if with_vectorization:
                dtype = data.dtype
            else:
                dtype = data.dtype.base_type
            return "{} *{}".format(dtype.ctype, var_name)
        if isinstance(data, dt.Stream):
            ctype = "dace::FIFO<{}, {}, {}>".format(data.dtype.base_type.ctype, cpp.sym2cpp(data.dtype.veclen),
                                                    cpp.sym2cpp(data.buffer_size))
            if data.shape[0] == 1:
                return "{} &{}".format(ctype, var_name)
            else:
                return "{} {}[{}]".format(ctype, var_name, data.shape[0])
        else:
            return data.as_arg(with_types=True, name=var_name)

    def generate_unroll_loop_pre(self, kernel_stream, factor, sdfg, state_id, node):
        pass

    @staticmethod
    def generate_unroll_loop_post(kernel_stream, factor, sdfg, state_id, node):
        if factor is None:
            kernel_stream.write("#pragma HLS UNROLL", sdfg, state_id, node)
        else:
            kernel_stream.write("#pragma HLS UNROLL factor={}".format(factor), sdfg, state_id, node)

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

    def generate_nsdfg_header(self, sdfg, state, state_id, node, memlet_references, sdfg_label):
        # TODO: Use a single method for GPU kernels, FPGA modules, and NSDFGs
        arguments = [f'{atype} {aname}' for atype, aname, _ in memlet_references]
        arguments += [
            f'{node.sdfg.symbols[aname].as_arg(aname)}' for aname in sorted(node.symbol_mapping.keys())
            if aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        return f'void {sdfg_label}({arguments}) {{\n#pragma HLS INLINE'

    def write_and_resolve_expr(self, sdfg, memlet, nc, outname, inname, indices=None, dtype=None):
        """
        Emits a conflict resolution call from a memlet.
        """
        redtype = operations.detect_reduction_type(memlet.wcr, openmp=True)
        ptrname = cpp.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg, self._frame)
        defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
        if isinstance(indices, str):
            ptr = '%s + %s' % (cpp.cpp_ptr_expr(sdfg,
                                                memlet,
                                                defined_type,
                                                is_write=True,
                                                codegen=self._frame,
                                                decouple_array_interface=self._decouple_array_interfaces), indices)
        else:
            ptr = cpp.cpp_ptr_expr(sdfg,
                                   memlet,
                                   defined_type,
                                   indices=indices,
                                   is_write=True,
                                   codegen=self._frame,
                                   decouple_array_interface=self._decouple_array_interfaces)

        if isinstance(dtype, dtypes.pointer):
            dtype = dtype.base_type

        # Special call for detected reduction types
        if redtype != dtypes.ReductionType.Custom:
            if redtype == dace.dtypes.ReductionType.Sub:
                # write this as an addition
                credtype = "dace::ReductionType::Sum"
                is_sub = True
            else:
                credtype = "dace::ReductionType::" + str(redtype)[str(redtype).find(".") + 1:]
                is_sub = False

            if isinstance(dtype, dtypes.vector):
                return (f'dace::xilinx_wcr_fixed_vec<{credtype}, '
                        f'{dtype.vtype.ctype}, {dtype.veclen}>::reduce('
                        f'{ptr}, {"-" if is_sub else ""}{inname})')
            return (f'dace::xilinx_wcr_fixed<{credtype}, {dtype.ctype}>::reduce('
                    f'{ptr}, {"-" if is_sub else ""}{inname})')

        # General reduction
        raise NotImplementedError('General reductions not yet implemented')

    @staticmethod
    def make_read(defined_type, dtype, var_name, expr, index, is_pack, packing_factor):
        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if " " in expr:
                expr = "(" + expr + ")"
            read_expr = "{}.pop()".format(expr)
        elif defined_type == DefinedType.Scalar:
            read_expr = var_name
        else:
            if index is not None and index != "0":
                read_expr = "{} + {}".format(expr, index)
            else:
                read_expr = expr
        if is_pack:
            return "dace::Pack<{}, {}>({})".format(dtype.base_type.ctype, packing_factor, read_expr)
        else:
            return "dace::Read<{}, {}>({})".format(dtype.base_type.ctype, dtype.veclen, read_expr)

    def generate_converter(*args, **kwargs):
        pass  # Handled in C++

    @staticmethod
    def make_write(defined_type, dtype, var_name, write_expr, index, read_expr, wcr, is_unpack, packing_factor):
        if defined_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if defined_type == DefinedType.StreamArray:
                write_expr = "{}[{}]".format(write_expr, "0" if not index else index)
            if is_unpack:
                return "\n".join("{}.push({}[{}]);".format(write_expr, read_expr, i) for i in range(packing_factor))
            else:
                return "{}.push({});".format(write_expr, read_expr)
        else:
            if defined_type == DefinedType.Scalar:
                write_expr = var_name
            elif index and index != "0":
                write_expr = "{} + {}".format(write_expr, index)
            if is_unpack:
                return "dace::Unpack<{}, {}>({}, {});".format(dtype.base_type.ctype, packing_factor, read_expr,
                                                              write_expr)
            else:
                # TODO: Temporary hack because we don't have the output
                #       vector length.
                veclen = max(dtype.veclen, packing_factor)
                return "dace::Write<{}, {}>({}, {});".format(dtype.base_type.ctype, veclen, write_expr, read_expr)

    def make_shift_register_write(self, defined_type, dtype, var_name, write_expr, index, read_expr, wcr, is_unpack,
                                  packing_factor, sdfg):
        raise NotImplementedError("Xilinx shift registers NYI")

    @staticmethod
    def generate_no_dependence_pre(kernel_stream, sdfg, state_id, node, var_name=None):
        pass

    def generate_no_dependence_post(self,
                                    kernel_stream,
                                    sdfg: SDFG,
                                    state_id: int,
                                    node: nodes.Node,
                                    var_name: str,
                                    accessed_subset: Union[int, subsets.Subset] = None):
        '''
        Adds post loop pragma for ignoring loop carried dependencies on a given variable
        '''
        defined_type, _ = self._dispatcher.defined_vars.get(var_name)

        if var_name in sdfg.arrays:
            array = sdfg.arrays[var_name]
        else:
            array = None

        var_name = fpga.fpga_ptr(var_name,
                                 array,
                                 sdfg,
                                 accessed_subset,
                                 True,
                                 self._dispatcher,
                                 is_array_interface=(defined_type == DefinedType.ArrayInterface),
                                 decouple_array_interfaces=self._decouple_array_interfaces)
        kernel_stream.write("#pragma HLS DEPENDENCE variable={} false".format(var_name), sdfg, state_id, node)

    def generate_kernel_boilerplate_pre(self, sdfg, state_id, kernel_name, parameters, bank_assignments, module_stream,
                                        kernel_stream, external_streams):

        # Write header
        module_stream.write("""#include <dace/fpga_device.h>
#include <dace/math.h>
#include <dace/complex.h>""", sdfg)
        self._frame.generate_fileheader(sdfg, module_stream, 'xilinx_device')
        module_stream.write("\n", sdfg)

        argname_to_bank_assignment = {}
        # Build kernel signature
        kernel_args = []
        array_args = []
        for is_output, data_name, data, interface in parameters:
            is_assigned = data_name in bank_assignments and bank_assignments[data_name] is not None
            if is_assigned and isinstance(data, dt.Array):
                memory_bank = bank_assignments[data_name]
                lowest_bank_index, _ = fpga.get_multibank_ranges_from_subset(memory_bank[1], sdfg)

                for bank, interface_id in fpga.iterate_multibank_interface_ids(data, interface):
                    kernel_arg = self.make_kernel_argument(data,
                                                           data_name,
                                                           bank,
                                                           sdfg,
                                                           is_output,
                                                           True,
                                                           interface_id,
                                                           decouple_array_interfaces=self._decouple_array_interfaces)
                    if kernel_arg:
                        kernel_args.append(kernel_arg)
                        array_args.append((kernel_arg, data_name))
                        argname_to_bank_assignment[kernel_arg] = (memory_bank[0], lowest_bank_index + bank)
            else:
                kernel_arg = self.make_kernel_argument(data,
                                                       data_name,
                                                       None,
                                                       None,
                                                       is_output,
                                                       True,
                                                       interface,
                                                       decouple_array_interfaces=self._decouple_array_interfaces)
                if kernel_arg:
                    kernel_args.append(kernel_arg)
                    if isinstance(data, dt.Array):
                        array_args.append((kernel_arg, data_name))
                        argname_to_bank_assignment[kernel_arg] = None

        stream_args = []
        for is_output, data_name, data, interface in external_streams:
            kernel_arg = self.make_kernel_argument(data,
                                                   data_name,
                                                   None,
                                                   None,
                                                   is_output,
                                                   True,
                                                   interface,
                                                   decouple_array_interfaces=self._decouple_array_interfaces)

            if kernel_arg:
                stream_args.append(kernel_arg)

        if not self._decouple_array_interfaces:
            kernel_args = dtypes.deduplicate(kernel_args)

        # Write kernel signature
        kernel_stream.write("DACE_EXPORTED void {}({}) {{\n".format(kernel_name, ', '.join(kernel_args + stream_args)),
                            sdfg, state_id)

        # Insert interface pragmas
        num_mapped_args = 0
        if not self._decouple_array_interfaces:
            array_args = dtypes.deduplicate(array_args)

        for arg, data_name in array_args:
            var_name = re.findall(r"\w+", arg)[-1]
            if "*" in arg:
                interface_name = "gmem{}".format(num_mapped_args)
                kernel_stream.write(
                    "#pragma HLS INTERFACE m_axi port={} "
                    "offset=slave bundle={}".format(var_name, interface_name), sdfg, state_id)
                # Map this interface to the corresponding location
                # specification to be passed to the Xilinx compiler
                memory_bank = argname_to_bank_assignment[arg]
                self._bank_assignments[(kernel_name, interface_name)] = memory_bank
                num_mapped_args += 1

        for arg in kernel_args + ["return"]:
            var_name = re.findall(r"\w+", arg)[-1]
            kernel_stream.write("#pragma HLS INTERFACE s_axilite port={} bundle=control".format(var_name))

        for _, var_name, node, _ in external_streams:
            arr_len = dace.symbolic.evaluate(node.shape[0], sdfg.constants)
            if arr_len > 1:
                kernel_stream.write("#pragma HLS ARRAY_PARTITION variable={} dim=1 complete".format(var_name))
            kernel_stream.write("#pragma HLS INTERFACE axis port={}".format(var_name))

        # TODO: add special case if there's only one module for niceness
        kernel_stream.write("\n#pragma HLS DATAFLOW")
        kernel_stream.write("\nHLSLIB_DATAFLOW_INIT();")

    @staticmethod
    def generate_kernel_boilerplate_post(kernel_stream, sdfg, state_id):
        kernel_stream.write("HLSLIB_DATAFLOW_FINALIZE();\n}\n", sdfg, state_id)

    def generate_host_function_body(self, sdfg: dace.SDFG, state: dace.SDFGState, kernel_name: str, predecessors: list,
                                    parameters: list, rtl_tasklet_names: list, kernel_stream: CodeIOStream,
                                    instrumentation_stream: CodeIOStream):
        '''
        Generate the host-specific code for spawning and synchronizing the given kernel.
        :param sdfg:
        :param state:
        :param predecessors: list containing all the name of kernels that must be finished before starting this one
        :param parameters: list containing the kernel parameters (of all kernels in this state)
        :param rtl_tasklet_names
        :param kernel_stream: Device-specific code stream
        :param instrumentation_stream: Code for profiling kernel execution time.
        '''

        kernel_args = []
        for _, name, p, interface_ids in parameters:
            if isinstance(p, dt.Array):
                for bank, interface_id in fpga.iterate_multibank_interface_ids(p, interface_ids):
                    # Keep track of the interface_id (if any), while creating kernel arguments.
                    # In Xilinx we may have kernel argument with the same name but we want to keep all of them
                    # if they have different interface IDs (this could be the case if the same data is accessed
                    # from different PEs)

                    kernel_args.append(
                        (p.as_arg(False,
                                  name=fpga.fpga_ptr(name,
                                                     p,
                                                     sdfg,
                                                     bank,
                                                     decouple_array_interfaces=self._decouple_array_interfaces)),
                         interface_id))
            elif isinstance(p, dt.Stream) and name in self._defined_external_streams:
                if p.is_stream_array():
                    kernel_args.append((f" hlslib::ocl::SimulationOnly(&{p.as_arg(False, name=name)}[0])", 0))
                else:
                    kernel_args.append((f" hlslib::ocl::SimulationOnly({p.as_arg(False, name=name)})", 0))
            else:
                kernel_args.append((p.as_arg(False, name=name), 0))

        kernel_function_name = kernel_name
        kernel_file_name = "{}.xclbin".format(kernel_name)

        # Check if this kernel depends from other kernels
        needs_synch = len(predecessors) > 0

        if needs_synch:
            # Build a vector containing all the events associated with the kernels from which this one depends
            kernel_deps_name = f"deps_{kernel_name}"
            kernel_stream.write(f"std::vector<hlslib::ocl::Event> {kernel_deps_name};")
            for pred in predecessors:
                # concatenate events from predecessor kernel
                kernel_stream.write(f"{kernel_deps_name}.push_back({pred}_event);")
        if not self._decouple_array_interfaces:
            kernel_args = dtypes.deduplicate(kernel_args)
        # Launch HLS kernel, passing synchronization events (if any)
        kernel_stream.write(
            f"""auto {kernel_name}_kernel = program.MakeKernel({kernel_function_name}, "{kernel_function_name}", {", ".join(ka[0] for ka in kernel_args)});"""
        )

        kernel_stream.write(
            f"""\
  hlslib::ocl::Event {kernel_name}_event = {kernel_name}_kernel.ExecuteTaskAsync({f'{kernel_deps_name}.begin(), {kernel_deps_name}.end()' if needs_synch else ''});
  all_events.push_back({kernel_name}_event);""", sdfg, sdfg.node_id(state))
        if state.instrument == dtypes.InstrumentationType.FPGA:
            self.instrument_opencl_kernel(kernel_name, sdfg.node_id(state), sdfg.sdfg_id, instrumentation_stream)

        # Join RTL tasklets
        for name in rtl_tasklet_names:
            kernel_stream.write(f"kernel_{name}.wait();\n", sdfg, sdfg.node_id(state))

    def generate_module(self, sdfg, state, kernel_name, name, subgraph, parameters, module_stream, entry_stream,
                        host_stream, instrumentation_stream):
        """Generates a module that will run as a dataflow function in the FPGA
           kernel."""

        state_id = sdfg.node_id(state)
        dfg = sdfg.nodes()[state_id]

        kernel_args_call = []
        kernel_args_module = []
        for is_output, pname, p, interface_ids in parameters:
            if isinstance(p, dt.Array):
                for bank, interface_id in fpga.iterate_multibank_interface_ids(p, interface_ids):
                    arr_name = fpga.fpga_ptr(pname,
                                             p,
                                             sdfg,
                                             bank,
                                             is_output,
                                             is_array_interface=True,
                                             decouple_array_interfaces=self._decouple_array_interfaces)
                    # Add interface ID to called module, but not to the module
                    # arguments
                    argname = fpga.fpga_ptr(pname,
                                            p,
                                            sdfg,
                                            bank,
                                            is_output,
                                            is_array_interface=True,
                                            interface_id=interface_id,
                                            decouple_array_interfaces=self._decouple_array_interfaces)

                    kernel_args_call.append(argname)
                    dtype = p.dtype

                    if self._decouple_array_interfaces:
                        kernel_args_module.append("{} {}*{}".format(dtype.ctype, "const " if not is_output else "",
                                                                    arr_name))
                    else:
                        # in this case we don't know if this is accessed read-only or not
                        kernel_args_module.append("{} *{}".format(dtype.ctype, arr_name))

            else:
                if isinstance(p, dt.Stream):
                    # if this is an external stream, its name may have been mangled in the kernel
                    call_name = self._external_streams[pname] if pname in self._external_streams else pname
                    kernel_args_call.append(p.as_arg(with_types=False, name=call_name))
                    if p.is_stream_array():
                        kernel_args_module.append("dace::FIFO<{}, {}, {}> {}[{}]".format(
                            p.dtype.base_type.ctype, cpp.sym2cpp(p.veclen), cpp.sym2cpp(p.buffer_size), pname,
                            p.size_string()))
                    else:
                        kernel_args_module.append("dace::FIFO<{}, {}, {}> &{}".format(
                            p.dtype.base_type.ctype, cpp.sym2cpp(p.veclen), cpp.sym2cpp(p.buffer_size), pname))
                else:
                    kernel_args_call.append(p.as_arg(with_types=False, name=pname))
                    kernel_args_module.append(p.as_arg(with_types=True, name=pname))

        # Check if we are generating an RTL module, in which case only the
        # accesses to the streams should be handled
        rtl_tasklet = None
        for n in subgraph.nodes():
            if (isinstance(n, dace.nodes.Tasklet) and n.language == dace.dtypes.Language.SystemVerilog):
                rtl_tasklet = n
                break
        if rtl_tasklet:
            entry_stream.write(f'// [RTL] HLSLIB_DATAFLOW_FUNCTION({name}, {", ".join(kernel_args_call)});')
            module_stream.write(f'// [RTL] void {name}({", ".join(kernel_args_module)});\n\n')

            rtl_name = self.rtl_tasklet_name(rtl_tasklet, state, sdfg)

            # _i in names are due to vitis
            for node in subgraph.source_nodes():
                if isinstance(sdfg.arrays[node.data], dt.Stream):
                    # TODO multiple readers accessing a single stream should fail
                    dst = subgraph.out_edges(node)[0].dst
                    if isinstance(dst, dace.nodes.MapEntry) and dst.map.unroll:
                        unrolled_map_range = dace.symbolic.evaluate(dst.map.range[0][1] + 1, sdfg.constants)
                    else:
                        unrolled_map_range = 1
                    if unrolled_map_range > 1:
                        elements_to_add = [f'{node.data}_{i}' for i in range(unrolled_map_range)]
                    else:
                        elements_to_add = [node.data]
                    for i in range(unrolled_map_range):
                        elem = elements_to_add[i]
                        postfix = f'_{i}' if unrolled_map_range > 1 else ''
                        if elem not in self._stream_connections:
                            self._stream_connections[elem] = [None, None]
                        for edge in subgraph.out_edges(node):
                            rtl_dst = state.memlet_path(edge)[-1].dst_conn
                            val = '{}_top_1.s_axis_{}{}'.format(rtl_name, rtl_dst, postfix)
                            self._stream_connections[elem][1] = val

            for node in subgraph.sink_nodes():
                if isinstance(sdfg.arrays[node.data], dt.Stream):
                    # TODO multiple writers accessing a single stream should fail
                    src = subgraph.in_edges(node)[0].src
                    if (isinstance(src, dace.nodes.MapExit) and src.map.unroll):
                        unrolled_map_range = dace.symbolic.evaluate(src.map.range[0][1] + 1, sdfg.constants)
                    else:
                        unrolled_map_range = 1
                    if unrolled_map_range > 1:
                        elements_to_add = [f'{node.data}_{i}' for i in range(unrolled_map_range)]
                    else:
                        elements_to_add = [node.data]
                    for i in range(unrolled_map_range):
                        elem = elements_to_add[i]
                        postfix = f'_{i}' if unrolled_map_range > 1 else ''
                        if elem not in self._stream_connections:
                            self._stream_connections[elem] = [None, None]
                        for edge in state.in_edges(node):
                            rtl_src = subgraph.memlet_path(edge)[0].src_conn
                            self._stream_connections[elem][0] = '{}_top_1.m_axis_{}{}'.format(
                                rtl_name, rtl_src, postfix)

            # Make the dispatcher trigger generation of the RTL module, but
            # ignore the generated code, as the RTL codegen will generate the
            # appropriate files.
            ignore_stream = CodeIOStream()
            self._dispatcher.dispatch_subgraph(sdfg,
                                               subgraph,
                                               state_id,
                                               ignore_stream,
                                               ignore_stream,
                                               skip_entry_node=False)

            # Launch the kernel from the host code
            rtl_name = self.rtl_tasklet_name(rtl_tasklet, state, sdfg)

            # kernel arguments
            host_stream.write(
                f"  auto kernel_{rtl_name} = program.MakeKernel(\"{rtl_name}_top\"{', '.join([''] + [name for _, name, p, _ in parameters if not (isinstance(p, dt.Stream))])}).ExecuteTaskAsync();",
                sdfg, state_id, rtl_tasklet)
            if state.instrument == dtypes.InstrumentationType.FPGA:
                self.instrument_opencl_kernel(rtl_name, state_id, sdfg.sdfg_id, instrumentation_stream)

            return

        # create a unique module name to prevent name clashes
        module_function_name = f"module_{name}_{sdfg.sdfg_id}"

        # Unrolling processing elements: if there first scope of the subgraph
        # is an unrolled map, generate a processing element for each iteration
        scope_children = subgraph.scope_children()
        top_scopes = [n for n in scope_children[None] if isinstance(n, dace.sdfg.nodes.EntryNode)]
        unrolled_loops = 0
        if len(top_scopes) == 1:
            scope = top_scopes[0]
            if scope.unroll:
                self._unrolled_pes.add(scope.map)
                kernel_args_call += ", ".join(scope.map.params)
                kernel_args_module += ["int " + p for p in scope.params]
                for p, r in zip(scope.map.params, scope.map.range):
                    if len(r) > 3:
                        raise cgx.CodegenError("Strided unroll not supported")
                    entry_stream.write("for (size_t {param} = {begin}; {param} < {end}; "
                                       "{param} += {increment}) {{\n#pragma HLS UNROLL".format(param=p,
                                                                                               begin=r[0],
                                                                                               end=r[1] + 1,
                                                                                               increment=r[2]))
                    unrolled_loops += 1

        # Generate caller code in top-level function
        if not self._decouple_array_interfaces:
            kernel_args_call = dtypes.deduplicate(kernel_args_call)
        entry_stream.write(
            "HLSLIB_DATAFLOW_FUNCTION({}, {});".format(module_function_name, ", ".join(kernel_args_call)), sdfg,
            state_id)

        for _ in range(unrolled_loops):
            entry_stream.write("}")

        # ----------------------------------------------------------------------
        # Generate kernel code
        # ----------------------------------------------------------------------

        self._dispatcher.defined_vars.enter_scope(subgraph)

        module_body_stream = CodeIOStream()

        if not self._decouple_array_interfaces:
            kernel_args_module = dtypes.deduplicate(kernel_args_module)

        module_body_stream.write("void {}({}) {{".format(module_function_name, ", ".join(kernel_args_module)), sdfg,
                                 state_id)

        # Register the array interface as a naked pointer for use inside the
        # FPGA kernel
        interfaces_added = set()

        for is_output, argname, arg, interface_id in parameters:
            for bank, _ in fpga.iterate_multibank_interface_ids(arg, interface_id):
                if isinstance(arg, dt.Stream) and argname in self._external_streams:
                    # This is an external stream being passed to the module
                    # Add this to defined vars
                    if not self._dispatcher.defined_vars.has(argname):
                        self._dispatcher.defined_vars.add(argname, DefinedType.Stream, arg.ctype)
                    continue

                if (not (isinstance(arg, dt.Array) and arg.storage == dace.dtypes.StorageType.FPGA_Global)):
                    continue
                ctype = dtypes.pointer(arg.dtype).ctype
                ptr_name = fpga.fpga_ptr(argname,
                                         arg,
                                         sdfg,
                                         bank,
                                         is_output,
                                         None,
                                         is_array_interface=True,
                                         decouple_array_interfaces=self._decouple_array_interfaces)
                if not is_output and self._decouple_array_interfaces:
                    ctype = f"const {ctype}"

                if self._decouple_array_interfaces:
                    self._dispatcher.defined_vars.add(ptr_name, DefinedType.Pointer, ctype)
                if argname in interfaces_added:
                    continue
                interfaces_added.add(argname)
                self._dispatcher.defined_vars.add(argname, DefinedType.ArrayInterface, ctype, allow_shadowing=True)
        module_body_stream.write("\n")

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

        self._dispatcher.defined_vars.exit_scope(subgraph)

    def rtl_tasklet_name(self, node: nodes.RTLTasklet, state, sdfg):
        return "{}_{}_{}_{}".format(node.name, sdfg.sdfg_id, sdfg.node_id(state), state.node_id(node))

    def generate_kernel_internal(self, sdfg: dace.SDFG, state: dace.SDFGState, kernel_name: str, predecessors: list,
                                 subgraphs: list, kernel_stream: CodeIOStream, state_host_header_stream: CodeIOStream,
                                 state_host_body_stream: CodeIOStream, instrumentation_stream: CodeIOStream,
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream, state_parameters: list):
        '''
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
        '''

        (global_data_parameters, top_level_local_data, subgraph_parameters, nested_global_transients, bank_assignments,
         external_streams) = self.make_parameters(sdfg, state, subgraphs)

        state_parameters.extend(global_data_parameters)

        # We need to pass external streams as parameters to module
        # (unless they are already there. This could be case of inter-PE intra-kernel streams)
        # TODO It doesn't break RTL, but the streams are passed to sub kernels that don't need the streams, in turn relying on Vitis to optimize them away again.
        for k, v in subgraph_parameters.items():
            for stream_is_out, stream_name, stream_desc, stream_iid in external_streams:
                for is_output, data_name, desc, interface_id in v:
                    if data_name == stream_name and stream_desc == desc:
                        break
                else:
                    v.append((stream_is_out, stream_name, stream_desc, stream_iid))

        # Xilinx does not like external streams name with leading underscores to be used as port names
        # We remove them, and we check that they are not defined anywhere else
        for es in external_streams:

            new_name = es[1].strip("_")
            self._external_streams[es[1]] = new_name

            if new_name != es[1]:
                clashes = [param for param in global_data_parameters if param[1] == new_name]
                clashes.extend([param for param in top_level_local_data if param[1] == new_name])
                clashes.extend([param for param in subgraph_parameters.values() if param[1] == new_name])
                clashes.extend([param for param in nested_global_transients if param[1] == new_name])
                if len(clashes) > 0:
                    raise cgx.CodegenError(
                        f"External stream sanitized name {new_name} clashes with other paramters: {clashes}")
                else:
                    external_streams.remove(es)
                    external_streams.append((es[0], new_name, es[2], es[3]))

        # Detect RTL tasklets, which will be launched as individual kernels
        rtl_tasklet_names = [
            self.rtl_tasklet_name(nd, state, sdfg) for nd in state.nodes() if isinstance(nd, nodes.RTLTasklet)
        ]

        # Generate host code
        self.generate_host_header(sdfg, kernel_name, global_data_parameters + external_streams,
                                  state_host_header_stream)
        self.generate_host_function_boilerplate(sdfg, state, nested_global_transients, state_host_body_stream)

        # Now we write the device code
        module_stream = CodeIOStream()
        entry_stream = CodeIOStream()

        state_id = sdfg.node_id(state)

        self.generate_kernel_boilerplate_pre(sdfg, state_id, kernel_name, global_data_parameters, bank_assignments,
                                             module_stream, entry_stream, external_streams)

        # Emit allocations
        for node in top_level_local_data:
            self._dispatcher.dispatch_allocate(sdfg, state, state_id, node, node.desc(sdfg), module_stream,
                                               entry_stream)

        for is_output, name, node, _ in external_streams:
            num_streams = dace.symbolic.evaluate(node.shape[0], sdfg.constants)
            self._dispatcher.defined_vars.add_global(name, DefinedType.Stream, node.ctype)
            key = 0 if is_output else 1
            val = '{}_1.{}'.format(kernel_name, name)

            # Define here external streams
            if name not in self._defined_external_streams:
                self.define_stream(node.dtype, node.buffer_size, name, node.total_size, None, state_host_body_stream,
                                   sdfg)
                self._defined_external_streams.add(name)

            if num_streams > 1:
                streams = [f'{name}_{i}' for i in range(num_streams)]
            else:  # _num should not be appended, when there is only one kernel
                streams = [name]
                if name not in self._defined_external_streams:
                    self.define_stream(node.dtype, node.buffer_size, name, node.total_size, None,
                                       state_host_body_stream, sdfg)
                    self._defined_external_streams.add(name)
            for stream in streams:
                if stream not in self._stream_connections:
                    self._stream_connections[stream] = [None, None]
                val = '{}_1.{}'.format(kernel_name, stream)
                self._stream_connections[stream][key] = val

        self.generate_modules(sdfg, state, kernel_name, subgraphs, subgraph_parameters, module_stream, entry_stream,
                              state_host_body_stream, instrumentation_stream)

        self.generate_host_function_body(sdfg, state, kernel_name, predecessors,
                                         global_data_parameters + external_streams, rtl_tasklet_names,
                                         state_host_body_stream, instrumentation_stream)

        # Store code to be passed to compilation phase
        # self._host_codes.append((kernel_name, host_code_stream.getvalue()))
        kernel_stream.write(module_stream.getvalue())
        kernel_stream.write(entry_stream.getvalue())

        self.generate_kernel_boilerplate_post(kernel_stream, sdfg, state_id)

    def generate_host_header(self, sdfg, kernel_function_name, parameters, host_code_stream):

        kernel_args = []
        for is_output, name, arg, interface_ids in parameters:
            if isinstance(arg, dt.Stream):

                if arg.is_stream_array():
                    kernel_args.append("dace::FIFO<{}, {}, {}> {}[{}]".format(arg.dtype.base_type.ctype,
                                                                              cpp.sym2cpp(arg.veclen),
                                                                              cpp.sym2cpp(arg.buffer_size), name,
                                                                              arg.size_string()))
                else:
                    kernel_args.append("dace::FIFO<{}, {}, {}> &{}".format(arg.dtype.base_type.ctype,
                                                                           cpp.sym2cpp(arg.veclen),
                                                                           cpp.sym2cpp(arg.buffer_size), name))
            elif isinstance(arg, dt.Array):
                for bank, interface_id in fpga.iterate_multibank_interface_ids(arg, interface_ids):
                    argname = fpga.fpga_ptr(name,
                                            arg,
                                            sdfg,
                                            bank,
                                            is_output,
                                            None,
                                            None,
                                            True,
                                            interface_id,
                                            decouple_array_interfaces=self._decouple_array_interfaces)
                    kernel_args.append(arg.as_arg(with_types=True, name=argname))
            else:
                kernel_args.append(arg.as_arg(with_types=True, name=name))
        if not self._decouple_array_interfaces:
            kernel_args = dtypes.deduplicate(kernel_args)
        host_code_stream.write(
            """\
// Signature of kernel function (with raw pointers) for argument matching
DACE_EXPORTED void {kernel_function_name}({kernel_args});\n\n""".format(kernel_function_name=kernel_function_name,
                                                                        kernel_args=", ".join(kernel_args)), sdfg)

    def generate_memlet_definition(self, sdfg, dfg, state_id, src_node, dst_node, edge, callsite_stream):
        memlet = edge.data
        ptrname = cpp.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg, self._frame)

        if (self._dispatcher.defined_vars.get(ptrname)[0] == DefinedType.FPGA_ShiftRegister):
            raise NotImplementedError("Shift register for Xilinx NYI")
        else:
            self._cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge, None, callsite_stream)

    def allocate_view(self, sdfg: dace.SDFG, dfg: dace.SDFGState, state_id: int, node: dace.nodes.AccessNode,
                      global_stream: CodeIOStream, declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        return self._cpu_codegen.allocate_view(sdfg, dfg, state_id, node, global_stream, declaration_stream,
                                               allocation_stream)

    def generate_nsdfg_arguments(self, sdfg, dfg, state, node):
        # Connectors that are both input and output share the same name, unless
        # they are pointers to global memory in device code, in which case they
        # are split into explicit input and output interfaces
        inout = set(node.in_connectors.keys() & node.out_connectors.keys())

        memlet_references = []
        for _, _, _, vconn, in_memlet in sorted(state.in_edges(node), key=lambda e: e.dst_conn or ""):
            if in_memlet.data is None:
                continue
            if not self._decouple_array_interfaces and vconn in inout:
                # Only one interface will be generated
                continue
            ptrname = cpp.ptr(in_memlet.data, sdfg.arrays[in_memlet.data], sdfg, self._frame)
            is_memory_interface = (self._dispatcher.defined_vars.get(ptrname, 1)[0] == DefinedType.ArrayInterface)
            desc = sdfg.arrays[in_memlet.data]
            if is_memory_interface:
                for bank in fpga.iterate_distributed_subset(sdfg.arrays[in_memlet.data], in_memlet, False, sdfg):
                    interface_name = fpga.fpga_ptr(vconn,
                                                   sdfg.arrays[in_memlet.data],
                                                   sdfg,
                                                   bank,
                                                   False,
                                                   is_array_interface=True,
                                                   decouple_array_interfaces=self._decouple_array_interfaces)
                    passed_memlet = copy.deepcopy(in_memlet)
                    passed_memlet.subset = fpga.modify_distributed_subset(passed_memlet.subset, bank)
                    interface_ref = cpp.emit_memlet_reference(self._dispatcher,
                                                              sdfg,
                                                              passed_memlet,
                                                              interface_name,
                                                              conntype=node.in_connectors[vconn],
                                                              is_write=False,
                                                              decouple_array_interfaces=self._decouple_array_interfaces)
                    memlet_references.append(interface_ref)

            if vconn in inout:
                continue
            if fpga.is_multibank_array_with_distributed_index(sdfg.arrays[in_memlet.data]):
                passed_memlet = copy.deepcopy(in_memlet)
                passed_memlet.subset = fpga.modify_distributed_subset(passed_memlet.subset,
                                                                      0)  # dummy so it works for HBM
            else:
                passed_memlet = in_memlet
            ref = cpp.emit_memlet_reference(self._dispatcher,
                                            sdfg,
                                            passed_memlet,
                                            vconn,
                                            conntype=node.in_connectors[vconn],
                                            is_write=False,
                                            decouple_array_interfaces=self._decouple_array_interfaces)
            if not is_memory_interface:
                memlet_references.append(ref)

        for _, uconn, _, _, out_memlet in sorted(state.out_edges(node), key=lambda e: e.src_conn or ""):
            if out_memlet.data is None:
                continue
            if fpga.is_multibank_array_with_distributed_index(sdfg.arrays[out_memlet.data]):
                passed_memlet = copy.deepcopy(out_memlet)
                passed_memlet.subset = fpga.modify_distributed_subset(passed_memlet.subset,
                                                                      0)  # dummy so it works for HBM
            else:
                passed_memlet = out_memlet
            desc = sdfg.arrays[out_memlet.data]
            ref = cpp.emit_memlet_reference(self._dispatcher,
                                            sdfg,
                                            passed_memlet,
                                            uconn,
                                            conntype=node.out_connectors[uconn],
                                            is_write=True,
                                            decouple_array_interfaces=self._decouple_array_interfaces)
            ptrname = cpp.ptr(out_memlet.data, sdfg.arrays[out_memlet.data], sdfg, self._frame)
            is_memory_interface = (self._dispatcher.defined_vars.get(ptrname, 1)[0] == DefinedType.ArrayInterface)

            if is_memory_interface:
                for bank in fpga.iterate_distributed_subset(sdfg.arrays[out_memlet.data], out_memlet, True, sdfg):
                    interface_name = fpga.fpga_ptr(uconn,
                                                   sdfg.arrays[out_memlet.data],
                                                   sdfg,
                                                   bank,
                                                   True,
                                                   is_array_interface=True,
                                                   decouple_array_interfaces=self._decouple_array_interfaces)
                    passed_memlet = copy.deepcopy(out_memlet)
                    passed_memlet.subset = fpga.modify_distributed_subset(passed_memlet.subset, bank)
                    interface_ref = cpp.emit_memlet_reference(self._dispatcher,
                                                              sdfg,
                                                              passed_memlet,
                                                              interface_name,
                                                              conntype=node.out_connectors[uconn],
                                                              is_write=True,
                                                              decouple_array_interfaces=self._decouple_array_interfaces)
                    memlet_references.append(interface_ref)
            else:
                memlet_references.append(ref)

        return memlet_references

    def unparse_tasklet(self, *args, **kwargs):
        # Pass this object for callbacks into the Xilinx codegen
        cpp.unparse_tasklet(*args, codegen=self, **kwargs)

    def make_ptr_assignment(self, src_expr, src_dtype, dst_expr, dst_dtype):
        """
        Write source to destination, where the source is a scalar, and the
        destination is a pointer.
        :return: String of C++ performing the write.
        """
        return self.make_write(DefinedType.Pointer, dst_dtype, None, "&" + dst_expr, None, src_expr, None,
                               dst_dtype.veclen < src_dtype.veclen, src_dtype.veclen)
