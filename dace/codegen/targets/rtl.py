from six import StringIO
import ast
import ctypes
import functools
import os
import sympy
import warnings

import dace
from dace.frontend import operations
from dace import registry, subsets, symbolic, dtypes, data as dt
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import ScopeSubgraphView, SDFG, SDFGState, scope_contains_scope, is_devicelevel_gpu, is_array_stream_view, has_dynamic_map_inputs, dynamic_map_inputs
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import (TargetCodeGenerator, IllegalCopy,
                                         make_absolute, DefinedType)
from dace.codegen.targets.cpp import (sym2cpp, unparse_cr, unparse_cr_split,
                                      cpp_array_expr, synchronize_streams,
                                      memlet_copy_to_absolute_strides,
                                      codeblock_to_cpp)
from dace.codegen import cppunparse
from dace.codegen.targets.cpu import CPUCodeGen

# class RTLCodeGen(CPUCodeGen):
@registry.autoregister_params(name='rtl')
class RTLCodeGen(TargetCodeGenerator):
    """ RTL (SystemVerilog) Code Generator """

    title = 'RTL'
    target_name = 'rtl'
    language = 'rtl'

    def __init__(self, frame_codegen, *args, **kwargs):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        # register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

    @staticmethod
    def cmake_options():
        """ Prepare CMake options. """
        # get flags from config
        verilator_flags = Config.get("compiler", "rtl", "verilator_flags")  # COVERAGE, TRACE
        # create options list
        options = [
            "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags)
        ]
        return options

    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
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