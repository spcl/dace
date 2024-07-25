# import
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, SDFGState, StateSubgraphView
import functools
import itertools
import warnings

from dace import data, dtypes, registry, memlet as mmlt, subsets, symbolic, Config
from dace.codegen import cppunparse, exceptions as cgx
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp, fpga
from dace.codegen.common import codeblock_to_cpp, sym2cpp, update_persistent_desc
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.frontend import operations
from dace.sdfg import nodes, utils as sdutils
from dace.sdfg import (ScopeSubgraphView, SDFG, scope_contains_scope, is_array_stream_view, NodeNotExpandedError,
                       dynamic_map_inputs)
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope
from dace.sdfg.validation import validate_memlet_data
from typing import TYPE_CHECKING, Optional, Tuple, Union
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.cpu import CPUCodeGen


@registry.autoregister_params(name='ipu')
class IPUCodeGen(TargetCodeGenerator):
    """ IPU(Graphcore) code generator. """
    target_name = 'ipu'
    title = 'IPU'
    language = 'cpp'

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: SDFG):
        self._codeobjects = []  # Holds any external files - src/cuda/xyz.cu, ...
        self._sdfg = sdfg
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_node_dispatcher(self)
        #self._cpu_codegen: CPUCodeGen = self._dispatcher.get_generic_node_dispatcher()
        
        

    # __dace_init_<TARGET> function is generated if True
    @property
    def has_initializer(self):
        return False

    # __dace_exit_<TARGET> function is generated if True
    @property
    def has_finalizer(self):
        return False

    @staticmethod
    def cmake_options():
        options = []

        linker_flags = Config.get("compiler", "ipu", "libs")
     
        if linker_flags:
            options.append(f'-DCMAKE_SHARED_LINKER_FLAGS="{linker_flags}"')


        return options

    # This will generate the src/cuda/xyz.cu files and folders using "codeObjects" class.
    # We don't need this now as we are mostly concerned about a single file codegen as of now.
    def get_generated_codeobjects(self):
        return self._codeobjects

    def generate_node(self, sdfg:SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node:nodes.Node, function_stream: CodeIOStream, callsite_stream:CodeIOStream):

        if isinstance(node, nodes.Map):
            callsite_stream.write(
                f'''
                Concurrency(Map/Consume)(omp loop)!
            '''
            , sdfg)
        elif isinstance(node, nodes.AccessNode):
            callsite_stream.write(
                f'''
                AccessNode(container=array/stream)!
            '''
            , sdfg)
        elif isinstance(node, nodes.CodeNode):
            callsite_stream.write(
                f'''
                CodeNode(Tasklet/nestedSDFG)!
            '''
            , sdfg)
        
