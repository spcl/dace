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


# @registry.autoregister_params(name='ipu')
# class IPUCodeGen(TargetCodeGenerator):
@registry.autoregister_params(name='loopy')
class MyCustomLoop(TargetCodeGenerator):
    """ IPU(Graphcore) code generator. """
    target_name = 'loopy'
    title = 'LOOPY'
    language = 'cpp'

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: SDFG):
        self._codeobjects = []  # Holds any external files - src/cuda/xyz.cu, ...
        self._sdfg = sdfg
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        # self._dispatcher.register_node_dispatcher(self)
        # self._dispatcher.register_state_dispatcher(self)
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.LoopyLoop, self)

        # self._cpu_codegen: CPUCodeGen = self._dispatcher.get_generic_node_dispatcher()
        
        

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
    
    # def generate_node(self, sdfg:SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, node:nodes.Node, function_stream: CodeIOStream, callsite_stream:CodeIOStream):
    #     callsite_stream.write(
    #         f'''
    #         Node!
    #     '''
    #     , sdfg)
    
    # def generate_state(self, sdfg: SDFG, state: SDFGState, function_stream: CodeIOStream, callsite_stream: CodeIOStream, generate_state_footer: bool) -> None:
    #     callsite_stream.write(
    #         f'''
    #         State!
    #     '''
    #     , sdfg)

  # A scope dispatcher will trigger a method called generate_scope whenever 
    # an SDFG has a scope with that schedule
    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        # The parameters here are:
        # sdfg: The SDFG we are currently generating.
        # scope: The subgraph of the state containing only the scope (map contents)
        #        we want to generate the code for.
        # state_id: The state in the SDFG the subgraph is taken from (i.e., 
        #           `sdfg.node(state_id)` is the same as `scope.graph`)
        # function_stream: A cursor to the global code (which can be used to define
        #                  functions, hence the name).
        # callsite_stream: A cursor to the current location in the code, most of
        #                  the code is generated here.
        
        # We can get the map entry node from the scope graph
        entry_node = scope.source_nodes()[0]
        
        # First, generate an opening brace (for instrumentation and dynamic map ranges)
        callsite_stream.write('{', sdfg, state_id, entry_node)
        
        ################################################################
        # Generate specific code: We will generate a reversed loop with a 
        # comment for each dimension of the map. For the sake of simplicity,
        # dynamic map ranges are not supported.
        
        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            # We use the sym2cpp function from the cpp support functions
            # to convert symbolic expressions to proper C++
            begin, end, stride = (sym2cpp(r) for r in rng)
            
            # Every write is optionally (but recommended to be) tagged with
            # 1-3 extra arguments, serving as line information to match
            # SDFG, state, and graph nodes/edges to written code.
            callsite_stream.write(f'''// Loopy-loop {param}
            for (auto {param} = {end}; {param} >= {begin}; {param} -= {stride}) {{''',
                                  sdfg, state_id, entry_node
            )
        
            # NOTE: CodeIOStream will automatically take care of indentation for us.
         
        # Now that the loops have been defined, use the dispatcher to invoke any
        # code generator (including this one) that is registered to deal with
        # the internal nodes in the subgraph. We skip the MapEntry node.
        self._dispatcher.dispatch_subgraph(sdfg, cfg, scope, state_id,
                                          function_stream, callsite_stream,
                                          skip_entry_node=True, skip_exit_node=True)
        
        # NOTE: Since skip_exit_node above is set to False, closing braces will
        #       be automatically generated
        # Change schedule
        # for node, _ in sdfg.all_nodes_recursive():
        #     if isinstance(node, dtypes.nodes.MapEntry):
        #         node.schedule = dtypes.ScheduleType.LoopyLoop
