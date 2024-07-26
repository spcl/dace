# import
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from dace.codegen.targets.ipu import IPUCodeGen
    from dace.codegen.targets.cpp import CPUCodeGen


@registry.autoregister_params(name='ipu')
class IPUCodeGen(TargetCodeGenerator):
    """ IPU(Graphcore) code generator. """
    target_name = 'ipu'
    title = 'IPU'
    language = 'cpp'

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._sdfg = sdfg
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._global_sdfg = sdfg
        
        # Register dispatchers
        # self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        
        # Register additional dispatchers
        # self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)
        # self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.MPI, self)
        # self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.Sequential, self)
        self._dispatcher.register_node_dispatcher(self)
        
    def state_dispatch_predicate(self, sdfg, state):
        return True

    # __dace_init_<TARGET> function is generated if True
    @property
    def has_initializer(self):
        return True

    # __dace_exit_<TARGET> function is generated if True
    @property
    def has_finalizer(self):
        return True

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
        fileheader = CodeIOStream()
        sdfg = self._global_sdfg
        
        # Adds <poplar.h>
        self._frame.generate_fileheader(self._global_sdfg, fileheader, 'poplar')    
        
        # cuda/mpi seemed to be using this follow 
        params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma
        codelet_file_code = """
// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
// Copied from tut3_vertices from Poplar SDK tutorials

#include <poplar/Vertex.hpp>

class SumVertex : public poplar::Vertex {
    public:
    // Fields
    poplar::Input<poplar::Vector<float>> in;
    poplar::Output<float> out;

    // Compute function
    bool compute() {
        *out = 0;
        for (const auto &v : in) {
        *out += v;
        }
        return true;
    }
};
"""
        
        codeobj = CodeObject(
            name=sdfg.name + '_codelets', 
            code=codelet_file_code,
            language='cpp', 
            target=IPUCodeGen, 
            title='IPU',
            target_type='ipu',
            linkable=False)
        
        # Fill in the list
        return [codeobj]

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
        
    # def generate_state(self, 
            #         sdfg:SDFG, 
            #         cfg: ControlFlowRegion, 
            #         state: SDFGState, 
            #         function_stream: CodeIOStream, 
            #         callsite_stream:CodeIOStream,
            #         generate_state_footer:bool = True):
            
            # callsite_stream.write(
            #     f'''
            #     State(CFG/Loops/Conditionals(if else, for, ...))
            # '''
            # , sdfg)

    # def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: StateSubgraphView, state_id: int,
    #                    function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
    #     # Take care of map header
    #     assert len(dfg_scope.source_nodes()) == 1
    #     map_header: nodes.MapEntry = dfg_scope.source_nodes()[0]

    #     function_stream.write('extern int __dace_comm_size, __dace_comm_rank;', cfg, state_id, map_header)

    #     # Add extra opening brace (dynamic map ranges, closed in MapExit
    #     # generator)
    #     callsite_stream.write('{', cfg, state_id, map_header)

    #     if len(map_header.map.params) > 1:
    #         raise NotImplementedError('Multi-dimensional MPI maps are not supported')

    #     state = cfg.state(state_id)
    #     symtypes = map_header.new_symbols(sdfg, state, state.symbols_defined_at(map_header))

    #     for var, r in zip(map_header.map.params, map_header.map.range):
    #         begin, end, skip = r

    #         callsite_stream.write('{\n', cfg, state_id, map_header)
    #         callsite_stream.write(
    #             '%s %s = %s + __dace_comm_rank * (%s);\n' %
    #             (symtypes[var], var, cppunparse.pyexpr2cpp(symbolic.symstr(begin, cpp_mode=True)),
    #              cppunparse.pyexpr2cpp(symbolic.symstr(skip, cpp_mode=True))), cfg, state_id, map_header)

    #     self._frame.allocate_arrays_in_scope(sdfg, cfg, map_header, function_stream, callsite_stream)

    #     self._dispatcher.dispatch_subgraph(sdfg,
    #                                        cfg,
    #                                        dfg_scope,
    #                                        state_id,
    #                                        function_stream,
    #                                        callsite_stream,
    #                                        skip_entry_node=True)
