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
        self._generated_nodes = set()
        self.calling_codegen = self
        
        # Register dispatchers
        # self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        ipu_storage = [dtypes.StorageType.Register]
        # Register additional dispatchers
        # self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.Sequential, self)
        self._dispatcher.register_node_dispatcher(self)
        self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)
        self._dispatcher.register_array_dispatcher(ipu_storage, self)
        # Register IPU copies (all internal pairs)
        for src_storage, dst_storage in itertools.product(ipu_storage, ipu_storage):
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)
        
        
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
        
        if Config.get("compiler", "ipu", "libs"):
            options.append('-DCMAKE_SHARED_LINKER_FLAGS="{}"'.format(Config.get("compiler", "ipu", "libs")))

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
            linkable=False)
        
        # Fill in the list
        return [codeobj]

############################################################################################################
#   IPU specific node/state generation
############################################################################################################
    # from cpu.py
    def generate_node(self, sdfg:SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node:nodes.Node, function_stream: CodeIOStream, callsite_stream:CodeIOStream):
        
        self._dispatcher.dispatch_allocate(sdfg, cfg, dfg, state_id, node, node.desc(sdfg), function_stream, callsite_stream)

        if isinstance(node, nodes.NestedSDFG):
            # Dynamically obtain node generator according to class name
            try:
                gen = getattr(self, "_generate_" + type(node).__name__) 
            except AttributeError:
                if isinstance(node, nodes.LibraryNode):
                    raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
                raise
            # _generate_Tasklet() example

            gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            # Mark node as "generated"
            self._generated_nodes.add(node)
            # self._locals.clear_scope(self._ldepth + 1)
            
    def generate_state(self, 
                    sdfg:SDFG, 
                    cfg: ControlFlowRegion, 
                    state: SDFGState, 
                    function_stream: CodeIOStream, 
                    callsite_stream:CodeIOStream,
                    generate_state_footer:bool = True):

        self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream, generate_state_footer=False)

    def generate_scope(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       dfg_scope: ScopeSubgraphView,
                       state_id: int,
                       function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream) -> None:
        # Get the first entry node of Map
        entry_node = dfg_scope.source_nodes()[0]

        # function_stream.write('extern int __dace_comm_size, __dace_comm_rank;', cfg, state_id, entry_node)
        callsite_stream.write('{', cfg, state_id, entry_node)

        # cpp.presynchronize_streams(sdfg, cfg, dfg_scope, state_id, entry_node, callsite_stream)   #TODO: add some other function of own.
        # Should we ?
        self.generate_node(sdfg, cfg, dfg_scope, state_id, entry_node, function_stream, callsite_stream)
        # generated nested subgraphs
        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def declare_array(self,
                      sdfg: SDFG,
                      cfg: ControlFlowRegion,
                      dfg: StateSubgraphView,
                      state_id: int,
                      node: nodes.Node,
                      nodedesc: data.Data,
                      function_stream: CodeIOStream,
                      declaration_stream: CodeIOStream) -> None:
        print("IN DECLARE_ARRAY")
        fsymbols = self._frame.symbols_and_constants(sdfg)
        # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
        # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
        # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
        # `nodedesc` is a View and `dfg` is None.
        if dfg and not sdutils.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError("The declare_array method should only be used for variables "
                                      "that must have their declaration and allocation separate.")

        name = node.root_data
        ptrname = cpp.ptr(name, nodedesc, sdfg, self._frame)

        if nodedesc.transient is False:
            return

        # Check if array is already declared
        if self._dispatcher.declared_arrays.has(ptrname):
            return

        # Compute array size
        arrsize = nodedesc.total_size
        if not isinstance(nodedesc.dtype, dtypes.opaque):
            arrsize_bytes = arrsize * nodedesc.dtype.bytes

        if (nodedesc.storage == dtypes.StorageType.Register):
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            declaration_stream.write(f'{nodedesc.dtype.ctype} *{name} = nullptr;\n', cfg, state_id, node)
            #Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
            declaration_stream.write(f'{nodedesc.dtype.ctype} {name}_const = graph.addConstant<{nodedesc.dtype.ctype}>({nodedesc.dtype.ctype.capitalize}, {arrsize}, {nodedesc.ctype}({nodedesc.dtype.ctype}));\n', cfg, state_id, node)
            self._dispatcher.declared_arrays.add(name, DefinedType.Pointer, ctypedef)
            return
        else:
            raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))


#### Helpers
    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label, state_struct=True):
        # prepend = []
        # if state_struct:
        #     prepend = ['__state']
        # fsyms = node.sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True)
        # args = ', '.join(prepend + [argval for _, _, argval in memlet_references] + [
        #     cpp.sym2cpp(symval) for symname, symval in sorted(node.symbol_mapping.items())
        #     if symname in fsyms and symname not in sdfg.constants
        # ])
        # return f'{sdfg_label}({args});'
        args = ''
        return f'{sdfg_label}({args});' #TODO: add args later
    
#### Node Generators(What node to generate) - callback from generate_node()
    def _generate_NestedSDFG(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: ScopeSubgraphView,
        state_id: int,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        state_dfg = cfg.nodes()[state_id]
         # Emit nested SDFG as a separate function
        nested_stream = CodeIOStream()
        nested_global_stream = CodeIOStream()

        # unique name generation of function
        sdfg_label = "%s_%d_%d_%d" % (node.sdfg.name, sdfg.cfg_id, state_id, dfg.node_id(node))
        
                    # Generate function call
        codegen = self.calling_codegen
        memlet_references = None    # TODO: add memlet references later                    
        callsite_stream.write(codegen.generate_nsdfg_call(sdfg, cfg, state_dfg, node, memlet_references,
                                                            sdfg_label),
                                  cfg, state_id, node)
        # callsite_stream.write(sdfg_label, cfg, state_id, node)



    # def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: StateSubgraphView, state_id: int,
    #                    function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:

    #     function_stream.write('extern int __dace_comm_size, __dace_comm_rank;', cfg, state_id, map_header)


    #     # Add extra opening brace (dynamic map ranges, closed in MapExit
    #     # generator)
    #     callsite_stream.write('{', cfg, state_id, map_header)

    #     if len(map_header.map.params) > 1:
    #         raise NotImplementedError('Multi-dimensional MPI maps are not supported')

    #     state = cfg.state(state_id)
    #     symtypes = map_header.new_symbols(sdfg, state, state.symbols_defined_at(map_header))


    #$$$$ First dace::copy()
    #     for var, r in zip(map_header.map.params, map_header.map.range):
    #         begin, end, skip = r

    #         callsite_stream.write('{\n', cfg, state_id, map_header)
    #         callsite_stream.write(
    #             '%s %s = %s + __dace_comm_rank * (%s);\n' %
    #             (symtypes[var], var, cppunparse.pyexpr2cpp(symbolic.symstr(begin, cpp_mode=True)),
    #              cppunparse.pyexpr2cpp(symbolic.symstr(skip, cpp_mode=True))), cfg, state_id, map_header)

    #     self._frame.allocate_arrays_in_scope(sdfg, cfg, map_header, function_stream, callsite_stream)
    # subgraphs_scope_call

