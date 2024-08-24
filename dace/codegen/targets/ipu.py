# import
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from io import StringIO
from typing import TYPE_CHECKING, Optional, Tuple, Union
from copy import deepcopy
from dace import (data, dtypes, registry, memlet as mmlt, subsets, symbolic, Config)
from dace import dtypes, memlet as mm
from dace import Memlet
from dace.codegen import cppunparse, exceptions as cgx

from dace.codegen.prettycode import CodeIOStream
import dace.codegen.targets
from dace.codegen.targets import cpp, fpga
from dace.codegen.targets.cpu import CPUCodeGen
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.common import codeblock_to_cpp, sym2cpp, update_persistent_desc
from dace.codegen.targets.target import IllegalCopy, TargetCodeGenerator, make_absolute
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.frontend import operations
from dace.sdfg import (ScopeSubgraphView, SDFG, scope_contains_scope, is_array_stream_view, NodeNotExpandedError,
                       dynamic_map_inputs, nodes, utils as sdutils)
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope
from dace.sdfg.state import ControlFlowRegion, SDFGState, StateSubgraphView
from dace.sdfg import graph, state, find_input_arraynode, find_output_arraynode
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from dace.sdfg.validation import validate_memlet_data
from dace.sdfg.graph import MultiConnectorEdge
from dace.codegen.targets.sve import util as util
import copy
import functools
import itertools
import warnings


@registry.autoregister_params(name='ipu')
class IPUCodeGen(TargetCodeGenerator):
    """ IPU(Graphcore) code generator. """
    target_name = 'ipu'
    title = 'IPU'
    language = 'cpp'

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        print("in IPUCodeGen")
        self.has_generated_header = False
        self.frame = frame_codegen
        self.dispatcher = frame_codegen._dispatcher
        self.cpu_codegen: dace.codegen.targets.CPUCodeGen = self.dispatcher.get_generic_node_dispatcher()
        self._locals = cppunparse.CPPLocals()
        # Scope depth (for defining locals)
        self._ldepth = 0
        # Keep nested SDFG schedule when descending into it
        self._toplevel_schedule = None
        

        # self.dispatcher.register_array_dispatcher(dtypes.StorageType.IPU_Tile_Local, self)
            
        # Storage
        # ipu_storage = [dtypes.StorageType.IPU_Memory]        
        gpu_storage = [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned, dtypes.StorageType.IPU_Memory]
       
        self.dispatcher.register_array_dispatcher(gpu_storage, self)   # allocate_array/deallocate_array
        for storage in gpu_storage:
            for other_storage in gpu_storage:
                self.dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self.dispatcher.register_copy_dispatcher(other_storage, storage, None, self)
        
                
        # # Dispatchers
        # self.dispatcher.register_map_dispatcher(dace.ScheduleType.IPU_Map, self)
        # self.dispatcher.register_node_dispatcher(self, self.is_ipu_map_scope)
        # self.dispatcher.register_node_dispatcher(self, self.is_node_tasklet)
        # self.dispatcher.register_copy_dispatcher(dtypes.StorageType.Register, dtypes.StorageType.IPU_Tile_Local, None, func=self)
        # self._dispatcher.register_map_dispatcher(dace.ScheduleType.IPU, self)
        # self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)


    def get_generated_codeobjects(self):
        res = super().get_generated_codeobjects()
        return res    

    # __dace_init_<TARGET> function
    @property
    def has_initializer(self):
        return False

    # __dace_exit_<TARGET> function
    @property
    def has_finalizer(self):
        return False

    @staticmethod
    def cmake_options():
        options = []
        # if Config.get("compiler", "ipu", "libs"):
        #     options.append('-DCMAKE_SHARED_LINKER_FLAGS="{}"'.format(Config.get("compiler", "ipu", "libs")))
        return options
    
    def is_node_tasklet(self, sdfg, state, node):
        if isinstance(node, nodes.Tasklet):
            return True
        return False
    
    """    if hasattr(node, 'schedule'):  # NOTE: Works on nodes and scopes(NestedSDFG, Consume, Map, LibraryNode)
            if node.schedule == dtypes.ScheduleType.Sequential:
                return True
        return False
        """
############################################################################################################
#   IPU specific node/state generation
############################################################################################################
    # def copy_memory(
    #     self,
    #     sdfg: SDFG,
    #     cfg: ControlFlowRegion,
    #     dfg: StateSubgraphView,
    #     state_id: int,
    #     src_node: Union[nodes.Tasklet, nodes.AccessNode],
    #     dst_node: Union[nodes.Tasklet, nodes.AccessNode],
    #     edge: MultiConnectorEdge,
    #     function_stream: CodeIOStream,
    #     callsite_stream: CodeIOStream,
    # ) -> None:  
    #     return self.cpu_codegen.copy_memory(sdfg, cfg, dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream)
    #     return super().copy_memory(sdfg, dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream)
    
    # def declare_array(self, sdfg: SDFG, cfg: state.ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
    #                   nodedesc: data.Data, global_stream: CodeIOStream, declaration_stream: CodeIOStream) -> None:
    #     self.cpu_codegen.declare_array(sdfg, cfg, dfg, state_id, node, nodedesc, global_stream, declaration_stream)
    
    # def allocate_array(self, sdfg: SDFG, cfg: state.ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
    #                    nodedesc: data.Data, global_stream: CodeIOStream, declaration_stream: CodeIOStream,
    #                    allocation_stream: CodeIOStream) -> None:
         
    #     # if user provided this storage type, then we dump what they said.
    #     if nodedesc.storage == dtypes.StorageType.IPU_Tile_Local:
    #         name = node.data
    #         size = nodedesc.total_size
    #         ipu_type = "FLOAT"
    #         self.dispatcher.defined_vars.add(name, DefinedType.Scalar, ipu_type)
    #         declaration_stream.write(f'_state->graph.addVariable({ipu_type}, [{size}], {name});', cfg, state_id, node)       
    #         return
    
    #     self.cpu_codegen.allocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, global_stream, declaration_stream,
    #                                     allocation_stream)
        
    # def deallocate_array(self, sdfg: SDFG, cfg: state.ControlFlowRegion, dfg: SDFGState, state_id: int,
    #                      node: nodes.Node, nodedesc: data.Data, function_stream: CodeIOStream,
    #                      callsite_stream: CodeIOStream) -> None:
    #     # unless any cpu allocations no need for IPUs
    #     pass
    #     # return self.cpu_codegen.deallocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream,
    #     #                                         callsite_stream)        
    
    # def allocate_array(self, sdfg: dace.SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int,
        #                node: nodes.AccessNode, nodedesc: data.Array, function_stream: CodeIOStream,
        #                declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        # # Make sure the codegen includes the appropriate header files
        # self.add_header(function_stream)

        # name = node.data
        # print("ALLOCATE ARRAY - ", name)
        # # # Based on the hardware, the total size must be 16^2
        # # assert nodedesc.total_size == 16 * 16
        # # # Majority is detected by the strides of the data
        # # maj = 'row' if nodedesc.strides[-1] == 1 else 'col'

        # # Write a fragment based on the storage type
        # if nodedesc.storage == dace.StorageType.TensorCore_Accumulator:
        #     ctype = 'wmma::fragment<wmma::accumulator, 16, 16, 16, float>'
        #     declaration_stream.write(f'{ctype} {name};', cfg, state_id, node)
        # # else:
        # #     ctype = 'wmma::fragment<wmma::matrix_{mat}, 16, 16, 16, half, wmma::{maj}_major>'.format(
        # #         mat=('a' if 'A' in nodedesc.storage.name else 'b'), maj=maj)
        # #     declaration_stream.write(f'{ctype} {name};', cfg, state_id, node)
            
        # # # Add the ctype to defined_vars so that the codegen can properly pass
        # # # fragments to functions as an object reference.
        # self._dispatcher.defined_vars.add(name, DefinedType.Object, ctype)
        # self.cpu_codegen.allocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
        #                                  allocation_stream)   
#     def allocate_ipu_stream(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
#                         node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
#                         declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
#         dataname = node.data
#         allocname = cpp.ptr(dataname, nodedesc, sdfg, self._frame)
#         if nodedesc.storage == dtypes.StorageType.GPU_Global:
#             fmtargs = {
#                 'name': allocname,  # TODO: Handle persistent streams
#                 'allocname': allocname,
#                 'type': nodedesc.dtype.ctype,
#                 'is_pow2': sym2cpp(sympy.log(nodedesc.buffer_size, 2).is_Integer),
#                 'location': '%s_%s_%s' % (cfg.cfg_id, state_id, dfg.node_id(node))
#             }

#             ctypedef = 'dace::GPUStream<{type}, {is_pow2}>'.format(**fmtargs)
#             self._dispatcher.defined_vars.add(allocname, DefinedType.Stream, ctypedef)

#             if is_array_stream_view(sdfg, dfg, node):
#                 edges = dfg.out_edges(node)
#                 if len(edges) > 1:
#                     raise NotImplementedError("Cannot handle streams writing to multiple arrays.")

#                 fmtargs['ptr'] = nodedesc.sink + ' + ' + cpp_array_expr(
#                     sdfg, edges[0].data, with_brackets=False, codegen=self._frame)

#                 # Assuming 1D subset of sink/src
#                 # sym2cpp(edges[0].data.subset[-1])
#                 fmtargs['size'] = sym2cpp(nodedesc.buffer_size)

#                 # (important) Ensure GPU array is allocated before the stream
#                 datanode = dfg.out_edges(node)[0].dst
#                 sinkdesc = sdfg.arrays[datanode.data]
#                 self._dispatcher.dispatch_allocate(sdfg, cfg, dfg, state_id, datanode, sinkdesc, function_stream,
#                                                    allocation_stream)

#                 function_stream.write(
#                     'DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'
#                     .format(**fmtargs), cfg, state_id, node)
#                 self._globalcode.write(
#                     """
# DACE_EXPORTED void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);
# void __dace_alloc_{location}({type} *ptr, uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result) {{
#     result = dace::AllocGPUArrayStreamView<{type}, {is_pow2}>(ptr, size);
# }}""".format(**fmtargs), cfg, state_id, node)
#                 declaration_stream.write('dace::GPUStream<{type}, {is_pow2}> {name};'.format(**fmtargs), cfg, state_id,
#                                          node)
#                 allocation_stream.write('__dace_alloc_{location}({ptr}, {size}, {allocname});'.format(**fmtargs), cfg,
#                                         state_id, node)
#             else:
#                 fmtargs['size'] = sym2cpp(nodedesc.buffer_size)

#                 function_stream.write(
#                     'DACE_EXPORTED void __dace_alloc_{location}(uint32_t size, dace::GPUStream<{type}, {is_pow2}>& result);'
#                     .format(**fmtargs), cfg, state_id, node)
#                 self._globalcode.write(
#                     """
# DACE_EXPORTED void __dace_alloc_{location}(uint32_t {size}, dace::GPUStream<{type}, {is_pow2}>& result);
# void __dace_alloc_{location}(uint32_t {size}, dace::GPUStream<{type}, {is_pow2}>& result) {{
#     result = dace::AllocGPUStream<{type}, {is_pow2}>({size});
# }}""".format(**fmtargs), cfg, state_id, node)
#                 declaration_stream.write('dace::GPUStream<{type}, {is_pow2}> {name};'.format(**fmtargs), cfg, state_id,
#                                          node)
#                 allocation_stream.write('__dace_alloc_{location}({size}, {allocname});'.format(**fmtargs), cfg,
#                                         state_id, node)

    
    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        self.add_header(function_stream)
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)

        try:
            self.dispatcher.defined_vars.get(dataname)
            return
        except KeyError:
            pass  # The variable was not defined, we can continue

        # Check if array is already declared
        declared = False
        try:
            self.dispatcher.declared_arrays.get(dataname)
            declared = True  # Array was already declared in this or upper scopes
        except KeyError:  # Array not declared yet
            pass


        # if isinstance(nodedesc, dace.data.Stream):
        #     return self.allocate_ipu_stream(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
        #                                 allocation_stream)
            
       
        #print nodedesc type
        
            #return self.allocate_poplar_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                                                # allocation_stream)
        # elif isinstance(nodedesc, dace.data.Scalar):
        #     return self.allocate_scalar(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
        #                                            allocation_stream)
        
            
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        arrsize_malloc = '%s * sizeof(%s)' % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = nodedesc.dtype.ctype
        shape = nodedesc.shape
        

        # Different types of GPU arrays
        if nodedesc.storage == dtypes.StorageType.IPU_Memory:
            # Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
            result_alloc.write("Tensor %s = _state->graph.addVariable(%s, {%s});\n" % (dataname, nodedesc.dtype.ctype.capitalize(), sym2cpp(arrsize)))
            self.dispatcher.defined_vars.add(dataname, DefinedType.ArrayInterface, ctypedef)            
        elif nodedesc.storage == dtypes.StorageType.Register:
            if is_dynamically_sized:
                raise ValueError('Dynamic allocation of registers not allowed')
            if nodedesc.start_offset != 0:
                raise NotImplementedError('Start offset unsupported for registers')
            szstr = ' = {0}' if node.setzero else ''
            result_decl.write("%s %s[%s]%s;\n" % (nodedesc.dtype.ctype, dataname, sym2cpp(arrsize), szstr))
            self.dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
        else:
            raise NotImplementedError("IPU: Unimplemented storage type " + str(nodedesc.storage))

        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        if nodedesc.storage == dtypes.StorageType.IPU_Memory or \
            nodedesc.storage == dtypes.StorageType.Register:
            pass    # IPU variables are C++ objects and are automatically deallocated
        else:
            raise NotImplementedError
        
    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    src_node: Union[nodes.Tasklet, nodes.AccessNode], dst_node: Union[nodes.CodeNode, nodes.AccessNode],
                    memlet: Memlet, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        state = cfg.state(state_id)
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            src_parent = state.entry_node(src_node)
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        dst_parent = state.entry_node(dst_node)
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule
        
        callsite_stream.write("poplar::copy calls")
        # # Emit actual copy
        # self._emit_copy(state_id, src_node, src_storage, dst_node, dst_storage, dst_schedule, memlet, sdfg, cfg, dfg,
        #                 callsite_stream)

    def generate_node(self, sdfg: SDFG, cfg: state.ControlFlowRegion, state: SDFGState, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """(TASKLET only)
            0. Declarations
            1. Generate pre tasklet
            2. Generate tasklet code
            3. Generate post tasklet
            4. Writes
        """
        inner_stream, codegen = self.declarations(cfg, state_id, node, function_stream)    
        self.dispatcher.defined_vars.enter_scope(node)
        ############################################################################################################
        # self.pre_tasklet(sdfg, cfg, state, state_id, node, function_stream, callsite_stream, inner_stream, codegen)
        for edge in state.in_edges(node):
            self.generate_read(sdfg, state, edge, inner_stream)
        self.tasklet(sdfg, cfg, state, state_id, node, function_stream, inner_stream)
        after_memlets_stream = self.post_tasklet(sdfg, cfg, state, state_id, node, function_stream, inner_stream, codegen)
        ############################################################################################################
        callsite_stream.write('{', cfg, state_id, node)
        callsite_stream.write(inner_stream.getvalue(), cfg, state_id, node)
        callsite_stream.write(after_memlets_stream.getvalue())
        callsite_stream.write('}', cfg, state_id, node)
        self._locals.clear_scope(self._ldepth + 1)
        self.dispatcher.defined_vars.exit_scope(node)

    def declarations(self, cfg, state_id, node, function_stream):
        self.add_header(function_stream)
        inner_stream = CodeIOStream()
        state_dfg: SDFGState = cfg.nodes()[state_id]
        codegen = self.cpu_codegen or self
        return inner_stream,codegen

    def post_tasklet(self, sdfg, cfg, state, state_id, node, function_stream, inner_stream, codegen):
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_postamble(sdfg, cfg, state, state_id, node, function_stream, inner_stream,
                                        after_memlets_stream)
        # Process outgoing memlets
        codegen.process_out_memlets(sdfg, cfg, state_id, node, state, self.dispatcher, inner_stream, True, function_stream)
        return after_memlets_stream

    def tasklet(self, sdfg, cfg, state, state_id, node, function_stream, inner_stream):
        inner_stream.write("\n    ///////////////////\n", cfg, state_id, node)
        # Currently cpu
        self.unparse_ipu_tasklet(sdfg, cfg, state_id, state, node, function_stream, inner_stream, self._locals,
                                    self._ldepth, self._toplevel_schedule)
        inner_stream.write("    ///////////////////\n\n", cfg, state_id, node)

    def pre_tasklet(self, sdfg, cfg, state, state_id, node, function_stream, callsite_stream, inner_stream, codegen):
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_preamble(sdfg, cfg, state, state_id, node, function_stream, callsite_stream,
                                          after_memlets_stream)
            # SOME VARIABLE DECLARATIONS
        # post-memlet tasklet-preamble code
        
        callsite_stream.write(after_memlets_stream.getvalue())
        self.add_pre_tasklet_declarations(sdfg, cfg, state_id, state, node, function_stream, inner_stream)

    def unparse_ipu_tasklet(self, sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                        toplevel_schedule):
        # Change it later to IPU specific
        self.cpu_codegen.unparse_tasklet(sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                            toplevel_schedule)

    def add_pre_tasklet_declarations(self, sdfg, cfg, state_id, state, node, function_stream, inner_stream):
        
        arrays = set()
        for edge in state.in_edges(node):
            u = edge.src
            memlet = edge.data
            src_node = state.memlet_path(edge)[0].src

            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError("Duplicates found in memlets")
                ctype = node.in_connectors[edge.dst_conn].ctype
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data
                    if not shared_data_name:
                        # Very unique name. TODO: Make more intuitive
                        shared_data_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, state.node_id(src_node),
                                                                      state.node_id(node), edge.src_conn)

                    # Read variable from shared storage
                    defined_type, _ = self.dispatcher.defined_vars.get(shared_data_name)
                    if defined_type in (DefinedType.Scalar, DefinedType.Pointer):
                        assign_str = (f"const {ctype} {edge.dst_conn} = {shared_data_name};")
                    else:
                        assign_str = (f"const {ctype} &{edge.dst_conn} = {shared_data_name};")
                    inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                    self.dispatcher.defined_vars.add(edge.dst_conn, defined_type, f"const {ctype}")

                else:
                    self.dispatcher.dispatch_copy(
                        src_node,
                        node,
                        edge,
                        sdfg,
                        cfg,
                        state,
                        state_id,
                        function_stream,
                        inner_stream,
                    )

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1, ctype)
                arrays.add(edge.dst_conn)

    def generate_read(self, sdfg: SDFG, state: SDFGState, edge: graph.MultiConnectorEdge[mm.Memlet],
                      code: CodeIOStream):
        """
            Responsible for generating code for reads into a Tasklet, given the ingoing edge.
        """
        if edge.dst_conn is None:
            return
        src_node = state.memlet_path(edge)[0].src
        dst_type = edge.dst.in_connectors[edge.dst_conn]
        dst_name = edge.dst_conn
        if isinstance(src_node, nodes.Tasklet):
            ##################
            # Code->Code edges
            src_type = edge.src.out_connectors[edge.src_conn]
            if util.is_vector(src_type) and util.is_vector(dst_type):
                # Directly read from shared vector register
                code.write(f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = {edge.data.data};')
            elif util.is_scalar(src_type) and util.is_scalar(dst_type):
                # Directly read from shared scalar register
                code.write(f'{dst_type} {dst_name} = {edge.data.data};')
            elif util.is_scalar(src_type) and util.is_vector(dst_type):
                # Scalar broadcast from shared scalar register
                code.write(
                    f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = svdup_{util.TYPE_TO_SVE_SUFFIX[dst_type.type]}({edge.data.data});'
                )
            else:
                raise util.NotSupportedError('Unsupported Code->Code edge')
        elif isinstance(src_node, nodes.AccessNode):
            ##################
            # Read from AccessNode
            desc = src_node.desc(sdfg)
            if isinstance(desc, data.Array):
                # Copy from array
                if util.is_pointer(dst_type):
                    ##################
                    # Pointer reference
                    code.write(
                        f'{dst_type} {dst_name} = {cpp.cpp_ptr_expr(sdfg, edge.data, None, codegen=self.frame)};')
                elif util.is_vector(dst_type):
                    raise util.NotSupportedError('Unsupported read from array which is vector type, util.is_vector()')
                else:
                    ##################
                    # Scalar read from array
                    code.write(f'{dst_type} {dst_name} = {cpp.cpp_array_expr(sdfg, edge.data, codegen=self.frame)};')
            elif isinstance(desc, data.Scalar):
                # Refer to shared variable
                src_type = desc.dtype
                if util.is_vector(src_type) and util.is_vector(dst_type):
                    # Directly read from shared vector register
                    code.write(f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = {edge.data.data};')
                elif util.is_scalar(src_type) and util.is_scalar(dst_type):
                    # Directly read from shared scalar register
                    code.write(f'{dst_type} {dst_name} = {edge.data.data};')
                elif util.is_scalar(src_type) and util.is_vector(dst_type):
                    # Scalar broadcast from shared scalar register
                    code.write(
                        f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = svdup_{util.TYPE_TO_SVE_SUFFIX[dst_type.type]}({edge.data.data});'
                    )
                else:
                    raise util.NotSupportedError('Unsupported Scalar->Code edge')
        else:
            raise util.NotSupportedError('Only copy from Tasklets and AccessNodes is supported')
                      
    # def generate_state(self, 
    #                 sdfg:SDFG, 
    #                 cfg: ControlFlowRegion, 
    #                 state: SDFGState, 
    #                 function_stream: CodeIOStream, 
    #                 callsite_stream:CodeIOStream,
    #                 generate_state_footer:bool = True):
    #     debug_print_self(self)
    #     self._frame.generate_state(sdfg, cfg, state, function_stream, callsite_stream)
        
    # def declare_array(self,
    #                   sdfg: SDFG,
    #                   cfg: ControlFlowRegion,
    #                   dfg: StateSubgraphView,
    #                   state_id: int,
    #                   node: nodes.Node,
    #                   nodedesc: data.Data,
    #                   function_stream: CodeIOStream,
    #                   declaration_stream: CodeIOStream) -> None:
    #     print("IN DECLARE_ARRAY")
    #     fsymbols = self._frame.symbols_and_constants(sdfg)
    #     # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
    #     # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
    #     # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
    #     # `nodedesc` is a View and `dfg` is None.
    #     if dfg and not sdutils.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
    #         raise NotImplementedError("The declare_array method should only be used for variables "
    #                                   "that must have their declaration and allocation separate.")

    #     name = node.root_data
    #     ptrname = cpp.ptr(name, nodedesc, sdfg, self._frame)

    #     if nodedesc.transient is False:
    #         return

    #     # Check if array is already declared
    #     if self._dispatcher.declared_arrays.has(ptrname):
    #         return

    #     # Compute array size
    #     arrsize = nodedesc.total_size
    #     if not isinstance(nodedesc.dtype, dtypes.opaque):
    #         arrsize_bytes = arrsize * nodedesc.dtype.bytes

    #     if (nodedesc.storage == dtypes.StorageType.Register):
    #         ctypedef = dtypes.pointer(nodedesc.dtype).ctype
    #         declaration_stream.write(f'{nodedesc.dtype.ctype} *{name} = nullptr;\n', cfg, state_id, node)
    #         #Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
    #         declaration_stream.write(f'{nodedesc.dtype.ctype} {name}_const = graph.addConstant<{nodedesc.dtype.ctype}>({nodedesc.dtype.ctype.capitalize}, {arrsize}, {nodedesc.ctype}({nodedesc.dtype.ctype}));\n', cfg, state_id, node)
    #         self._dispatcher.declared_arrays.add(name, DefinedType.Pointer, ctypedef)
    #         return
    #     else:
    #         raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

############################################################################################################
# #### Helpers

    def add_header(self, function_stream: CodeIOStream):
        if self.has_generated_header:
            return
        self.has_generated_header = True

        # headers
        function_stream.write("#include <poplar/Vertex.hpp>\n")
        function_stream.write("#include <poplar/Graph.hpp>\n")
        function_stream.write("#include <poplar/Engine.hpp>\n")
        function_stream.write("#include <poplar/IPUModel.hpp>\n")
        function_stream.write("#include <poplar/DeviceManager.hpp>\n")
        function_stream.write("#include <poplar/Target.hpp>\n")
        function_stream.write("#include <poplar/Program.hpp>\n")
        function_stream.write("#include <poplar/Type.hpp>\n")
        function_stream.write("#include <poplar/exceptions.hpp>\n")
        function_stream.write("#include <poplar/OptionFlags.hpp>\n")
        function_stream.write("#include <poplar/EngineConnection.hpp>\n")
        function_stream.write("#include <poplar/ControlFlow.hpp>\n")
        function_stream.write("#include <popops/codelets.hpp>\n")
        function_stream.write("#include <popops/ElementWise.hpp>\n")
        function_stream.write("#include <popops/Reduce.hpp>\n")
        function_stream.write("#include <popops/Select.hpp>\n")
        function_stream.write("#include <popops/Sort.hpp>\n")
        # namespace
        function_stream.write(f'using namespace poplar; \n')
        function_stream.write(f'using namespace poplar::program; \n')
        
    # def debug_print_self(self):
    #     print("IN GENERATE_STATE")
    #     # print below ones as well
    #     print("TargetDispatcher:", self._dispatcher)
    #     print("init_code", self._frame._initcode.getvalue())
    #     print("exit_code", self._frame._exitcode.getvalue())
    #     print("Len env:", len(self._frame.environments))
    #     for _x in self._frame.statestruct:
    #         print("statestruct:", _x)
    #     print("environments:", self._frame.environments)
    #     print("targets:", self._frame.targets)
    #     print("to_allocate:", self._frame.to_allocate)
    #     print("where_allocated:", self._frame.where_allocated)
    #     print("fsyms:", self._frame.fsyms)
    #     print("_symbols_and_constants:", self._frame._symbols_and_constants)
    #     print("arglist:", self._frame.arglist)  
    #     print ("DONE")
    #     print("DISPATCHER Data")
    #     print ("used_env", self._dispatcher.used_environments)
    #     print ("used_targets", self._frame.dispatcher.used_targets)
    #     print("DONE")
    #     #######
    #     print("TargetCodeGenerator:", self)
    #     print("language", self.language)
    #     # print("TargetDispatcher:", self._dispatcher.used_targets)

    # def generate_scope(self,
    #                    sdfg: SDFG,
    #                    cfg: ControlFlowRegion,
    #                    dfg_scope: ScopeSubgraphView,
    #                    state_id: int,
    #                    function_stream: CodeIOStream,
    #                    callsite_stream: CodeIOStream) -> None:
    #     # Get the first entry node of Map
    #     entry_node = dfg_scope.source_nodes()[0]

    #     # function_stream.write('extern int __dace_comm_size, __dace_comm_rank;', cfg, state_id, entry_node)
    #     callsite_stream.write('{', cfg, state_id, entry_node)

    #     # cpp.presynchronize_streams(sdfg, cfg, dfg_scope, state_id, entry_node, callsite_stream)   #TODO: add some other function of own.
    #     # Should we ?
    #     # self.generate_node(sdfg, cfg, dfg_scope, state_id, entry_node, function_stream, callsite_stream)
    #     # generated nested subgraphs
    #     self._dispatcher.dispatch_subgraph(sdfg,
    #                                        cfg,
    #                                        dfg_scope,
    #                                        state_id,
    #                                        function_stream,
    #                                        callsite_stream,
    #                                        skip_entry_node=True)

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


    # #$$$$ First dace::copy()
    #     for var, r in zip(map_header.map.params, map_header.map.range):
    #         begin, end, skip = r

    #         callsite_stream.write('{\n', cfg, state_id, map_header)
    #         callsite_stream.write(
    #             '%s %s = %s + __dace_comm_rank * (%s);\n' %
    #             (symtypes[var], var, cppunparse.pyexpr2cpp(symbolic.symstr(begin, cpp_mode=True)),
    #              cppunparse.pyexpr2cpp(symbolic.symstr(skip, cpp_mode=True))), cfg, state_id, map_header)

    #     self._frame.allocate_arrays_in_scope(sdfg, cfg, map_header, function_stream, callsite_stream)


    # This will generate the src/cuda/xyz.cu files and folders using "codeObjects" class.
    # We don't need this now as we are mostly concerned about a single file codegen as of now.
    # def get_generated_codeobjects(self):
        # fileheader = CodeIOStream()
        # sdfg = self._global_sdfg  
        
        # # cuda/mpi seemed to be using this follow 
        # params_comma = self._global_sdfg.init_signature(free_symbols=self._frame.free_symbols(self._global_sdfg))
        # if params_comma:
        #     params_comma = ', ' + params_comma
        # codelet_file_code = """
        #                     // Copyright (c) 2018 Graphcore Ltd. All rights reserved.
        #                     // Copied from tut3_vertices from Poplar SDK tutorials

        #                     #include <poplar/Vertex.hpp>

        #                     class SumVertex : public poplar::Vertex {
        #                         public:
        #                         // Fields
        #                         poplar::Input<poplar::Vector<float>> in;
        #                         poplar::Output<float> out;

        #                         // Compute function
        #                         bool compute() {
        #                             *out = 0;
        #                             for (const auto &v : in) {
        #                             *out += v;
        #                             }
        #                             return true;
        #                         }
        #                     };
        #                     """
        
        # codeobj = CodeObject(
        #     name=sdfg.name + '_codelets', 
        #     code=codelet_file_code,
        #     language='cpp', 
        #     target=IPUCodeGen, 
        #     title='IPU',
        #     linkable=False)
        
        # # Fill in the list
        # return [codeobj]