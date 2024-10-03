# import
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from io import StringIO
from dace.codegen.codeobject import CodeObject
import sympy
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union
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
from dace.codegen.targets.ipu_files import ipu_utils as ipu_utils
from dace.codegen.targets.cpp import (codeblock_to_cpp, cpp_array_expr, memlet_copy_to_absolute_strides, sym2cpp,
                                      synchronize_streams, unparse_cr, mangle_dace_state_struct_name)

import copy
import functools
import itertools
import warnings

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator
    from dace.codegen.targets.cpu import CPUCodeGen
import pdb; 

def is_ipu_kernel(sdfg, state):
        """
        Returns whether the given state is an FPGA kernel and should be dispatched
        to the FPGA code generator.

        :return: True if this is an FPGA kernel, False otherwise.
        """
        # pdb.set_trace()
        data_nodes = state.data_nodes()
        at_least_one_ipu_allocated_array = False
        for n in data_nodes:
            desc = n.desc(sdfg)
            # print(desc.storage.name, desc.storage, desc)
            if desc.storage == dtypes.StorageType.IPU_Memory:
                at_least_one_ipu_allocated_array = True
            if isinstance(desc, data.Scalar):
                continue
            if desc.storage != dtypes.StorageType.IPU_Memory:
                return False
        return at_least_one_ipu_allocated_array
    
@registry.autoregister_params(name='ipu')
class IPUCodeGen(TargetCodeGenerator):
    """ IPU(Graphcore) code generator. """
    target_name = 'ipu'
    title = 'IPU'
    language = 'cpp'
    _in_device_code = False

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: SDFG):

        self.program_name = sdfg.name
        
        self.has_generated_header = False
        self.frame = frame_codegen
        self.dispatcher = frame_codegen._dispatcher
        self.cpu_codegen: Optional['CPUCodeGen'] = None
        # self._locals = cppunparse.CPPLocals()
        # Scope depth (for defining locals)
        self._ldepth = 0
        # Keep nested SDFG schedule when descending into it
        self._toplevel_schedule = None
        self._localcode = CodeIOStream()
        self._globalcode = CodeIOStream()
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self._global_sdfg: SDFG = sdfg
        self._arglists: Dict[nodes.MapEntry, Dict[str, data.Data]] = {}
        # Keep track of current "scope entry/exit" code streams for extra
        # code generation
        self.scope_entry_stream = self._initcode
        self.scope_exit_stream = self._exitcode
        self._ipu_streams, self._ipu_events = 0, 0
        self._kernels_dependencies = dict()
        self._kernels_names_to_id = dict()
        self._num_kernels = 0
        self._host_codes = []   
        self._kernel_codes = []
        self._generated_nodes = []
        

        # Register dispatchers
        self.cpu_codegen = self.dispatcher.get_generic_node_dispatcher()        
        
        self.dispatcher.register_state_dispatcher(self, predicate=is_ipu_kernel)
        # self.dispatcher.register_array_dispatcher(dtypes.StorageType.IPU_Tile_Local, self)
            
        # Storage
        # ipu_storage = [dtypes.StorageType.IPU_Memory]        
        ipu_storage = [dtypes.StorageType.IPU_Memory]
        self.dispatcher.register_array_dispatcher(ipu_storage, self)   # allocate_array/deallocate_array
        for storage in ipu_storage:
            for other_storage in dtypes.StorageType:
                self.dispatcher.register_copy_dispatcher(storage, other_storage, None, self)
                self.dispatcher.register_copy_dispatcher(other_storage, storage, None, self)
        
        
        
                
                
                
        # # Dispatchers
        # self.dispatcher.register_map_dispatcher(dace.ScheduleType.IPU_Map, self)
        # self.dispatcher.register_node_dispatcher(self, self.is_ipu_map_scope)
        # self.dispatcher.register_node_dispatcher(self, self.is_node_library_node)
        # self.dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
        # self.dispatcher.register_copy_dispatcher(dtypes.StorageType.Register, dtypes.StorageType.IPU_Tile_Local, None, func=self)
        # self._dispatcher.register_map_dispatcher(dace.ScheduleType.IPU, self)
        # self._dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

    def preprocess(self, sdfg: SDFG) -> None:
        self.frame.statestruct.append('dace_poplar_context *poplar_context;')
        pass

    def get_generated_codeobjects(self):
        params_comma = self._global_sdfg.init_signature(free_symbols=self.frame.free_symbols(self._global_sdfg))
        if params_comma:
            params_comma = ', ' + params_comma
 
        host_code = CodeIOStream()       
        host_code.write("""
#include <dace/dace.h>
""")
        fileheader = CodeIOStream()
        self.frame.generate_fileheader(self._global_sdfg, fileheader, 'poplar')
        
        host_code.write("""
{file_header}

{other_globalcode}

DACE_EXPORTED int __dace_init_ipu({sdfg_state_name} *__state{params}) {{
    __state->poplar_context = new dace_poplar_context();
    return 0;
}}

DACE_EXPORTED int __dace_exit_ipu({sdfg_state_name} *__state) {{
    delete __state->poplar_context;
    return 0;
}}

DACE_EXPORTED auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> 
{{
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {{
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {{
            std::cout << " - attached" << std::endl;
            device = {{std::move(d)}};
            break;
        }} else {{
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }}
    }}
    return device;
}}

DACE_EXPORTED auto defineDataStreams({sdfg_state_name} &__state)
{{
    auto toIpuStream = __state.poplar_context->graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = __state.poplar_context->graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    __state.poplar_context->programs["copy_to_ipu"] = Copy(toIpuStream, __state.poplar_context->tensors["data"]);
    __state.poplar_context->programs["copy_to_host"] = Copy(__state.poplar_context->tensors["data"], fromIpuStream);
}}

{host_code_seperator}""".format(params=params_comma,
           sdfg_state_name=mangle_dace_state_struct_name(self._global_sdfg),
           other_globalcode=self._globalcode.getvalue(),
           file_header=fileheader.getvalue(),
           sdfg=self._global_sdfg, 
           host_code_seperator="".join([
                          "{separator}\n// Dataflow graph building: {kernel_name}"
                          "\n{separator}\n\n{code}\n\n".format(separator="/" * 79, kernel_name=name, code=code)
                          for (name, code) in self._host_codes])))

        host_code_obj = CodeObject(self.program_name,
                                   host_code.getvalue(),
                                   "cpp",
                                   IPUCodeGen,
                                   "IPU",
                                   target_type="host")

        # Device object
        kernel_code_objs = [
            CodeObject(kernel_name,
                       code,
                       "cpp",
                       IPUCodeGen,
                       "IPU",
                       target_type="device") for (kernel_name, code) in self._kernel_codes
        ]
                
        return [host_code_obj] +  kernel_code_objs

    # __dace_init_<TARGET> function
    @property
    def has_initializer(self):
        return True

    # __dace_exit_<TARGET> function
    @property
    def has_finalizer(self):
        return True
    
    def state_dispatch_predicate(self, sdfg, state):
        if self._toplevel_schedule == dtypes.ScheduleType.IPU_SCHEDULE:
            return True
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
    
    def is_node_library_node(self, sdfg, state, node):
        print("NODE is = ", type(node).__name__)
        if isinstance(node, nodes.LibraryNode):
            return True   
        return False
    
    def node_dispatch_predicate(self, sdfg, state, node):
        return True
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
    def allocate_ipu_scalar(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                        declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
            
        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        #arrsize_malloc = '%s * sizeof(%s)' % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = 'Tensor *'
        shape = nodedesc.shape
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)
        
        # Check if array is already declared
        declared = self.dispatcher.declared_arrays.has(dataname)
        # Different types of memories
        if nodedesc.storage == dtypes.StorageType.IPU_Memory:
            if not declared:
                result_decl.write('%s %s;\n' % (ctypedef, dataname))    # Tensor *p;
            self.dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
            
            if nodedesc.pool:
                raise NotImplementedError("Pool not implemented yet " + str(nodedesc.storage))
            else:
                shape_poplar_format = ', '.join([str(sh) for sh in shape])
                result_alloc.write("%s = _state->graph.addVariable(%s, {%s});\n" % (dataname, ipu_utils.TYPE_TO_IPU[nodedesc.dtype], shape_poplar_format))           
        else:
            raise NotImplementedError("IPU: Unimplemented StorageType " + str(nodedesc.storage))
        
        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)
          
    def allocate_ipu_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                        declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        
        result_decl = StringIO()
        result_alloc = StringIO()
        arrsize = nodedesc.total_size
        is_dynamically_sized = symbolic.issymbolic(arrsize, sdfg.constants)
        #arrsize_malloc = '%s * sizeof(%s)' % (sym2cpp(arrsize), nodedesc.dtype.ctype)
        ctypedef = 'Tensor *'
        shape = nodedesc.shape
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)
        
        # Check if array is already declared
        declared = self.dispatcher.declared_arrays.has(dataname)
        # Different types of memories
        if nodedesc.storage == dtypes.StorageType.IPU_Memory:
            if not declared:
                result_decl.write('%s %s;\n' % (ctypedef, dataname))    # Tensor *p;
            self.dispatcher.defined_vars.add(dataname, DefinedType.Pointer, ctypedef)
            
            if nodedesc.pool:
                raise NotImplementedError("Pool not implemented yet " + str(nodedesc.storage))
            else:
                shape_poplar_format = ', '.join([str(sh) for sh in shape])
                result_alloc.write("%s = _state->graph.addVariable(%s, {%s});\n" % (dataname, ipu_utils.TYPE_TO_IPU[nodedesc.dtype], shape_poplar_format))           
        else:
            raise NotImplementedError("IPU: Unimplemented StorageType " + str(nodedesc.storage))
        
        declaration_stream.write(result_decl.getvalue(), cfg, state_id, node)
        allocation_stream.write(result_alloc.getvalue(), cfg, state_id, node)
            
    def allocate_ipu_stream(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                        node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                        declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        return NotImplementedError("IPU Stream not implemented yet")
#         dataname = node.data
#         allocname = cpp.ptr(dataname, nodedesc, sdfg, self.frame)
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

    def decidemapping(self, dataname, nodedesc, sdfg):

        # Get the shape of the data descriptor
        shape = nodedesc.shape
        # Get the total size of the data descriptor
        size = nodedesc.total_size

        # CREATE a dictionary to store the mapping of the data to the tile
        dataToTileMap = {}
        # Get the number of tiles
        numTiles = 10
        # Get the number of elements in the data descriptor
        numElements = size
        
        if (numElements < numTiles):    # special case
            numTiles = numElements

        # Get the number of elements per tile
        numElementsPerTile = numElements // numTiles
        # Get the number of elements in the last tile
        numElementsLastTile = numElements % numTiles

        # Loop over the number of tiles
        for i in range(numTiles):
            # Get the start index of the tile
            start = i * numElementsPerTile
            # Get the end index of the tile
            end = start + numElementsPerTile
            if (end - start > 1):
                # Get the data of the tile with slicing
                data = dataname + ".slice(" + "[" + str(start) + ":" + str(end) + "]" + ")"
            else:
                data = dataname + "[" + str(start) + "]"
                
            # Add the data to the tile mapping
            dataToTileMap[data] = i
        
        # # Get the start index of the last tile
        # start = numTiles * numElementsPerTile
        # # Get the end index of the last tile
        # end = start + numElementsLastTile
        # # Get the data of the last tile
        # data = dataname + "[" + str(start) + ":" + str(end) + "]"
        # # Add the data to the tile mapping
        # dataToTileMap[data] = numTiles - 1
        
        return dataToTileMap

    # TODO:Similar mapVertexOntile            
    def mapdataontile(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                    node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                    declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        if isinstance(nodedesc, dace.data.Array):
            self.mapArrayOnTile(sdfg, cfg, state_id, node, nodedesc, allocation_stream)
        elif isinstance(nodedesc, dace.data.Scalar):
            self.mapScalarOnTile(sdfg, cfg, state_id, node, nodedesc, allocation_stream)
        else:
            raise NotImplementedError("Unimplemented mapping for this AccessNode: {}".format(type(nodedesc)))

    def mapArrayOnTile(self, sdfg, cfg, state_id, node, nodedesc, allocation_stream):
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)
            # Map array intelligently
        spreadOverTiles = True
        if spreadOverTiles:
            dataToTileMap = self.decidemapping(dataname, nodedesc, sdfg)
                # Map array over multiple tiles
                # loop over the dataToTileMap and set the mapping
                # import pprint
                # pprint.pprint(dataToTileMap)
                
            for data, tilenumber in dataToTileMap.items():
                setTileMappingCall = f"_state->graph.setTileMapping({data}, {tilenumber});"
                allocation_stream.write(setTileMappingCall, cfg, state_id, node)
        else:
                # Map array, given only 1 element maps on one tile
            tilenumber = 0
            setTileMappingCall = f"_state->graph.setTileMapping({dataname}, {tilenumber});"
            allocation_stream.write(setTileMappingCall, cfg, state_id, node)

    def mapScalarOnTile(self, sdfg, cfg, state_id, node, nodedesc, allocation_stream):
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)
            # Map scalar, given only 1 element maps on one tile
        tilenumber = 0
        setTileMappingCall = f"_state->graph.setTileMapping({dataname}, {tilenumber});"
        allocation_stream.write(setTileMappingCall, cfg, state_id, node)
        
    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                       node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> None:
        
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            nodedesc = update_persistent_desc(nodedesc, sdfg)        
        
        dataname = cpp.ptr(node.data, nodedesc, sdfg, self.frame)

        try:
            self.dispatcher.defined_vars.get(dataname)
            return
        except KeyError:
            pass  # The variable was not defined, we can continue

        if isinstance(nodedesc, dace.data.Stream):
            self.allocate_ipu_stream(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                                        allocation_stream)
        elif isinstance(nodedesc, dace.data.View):
            self._cpu_codegen.allocate_view(sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream,
                                                   allocation_stream)
        elif isinstance(nodedesc, dace.data.Reference):
            self._cpu_codegen.allocate_reference(sdfg, cfg, dfg, state_id, node, function_stream,
                                                        declaration_stream, allocation_stream)
        elif isinstance(nodedesc, dace.data.Array):
            self.allocate_ipu_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
            self.mapdataontile(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
        elif isinstance(nodedesc, dace.data.Scalar):
            self.allocate_ipu_scalar(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
            self.mapdataontile(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream, allocation_stream)
        else:
            raise NotImplementedError("Unimplemented type: {}".format(type(nodedesc)))
        
        # Mapping on tiles
        

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        if nodedesc.storage == dtypes.StorageType.IPU_Memory or \
            nodedesc.storage == dtypes.StorageType.Register:
            pass    # IPU variables are C++ objects and are automatically deallocated
        else:
            raise NotImplementedError("Unimplemented deallocate() for StorageType " + str(nodedesc.storage))
        
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

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.Node, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        print("Generating node: ", node.label)
            # Dynamically obtain node generator according to class name
            # gen = getattr(self, '_generate_' + type(node).__name__, False)
            # if gen is not False:  # Not every node type has a code generator here
            #     gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
            #     return

        # self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        
    # def generate_node(self, sdfg: SDFG, cfg: state.ControlFlowRegion, state: SDFGState, state_id: int, node: nodes.Node,
    #                   function_stream: CodeIOStream, callsite_stream: CodeIOStream):
    #     """(TASKLET only)
    #         0. Declarations
    #         1. Generate pre tasklet
    #         2. Generate tasklet code
    #         3. Generate post tasklet
    #         4. Writes
    #     """
    #     callsite_stream.write(f"// Generating node {node.label}\n")
        # inner_stream, codegen = self.declarations(cfg, state_id, node, function_stream)    
        # self.dispatcher.defined_vars.enter_scope(node)
        # ############################################################################################################
        # # self.pre_tasklet(sdfg, cfg, state, state_id, node, function_stream, callsite_stream, inner_stream, codegen)
        # for edge in state.in_edges(node):
        #     self.generate_read(sdfg, state, edge, inner_stream)
        # callsite_stream.write('SJJ:TASKLET', cfg, state_id, node)
        # function_stream.write("SJJ:TASKLET Call  {0}() {{\n".format(node.label), cfg, state_id, node)
        # self.tasklet(sdfg, cfg, state, state_id, node, function_stream, inner_stream)
        # after_memlets_stream = self.post_tasklet(sdfg, cfg, state, state_id, node, function_stream, inner_stream, codegen)
        # ############################################################################################################
        # callsite_stream.write('{', cfg, state_id, node)
        # callsite_stream.write(inner_stream.getvalue(), cfg, state_id, node)
        # callsite_stream.write(after_memlets_stream.getvalue())
        # callsite_stream.write('}', cfg, state_id, node)
        # self._locals.clear_scope(self._ldepth + 1)
        # self.dispatcher.defined_vars.exit_scope(node)

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
        function_stream.write(f"SJJ:  {node.label}() {{\n", cfg, state_id, node)
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
                      
    def generate_state(self, 
                    sdfg:SDFG, 
                    cfg: ControlFlowRegion, 
                    state: SDFGState, 
                    function_stream: CodeIOStream, 
                    callsite_stream:CodeIOStream,
                    generate_state_footer:bool = True):
        print("IPU STATE\n")
        # disp = self.dispatcher.get_scope_dispatcher(dtypes.ScheduleType.Unrolled)
        ipu_disp = self.dispatcher.get_state_dispatcher(sdfg, state=state)
        cpu_disp = self.cpu_codegen
        self.dispatcher._used_targets.add(ipu_disp)
        self.dispatcher._used_targets.add(cpu_disp)
        
        state_id = state.block_id
        
        if IPUCodeGen._in_device_code:
            print("IN DEVICE CODE")
            
            to_allocate = dace.sdfg.local_transients(sdfg, state, None)
            allocated = set()
            subgraphs = dace.sdfg.concurrent_subgraphs(state)

            for node in state.data_nodes():
                data = node.desc(sdfg)
                if node.data not in to_allocate or node.data in allocated:
                    continue
                # Make sure there are no global transients in the nested state
                # that are thus not gonna be allocated
                if data.storage == dtypes.StorageType.IPU_Memory and not isinstance(data, data.View):
                    raise cgx.CodegenError("Cannot allocate global memory from device code.")
                allocated.add(node.data)
                # Allocate transients
                self._dispatcher.dispatch_allocate(sdfg, cfg, state, state_id, node, data, function_stream,
                                                   callsite_stream)

            self.generate_nested_state(sdfg, cfg, state, state.label, subgraphs, function_stream, callsite_stream)
            
        else:
            print("IN HOST CODE")
            sdfg_state_name = cpp.mangle_dace_state_struct_name(self._global_sdfg)
            print("SDFG STATE NAME: ", sdfg_state_name)
            formatted_string = """
                                              
            // hack to make the files compile by forward declaring the functions
            extern "C" auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device>;
            extern "C" void defineDataStreams({sdfg_state_name} &__state);
            extern "C" void kernel_buildComputeGraph({sdfg_state_name} &__state);
                                              """.format(sdfg_state_name=sdfg_state_name)
            
            function_stream.write(formatted_string)
            
            # self.frame.generate_ipu_state(sdfg, cfg, state, function_stream, callsite_stream, generate_state_footer=False)
            self.generate_ipu_cpuside_state(sdfg, cfg, state, function_stream, callsite_stream, generate_state_footer=False)
            
############################################################################################################
# #### Helpers

    def generate_ipu_cpuside_state(self,
                                sdfg: SDFG,
                                cfg: ControlFlowRegion,
                                state: SDFGState,
                                function_stream: CodeIOStream,
                                callsite_stream: CodeIOStream,
                                generate_state_footer: bool = True):
        sid = state.block_id
        
        callsite_stream.write(f'// Ipu pipeline \n', sdfg)
        callsite_stream.write(f"""
            // Data initialization
            __state->poplar_context->hostData = vector<float>(NUM_DATA_ITEMS, 1);

            // Real code pipeline starts from here.
            std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
            __state->poplar_context->device = getIpuDevice(1);
            if (!__state->poplar_context->device.has_value()) {{
                std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
                return;
            }}
        """)
        #####################
        # Create dataflow graph for state's children.

        
        # Start a new state code generation: reset previous dependencies if any
        self._kernels_dependencies.clear()
        self._kernels_names_to_id.clear()
        
        # For now only 1 kernel.
        kernels = [(state, 0)]


        state_host_header_stream = CodeIOStream()
        state_host_body_stream = CodeIOStream()
        instrumentation_stream = CodeIOStream()
    
        for kern, kern_id in kernels:
            if sdfg.parent_nsdfg_node is not None:
                kernel_name = f"{sdfg.parent_nsdfg_node.label}_{state.label}_{kern_id}_{cfg.cfg_id}"
            else:
                kernel_name = f"{state.label}_{kern_id}_{cfg.cfg_id}"
            self._kernels_names_to_id[kernel_name] = kern_id

        kernel_host_stream = CodeIOStream()            
        function_stream.write(f"// kernel_name = {kernel_name}\n")
        self.generate_host_function(sdfg, cfg, state, sid, function_stream, callsite_stream, state_host_header_stream, state_host_body_stream, instrumentation_stream, kernel_host_stream)

        # Store code strings to be passed to compilation phase
        self._host_codes.append((kernel_name, kernel_host_stream.getvalue()))
            
        #####################
        # Write state footer(After kernel call?)
        callsite_stream.write(f"""
            std::cout << "STEP 3: Define data streams" << std::endl;
            defineDataStreams(*__state);  // Pass the state directly

            std::cout << "STEP 4: Create engine and compile graph" << std::endl;
            __state->poplar_context->engineOptions = OptionFlags{{
                {{"target.saveArchive", "archive.a"}},
                {{"debug.instrument", "true"}},
                {{"debug.instrumentCompute", "true"}},
                {{"debug.instrumentControlFlow", "true"}},
                {{"debug.computeInstrumentationLevel", "tile"}},
                {{"debug.outputAllSymbols", "true"}},
                {{"autoReport.all", "true"}},
                {{"autoReport.outputSerializedGraph", "true"}},
                {{"debug.retainDebugInformation", "true"}},
            }};

            __state->poplar_context->programIds = map<string, int>();
            __state->poplar_context->programsList = vector<Program>(__state->poplar_context->programs.size());  // Removing the size causes segfault
            int index = 0;
            for (auto &nameToProgram : __state->poplar_context->programs) {{
                __state->poplar_context->programIds[nameToProgram.first] = index;
                __state->poplar_context->programsList[index] = nameToProgram.second;
                index++;
            }}

            // Now construct the Engine using the constructor
            auto engine = Engine(__state->poplar_context->graph, __state->poplar_context->programsList, __state->poplar_context->engineOptions);

            std::cout << "STEP 5: Load compiled graph onto the IPU tiles" << std::endl;
            engine.load(*__state->poplar_context->device);
            // engine.enableExecutionProfiling();

            std::cout << "STEP 6: Attach data streams" << std::endl;
            
            engine.connectStream("TO_IPU", __state->poplar_context->hostData.data());
            engine.connectStream("FROM_IPU", __state->poplar_context->hostData.data());

            std::cout << "STEP 7: Run programs" << std::endl;
            engine.run(__state->poplar_context->programIds["copy_to_ipu"]);  // Copy to IPU
            engine.run(__state->poplar_context->programIds["main"]);         // Main program
            engine.run(__state->poplar_context->programIds["copy_to_host"]); // Copy from IPU
        """)

            ## Generate the global function here
    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[mmlt.Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        self.dispatcher.dispatch_copy(src_node, dst_node, edge, sdfg, cfg, state_dfg, state_id, function_stream,
                                       callsite_stream)
        
    def generate_nested_state(self, sdfg: SDFG, cfg: ControlFlowRegion, state: dace.SDFGState, nest_name: str,
                              subgraphs: List[ScopeSubgraphView], function_stream: CodeIOStream,
                              callsite_stream: CodeIOStream) -> None:

        for sg in subgraphs:
            self.dispatcher.dispatch_subgraph(sdfg,
                                               cfg,
                                               sg,
                                               sdfg.node_id(state),
                                               function_stream,
                                               callsite_stream,
                                               skip_entry_node=False)
            
    def generate_host_function(self, sdfg, cfg, state, state_id, function_stream, callsite_stream, state_host_header_stream, state_host_body_stream, instrumentation_stream, kernel_host_stream):
        # Basic arguments setting
        kernel_args_call_host = []
        kernel_args_opencl = []
        # Include state in args
        kernel_args_opencl.append(f"{cpp.mangle_dace_state_struct_name(self._global_sdfg)} &__state")
        kernel_args_call_host.append(f"*__state")

        # real code starts
        host_function_name = f"kernel_buildComputeGraph"
        
        callsite_stream.write("////////////////////////////////////////KERNEL")
        callsite_stream.write("std::cout << \"STEP 2: Building the compute graph\" << std::endl;")
        callsite_stream.write("{}({});".format(host_function_name, ", ".join(kernel_args_call_host)))
        callsite_stream.write("////////////////////////////////////////")
        
        # function_stream.write("\n\nDACE_EXPORTED auto {}({});\n\n".format(host_function_name,
                                                                            # ", ".join(kernel_args_opencl)))
        
        #///////////////////////////
        # add generated header information
        kernel_host_stream.write(state_host_header_stream.getvalue())

        kernel_host_stream.write(f"""\
    DACE_EXPORTED void {host_function_name}({', '.join(kernel_args_opencl)}) {{""")
        
        # write the kernel_host_stream withe the commands I have copied
        kernel_host_stream.write(f"""\
                std::cout << "  STEP 2.1: Create graph and compile codelets" << std::endl;
                
                // Step 1: Create graph and add codelets
                __state.poplar_context->graph = poplar::Graph(__state.poplar_context->device->getTarget());
                //__state.poplar_context->graph.addCodelets({{"src/codelets/SkeletonCodelets.cpp"}}, "-O3 -I codelets");
                popops::addCodelets(__state.poplar_context->graph);
            """)
        
        kernel_host_stream.write("""
                // Step 2: Add data to the graph
                std::cout << "  STEP 2.2: Add data to the graph" << std::endl;""")
        # Emit internal transient array allocation
        # __state.poplar_context->tensors["data"] = __state.poplar_context->graph.addVariable(poplar::FLOAT, {{NUM_DATA_ITEMS}}, "data");            
        self.frame.allocate_arrays_in_scope(sdfg, cfg, state, function_stream, kernel_host_stream)
        kernel_host_stream.write('\n')
        
        kernel_host_stream.write("""
                poputil::mapTensorLinearly(__state.poplar_context->graph, __state.poplar_context->tensors["data"]);
                """)
        kernel_host_stream.write("""
                const int numTiles = __state.poplar_context->device->getTarget().getNumTiles();
                // Add programs and wire up data
                const auto NumElemsPerTile = NUM_DATA_ITEMS / numTiles;
                //auto cs = __state.poplar_context->graph.addComputeSet("loopBody");
                //
                //for (auto tileNum = 0; tileNum < numTiles; tileNum++) {{
                //    const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)NUM_DATA_ITEMS);
                //    const auto sliceStart = tileNum * NumElemsPerTile;
                //    auto v = __state.poplar_context->graph.addVertex(cs, "SkeletonVertex", {{"data", __state.poplar_context->tensors["data"].slice(sliceStart, sliceEnd)}});
                //    __state.poplar_context->graph.setInitialValue(v["howMuchToAdd"], tileNum);
                //    __state.poplar_context->graph.setPerfEstimate(v, 100);
                //    __state.poplar_context->graph.setTileMapping(v, tileNum);
                //}}
                //
                //__state.poplar_context->programs["main"] = Repeat(10, Execute(cs));
                //                 """)

        kernel_host_stream.write("}\n")
        
        self.frame.deallocate_arrays_in_scope(sdfg, cfg, state, function_stream, callsite_stream)

    def generate_kernel(self,
                        sdfg: dace.SDFG,
                        cfg: ControlFlowRegion,
                        state: dace.SDFGState,
                        kernel_name: str,
                        subgraphs: list,
                        function_stream: CodeIOStream,
                        callsite_stream: CodeIOStream,
                        state_host_header_stream: CodeIOStream,
                        state_host_body_stream: CodeIOStream,
                        instrumentation_stream: CodeIOStream,
                        state_parameters: list,
                        kernel_id: int = None):
        """
        Entry point for generating an FPGA Kernel out of the given subgraphs.

        :param sdfg:
        :param state:
        :param kernel_name: the generated kernel name.
        :param subgraphs: the connected components that constitute this kernel.
        :param function_stream: CPU code stream, contains global declarations.
        :param callsite_stream: CPU code stream, contains code for invoking kernels, ...
        :param state_host_header_stream: Device-specific host code stream: contains the host code
            for the state global declarations.
        :param state_host_body_stream: Device-specific host code stream: contains all the code related
            to this state, for creating transient buffers, spawning kernels, and synchronizing them.
        :param instrumentation_stream: Code for profiling kernel execution time.
        :param state_parameters: a list of parameters that must be passed to the state. It will get populated
            considering all the parameters needed by the kernels in this state.
        :param kernel_id: Unique ID of this kernels as computed in the generate_state function
        """
        kernel_stream = CodeIOStream()
        #   # Actual kernel code generation
        # self.generate_kernel_internal(sdfg, cfg, state, kernel_name, predecessors, subgraphs, kernel_stream,
        #                               state_host_header_stream, state_host_body_stream, instrumentation_stream,
        #                               function_stream, callsite_stream, state_parameters)
        kernel_stream.write(f"// Kernel {kernel_name} called here", sdfg, state)
        # Store code strings to be passed to compilation phase
        self._kernel_codes.append((kernel_name, kernel_stream.getvalue()))
        
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
    #     print("TargetDispatcher:", self.dispatcher)
    #     print("init_code", self.frame._initcode.getvalue())
    #     print("exit_code", self.frame._exitcode.getvalue())
    #     print("Len env:", len(self.frame.environments))
    #     for _x in self.frame.statestruct:
    #         print("statestruct:", _x)
    #     print("environments:", self.frame.environments)
    #     print("targets:", self.frame.targets)
    #     print("to_allocate:", self.frame.to_allocate)
    #     print("where_allocated:", self.frame.where_allocated)
    #     print("fsyms:", self.frame.fsyms)
    #     print("_symbols_and_constants:", self.frame._symbols_and_constants)
    #     print("arglist:", self.frame.arglist)  
    #     print ("DONE")
    #     print("DISPATCHER Data")
    #     print ("used_env", self.dispatcher.used_environments)
    #     print ("used_targets", self.frame.dispatcher.used_targets)
    #     print("DONE")
    #     #######
    #     print("TargetCodeGenerator:", self)
    #     print("language", self.language)
        # print("TargetDispatcher:", self._dispatcher.used_targets)

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
