# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Code generation: This module is responsible for converting an SDFG into SVE code.
"""

from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.cpp import sym2cpp
from dace import dtypes, memlet as mm
from dace.sdfg import graph, state, find_input_arraynode, find_output_arraynode
from dace.sdfg.scope import is_in_scope
import itertools
from dace.codegen.targets.sve import util as util
from typing import List
import copy
from six import StringIO
import dace.codegen.targets.sve.unparse
from dace import registry, symbolic, dtypes
from dace.codegen.targets import cpp as cpp
from dace.frontend.operations import detect_reduction_type
import dace.symbolic
from dace.codegen.targets.cpp import sym2cpp
from dace.sdfg import utils as sdutil
from dace.codegen.dispatcher import DefinedType
import copy
import numpy as np
from dace.codegen.targets.cpp import is_write_conflicted
from dace import data as data
from dace.frontend.operations import detect_reduction_type
import dace.codegen.targets


@dace.registry.autoregister_params(name='sve')
class SVECodeGen(TargetCodeGenerator):
    target_name = 'sve'
    title = 'Arm SVE'

    def add_header(self, function_stream: CodeIOStream):
        if self.has_generated_header:
            return
        self.has_generated_header = True

        function_stream.write('#include <arm_sve.h>\n')

        # TODO: Find this automatically at compile time
        function_stream.write(f'#define {util.REGISTER_BYTE_SIZE} 64\n')

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self.has_generated_header = False

        self.frame = frame_codegen
        self.dispatcher = frame_codegen._dispatcher
        self.dispatcher.register_map_dispatcher(dace.ScheduleType.SVE_Map, self)
        self.dispatcher.register_node_dispatcher(
            self, lambda state, sdfg, node: is_in_scope(
                state, sdfg, node, [dace.ScheduleType.SVE_Map]))

        cpu_storage = [
            dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal,
            dtypes.StorageType.Register, dtypes.StorageType.SVE_Register
        ]

        # This dispatcher is required to catch the allocation of Code->Code registers
        # because we want SVE registers instead of dace::vec<>'s.
        # In any other case it will call the default codegen.
        self.dispatcher.register_array_dispatcher(cpu_storage, self)
        self.dispatcher.register_copy_dispatcher(
            dtypes.StorageType.SVE_Register, dtypes.StorageType.CPU_Heap, None,
            self)

        self.cpu_codegen: dace.codegen.targets.CPUCodeGen = self.dispatcher.get_generic_node_dispatcher(
        )

    def get_generated_codeobjects(self):
        res = super().get_generated_codeobjects()
        print(res)
        return res

    def copy_memory(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                    src_node: nodes.Node, dst_node: nodes.Node,
                    edge: gr.MultiConnectorEdge[mm.Memlet],
                    function_stream: CodeIOStream,
                    callsite_stream: CodeIOStream) -> None:

        # Check whether it is a known reduction that is possible in SVE
        reduction_type = detect_reduction_type(edge.data.wcr)
        if reduction_type not in util.REDUCTION_TYPE_TO_SVE:
            raise util.NotSupportedError('Unsupported reduction in SVE')

        nc = not is_write_conflicted(dfg, edge)
        desc = edge.src.desc(sdfg)
        if not nc or not isinstance(desc.dtype,
                                    (dtypes.pointer, dtypes.vector)):
            # WCR on vectors works in two steps:
            # 1. Reduce the SVE register using SVE instructions into a scalar
            # 2. WCR the scalar to memory using DaCe functionality
            wcr = self.cpu_codegen.write_and_resolve_expr(sdfg,
                                                          edge.data,
                                                          not nc,
                                                          None,
                                                          '@',
                                                          dtype=desc.dtype)
            callsite_stream.write(
                wcr[:wcr.find('@')] +
                util.REDUCTION_TYPE_TO_SVE[reduction_type] +
                f'(svptrue_{util.TYPE_TO_SVE_SUFFIX[desc.dtype]}(), ' +
                src_node.label + wcr[wcr.find('@') + 1:] + ');')
            return
        else:
            ######################
            # Horizontal non-atomic reduction
            raise NotImplementedError()

        return super().copy_memory(sdfg, dfg, state_id, src_node, dst_node,
                                   edge, function_stream, callsite_stream)

    def generate_node(self, sdfg: SDFG, state: SDFGState, state_id: int,
                      node: nodes.Node, function_stream: CodeIOStream,
                      callsite_stream: CodeIOStream):
        self.add_header(function_stream)

        if not isinstance(node, nodes.Tasklet):
            return

        scope = util.get_sve_scope(sdfg, state, node)

        # Reset the stream variable mappings
        self.stream_associations = dict()
        self.wcr_associations = dict()

        callsite_stream.write('{')
        self.dispatcher.defined_vars.enter_scope(node)

        ##################
        # Generate tasklet

        # Inputs
        for edge in state.in_edges(node):
            self.generate_read(sdfg, state, scope.map, edge, callsite_stream)

        requires_wb = []

        # Temporary output registers
        for edge in state.out_edges(node):
            if self.generate_out_register(sdfg, state, edge, callsite_stream):
                requires_wb.append(edge)

        # Tasklet code
        self.unparse_tasklet(sdfg, state, state_id, node, function_stream,
                             callsite_stream)

        # Writeback from temporary registers to memory
        for edge in requires_wb:
            self.generate_writeback(sdfg, state, scope, edge, callsite_stream)

        self.dispatcher.defined_vars.exit_scope(node)
        callsite_stream.write('}')

    def generate_read(self, sdfg: SDFG, state: SDFGState, map: nodes.Map,
                      edge: graph.MultiConnectorEdge[mm.Memlet],
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
                code.write(
                    f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = {edge.data.data};'
                )
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
                        f'{dst_type} {dst_name} = {cpp.cpp_ptr_expr(sdfg, edge.data, None)};'
                    )
                elif util.is_vector(dst_type):
                    ##################
                    # Vector load

                    stride = edge.data.get_stride(sdfg, map)

                    # First part of the declaration is `type name`
                    load_lhs = '{} {}'.format(util.TYPE_TO_SVE[dst_type.type],
                                              dst_name)

                    # long long issue casting
                    ptr_cast = ''
                    if dst_type.type == np.int64:
                        ptr_cast = '(int64_t*) '
                    elif dst_type.type == np.uint64:
                        ptr_cast = '(uint64_t*) '

                    # Regular load and gather share the first arguments
                    load_args = '{}, {}'.format(
                        util.get_loop_predicate(sdfg, state,
                                                edge.dst), ptr_cast +
                        cpp.cpp_ptr_expr(sdfg, edge.data, DefinedType.Pointer))

                    if stride == 1:
                        code.write('{} = svld1({});'.format(
                            load_lhs, load_args))
                    else:
                        code.write(
                            '{} = svld1_gather_index({}, svindex_s{}(0, {}));'.
                            format(load_lhs, load_args,
                                   util.get_base_type(dst_type).bytes * 8,
                                   sym2cpp(stride)))
                else:
                    ##################
                    # Scalar read from array
                    code.write(
                        f'{dst_type} {dst_name} = {cpp.cpp_array_expr(sdfg, edge.data)};'
                    )
            elif isinstance(desc, data.Scalar):
                # Refer to shared variable
                src_type = desc.dtype
                if util.is_vector(src_type) and util.is_vector(dst_type):
                    # Directly read from shared vector register
                    code.write(
                        f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = {edge.data.data};'
                    )
                elif util.is_scalar(src_type) and util.is_scalar(dst_type):
                    # Directly read from shared scalar register
                    code.write(f'{dst_type} {dst_name} = {edge.data.data};')
                elif util.is_scalar(src_type) and util.is_vector(dst_type):
                    # Scalar broadcast from shared scalar register
                    code.write(
                        f'{util.TYPE_TO_SVE[dst_type.type]} {dst_name} = svdup_{util.TYPE_TO_SVE_SUFFIX[dst_type.type]}({edge.data.data});'
                    )
                else:
                    raise util.NotSupportedError(
                        'Unsupported Scalar->Code edge')
        else:
            raise util.NotSupportedError(
                'Only copy from Tasklets and AccessNodes is supported')

    def generate_out_register(self,
                              sdfg: SDFG,
                              state: SDFGState,
                              edge: graph.MultiConnectorEdge[mm.Memlet],
                              code: CodeIOStream,
                              use_data_name: bool = False) -> bool:
        """
            Responsible for generating temporary out registers in a Tasklet, given an outgoing edge.
            Returns `True` if a writeback of this register is needed.
        """
        if edge.src_conn is None:
            return

        dst_node = state.memlet_path(edge)[-1].dst

        src_type = edge.src.out_connectors[edge.src_conn]
        src_name = edge.src_conn

        if use_data_name:
            src_name = edge.data.data

        if isinstance(dst_node, nodes.AccessNode) and isinstance(
                dst_node.desc(sdfg), data.Stream):
            # Streams don't need writeback and are treated differently
            self.stream_associations[edge.src_conn] = (edge.data.data,
                                                       src_type.base_type)
            return False
        elif edge.data.wcr is not None:
            # WCR is addressed within the unparser to capture conditionals
            self.wcr_associations[edge.src_conn] = (dst_node, edge,
                                                    src_type.base_type)
            return False

        # Create temporary registers
        ctype = None
        if util.is_vector(src_type):
            ctype = util.TYPE_TO_SVE[src_type.type]
        elif util.is_scalar(src_type):
            ctype = src_type.ctype
        else:
            raise util.NotSupportedError(
                'Unsupported Code->Code edge (pointer)')

        self.dispatcher.defined_vars.add(src_name, DefinedType.Scalar, ctype)
        code.write(f'{ctype} {src_name};')

        return True

    def generate_writeback(self, sdfg: SDFG, state: SDFGState, map: nodes.Map,
                           edge: graph.MultiConnectorEdge[mm.Memlet],
                           code: CodeIOStream):
        """
            Responsible for generating code for a writeback in a Tasklet, given the outgoing edge.
            This is mainly taking the temporary register and writing it back.
        """
        if edge.src_conn is None:
            return

        dst_node = state.memlet_path(edge)[-1].dst

        src_type = edge.src.out_connectors[edge.src_conn]
        src_name = edge.src_conn

        if isinstance(dst_node, nodes.Tasklet):
            ##################
            # Code->Code edges
            dst_type = edge.dst.in_connectors[edge.dst_conn]

            if (util.is_vector(src_type) and util.is_vector(dst_type)) or (
                    util.is_scalar(src_type) and util.is_scalar(dst_type)):
                # Simply write back to shared register
                code.write(f'{edge.data.data} = {src_name};')
            elif util.is_scalar(src_type) and util.is_vector(dst_type):
                # Scalar broadcast to shared vector register
                code.write(
                    f'{edge.data.data} = svdup_{util.TYPE_TO_SVE_SUFFIX[dst_type.type]}({src_name});'
                )
            else:
                raise util.NotSupportedError('Unsupported Code->Code edge')
        elif isinstance(dst_node, nodes.AccessNode):
            ##################
            # Write to AccessNode
            desc = dst_node.desc(sdfg)
            if isinstance(desc, data.Array):
                ##################
                # Write into Array
                if util.is_pointer(src_type):
                    raise util.NotSupportedError('Unsupported writeback')
                elif util.is_vector(src_type):
                    ##################
                    # Scatter vector store into array

                    stride = edge.data.get_stride(sdfg, map)

                    # long long fix
                    ptr_cast = ''
                    if src_type.type == np.int64:
                        ptr_cast = '(int64_t*) '
                    elif src_type.type == np.uint64:
                        ptr_cast = '(uint64_t*) '

                    store_args = '{}, {}'.format(
                        util.get_loop_predicate(sdfg, state, edge.src),
                        ptr_cast +
                        cpp.cpp_ptr_expr(sdfg, edge.data, DefinedType.Pointer),
                    )

                    if stride == 1:
                        code.write(f'svst1({store_args}, {src_name});')
                    else:
                        code.write(
                            f'svst1_scatter_index({store_args}, svindex_s{util.get_base_type(src_type).bytes * 8}(0, {sym2cpp(stride)}), {src_name});'
                        )
                else:
                    ##################
                    # Scalar write into array
                    code.write(
                        f'{cpp.cpp_array_expr(sdfg, edge.data)} = {src_name};')
            elif isinstance(desc, data.Scalar):
                ##################
                # Write into Scalar
                if util.is_pointer(src_type):
                    raise util.NotSupportedError('Unsupported writeback')
                elif util.is_vector(src_type):
                    if util.is_vector(desc.dtype):
                        ##################
                        # Vector write into vector Scalar access node
                        code.write(f'{edge.data.data} = {src_name};')
                    else:
                        raise util.NotSupportedError('Unsupported writeback')
                else:
                    if util.is_vector(desc.dtype):
                        ##################
                        # Broadcast into scalar AccessNode
                        code.write(
                            f'{edge.data.data} = svdup_{util.TYPE_TO_SVE_SUFFIX[src_type]}({src_name});'
                        )
                    else:
                        ##################
                        # Scalar write into scalar AccessNode
                        code.write(f'{edge.data.data} = {src_name};')

        else:
            raise util.NotSupportedError(
                'Only writeback to Tasklets and AccessNodes is supported')

    def declare_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                      node: nodes.Node, nodedesc: data.Data,
                      global_stream: CodeIOStream,
                      declaration_stream: CodeIOStream) -> None:
        self.cpu_codegen.declare_array(sdfg, dfg, state_id, node, nodedesc,
                                       global_stream, declaration_stream)

    def allocate_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                       node: nodes.Node, nodedesc: data.Data,
                       global_stream: CodeIOStream,
                       declaration_stream: CodeIOStream,
                       allocation_stream: CodeIOStream) -> None:
        if nodedesc.storage == dtypes.StorageType.SVE_Register:
            sve_type = util.TYPE_TO_SVE[nodedesc.dtype]
            self.dispatcher.defined_vars.add(node.data, DefinedType.Scalar,
                                             sve_type)
            return

        if util.get_sve_scope(sdfg, dfg, node) is not None and isinstance(
                nodedesc, data.Scalar) and isinstance(nodedesc.dtype,
                                                      dtypes.vector):
            # Special allocation if vector Code->Code register in SVE scope
            # We prevent dace::vec<>'s and allocate SVE registers instead
            if self.dispatcher.defined_vars.has(node.data):
                sve_type = util.TYPE_TO_SVE[nodedesc.dtype.vtype]
                self.dispatcher.defined_vars.add(node.data, DefinedType.Scalar,
                                                 sve_type)
                declaration_stream.write(f'{sve_type} {node.data};')
            return

        self.cpu_codegen.allocate_array(sdfg, dfg, state_id, node, nodedesc,
                                        global_stream, declaration_stream,
                                        allocation_stream)

    def deallocate_array(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                         node: nodes.Node, nodedesc: data.Data,
                         function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        return self.cpu_codegen.deallocate_array(sdfg, dfg, state_id, node,
                                                 nodedesc, function_stream,
                                                 callsite_stream)

    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        entry_node = scope.source_nodes()[0]
        current_map = entry_node.map
        self.current_map = current_map

        if len(current_map.params) > 1:
            raise util.NotSupportedError('SVE map must be one dimensional')

        loop_types = list(
            set([util.get_base_type(sdfg.arrays[a].dtype)
                 for a in sdfg.arrays]))

        # Edge case if no arrays are used
        loop_type = loop_types[0] if len(loop_types) > 0 else dace.int64

        ltype_size = loop_type.bytes

        long_type = copy.copy(dace.int64)
        long_type.ctype = 'int64_t'

        self.counter_type = {
            1: dace.int8,
            2: dace.int16,
            4: dace.int32,
            8: long_type
        }[ltype_size]

        callsite_stream.write('{')
        self.dispatcher.defined_vars.enter_scope(scope)

        # Define all dynamic input connectors of the map entry
        state_dfg = sdfg.node(state_id)
        for e in dace.sdfg.dynamic_map_inputs(state_dfg, entry_node):
            if e.data.data != e.dst_conn:
                callsite_stream.write(
                    self.cpu_codegen.memlet_definition(
                        sdfg, e.data, False, e.dst_conn,
                        e.dst.in_connectors[e.dst_conn]), sdfg, state_id,
                    entry_node)

        param = current_map.params[0]
        rng = current_map.range[0]
        begin, end, stride = (sym2cpp(r) for r in rng)

        # Generate the SVE loop header
        # The name of our loop predicate is always __pg_{param}
        self.dispatcher.defined_vars.add('__pg_' + param, DefinedType.Scalar,
                                         'svbool_t')

        # Declare our counting variable (e.g. i) and precompute the loop predicate for our range
        callsite_stream.write(f'{self.counter_type} {param} = {begin};')

        end_param = f'__{param}_to'
        callsite_stream.write(f'{self.counter_type} {end_param} = {end};')

        callsite_stream.write(
            f'svbool_t __pg_{param} = svwhilele_b{ltype_size * 8}({param}, {end_param});'
        )

        # Test for the predicate
        callsite_stream.write(
            f'while(svptest_any(svptrue_b{ltype_size * 8}(), __pg_{param})) {{')

        # Allocate scope related memory
        for node, _ in scope.all_nodes_recursive():
            if isinstance(node, nodes.Tasklet):
                # Create empty shared registers for outputs into other tasklets
                for edge in state_dfg.out_edges(node):
                    if isinstance(edge.dst, dace.nodes.Tasklet):
                        self.generate_out_register(sdfg, state_dfg, edge,
                                                   callsite_stream, True)

        # Dispatch the subgraph generation
        self.dispatcher.dispatch_subgraph(sdfg,
                                          scope,
                                          state_id,
                                          function_stream,
                                          callsite_stream,
                                          skip_entry_node=True,
                                          skip_exit_node=True)

        # Increase the counting variable (according to the number of processed elements)
        size_letter = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}[ltype_size]
        callsite_stream.write(f'{param} += svcnt{size_letter}() * {stride};')

        # Then recompute the loop predicate
        callsite_stream.write(
            f'__pg_{param} = svwhilele_b{ltype_size * 8}({param}, {end_param});'
        )

        callsite_stream.write('}')

        self.dispatcher.defined_vars.exit_scope(scope)
        callsite_stream.write('}')

    def unparse_tasklet(self, sdfg: SDFG, dfg: state.StateSubgraphView,
                        state_id: int, node: nodes.Node,
                        function_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):
        state_dfg: SDFGState = sdfg.nodes()[state_id]

        callsite_stream.write('\n///////////////////')
        callsite_stream.write(f'// Tasklet code ({node.label})')

        # Determine all defined symbols for the Unparser (for inference)

        # Constants and other defined symbols
        defined_symbols = state_dfg.symbols_defined_at(node)
        defined_symbols.update({
            k: v.dtype if hasattr(v, 'dtype') else dtypes.typeclass(type(v))
            for k, v in sdfg.constants.items()
        })

        # All memlets of that node
        memlets = {}
        for edge in state_dfg.all_edges(node):
            u, uconn, v, vconn, _ = edge
            if u == node and uconn in u.out_connectors:
                defined_symbols.update({uconn: u.out_connectors[uconn]})
            elif v == node and vconn in v.in_connectors:
                defined_symbols.update({vconn: v.in_connectors[vconn]})

        body = node.code.code
        for stmt in body:
            stmt = copy.deepcopy(stmt)
            result = StringIO()
            dace.codegen.targets.sve.unparse.SVEUnparser(
                sdfg, dfg, self.current_map, self.cpu_codegen,
                stmt, result, body, memlets,
                util.get_loop_predicate(sdfg, dfg, node), self.counter_type,
                defined_symbols, self.stream_associations,
                self.wcr_associations)
            callsite_stream.write(result.getvalue(), sdfg, state_id, node)

        callsite_stream.write('///////////////////\n\n')
