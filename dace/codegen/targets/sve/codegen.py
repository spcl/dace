# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Code generation: This module is responsible for converting an SDFG into SVE code.
"""

import dace
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
import dace.codegen.targets.sve.util as util
import copy
from six import StringIO
import dace.codegen.targets.sve.unparse
from dace import registry, symbolic, dtypes
import dace.codegen.targets.cpp as cpp
from dace.frontend.operations import detect_reduction_type
import dace.symbolic
from dace.codegen.targets.cpp import sym2cpp
from dace.sdfg import utils as sdutil
from dace.codegen.dispatcher import DefinedType
import copy
import numpy as np


def contains_any_sve(sdfg: SDFG):
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node,
                      nodes.Map) and node.schedule == dace.ScheduleType.SVE_Map:
            return True
    return False


@dace.registry.autoregister_params(name='sve')
class SVECodeGenerator(TargetCodeGenerator):
    target_name = 'armv8'
    title = 'sve'
    language = 'cpp'

    def get_load_stride(self, sdfg: SDFG, state: SDFGState, node: nodes.Node,
                        memlet: dace.Memlet) -> symbolic.SymExpr:
        """Determines the stride of a load/store based on:
            - The memlet subset
            - The array strides
            - The involved SVE loop stride"""

        scope = util.get_sve_scope(sdfg, state, node)
        if scope is None:
            raise NotImplementedError('Not in an SVE scope')

        sve_param = scope.map.params[-1]
        sve_range = scope.map.range[-1]
        sve_sym = dace.symbolic.symbol(sve_param)

        array = sdfg.arrays[memlet.data]

        # 1. Flatten the subset to a 1D-offset (using the array strides)
        offset_1 = memlet.subset.at([0] * len(array.strides), array.strides)

        if not offset_1.has(sve_sym):
            raise util.NotSupportedError("SVE param does not occur in subset")

        # 2. Replace the SVE loop param with its next (possibly strided) value
        offset_2 = offset_1.subs(sve_sym, sve_sym + sve_range[2])

        # 3. The load stride is the difference between both
        stride = (offset_2 - offset_1).simplify()

        return stride

    def add_header(self, function_stream: CodeIOStream):
        if self.has_generated_header:
            return
        self.has_generated_header = True

        function_stream.write('#include <arm_sve.h>\n')

        # TODO: Find this automatically at compile time
        function_stream.write(f'#define {util.REGISTER_BYTE_SIZE} 64\n')

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        dace.SCOPEDEFAULT_SCHEDULE[
            dace.ScheduleType.SVE_Map] = dace.ScheduleType.Sequential
        dace.SCOPEDEFAULT_STORAGE[
            dace.ScheduleType.SVE_Map] = dace.StorageType.CPU_Heap
        self.has_generated_header = False

        self.frame = frame_codegen
        self.dispatcher = frame_codegen._dispatcher
        self.dispatcher.register_map_dispatcher(dace.ScheduleType.SVE_Map, self)
        self.dispatcher.register_node_dispatcher(
            self, lambda state, sdfg, node: is_in_scope(
                state, sdfg, node, [dace.ScheduleType.SVE_Map]))
        #self.dispatcher.register_state_dispatcher(self, lambda sdfg, state: contains_any_sve(sdfg))

        self.cpu_codegen = self.dispatcher.get_generic_node_dispatcher()
        self.state_gen = self.dispatcher.get_generic_state_dispatcher()

        for src_storage, dst_storage in itertools.product(
                dtypes.StorageType, dtypes.StorageType):
            self.dispatcher.register_copy_dispatcher(src_storage, dst_storage,
                                                     dace.ScheduleType.SVE_Map,
                                                     self)

    def create_empty_definition(self,
                                conn: dace.typeclass,
                                edge: gr.MultiConnectorEdge[mm.Memlet],
                                callsite_stream: CodeIOStream,
                                output: bool = False,
                                is_code_code: bool = False):
        """ Creates a simple variable definition `type name;`, which works for both vectors and regular data types. """

        var_name = None
        var_type = None
        var_ctype = None

        if output:
            var_name = edge.dst_conn
        else:
            var_name = edge.src_conn

        if is_code_code:
            # For edges between Tasklets (Code->Code), we use the data as name because these registers are temporary and shared
            var_name = edge.data.data

        if isinstance(conn, dtypes.vector):
            # Creates an SVE register

            if conn.type not in util.TYPE_TO_SVE:
                raise util.NotSupportedError('Data type not supported')

            # In case of a WCR, we must initialize it with the identity value.
            # This is to prevent cases in a conditional WCR, where we don't write and it is filled with garbage.
            # Currently, the initial is 0, because product reduction is not supported in SVE.
            init_str = ''
            if edge.data.wcr:
                init_str = ' = {}(0)'.format(util.instr('dup', type=conn.type))

            var_type = conn.type
            var_ctype = util.TYPE_TO_SVE[var_type]

            callsite_stream.write('{} {}{};'.format(var_ctype, var_name,
                                                    init_str))
        else:
            raise NotImplementedError(
                f'Output into scalar or pointer is not supported ({var_name})')

        self.dispatcher.defined_vars.add(var_name, var_type, var_ctype)

    def generate_node(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                      node: nodes.Node, function_stream: CodeIOStream,
                      callsite_stream: CodeIOStream):
        self.add_header(function_stream)

        # Reset the mappings
        self.stream_associations = dict()

        # Create empty shared registers for outputs into other tasklets
        for edge in dfg.out_edges(node):
            if isinstance(edge.dst, dace.nodes.Tasklet):
                self.create_empty_definition(node.out_connectors[edge.src_conn],
                                             edge,
                                             callsite_stream,
                                             is_code_code=True)

        callsite_stream.write('{')

        # Create input registers (and fill them accordingly)
        for edge in dfg.in_edges(node):
            if isinstance(edge.src, nodes.Tasklet):
                # Copy from tasklet is treated differently (because it involves a shared register)
                # Changing src_node to a Tasklet will trigger a different copy
                self.dispatcher.dispatch_copy(edge.src, node, edge, sdfg, dfg,
                                              state_id, function_stream,
                                              callsite_stream)
            else:
                # Copy from some array (or stream)
                src_node = find_input_arraynode(dfg, edge)
                self.dispatcher.dispatch_copy(src_node, node, edge, sdfg, dfg,
                                              state_id, function_stream,
                                              callsite_stream)

        # Keep track of (edge, node) that need a writeback
        requires_wb = []

        # Create output registers
        for edge in dfg.out_edges(node):
            if isinstance(edge.dst, nodes.Tasklet):
                # Output into another tasklet again is treated differently similar to the input registers
                self.dispatcher.dispatch_output_definition(
                    node, edge.dst, edge, sdfg, dfg, state_id, function_stream,
                    callsite_stream)

                requires_wb.append((edge, node))
            else:
                dst_node = find_output_arraynode(dfg, edge)
                dst_desc = dst_node.desc(sdfg)

                # Streams neither need an output register (pushes can happen at any time in a tasklet) nor a writeback
                if isinstance(dst_desc, dace.data.Stream):
                    # We flag the name of the stream variable
                    self.stream_associations[edge.src_conn] = (dst_node.data,
                                                               dst_desc.dtype)
                else:
                    self.dispatcher.dispatch_output_definition(
                        node, dst_node, edge, sdfg, dfg, state_id,
                        function_stream, callsite_stream)

                    requires_wb.append((edge, dst_node))

        # Generate tasklet code
        if isinstance(node, nodes.Tasklet):
            self.unparse_tasklet(sdfg, dfg, state_id, node, function_stream,
                                 callsite_stream)

        # Write back output registers to memory
        for edge, dst_node in requires_wb:
            self.write_back(sdfg, dfg, state_id, node, dst_node, edge,
                            function_stream, callsite_stream)

        callsite_stream.write('}')

    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        entry_node = scope.source_nodes()[0]

        loop_type = list(set([sdfg.arrays[a].dtype for a in sdfg.arrays]))[0]
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

        # Define all input connectors of the map entry
        state_dfg = sdfg.node(state_id)
        for e in dace.sdfg.dynamic_map_inputs(state_dfg, entry_node):
            if e.data.data != e.dst_conn:
                callsite_stream.write(
                    self.cpu_codegen.memlet_definition(
                        sdfg, e.data, False, e.dst_conn,
                        e.dst.in_connectors[e.dst_conn]), sdfg, state_id,
                    entry_node)

        # We only create an SVE do-while in the innermost loop
        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            begin, end, stride = (sym2cpp(r) for r in rng)

            self.dispatcher.defined_vars.enter_scope(sdfg)

            # Check whether we are in the innermost loop
            if param != entry_node.map.params[-1]:
                # Default C++ for-loop
                callsite_stream.write(
                    f'for(auto {param} = {begin}; {param} <= {end}; {param} += {stride}) {{'
                )
            else:
                # Generate the SVE loop header

                # The name of our loop predicate is always __pg_{param}
                self.dispatcher.defined_vars.add('__pg_' + param,
                                                 DefinedType.Scalar, 'svbool_t')

                # Declare our counting variable (e.g. i) and precompute the loop predicate for our range
                callsite_stream.write(
                    f'''{self.counter_type} {param} = {begin};
                    svbool_t __pg_{param} = svwhilele_b{ltype_size * 8}({param}, ({self.counter_type}) {end});
                    do {{''', sdfg, state_id, entry_node)

        # Dispatch the subgraph generation
        self.dispatcher.dispatch_subgraph(sdfg,
                                          scope,
                                          state_id,
                                          function_stream,
                                          callsite_stream,
                                          skip_entry_node=True,
                                          skip_exit_node=True)

        # Close the loops from above (in reverse)
        for param, rng in zip(reversed(entry_node.map.params),
                              reversed(entry_node.map.range)):
            # The innermost loop is SVE and needs a special while-footer, otherwise we just add the closing bracket
            if param != entry_node.map.params[-1]:
                # Close the default C++ for-loop
                callsite_stream.write('}')
            else:
                # Generate the SVE loop footer

                _, end, stride = (sym2cpp(r) for r in rng)

                # Increase the counting variable (according to the number of processed elements)
                # Then recompute the loop predicate and test for it
                callsite_stream.write(
                    f'''{param} += svcntp_b{ltype_size * 8}(__pg_{param}, __pg_{param}) * {stride};
                    __pg_{param} = svwhilele_b{ltype_size * 8}({param}, ({self.counter_type}) {end});
                    }} while(svptest_any(svptrue_b{ltype_size * 8}(), __pg_{param}));''',
                    sdfg, state_id, entry_node)

            self.dispatcher.defined_vars.exit_scope(sdfg)

        callsite_stream.write('}')

    def copy_memory(self, sdfg: SDFG, dfg: SDFGState, state_id: int,
                    src_node: nodes.Node, dst_node: nodes.Node,
                    edge: gr.MultiConnectorEdge[mm.Memlet],
                    function_stream: CodeIOStream,
                    callsite_stream: CodeIOStream):
        # We should always be in an SVE scope
        scope = util.get_sve_scope(sdfg, dfg, dst_node)
        if scope is None:
            raise NotImplementedError('Not in an SVE scope')

        in_conn = dst_node.in_connectors[edge.dst_conn]

        if isinstance(src_node, dace.nodes.Tasklet):
            # Copy from tasklet is just copying the shared register
            # Use defined_vars to get the C++ type of the shared register
            callsite_stream.write(
                f'{self.dispatcher.defined_vars.get(edge.data.data)[1]} {edge.dst_conn} = {edge.data.data};'
            )
            return

        if not isinstance(src_node, dace.nodes.AccessNode):
            raise util.NotSupportedError(
                'Copy neither from Tasklet nor AccessNode')

        src_desc = src_node.desc(sdfg)

        if isinstance(src_desc, dace.data.Stream):
            # A copy from a stream will trigger a vector pop
            raise NotImplementedError()

            # FIXME: Issue when we can pop different amounts of data!
            # If we limit to the smallest amount, certain data will be lost (never processed)
            """
            # SVE register where the stream will be popped to
            self.create_empty_definition(in_conn, edge, callsite_stream, output=True)

            var_name = edge.dst_conn

            callsite_stream.write(
                f'{util.TYPE_TO_SVE[in_conn.type]} {var_name};')

            callsite_stream.write('{')
            callsite_stream.write('// Stream pop')

            # Pop into local buffer
            # 256 // in_conn.vtype.bytes
            n_vec = f'{util.REGISTER_BYTE_SIZE} / {in_conn.vtype.bytes}'
            callsite_stream.write(f'{in_conn.vtype.ctype} __tmp[{n_vec}];')
            callsite_stream.write(
                f'size_t __cnt = {edge.data.data}.pop_try(__tmp, {n_vec});')

            # Limit the loop predicate
            loop_pred = util.get_loop_predicate(sdfg, dfg, dst_node)
            callsite_stream.write(
                f'{loop_pred} = svand_z({loop_pred}, {loop_pred}, svwhilelt_b{in_conn.vtype.bytes * 8}(0ll, __cnt));')

            # Transfer to register
            callsite_stream.write(f'{var_name} = svld1({loop_pred}, __tmp);')

            callsite_stream.write('}')
            """
            return

        if isinstance(in_conn, dtypes.vector):
            # Copy from vector, so we can use svld

            if in_conn.type not in util.TYPE_TO_SVE:
                raise NotImplementedError(
                    f'Data type {in_conn.type} not supported')

            self.dispatcher.defined_vars.add(edge.dst_conn, dtypes.vector,
                                             in_conn.ctype)

            # Determine the stride of the load and use a gather if applicable
            stride = self.get_load_stride(sdfg, dfg, dst_node, edge.data)

            # First part of the declaration is `type name`
            load_lhs = '{} {}'.format(util.TYPE_TO_SVE[in_conn.type],
                                      edge.dst_conn)

            ptr_cast = ''
            if in_conn.type == np.int64:
                ptr_cast = '(int64_t*) '
            elif in_conn.type == np.uint64:
                ptr_cast = '(uint64_t*) '

            # Regular load and gather share the first arguments
            load_args = '{}, {}'.format(
                util.get_loop_predicate(sdfg, dfg, dst_node), ptr_cast +
                cpp.cpp_ptr_expr(sdfg, edge.data, DefinedType.Pointer))

            if stride == 1:
                callsite_stream.write('{} = svld1({});'.format(
                    load_lhs, load_args))
            else:
                callsite_stream.write(
                    '{} = svld1_gather_index({}, svindex_s{}(0, {}));'.format(
                        load_lhs, load_args,
                        util.get_base_type(in_conn).bytes * 8, sym2cpp(stride)))
        else:
            # Any other copy (e.g. pointer or scalar) is handled by the default CPU codegen
            self.cpu_codegen.copy_memory(sdfg, dfg, state_id, src_node,
                                         dst_node, edge, function_stream,
                                         callsite_stream)

    def define_out_memlet(self, sdfg: SDFG, dfg: state.StateSubgraphView,
                          state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge: graph.MultiConnectorEdge,
                          function_stream: CodeIOStream,
                          callsite_stream: CodeIOStream):
        scope = util.get_sve_scope(sdfg, dfg, src_node)
        if scope is None:
            raise NotImplementedError('Not in an SVE scope')

        self.create_empty_definition(src_node.out_connectors[edge.src_conn],
                                     edge, callsite_stream)

    def write_back(self, sdfg: SDFG, dfg: state.StateSubgraphView,
                   state_id: int, src_node: nodes.Node, dst_node: nodes.Node,
                   edge: graph.MultiConnectorEdge,
                   function_stream: CodeIOStream,
                   callsite_stream: CodeIOStream):
        scope = util.get_sve_scope(sdfg, dfg, src_node)
        if scope is None:
            raise NotImplementedError('Not in an SVE scope')

        out_conn = src_node.out_connectors[edge.src_conn]
        if out_conn.type not in util.TYPE_TO_SVE:
            raise NotImplementedError(
                f'Data type {out_conn.type} not supported')

        if edge.data.wcr is None:
            # No WCR required

            if isinstance(dst_node, dace.nodes.Tasklet):
                # Writeback into a tasklet is just writing into the shared register
                callsite_stream.write(f'{edge.data.data} = {edge.src_conn};')
                return

            if isinstance(out_conn, dtypes.vector):
                # If no WCR, we can directly store the vector (SVE register) in memory
                # Determine the stride of the store and use a scatter load if applicable

                stride = self.get_load_stride(sdfg, dfg, src_node, edge.data)

                ptr_cast = ''
                if out_conn.type == np.int64:
                    ptr_cast = '(int64_t*) '
                elif out_conn.type == np.uint64:
                    ptr_cast = '(uint64_t*) '

                store_args = '{}, {}'.format(
                    util.get_loop_predicate(sdfg, dfg, src_node),
                    ptr_cast +
                    cpp.cpp_ptr_expr(sdfg, edge.data, DefinedType.Pointer),
                )

                if stride == 1:
                    callsite_stream.write(
                        f'svst1({store_args}, {edge.src_conn});')
                else:
                    callsite_stream.write(
                        f'svst1_scatter_index({store_args}, svindex_s{util.get_base_type(out_conn).bytes * 8}(0, {sym2cpp(stride)}), {edge.src_conn});'
                    )
            else:
                raise NotImplementedError('Writeback into non-vector')
        else:
            # TODO: Check what are we WCR'ing in?

            # Since we have WCR, we must determine a suitable SVE reduce instruction
            # Check whether it is a known reduction that is possible in SVE
            reduction_type = detect_reduction_type(edge.data.wcr)
            if reduction_type not in util.REDUCTION_TYPE_TO_SVE:
                raise util.NotSupportedError('Unsupported reduction in SVE')

            # If the memlet contains the innermost SVE param, we have a problem, because
            # SVE doesn't support WCR stores. This would require unrolling the loop.
            if scope.params[-1] in edge.data.free_symbols:
                raise util.NotSupportedError(
                    'SVE loop param used in WCR memlet')

            # WCR on vectors works in two steps:
            # 1. Reduce the SVE register using SVE instructions into a scalar
            # 2. WCR the scalar to memory using DaCe functionality

            sve_reduction = '{}({}, {})'.format(
                util.REDUCTION_TYPE_TO_SVE[reduction_type],
                util.get_loop_predicate(sdfg, dfg, src_node), edge.src_conn)

            ptr_cast = ''
            if out_conn.type == np.int64:
                ptr_cast = '(long long*) '
            elif out_conn.type == np.uint64:
                ptr_cast = '(unsigned long long*) '

            wcr_expr = self.cpu_codegen.write_and_resolve_expr(
                sdfg,
                edge.data,
                edge.data.wcr_nonatomic,
                None,
                ptr_cast + sve_reduction,
                dtype=out_conn.vtype)

            callsite_stream.write(wcr_expr + ';')

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
                sdfg, stmt, result, body, memlets,
                util.get_loop_predicate(sdfg, dfg, node), self.counter_type,
                defined_symbols, self.stream_associations)
            callsite_stream.write(result.getvalue(), sdfg, state_id, node)

        callsite_stream.write('///////////////////\n\n')
