import ast
import copy
import functools
import itertools
import sympy as sp
from six import StringIO

from dace.codegen import cppunparse

import dace
from dace.config import Config
from dace.frontend import operations
from dace import data, subsets, symbolic, types, memlet as mmlt
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import framecode
from dace.codegen.targets.target import (TargetCodeGenerator, make_absolute,
                                         DefinedType)
from dace.graph import nodes, nxutil
from dace.sdfg import ScopeSubgraphView, SDFG, scope_contains_scope, find_input_arraynode, find_output_arraynode, is_devicelevel

from dace.frontend.python.astutils import ExtNodeTransformer, rname, unparse
from dace.properties import LambdaProperty

from dace.codegen.instrumentation.perfsettings import PerfSettings, PerfUtils, PerfMetaInfo, PerfMetaInfoStatic

_REDUCTION_TYPE_TO_OPENMP = {
    types.ReductionType.Max: 'max',
    types.ReductionType.Min: 'min',
    types.ReductionType.Sum: '+',
    types.ReductionType.Product: '*',
    types.ReductionType.Bitwise_And: '&',
    types.ReductionType.Logical_And: '&&',
    types.ReductionType.Bitwise_Or: '|',
    types.ReductionType.Logical_Or: '||',
    types.ReductionType.Bitwise_Xor: '^',
}


class CPUCodeGen(TargetCodeGenerator):
    """ SDFG CPU code generator. """

    title = 'CPU'
    target_name = 'cpu'
    language = 'cpp'

    def __init__(self, frame_codegen, sdfg):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self._locals = cppunparse.CPPLocals()
        # Scope depth (for use of the 'auto' keyword when
        # defining locals)
        self._ldepth = 0

        # FIXME: this allows other code generators to change the CPU
        # behavior to assume that arrays point to packed types, thus dividing
        # all addresess by the vector length.
        self._packed_types = False

        # Keep track of traversed nodes
        self._generated_nodes = set()
        self._allocated_arrays = set()
        # Keeps track of generated connectors, so we know how to access them in
        # nested scopes
        for name, arg_type in sdfg.arglist().items():
            if (isinstance(arg_type, dace.data.Scalar)
                    or isinstance(arg_type, dace.types.typeclass)):
                self._dispatcher.defined_vars.add(name, DefinedType.Scalar)
            elif isinstance(arg_type, dace.data.Array):
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            elif isinstance(arg_type, dace.data.Stream):
                if arg_type.is_stream_array():
                    self._dispatcher.defined_vars.add(name,
                                                      DefinedType.StreamArray)
                else:
                    self._dispatcher.defined_vars.add(name, DefinedType.Stream)
            else:
                raise TypeError("Unrecognized argument type: {}".format(
                    type(arg_type).__name__))

        # Register dispatchers
        dispatcher.register_node_dispatcher(self)
        dispatcher.register_map_dispatcher(
            [types.ScheduleType.CPU_Multicore, types.ScheduleType.Sequential],
            self)

        cpu_storage = [
            types.StorageType.CPU_Heap, types.StorageType.CPU_Pinned,
            types.StorageType.CPU_Stack, types.StorageType.Register
        ]
        dispatcher.register_array_dispatcher(cpu_storage, self)

        # Register CPU copies (all internal pairs)
        for src_storage, dst_storage in itertools.product(
                cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None,
                                                self)

    @staticmethod
    def cmake_options():
        compiler = make_absolute(Config.get("compiler", "cpu", "executable"))
        flags = Config.get("compiler", "cpu", "args")
        flags += Config.get("compiler", "cpu", "additional_args")

        # Args for vectorization output
        if PerfSettings.perf_enable_vectorization_analysis():
            flags += " -fopt-info-vec-optimized-missed=vecreport.txt "

        options = [
            "-DCMAKE_CXX_COMPILER=\"{}\"".format(compiler),
            "-DCMAKE_CXX_FLAGS=\"{}\"".format(flags),
        ]
        return options

    def get_generated_codeobjects(self):
        # CPU target generates inline code
        return []

    @property
    def has_initializer(self):
        return False

    @property
    def has_finalizer(self):
        return False

    def generate_scope(self, sdfg: SDFG, dfg_scope: ScopeSubgraphView,
                       state_id, function_stream, callsite_stream):
        entry_node = dfg_scope.source_nodes()[0]
        presynchronize_streams(sdfg, dfg_scope, state_id, entry_node,
                               callsite_stream)

        self.generate_node(sdfg, dfg_scope, state_id, entry_node,
                           function_stream, callsite_stream)
        self._dispatcher.dispatch_subgraph(
            sdfg,
            dfg_scope,
            state_id,
            function_stream,
            callsite_stream,
            skip_entry_node=True)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        # Dynamically obtain node generator according to class name
        gen = getattr(self, '_generate_' + type(node).__name__)

        gen(sdfg, dfg, state_id, node, function_stream, callsite_stream)

        # Mark node as "generated"
        self._generated_nodes.add(node)

        self._locals.clear_scope(self._ldepth + 1)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)
        if ((state_id, node.data) in self._allocated_arrays
                or (None, node.data) in self._allocated_arrays
                or nodedesc.transient == False):
            return
        self._allocated_arrays.add((state_id, node.data))

        # Compute array size
        arrsize = ' * '.join([sym2cpp(s) for s in nodedesc.strides])

        if isinstance(nodedesc, data.Scalar):
            callsite_stream.write("%s %s;\n" % (nodedesc.dtype.ctype, name),
                                  sdfg, state_id, node)
            self._dispatcher.defined_vars.add(name, DefinedType.Scalar)
        elif isinstance(nodedesc, data.Stream):
            ###################################################################
            # Stream directly connected to an array

            if is_array_stream_view(sdfg, dfg, node):
                if state_id is None:
                    raise SyntaxError(
                        'Stream-view of array may not be defined '
                        'in more than one state')

                arrnode = sdfg.arrays[nodedesc.sink]
                state = sdfg.nodes()[state_id]
                edges = state.out_edges(node)
                if len(edges) > 1:
                    raise NotImplementedError('Cannot handle streams writing '
                                              'to multiple arrays.')

                memlet_path = state.memlet_path(edges[0])
                # Allocate the array before its stream view, if necessary
                self.allocate_array(sdfg, dfg, state_id, memlet_path[-1].dst,
                                    function_stream, callsite_stream)

                array_expr = self.copy_expr(sdfg, nodedesc.sink, edges[0].data)
                threadlocal = ''
                threadlocal_stores = [
                    types.StorageType.CPU_Stack, types.StorageType.Register
                ]
                if (sdfg.arrays[nodedesc.sink].storage in threadlocal_stores
                        or nodedesc.storage in threadlocal_stores):
                    threadlocal = 'Threadlocal'
                callsite_stream.write(
                    'dace::ArrayStreamView%s<%s> %s (%s);\n' %
                    (threadlocal, arrnode.dtype.ctype, name, array_expr), sdfg,
                    state_id, node)
                self._dispatcher.defined_vars.add(name, DefinedType.Stream)
                return

            ###################################################################
            # Regular stream

            dtype = "dace::vec<{}, {}>".format(nodedesc.dtype.ctype,
                                               sym2cpp(nodedesc.veclen))

            if nodedesc.buffer_size != 0:
                definition = "dace::Stream<{}> {}({});".format(
                    dtype, name, nodedesc.buffer_size)
            else:
                definition = "dace::Stream<{}> {};".format(dtype, name)

            callsite_stream.write(definition, sdfg, state_id, node)
            self._dispatcher.defined_vars.add(name, DefinedType.Stream)

        elif (nodedesc.storage == types.StorageType.CPU_Heap
              or nodedesc.storage == types.StorageType.Immaterial
              ):  # TODO: immaterial arrays should not allocate memory
            callsite_stream.write(
                "%s *%s = new %s DACE_ALIGN(64)[%s];\n" %
                (nodedesc.dtype.ctype, name, nodedesc.dtype.ctype, arrsize),
                sdfg, state_id, node)
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            if node.setzero:
                callsite_stream.write('memset(%s, 0, sizeof(%s)*%s);' %
                                      (name, nodedesc.dtype.ctype, arrsize))
            return
        elif (nodedesc.storage == types.StorageType.CPU_Stack
              or nodedesc.storage == types.StorageType.Register):
            if node.setzero:
                callsite_stream.write(
                    "%s %s[%s]  DACE_ALIGN(64) = {0};\n" %
                    (nodedesc.dtype.ctype, name, arrsize), sdfg, state_id,
                    node)
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
                return
            callsite_stream.write(
                "%s %s[%s]  DACE_ALIGN(64);\n" %
                (nodedesc.dtype.ctype, name, arrsize), sdfg, state_id, node)
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            return
        else:
            raise NotImplementedError('Unimplemented storage type ' +
                                      str(nodedesc.storage))

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        if isinstance(dfg, SDFG):
            result = StringIO()
            for sid, state in enumerate(dfg.nodes()):
                if node in state.nodes():
                    self.initialize_array(sdfg, state, sid, node,
                                          function_stream, callsite_stream)
                    break
            return

        parent_node = dfg.scope_dict()[node]
        nodedesc = node.desc(sdfg)
        name = node.data

        # Traverse the DFG, looking for WCR with an identity element
        def traverse(u, uconn, v, vconn, d):
            if d.wcr:
                if d.data == name:
                    if d.wcr_identity is not None:
                        return d.wcr_identity
            return None

        identity = None
        if parent_node is not None:
            for u, uconn, v, vconn, d, s in nxutil.traverse_sdfg_scope(
                    dfg, parent_node):
                identity = traverse(u, uconn, v, vconn, d)
                if identity is not None: break
        else:
            for u, uconn, v, vconn, d in dfg.edges():
                identity = traverse(u, uconn, v, vconn, d)
                if identity is not None: break

        if identity is None:
            return

        # If we should generate an initialization expression
        if isinstance(nodedesc, data.Scalar):
            callsite_stream.write('%s = %s;\n' % (name, sym2cpp(identity)),
                                  sdfg, state_id, node)
            return

        params = [name, sym2cpp(identity)]
        shape = [sym2cpp(s) for s in nodedesc.shape]
        params.append(' * '.join(shape))

        # Faster
        if identity == 0:
            params[-1] += ' * sizeof(%s[0])' % name
            callsite_stream.write('memset(%s);\n' % (', '.join(params)), sdfg,
                                  state_id, node)
            return

        callsite_stream.write('dace::InitArray(%s);\n' % (', '.join(params)),
                              sdfg, state_id, node)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        nodedesc = node.desc(sdfg)
        if isinstance(nodedesc, data.Scalar):
            return
        elif isinstance(nodedesc, data.Stream):
            return
        elif nodedesc.storage == types.StorageType.CPU_Heap:
            callsite_stream.write("delete[] %s;\n" % node.data, sdfg, state_id,
                                  node)
        else:
            return

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):
        if isinstance(src_node, nodes.Tasklet):
            src_storage = types.StorageType.Register
            try:
                src_parent = dfg.scope_dict()[src_node]
            except KeyError:
                src_parent = None
            dst_schedule = (None
                            if src_parent is None else src_parent.map.schedule)
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = types.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.scope_dict()[dst_node]
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = sdfg.nodes()[state_id]

        # Emit actual copy
        self._emit_copy(sdfg, state_id, src_node, src_storage, dst_node,
                        dst_storage, dst_schedule, edge, state_dfg,
                        callsite_stream)

    def _emit_copy(self, sdfg, state_id, src_node, src_storage, dst_node,
                   dst_storage, dst_schedule, edge, dfg, stream):
        u, uconn, v, vconn, memlet = edge

        #############################################################
        # Instrumentation: Pre-copy

        # For perfcounters, we have to make sure that:
        # 1) No other measurements are done for the containing scope (no map operation containing this copy is instrumented)
        src_instrumented = PerfUtils.has_surrounding_perfcounters(
            src_node, dfg)
        dst_instrumented = PerfUtils.has_surrounding_perfcounters(
            dst_node, dfg)

        # From cuda.py
        cpu_storage_types = [
            types.StorageType.CPU_Heap, types.StorageType.CPU_Stack,
            types.StorageType.CPU_Pinned, types.StorageType.Register
        ]

        perf_cpu_only = (src_storage in cpu_storage_types) and (
            dst_storage in cpu_storage_types)

        perf_should_instrument = PerfSettings.perf_enable_instrumentation_for(
            sdfg) and (not src_instrumented) and (
                not dst_instrumented) and perf_cpu_only

        #############################################################

        # Determine memlet directionality
        if (isinstance(src_node, nodes.AccessNode)
                and memlet.data == src_node.data):
            write = True
        elif (isinstance(dst_node, nodes.AccessNode)
              and memlet.data == dst_node.data):
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(
                dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError(
                'Copying between code nodes is only supported as'
                ' part of the participating nodes')
        else:
            raise LookupError('Memlet does not point to any of the nodes')

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            stream.write(
                '    ' + self.memlet_definition(sdfg, memlet, False, vconn),
                sdfg, state_id, [src_node, dst_node])
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            stream.write(
                '    ' + self.memlet_definition(sdfg, memlet, True, uconn),
                sdfg, state_id, [src_node, dst_node])
            return
        else:  # Copy array-to-array
            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)

            if write:
                vconn = dst_node.data
            ctype = 'dace::vec<%s, %d>' % (dst_nodedesc.dtype.ctype,
                                           memlet.veclen)

            #############################################
            # Corner cases

            # Writing one index
            if isinstance(memlet.subset,
                          subsets.Indices) and memlet.wcr is None:
                stream.write(
                    '%s = %s;' % (vconn, self.memlet_ctor(
                        sdfg, memlet, False)), sdfg, state_id,
                    [src_node, dst_node])
                return
            # Writing from/to a stream
            if (isinstance(sdfg.arrays[memlet.data], data.Stream) or \
                (isinstance(src_node, nodes.AccessNode) and isinstance(src_nodedesc,
                                                                     data.Stream))):
                # Identify whether a stream is writing to an array
                if (isinstance(dst_nodedesc, (data.Scalar, data.Array))
                        and isinstance(src_nodedesc, data.Stream)):
                    return  # Do nothing (handled by ArrayStreamView)

                # Array -> Stream - push bulk
                if (isinstance(src_nodedesc, (data.Scalar, data.Array))
                        and isinstance(dst_nodedesc, data.Stream)):
                    if hasattr(src_nodedesc, 'src'):  # ArrayStreamView
                        stream.write(
                            '{s}.push({arr});'.format(
                                s=dst_node.data, arr=src_nodedesc.src), sdfg,
                            state_id, [src_node, dst_node])
                    else:
                        copysize = ' * '.join(
                            [sym2cpp(s) for s in memlet.subset.size()])
                        stream.write(
                            '{s}.push({arr}, {size});'.format(
                                s=dst_node.data,
                                arr=src_node.data,
                                size=copysize), sdfg, state_id,
                            [src_node, dst_node])
                    return
                else:
                    # Unknown case
                    raise NotImplementedError

            #############################################

            state_dfg = sdfg.nodes()[state_id]

            copy_shape, src_strides, dst_strides, src_expr, dst_expr = (
                self.memlet_copy_to_absolute_strides(sdfg, memlet, src_node,
                                                     dst_node))

            # Which numbers to include in the variable argument part
            dynshape, dynsrc, dyndst = 1, 1, 1

            # Dynamic copy dimensions
            if any(symbolic.issymbolic(s, sdfg.constants) for s in copy_shape):
                copy_tmpl = 'Dynamic<{type}, {veclen}, {aligned}, {dims}>'.format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned='false',
                    dims=len(copy_shape))
            else:  # Static copy dimensions
                copy_tmpl = '<{type}, {veclen}, {aligned}, {dims}>'.format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned='false',
                    dims=', '.join(sym2cpp(copy_shape)))
                dynshape = 0

            # Constant src/dst dimensions
            if not any(
                    symbolic.issymbolic(s, sdfg.constants)
                    for s in dst_strides):
                # Constant destination
                shape_tmpl = 'template ConstDst<%s>' % ', '.join(
                    sym2cpp(dst_strides))
                dyndst = 0
            elif not any(
                    symbolic.issymbolic(s, sdfg.constants)
                    for s in src_strides):
                # Constant source
                shape_tmpl = 'template ConstSrc<%s>' % ', '.join(
                    sym2cpp(src_strides))
                dynsrc = 0
            else:
                # Both dynamic
                shape_tmpl = 'Dynamic'

            # Parameter pack handling
            stride_tmpl_args = [0] * (
                dynshape + dynsrc + dyndst) * len(copy_shape)
            j = 0
            for shape, src, dst in zip(copy_shape, src_strides, dst_strides):
                if dynshape > 0:
                    stride_tmpl_args[j] = shape
                    j += 1
                if dynsrc > 0:
                    stride_tmpl_args[j] = src
                    j += 1
                if dyndst > 0:
                    stride_tmpl_args[j] = dst
                    j += 1

            copy_args = ([src_expr, dst_expr] + ([] if memlet.wcr is None else
                                                 [unparse_cr(memlet.wcr)]) +
                         sym2cpp(stride_tmpl_args))

            #############################################################
            # Instrumentation: Pre-copy 2
            unique_cpy_id = PerfSettings.get_unique_number()

            if perf_should_instrument:
                fac3 = ' * '.join(sym2cpp(copy_shape)) + " / " + '/'.join(
                    sym2cpp(dst_strides))
                copy_size = "sizeof(%s) * %s * (%s)" % (ctype, memlet.veclen,
                                                        fac3)
                node_id = PerfUtils.unified_id(dfg.node_id(dst_node), state_id)
                # Mark a section start (this is not really a section in itself (it would be a section with 1 entry))
                stream.write(
                    "__perf_store.markSectionStart(%d, (long long)%s, PAPI_thread_id());\n"
                    % (node_id, copy_size), sdfg, state_id,
                    [src_node, dst_node])
                stream.write((
                    "dace_perf::{pcs} __perf_cpy_{nodeid}_{unique_id};\n" +
                    "auto& __vs_cpy_{nodeid}_{unique_id} = __perf_store.getNewValueSet(__perf_cpy_{nodeid}_{unique_id}, {nodeid}, PAPI_thread_id(), {size}, dace_perf::ValueSetType::Copy);\n"
                    + "__perf_cpy_{nodeid}_{unique_id}.enterCritical();\n"
                ).format(
                    pcs=PerfUtils.perf_counter_string(dst_node),
                    nodeid=node_id,
                    unique_id=unique_cpy_id,
                    size=copy_size), sdfg, state_id, [src_node, dst_node])
            #############################################################

            nc = True
            if memlet.wcr is not None:
                nc = not is_write_conflicted(dfg, edge)
            if nc:
                stream.write(
                    """
                    dace::CopyND{copy_tmpl}::{shape_tmpl}::{copy_func}(
                        {copy_args});""".format(
                        copy_tmpl=copy_tmpl,
                        shape_tmpl=shape_tmpl,
                        copy_func='Copy'
                        if memlet.wcr is None else 'Accumulate',
                        copy_args=', '.join(copy_args)), sdfg, state_id,
                    [src_node, dst_node])
            else:  # Conflicted WCR
                if dynshape == 1:
                    raise NotImplementedError(
                        'Accumulation of dynamically-shaped '
                        'arrays not yet implemented')
                elif copy_shape == [
                        1
                ]:  # Special case: accumulating one element
                    dst_expr = self.memlet_view_ctor(sdfg, memlet, True)
                    stream.write(
                        write_and_resolve_expr(memlet, nc, dst_expr,
                                               '*(' + src_expr + ')'), sdfg,
                        state_id, [src_node, dst_node])
                else:
                    raise NotImplementedError('Accumulation of arrays '
                                              'with WCR not yet implemented')

        #############################################################
        # Instrumentation: Post-copy
        if perf_should_instrument:
            stream.write(("__perf_cpy_%d_%d.leaveCritical(__vs_cpy_%d_%d);\n")
                         % (node_id, unique_cpy_id, node_id, unique_cpy_id),
                         sdfg, state_id, [src_node, dst_node])
        #############################################################

    ###########################################################################
    # Memlet handling

    def process_out_memlets(self, sdfg, state_id, node, dfg, dispatcher,
                            result, locals_defined, function_stream):

        scope_dict = sdfg.nodes()[state_id].scope_dict()

        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            dst_node = dfg.memlet_path(edge)[-1].dst

            # Target is neither a data nor a tasklet node
            if (isinstance(node, nodes.AccessNode)
                    and (not isinstance(dst_node, nodes.AccessNode)
                         and not isinstance(dst_node, nodes.CodeNode))):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(
                    v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(v, nodes.CodeNode):
                shared_data_name = 's%d_n%d%s_n%d%s' % (
                    state_id, dfg.node_id(edge.src), edge.src_conn,
                    dfg.node_id(edge.dst), edge.dst_conn)
                result.write('__%s = %s;' % (shared_data_name, edge.src_conn),
                             sdfg, state_id, [edge.src, edge.dst])
                continue

            # If the memlet is not pointing to a data node (e.g. tasklet), then
            # the tasklet will take care of the copy
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            # If the memlet is pointing into an array in an inner scope, then
            # the inner scope (i.e., the output array) must handle it
            if (scope_dict[node] != scope_dict[dst_node]
                    and scope_contains_scope(scope_dict, node, dst_node)):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array
            if isinstance(node, nodes.CodeNode):
                if not uconn:
                    raise SyntaxError(
                        'Cannot copy memlet without a local connector: {} to {}'
                        .format(str(edge.src), str(edge.dst)))

                try:
                    positive_accesses = bool(memlet.num_accesses >= 0)
                except TypeError:
                    positive_accesses = False

                if memlet.subset.data_dims() == 0 and positive_accesses:
                    out_local_name = '    __' + uconn
                    in_local_name = uconn
                    if not locals_defined:
                        out_local_name = self.memlet_ctor(sdfg, memlet, True)
                        in_memlets = [
                            d for _, _, _, _, d in dfg.in_edges(node)
                        ]
                        assert len(in_memlets) == 1
                        in_local_name = self.memlet_ctor(
                            sdfg, in_memlets[0], False)

                    state_dfg = sdfg.nodes()[state_id]

                    if memlet.wcr is not None:
                        nc = not is_write_conflicted(dfg, edge)
                        result.write(
                            write_and_resolve_expr(memlet, nc, out_local_name,
                                                   in_local_name), sdfg,
                            state_id, node)
                    else:
                        result.write(
                            '%s.write(%s);\n' % (out_local_name,
                                                 in_local_name), sdfg,
                            state_id, node)
            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                if dst_node != node and not isinstance(dst_node,
                                                       nodes.Tasklet):
                    dispatcher.dispatch_copy(node, dst_node, edge, sdfg, dfg,
                                             state_id, function_stream, result)

    def memlet_view_ctor(self, sdfg, memlet, is_output):
        memlet_params = []

        memlet_name = memlet.data
        def_type = self._dispatcher.defined_vars.get(memlet_name)

        if def_type == DefinedType.Pointer:
            memlet_expr = memlet_name  # Common case
        elif (def_type == DefinedType.Scalar
              or def_type == DefinedType.ScalarView):
            memlet_expr = '&' + memlet_name
        elif def_type == DefinedType.ArrayView:
            memlet_expr = memlet_name + ".ptr()"
        else:
            raise TypeError("Unsupported connector type {}".format(def_type))

        if isinstance(memlet.subset, subsets.Indices):

            # FIXME: _packed_types influences how this offset is
            # generated from the FPGA codegen. We should find a nicer solution.
            if self._packed_types is True:
                offset = cpp_array_expr(
                    sdfg, memlet, False, packed_veclen=memlet.veclen)
            else:
                offset = cpp_array_expr(sdfg, memlet, False)

            # Compute address
            memlet_params.append(memlet_expr + ' + ' + offset)
            dims = 0

        else:

            if isinstance(memlet.subset, subsets.Range):

                dims = len(memlet.subset.ranges)

                # FIXME: _packed_types influences how this offset is
                # generated from the FPGA codegen. We should find a nicer
                # solution.
                if self._packed_types is True:
                    offset = cpp_offset_expr(
                        sdfg.arrays[memlet.data],
                        memlet.subset,
                        packed_veclen=memlet.veclen)
                else:
                    offset = cpp_offset_expr(sdfg.arrays[memlet.data],
                                             memlet.subset)
                if offset == "0":
                    memlet_params.append(memlet_expr)
                else:
                    if (def_type not in [
                            DefinedType.Pointer, DefinedType.ArrayView
                    ]):
                        raise dace.codegen.codegen.CodegenError(
                            "Cannot offset address of connector {} of type {}".
                            format(memlet_name, def_type))
                    memlet_params.append(memlet_expr + ' + ' + offset)

                # Dimensions to remove from view (due to having one value)
                indexdims = []

                # Figure out dimensions for scalar version
                for dim, (rb, re, rs) in enumerate(memlet.subset.ranges):
                    try:
                        if (re - rb) == 0:
                            indexdims.append(dim)
                    except TypeError:  # cannot determine truth value of Relational
                        pass

                # Remove index (one scalar) dimensions
                dims -= len(indexdims)

                if dims > 0:
                    strides = memlet.subset.absolute_strides(
                        sdfg.arrays[memlet.data].strides)
                    # Filter out index dims
                    strides = [
                        s for i, s in enumerate(strides) if i not in indexdims
                    ]
                    # FIXME: _packed_types influences how this offset is
                    # generated from the FPGA codegen. We should find a nicer
                    # solution.
                    if self._packed_types and memlet.veclen > 1:
                        for i in range(len(strides) - 1):
                            strides[i] /= memlet.veclen
                    memlet_params.extend(sym2cpp(strides))
                    dims = memlet.subset.data_dims()

            else:
                raise RuntimeError(
                    'Memlet type "%s" not implemented' % memlet.subset)

        if memlet.num_accesses == 1:
            num_accesses_str = "1"
        else:  # symbolic.issymbolic(memlet.num_accesses, sdfg.constants):
            num_accesses_str = 'dace::NA_RUNTIME'

        return 'dace::ArrayView%s<%s, %d, %s, %s> (%s)' % (
            "Out"
            if is_output else "In", sdfg.arrays[memlet.data].dtype.ctype, dims,
            sym2cpp(memlet.veclen), num_accesses_str, ', '.join(memlet_params))

    def memlet_definition(self, sdfg, memlet, output, local_name):
        result = ('auto __%s = ' % local_name + self.memlet_ctor(
            sdfg, memlet, output) + ';\n')

        # Allocate variable type
        memlet_type = 'dace::vec<%s, %s>' % (
            sdfg.arrays[memlet.data].dtype.ctype, sym2cpp(memlet.veclen))

        var_type = self._dispatcher.defined_vars.get(memlet.data)

        # ** Concerning aligned vs. non-aligned values:
        # We prefer aligned values, so in every case where we are assigning to
        # a local _value_, we explicitly assign to an aligned type
        # (memlet_type). In all other cases, where we need either a pointer or
        # a reference, typically due to variable number of accesses, we have to
        # use the underlying type of the ArrayView, be it aligned or unaligned,
        # to avoid runtime crashes. We use auto for this, so the ArrayView can
        # return whatever it supports.

        if var_type == DefinedType.Scalar:
            if memlet.num_accesses == 1:
                if not output:
                    # We can pre-read the value
                    result += "{} {} = __{}.val<{}>();".format(
                        memlet_type, local_name, local_name, memlet.veclen)
                else:
                    # The value will be written during the tasklet, and will be
                    # automatically written out after
                    result += "{} {};".format(memlet_type, local_name)
                self._dispatcher.defined_vars.add(local_name,
                                                  DefinedType.Scalar)
            elif memlet.num_accesses == -1:
                if output:
                    # Variable number of writes: get reference to the target of
                    # the view to reflect writes at the data
                    result += "auto &{} = __{}.ref<{}>();".format(
                        local_name, local_name, memlet.veclen)
                else:
                    # Variable number of reads: get a const reference that can
                    # be read if necessary
                    result += "auto const &{} = __{}.ref<{}>();".format(
                        local_name, local_name, memlet.veclen)
                self._dispatcher.defined_vars.add(local_name,
                                                  DefinedType.Scalar)
            else:
                raise dace.codegen.codegen.CodegenError(
                    "Unsupported number of accesses {} for scalar {}".format(
                        memlet.num_accesses, local_name))
        elif var_type == DefinedType.Pointer:
            if memlet.num_accesses == 1:
                if output:
                    result += "{} {};".format(memlet_type, local_name)
                else:
                    result += "{} {} = __{}.val<{}>();".format(
                        memlet_type, local_name, local_name, memlet.veclen)
                self._dispatcher.defined_vars.add(local_name,
                                                  DefinedType.Scalar)
            else:
                if memlet.subset.data_dims() == 0:
                    # Forward ArrayView
                    result += "auto &{} = __{}.ref<{}>();".format(
                        local_name, local_name, memlet.veclen)
                    self._dispatcher.defined_vars.add(local_name,
                                                      DefinedType.Scalar)
                else:
                    result += "auto *{} = __{}.ptr<{}>();".format(
                        local_name, local_name, memlet.veclen)
                    self._dispatcher.defined_vars.add(local_name,
                                                      DefinedType.Pointer)
        elif (var_type == DefinedType.Stream
              or var_type == DefinedType.StreamArray):
            if memlet.num_accesses == 1:
                if output:
                    result += "{} {};".format(memlet_type, local_name)
                else:
                    result += "auto {} = __{}.pop();".format(
                        local_name, local_name)
                self._dispatcher.defined_vars.add(local_name,
                                                  DefinedType.Scalar)
            else:
                # Just forward actions to the underlying object
                result += "auto &{} = __{};".format(local_name, local_name)
                self._dispatcher.defined_vars.add(local_name,
                                                  DefinedType.Stream)
        else:
            raise TypeError("Unknown variable type: {}".format(var_type))

        return result

    def memlet_stream_ctor(self, sdfg, memlet):
        stream = sdfg.arrays[memlet.data]
        dtype = "dace::vec<{}, {}>".format(stream.dtype.ctype,
                                           symbolic.symstr(memlet.veclen))
        return "dace::make_streamview({})".format(memlet.data + (
            "[{}]".format(cpp_offset_expr(stream, memlet.subset))
            if isinstance(stream, dace.data.Stream)
            and stream.is_stream_array() else ""))

    def memlet_ctor(self, sdfg, memlet, is_output):

        def_type = self._dispatcher.defined_vars.get(memlet.data)

        if (def_type == DefinedType.Stream
                or def_type == DefinedType.StreamArray):
            return self.memlet_stream_ctor(sdfg, memlet)

        elif (def_type == DefinedType.Pointer or def_type == DefinedType.Scalar
              or def_type == DefinedType.ScalarView
              or def_type == DefinedType.ArrayView):
            return self.memlet_view_ctor(sdfg, memlet, is_output)

        else:
            raise NotImplementedError(
                "Connector type {} not yet implemented".format(def_type))

    def copy_expr(self,
                  sdfg,
                  dataname,
                  memlet,
                  offset=None,
                  relative_offset=True,
                  packed_types=False):
        datadesc = sdfg.arrays[dataname]
        if relative_offset:
            s = memlet.subset
            o = offset
        else:
            if offset is None:
                s = None
            elif not isinstance(offset, subsets.Subset):
                s = subsets.Indices(offset)
            else:
                s = offset
            o = None
        if s != None:
            offset_cppstr = cpp_offset_expr(
                datadesc, s, o, memlet.veclen if packed_types else 1)
        else:
            offset_cppstr = '0'
        dt = ''

        if memlet.veclen != 1 and not packed_types:
            offset_cppstr = '(%s) / %s' % (offset_cppstr, sym2cpp(
                memlet.veclen))
            dt = '(dace::vec<%s, %s> *)' % (datadesc.dtype.ctype,
                                            sym2cpp(memlet.veclen))

        expr = dataname

        def_type = self._dispatcher.defined_vars.get(dataname)

        add_offset = (offset_cppstr != "0")

        if def_type == DefinedType.Pointer:
            return "{}{}{}".format(
                dt, expr, " + {}".format(offset_cppstr) if add_offset else "")

        elif def_type == DefinedType.ArrayView:
            return "{}{}.ptr(){}".format(
                dt, expr, " + {}".format(offset_cppstr) if add_offset else "")

        elif def_type == DefinedType.StreamArray:
            return "{}[{}]".format(expr, offset_cppstr)

        elif (def_type == DefinedType.Scalar
              or def_type == DefinedType.ScalarView
              or def_type == DefinedType.Stream):

            if add_offset:
                raise TypeError(
                    "Tried to offset address of scalar {}: {}".format(
                        dataname, offset_cppstr))

            if (def_type == DefinedType.Scalar
                    or def_type == DefinedType.ScalarView):
                return "{}&{}".format(dt, expr)
            else:
                return dataname

        else:
            raise NotImplementedError(
                "copy_expr not implemented "
                "for connector type: {}".format(def_type))

    def memlet_copy_to_absolute_strides(self,
                                        sdfg,
                                        memlet,
                                        src_node,
                                        dst_node,
                                        packed_types=False):
        # Ignore vectorization flag is a hack to accommmodate FPGA behavior,
        # where the pointer type is changed to a vector type, and addresses
        # thus shouldn't take vectorization into account.
        copy_shape = memlet.subset.size()
        copy_shape = [symbolic.overapproximate(s) for s in copy_shape]
        src_nodedesc = src_node.desc(sdfg)
        dst_nodedesc = dst_node.desc(sdfg)

        if memlet.data == src_node.data:
            src_expr = self.copy_expr(
                sdfg, src_node.data, memlet, packed_types=packed_types)
            dst_expr = self.copy_expr(
                sdfg,
                dst_node.data,
                memlet,
                None,
                False,
                packed_types=packed_types)
            if memlet.other_subset is not None:
                dst_expr = self.copy_expr(
                    sdfg,
                    dst_node.data,
                    memlet,
                    memlet.other_subset,
                    False,
                    packed_types=packed_types)
                dst_subset = memlet.other_subset
            else:
                dst_subset = subsets.Range.from_array(dst_nodedesc)
            src_subset = memlet.subset

        else:
            src_expr = self.copy_expr(
                sdfg,
                src_node.data,
                memlet,
                None,
                False,
                packed_types=packed_types)
            dst_expr = self.copy_expr(
                sdfg, dst_node.data, memlet, packed_types=packed_types)
            if memlet.other_subset is not None:
                src_expr = self.copy_expr(
                    sdfg,
                    src_node.data,
                    memlet,
                    memlet.other_subset,
                    False,
                    packed_types=packed_types)
                src_subset = memlet.other_subset
            else:
                src_subset = subsets.Range.from_array(src_nodedesc)
            dst_subset = memlet.subset

        src_strides = src_subset.absolute_strides(src_nodedesc.strides)
        dst_strides = dst_subset.absolute_strides(dst_nodedesc.strides)

        # Try to turn into degenerate/strided ND copies
        result = ndcopy_to_strided_copy(copy_shape, src_nodedesc.strides,
                                        src_strides, dst_nodedesc.strides,
                                        dst_strides, memlet.subset)
        if result is not None:
            copy_shape, src_strides, dst_strides = result
        else:
            # If other_subset is defined, reduce its dimensionality by
            # removing the "empty" dimensions (size = 1) and filter the
            # corresponding strides out
            src_strides = [
                stride for stride, s in zip(src_strides, src_subset.size())
                if s != 1
            ] + src_strides[len(src_subset):]  # Include tiles
            if not src_strides:
                src_strides = [1]
            dst_strides = [
                stride for stride, s in zip(dst_strides, dst_subset.size())
                if s != 1
            ] + dst_strides[len(dst_subset):]  # Include tiles
            if not dst_strides:
                dst_strides = [1]
            copy_shape = [s for s in copy_shape if s != 1]
            if not copy_shape:
                copy_shape = [1]

        # Extend copy shape to the largest among the data dimensions,
        # and extend other array with the appropriate strides
        if (len(dst_strides) != len(copy_shape)
                or len(src_strides) != len(copy_shape)):
            if memlet.data == src_node.data:
                copy_shape, dst_strides = _reshape_strides(
                    src_subset, src_strides, dst_strides, copy_shape)
            elif memlet.data == dst_node.data:
                copy_shape, src_strides = _reshape_strides(
                    dst_subset, dst_strides, src_strides, copy_shape)

        if memlet.veclen != 1:
            int_floor = sp.Function('int_floor')
            src_strides[:-1] = [
                int_floor(s, memlet.veclen) for s in src_strides[:-1]
            ]
            dst_strides[:-1] = [
                int_floor(s, memlet.veclen) for s in dst_strides[:-1]
            ]
            if not packed_types:
                copy_shape[-1] = int_floor(copy_shape[-1], memlet.veclen)

        return copy_shape, src_strides, dst_strides, src_expr, dst_expr

    #########################################################################
    # Dynamically-called node dispatchers

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        callsite_stream.write('{\n', sdfg, state_id, node)

        # Add code to init and exit functions
        self._frame._initcode.write(node.code_init, sdfg)
        self._frame._exitcode.write(node.code_exit, sdfg)

        state_dfg = sdfg.nodes()[state_id]

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in state_dfg.in_edges(node):
            u = edge.src
            memlet = edge.data

            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError('Duplicates found in memlets')
                # Special case: code->code
                if isinstance(edge.src, nodes.CodeNode):
                    shared_data_name = 's%d_n%d%s_n%d%s' % (
                        state_id, dfg.node_id(edge.src), edge.src_conn,
                        dfg.node_id(edge.dst), edge.dst_conn)

                    # Read variable from shared storage
                    callsite_stream.write(
                        'const dace::vec<%s, %s>& %s = __%s;' %
                        (sdfg.arrays[memlet.data].dtype.ctype,
                         sym2cpp(memlet.veclen), edge.dst_conn,
                         shared_data_name), sdfg, state_id,
                        [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn,
                                                      DefinedType.Scalar)

                else:
                    src_node = find_input_arraynode(state_dfg, edge)

                    self._dispatcher.dispatch_copy(
                        src_node, node, edge, sdfg, state_dfg, state_id,
                        function_stream, callsite_stream)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1)
                arrays.add(edge.dst_conn)

        callsite_stream.write('\n', sdfg, state_id, node)

        # Use outgoing edges to preallocate output local vars
        for edge in state_dfg.out_edges(node):
            v = edge.dst
            memlet = edge.data

            if edge.src_conn:
                if edge.src_conn in arrays:  # Disallow duplicates
                    continue
                # Special case: code->code
                if isinstance(edge.dst, nodes.CodeNode):
                    callsite_stream.write(
                        'dace::vec<%s, %s> %s;' %
                        (sdfg.arrays[memlet.data].dtype.ctype,
                         sym2cpp(memlet.veclen), edge.src_conn), sdfg,
                        state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.src_conn,
                                                      DefinedType.Scalar)
                else:
                    dst_node = find_output_arraynode(state_dfg, edge)

                    self._dispatcher.dispatch_copy(
                        node, dst_node, edge, sdfg, state_dfg, state_id,
                        function_stream, callsite_stream)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1)
                arrays.add(edge.src_conn)

        callsite_stream.write('\n    ///////////////////\n', sdfg, state_id,
                              node)

        unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                        callsite_stream, self._locals, self._ldepth)

        callsite_stream.write('    ///////////////////\n\n', sdfg, state_id,
                              node)

        # Process outgoing memlets
        self.process_out_memlets(sdfg, state_id, node, state_dfg,
                                 self._dispatcher, callsite_stream, True,
                                 function_stream)

        #############################################################
        # Instrumentation: Post-tasklet
        if PerfSettings.perf_enable_instrumentation(
        ) and PerfUtils.has_surrounding_perfcounters(node, dfg):
            # Add bytes moved
            callsite_stream.write(
                "__perf_store.addBytesMoved(%s);" %
                PerfUtils.get_tasklet_byte_accesses(node, dfg, sdfg, state_id))
        #############################################################

        callsite_stream.write('}\n', sdfg, state_id, node)

        self._dispatcher.defined_vars.exit_scope(node)

    def _generate_EmptyTasklet(self, sdfg, dfg, state_id, node,
                               function_stream, callsite_stream):
        self._generate_Tasklet(sdfg, dfg, state_id, node, function_stream,
                               callsite_stream)

    def _generate_NestedSDFG(self, sdfg, dfg: ScopeSubgraphView, state_id,
                             node, function_stream: CodeIOStream,
                             callsite_stream: CodeIOStream):

        self._dispatcher.defined_vars.enter_scope(sdfg)

        # If SDFG parent is not set, set it
        node.sdfg._parent = sdfg
        state_dfg = sdfg.nodes()[state_id]

        # Take care of nested SDFG I/O
        for _, _, _, vconn, in_memlet in state_dfg.in_edges(node):
            callsite_stream.write(
                self.memlet_definition(sdfg, in_memlet, False, vconn), sdfg,
                state_id, node)
        for _, uconn, _, _, out_memlet in state_dfg.out_edges(node):
            callsite_stream.write(
                self.memlet_definition(sdfg, out_memlet, True, uconn), sdfg,
                state_id, node)

        callsite_stream.write('\n    ///////////////////\n', sdfg, state_id,
                              node)

        sdfg_label = '_%d_%d' % (state_id, dfg.node_id(node))
        # Generate code for internal SDFG
        global_code, local_code, used_targets = \
            self._frame.generate_code(node.sdfg, node.schedule, sdfg_label)

        # Write generated code in the proper places (nested SDFG writes
        # location info)
        function_stream.write(global_code)
        callsite_stream.write(local_code)

        callsite_stream.write('    ///////////////////\n\n', sdfg, state_id,
                              node)

        # Process outgoing memlets with the internal SDFG
        self.process_out_memlets(sdfg, state_id, node, state_dfg,
                                 self._dispatcher, callsite_stream, True,
                                 function_stream)

        self._dispatcher.defined_vars.exit_scope(sdfg)

    def _generate_MapEntry(self, sdfg, dfg, state_id, node: nodes.MapEntry,
                           function_stream, callsite_stream):
        map_params = node.map.params
        map_name = '__DACEMAP_' + str(state_id) + '_' + str(dfg.node_id(node))

        unified_id = PerfUtils.unified_id(dfg.node_id(node), state_id)

        #############################################################
        # Instrumentation: Pre-MapEntry

        # Intrusively set the depth
        PerfUtils.set_map_depth(node, dfg)

        result = callsite_stream

        map_header = ''

        if PerfSettings.perf_enable_instrumentation():
            idstr = "// (Node %d)\n" % unified_id
            map_header += idstr  # Used to identify line numbers later
            PerfMetaInfoStatic.info.add_node(node, idstr)

        if node.map.schedule == types.ScheduleType.CPU_Multicore:
            # We have to find out if we should mark a section start here or later.
            children = PerfUtils.all_maps(node, dfg)

            for x in children:
                if PerfUtils.map_depth(
                        x) > PerfSettings.perf_max_scope_depth():
                    break  # We have our relevant nodes.
                if x.map.schedule == types.ScheduleType.CPU_Multicore:
                    # nested SuperSections are not well-supported
                    # We have to mark the outermost section,
                    # which also means that we have to somehow tell the
                    # lower nodes to not mark the section start.
                    x.map._can_be_supersection_start = False

            if PerfSettings.perf_enable_instrumentation_for(
                    sdfg, node
            ) and PerfUtils.map_depth(
                    node
            ) <= PerfSettings.perf_max_scope_depth(
            ) and node.map._can_be_supersection_start and not dfg.is_parallel(
            ):
                map_header += "__perf_store.markSuperSectionStart(%d);\n" % unified_id
            elif PerfSettings.perf_supersection_emission_debug():
                reasons = []
                if not node.map._can_be_supersection_start:
                    reasons.append("CANNOT_BE_SS")
                if dfg.is_parallel():
                    reasons.append("CONTAINER_IS_PARALLEL")
                if PerfUtils.map_depth(
                        node) > PerfSettings.perf_max_scope_depth():
                    reasons.append("EXCEED_MAX_DEPTH")
                if not PerfSettings.perf_enable_instrumentation_for(
                        sdfg, node):
                    reasons.append("MISC")

                map_header += "// SuperSection start not emitted. Reasons: " + ",".join(
                    reasons) + "\n"

        elif PerfSettings.perf_enable_instrumentation_for(
                sdfg, node
        ) and PerfUtils.map_depth(node) == PerfSettings.perf_max_scope_depth(
        ) and node.map._can_be_supersection_start and not dfg.is_parallel():
            # even if the schedule is sequential, we can serialize to
            # keep buffer usage low
            map_header += "__perf_store.markSuperSectionStart(%d);\n" % unified_id

        if PerfUtils.instrument_entry(
                node, dfg) and PerfSettings.perf_enable_instrumentation_for(
                    sdfg, node):

            size = PerfUtils.accumulate_byte_movements_v2(
                node, node, dfg, sdfg, state_id)
            size = sp.simplify(size)

            used_symbols = symbolic.symbols_in_sympy_expr(size)
            defined_symbols = sdfg.symbols_defined_at(node)
            undefined_symbols = [
                x for x in used_symbols if x not in defined_symbols
            ]
            if len(undefined_symbols) > 0:
                # We cannot statically determine the size at this point
                print(
                    "Failed to determine size because of undefined symbols (\""
                    + str(undefined_symbols) + "\") in \"" + str(size) +
                    "\", falling back to 0")
                size = 0

            size = sym2cpp(size)

            map_header += "__perf_store.markSectionStart(%d, (long long)%s, PAPI_thread_id());\n" % (
                unified_id, size)

        #############################################################

        if node.map.schedule == types.ScheduleType.CPU_Multicore:
            map_header += '#pragma omp parallel for'
            openmp_parallel_for_defined = True

            # The code below is disabled since we now use pragma omp atomic
            # TODO(later): set up register outside loop
            #exit_node = dfg.exit_nodes(node)[0]
            reduction_stmts = []
            #for outedge in dfg.in_edges(exit_node):
            #    if (isinstance(outedge.src, nodes.CodeNode)
            #            and outedge.data.wcr is not None):
            #        redt = operations.detect_reduction_type(outedge.data.wcr)
            #        if redt != types.ReductionType.Custom:
            #            reduction_stmts.append('reduction({typ}:{var})'.format(
            #                typ=_REDUCTION_TYPE_TO_OPENMP[redt],
            #                var=outedge.src_conn))
            #            reduced_variables.append(outedge)

            map_header += ' %s\n' % ', '.join(reduction_stmts)

        # TODO: Explicit map unroller
        if node.map.unroll:
            if node.map.schedule == types.ScheduleType.CPU_Multicore:
                raise ValueError('An Multicore CPU map cannot be unrolled (' +
                                 node.map.label + ')')

        constsize = all([
            not symbolic.issymbolic(v, sdfg.constants) for r in node.map.range
            for v in r
        ])

        # Construct (EXCLUSIVE) map range as a list of comma-delimited C++
        # strings.
        maprange_cppstr = [
            '%s, %s, %s' % (sym2cpp(rb), sym2cpp(re + 1), sym2cpp(rs))
            for rb, re, rs in node.map.range
        ]

        # Map flattening
        if node.map.flatten:

            #############################################################
            # Instrumentation: Post-MapEntry (pre-definitions)
            perf_entry_string = (
                'dace_perf::%s __perf_%d;\n' +
                'auto& __vs_%d = __perf_store.getNewValueSet(__perf_%d, %d, PAPI_thread_id(), %%s);\n'
                + '__perf_%d.enterCritical();\n') % (
                    PerfUtils.perf_counter_string(node), unified_id,
                    unified_id, unified_id, unified_id, unified_id)
            #############################################################

            # If the integer set is constant-sized, emit const_int_range
            if constsize:
                # Generate the loop
                result.write(
                    """
typedef dace::const_int_range<{range}> {mapname}_rng;
{map_header}
for (int {mapname}_iter = 0; {mapname}_iter < {mapname}_rng::size; ++{mapname}_iter) {{
                             """.format(
                        range=', '.join(maprange_cppstr),
                        map_header=map_header,
                        mapname=map_name), sdfg, state_id, node)

                #############################################################
                # Instrumentation: Post-MapEntry (pre-definitions)
                # Perfcounters for flattened maps include the calculations
                # made to obtain the different axis indices
                if PerfUtils.instrument_entry(
                        node,
                        dfg) and PerfSettings.perf_enable_instrumentation_for(
                            sdfg, node):
                    result.write(perf_entry_string % (map_name + "_iter"),
                                 sdfg, state_id, node)
                    # remember which map has the counters enabled
                    node.map._has_papi_counters = True
                #############################################################

                # Generate the variables
                for ind, var in enumerate(map_params):
                    result.write(
                        ('auto {var} = {mapname}_rng' +
                         '::index_value({mapname}_iter, ' + '{ind});').format(
                             ind=ind, var=var,
                             mapname=map_name), sdfg, state_id, node)
            else:  # Runtime-size integer range set
                # Generate the loop
                result.write(
                    """
auto {mapname}_rng = dace::make_range({tuplerange});
{map_header}
for (int {mapname}_iter = 0; {mapname}_iter < {mapname}_rng.size(); ++{mapname}_iter) {{
                                 """.format(
                        tuplerange=', '.join([
                            'std::make_tuple(%s)' % cppr
                            for cppr in maprange_cppstr
                        ]),
                        map_header=map_header,
                        mapname=map_name), sdfg, state_id, node)

                #############################################################
                # Instrumentation: Post-MapEntry (pre-definitions)
                # Perfcounters for flattened maps include the calculations
                # made to obtain the different axis indices
                if PerfUtils.instrument_entry(
                        node,
                        dfg) and PerfSettings.perf_enable_instrumentation_for(
                            sdfg, node):
                    result.write(perf_entry_string % (map_name + "_iter"),
                                 sdfg, state_id, node)
                    # remember which map has the counters enabled
                    node.map._has_papi_counters = True
                #############################################################

                # Generate the variables
                for ind, var in enumerate(map_params):
                    result.write(
                        ('auto {var} = {mapname}_rng' +
                         '.index_value({mapname}_iter, ' + '{ind});').format(
                             ind=ind, var=var,
                             mapname=map_name), sdfg, state_id, node)

        else:  # Nested loops
            result.write(map_header, sdfg, state_id, node)
            for i, r in enumerate(node.map.range):
                #var = '__DACEMAP_%s_%d' % (node.map.label, i)
                var = map_params[i]
                begin, end, skip = r

                if node.map.unroll:
                    result.write('#pragma unroll', sdfg, state_id, node)

                result.write(
                    'for (auto %s = %s; %s < %s; %s += %s) {\n' %
                    (var, sym2cpp(begin), var, sym2cpp(end + 1), var,
                     sym2cpp(skip)), sdfg, state_id, node)

                #############################################################
                # Instrumentation: Post-MapEntry (pre-definitions)
                if PerfUtils.instrument_entry(node, dfg) and (
                    (not PerfSettings.perf_debug_profile_innermost and i == 0)
                        or (PerfSettings.perf_debug_profile_innermost
                            and i == len(node.map.range) - 1)
                ) and PerfSettings.perf_enable_instrumentation_for(sdfg, node):
                    result.write(
                        ('dace_perf::%s __perf_%d;\n' +
                         'auto& __vs_%d = __perf_store.getNewValueSet(__perf_%d, %d, PAPI_thread_id(), %s);\n'
                         + '__perf_%d.enterCritical();\n') %
                        (PerfUtils.perf_counter_string(node), unified_id,
                         unified_id, unified_id, unified_id, var, unified_id),
                        sdfg, state_id, node)
                    # remember which map has the counters enabled
                    node.map._has_papi_counters = True
                #############################################################

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(sdfg, dfg, node)
        allocated = set()
        for child in dfg.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child,
                                               None, result)
            self._dispatcher.dispatch_initialize(sdfg, dfg, state_id, child,
                                                 None, result)

        # Generate register definitions for inter-tasklet memlets
        scope_dict = dfg.scope_dict()
        for edge in dfg.edges():
            # Only interested in edges within current scope
            if scope_dict[edge.src] != node or scope_dict[edge.dst] != node:
                continue
            if (isinstance(edge.src, nodes.CodeNode)
                    and isinstance(edge.dst, nodes.CodeNode)):
                local_name = '__s%d_n%d%s_n%d%s' % (
                    state_id, dfg.node_id(edge.src), edge.src_conn,
                    dfg.node_id(edge.dst), edge.dst_conn)
                # Allocate variable type
                code = 'dace::vec<%s, %s> %s;' % (
                    sdfg.arrays[edge.data.data].dtype.ctype,
                    sym2cpp(edge.data.veclen), local_name)
                result.write(code, sdfg, state_id, [edge.src, edge.dst])

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        map_node = scope_dict[node]

        if map_node is None:
            raise ValueError('Exit node ' + str(node.map.label) +
                             ' is not dominated by a scope entry node')

        #############################################################
        # Instrumentation: Pre-MapExit
        unified_id = PerfUtils.unified_id(dfg.node_id(map_node), state_id)
        #############################################################

        # Emit internal transient array deallocation
        to_allocate = dace.sdfg.local_transients(sdfg, dfg, map_node)
        deallocated = set()
        for child in dfg.scope_dict(node_to_children=True)[map_node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in deallocated:
                continue
            deallocated.add(child.data)
            self._dispatcher.dispatch_deallocate(sdfg, dfg, state_id, child,
                                                 None, result)

        # If there are other non-visited map exits, they are responsible for
        # closing braces
        map_exits = [
            k for k, v in scope_dict.items()
            if v == map_node and isinstance(k, nodes.ExitNode)
            and k not in self._generated_nodes
        ]
        if len(map_exits) > 1:
            return

        # Map flattening
        if map_node.map.flatten:
            #############################################################
            # Instrumentation: Pre-MapExit
            if PerfSettings.perf_enable_instrumentation(
            ) and map_node.map._has_papi_counters:
                result.write(
                    '__perf_%d.leaveCritical(__vs_%d);\n' %
                    (unified_id, unified_id), sdfg, state_id, node)
            if PerfSettings.perf_debug_annotate_scopes:
                result.write('// %s\n' % str(map_node), sdfg, state_id, node)
            #############################################################
            result.write('}', sdfg, state_id, node)
        else:
            for i, r in enumerate(map_node.map.range):
                #############################################################
                # Instrumentation: Pre-MapExit
                if PerfSettings.perf_enable_instrumentation(
                ) and map_node.map._has_papi_counters and (
                    (PerfSettings.perf_debug_profile_innermost and i == 0) or
                    (not PerfSettings.perf_debug_profile_innermost
                     and i == len(map_node.map.range) - 1)):
                    result.write(
                        '__perf_%d.leaveCritical(__vs_%d);\n' %
                        (unified_id, unified_id), sdfg, state_id, node)
                if PerfSettings.perf_debug_annotate_scopes and i == len(
                        map_node.map.range) - 1:
                    result.write('// %s\n' % str(map_node), sdfg, state_id,
                                 node)
                #############################################################
                result.write('}', sdfg, state_id, node)

        #############################################################
        # Instrumentation: Post-MapExit
        if PerfSettings.perf_enable_vectorization_analysis():
            idstr = "// end (Node %d)\n" % unified_id
            result.write(idstr, sdfg, state_id, node)
            PerfMetaInfoStatic.info.add_node(node, idstr)
        #############################################################

    def _generate_ConsumeEntry(self, sdfg, dfg, state_id, node: nodes.MapEntry,
                               function_stream, callsite_stream):
        result = callsite_stream

        constsize = all([
            not symbolic.issymbolic(v, sdfg.constants) for r in node.map.range
            for v in r
        ])
        state_dfg = sdfg.nodes()[state_id]

        input_sedge = next(
            e for e in state_dfg.in_edges(node) if e.dst_conn == 'IN_stream')
        output_sedge = next(
            e for e in state_dfg.out_edges(node) if e.src_conn == 'OUT_stream')
        input_stream = state_dfg.memlet_path(input_sedge)[0].src
        input_streamdesc = input_stream.desc(sdfg)

        # Take chunks into account
        if node.consume.chunksize == 1:
            chunk = 'const %s& %s' % (input_streamdesc.dtype.ctype,
                                      node.consume.label + '_element')
            self._dispatcher.defined_vars.add(node.consume.label + "_element",
                                              DefinedType.Scalar)
        else:
            chunk = 'const %s *%s, size_t %s' % (
                input_streamdesc.dtype.ctype, node.consume.label + '_elements',
                node.consume.label + '_numelems')
            self._dispatcher.defined_vars.add(node.consume.label + "_elements",
                                              DefinedType.Pointer)
            self._dispatcher.defined_vars.add(node.consume.label + "_numelems",
                                              DefinedType.Scalar)

        # Take quiescence condition into account
        if node.consume.condition is not None:
            condition_string = (
                '[&]() { return %s; }, ' % cppunparse.cppunparse(
                    node.consume.condition, False))
        else:
            condition_string = ''

        result.write(
            'dace::Consume<{chunksz}>::template consume{cond}({stream_in}, '
            '{num_pes}, {condition}'
            '[&](int {pe_index}, {element_or_chunk}) {{'.format(
                chunksz=node.consume.chunksize,
                cond='' if node.consume.condition is None else '_cond',
                condition=condition_string,
                stream_in=input_stream.data,  # TODO: stream arrays
                element_or_chunk=chunk,
                num_pes=sym2cpp(node.consume.num_pes),
                pe_index=node.consume.pe_index),
            sdfg,
            state_id,
            node)

        # Since consume is an alias node, we create an actual array for the
        # consumed element and modify the outgoing memlet path ("OUT_stream")
        # TODO: do this before getting to the codegen
        if node.consume.chunksize == 1:
            consumed_element = sdfg.add_scalar(
                node.consume.label + '_element',
                input_streamdesc.dtype,
                transient=True,
                storage=types.StorageType.Register)
            ce_node = nodes.AccessNode(node.consume.label + '_element',
                                       types.AccessType.ReadOnly)
        else:
            consumed_element = sdfg.add_array(
                node.consume.label + '_elements', [node.consume.chunksize],
                input_streamdesc.dtype,
                transient=True,
                storage=types.StorageType.Register)
            ce_node = nodes.AccessNode(node.consume.label + '_elements',
                                       types.AccessType.ReadOnly)
        state_dfg.add_node(ce_node)
        out_memlet_path = state_dfg.memlet_path(output_sedge)
        state_dfg.remove_edge(out_memlet_path[0])
        state_dfg.add_edge(
            out_memlet_path[0].src, out_memlet_path[0].src_conn, ce_node, None,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)))
        state_dfg.add_edge(
            ce_node, None, out_memlet_path[0].dst, out_memlet_path[0].dst_conn,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)))
        for e in out_memlet_path[1:]:
            e.data.data = ce_node.data
        ## END of SDFG-rewriting code

        # Emit internal transient array allocation
        to_allocate = dace.sdfg.local_transients(sdfg, dfg, node)
        allocated = set()
        for child in dfg.scope_dict(node_to_children=True)[node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, child,
                                               None, result)
            self._dispatcher.dispatch_initialize(sdfg, dfg, state_id, child,
                                                 None, result)

        # Generate register definitions for inter-tasklet memlets
        scope_dict = dfg.scope_dict()
        for edge in dfg.edges():
            # Only interested in edges within current scope
            if scope_dict[edge.src] != node or scope_dict[edge.dst] != node:
                continue
            if (isinstance(edge.src, nodes.CodeNode)
                    and isinstance(edge.dst, nodes.CodeNode)):
                local_name = '__s%d_n%d%s_n%d%s' % (
                    state_id, dfg.node_id(edge.src), edge.src_conn,
                    dfg.node_id(edge.dst), edge.dst_conn)
                # Allocate variable type
                code = 'dace::vec<%s, %s> %s;' % (
                    sdfg.arrays[edge.data.data].dtype.ctype,
                    sym2cpp(edge.data.veclen), local_name)
                result.write(code, sdfg, state_id, [edge.src, edge.dst])

    def _generate_ConsumeExit(self, sdfg, dfg, state_id, node, function_stream,
                              callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]

        if entry_node is None:
            raise ValueError('Exit node ' + str(node.consume.label) +
                             ' is not dominated by a scope entry node')

        # Emit internal transient array deallocation
        to_allocate = dace.sdfg.local_transients(sdfg, dfg, entry_node)
        deallocated = set()
        for child in dfg.scope_dict(node_to_children=True)[entry_node]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in deallocated:
                continue
            deallocated.add(child.data)
            self._dispatcher.dispatch_deallocate(sdfg, dfg, state_id, child,
                                                 None, result)

        result.write('});', sdfg, state_id, node)

    def _generate_Reduce(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):

        unified_id = PerfUtils.unified_id(dfg.node_id(node), state_id)

        # Try to autodetect reduction type
        redtype = operations.detect_reduction_type(node.wcr)

        loop_header = ''

        perf_should_instrument = PerfSettings.perf_enable_instrumentation(
        ) and not PerfUtils.has_surrounding_perfcounters(
            node, dfg) and PerfSettings.perf_enable_instrumentation_for(
                sdfg, node)

        if node.schedule == types.ScheduleType.CPU_Multicore:
            if PerfSettings.perf_enable_vectorization_analysis():
                idstr = "// (Node %d)\n" % dfg.node_id(node)
                loop_header += idstr
                PerfMetaInfoStatic.info.add_node(node, idstr)
            loop_header += '#pragma omp parallel for'

        end_braces = 0

        axes = node.axes
        state_dfg = sdfg.nodes()[state_id]
        input_memlet = state_dfg.in_edges(node)[0].data
        output_edge = state_dfg.out_edges(node)[0]
        output_memlet = output_edge.data

        output_type = 'dace::vec<%s, %s>' % (
            sdfg.arrays[output_memlet.data].dtype.ctype, output_memlet.veclen)

        # If axes were not defined, use all input dimensions
        input_dims = input_memlet.subset.dims()
        output_dims = output_memlet.subset.data_dims()
        if axes is None:
            axes = tuple(range(input_dims))

        # Obtain variable names per output and reduction axis
        axis_vars = []
        octr = 0
        for d in range(input_dims):
            if d in axes:
                axis_vars.append('__i%d' % d)
            else:
                axis_vars.append('__o%d' % octr)
                octr += 1

        #############################################################
        # Instrumentation: Pre-reduce
        # For measuring the memory bandwidth, we analyze the amount of data
        # moved.
        if perf_should_instrument:
            perf_expected_data_movement_sympy = 1

            for axis in range(output_dims):
                ao = output_memlet.subset[axis]
                perf_expected_data_movement_sympy *= (
                    (ao[1] + 1 - ao[0]) / ao[2])

            for axis in axes:
                ai = input_memlet.subset[axis]
                perf_expected_data_movement_sympy *= (
                    (ai[1] + 1 - ai[0]) / ai[2])

            if not dfg.is_parallel():
                # Now we put a start marker, but only if we are in a serial state
                callsite_stream.write(
                    '__perf_store.markSuperSectionStart(%d);\n' % (unified_id))

            callsite_stream.write(
                '__perf_store.markSectionStart(%d, (long long)%s, PAPI_thread_id());\n'
                % (unified_id,
                   str(sp.simplify(perf_expected_data_movement_sympy)) +
                   (" * (sizeof(%s) + sizeof(%s))" %
                    (sdfg.arrays[output_memlet.data].dtype.ctype,
                     sdfg.arrays[input_memlet.data].dtype.ctype))), sdfg,
                state_id, node)
        #############################################################

        # Write OpenMP loop pragma if there are output dimensions
        if output_dims > 0:
            callsite_stream.write(loop_header, sdfg, state_id, node)

        # Generate outer loops
        output_subset = output_memlet.subset
        for axis in range(output_dims):
            callsite_stream.write(
                'for (int {var} = {begin}; {var} < {end}; {var} += {skip}) {{'.
                format(
                    var='__o%d' % axis,
                    begin=output_subset[axis][0],
                    end=output_subset[axis][1] + 1,
                    skip=output_subset[axis][2]), sdfg, state_id, node)

            #############################################################
            # Instrumentation: Reduce (part 1)
            # This could prevent the compiler from parallelizing/vectorizing
            if perf_should_instrument:
                if ((end_braces == 0
                     and not PerfSettings.perf_debug_profile_innermost)
                        or (end_braces == output_dims - 1
                            and PerfSettings.perf_debug_profile_innermost)):
                    callsite_stream.write(
                        'dace_perf::%s __perf_%d;\n' %
                        (PerfUtils.perf_counter_string(node), unified_id),
                        sdfg, state_id, node)
                    callsite_stream.write(
                        'auto& __perf_%d_vs = __perf_store.getNewValueSet(__perf_%d, %d, PAPI_thread_id(), __o%d);\n'
                        % (unified_id, unified_id, unified_id, axis), sdfg,
                        state_id, node)
                    callsite_stream.write(
                        '__perf_%d.enterCritical();\n' % unified_id, sdfg,
                        state_id, node)
            #############################################################
            end_braces += 1

        #############################################################
        # Instrumentation: Reduce (part 2)
        if end_braces == 0 and perf_should_instrument:
            callsite_stream.write(
                'dace_perf::%s __perf_%d;\n' %
                (PerfUtils.perf_counter_string(node), unified_id), sdfg,
                state_id, node)
            callsite_stream.write(
                'auto& __perf_%d_vs = __perf_store.getNewValueSet(__perf_%d, %d, PAPI_thread_id(),  0);\n'
                % (unified_id, unified_id, unified_id), sdfg, state_id, node)
            callsite_stream.write('__perf_%d.enterCritical();\n' % unified_id,
                                  sdfg, state_id, node)
        #############################################################

        use_tmpout = False
        if len(axes) == input_dims:
            # Add OpenMP reduction clause if reducing all axes
            if (redtype != types.ReductionType.Custom
                    and node.schedule == types.ScheduleType.CPU_Multicore):
                loop_header += ' reduction(%s: __tmpout)' % (
                    _REDUCTION_TYPE_TO_OPENMP[redtype])

            # Output initialization
            identity = ''
            if node.identity is not None:
                identity = ' = %s' % sym2cpp(node.identity)
            callsite_stream.write(
                '{\n%s __tmpout%s;' % (output_type, identity), sdfg, state_id,
                node)
            callsite_stream.write(loop_header, sdfg, state_id, node)
            end_braces += 1
            use_tmpout = True

        # Generate inner loops (reducing)
        input_subset = input_memlet.subset
        for axis in axes:
            callsite_stream.write(
                'for (int {var} = {begin}; {var} < {end}; {var} += {skip}) {{'.
                format(
                    var='__i%d' % axis,
                    begin=input_subset[axis][0],
                    end=input_subset[axis][1] + 1,
                    skip=input_subset[axis][2]), sdfg, state_id, node)
            end_braces += 1

        # Generate reduction code
        credtype = 'dace::ReductionType::' + str(
            redtype)[str(redtype).find('.') + 1:]

        # Use index expressions
        outvar = ('__tmpout' if use_tmpout else cpp_array_expr(
            sdfg,
            output_memlet,
            offset=['__o%d' % i for i in range(output_dims)],
            relative_offset=False))
        invar = cpp_array_expr(
            sdfg, input_memlet, offset=axis_vars, relative_offset=False)

        if redtype != types.ReductionType.Custom:
            callsite_stream.write(
                'dace::wcr_fixed<%s, %s>::reduce_atomic(&%s, %s);' %
                (credtype, output_type, outvar, invar), sdfg, state_id,
                node)  #cpp_array_expr(), cpp_array_expr()
        else:
            callsite_stream.write(
                'dace::wcr_custom<%s>::template reduce_atomic(%s, &%s, %s);' %
                (output_type, unparse_cr(node.wcr), outvar, invar), sdfg,
                state_id, node)  #cpp_array_expr(), cpp_array_expr()

        #############################################################
        # Instrumentation: Post-Reduce (pre-braces)
        byte_moved_measurement = "__perf_store.addBytesMoved(%s);\n"

        # For reductions, we assume Read-Modify-Write for all operations
        # Every reduction statement costs sizeof(input) + sizeof(output).
        # This is wrong with some custom reductions or extending operations
        # (e.g., i32 * i32 => i64)
        # It also is wrong for write-avoiding min/max (min/max that only
        # overwrite the reduced variable when it needs to be changed)

        if perf_should_instrument:
            callsite_stream.write(
                byte_moved_measurement % ("(sizeof(%s) + sizeof(%s))" %
                                          (outvar, invar)), sdfg, state_id,
                node)
        #############################################################

        # Generate closing braces
        for i in range(end_braces):
            # Store back tmpout into the true output
            if i == end_braces - 1 and use_tmpout:
                callsite_stream.write(
                    '%s = __tmpout;' % cpp_array_expr(sdfg, output_memlet),
                    sdfg, state_id, node)
            #############################################################
            # Instrumentation: Post-Reduce (in-braces)
            if perf_should_instrument and (
                (i == end_braces - 1
                 and not PerfSettings.perf_debug_profile_innermost) or
                (i == len(axes)
                 and PerfSettings.perf_debug_profile_innermost)):
                callsite_stream.write(
                    '__perf_%d.leaveCritical(__perf_%d_vs);\n' %
                    (unified_id, unified_id), sdfg, state_id, node)
            #############################################################
            callsite_stream.write('}', sdfg, state_id, node)

    def _generate_AccessNode(self, sdfg, dfg, state_id, node, function_stream,
                             callsite_stream):
        state_dfg = sdfg.nodes()[state_id]

        if node not in state_dfg.sink_nodes():
            # NOTE: sink nodes are synchronized at the end of a state
            presynchronize_streams(sdfg, state_dfg, state_id, node,
                                   callsite_stream)

        sdict = state_dfg.scope_dict()
        for edge in state_dfg.in_edges(node):
            predecessor, _, _, _, memlet = edge
            if memlet.data is None:
                continue  # If the edge has to be skipped

            # Determines if this path ends here or has a definite source (array) node
            memlet_path = state_dfg.memlet_path(edge)
            if memlet_path[-1].dst == node:
                src_node = memlet_path[0].src
                # Only generate code in case this is the innermost scope
                # (copies are generated at the inner scope, where both arrays exist)
                if (scope_contains_scope(sdict, src_node, node)
                        and sdict[src_node] != sdict[node]):
                    self._dispatcher.dispatch_copy(
                        src_node, node, edge, sdfg, dfg, state_id,
                        function_stream, callsite_stream)

        # Process outgoing memlets (array-to-array write should be emitted
        # from the first leading edge out of the array)
        self.process_out_memlets(sdfg, state_id, node, state_dfg,
                                 self._dispatcher, callsite_stream, False,
                                 function_stream)


########################################################################
########################################################################
########################################################################
########################################################################
# Helper functions and classes


def _reshape_strides(subset, strides, original_strides, copy_shape):
    """ Helper function that reshapes a shape to the given strides. """
    # TODO(later): Address original strides in the computation of the
    #              result strides.
    original_copy_shape = subset.size()
    dims = len(copy_shape)

    reduced_tile_sizes = [
        ts for ts, s in zip(subset.tile_sizes, original_copy_shape) if s != 1
    ]

    reshaped_copy = copy_shape + [ts for ts in subset.tile_sizes if ts != 1]
    reshaped_copy[:len(copy_shape)] = [
        s / ts for s, ts in zip(copy_shape, reduced_tile_sizes)
    ]

    new_strides = [0] * len(reshaped_copy)
    elements_remaining = functools.reduce(sp.mul.Mul, copy_shape, 1)
    tiledim = 0
    for i in range(len(copy_shape)):
        new_strides[i] = elements_remaining / reshaped_copy[i]
        elements_remaining = new_strides[i]
        if reduced_tile_sizes[i] != 1:
            new_strides[dims + tiledim] = (
                elements_remaining / reshaped_copy[dims + tiledim])
            elements_remaining = new_strides[dims + tiledim]
            tiledim += 1

    return reshaped_copy, new_strides


def ndcopy_to_strided_copy(copy_shape, src_shape, src_strides, dst_shape,
                           dst_strides, subset):
    """ Detects situations where an N-dimensional copy can be degenerated into
        a (faster) 1D copy or 2D strided copy. Returns new copy
        dimensions and offsets to emulate the requested copy.

        @return: a 3-tuple: copy_shape, src_strides, dst_strides
    """
    dims = len(copy_shape)

    # Cannot degenerate tiled copies
    if any(ts != 1 for ts in subset.tile_sizes):
        return None

    # 1D copy of the whole array
    if (tuple(copy_shape) == tuple(src_shape)
            and tuple(copy_shape) == tuple(dst_shape)):
        copy_shape = [functools.reduce(lambda x, y: x * y, copy_shape)]
        return copy_shape, [1], [1]
    # 1D strided copy
    elif sum([0 if c == 1 else 1 for c in copy_shape]) == 1:
        # Find the copied dimension:
        # In copy shape
        copydim = next(i for i, c in enumerate(copy_shape) if c != 1)

        # In source strides
        if len(copy_shape) == len(src_shape):
            srcdim = copydim
        else:
            srcdim = next(i for i, c in enumerate(src_shape) if c != 1)

        # In destination strides
        if len(copy_shape) == len(dst_shape):
            dstdim = copydim
        else:
            dstdim = next(i for i, c in enumerate(dst_shape) if c != 1)

        # Return new copy
        return [copy_shape[copydim]], [src_strides[srcdim]], [
            dst_strides[dstdim]
        ]
    else:
        return None


def ndslice_cpp(slice, dims, rowmajor=True):
    result = StringIO()

    if len(slice) == 0:  # Scalar
        return '0'

    for i, d in enumerate(slice):
        if isinstance(d, tuple):
            raise SyntaxError(
                'CPU backend does not yet support ranges as inputs/outputs')

        # TODO(later): Use access order

        result.write(sym2cpp(d))

        # If not last
        if i < len(slice) - 1:
            strdims = [str(dim) for dim in dims[i + 1:]]
            result.write(
                '*%s + ' % '*'.join(strdims))  # Multiply by leading dimensions

    return result.getvalue()


def cpp_offset_expr(d: data.Data,
                    subset_in: subsets.Subset,
                    offset=None,
                    packed_veclen=1):
    """ Creates a C++ expression that can be added to a pointer in order
        to offset it to the beginning of the given subset and offset.
        @param d: The data structure to use for sizes/strides.
        @param subset: The subset to offset by.
        @param offset: An additional list of offsets or a Subset object
        @param packed_veclen: If packed types are targeted, specifies the
                              vector length that the final offset should be 
                              divided by.
        @return: A string in C++ syntax with the correct offset
    """
    subset = copy.deepcopy(subset_in)

    # Offset according to parameters
    if offset is not None:
        if isinstance(offset, subsets.Subset):
            subset.offset(offset, False)
        else:
            subset.offset(subsets.Indices(offset), False)

    # Then, offset according to array
    subset.offset(subsets.Indices(d.offset), False)

    # Obtain start range from offsetted subset
    slice = [0] * len(d.strides)  #subset.min_element()

    index = subset.at(slice, d.strides)
    if packed_veclen > 1:
        index /= packed_veclen

    return sym2cpp(index)


def cpp_array_expr(sdfg,
                   memlet,
                   with_brackets=True,
                   offset=None,
                   relative_offset=True,
                   packed_veclen=1):
    """ Converts an Indices/Range object to a C++ array access string. """
    s = memlet.subset if relative_offset else subsets.Indices(offset)
    o = offset if relative_offset else None
    offset_cppstr = cpp_offset_expr(sdfg.arrays[memlet.data], s, o,
                                    packed_veclen)

    if with_brackets:
        return '%s[%s]' % (memlet.data, offset_cppstr)
    else:
        return offset_cppstr


def write_and_resolve_expr(memlet, nc, outname, inname, indices=None):
    """ Helper function that emits a write_and_resolve call from a memlet. """

    redtype = operations.detect_reduction_type(memlet.wcr)

    nc = '_nc' if nc else ''
    indstr = (', ' + indices) if indices is not None else ''

    reduction_tmpl = ''
    custom_reduction = ''

    # Special call for detected reduction types
    if redtype != types.ReductionType.Custom:
        credtype = ('dace::ReductionType::' +
                    str(redtype)[str(redtype).find('.') + 1:])
        reduction_tmpl = '<%s>' % credtype
    else:
        custom_reduction = ', %s' % unparse_cr(memlet.wcr)

    return '{oname}.write_and_resolve{nc}{tmpl}({iname}{wcr}{ind});'.format(
        oname=outname,
        nc=nc,
        tmpl=reduction_tmpl,
        iname=inname,
        wcr=custom_reduction,
        ind=indstr)


def is_write_conflicted(dfg, edge, datanode=None):
    """ Detects whether a write-conflict-resolving edge can be emitted without
        using atomics or critical sections. """

    if edge.data.wcr_conflict is not None and not edge.data.wcr_conflict:
        return False

    if edge is None:
        start_node = None
        memlet = None
    else:
        start_node = edge.dst
        memlet = edge.data

    # If it's an entire SDFG, it's probably write-conflicted
    if isinstance(dfg, SDFG):
        if datanode is None: return True
        in_edges = find_incoming_edges(datanode, dfg)
        if len(in_edges) != 1: return True
        if (isinstance(in_edges[0].src, nodes.ExitNode) and
                in_edges[0].src.map.schedule == types.ScheduleType.Sequential):
            return False
        return True

    # Traverse memlet path to determine conflicts.
    # If no conflicts will occur, write without atomics
    # (e.g., if the array has been defined in a non-parallel schedule context)
    # TODO: This is not perfect (need to take indices into consideration)
    path = dfg.memlet_path(edge)
    for e in path:
        if (isinstance(e.dst, nodes.ExitNode)
                and e.dst.map.schedule != types.ScheduleType.Sequential):
            return True
        # Should never happen (no such thing as write-conflicting reads)
        if (isinstance(e.src, nodes.EntryNode)
                and e.src.map.schedule != types.ScheduleType.Sequential):
            return True

    return False


def unparse_cr(wcr_ast):
    """ Outputs a C++ version of a conflict resolution lambda. """

    if isinstance(wcr_ast, ast.Lambda):
        return cppunparse.cppunparse(wcr_ast, expr_semicolon=False)
    elif isinstance(wcr_ast, ast.FunctionDef):
        # Construct a lambda function out of a function
        return '[] (%s) { %s }' % (
            cppunparse.cppunparse(wcr_ast.args, expr_semicolon=False),
            cppunparse.cppunparse(wcr_ast.body, expr_semicolon=False))
    elif isinstance(wcr_ast, ast.Module):
        return unparse_cr(wcr_ast.body[0].value)
    elif isinstance(wcr_ast, str):
        return unparse_cr(LambdaProperty.from_string(wcr_ast))
    else:
        raise NotImplementedError('INVALID TYPE OF WCR: ' +
                                  type(wcr_ast).__name__)


def unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                    callsite_stream, locals, ldepth):

    if node.label is None or node.label == "":
        return ''

    state_dfg = sdfg.nodes()[state_id]
    unified_id = PerfUtils.unified_id(dfg.node_id(node), state_id)

    # Not [], "" or None
    if not node.code:
        return ''

    # Not [], "" or None
    if node.code_global:
        if node.language is not types.Language.CPP:
            raise ValueError(
                "Global code only supported for C++ tasklets: got {}".format(
                    node.language))
        function_stream.write(
            type(node).__properties__["code_global"].to_string(
                node.code_global), sdfg, state_id, node)
        function_stream.write("\n", sdfg, state_id, node)

    # If raw C++ code, return the code directly
    if node.language != types.Language.Python:
        # If this code runs on the host and is associated with a CUDA stream,
        # set the stream to a local variable.
        max_streams = Config.get('compiler', 'cuda', 'max_concurrent_streams')
        if (max_streams >= 0 and not is_devicelevel(sdfg, state_dfg, node)
                and hasattr(node, '_cuda_stream')):
            callsite_stream.write(
                'cudaStream_t __dace_current_stream = dace::cuda::__streams[%d];'
                % node._cuda_stream, sdfg, state_id, node)

        if node.language != types.Language.CPP:
            raise ValueError(
                "Only Python or C++ code supported in CPU codegen, got: {}".
                format(node.language))
        callsite_stream.write(
            type(node).__properties__["code"].to_string(node.code), sdfg,
            state_id, node)

        if (hasattr(node, '_cuda_stream')
                and not is_devicelevel(sdfg, state_dfg, node)):
            synchronize_streams(sdfg, state_dfg, state_id, node, node,
                                callsite_stream)
        return

    body = node.code

    # Map local names to memlets (for WCR detection)
    memlets = {}
    for edge in state_dfg.all_edges(node):
        u, uconn, v, vconn, memlet = edge
        if u == node:
            memlet_nc = not is_write_conflicted(dfg, edge)
            memlet_wcr = memlet.wcr

            memlets[uconn] = (memlet, memlet_nc, memlet_wcr)
        elif v == node:
            memlets[vconn] = (memlet, False, None)

    #############################################################
    # Instrumentation: Pre-Tasklet
    if PerfSettings.perf_tasklets and PerfSettings.perf_enable_instrumentation(
    ):
        callsite_stream.write(
            'dace_perf::%s __perf_%s;\n' %
            (PerfUtils.perf_counter_string(node), node.label), sdfg, state_id,
            node)
        callsite_stream.write(
            'auto& __perf_vs_%s = __perf_store.getNewValueSet(__perf_%s, %d, PAPI_thread_id(), 0);\n'
            % (node.label, node.label, unified_id), sdfg, state_id, node)

        callsite_stream.write('__perf_%s.enterCritical();\n' % node.label,
                              sdfg, state_id, node)

    #############################################################

    callsite_stream.write('// Tasklet code (%s)\n' % node.label, sdfg,
                          state_id, node)
    for stmt in body:
        if isinstance(stmt, ast.Expr):
            rk = DaCeKeywordRemover(memlets,
                                    sdfg.constants).visit_TopLevelExpr(stmt)
        else:
            rk = DaCeKeywordRemover(memlets, sdfg.constants).visit(stmt)

        if rk is not None:
            # Unparse to C++ and add 'auto' declarations if locals not declared
            result = StringIO()
            cppunparse.CPPUnparser(rk, ldepth + 1, locals, result)
            callsite_stream.write(result.getvalue(), sdfg, state_id, node)

    #############################################################
    # Instrumentation: Post-Tasklet
    if PerfSettings.perf_tasklets and PerfSettings.perf_enable_instrumentation(
    ):
        callsite_stream.write(
            '__perf_%s.leaveCritical(__perf_vs_%s);' %
            (node.label, node.label), sdfg, state_id, node)
    #############################################################


def is_array_stream_view(sdfg, dfg, node):
    """ Test whether a stream is directly connected to an array. """

    # Test all memlet paths from the array. If the path goes directly
    # to/from a stream, construct a stream array view
    source_paths = []
    sink_paths = []
    for e in dfg.in_edges(node):
        src_node = dfg.memlet_path(e)[0].src
        if (isinstance(src_node, nodes.AccessNode)
                and isinstance(src_node.desc(sdfg), data.Array)):
            source_paths.append(src_node)
    for e in dfg.out_edges(node):
        sink_node = dfg.memlet_path(e)[-1].dst
        if (isinstance(sink_node, nodes.AccessNode)
                and isinstance(sink_node.desc(sdfg), data.Array)):
            sink_paths.append(sink_node)

    # Special case: stream can be represented as a view of an array
    if len(source_paths) == 1 or len(sink_paths) == 1:
        # TODO: What about a source path?
        arrnode = sink_paths[0]
        # Only works if the stream itself is not an array of streams
        if list(node.desc(sdfg).shape) == [1]:
            node.desc(sdfg).sink = arrnode.data  # For memlet generation
            arrnode.desc(
                sdfg).src = node.data  # TODO: Move src/sink to node, not array
            return True
    return False


def find_incoming_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.in_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.in_edges(node))


def find_outgoing_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.out_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.out_edges(node))


def sym2cpp(s):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(s, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(s))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d)) for d in s]


class DaCeKeywordRemover(ExtNodeTransformer):
    """ Removes memlets and other DaCe keywords from a Python AST, and 
        converts array accesses to C++ methods that can be generated.
        
        Used for unparsing Python tasklets into C++ that uses the DaCe 
        runtime.
        
        @note: Assumes that the DaCe syntax is correct (as verified by the
               Python frontend).
    """

    def __init__(self, memlets, constants):
        self.memlets = memlets
        self.constants = constants

    def visit_TopLevelExpr(self, node):
        # This is a DaCe shift, omit it
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.LShift) or isinstance(
                    node.value.op, ast.RShift):
                return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Subscript):
            return self.generic_visit(node)

        target = rname(node.target)
        if target not in self.memlets:
            return self.generic_visit(node)

        raise SyntaxError('Augmented assignments (e.g. +=) not allowed on ' +
                          'array memlets')

    def visit_Assign(self, node):
        target = rname(node.targets[0])
        if target not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr = self.memlets[target]
        value = self.visit(node.value)

        if not isinstance(node.targets[0], ast.Subscript):
            # Dynamic accesses -> every access counts
            try:
                if memlet is not None and memlet.num_accesses < 0:
                    if wcr is not None:
                        newnode = ast.Name(
                            id=write_and_resolve_expr(
                                memlet, nc, '__' + target,
                                cppunparse.cppunparse(
                                    value, expr_semicolon=False)))
                    else:
                        newnode = ast.Name(id='__%s.write(%s);' % (
                            target,
                            cppunparse.cppunparse(value, expr_semicolon=False))
                                           )

                    return ast.copy_location(newnode, node)
            except TypeError:  # cannot determine truth value of Relational
                pass

            return self.generic_visit(node)

        slice = self.visit(node.targets[0].slice)
        if not isinstance(slice, ast.Index):
            raise NotImplementedError('Range subscripting not implemented')

        if isinstance(slice.value, ast.Tuple):
            subscript = unparse(slice)[1:-1]
        else:
            subscript = unparse(slice)

        if wcr is not None:
            newnode = ast.Name(
                id=write_and_resolve_expr(
                    memlet,
                    nc,
                    '__' + target,
                    cppunparse.cppunparse(value, expr_semicolon=False),
                    indices=subscript))
        else:
            newnode = ast.Name(id='__%s.write(%s, %s);' % (
                target, cppunparse.cppunparse(value, expr_semicolon=False),
                subscript))

        return ast.copy_location(newnode, node)

    def visit_Subscript(self, node):
        target = rname(node)
        if target not in self.memlets and target not in self.constants:
            return self.generic_visit(node)

        slice = self.visit(node.slice)
        if not isinstance(slice, ast.Index):
            raise NotImplementedError('Range subscripting not implemented')

        if isinstance(slice.value, ast.Tuple):
            subscript = unparse(slice)[1:-1]
        else:
            subscript = unparse(slice)

        if target in self.constants:
            slice_str = ndslice_cpp(
                subscript.split(', '), self.constants[target].shape)
            newnode = ast.parse('%s[%s]' % (target, slice_str)).body[0].value
        else:
            newnode = ast.parse('__%s(%s)' % (target, subscript)).body[0].value
        return ast.copy_location(newnode, node)

    def visit_Expr(self, node):
        # Check for DaCe function calls
        if isinstance(node.value, ast.Call):
            # Some calls should not be parsed
            if rname(node.value.func) == "define_local":
                return None
            elif rname(node.value.func) == "define_local_scalar":
                return None
            elif rname(node.value.func) == "define_stream":
                return None
            elif rname(node.value.func) == "define_streamarray":
                return None

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Do not parse internal functions
        return None

    # Replace default modules (e.g., math) with dace::math::
    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind('.')]
        func_name = attrname[attrname.rfind('.') + 1:]
        if module_name in types._ALLOWED_MODULES:
            cppmodname = types._ALLOWED_MODULES[module_name]
            return ast.copy_location(
                ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
        return self.generic_visit(node)


def unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def presynchronize_streams(sdfg, dfg, state_id, node, callsite_stream):
    state_dfg = sdfg.nodes()[state_id]
    if hasattr(node, '_cuda_stream') or is_devicelevel(sdfg, state_dfg, node):
        return
    for e in state_dfg.in_edges(node):
        if hasattr(e.src, '_cuda_stream'):
            cudastream = 'dace::cuda::__streams[%d]' % e.src._cuda_stream
            callsite_stream.write('cudaStreamSynchronize(%s);' % cudastream,
                                  sdfg, state_id, [e.src, e.dst])


# TODO: This should be in the CUDA code generator. Add appropriate conditions to node dispatch predicate
def synchronize_streams(sdfg, dfg, state_id, node, scope_exit,
                        callsite_stream):
    # Post-kernel stream synchronization (with host or other streams)
    max_streams = Config.get('compiler', 'cuda', 'max_concurrent_streams')
    if max_streams >= 0:
        cudastream = 'dace::cuda::__streams[%d]' % node._cuda_stream
        for edge in dfg.out_edges(scope_exit):
            # Synchronize end of kernel with output data (multiple kernels
            # lead to same data node)
            if (isinstance(edge.dst, nodes.AccessNode)
                    and edge.dst._cuda_stream != node._cuda_stream):
                callsite_stream.write(
                    '''cudaEventRecord(dace::cuda::__events[{ev}], {src_stream});
cudaStreamWaitEvent(dace::cuda::__streams[{dst_stream}], dace::cuda::__events[{ev}], 0);'''
                    .format(
                        ev=edge._cuda_event,
                        src_stream=cudastream,
                        dst_stream=edge.dst._cuda_stream), sdfg, state_id,
                    [edge.src, edge.dst])
                continue

            # We need the streams leading out of the output data
            for e in dfg.out_edges(edge.dst):
                if isinstance(e.dst, nodes.AccessNode):
                    continue
                # If no stream at destination: synchronize stream with host.
                if not hasattr(e.dst, '_cuda_stream'):
                    pass
                    # Done at destination

                # If different stream at destination: record event and wait
                # for it in target stream.
                elif e.dst._cuda_stream != node._cuda_stream:
                    callsite_stream.write(
                        '''cudaEventRecord(dace::cuda::__events[{ev}], {src_stream});
    cudaStreamWaitEvent(dace::cuda::__streams[{dst_stream}], dace::cuda::__events[{ev}], 0);'''
                        .format(
                            ev=e._cuda_event,
                            src_stream=cudastream,
                            dst_stream=e.dst._cuda_stream), sdfg, state_id,
                        [e.src, e.dst])
                # Otherwise, no synchronization necessary
