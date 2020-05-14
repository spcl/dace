import itertools
import warnings

from dace import data, registry, memlet as mm
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.common import codeblock_to_cpp
from dace.codegen.targets.cpp import *
from dace.codegen.targets.target import TargetCodeGenerator, make_absolute, \
    DefinedType
from dace.graph import nodes, nxutil
from dace.sdfg import (ScopeSubgraphView, SDFG, scope_contains_scope,
                       is_devicelevel, is_array_stream_view,
                       NodeNotExpandedError)


@registry.autoregister_params(name='cpu')
class CPUCodeGen(TargetCodeGenerator):
    """ SDFG CPU code generator. """

    title = "CPU"
    target_name = "cpu"
    language = "cpp"

    def __init__(self, frame_codegen, sdfg):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        self._locals = cppunparse.CPPLocals()
        # Scope depth (for use of the 'auto' keyword when
        # defining locals)
        self._ldepth = 0

        # Keep nested SDFG schedule when descending into it
        self._toplevel_schedule = None

        # FIXME: this allows other code generators to change the CPU
        # behavior to assume that arrays point to packed types, thus dividing
        # all addresess by the vector length.
        self._packed_types = False

        # Keep track of traversed nodes
        self._generated_nodes = set()
        # Keeps track of generated connectors, so we know how to access them in
        # nested scopes
        for name, arg_type in sdfg.arglist().items():
            if (isinstance(arg_type, dace.data.Scalar)
                    or isinstance(arg_type, dace.dtypes.typeclass)):
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
                raise TypeError(
                    "Unrecognized argument type: {t} (value {v})".format(
                        t=type(arg_type).__name__, v=str(arg_type)))

        # Register dispatchers
        dispatcher.register_node_dispatcher(self)
        dispatcher.register_map_dispatcher([
            dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential
        ], self)

        cpu_storage = [
            dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal,
            dtypes.StorageType.Register
        ]
        dispatcher.register_array_dispatcher(cpu_storage, self)

        # Register CPU copies (all internal pairs)
        for src_storage, dst_storage in itertools.product(
                cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None,
                                                self)

    @staticmethod
    def cmake_options():
        options = []

        if Config.get('compiler', 'cpu', 'executable'):
            compiler = make_absolute(
                Config.get('compiler', 'cpu', 'executable'))
            options.append('-DCMAKE_CXX_COMPILER="{}"'.format(compiler))

        if Config.get('compiler', 'cpu', 'args'):
            flags = Config.get('compiler', 'cpu', 'args')
            options.append('-DCMAKE_CXX_FLAGS="{}"'.format(flags))

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

    def generate_scope(
        self,
        sdfg: SDFG,
        dfg_scope: ScopeSubgraphView,
        state_id,
        function_stream,
        callsite_stream,
    ):
        entry_node = dfg_scope.source_nodes()[0]
        presynchronize_streams(sdfg, dfg_scope, state_id, entry_node,
                               callsite_stream)

        self.generate_node(sdfg, dfg_scope, state_id, entry_node,
                           function_stream, callsite_stream)
        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        # Dynamically obtain node generator according to class name
        try:
            gen = getattr(self, "_generate_" + type(node).__name__)
        except AttributeError:
            if isinstance(node, nodes.LibraryNode):
                raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
            raise

        gen(sdfg, dfg, state_id, node, function_stream, callsite_stream)

        # Mark node as "generated"
        self._generated_nodes.add(node)

        self._locals.clear_scope(self._ldepth + 1)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)

        if nodedesc.transient is False:
            return

        # Check if array is already allocated
        try:
            self._dispatcher.defined_vars.get(name)
            return  # Array was already allocated in this or upper scopes
        except KeyError:  # Array not allocated yet
            pass

        # Compute array size
        arrsize = nodedesc.total_size

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
                        "Stream-view of array may not be defined "
                        "in more than one state")

                arrnode = sdfg.arrays[nodedesc.sink]
                state = sdfg.nodes()[state_id]
                edges = state.out_edges(node)
                if len(edges) > 1:
                    raise NotImplementedError("Cannot handle streams writing "
                                              "to multiple arrays.")

                memlet_path = state.memlet_path(edges[0])
                # Allocate the array before its stream view, if necessary
                self.allocate_array(
                    sdfg,
                    dfg,
                    state_id,
                    memlet_path[-1].dst,
                    function_stream,
                    callsite_stream,
                )

                array_expr = copy_expr(self._dispatcher,
                                       sdfg,
                                       nodedesc.sink,
                                       edges[0].data,
                                       packed_types=self._packed_types)
                threadlocal = ""
                threadlocal_stores = [
                    dtypes.StorageType.CPU_ThreadLocal,
                    dtypes.StorageType.Register
                ]
                if (sdfg.arrays[nodedesc.sink].storage in threadlocal_stores
                        or nodedesc.storage in threadlocal_stores):
                    threadlocal = "Threadlocal"
                callsite_stream.write(
                    "dace::ArrayStreamView%s<%s> %s (%s);\n" %
                    (threadlocal, arrnode.dtype.ctype, name, array_expr),
                    sdfg,
                    state_id,
                    node,
                )
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

        # TODO: immaterial arrays should not allocate memory
        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or nodedesc.storage == dtypes.StorageType.Immaterial
              or (nodedesc.storage == dtypes.StorageType.Register
                  and symbolic.issymbolic(arrsize, sdfg.constants))):

            if nodedesc.storage == dtypes.StorageType.Register:
                warnings.warn('Variable-length array %s with size %s '
                              'detected and was allocated on heap instead of '
                              '%s' %
                              (name, sym2cpp(arrsize), nodedesc.storage))

            callsite_stream.write(
                "%s *%s = new %s DACE_ALIGN(64)[%s];\n" %
                (nodedesc.dtype.ctype, name, nodedesc.dtype.ctype,
                 sym2cpp(arrsize)),
                sdfg,
                state_id,
                node,
            )
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)

            if node.setzero:
                callsite_stream.write(
                    "memset(%s, 0, sizeof(%s)*%s);" %
                    (name, nodedesc.dtype.ctype, sym2cpp(arrsize)))
            return
        elif (nodedesc.storage == dtypes.StorageType.Register):
            if node.setzero:
                callsite_stream.write(
                    "%s %s[%s]  DACE_ALIGN(64) = {0};\n" %
                    (nodedesc.dtype.ctype, name, sym2cpp(arrsize)),
                    sdfg,
                    state_id,
                    node,
                )
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
                return
            callsite_stream.write(
                "%s %s[%s]  DACE_ALIGN(64);\n" %
                (nodedesc.dtype.ctype, name, sym2cpp(arrsize)),
                sdfg,
                state_id,
                node,
            )
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Define pointer once
            # NOTE: OpenMP threadprivate storage MUST be declared globally.
            if not self._dispatcher.defined_vars.has(name):
                function_stream.write(
                    "{ctype} *{name};\n#pragma omp threadprivate({name})".
                    format(ctype=nodedesc.dtype.ctype, name=name),
                    sdfg,
                    state_id,
                    node,
                )
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer)

            # Allocate in each OpenMP thread
            callsite_stream.write(
                """
                #pragma omp parallel
                {{
                    {name} = new {ctype} DACE_ALIGN(64)[{arrsize}];""".format(
                    ctype=nodedesc.dtype.ctype,
                    name=name,
                    arrsize=sym2cpp(arrsize)),
                sdfg,
                state_id,
                node,
            )
            if node.setzero:
                callsite_stream.write(
                    "memset(%s, 0, sizeof(%s)*%s);" %
                    (name, nodedesc.dtype.ctype, sym2cpp(arrsize)))
            # Close OpenMP parallel section
            callsite_stream.write('}')
        else:
            raise NotImplementedError("Unimplemented storage type " +
                                      str(nodedesc.storage))

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        nodedesc = node.desc(sdfg)
        arrsize = nodedesc.total_size
        if isinstance(nodedesc, data.Scalar):
            return
        elif isinstance(nodedesc, data.Stream):
            return
        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Register
                  and symbolic.issymbolic(arrsize, sdfg.constants))):
            callsite_stream.write("delete[] %s;\n" % node.data, sdfg, state_id,
                                  node)
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Deallocate in each OpenMP thread
            callsite_stream.write(
                """#pragma omp parallel
                {{
                    delete[] {name};
                }}""".format(name=node.data),
                sdfg,
                state_id,
                node,
            )
        else:
            return

    def copy_memory(
        self,
        sdfg,
        dfg,
        state_id,
        src_node,
        dst_node,
        edge,
        function_stream,
        callsite_stream,
    ):
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            try:
                src_parent = dfg.scope_dict()[src_node]
            except KeyError:
                src_parent = None
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.scope_dict()[dst_node]
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = sdfg.nodes()[state_id]

        # Emit actual copy
        self._emit_copy(
            sdfg,
            state_id,
            src_node,
            src_storage,
            dst_node,
            dst_storage,
            dst_schedule,
            edge,
            state_dfg,
            callsite_stream,
        )

    def _emit_copy(
        self,
        sdfg,
        state_id,
        src_node,
        src_storage,
        dst_node,
        dst_storage,
        dst_schedule,
        edge,
        dfg,
        stream,
    ):
        u, uconn, v, vconn, memlet = edge

        # Determine memlet directionality
        if isinstance(src_node,
                      nodes.AccessNode) and memlet.data == src_node.data:
            write = True
        elif isinstance(dst_node,
                        nodes.AccessNode) and memlet.data == dst_node.data:
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(
                dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError(
                "Copying between code nodes is only supported as"
                " part of the participating nodes")
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, False, vconn),
                sdfg,
                state_id,
                [src_node, dst_node],
            )
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, True, uconn),
                sdfg,
                state_id,
                [src_node, dst_node],
            )
            return
        else:  # Copy array-to-array
            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)

            if write:
                vconn = dst_node.data
            ctype = "dace::vec<%s, %d>" % (dst_nodedesc.dtype.ctype,
                                           memlet.veclen)

            #############################################
            # Corner cases

            # Writing one index
            if (isinstance(memlet.subset, subsets.Indices)
                    and memlet.wcr is None
                    and self._dispatcher.defined_vars.get(
                        vconn) == DefinedType.Scalar):
                stream.write(
                    "%s = %s;" %
                    (vconn, self.memlet_ctor(sdfg, memlet, False)),
                    sdfg,
                    state_id,
                    [src_node, dst_node],
                )
                return
            # Writing from/to a stream
            if isinstance(sdfg.arrays[memlet.data], data.Stream) or (
                    isinstance(src_node, nodes.AccessNode)
                    and isinstance(src_nodedesc, data.Stream)):
                # Identify whether a stream is writing to an array
                if isinstance(dst_nodedesc,
                              (data.Scalar, data.Array)) and isinstance(
                                  src_nodedesc, data.Stream):
                    # Stream -> Array - pop bulk
                    if is_array_stream_view(sdfg, dfg, src_node):
                        return  # Do nothing (handled by ArrayStreamView)

                    array_subset = (memlet.subset
                                    if memlet.data == dst_node.data else
                                    memlet.other_subset)
                    if array_subset is None:  # Need to use entire array
                        array_subset = subsets.Range.from_array(dst_nodedesc)

                    # stream_subset = (memlet.subset
                    #                  if memlet.data == src_node.data else
                    #                  memlet.other_subset)
                    stream_subset = memlet.subset
                    if memlet.data != src_node.data and memlet.other_subset:
                        stream_subset = memlet.other_subset

                    stream_expr = cpp_offset_expr(src_nodedesc, stream_subset)
                    array_expr = cpp_offset_expr(dst_nodedesc, array_subset)
                    assert functools.reduce(lambda a, b: a * b,
                                            src_nodedesc.shape, 1) == 1
                    stream.write(
                        "{s}.pop(&{arr}[{aexpr}], {maxsize});".format(
                            s=src_node.data,
                            arr=dst_node.data,
                            aexpr=array_expr,
                            maxsize=sym2cpp(array_subset.num_elements())),
                        sdfg,
                        state_id,
                        [src_node, dst_node],
                    )
                    return
                # Array -> Stream - push bulk
                if isinstance(src_nodedesc,
                              (data.Scalar, data.Array)) and isinstance(
                                  dst_nodedesc, data.Stream):
                    if hasattr(src_nodedesc, "src"):  # ArrayStreamView
                        stream.write(
                            "{s}.push({arr});".format(s=dst_node.data,
                                                      arr=src_nodedesc.src),
                            sdfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    else:
                        copysize = " * ".join(
                            [sym2cpp(s) for s in memlet.subset.size()])
                        stream.write(
                            "{s}.push({arr}, {size});".format(
                                s=dst_node.data,
                                arr=src_node.data,
                                size=copysize),
                            sdfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    return
                else:
                    # Unknown case
                    raise NotImplementedError

            #############################################

            state_dfg = sdfg.nodes()[state_id]

            copy_shape, src_strides, dst_strides, src_expr, dst_expr = \
                memlet_copy_to_absolute_strides(
                    self._dispatcher, sdfg, memlet, src_node, dst_node,
                    self._packed_types)

            # Which numbers to include in the variable argument part
            dynshape, dynsrc, dyndst = 1, 1, 1

            # Dynamic copy dimensions
            if any(symbolic.issymbolic(s, sdfg.constants) for s in copy_shape):
                copy_tmpl = "Dynamic<{type}, {veclen}, {aligned}, {dims}>".format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned="false",
                    dims=len(copy_shape),
                )
            else:  # Static copy dimensions
                copy_tmpl = "<{type}, {veclen}, {aligned}, {dims}>".format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned="false",
                    dims=", ".join(sym2cpp(copy_shape)),
                )
                dynshape = 0

            # Constant src/dst dimensions
            if not any(
                    symbolic.issymbolic(s, sdfg.constants)
                    for s in dst_strides):
                # Constant destination
                shape_tmpl = "template ConstDst<%s>" % ", ".join(
                    sym2cpp(dst_strides))
                dyndst = 0
            elif not any(
                    symbolic.issymbolic(s, sdfg.constants)
                    for s in src_strides):
                # Constant source
                shape_tmpl = "template ConstSrc<%s>" % ", ".join(
                    sym2cpp(src_strides))
                dynsrc = 0
            else:
                # Both dynamic
                shape_tmpl = "Dynamic"

            # Parameter pack handling
            stride_tmpl_args = [0] * (dynshape + dynsrc +
                                      dyndst) * len(copy_shape)
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

            copy_args = ([src_expr, dst_expr] +
                         ([] if memlet.wcr is None else
                          [unparse_cr(sdfg, memlet.wcr)]) +
                         sym2cpp(stride_tmpl_args))

            # Instrumentation: Pre-copy
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_copy_begin(sdfg, state_dfg, src_node, dst_node,
                                        edge, stream, None, copy_shape,
                                        src_strides, dst_strides)

            nc = True
            if memlet.wcr is not None:
                nc = not is_write_conflicted(
                    dfg, edge, sdfg_schedule=self._toplevel_schedule)
            if nc:
                stream.write(
                    """
                    dace::CopyND{copy_tmpl}::{shape_tmpl}::{copy_func}(
                        {copy_args});""".format(
                        copy_tmpl=copy_tmpl,
                        shape_tmpl=shape_tmpl,
                        copy_func="Copy"
                        if memlet.wcr is None else "Accumulate",
                        copy_args=", ".join(copy_args),
                    ),
                    sdfg,
                    state_id,
                    [src_node, dst_node],
                )
            else:  # Conflicted WCR
                if dynshape == 1:
                    raise NotImplementedError(
                        "Accumulation of dynamically-shaped "
                        "arrays not yet implemented")
                elif copy_shape == [
                        1
                ]:  # Special case: accumulating one element
                    dst_expr = self.memlet_view_ctor(sdfg, memlet, True)
                    stream.write(
                        write_and_resolve_expr(sdfg, memlet, nc, dst_expr,
                                               '*(' + src_expr + ')'), sdfg,
                        state_id, [src_node, dst_node])
                else:
                    raise NotImplementedError("Accumulation of arrays "
                                              "with WCR not yet implemented")

        #############################################################
        # Instrumentation: Post-copy
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_copy_end(sdfg, state_dfg, src_node, dst_node, edge,
                                  stream, None)
        #############################################################

    ###########################################################################
    # Memlet handling

    def process_out_memlets(self,
                            sdfg,
                            state_id,
                            node,
                            dfg,
                            dispatcher,
                            result,
                            locals_defined,
                            function_stream,
                            skip_wcr=False):

        scope_dict = sdfg.nodes()[state_id].scope_dict()

        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            if skip_wcr and memlet.wcr is not None:
                continue
            dst_node = dfg.memlet_path(edge)[-1].dst

            # Target is neither a data nor a tasklet node
            if isinstance(node, nodes.AccessNode) and (
                    not isinstance(dst_node, nodes.AccessNode)
                    and not isinstance(dst_node, nodes.CodeNode)):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(
                    v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(dst_node, nodes.CodeNode) and edge.src_conn:
                shared_data_name = edge.data.data
                result.write(
                    "%s = %s;" % (shared_data_name, edge.src_conn),
                    sdfg,
                    state_id,
                    [edge.src, edge.dst],
                )
                continue

            # If the memlet is not pointing to a data node (e.g. tasklet), then
            # the tasklet will take care of the copy
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            # If the memlet is pointing into an array in an inner scope, then
            # the inner scope (i.e., the output array) must handle it
            if scope_dict[
                    node] != scope_dict[dst_node] and scope_contains_scope(
                        scope_dict, node, dst_node):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array
            if isinstance(node, nodes.CodeNode):
                if not uconn:
                    raise SyntaxError(
                        "Cannot copy memlet without a local connector: {} to {}"
                        .format(str(edge.src), str(edge.dst)))

                could_be_scalar = True
                if isinstance(node, nodes.NestedSDFG):
                    could_be_scalar = not isinstance(node.sdfg.arrays[uconn],
                                                     data.Array)

                try:
                    positive_accesses = bool(memlet.num_accesses >= 0)
                except TypeError:
                    positive_accesses = False

                if (memlet.subset.data_dims() == 0 and positive_accesses
                        and could_be_scalar):
                    out_local_name = "    __" + uconn
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
                        nc = not is_write_conflicted(
                            dfg, edge, sdfg_schedule=self._toplevel_schedule)
                        result.write(
                            write_and_resolve_expr(sdfg, memlet, nc,
                                                   out_local_name,
                                                   in_local_name), sdfg,
                            state_id, node)
                    else:
                        result.write(
                            "%s.write(%s);\n" %
                            (out_local_name, in_local_name),
                            sdfg,
                            state_id,
                            node,
                        )
            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                if dst_node != node and not isinstance(dst_node,
                                                       nodes.Tasklet):
                    dispatcher.dispatch_copy(
                        node,
                        dst_node,
                        edge,
                        sdfg,
                        dfg,
                        state_id,
                        function_stream,
                        result,
                    )

    def memlet_view_ctor(self, sdfg, memlet, is_output):
        memlet_params = []

        memlet_name = memlet.data
        def_type = self._dispatcher.defined_vars.get(memlet_name)

        if def_type == DefinedType.Pointer:
            memlet_expr = memlet_name  # Common case
        elif def_type == DefinedType.Scalar:
            memlet_expr = "&" + memlet_name
        elif def_type == DefinedType.ArrayView:
            memlet_expr = memlet_name + ".ptr()"
        else:
            raise TypeError("Unsupported connector type {}".format(def_type))

        if isinstance(memlet.subset, subsets.Indices):

            # FIXME: _packed_types influences how this offset is
            # generated from the FPGA codegen. We should find a nicer solution.
            if self._packed_types is True:
                offset = cpp_array_expr(sdfg,
                                        memlet,
                                        False,
                                        packed_veclen=memlet.veclen)
            else:
                offset = cpp_array_expr(sdfg, memlet, False)

            # Compute address
            memlet_params.append(memlet_expr + " + " + offset)
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
                        packed_veclen=memlet.veclen,
                    )
                else:
                    offset = cpp_offset_expr(sdfg.arrays[memlet.data],
                                             memlet.subset)
                if offset == "0":
                    memlet_params.append(memlet_expr)
                else:
                    if def_type not in [
                            DefinedType.Pointer, DefinedType.ArrayView
                    ]:
                        from dace.codegen.codegen import CodegenError
                        raise CodegenError(
                            "Cannot offset address of connector {} of type {}".
                            format(memlet_name, def_type))
                    memlet_params.append(memlet_expr + " + " + offset)

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
                raise RuntimeError('Memlet type "%s" not implemented' %
                                   memlet.subset)

        if memlet.num_accesses == 1:
            num_accesses_str = "1"
        else:  # symbolic.issymbolic(memlet.num_accesses, sdfg.constants):
            num_accesses_str = "dace::NA_RUNTIME"

        return "dace::ArrayView%s<%s, %d, %s, %s> (%s)" % (
            "Out" if is_output else "In",
            sdfg.arrays[memlet.data].dtype.ctype,
            dims,
            sym2cpp(memlet.veclen),
            num_accesses_str,
            ", ".join(memlet_params),
        )

    def memlet_definition(self,
                          sdfg: SDFG,
                          memlet: mm.Memlet,
                          output: bool,
                          local_name: str,
                          conntype: data.Data = None,
                          allow_shadowing=False):
        could_be_scalar = not conntype or not isinstance(conntype, data.Array)
        result = ("auto __%s = " % local_name +
                  self.memlet_ctor(sdfg, memlet, output) + ";\n")

        # Allocate variable type
        memlet_type = "dace::vec<%s, %s>" % (
            sdfg.arrays[memlet.data].dtype.ctype,
            sym2cpp(memlet.veclen),
        )

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
                self._dispatcher.defined_vars.add(
                    local_name,
                    DefinedType.Scalar,
                    allow_shadowing=allow_shadowing)
            else:
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
                self._dispatcher.defined_vars.add(
                    local_name,
                    DefinedType.Scalar,
                    allow_shadowing=allow_shadowing)
        elif var_type == DefinedType.Pointer:
            if (memlet.num_accesses == 1 and memlet.subset.num_elements() == 1
                    and could_be_scalar):
                if output:
                    result += "{} {};".format(memlet_type, local_name)
                else:
                    result += "{} {} = __{}.val<{}>();".format(
                        memlet_type, local_name, local_name, memlet.veclen)
                self._dispatcher.defined_vars.add(
                    local_name,
                    DefinedType.Scalar,
                    allow_shadowing=allow_shadowing)
            else:
                if memlet.subset.data_dims() == 0 and could_be_scalar:
                    # Forward ArrayView
                    result += "auto &{} = __{}.ref<{}>();".format(
                        local_name, local_name, memlet.veclen)
                    self._dispatcher.defined_vars.add(
                        local_name,
                        DefinedType.Scalar,
                        allow_shadowing=allow_shadowing)
                else:
                    result += "auto *{} = __{}.ptr<{}>();".format(
                        local_name, local_name, memlet.veclen)
                    self._dispatcher.defined_vars.add(
                        local_name,
                        DefinedType.Pointer,
                        allow_shadowing=allow_shadowing)
        elif var_type in [
                DefinedType.Stream, DefinedType.StreamArray,
                DefinedType.StreamView
        ]:
            if memlet.num_accesses == 1:
                if output:
                    result += "{} {};".format(memlet_type, local_name)
                else:
                    result += "auto {} = __{}.pop();".format(
                        local_name, local_name)
                self._dispatcher.defined_vars.add(
                    local_name,
                    DefinedType.Scalar,
                    allow_shadowing=allow_shadowing)
            else:
                # Just forward actions to the underlying object
                result += "auto &{} = __{};".format(local_name, local_name)
                self._dispatcher.defined_vars.add(
                    local_name,
                    DefinedType.StreamView,
                    allow_shadowing=allow_shadowing)
        else:
            raise TypeError("Unknown variable type: {}".format(var_type))

        return result

    def memlet_stream_ctor(self, sdfg, memlet):
        def_type = self._dispatcher.defined_vars.get(memlet.data)

        stream = sdfg.arrays[memlet.data]
        expr = memlet.data + ("[{}]".format(
            cpp_offset_expr(stream, memlet.subset)) if isinstance(
                stream, dace.data.Stream) and stream.is_stream_array() else "")

        if def_type == DefinedType.StreamView:
            return expr

        return "dace::make_streamview({})".format(expr)

    def memlet_ctor(self, sdfg, memlet, is_output):

        def_type = self._dispatcher.defined_vars.get(memlet.data)

        if def_type in [
                DefinedType.Stream, DefinedType.StreamArray,
                DefinedType.StreamView
        ]:
            return self.memlet_stream_ctor(sdfg, memlet)

        elif def_type in [
                DefinedType.Pointer, DefinedType.Scalar, DefinedType.ArrayView
        ]:
            return self.memlet_view_ctor(sdfg, memlet, is_output)

        else:
            raise NotImplementedError(
                "Connector type {} not yet implemented".format(def_type))

    #########################################################################
    # Dynamically-called node dispatchers

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        outer_stream_begin = CodeIOStream()
        outer_stream_end = CodeIOStream()
        inner_stream = CodeIOStream()

        state_dfg = sdfg.nodes()[state_id]

        # Prepare preamble and code for after memlets
        after_memlets_stream = CodeIOStream()
        self.generate_tasklet_preamble(sdfg, dfg, state_id, node,
                                       function_stream, callsite_stream,
                                       after_memlets_stream)

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in state_dfg.in_edges(node):
            u = edge.src
            memlet = edge.data
            src_node = state_dfg.memlet_path(edge)[0].src

            if edge.dst_conn:  # Not (None or "")
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError("Duplicates found in memlets")
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data

                    # Read variable from shared storage
                    inner_stream.write(
                        "const dace::vec<%s, %s>& %s = %s;" % (
                            sdfg.arrays[memlet.data].dtype.ctype,
                            sym2cpp(memlet.veclen),
                            edge.dst_conn,
                            shared_data_name,
                        ),
                        sdfg,
                        state_id,
                        [edge.src, edge.dst],
                    )
                    self._dispatcher.defined_vars.add(edge.dst_conn,
                                                      DefinedType.Scalar)

                else:
                    self._dispatcher.dispatch_copy(
                        src_node,
                        node,
                        edge,
                        sdfg,
                        state_dfg,
                        state_id,
                        function_stream,
                        inner_stream,
                    )

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1,
                                    sdfg.arrays[memlet.data].dtype.ctype)
                arrays.add(edge.dst_conn)

        inner_stream.write("\n", sdfg, state_id, node)

        # Use outgoing edges to preallocate output local vars
        # in two stages: first we preallocate for data<->code cases,
        # followed by code<->code
        tasklet_out_connectors = set()
        for edge in state_dfg.out_edges(node):
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                # Handling this in a separate pass just below
                continue

            if edge.src_conn:
                if edge.src_conn in tasklet_out_connectors:  # Disallow duplicates
                    continue
                else:
                    self._dispatcher.dispatch_copy(
                        node,
                        dst_node,
                        edge,
                        sdfg,
                        state_dfg,
                        state_id,
                        function_stream,
                        inner_stream,
                    )

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1,
                                    sdfg.arrays[edge.data.data].dtype.ctype)
                tasklet_out_connectors.add(edge.src_conn)

        for edge in state_dfg.out_edges(node):
            # Special case: code->code
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if edge.src_conn is None:
                continue
            elif (isinstance(dst_node, nodes.CodeNode)
                  and edge.src_conn not in tasklet_out_connectors):
                memlet = edge.data

                # Generate register definitions for inter-tasklet memlets
                local_name = edge.data.data
                # Allocate variable type
                code = "dace::vec<%s, %s> %s;" % (
                    sdfg.arrays[edge.data.data].dtype.ctype,
                    sym2cpp(edge.data.veclen),
                    local_name,
                )
                outer_stream_begin.write(code, sdfg, state_id,
                                         [edge.src, dst_node])
                arg_type = sdfg.arrays[edge.data.data]
                if (isinstance(arg_type, dace.data.Scalar)
                        or isinstance(arg_type, dace.dtypes.typeclass)):
                    self._dispatcher.defined_vars.add(local_name,
                                                      DefinedType.Scalar,
                                                      ancestor=1)
                elif isinstance(arg_type, dace.data.Array):
                    self._dispatcher.defined_vars.add(local_name,
                                                      DefinedType.Pointer,
                                                      ancestor=1)
                elif isinstance(arg_type, dace.data.Stream):
                    if arg_type.is_stream_array():
                        self._dispatcher.defined_vars.add(
                            local_name, DefinedType.StreamArray, ancestor=1)
                    else:
                        self._dispatcher.defined_vars.add(local_name,
                                                          DefinedType.Stream,
                                                          ancestor=1)
                else:
                    raise TypeError("Unrecognized argument type: {}".format(
                        type(arg_type).__name__))

                inner_stream.write(
                    "dace::vec<%s, %s> %s;" % (
                        sdfg.arrays[memlet.data].dtype.ctype,
                        sym2cpp(memlet.veclen),
                        edge.src_conn,
                    ),
                    sdfg,
                    state_id,
                    [edge.src, edge.dst],
                )
                tasklet_out_connectors.add(edge.src_conn)
                self._dispatcher.defined_vars.add(edge.src_conn,
                                                  DefinedType.Scalar)
                self._locals.define(edge.src_conn, -1, self._ldepth + 1,
                                    sdfg.arrays[memlet.data].dtype.ctype)
                locals_defined = True

        # Emit post-memlet tasklet preamble code
        callsite_stream.write(after_memlets_stream.getvalue())

        # Instrumentation: Pre-tasklet
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, state_dfg, node, outer_stream_begin,
                                inner_stream, function_stream)

        inner_stream.write("\n    ///////////////////\n", sdfg, state_id, node)

        unparse_tasklet(sdfg, state_id, dfg, node, function_stream,
                        inner_stream, self._locals, self._ldepth,
                        self._toplevel_schedule)

        inner_stream.write("    ///////////////////\n\n", sdfg, state_id, node)

        # Generate pre-memlet tasklet postamble
        after_memlets_stream = CodeIOStream()
        self.generate_tasklet_postamble(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream,
                                        after_memlets_stream)

        # Process outgoing memlets
        self.process_out_memlets(
            sdfg,
            state_id,
            node,
            state_dfg,
            self._dispatcher,
            inner_stream,
            True,
            function_stream,
        )

        # Instrumentation: Post-tasklet
        if instr is not None:
            instr.on_node_end(sdfg, state_dfg, node, outer_stream_end,
                              inner_stream, function_stream)

        callsite_stream.write(outer_stream_begin.getvalue(), sdfg, state_id,
                              node)
        callsite_stream.write("{\n", sdfg, state_id, node)
        callsite_stream.write(inner_stream.getvalue(), sdfg, state_id, node)
        callsite_stream.write("}\n", sdfg, state_id, node)
        callsite_stream.write(outer_stream_end.getvalue(), sdfg, state_id,
                              node)

        self._dispatcher.defined_vars.exit_scope(node)

        callsite_stream.write(after_memlets_stream.getvalue())

    def _generate_EmptyTasklet(self, sdfg, dfg, state_id, node,
                               function_stream, callsite_stream):
        self._generate_Tasklet(sdfg, dfg, state_id, node, function_stream,
                               callsite_stream)

    def _generate_NestedSDFG(
        self,
        sdfg,
        dfg: ScopeSubgraphView,
        state_id,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        callsite_stream.write('{', sdfg, state_id, node)
        self._dispatcher.defined_vars.enter_scope(sdfg)

        # If SDFG parent is not set, set it
        state_dfg = sdfg.nodes()[state_id]
        node.sdfg.parent = state_dfg
        node.sdfg._parent_sdfg = sdfg

        # Connectors that are both input and output share the same name
        inout = set(node.in_connectors & node.out_connectors)

        # TODO: Emit nested SDFG as a separate function

        # Emit accessors as pointers/references (rather than new objects)
        # Take care of nested SDFG I/O
        for _, _, _, vconn, in_memlet in state_dfg.in_edges(node):
            if vconn in inout or in_memlet.data is None:
                continue
            callsite_stream.write(
                self.memlet_definition(sdfg,
                                       in_memlet,
                                       False,
                                       vconn,
                                       conntype=node.sdfg.arrays[vconn],
                                       allow_shadowing=True), sdfg, state_id,
                node)
        for _, uconn, _, _, out_memlet in state_dfg.out_edges(node):
            if out_memlet.data is not None:
                if out_memlet.wcr is not None:
                    out_code = emit_memlet_reference(self._dispatcher, sdfg,
                                                     out_memlet, uconn)
                else:
                    out_code = self.memlet_definition(
                        sdfg,
                        out_memlet,
                        True,
                        uconn,
                        conntype=node.sdfg.arrays[uconn],
                        allow_shadowing=True)

                callsite_stream.write(out_code, sdfg, state_id, node)

        callsite_stream.write("\n{    ///////////////////\n", sdfg, state_id,
                              node)

        # Emit symbol mappings
        # HACK: We first emit variables of the form __dacesym_X = Y to avoid
        #       overriding symbolic expressions when the symbol names match
        # TODO: When emitting nested SDFGs as separate functions,
        #       remove the workaround
        for symname, symval in sorted(node.symbol_mapping.items()):
            callsite_stream.write(
                '{dtype} __dacesym_{symname} = {symval};\n'.format(
                    dtype=symbolic.symtype(symval),
                    symname=symname,
                    symval=sym2cpp(symval)), sdfg, state_id, node)
        for sym in sorted(node.symbol_mapping.keys()):
            callsite_stream.write(
                'auto {symname} = __dacesym_{symname};\n'.format(symname=sym),
                sdfg, state_id, node)
        ## End of symbol mappings

        old_schedule = self._toplevel_schedule
        self._toplevel_schedule = node.schedule

        sdfg_label = "_%d_%d" % (state_id, dfg.node_id(node))
        # Generate code for internal SDFG
        global_code, local_code, used_targets, used_environments = self._frame.generate_code(
            node.sdfg, node.schedule, sdfg_label)
        self._dispatcher._used_environments |= used_environments

        # Write generated code in the proper places (nested SDFG writes
        # location info)
        function_stream.write(global_code)
        callsite_stream.write(local_code)

        self._toplevel_schedule = old_schedule

        callsite_stream.write("}    ///////////////////\n\n", sdfg, state_id,
                              node)

        # Process outgoing memlets with the internal SDFG
        self.process_out_memlets(sdfg,
                                 state_id,
                                 node,
                                 state_dfg,
                                 self._dispatcher,
                                 callsite_stream,
                                 True,
                                 function_stream,
                                 skip_wcr=True)

        self._dispatcher.defined_vars.exit_scope(sdfg)
        callsite_stream.write('}', sdfg, state_id, node)

    def _generate_MapEntry(
        self,
        sdfg,
        dfg,
        state_id,
        node: nodes.MapEntry,
        function_stream,
        callsite_stream,
    ):
        state_dfg = sdfg.node(state_id)
        map_params = node.map.params
        map_name = "__DACEMAP_" + str(state_id) + "_" + str(dfg.node_id(node))

        result = callsite_stream
        map_header = ""

        # Encapsulate map with a C scope
        # TODO: Refactor out of MapEntry generation (generate_scope_header?)
        callsite_stream.write('{', sdfg, state_id, node)

        # Define all input connectors of this map entry
        for e in dace.sdfg.dynamic_map_inputs(state_dfg, node):
            callsite_stream.write(
                self.memlet_definition(sdfg, e.data, False, e.dst_conn), sdfg,
                state_id, node)

        inner_stream = CodeIOStream()
        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream,
                                     callsite_stream, inner_stream)

        # Instrumentation: Pre-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state_dfg, node, callsite_stream,
                                 inner_stream, function_stream)

        # TODO: Refactor to generate_scope_preamble once a general code
        #  generator (that CPU inherits from) is implemented
        if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
            map_header += "#pragma omp parallel for"
            if node.map.collapse > 1:
                map_header += ' collapse(%d)' % node.map.collapse
            # Loop over outputs, add OpenMP reduction clauses to detected cases
            # TODO: set up register outside loop
            # exit_node = dfg.exit_node(node)
            reduction_stmts = []
            # for outedge in dfg.in_edges(exit_node):
            #    if (isinstance(outedge.src, nodes.CodeNode)
            #            and outedge.data.wcr is not None):
            #        redt = operations.detect_reduction_type(outedge.data.wcr)
            #        if redt != dtypes.ReductionType.Custom:
            #            reduction_stmts.append('reduction({typ}:{var})'.format(
            #                typ=_REDUCTION_TYPE_TO_OPENMP[redt],
            #                var=outedge.src_conn))
            #            reduced_variables.append(outedge)

            map_header += " %s\n" % ", ".join(reduction_stmts)

        # TODO: Explicit map unroller
        if node.map.unroll:
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                raise ValueError("A Multicore CPU map cannot be unrolled (" +
                                 node.map.label + ")")

        constsize = all([
            not symbolic.issymbolic(v, sdfg.constants) for r in node.map.range
            for v in r
        ])

        # Construct (EXCLUSIVE) map range as a list of comma-delimited C++
        # strings.
        maprange_cppstr = [
            "%s, %s, %s" % (sym2cpp(rb), sym2cpp(re + 1), sym2cpp(rs))
            for rb, re, rs in node.map.range
        ]

        # Nested loops
        result.write(map_header, sdfg, state_id, node)
        for i, r in enumerate(node.map.range):
            # var = '__DACEMAP_%s_%d' % (node.map.label, i)
            var = map_params[i]
            begin, end, skip = r

            if node.map.unroll:
                result.write("#pragma unroll", sdfg, state_id, node)

            result.write(
                "for (auto %s = %s; %s < %s; %s += %s) {\n" %
                (var, sym2cpp(begin), var, sym2cpp(end + 1), var,
                 sym2cpp(skip)),
                sdfg,
                state_id,
                node,
            )

        callsite_stream.write(inner_stream.getvalue())

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

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream,
                          callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        map_node = scope_dict[node]
        state_dfg = sdfg.node(state_id)

        if map_node is None:
            raise ValueError("Exit node " + str(node.map.label) +
                             " is not dominated by a scope entry node")

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

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None and not is_devicelevel(sdfg, state_dfg, node):
            instr.on_scope_exit(sdfg, state_dfg, node, outer_stream,
                                callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream,
                                      outer_stream, callsite_stream)

        for _ in map_node.map.range:
            result.write("}", sdfg, state_id, node)

        result.write(outer_stream.getvalue())

        callsite_stream.write('}', sdfg, state_id, node)

    def _generate_ConsumeEntry(
        self,
        sdfg,
        dfg,
        state_id,
        node: nodes.MapEntry,
        function_stream,
        callsite_stream,
    ):
        result = callsite_stream

        constsize = all([
            not symbolic.issymbolic(v, sdfg.constants) for r in node.map.range
            for v in r
        ])
        state_dfg = sdfg.nodes()[state_id]

        input_sedge = next(e for e in state_dfg.in_edges(node)
                           if e.dst_conn == "IN_stream")
        output_sedge = next(e for e in state_dfg.out_edges(node)
                            if e.src_conn == "OUT_stream")
        input_stream = state_dfg.memlet_path(input_sedge)[0].src
        input_streamdesc = input_stream.desc(sdfg)

        # Take chunks into account
        if node.consume.chunksize == 1:
            chunk = "const %s& %s" % (
                input_streamdesc.dtype.ctype,
                node.consume.label + "_element",
            )
            self._dispatcher.defined_vars.add(node.consume.label + "_element",
                                              DefinedType.Scalar)
        else:
            chunk = "const %s *%s, size_t %s" % (
                input_streamdesc.dtype.ctype,
                node.consume.label + "_elements",
                node.consume.label + "_numelems",
            )
            self._dispatcher.defined_vars.add(node.consume.label + "_elements",
                                              DefinedType.Pointer)
            self._dispatcher.defined_vars.add(node.consume.label + "_numelems",
                                              DefinedType.Scalar)

        # Take quiescence condition into account
        if node.consume.condition.code is not None:
            condition_string = "[&]() { return %s; }, " % cppunparse.cppunparse(
                node.consume.condition.code, False)
        else:
            condition_string = ""

        inner_stream = CodeIOStream()

        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream,
                                     callsite_stream, inner_stream)

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state_dfg, node, callsite_stream,
                                 inner_stream, function_stream)

        result.write(
            "dace::Consume<{chunksz}>::template consume{cond}({stream_in}, "
            "{num_pes}, {condition}"
            "[&](int {pe_index}, {element_or_chunk}) {{".format(
                chunksz=node.consume.chunksize,
                cond="" if node.consume.condition.code is None else "_cond",
                condition=condition_string,
                stream_in=input_stream.data,  # TODO: stream arrays
                element_or_chunk=chunk,
                num_pes=sym2cpp(node.consume.num_pes),
                pe_index=node.consume.pe_index,
            ),
            sdfg,
            state_id,
            node,
        )

        # Since consume is an alias node, we create an actual array for the
        # consumed element and modify the outgoing memlet path ("OUT_stream")
        # TODO: do this before getting to the codegen
        if node.consume.chunksize == 1:
            newname, _ = sdfg.add_scalar(node.consume.label + "_element",
                                         input_streamdesc.dtype,
                                         transient=True,
                                         storage=dtypes.StorageType.Register,
                                         find_new_name=True)
            ce_node = nodes.AccessNode(newname, dtypes.AccessType.ReadOnly)
        else:
            newname, _ = sdfg.add_array(node.consume.label + '_elements',
                                        [node.consume.chunksize],
                                        input_streamdesc.dtype,
                                        transient=True,
                                        storage=dtypes.StorageType.Register,
                                        find_new_name=True)
            ce_node = nodes.AccessNode(newname, dtypes.AccessType.ReadOnly)
        state_dfg.add_node(ce_node)
        out_memlet_path = state_dfg.memlet_path(output_sedge)
        state_dfg.remove_edge(out_memlet_path[0])
        state_dfg.add_edge(
            out_memlet_path[0].src,
            out_memlet_path[0].src_conn,
            ce_node,
            None,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)),
        )
        state_dfg.add_edge(
            ce_node,
            None,
            out_memlet_path[0].dst,
            out_memlet_path[0].dst_conn,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)),
        )
        for e in out_memlet_path[1:]:
            e.data.data = ce_node.data
        # END of SDFG-rewriting code

        result.write(inner_stream.getvalue())

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

            # Generate register definitions for inter-tasklet memlets
            scope_dict = dfg.scope_dict()
            for edge in dfg.edges():
                # Only interested in edges within current scope
                if scope_dict[edge.src] != node or scope_dict[
                        edge.dst] != node:
                    continue
                if (isinstance(edge.src, nodes.CodeNode)
                        and isinstance(edge.dst, nodes.CodeNode)):
                    local_name = edge.data.data
                    # Allocate variable type
                    code = 'dace::vec<%s, %s> %s;' % (
                        sdfg.arrays[edge.data.data].dtype.ctype,
                        sym2cpp(edge.data.veclen), local_name)
                    result.write(code, sdfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(local_name,
                                                      DefinedType.Scalar)

    def _generate_ConsumeExit(self, sdfg, dfg, state_id, node, function_stream,
                              callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]
        state_dfg = sdfg.node(state_id)

        if entry_node is None:
            raise ValueError("Exit node " + str(node.consume.label) +
                             " is not dominated by a scope entry node")

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

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_exit(sdfg, state_dfg, node, outer_stream,
                                callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream,
                                      outer_stream, callsite_stream)

        result.write("});", sdfg, state_id, node)

        result.write(outer_stream.getvalue())

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
                        src_node,
                        node,
                        edge,
                        sdfg,
                        dfg,
                        state_id,
                        function_stream,
                        callsite_stream,
                    )

        # Process outgoing memlets (array-to-array write should be emitted
        # from the first leading edge out of the array)
        self.process_out_memlets(
            sdfg,
            state_id,
            node,
            state_dfg,
            self._dispatcher,
            callsite_stream,
            False,
            function_stream,
        )

    # Methods for subclasses to override

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id,
                                function_stream, outer_stream, inner_stream):
        """
        Generates code for the beginning of an SDFG scope, outputting it to
        the given code streams.
        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param outer_stream: A `CodeIOStream` object that points
                             to the code before the scope generation (e.g.,
                             before for-loops or kernel invocations).
        :param inner_stream: A `CodeIOStream` object that points
                             to the beginning of the scope code (e.g.,
                             inside for-loops or beginning of kernel).
        """
        pass

    def generate_scope_postamble(self, sdfg, dfg_scope, state_id,
                                 function_stream, outer_stream, inner_stream):
        """
        Generates code for the end of an SDFG scope, outputting it to
        the given code streams.
        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param outer_stream: A `CodeIOStream` object that points
                             to the code after the scope (e.g., after
                             for-loop closing braces or kernel invocations).
        :param inner_stream: A `CodeIOStream` object that points
                             to the end of the inner scope code (e.g.,
                             before for-loop closing braces or end of
                             kernel).
        """
        pass

    def generate_tasklet_preamble(self, sdfg, dfg_scope, state_id, node,
                                  function_stream, before_memlets_stream,
                                  after_memlets_stream):
        """
        Generates code for the beginning of a tasklet. This method is
        intended to be overloaded by subclasses.
        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param node: The tasklet node in the state.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param before_memlets_stream: A `CodeIOStream` object that will emit
                                      code before input memlets are generated.
        :param after_memlets_stream: A `CodeIOStream` object that will emit code
                                     after input memlets are generated.
        """
        pass

    def generate_tasklet_postamble(self, sdfg, dfg_scope, state_id, node,
                                   function_stream, before_memlets_stream,
                                   after_memlets_stream):
        """
        Generates code for the end of a tasklet. This method is intended to be
        overloaded by subclasses.
        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param node: The tasklet node in the state.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param before_memlets_stream: A `CodeIOStream` object that will emit
                                      code before output memlets are generated.
        :param after_memlets_stream: A `CodeIOStream` object that will emit code
                                     after output memlets are generated.
        """
        pass
