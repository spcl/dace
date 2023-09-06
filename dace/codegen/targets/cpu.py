# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace.sdfg.state import SDFGState
import functools
import itertools
import warnings

from sympy.functions.elementary.complexes import arg

from dace import data, dtypes, registry, memlet as mmlt, sdfg as sd, subsets, symbolic, Config
from dace.codegen import cppunparse, exceptions as cgx
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import codeblock_to_cpp, sym2cpp, update_persistent_desc
from dace.codegen.targets.target import TargetCodeGenerator, make_absolute
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.frontend import operations
from dace.sdfg import nodes, utils as sdutils
from dace.sdfg import (ScopeSubgraphView, SDFG, scope_contains_scope, is_array_stream_view, NodeNotExpandedError,
                       dynamic_map_inputs, local_transients)
from dace.sdfg.scope import is_devicelevel_gpu, is_devicelevel_fpga, is_in_scope
from typing import Union
from dace.codegen.targets import fpga


@registry.autoregister_params(name='cpu')
class CPUCodeGen(TargetCodeGenerator):
    """ SDFG CPU code generator. """

    title = "CPU"
    target_name = "cpu"
    language = "cpp"

    def __init__(self, frame_codegen, sdfg):
        self._frame = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher
        self.calling_codegen = self
        dispatcher = self._dispatcher

        self._locals = cppunparse.CPPLocals()
        # Scope depth (for defining locals)
        self._ldepth = 0

        # Keep nested SDFG schedule when descending into it
        self._toplevel_schedule = None

        # FIXME: this allows other code generators to change the CPU
        # behavior to assume that arrays point to packed types, thus dividing
        # all addresess by the vector length.
        self._packed_types = False

        # Keep track of traversed nodes
        self._generated_nodes = set()

        # Keep track of generated NestedSDG, and the name of the assigned function
        self._generated_nested_sdfg = dict()

        # NOTE: Multi-nesting with StructArrays must be further investigated.
        def _visit_structure(struct: data.Structure, args: dict, prefix: str = ''):
            for k, v in struct.members.items():
                if isinstance(v, data.Structure):
                    _visit_structure(v, args, f'{prefix}.{k}')
                elif isinstance(v, data.StructArray):
                    _visit_structure(v.stype, args, f'{prefix}.{k}')
                elif isinstance(v, data.Data):
                    args[f'{prefix}.{k}'] = v

        # Keeps track of generated connectors, so we know how to access them in nested scopes
        arglist = dict(self._frame.arglist)
        for name, arg_type in self._frame.arglist.items():
            if isinstance(arg_type, data.Structure):
                desc = sdfg.arrays[name]
                _visit_structure(arg_type, arglist, name)
            elif isinstance(arg_type, data.StructArray):
                desc = sdfg.arrays[name]
                desc = desc.stype
                _visit_structure(desc, arglist, name)

        for name, arg_type in arglist.items():
            if isinstance(arg_type, data.Scalar):
                # GPU global memory is only accessed via pointers
                # TODO(later): Fix workaround somehow
                if arg_type.storage is dtypes.StorageType.GPU_Global:
                    self._dispatcher.defined_vars.add(name, DefinedType.Pointer, dtypes.pointer(arg_type.dtype).ctype)
                    continue

                self._dispatcher.defined_vars.add(name, DefinedType.Scalar, arg_type.dtype.ctype)
            elif isinstance(arg_type, data.Array):
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer, dtypes.pointer(arg_type.dtype).ctype)
            elif isinstance(arg_type, data.Stream):
                if arg_type.is_stream_array():
                    self._dispatcher.defined_vars.add(name, DefinedType.StreamArray, arg_type.as_arg(name=''))
                else:
                    self._dispatcher.defined_vars.add(name, DefinedType.Stream, arg_type.as_arg(name=''))
            elif isinstance(arg_type, data.Structure):
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer, arg_type.dtype.ctype)
            else:
                raise TypeError("Unrecognized argument type: {t} (value {v})".format(t=type(arg_type).__name__,
                                                                                     v=str(arg_type)))

        # Register dispatchers
        dispatcher.register_node_dispatcher(self)
        dispatcher.register_map_dispatcher(
            [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent, dtypes.ScheduleType.Sequential],
            self)

        cpu_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.Register]
        dispatcher.register_array_dispatcher(cpu_storage, self)

        # Register CPU copies (all internal pairs)
        for src_storage, dst_storage in itertools.product(cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)

    @staticmethod
    def cmake_options():
        options = []

        if Config.get('compiler', 'cpu', 'executable'):
            compiler = make_absolute(Config.get('compiler', 'cpu', 'executable'))
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
        cpp.presynchronize_streams(sdfg, dfg_scope, state_id, entry_node, callsite_stream)

        self.generate_node(sdfg, dfg_scope, state_id, entry_node, function_stream, callsite_stream)
        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def generate_node(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
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

    def allocate_view(self, sdfg: SDFG, dfg: SDFGState, state_id: int, node: nodes.AccessNode,
                      global_stream: CodeIOStream, declaration_stream: CodeIOStream, allocation_stream: CodeIOStream):
        """
        Allocates (creates pointer and refers to original) a view of an
        existing array, scalar, or view.
        """

        name = node.data
        nodedesc = node.desc(sdfg)
        ptrname = cpp.ptr(name, nodedesc, sdfg, self._frame)

        # Check if array is already declared
        declared = self._dispatcher.declared_arrays.has(ptrname)

        # Check directionality of view (referencing dst or src)
        edge = sdutils.get_view_edge(dfg, node)

        # When emitting ArrayInterface, we need to know if this is a read or
        # write variation
        is_write = edge.src is node

        # Allocate the viewed data before the view, if necessary
        mpath = dfg.memlet_path(edge)
        viewed_dnode = mpath[-1].dst if is_write else mpath[0].src
        self._dispatcher.dispatch_allocate(sdfg, dfg, state_id, viewed_dnode, viewed_dnode.desc(sdfg), global_stream,
                                           allocation_stream)

        # Memlet points to view, construct mirror memlet
        memlet = edge.data
        if memlet.data == node.data:
            memlet = deepcopy(memlet)
            memlet.data = viewed_dnode.data
            memlet.subset = memlet.dst_subset if is_write else memlet.src_subset
            if memlet.subset is None:
                memlet.subset = subsets.Range.from_array(viewed_dnode.desc(sdfg))

        # Emit memlet as a reference and register defined variable
        atype, aname, value = cpp.emit_memlet_reference(self._dispatcher,
                                                        sdfg,
                                                        memlet,
                                                        name,
                                                        dtypes.pointer(nodedesc.dtype),
                                                        ancestor=0,
                                                        is_write=is_write)
        if not declared:
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            self._dispatcher.declared_arrays.add(aname, DefinedType.Pointer, ctypedef)
            if isinstance(nodedesc, data.StructureView):
                for k, v in nodedesc.members.items():
                    if isinstance(v, data.Data):
                        ctypedef = dtypes.pointer(v.dtype).ctype if isinstance(v, data.Array) else v.dtype.ctype
                        defined_type = DefinedType.Scalar if isinstance(v, data.Scalar) else DefinedType.Pointer
                        self._dispatcher.declared_arrays.add(f"{name}.{k}", defined_type, ctypedef)
                        self._dispatcher.defined_vars.add(f"{name}.{k}", defined_type, ctypedef)
                # TODO: Find a better way to do this (the issue is with pointers of pointers)
                if atype.endswith('*'):
                    atype = atype[:-1]
                if value.startswith('&'):
                    value = value[1:]
            declaration_stream.write(f'{atype} {aname};', sdfg, state_id, node)
        allocation_stream.write(f'{aname} = {value};', sdfg, state_id, node)

    def allocate_reference(self, sdfg: SDFG, dfg: SDFGState, state_id: int, node: nodes.AccessNode,
                           global_stream: CodeIOStream, declaration_stream: CodeIOStream,
                           allocation_stream: CodeIOStream):
        name = node.data
        nodedesc = node.desc(sdfg)
        ptrname = cpp.ptr(name, nodedesc, sdfg, self._frame)

        # Check if reference is already declared
        declared = self._dispatcher.declared_arrays.has(ptrname)

        if not declared:
            declaration_stream.write(f'{nodedesc.dtype.ctype} *{ptrname};', sdfg, state_id, node)
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            self._dispatcher.declared_arrays.add(ptrname, DefinedType.Pointer, ctypedef)
            self._dispatcher.defined_vars.add(ptrname, DefinedType.Pointer, ctypedef)

    def declare_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream):

        fsymbols = self._frame.symbols_and_constants(sdfg)
        # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
        # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
        # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
        # `nodedesc` is a View and `dfg` is None.
        if dfg and not sdutils.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError("The declare_array method should only be used for variables "
                                      "that must have their declaration and allocation separate.")

        name = node.data
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

        if (nodedesc.storage == dtypes.StorageType.CPU_Heap or nodedesc.storage == dtypes.StorageType.Register):

            ctypedef = dtypes.pointer(nodedesc.dtype).ctype

            declaration_stream.write(f'{nodedesc.dtype.ctype} *{name} = nullptr;\n', sdfg, state_id, node)
            self._dispatcher.declared_arrays.add(name, DefinedType.Pointer, ctypedef)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Define pointer once
            # NOTE: OpenMP threadprivate storage MUST be declared globally.
            function_stream.write(
                "{ctype} *{name} = nullptr;\n"
                "#pragma omp threadprivate({name})".format(ctype=nodedesc.dtype.ctype, name=name),
                sdfg,
                state_id,
                node,
            )
            self._dispatcher.declared_arrays.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)
        else:
            raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

    def allocate_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                       allocation_stream):
        name = node.data
        alloc_name = cpp.ptr(name, nodedesc, sdfg, self._frame)
        name = alloc_name
        # NOTE: `expr` may only be a name or a sequence of names and dots. The latter indicates nested data and
        # NOTE: structures. Since structures are implemented as pointers, we replace dots with arrows.
        alloc_name = alloc_name.replace('.', '->')

        if nodedesc.transient is False:
            return

        # Check if array is already allocated
        if self._dispatcher.defined_vars.has(name):
            return

        # Check if array is already declared
        declared = self._dispatcher.declared_arrays.has(name)

        define_var = self._dispatcher.defined_vars.add
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            define_var = self._dispatcher.defined_vars.add_global
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # Compute array size
        arrsize = nodedesc.total_size
        arrsize_bytes = None
        if not isinstance(nodedesc.dtype, dtypes.opaque):
            arrsize_bytes = arrsize * nodedesc.dtype.bytes

        if isinstance(nodedesc, data.Structure) and not isinstance(nodedesc, data.StructureView):
            declaration_stream.write(f"{nodedesc.ctype} {name} = new {nodedesc.dtype.base_type};\n")
            define_var(name, DefinedType.Pointer, nodedesc.ctype)
            for k, v in nodedesc.members.items():
                if isinstance(v, data.Data):
                    ctypedef = dtypes.pointer(v.dtype).ctype if isinstance(v, data.Array) else v.dtype.ctype
                    defined_type = DefinedType.Scalar if isinstance(v, data.Scalar) else DefinedType.Pointer
                    self._dispatcher.declared_arrays.add(f"{name}.{k}", defined_type, ctypedef)
                    self.allocate_array(sdfg, dfg, state_id, nodes.AccessNode(f"{name}.{k}"), v, function_stream,
                                        declaration_stream, allocation_stream)
            return
        if isinstance(nodedesc, (data.StructureView, data.View)):
            return self.allocate_view(sdfg, dfg, state_id, node, function_stream, declaration_stream, allocation_stream)
        if isinstance(nodedesc, data.Reference):
            return self.allocate_reference(sdfg, dfg, state_id, node, function_stream, declaration_stream,
                                           allocation_stream)
        if isinstance(nodedesc, data.Scalar):
            if node.setzero:
                declaration_stream.write("%s %s = 0;\n" % (nodedesc.dtype.ctype, name), sdfg, state_id, node)
            else:
                declaration_stream.write("%s %s;\n" % (nodedesc.dtype.ctype, name), sdfg, state_id, node)
            define_var(name, DefinedType.Scalar, nodedesc.dtype.ctype)
        elif isinstance(nodedesc, data.Stream):
            ###################################################################
            # Stream directly connected to an array

            if is_array_stream_view(sdfg, dfg, node):
                if state_id is None:
                    raise SyntaxError("Stream-view of array may not be defined in more than one state")

                arrnode = sdfg.arrays[nodedesc.sink]
                state = sdfg.nodes()[state_id]
                edges = state.out_edges(node)
                if len(edges) > 1:
                    raise NotImplementedError("Cannot handle streams writing to multiple arrays.")

                memlet_path = state.memlet_path(edges[0])
                # Allocate the array before its stream view, if necessary
                self.allocate_array(sdfg, dfg, state_id, memlet_path[-1].dst, memlet_path[-1].dst.desc(sdfg),
                                    function_stream, declaration_stream, allocation_stream)

                array_expr = cpp.copy_expr(self._dispatcher,
                                           sdfg,
                                           nodedesc.sink,
                                           edges[0].data,
                                           packed_types=self._packed_types)
                threadlocal = ""
                threadlocal_stores = [dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.Register]
                if (sdfg.arrays[nodedesc.sink].storage in threadlocal_stores or nodedesc.storage in threadlocal_stores):
                    threadlocal = "Threadlocal"
                ctype = 'dace::ArrayStreamView%s<%s>' % (threadlocal, arrnode.dtype.ctype)
                declaration_stream.write(
                    "%s %s (%s);\n" % (ctype, name, array_expr),
                    sdfg,
                    state_id,
                    node,
                )
                define_var(name, DefinedType.Stream, ctype)
                return

            ###################################################################
            # Regular stream

            dtype = nodedesc.dtype.ctype
            ctypedef = 'dace::Stream<{}>'.format(dtype)
            if nodedesc.buffer_size != 0:
                definition = "{} {}({});".format(ctypedef, name, nodedesc.buffer_size)
            else:
                definition = "{} {};".format(ctypedef, name)

            declaration_stream.write(definition, sdfg, state_id, node)
            define_var(name, DefinedType.Stream, ctypedef)

        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Register and
                  ((symbolic.issymbolic(arrsize, sdfg.constants)) or
                   (arrsize_bytes and ((arrsize_bytes > Config.get("compiler", "max_stack_array_size")) == True))))):

            if nodedesc.storage == dtypes.StorageType.Register:

                if symbolic.issymbolic(arrsize, sdfg.constants):
                    warnings.warn('Variable-length array %s with size %s '
                                  'detected and was allocated on heap instead of '
                                  '%s' % (name, cpp.sym2cpp(arrsize), nodedesc.storage))
                elif (arrsize_bytes > Config.get("compiler", "max_stack_array_size")) == True:
                    warnings.warn("Array {} with size {} detected and was allocated on heap instead of "
                                  "{} since its size is greater than max_stack_array_size ({})".format(
                                      name, cpp.sym2cpp(arrsize_bytes), nodedesc.storage,
                                      Config.get("compiler", "max_stack_array_size")))

            ctypedef = dtypes.pointer(nodedesc.dtype).ctype

            if not declared:
                declaration_stream.write(f'{nodedesc.dtype.ctype} *{name};\n', sdfg, state_id, node)
            allocation_stream.write(
                "%s = new %s DACE_ALIGN(64)[%s];\n" % (alloc_name, nodedesc.dtype.ctype, cpp.sym2cpp(arrsize)), sdfg,
                state_id, node)
            define_var(name, DefinedType.Pointer, ctypedef)

            if node.setzero:
                allocation_stream.write("memset(%s, 0, sizeof(%s)*%s);" %
                                        (alloc_name, nodedesc.dtype.ctype, cpp.sym2cpp(arrsize)))
            if nodedesc.start_offset != 0:
                allocation_stream.write(f'{alloc_name} += {cpp.sym2cpp(nodedesc.start_offset)};\n', sdfg, state_id,
                                        node)

            return
        elif (nodedesc.storage == dtypes.StorageType.Register):
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            if nodedesc.start_offset != 0:
                raise NotImplementedError('Start offset unsupported for registers')
            if node.setzero:
                declaration_stream.write(
                    "%s %s[%s]  DACE_ALIGN(64) = {0};\n" % (nodedesc.dtype.ctype, name, cpp.sym2cpp(arrsize)),
                    sdfg,
                    state_id,
                    node,
                )
                define_var(name, DefinedType.Pointer, ctypedef)
                return
            declaration_stream.write(
                "%s %s[%s]  DACE_ALIGN(64);\n" % (nodedesc.dtype.ctype, name, cpp.sym2cpp(arrsize)),
                sdfg,
                state_id,
                node,
            )
            define_var(name, DefinedType.Pointer, ctypedef)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Define pointer once
            # NOTE: OpenMP threadprivate storage MUST be declared globally.
            if not declared:
                function_stream.write(
                    "{ctype} *{name};\n#pragma omp threadprivate({name})".format(ctype=nodedesc.dtype.ctype, name=name),
                    sdfg,
                    state_id,
                    node,
                )
                self._dispatcher.declared_arrays.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)

            # Allocate in each OpenMP thread
            allocation_stream.write(
                """
                #pragma omp parallel
                {{
                    {name} = new {ctype} DACE_ALIGN(64)[{arrsize}];""".format(ctype=nodedesc.dtype.ctype,
                                                                              name=alloc_name,
                                                                              arrsize=cpp.sym2cpp(arrsize)),
                sdfg,
                state_id,
                node,
            )
            if node.setzero:
                allocation_stream.write("memset(%s, 0, sizeof(%s)*%s);" %
                                        (alloc_name, nodedesc.dtype.ctype, cpp.sym2cpp(arrsize)))
            if nodedesc.start_offset != 0:
                allocation_stream.write(f'{alloc_name} += {cpp.sym2cpp(nodedesc.start_offset)};\n', sdfg, state_id,
                                        node)

            # Close OpenMP parallel section
            allocation_stream.write('}')
            self._dispatcher.defined_vars.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)
        else:
            raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

    def deallocate_array(self, sdfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        arrsize = nodedesc.total_size
        alloc_name = cpp.ptr(node.data, nodedesc, sdfg, self._frame)
        if isinstance(nodedesc, data.Array) and nodedesc.start_offset != 0:
            alloc_name = f'({alloc_name} - {cpp.sym2cpp(nodedesc.start_offset)})'

        if self._dispatcher.declared_arrays.has(alloc_name):
            is_global = nodedesc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
            self._dispatcher.declared_arrays.remove(alloc_name, is_global=is_global)

        if isinstance(nodedesc, (data.Scalar, data.StructureView, data.View, data.Stream, data.Reference)):
            return
        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Register and symbolic.issymbolic(arrsize, sdfg.constants))):
            callsite_stream.write("delete[] %s;\n" % alloc_name, sdfg, state_id, node)
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Deallocate in each OpenMP thread
            callsite_stream.write(
                """#pragma omp parallel
                {{
                    delete[] {name};
                }}""".format(name=alloc_name),
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
                src_parent = dfg.entry_node(src_node)
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
            dst_parent = dfg.entry_node(dst_node)
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = sdfg.node(state_id)

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
        orig_vconn = vconn

        # Determine memlet directionality
        if isinstance(src_node, nodes.AccessNode) and memlet.data == src_node.data:
            write = True
        elif isinstance(dst_node, nodes.AccessNode) and memlet.data == dst_node.data:
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError("Copying between code nodes is only supported as part of the participating nodes")
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, False, vconn, dst_node.in_connectors[vconn]),
                sdfg,
                state_id,
                [src_node, dst_node],
            )
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, True, uconn, src_node.out_connectors[uconn]),
                sdfg,
                state_id,
                [src_node, dst_node],
            )
            return
        else:  # Copy array-to-array
            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)

            if write:
                vconn = cpp.ptr(dst_node.data, dst_nodedesc, sdfg, self._frame)
            ctype = dst_nodedesc.dtype.ctype

            #############################################
            # Corner cases

            # Writing one index
            # if (isinstance(memlet.subset, subsets.Indices) and memlet.wcr is None
            #         and self._dispatcher.defined_vars.get(vconn)[0] == DefinedType.Scalar):
            #     stream.write(
            #         "%s = %s;" % (vconn, self.memlet_ctor(sdfg, memlet, dst_nodedesc.dtype, False)),
            #         sdfg,
            #         state_id,
            #         [src_node, dst_node],
            #     )
            #     return

            # Setting a reference
            if isinstance(dst_nodedesc, data.Reference) and orig_vconn == 'set':
                srcptr = cpp.ptr(src_node.data, src_nodedesc, sdfg, self._frame)
                defined_type, _ = self._dispatcher.defined_vars.get(srcptr)
                stream.write(
                    "%s = %s;" % (vconn, cpp.cpp_ptr_expr(sdfg, memlet, defined_type)),
                    sdfg,
                    state_id,
                    [src_node, dst_node],
                )
                return

            # Writing from/to a stream
            if isinstance(sdfg.arrays[memlet.data], data.Stream) or (isinstance(src_node, nodes.AccessNode)
                                                                     and isinstance(src_nodedesc, data.Stream)):
                # Identify whether a stream is writing to an array
                if isinstance(dst_nodedesc, (data.Scalar, data.Array)) and isinstance(src_nodedesc, data.Stream):
                    # Stream -> Array - pop bulk
                    if is_array_stream_view(sdfg, dfg, src_node):
                        return  # Do nothing (handled by ArrayStreamView)

                    array_subset = (memlet.subset if memlet.data == dst_node.data else memlet.other_subset)
                    if array_subset is None:  # Need to use entire array
                        array_subset = subsets.Range.from_array(dst_nodedesc)

                    # stream_subset = (memlet.subset
                    #                  if memlet.data == src_node.data else
                    #                  memlet.other_subset)
                    stream_subset = memlet.subset
                    if memlet.data != src_node.data and memlet.other_subset:
                        stream_subset = memlet.other_subset

                    stream_expr = cpp.cpp_offset_expr(src_nodedesc, stream_subset)
                    array_expr = cpp.cpp_offset_expr(dst_nodedesc, array_subset)
                    assert functools.reduce(lambda a, b: a * b, src_nodedesc.shape, 1) == 1
                    stream.write(
                        "{s}.pop(&{arr}[{aexpr}], {maxsize});".format(s=cpp.ptr(src_node.data, src_nodedesc, sdfg,
                                                                                self._frame),
                                                                      arr=cpp.ptr(dst_node.data, dst_nodedesc, sdfg,
                                                                                  self._frame),
                                                                      aexpr=array_expr,
                                                                      maxsize=cpp.sym2cpp(array_subset.num_elements())),
                        sdfg,
                        state_id,
                        [src_node, dst_node],
                    )
                    return
                # Array -> Stream - push bulk
                if isinstance(src_nodedesc, (data.Scalar, data.Array)) and isinstance(dst_nodedesc, data.Stream):
                    if isinstance(src_nodedesc, data.Scalar):
                        stream.write(
                            "{s}.push({arr});".format(s=cpp.ptr(dst_node.data, dst_nodedesc, sdfg, self._frame),
                                                      arr=cpp.ptr(src_node.data, src_nodedesc, sdfg, self._frame)),
                            sdfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    elif hasattr(src_nodedesc, "src"):  # ArrayStreamView
                        stream.write(
                            "{s}.push({arr});".format(s=cpp.ptr(dst_node.data, dst_nodedesc, sdfg, self._frame),
                                                      arr=cpp.ptr(src_nodedesc.src, sdfg.arrays[src_nodedesc.src], sdfg,
                                                                  self._frame)),
                            sdfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    else:
                        copysize = " * ".join([cpp.sym2cpp(s) for s in memlet.subset.size()])
                        stream.write(
                            "{s}.push({arr}, {size});".format(s=cpp.ptr(dst_node.data, dst_nodedesc, sdfg, self._frame),
                                                              arr=cpp.ptr(src_node.data, src_nodedesc, sdfg,
                                                                          self._frame),
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
                cpp.memlet_copy_to_absolute_strides(
                    self._dispatcher, sdfg, state_dfg, edge, src_node, dst_node,
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
                    dims=", ".join(cpp.sym2cpp(copy_shape)),
                )
                dynshape = 0

            # Constant src/dst dimensions
            if not any(symbolic.issymbolic(s, sdfg.constants) for s in dst_strides):
                # Constant destination
                shape_tmpl = "template ConstDst<%s>" % ", ".join(cpp.sym2cpp(dst_strides))
                dyndst = 0
            elif not any(symbolic.issymbolic(s, sdfg.constants) for s in src_strides):
                # Constant source
                shape_tmpl = "template ConstSrc<%s>" % ", ".join(cpp.sym2cpp(src_strides))
                dynsrc = 0
            else:
                # Both dynamic
                shape_tmpl = "Dynamic"

            # Parameter pack handling
            stride_tmpl_args = [0] * (dynshape + dynsrc + dyndst) * len(copy_shape)
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
                         ([] if memlet.wcr is None else [cpp.unparse_cr(sdfg, memlet.wcr, dst_nodedesc.dtype)]) +
                         cpp.sym2cpp(stride_tmpl_args))

            # Instrumentation: Pre-copy
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_copy_begin(sdfg, state_dfg, src_node, dst_node, edge, stream, None, copy_shape,
                                        src_strides, dst_strides)

            nc = True
            if memlet.wcr is not None:
                nc = not cpp.is_write_conflicted(dfg, edge, sdfg_schedule=self._toplevel_schedule)
            if nc:
                stream.write(
                    """
                    dace::CopyND{copy_tmpl}::{shape_tmpl}::{copy_func}(
                        {copy_args});""".format(
                        copy_tmpl=copy_tmpl,
                        shape_tmpl=shape_tmpl,
                        copy_func="Copy" if memlet.wcr is None else "Accumulate",
                        copy_args=", ".join(copy_args),
                    ),
                    sdfg,
                    state_id,
                    [src_node, dst_node],
                )
            else:  # Conflicted WCR
                if dynshape == 1:
                    warnings.warn('Performance warning: Emitting dynamically-'
                                  'shaped atomic write-conflict resolution of an array.')
                    stream.write(
                        """
                        dace::CopyND{copy_tmpl}::{shape_tmpl}::Accumulate_atomic(
                        {copy_args});""".format(
                            copy_tmpl=copy_tmpl,
                            shape_tmpl=shape_tmpl,
                            copy_args=", ".join(copy_args),
                        ),
                        sdfg,
                        state_id,
                        [src_node, dst_node],
                    )
                elif copy_shape == [1]:  # Special case: accumulating one element
                    dst_expr = self.memlet_view_ctor(sdfg, memlet, dst_nodedesc.dtype, True)
                    stream.write(
                        self.write_and_resolve_expr(
                            sdfg, memlet, nc, dst_expr, '*(' + src_expr + ')', dtype=dst_nodedesc.dtype) + ';', sdfg,
                        state_id, [src_node, dst_node])
                else:
                    warnings.warn('Minor performance warning: Emitting statically-'
                                  'shaped atomic write-conflict resolution of an array.')
                    stream.write(
                        """
                        dace::CopyND{copy_tmpl}::{shape_tmpl}::Accumulate_atomic(
                        {copy_args});""".format(
                            copy_tmpl=copy_tmpl,
                            shape_tmpl=shape_tmpl,
                            copy_args=", ".join(copy_args),
                        ),
                        sdfg,
                        state_id,
                        [src_node, dst_node],
                    )

        #############################################################
        # Instrumentation: Post-copy
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_copy_end(sdfg, state_dfg, src_node, dst_node, edge, stream, None)
        #############################################################

    ###########################################################################
    # Memlet handling

    def write_and_resolve_expr(self, sdfg, memlet, nc, outname, inname, indices=None, dtype=None):
        """
        Emits a conflict resolution call from a memlet.
        """

        redtype = operations.detect_reduction_type(memlet.wcr)
        atomic = "_atomic" if not nc else ""
        ptrname = cpp.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg, self._frame)
        defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
        if isinstance(indices, str):
            ptr = '%s + %s' % (cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self._frame), indices)
        else:
            ptr = cpp.cpp_ptr_expr(sdfg, memlet, defined_type, indices=indices, codegen=self._frame)

        if isinstance(dtype, dtypes.pointer):
            dtype = dtype.base_type

        # If there is a type mismatch and more than one element is used, cast
        # pointer (vector->vector WCR). Otherwise, generate vector->scalar
        # (horizontal) reduction.
        vec_prefix = ''
        vec_suffix = ''
        dst_dtype = sdfg.arrays[memlet.data].dtype
        if (isinstance(dtype, dtypes.vector) and not isinstance(dst_dtype, dtypes.vector)):
            if memlet.subset.num_elements() != 1:
                ptr = f'({dtype.ctype} *)({ptr})'
            else:
                vec_prefix = 'v'
                vec_suffix = f'<{dtype.veclen}>'
                dtype = dtype.base_type

        func = f'{vec_prefix}reduce{atomic}{vec_suffix}'

        # Special call for detected reduction types
        if redtype != dtypes.ReductionType.Custom:
            credtype = "dace::ReductionType::" + str(redtype)[str(redtype).find(".") + 1:]
            return (f'dace::wcr_fixed<{credtype}, {dtype.ctype}>::{func}({ptr}, {inname})')

        # General reduction
        custom_reduction = cpp.unparse_cr(sdfg, memlet.wcr, dtype)
        return (f'dace::wcr_custom<{dtype.ctype}>:: template {func}({custom_reduction}, {ptr}, {inname})')

    def process_out_memlets(self,
                            sdfg,
                            state_id,
                            node,
                            dfg,
                            dispatcher,
                            result,
                            locals_defined,
                            function_stream,
                            skip_wcr=False,
                            codegen=None):
        codegen = codegen or self
        scope_dict = sdfg.nodes()[state_id].scope_dict()

        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            if skip_wcr and memlet.wcr is not None:
                continue
            dst_node = dfg.memlet_path(edge)[-1].dst

            # Target is neither a data nor a tasklet node
            if isinstance(node, nodes.AccessNode) and (not isinstance(dst_node, nodes.AccessNode)
                                                       and not isinstance(dst_node, nodes.CodeNode)):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(dst_node, nodes.CodeNode) and edge.src_conn:
                shared_data_name = edge.data.data
                if not shared_data_name:
                    # Very unique name. TODO: Make more intuitive
                    shared_data_name = '__dace_%d_%d_%d_%d_%s' % (sdfg.sdfg_id, state_id, dfg.node_id(node),
                                                                  dfg.node_id(dst_node), edge.src_conn)

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
            if scope_dict[node] != scope_dict[dst_node] and scope_contains_scope(scope_dict, node, dst_node):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array
            if isinstance(node, nodes.CodeNode):
                if not uconn:
                    raise SyntaxError("Cannot copy memlet without a local connector: {} to {}".format(
                        str(edge.src), str(edge.dst)))

                conntype = node.out_connectors[uconn]
                is_scalar = not isinstance(conntype, dtypes.pointer)
                is_stream = isinstance(sdfg.arrays[memlet.data], data.Stream)

                if is_scalar and not memlet.dynamic and not is_stream:
                    out_local_name = "    __" + uconn
                    in_local_name = uconn
                    if not locals_defined:
                        out_local_name = self.memlet_ctor(sdfg, memlet, node.out_connectors[uconn], True)
                        in_memlets = [d for _, _, _, _, d in dfg.in_edges(node)]
                        assert len(in_memlets) == 1
                        in_local_name = self.memlet_ctor(sdfg, in_memlets[0], node.out_connectors[uconn], False)

                    state_dfg = sdfg.nodes()[state_id]

                    if memlet.wcr is not None:
                        nc = not cpp.is_write_conflicted(dfg, edge, sdfg_schedule=self._toplevel_schedule)
                        write_expr = codegen.write_and_resolve_expr(
                            sdfg, memlet, nc, out_local_name, in_local_name, dtype=node.out_connectors[uconn]) + ";"
                    else:
                        if isinstance(node, nodes.NestedSDFG):
                            # This case happens with nested SDFG outputs,
                            # which we skip since the memlets are references
                            continue
                        desc = sdfg.arrays[memlet.data]
                        ptrname = cpp.ptr(memlet.data, desc, sdfg, self._frame)
                        is_global = desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                      dtypes.AllocationLifetime.Persistent,
                                                      dtypes.AllocationLifetime.External)
                        try:
                            defined_type, _ = self._dispatcher.declared_arrays.get(ptrname, is_global=is_global)
                        except KeyError:
                            defined_type, _ = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)

                        if defined_type == DefinedType.Scalar:
                            mname = cpp.ptr(memlet.data, desc, sdfg, self._frame)
                            write_expr = f"{mname} = {in_local_name};"
                        elif (defined_type == DefinedType.ArrayInterface and not isinstance(desc, data.View)):
                            # Special case: No need to write anything between
                            # array interfaces going out
                            try:
                                deftype, _ = self._dispatcher.defined_vars.get(in_local_name)
                            except KeyError:
                                deftype = None
                            if deftype == DefinedType.ArrayInterface:
                                continue
                            array_expr = cpp.cpp_array_expr(sdfg, memlet, with_brackets=False, codegen=self._frame)
                            decouple_array_interfaces = Config.get_bool("compiler", "xilinx",
                                                                        "decouple_array_interfaces")
                            ptr_str = fpga.fpga_ptr(  # we are on fpga, since this is array interface
                                memlet.data,
                                desc,
                                sdfg,
                                memlet.subset,
                                True,
                                None,
                                None,
                                True,
                                decouple_array_interfaces=decouple_array_interfaces)
                            write_expr = f"*({ptr_str} + {array_expr}) = {in_local_name};"
                        else:
                            desc_dtype = desc.dtype
                            expr = cpp.cpp_array_expr(sdfg, memlet, codegen=self._frame)
                            write_expr = codegen.make_ptr_assignment(in_local_name, conntype, expr, desc_dtype)

                    # Write out
                    result.write(write_expr, sdfg, state_id, node)

            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                if dst_node != node and not isinstance(dst_node, nodes.Tasklet):
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

    def make_ptr_assignment(self, src_expr, src_dtype, dst_expr, dst_dtype, codegen=None):
        """
        Write source to destination, where the source is a scalar, and the
        destination is a pointer.
        
        :return: String of C++ performing the write.
        """
        codegen = codegen or self
        # If there is a type mismatch, cast pointer
        dst_expr = codegen.make_ptr_vector_cast(dst_expr, dst_dtype, src_dtype, True, DefinedType.Pointer)
        return f"{dst_expr} = {src_expr};"

    def memlet_view_ctor(self, sdfg, memlet, dtype, is_output):
        memlet_params = []

        memlet_name = cpp.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg, self._frame)
        def_type, _ = self._dispatcher.defined_vars.get(memlet_name)

        if def_type == DefinedType.Pointer:
            memlet_expr = memlet_name  # Common case
        elif def_type == DefinedType.Scalar:
            memlet_expr = "&" + memlet_name
        else:
            raise TypeError("Unsupported connector type {}".format(def_type))

        pointer = ''

        if isinstance(memlet.subset, subsets.Indices):

            # FIXME: _packed_types influences how this offset is
            # generated from the FPGA codegen. We should find a nicer solution.
            if self._packed_types is True:
                offset = cpp.cpp_array_expr(sdfg, memlet, False, codegen=self._frame)
            else:
                offset = cpp.cpp_array_expr(sdfg, memlet, False, codegen=self._frame)

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
                    offset = cpp.cpp_offset_expr(sdfg.arrays[memlet.data], memlet.subset)
                else:
                    offset = cpp.cpp_offset_expr(sdfg.arrays[memlet.data], memlet.subset)
                if offset == "0":
                    memlet_params.append(memlet_expr)
                else:
                    if def_type != DefinedType.Pointer:
                        raise cgx.CodegenError("Cannot offset address of connector {} of type {}".format(
                            memlet_name, def_type))
                    memlet_params.append(memlet_expr + " + " + offset)

                # Dimensions to remove from view (due to having one value)
                indexdims = []
                strides = sdfg.arrays[memlet.data].strides

                # Figure out dimensions for scalar version
                dimlen = dtype.veclen if isinstance(dtype, dtypes.vector) else 1
                for dim, (rb, re, rs) in enumerate(memlet.subset.ranges):
                    try:
                        # Check for number of elements in contiguous dimension
                        # (with respect to vector length)
                        if strides[dim] == 1 and (re - rb) == dimlen - 1:
                            indexdims.append(dim)
                        elif (re - rb) == 0:  # Elements in other dimensions
                            indexdims.append(dim)
                    except TypeError:
                        # Cannot determine truth value of Relational
                        pass

                # Remove index (one scalar) dimensions
                dims -= len(indexdims)

                if dims > 0:
                    strides = memlet.subset.absolute_strides(strides)
                    # Filter out index dims
                    strides = [s for i, s in enumerate(strides) if i not in indexdims]
                    # Use vector length to adapt strides
                    for i in range(len(strides) - 1):
                        strides[i] /= dimlen
                    memlet_params.extend(sym2cpp(strides))
                    dims = memlet.subset.data_dims()

            else:
                raise RuntimeError('Memlet type "%s" not implemented' % memlet.subset)

        # If there is a type mismatch, cast pointer (used in vector
        # packing/unpacking)
        if dtype != sdfg.arrays[memlet.data].dtype:
            memlet_params[0] = '(%s *)(%s)' % (dtype.ctype, memlet_params[0])

        return "dace::ArrayView%s<%s, %d, 1, 1> (%s)" % (
            "Out" if is_output else "In",
            dtype.ctype,
            dims,
            ", ".join(memlet_params),
        )

    def memlet_definition(self,
                          sdfg: SDFG,
                          memlet: mmlt.Memlet,
                          output: bool,
                          local_name: str,
                          conntype: Union[data.Data, dtypes.typeclass] = None,
                          allow_shadowing=False,
                          codegen=None):
        # TODO: Robust rule set
        if conntype is None:
            raise ValueError('Cannot define memlet for "%s" without connector type' % local_name)
        codegen = codegen or self
        # Convert from Data to typeclass
        if isinstance(conntype, data.Data):
            if isinstance(conntype, data.Array):
                conntype = dtypes.pointer(conntype.dtype)
            else:
                conntype = conntype.dtype

        desc = sdfg.arrays[memlet.data]

        is_scalar = not isinstance(conntype, dtypes.pointer) or desc.dtype == conntype
        is_pointer = isinstance(conntype, dtypes.pointer)

        # Allocate variable type
        memlet_type = conntype.dtype.ctype

        ptr = cpp.ptr(memlet.data, desc, sdfg, self._frame)
        types = None
        # Non-free symbol dependent Arrays due to their shape
        dependent_shape = (isinstance(desc, data.Array) and not isinstance(desc, data.View) and any(
            str(s) not in self._frame.symbols_and_constants(sdfg) for s in self._frame.free_symbols(desc)))
        try:
            # NOTE: It is hard to get access to the view-edge here, so always
            # check the declared-arrays dictionary for Views.
            if dependent_shape or isinstance(desc, data.View):
                types = self._dispatcher.declared_arrays.get(ptr)
        except KeyError:
            pass
        if not types:
            types = self._dispatcher.defined_vars.get(ptr, is_global=True)
        var_type, ctypedef = types
        # NOTE: `expr` may only be a name or a sequence of names and dots. The latter indicates nested data and
        # NOTE: structures. Since structures are implemented as pointers, we replace dots with arrows.
        ptr = ptr.replace('.', '->')

        if fpga.is_fpga_array(desc):
            decouple_array_interfaces = Config.get_bool("compiler", "xilinx", "decouple_array_interfaces")
            ptr = fpga.fpga_ptr(memlet.data,
                                desc,
                                sdfg,
                                memlet.subset,
                                output,
                                self._dispatcher,
                                0,
                                var_type == DefinedType.ArrayInterface and not isinstance(desc, data.View),
                                decouple_array_interfaces=decouple_array_interfaces)

        result = ''
        expr = (cpp.cpp_array_expr(sdfg, memlet, with_brackets=False, codegen=self._frame)
                if var_type in [DefinedType.Pointer, DefinedType.StreamArray, DefinedType.ArrayInterface] else ptr)

        if expr != ptr:
            expr = '%s[%s]' % (ptr, expr)
        # If there is a type mismatch, cast pointer
        expr = codegen.make_ptr_vector_cast(expr, desc.dtype, conntype, is_scalar, var_type)

        defined = None

        if var_type in [DefinedType.Scalar, DefinedType.Pointer, DefinedType.ArrayInterface]:
            if output:
                if is_pointer and var_type == DefinedType.ArrayInterface:
                    result += "{} {} = {};".format(memlet_type, local_name, expr)
                elif not memlet.dynamic or (memlet.dynamic and memlet.wcr is not None):
                    # Dynamic WCR memlets start uninitialized
                    result += "{} {};".format(memlet_type, local_name)
                    defined = DefinedType.Scalar

            else:
                if not memlet.dynamic:
                    if is_scalar:
                        # We can pre-read the value
                        result += "{} {} = {};".format(memlet_type, local_name, expr)
                    else:
                        # constexpr arrays
                        if memlet.data in self._frame.symbols_and_constants(sdfg):
                            result += "const {} {} = {};".format(memlet_type, local_name, expr)
                        else:
                            # Pointer reference
                            result += "{} {} = {};".format(ctypedef, local_name, expr)
                else:
                    # Variable number of reads: get a const reference that can
                    # be read if necessary
                    memlet_type = 'const %s' % memlet_type
                    if is_pointer:
                        # This is done to make the reference constant, otherwise
                        # compilers error out with initial reference value.
                        memlet_type += ' const'
                    result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = (DefinedType.Scalar if is_scalar else DefinedType.Pointer)
        elif var_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if not memlet.dynamic and memlet.num_accesses == 1:
                if not output:
                    if isinstance(desc, data.Stream) and desc.is_stream_array():
                        index = cpp.cpp_offset_expr(desc, memlet.subset)
                        expr = f"{memlet.data}[{index}]"
                    result += f'{memlet_type} {local_name} = ({expr}).pop();'
                    defined = DefinedType.Scalar
            else:
                # Just forward actions to the underlying object
                memlet_type = ctypedef
                result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = DefinedType.Stream
        else:
            raise TypeError("Unknown variable type: {}".format(var_type))

        if defined is not None:
            self._dispatcher.defined_vars.add(local_name, defined, memlet_type, allow_shadowing=allow_shadowing)

        return result

    def memlet_stream_ctor(self, sdfg, memlet):
        stream = sdfg.arrays[memlet.data]
        ptrname = cpp.ptr(memlet.data, stream, sdfg, self._frame)

        def_type, _ = self._dispatcher.defined_vars.get(ptrname)

        return memlet.data + ("[{}]".format(cpp.cpp_offset_expr(stream, memlet.subset))
                              if isinstance(stream, data.Stream) and stream.is_stream_array() else "")

    def memlet_ctor(self, sdfg, memlet, dtype, is_output):
        ptrname = cpp.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg, self._frame)
        def_type, _ = self._dispatcher.defined_vars.get(ptrname)

        if def_type in [DefinedType.Stream, DefinedType.Object, DefinedType.StreamArray]:
            return self.memlet_stream_ctor(sdfg, memlet)

        elif def_type in [DefinedType.Pointer, DefinedType.Scalar]:
            return self.memlet_view_ctor(sdfg, memlet, dtype, is_output)

        else:
            raise NotImplementedError("Connector type {} not yet implemented".format(def_type))

    #########################################################################
    # Dynamically-called node dispatchers

    def _generate_Tasklet(self, sdfg, dfg, state_id, node, function_stream, callsite_stream, codegen=None):

        # Allow other code generators to call this with a callback
        codegen = codegen or self

        outer_stream_begin = CodeIOStream()
        outer_stream_end = CodeIOStream()
        inner_stream = CodeIOStream()

        # Add code to init and exit functions
        self._frame._initcode.write(codeblock_to_cpp(node.code_init), sdfg)
        self._frame._exitcode.write(codeblock_to_cpp(node.code_exit), sdfg)

        state_dfg: SDFGState = sdfg.nodes()[state_id]

        # Free tasklets need to be presynchronized (e.g., CPU tasklet after
        # GPU->CPU copy)
        if state_dfg.entry_node(node) is None:
            cpp.presynchronize_streams(sdfg, state_dfg, state_id, node, callsite_stream)

        # Prepare preamble and code for after memlets
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_preamble(sdfg, dfg, state_id, node, function_stream, callsite_stream,
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
                ctype = node.in_connectors[edge.dst_conn].ctype
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data
                    if not shared_data_name:
                        # Very unique name. TODO: Make more intuitive
                        shared_data_name = '__dace_%d_%d_%d_%d_%s' % (sdfg.sdfg_id, state_id, dfg.node_id(src_node),
                                                                      dfg.node_id(node), edge.src_conn)

                    # Read variable from shared storage
                    defined_type, _ = self._dispatcher.defined_vars.get(shared_data_name)
                    if defined_type in (DefinedType.Scalar, DefinedType.Pointer):
                        assign_str = (f"const {ctype} {edge.dst_conn} = {shared_data_name};")
                    else:
                        assign_str = (f"const {ctype} &{edge.dst_conn} = {shared_data_name};")
                    inner_stream.write(assign_str, sdfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn, defined_type, f"const {ctype}")

                else:
                    self._dispatcher.dispatch_copy(
                        src_node,
                        node,
                        edge,
                        sdfg,
                        dfg,
                        state_id,
                        function_stream,
                        inner_stream,
                    )

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1, ctype)
                arrays.add(edge.dst_conn)

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

                self._dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, dfg, state_id, function_stream,
                                                            inner_stream)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, node.out_connectors[edge.src_conn].ctype)
                tasklet_out_connectors.add(edge.src_conn)

        for edge in state_dfg.out_edges(node):
            # Special case: code->code
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if edge.src_conn is None:
                continue
            cdtype = node.out_connectors[edge.src_conn]
            ctype = cdtype.ctype
            # Convert dtype to data descriptor
            if isinstance(cdtype, dtypes.pointer):
                arg_type = data.Array(cdtype._typeclass, [1])
            else:
                arg_type = data.Scalar(cdtype)

            if (isinstance(dst_node, nodes.CodeNode) and edge.src_conn not in tasklet_out_connectors):
                memlet = edge.data

                # Generate register definitions for inter-tasklet memlets
                local_name = edge.data.data
                if not local_name:
                    # Very unique name. TODO: Make more intuitive
                    local_name = '__dace_%d_%d_%d_%d_%s' % (sdfg.sdfg_id, state_id, dfg.node_id(node),
                                                            dfg.node_id(dst_node), edge.src_conn)

                # Allocate variable type
                code = "%s %s;" % (ctype, local_name)
                outer_stream_begin.write(code, sdfg, state_id, [edge.src, dst_node])
                if (isinstance(arg_type, data.Scalar) or isinstance(arg_type, dtypes.typeclass)):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype, ancestor=1)
                elif isinstance(arg_type, data.Array):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Pointer, ctype, ancestor=1)
                elif isinstance(arg_type, data.Stream):
                    if arg_type.is_stream_array():
                        self._dispatcher.defined_vars.add(local_name, DefinedType.StreamArray, ctype, ancestor=1)
                    else:
                        self._dispatcher.defined_vars.add(local_name, DefinedType.Stream, ctype, ancestor=1)
                else:
                    raise TypeError("Unrecognized argument type: {}".format(type(arg_type).__name__))

                inner_stream.write("%s %s;" % (ctype, edge.src_conn), sdfg, state_id, [edge.src, edge.dst])
                tasklet_out_connectors.add(edge.src_conn)
                self._dispatcher.defined_vars.add(edge.src_conn, DefinedType.Scalar, ctype)
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, ctype)
                locals_defined = True

        # Emit post-memlet tasklet preamble code
        callsite_stream.write(after_memlets_stream.getvalue())

        # Instrumentation: Pre-tasklet
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, state_dfg, node, outer_stream_begin, inner_stream, function_stream)

        inner_stream.write("\n    ///////////////////\n", sdfg, state_id, node)

        codegen.unparse_tasklet(sdfg, state_id, dfg, node, function_stream, inner_stream, self._locals, self._ldepth,
                                self._toplevel_schedule)

        inner_stream.write("    ///////////////////\n\n", sdfg, state_id, node)

        # Generate pre-memlet tasklet postamble
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_postamble(sdfg, dfg, state_id, node, function_stream, inner_stream,
                                           after_memlets_stream)

        # Process outgoing memlets
        codegen.process_out_memlets(
            sdfg,
            state_id,
            node,
            dfg,
            self._dispatcher,
            inner_stream,
            True,
            function_stream,
        )

        # Instrumentation: Post-tasklet
        if instr is not None:
            instr.on_node_end(sdfg, state_dfg, node, outer_stream_end, inner_stream, function_stream)

        callsite_stream.write(outer_stream_begin.getvalue(), sdfg, state_id, node)
        callsite_stream.write('{', sdfg, state_id, node)
        callsite_stream.write(inner_stream.getvalue(), sdfg, state_id, node)
        callsite_stream.write(after_memlets_stream.getvalue())
        callsite_stream.write('}', sdfg, state_id, node)
        callsite_stream.write(outer_stream_end.getvalue(), sdfg, state_id, node)

        self._dispatcher.defined_vars.exit_scope(node)

    def unparse_tasklet(self, sdfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                        toplevel_schedule):
        # Call the generic CPP unparse_tasklet method
        cpp.unparse_tasklet(sdfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth, toplevel_schedule,
                            self)

    def define_out_memlet(self, sdfg, state_dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream):
        cdtype = src_node.out_connectors[edge.src_conn]
        if isinstance(sdfg.arrays[edge.data.data], data.Stream):
            pass
        elif isinstance(cdtype, dtypes.pointer):
            # If pointer, also point to output
            desc = sdfg.arrays[edge.data.data]
            ptrname = cpp.ptr(edge.data.data, desc, sdfg, self._frame)
            is_global = desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                          dtypes.AllocationLifetime.External)
            defined_type, _ = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)
            base_ptr = cpp.cpp_ptr_expr(sdfg, edge.data, defined_type, codegen=self._frame)
            callsite_stream.write(f'{cdtype.ctype} {edge.src_conn} = {base_ptr};', sdfg, state_id, src_node)
        else:
            callsite_stream.write(f'{cdtype.ctype} {edge.src_conn};', sdfg, state_id, src_node)

    def generate_nsdfg_header(self, sdfg, state, state_id, node, memlet_references, sdfg_label, state_struct=True):
        # TODO: Use a single method for GPU kernels, FPGA modules, and NSDFGs
        arguments = []

        if state_struct:
            toplevel_sdfg: SDFG = sdfg.sdfg_list[0]
            arguments.append(f'{toplevel_sdfg.name}_t *__state')

        # Add "__restrict__" keywords to arguments that do not alias with others in the context of this SDFG
        restrict_args = []
        for atype, aname, _ in memlet_references:

            def make_restrict(expr: str) -> str:
                # Check whether "restrict" has already been added before and can be added
                if expr.strip().endswith('*'):
                    return '__restrict__'
                else:
                    return ''

            if aname in node.sdfg.arrays and not node.sdfg.arrays[aname].may_alias:
                restrict_args.append(make_restrict(atype))
            else:
                restrict_args.append('')

        arguments += [
            f'{atype} {restrict} {aname}' for (atype, aname, _), restrict in zip(memlet_references, restrict_args)
        ]
        arguments += [
            f'{node.sdfg.symbols[aname].as_arg(aname)}' for aname in sorted(node.symbol_mapping.keys())
            if aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        return f'void {sdfg_label}({arguments}) {{'

    def generate_nsdfg_call(self, sdfg, state, node, memlet_references, sdfg_label, state_struct=True):
        prepend = []
        if state_struct:
            prepend = ['__state']
        args = ', '.join(prepend + [argval for _, _, argval in memlet_references] + [
            cpp.sym2cpp(symval)
            for symname, symval in sorted(node.symbol_mapping.items()) if symname not in sdfg.constants
        ])
        return f'{sdfg_label}({args});'

    def generate_nsdfg_arguments(self, sdfg, dfg, state, node):
        # Connectors that are both input and output share the same name
        inout = set(node.in_connectors.keys() & node.out_connectors.keys())

        for _, _, _, vconn, memlet in state.all_edges(node):
            if (memlet.data in sdfg.arrays and fpga.is_multibank_array(sdfg.arrays[memlet.data])
                    and fpga.parse_location_bank(sdfg.arrays[memlet.data])[0] == "HBM"):

                raise NotImplementedError("HBM in nested SDFGs not supported in non-FPGA code.")

        memlet_references = []
        for _, _, _, vconn, in_memlet in sorted(state.in_edges(node), key=lambda e: e.dst_conn or ''):
            if vconn in inout or in_memlet.data is None:
                continue
            memlet_references.append(
                cpp.emit_memlet_reference(self._dispatcher,
                                          sdfg,
                                          in_memlet,
                                          vconn,
                                          is_write=vconn in node.out_connectors,
                                          conntype=node.in_connectors[vconn]))

        for _, uconn, _, _, out_memlet in sorted(state.out_edges(node), key=lambda e: e.src_conn or ''):
            if out_memlet.data is not None:
                memlet_references.append(
                    cpp.emit_memlet_reference(self._dispatcher,
                                              sdfg,
                                              out_memlet,
                                              uconn,
                                              conntype=node.out_connectors[uconn]))
        return memlet_references

    def _generate_NestedSDFG(
        self,
        sdfg,
        dfg: ScopeSubgraphView,
        state_id,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        inline = Config.get_bool('compiler', 'inline_sdfgs')
        self._dispatcher.defined_vars.enter_scope(sdfg, can_access_parent=inline)
        state_dfg = sdfg.nodes()[state_id]

        # Quick sanity check.
        # TODO(later): Is this necessary or "can_access_parent" should always be False?
        if inline:
            for nestedarr, ndesc in node.sdfg.arrays.items():
                if (self._dispatcher.defined_vars.has(nestedarr) and ndesc.transient):
                    raise NameError(f'Data name "{nestedarr}" in SDFG "{node.sdfg.name}" '
                                    'already defined in higher scopes and will be shadowed. '
                                    'Please rename or disable inline_sdfgs in the DaCe '
                                    'configuration to compile.')

        # Emit nested SDFG as a separate function
        nested_stream = CodeIOStream()
        nested_global_stream = CodeIOStream()

        unique_functions_conf = Config.get('compiler', 'unique_functions')

        # Backwards compatibility
        if unique_functions_conf is True:
            unique_functions_conf = 'hash'
        elif unique_functions_conf is False:
            unique_functions_conf = 'none'

        if unique_functions_conf == 'hash':
            unique_functions = True
            unique_functions_hash = True
        elif unique_functions_conf == 'unique_name':
            unique_functions = True
            unique_functions_hash = False
        elif unique_functions_conf == 'none':
            unique_functions = False
        else:
            raise ValueError(f'Unknown unique_functions configuration: {unique_functions_conf}')

        if unique_functions and not unique_functions_hash and node.unique_name != "":
            # If the SDFG has a unique name, use it
            sdfg_label = node.unique_name
        else:
            sdfg_label = "%s_%d_%d_%d" % (node.sdfg.name, sdfg.sdfg_id, state_id, dfg.node_id(node))

        code_already_generated = False
        if unique_functions and not inline:
            hash = node.sdfg.hash_sdfg()
            if unique_functions_hash:
                # Use hashing to check whether this Nested SDFG has been already generated. If that is the case,
                # use the saved name to call it, otherwise save the hash and the associated name
                if hash in self._generated_nested_sdfg:
                    code_already_generated = True
                    sdfg_label = self._generated_nested_sdfg[hash]
                else:
                    self._generated_nested_sdfg[hash] = sdfg_label
            else:
                # Use the SDFG label to check if this has been already code generated.
                # Check the hash of the formerly generated SDFG to check that we are not
                # generating different SDFGs with the same name
                if sdfg_label in self._generated_nested_sdfg:
                    code_already_generated = True
                    if hash != self._generated_nested_sdfg[sdfg_label]:
                        raise ValueError(f'Different Nested SDFGs have the same unique name: {sdfg_label}')
                else:
                    self._generated_nested_sdfg[sdfg_label] = hash

        #########################################
        # Take care of nested SDFG I/O (arguments)
        # Arguments are input connectors, output connectors, and symbols
        codegen = self.calling_codegen
        memlet_references = codegen.generate_nsdfg_arguments(sdfg, dfg, state_dfg, node)

        if not inline and (not unique_functions or not code_already_generated):
            nested_stream.write(
                ('inline ' if codegen is self else '') +
                codegen.generate_nsdfg_header(sdfg, state_dfg, state_id, node, memlet_references, sdfg_label), sdfg,
                state_id, node)

        #############################
        # Generate function contents

        if inline:
            callsite_stream.write('{', sdfg, state_id, node)
            for ref in memlet_references:
                callsite_stream.write('%s %s = %s;' % ref, sdfg, state_id, node)
            # Emit symbol mappings
            # We first emit variables of the form __dacesym_X = Y to avoid
            # overriding symbolic expressions when the symbol names match
            for symname, symval in sorted(node.symbol_mapping.items()):
                if symname in sdfg.constants:
                    continue
                callsite_stream.write(
                    '{dtype} __dacesym_{symname} = {symval};\n'.format(dtype=node.sdfg.symbols[symname],
                                                                       symname=symname,
                                                                       symval=cpp.sym2cpp(symval)), sdfg, state_id,
                    node)
            for symname in sorted(node.symbol_mapping.keys()):
                if symname in sdfg.constants:
                    continue
                callsite_stream.write(
                    '{dtype} {symname} = __dacesym_{symname};\n'.format(symname=symname,
                                                                        dtype=node.sdfg.symbols[symname]), sdfg,
                    state_id, node)
            ## End of symbol mappings
            #############################
            nested_stream = callsite_stream
            nested_global_stream = function_stream

        if not unique_functions or not code_already_generated:
            if not inline:
                self._frame.generate_constants(node.sdfg, nested_stream)

            old_schedule = self._toplevel_schedule
            self._toplevel_schedule = node.schedule

            # Generate code for internal SDFG
            global_code, local_code, used_targets, used_environments = self._frame.generate_code(
                node.sdfg, node.schedule, sdfg_label)
            self._dispatcher._used_environments |= used_environments

            self._toplevel_schedule = old_schedule

            nested_stream.write(local_code)

            # Process outgoing memlets with the internal SDFG
            codegen.process_out_memlets(sdfg,
                                        state_id,
                                        node,
                                        state_dfg,
                                        self._dispatcher,
                                        nested_stream,
                                        True,
                                        nested_global_stream,
                                        skip_wcr=True)

            nested_stream.write('}\n\n', sdfg, state_id, node)

        ########################
        if not inline:
            # Generate function call
            callsite_stream.write(codegen.generate_nsdfg_call(sdfg, state_dfg, node, memlet_references, sdfg_label),
                                  sdfg, state_id, node)

            ###############################################################
            # Write generated code in the proper places (nested SDFG writes
            # location info)
            if not unique_functions or not code_already_generated:
                function_stream.write(global_code)
            function_stream.write(nested_global_stream.getvalue())
            function_stream.write(nested_stream.getvalue())

        self._dispatcher.defined_vars.exit_scope(sdfg)

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
        for e in dynamic_map_inputs(state_dfg, node):
            if e.data.data != e.dst_conn:
                callsite_stream.write(
                    self.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]), sdfg,
                    state_id, node)

        inner_stream = CodeIOStream()
        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream, callsite_stream, inner_stream)

        # Instrumentation: Pre-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state_dfg, node, callsite_stream, inner_stream, function_stream)

        # TODO: Refactor to generate_scope_preamble once a general code
        #  generator (that CPU inherits from) is implemented
        if node.map.schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent):
            # OpenMP header
            in_persistent = False
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                in_persistent = is_in_scope(sdfg, state_dfg, node, [dtypes.ScheduleType.CPU_Persistent])
                if in_persistent:
                    # If already in a #pragma omp parallel, no need to use it twice
                    map_header += "#pragma omp for"
                    # TODO(later): barriers and map_header += " nowait"
                else:
                    map_header += "#pragma omp parallel for"

            elif node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
                map_header += "#pragma omp parallel"

            # OpenMP schedule properties
            if not in_persistent:
                if node.map.omp_schedule != dtypes.OMPScheduleType.Default:
                    schedule = " schedule("
                    if node.map.omp_schedule == dtypes.OMPScheduleType.Static:
                        schedule += "static"
                    elif node.map.omp_schedule == dtypes.OMPScheduleType.Dynamic:
                        schedule += "dynamic"
                    elif node.map.omp_schedule == dtypes.OMPScheduleType.Guided:
                        schedule += "guided"
                    else:
                        raise ValueError("Unknown OpenMP schedule type")
                    if node.map.omp_chunk_size > 0:
                        schedule += f", {node.map.omp_chunk_size}"
                    schedule += ")"
                    map_header += schedule

                if node.map.omp_num_threads > 0:
                    map_header += f" num_threads({node.map.omp_num_threads})"

            # OpenMP nested loop properties
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore and node.map.collapse > 1:
                map_header += ' collapse(%d)' % node.map.collapse

        if node.map.unroll:
            if node.map.schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent):
                raise ValueError("An OpenMP map cannot be unrolled (" + node.map.label + ")")

        result.write(map_header, sdfg, state_id, node)

        if node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            result.write('{\n', sdfg, state_id, node)

            # Find if bounds are used within the scope
            scope = state_dfg.scope_subgraph(node, False, False)
            fsyms = scope.free_symbols
            # Include external edges
            for n in scope.nodes():
                for e in state_dfg.all_edges(n):
                    fsyms |= e.data.free_symbols
            fsyms = set(map(str, fsyms))

            ntid_is_used = '__omp_num_threads' in fsyms
            tid_is_used = node.map.params[0] in fsyms
            if tid_is_used or ntid_is_used:
                function_stream.write('#include <omp.h>', sdfg, state_id, node)
            if tid_is_used:
                result.write(f'auto {node.map.params[0]} = omp_get_thread_num();', sdfg, state_id, node)
            if ntid_is_used:
                result.write(f'auto __omp_num_threads = omp_get_num_threads();', sdfg, state_id, node)
        else:
            # Emit nested loops
            for i, r in enumerate(node.map.range):
                var = map_params[i]
                begin, end, skip = r

                if node.map.unroll:
                    result.write("#pragma unroll", sdfg, state_id, node)

                result.write(
                    "for (auto %s = %s; %s < %s; %s += %s) {\n" %
                    (var, cpp.sym2cpp(begin), var, cpp.sym2cpp(end + 1), var, cpp.sym2cpp(skip)),
                    sdfg,
                    state_id,
                    node,
                )

        callsite_stream.write(inner_stream.getvalue())

        # Emit internal transient array allocation
        self._frame.allocate_arrays_in_scope(sdfg, node, function_stream, result)

    def _generate_MapExit(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        map_node = scope_dict[node]
        state_dfg = sdfg.node(state_id)

        if map_node is None:
            raise ValueError("Exit node " + str(node.map.label) + " is not dominated by a scope entry node")

        # Emit internal transient array deallocation
        self._frame.deallocate_arrays_in_scope(sdfg, map_node, function_stream, result)

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None and not is_devicelevel_gpu(sdfg, state_dfg, node):
            instr.on_scope_exit(sdfg, state_dfg, node, outer_stream, callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream, outer_stream, callsite_stream)

        if map_node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            result.write("}", sdfg, state_id, node)
        else:
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

        constsize = all([not symbolic.issymbolic(v, sdfg.constants) for r in node.map.range for v in r])
        state_dfg = sdfg.nodes()[state_id]

        input_sedge = next(e for e in state_dfg.in_edges(node) if e.dst_conn == "IN_stream")
        output_sedge = next(e for e in state_dfg.out_edges(node) if e.src_conn == "OUT_stream")
        input_stream = state_dfg.memlet_path(input_sedge)[0].src
        input_streamdesc = input_stream.desc(sdfg)

        # Take chunks into account
        if node.consume.chunksize == 1:
            ctype = 'const %s' % input_streamdesc.dtype.ctype
            chunk = "%s& %s" % (ctype, "__dace_" + node.consume.label + "_element")
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_element", DefinedType.Scalar, ctype)
        else:
            ctype = 'const %s *' % input_streamdesc.dtype.ctype
            chunk = "%s %s, size_t %s" % (ctype, "__dace_" + node.consume.label + "_elements",
                                          "__dace_" + node.consume.label + "_numelems")
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_elements", DefinedType.Pointer, ctype)
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_numelems", DefinedType.Scalar,
                                              'size_t')

        # Take quiescence condition into account
        if node.consume.condition.code is not None:
            condition_string = "[&]() { return %s; }, " % cppunparse.cppunparse(node.consume.condition.code, False)
        else:
            condition_string = ""

        inner_stream = CodeIOStream()

        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream, callsite_stream, inner_stream)

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state_dfg, node, callsite_stream, inner_stream, function_stream)

        result.write(
            "dace::Consume<{chunksz}>::template consume{cond}({stream_in}, "
            "{num_pes}, {condition}"
            "[&](int {pe_index}, {element_or_chunk}) {{".format(
                chunksz=node.consume.chunksize,
                cond="" if node.consume.condition.code is None else "_cond",
                condition=condition_string,
                stream_in=input_stream.data,  # TODO: stream arrays
                element_or_chunk=chunk,
                num_pes=cpp.sym2cpp(node.consume.num_pes),
                pe_index=node.consume.pe_index,
            ),
            sdfg,
            state_id,
            node,
        )

        # Since consume is an alias node, we create an actual array for the
        # consumed element and modify the outgoing memlet path ("OUT_stream")
        # TODO: do this before getting to the codegen (preprocess)
        if node.consume.chunksize == 1:
            newname, _ = sdfg.add_scalar("__dace_" + node.consume.label + "_element",
                                         input_streamdesc.dtype,
                                         transient=True,
                                         storage=dtypes.StorageType.Register,
                                         find_new_name=True)
            ce_node = nodes.AccessNode(newname)
        else:
            newname, _ = sdfg.add_array("__dace_" + node.consume.label + '_elements', [node.consume.chunksize],
                                        input_streamdesc.dtype,
                                        transient=True,
                                        storage=dtypes.StorageType.Register,
                                        find_new_name=True)
            ce_node = nodes.AccessNode(newname)
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
        self._frame.allocate_arrays_in_scope(sdfg, node, function_stream, result)

        # Generate register definitions for inter-tasklet memlets
        scope_dict = dfg.scope_dict()
        for child in dfg.scope_children()[node]:
            if not isinstance(child, nodes.AccessNode):
                continue

            for edge in dfg.edges():
                # Only interested in edges within current scope
                if scope_dict[edge.src] != node or scope_dict[edge.dst] != node:
                    continue
                # code->code edges
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    local_name = edge.data.data
                    ctype = node.out_connectors[edge.src_conn].ctype
                    if not local_name:
                        # Very unique name. TODO: Make more intuitive
                        local_name = '__dace_%d_%d_%d_%d_%s' % (sdfg.sdfg_id, state_id, dfg.node_id(
                            edge.src), dfg.node_id(edge.dst), edge.src_conn)

                    # Allocate variable type
                    code = '%s %s;' % (ctype, local_name)
                    result.write(code, sdfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype)

    def _generate_ConsumeExit(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]
        state_dfg = sdfg.node(state_id)

        if entry_node is None:
            raise ValueError("Exit node " + str(node.consume.label) + " is not dominated by a scope entry node")

        # Emit internal transient array deallocation
        self._frame.deallocate_arrays_in_scope(sdfg, entry_node, function_stream, result)

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_exit(sdfg, state_dfg, node, outer_stream, callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream, outer_stream, callsite_stream)

        result.write("});", sdfg, state_id, node)

        result.write(outer_stream.getvalue())

    def _generate_AccessNode(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
        state_dfg = sdfg.nodes()[state_id]

        if node not in state_dfg.sink_nodes():
            # NOTE: sink nodes are synchronized at the end of a state
            cpp.presynchronize_streams(sdfg, state_dfg, state_id, node, callsite_stream)

        # Instrumentation: Pre-node
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

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
                if (scope_contains_scope(sdict, src_node, node) and sdict[src_node] != sdict[node]):
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
            dfg,
            self._dispatcher,
            callsite_stream,
            False,
            function_stream,
        )

        # Instrumentation: Post-node
        if instr is not None:
            instr.on_node_end(sdfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

    # Methods for subclasses to override

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
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

    def generate_scope_postamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
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

    def generate_tasklet_preamble(self, sdfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
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

    def generate_tasklet_postamble(self, sdfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
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

    def make_ptr_vector_cast(self, *args, **kwargs):
        return cpp.make_ptr_vector_cast(*args, **kwargs)
