# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import warnings

import dace
from dace import registry, symbolic, dtypes, data, subsets
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import cpp
from dace.codegen.targets.target import (TargetCodeGenerator, make_absolute,
                                         DefinedType)
from dace.sdfg import nodes, scope_contains_scope
from dace.config import Config

from dace.codegen import cppunparse


def _dot(seq1, seq2):
    seq3 = [a * b for a, b in zip(seq1, seq2)]
    return functools.reduce(lambda a, b: a + b, seq3, 0)


@registry.autoregister_params(name='mpi')
class MPICodeGen(TargetCodeGenerator):
    """ An MPI code generator. """
    target_name = 'mpi'
    title = 'MPI'
    language = 'cpp'

    def __init__(self, frame_codegen, sdfg):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        fileheader = CodeIOStream()
        self._frame.generate_fileheader(sdfg, fileheader)

        self._codeobj = CodeObject(
            sdfg.name + '_mpi', """
#include <dace/dace.h>
#include <mpi.h>

MPI_Comm __dace_mpi_comm;
int __dace_comm_size = 1;
int __dace_comm_rank = 0;

{file_header}

DACE_EXPORTED int __dace_init_mpi({params});
DACE_EXPORTED void __dace_exit_mpi({params});

int __dace_init_mpi({params}) {{
    int isinit = 0;
    if (MPI_Initialized(&isinit) != MPI_SUCCESS)
        return 1;
    if (!isinit) {{
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
            return 1;
    }}

    MPI_Comm_dup(MPI_COMM_WORLD, &__dace_mpi_comm);
    MPI_Comm_rank(__dace_mpi_comm, &__dace_comm_rank);
    MPI_Comm_size(__dace_mpi_comm, &__dace_comm_size);

    printf(\"MPI was initialized on proc %i of %i\\n\", __dace_comm_rank,
           __dace_comm_size);
    return 0;
}}

void __dace_exit_mpi({params}) {{
    MPI_Comm_free(&__dace_mpi_comm);
    MPI_Finalize();

    printf(\"MPI was finalized on proc %i of %i\\n\", __dace_comm_rank,
           __dace_comm_size);
}}
""".format(params=sdfg.signature(), file_header=fileheader.getvalue()), 'cpp',
            MPICodeGen, 'MPI')

        # Register dispatchers
        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()
        self._dispatcher.register_node_dispatcher(
            self, predicate=lambda *_: True)

        # Register dispatchers
        dispatcher.register_map_dispatcher(dtypes.ScheduleType.MPI, self)
        dispatcher.register_array_dispatcher(dtypes.StorageType.Distributed, self)
        for schedule in dtypes.ScheduleType:
            dispatcher.register_copy_dispatcher(
                dtypes.StorageType.Distributed, dtypes.StorageType.Register,
                schedule, self)
            dispatcher.register_copy_dispatcher(
                dtypes.StorageType.Register, dtypes.StorageType.Distributed,
                schedule, self)
        # dispatcher.register_node_dispatcher(self, predicate)

    def get_generated_codeobjects(self):
        return [self._codeobj]

    @staticmethod
    def cmake_options():
        options = ['-DDACE_ENABLE_MPI=ON']

        if Config.get("compiler", "mpi", "executable"):
            compiler = make_absolute(Config.get("compiler", "mpi",
                                                "executable"))
            options.append("-DMPI_CXX_COMPILER=\"{}\"".format(compiler))

        return options

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return True
    
    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)

        if (isinstance(nodedesc, data.Array) and
                nodedesc.storage == dtypes.StorageType.Distributed):
            ndims = len(nodedesc.dist_shape)
            arrsize = cpp.sym2cpp(nodedesc.total_size)
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            ctype = nodedesc.dtype.ctype
            # comm_name = "__dace_comm_{n}".format(n=name)
            cart_name = "__dace_cart_{n}".format(n=name)
            dims = "__dace_dims_{n}".format(n=name)
            coords = "__dace_coords_{n}".format(n=name)
            # periods = "__dace_periods_{n}".format(n=name)
            # reorder = "__dace_reorder_{n}".format(n=name)
            win_name = "__dace_win_{n}".format(n=name)
            # callsite_stream.write(
            #     "MPI_Comm {c};\n"
            #     "int {dims}[{d}];\n"
            #     "int {coords}[{d}];\n"
            #     "int {periods}[{d}];\n"
            #     "int {reorder} = 0;".format(
            #         c=comm_name, d=ndims, dims=dims, coords=coords,
            #         periods=periods, reorder=reorder),
            #     sdfg, state_id, node)
            callsite_stream.write(
                "int {dims}[{d}];\n"
                "int {coords}[{d}];".format(d=ndims, dims=dims, coords=coords),
                sdfg, state_id, node)
            for i, s in enumerate(nodedesc.dist_shape):
                # TODO: Assume non-periodic for now
                # callsite_stream.write(
                #     "{dims}[{i}] = {s};\n"
                #     "{periods}[{i}] = 0;".format(
                #         i=i, s=s, dims=dims, periods=periods),
                #     sdfg, state_id, node)
                callsite_stream.write(
                    "{dims}[{i}] = {s};\n".format(i=i, s=s, dims=dims),
                    sdfg, state_id, node)
            # callsite_stream.write(
            #     "MPI_Cart_create(MPI_COMM_WORLD, {n}, {dims}, "
            #     "{periods}, {reorder}, &{c});\n"
            #     "MPI_Cart_coords({c}, __dace_comm_rank, "
            #     "{n}, {coords});".format(
            #         c=comm_name, n=ndims, dims=dims, periods=periods,
            #         reorder=reorder, coords=coords),
            #     sdfg, state_id, node)
            callsite_stream.write(
                "Cart {cart}({n}, {dims});\n"
                "{cart}.coords(__dace_comm_rank, {coords});".format(
                    n=ndims, dims=dims, coords=coords, cart=cart_name),
                sdfg, state_id, node)
            if nodedesc.transient:
                # TODO: Is there any reason to use MPI_Alloc_mem?
                callsite_stream.write(
                    "{t} *{n} = new {t} DACE_ALIGN(64)[{s}];".format(
                        t=ctype, n=name, s=arrsize),
                    sdfg, state_id, node)
            callsite_stream.write("MPI_Win {w};".format(w=win_name),
                                  sdfg, state_id, node)
            callsite_stream.write(
                "MPI_Win_create({n}, {s} * sizeof({t}), sizeof({t}), "
                "MPI_INFO_NULL, {c}, &{w});".format(
                    n=name, s=arrsize, t=ctype, c="MPI_COMM_WORLD", w=win_name),  # c=comm_name
                sdfg, state_id, node)
            if nodedesc.transient:
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer,
                                                  ctypedef)
        else:
            return

        # TODO: Fix other Data and Storage options

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)
        
        if (isinstance(nodedesc, data.Array) and
                nodedesc.storage == dtypes.StorageType.Distributed):
            winname = "__dace_win_{}".format(name)
            callsite_stream.write("MPI_Win_free(&{w});".format(w=winname),
                                  sdfg, state_id, node)
            if nodedesc.transient:
                callsite_stream.write("delete[] {n};".format(n=name),
                                      sdfg, state_id, node)
        else:
            return

        # TODO: Fix other Data and Storage options

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        # Take care of map header
        assert len(dfg_scope.source_nodes()) == 1
        map_header = dfg_scope.source_nodes()[0]
        map_footer = dfg_scope.sink_nodes()[0]

        function_stream.write("#include <mpi.h>\n"  # TODO: Where does this actually belong?
                              "#include <dace/mpi/cart.h>\n"
                              "extern int __dace_comm_size, __dace_comm_rank;",
                              sdfg, state_id, map_header)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        # comm_name = "__dace_comm_{}".format(map_header.map.label)
        # callsite_stream.write("MPI_Comm {c};\n".format(c=comm_name),
        #                       sdfg, state_id, map_header)
        # callsite_stream.write('{', sdfg, state_id, map_header)

        # if len(map_header.map.params) > 1:
        #     raise NotImplementedError(
        #         'Multi-dimensional MPI maps are not supported')

        ndims = len(map_header.map.params)
        sdims = [e + 1 for _, e, _ in map_header.map.range]
        name = "{sd}_{st}_{n}".format(
            sd=sdfg.sdfg_list.index(sdfg),
            st=state_id,
            n=sdfg.nodes()[state_id].node_id(map_header))
        cart_name = "__dace_cart_{n}".format(n=name)
        dims = "__dace_dims_{n}".format(n=name)
        coords = "__dace_coords_{n}".format(n=name)
        callsite_stream.write(
            "int {dims}[{d}];\n"
            "int {coords}[{d}];".format(d=ndims, dims=dims, coords=coords),
            sdfg, state_id, map_header)
        for i, s in enumerate(sdims):
            callsite_stream.write(
                "{dims}[{i}] = {s};\n".format(i=i, s=s, dims=dims),
                sdfg, state_id, map_header)
        callsite_stream.write(
            "Cart {cart}({n}, {dims});\n"
            "bool fits_cart_{name} = {cart}.fits(__dace_comm_rank);\n"
            "{cart}.coords(__dace_comm_rank, {coords});".format(
                n=ndims, dims=dims, name=name, coords=coords, cart=cart_name),
            sdfg, state_id, map_header)

        inp_windows = set()
        state = sdfg.nodes()[state_id]
        for e in state.in_edges(map_header):
            path = state.memlet_path(e)
            # TODO: Input path, therefore input data in path[0]?
            if isinstance(path[0].src, nodes.AccessNode):
                data_name = path[0].src.data
                nodedesc = sdfg.arrays[data_name]
                if nodedesc.storage == dtypes.StorageType.Distributed:
                    inp_windows.add("__dace_win_{n}".format(n=data_name))
        out_windows = set()
        state = sdfg.nodes()[state_id]
        for e in state.out_edges(map_footer):
            path = state.memlet_path(e)
            # TODO: Output path, therefore input data in path[-1]?
            if isinstance(path[-1].dst, nodes.AccessNode):
                data_name = path[-1].dst.data
                nodedesc = sdfg.arrays[data_name]
                if nodedesc.storage == dtypes.StorageType.Distributed:
                    out_windows.add("__dace_win_{n}".format(n=data_name))
        for win in inp_windows:
            callsite_stream.write("MPI_Win_lock_all(0, {w});".format(w=win),
                                  sdfg, state_id, map_header)
        for win in out_windows:
            callsite_stream.write("MPI_Win_fence(0, {w});".format(w=win),
                                  sdfg, state_id, map_header)

        callsite_stream.write('if (fits_cart_{name}) {{'.format(name=name),
                              sdfg, state_id, map_header)
        
        state = sdfg.node(state_id)
        symtypes = map_header.new_symbols(sdfg, state,
                                          state.symbols_defined_at(map_header))

        for i, (var, r) in enumerate(zip(map_header.map.params,
                                         map_header.map.range)):
            begin, end, skip = r

            callsite_stream.write('{\n', sdfg, state_id, map_header)
            callsite_stream.write(
                "{t} {v} = {coords}[{i}];\n".format(
                    t=symtypes[var], coords=coords, v=var, i=i),
                sdfg, state_id, map_header)  # TODO: Bound checking

        to_allocate = dace.sdfg.local_transients(sdfg, dfg_scope, map_header)
        allocated = set()
        for child in dfg_scope.scope_dict(node_to_children=True)[map_header]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg_scope, state_id, child,
                                               function_stream, callsite_stream)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

        for win in inp_windows:
            callsite_stream.write("MPI_Win_unlock_all({w});".format(w=win),
                                  sdfg, state_id, map_header)
        for win in out_windows:
            callsite_stream.write("MPI_Win_fence(0, {w});".format(w=win),
                                  sdfg, state_id, map_header)
        if not out_windows:
            callsite_stream.write(
                "MPI_Barrier({c});".format(c="MPI_COMM_WORLD"),
                sdfg, state_id, map_header)

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
        u, uconn, v, vconn, memlet = edge

        # Determine memlet directionality
        if isinstance(src_node,
                      nodes.AccessNode) and memlet.data == src_node.data:
            write = True
            parent = dst_node
        elif isinstance(dst_node,
                        nodes.AccessNode) and memlet.data == dst_node.data:
            write = False
            parent = src_node
        elif isinstance(src_node, nodes.CodeNode) and isinstance(
                dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError(
                "Copying between code nodes is only supported as"
                " part of the participating nodes")
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        # Find the map/map entry of the MPI scope we are in.
        while (not isinstance(parent, nodes.MapEntry)
                or parent.map.schedule != dtypes.ScheduleType.MPI):
            parent = dfg.scope_dict()[parent]
            if parent is None:
                raise Exception("Distributed copy outside MPI map")

        data = sdfg.arrays[memlet.data]

        # Extract the distributed subset index
        # This is needed because we may either have subsets.Indices
        # or subsets.Range in the distributed subset.
        # Currently we only support Indices and
        # Range of length 1 in all dimensions.
        if isinstance(edge.data.dist_subset, subsets.Indices):
            index = edge.data.dist_subset
        else:
            index = []
            for r in edge.data.dist_subset:
                if len(r) == 3:
                    begin, end, _ = r
                else:
                    begin, end, _, _ = r
                if begin != end:
                    raise NotImplementedError(
                        "Only distributed indices are currently supported")
                index.append(begin)

        if isinstance(edge.data.subset, subsets.Indices):
            local_index = edge.data.subset
        else:
            local_index = []
            for r in edge.data.subset:
                if len(r) == 3:
                    begin, end, _ = r
                else:
                    begin, end, _, _ = r
                if begin != end:
                    raise NotImplementedError(
                        "Only distributed indices are currently supported")
                local_index.append(begin)

        ndims = len(index)
        if isinstance(dst_node, nodes.Tasklet):
            name = vconn
            ctype = dst_node.in_connectors[vconn].dtype.ctype
        elif isinstance(src_node, nodes.Tasklet):
            name = uconn
            ctype = src_node.out_connectors[uconn].dtype.ctype
        mpitype = dtypes._MPITYPES[ctype]
        data_name = memlet.data
        # comm_name = "__dace_comm_{n}".format(n=data_name)
        win_name = "__dace_win_{n}".format(n=data_name)
        cart_name = "__dace_cart_{n}".format(n=data_name)
        trank = "__dace_target_rank_{n}".format(n=name)
        tcoords = "__dace_target_coords_{n}".format(n=name)
        callsite_stream.write(
            "int {tcoords}[{n}];".format(tcoords=tcoords, n=ndims),
            sdfg, state_id, [src_node, dst_node])
        for i, s in enumerate(index):
            callsite_stream.write(
                "{tcoords}[{i}] = {s};".format(tcoords=tcoords, i=i, s=s),
                sdfg, state_id, [src_node, dst_node])
        # callsite_stream.write(
        #     "int {trank};\n"
        #     "MPI_Cart_rank({c}, {tcoords}, &{trank});".format(
        #         trank=trank, c=comm_name, tcoords=tcoords),
        #     sdfg, state_id, [src_node, dst_node])
        callsite_stream.write(
            "int {trank};\n"
            "{cart}.rank({tcoords}, {trank});".format(
                trank=trank, tcoords=tcoords, cart=cart_name),
            sdfg, state_id, [src_node, dst_node])
        disp = _dot(local_index, data.strides)

        if isinstance(dst_node, nodes.Tasklet):
            callsite_stream.write(
                "{ct} {conn};\n"
                "MPI_Get(&{conn}, 1, {mt}, {trank}, {dp}, 1, {mt}, {w});\n"
                "MPI_Win_flush_local({trank}, {w});".format(
                    ct=ctype, conn=vconn, mt=mpitype, trank=trank,
                    dp=disp, w=win_name),
                sdfg, state_id, [src_node, dst_node])
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            if memlet.wcr:
                # TODO: Match wcr with MPI_Op (dtypes?)
                callsite_stream.write(
                    "MPI_Accumulate(&{conn}, 1, {mt}, {trank}, {dp}, 1, "
                    "{mt}, {op}, {w});".format(
                        ct=ctype, conn=uconn, mt=mpitype,
                        trank=trank, dp=disp, op="MPI_SUM", w=win_name),
                sdfg, state_id, [src_node, dst_node])
            else:
                callsite_stream.write(
                    "MPI_Put(&{conn}, 1, {mt}, {trank}, {dp}, 1, "
                    "{mt}, {w});".format(ct=ctype, conn=uconn, mt=mpitype,
                                         trank=trank, dp=disp, w=win_name),
                sdfg, state_id, [src_node, dst_node])
        else:  # Copy array-to-array
            # src_nodedesc = src_node.desc(sdfg)
            # dst_nodedesc = dst_node.desc(sdfg)

            # if write:
            #     vconn = dst_node.data
            # ctype = "dace::vec<%s, %d>" % (dst_nodedesc.dtype.ctype,
            #                                memlet.veclen)
            raise NotImplementedError
    
    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        method_name = "_generate_" + type(node).__name__
        # Fake inheritance... use this class' method if it exists,
        # otherwise fall back on CPU codegen
        if hasattr(self, method_name):

            # if hasattr(node, "schedule") and node.schedule not in [
            #         dace.dtypes.ScheduleType.Default,
            #         dace.dtypes.ScheduleType.FPGA_Device
            # ]:
            #     warnings.warn("Found schedule {} on {} node in FPGA code. "
            #                   "Ignoring.".format(node.schedule,
            #                                      type(node).__name__))

            getattr(self, method_name)(sdfg, dfg, state_id, node,
                                       function_stream, callsite_stream)
        else:
            old_codegen = self._cpu_codegen.calling_codegen
            self._cpu_codegen.calling_codegen = self

            self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                            function_stream, callsite_stream)

            self._cpu_codegen.calling_codegen = old_codegen

    def _generate_Tasklet(self, *args, **kwargs):
        # Call CPU implementation with this code generator as callback
        self._cpu_codegen._generate_Tasklet(*args, codegen=self, **kwargs)

    # def define_out_memlet(self, sdfg, state_dfg, state_id, src_node, dst_node,
    #                       edge, function_stream, callsite_stream):
    #     self._dispatcher.dispatch_copy(src_node, dst_node, edge, sdfg,
    #                                    state_dfg, state_id, function_stream,
    #                                    callsite_stream)

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
                if not shared_data_name:
                    # Very unique name. TODO: Make more intuitive
                    shared_data_name = '__dace_%d_%d_%d_%d_%s' % (
                        sdfg.sdfg_id, state_id, dfg.node_id(node),
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
            if scope_dict[node] != scope_dict[dst_node] and scope_contains_scope(
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

                conntype = node.out_connectors[uconn]
                is_scalar = not isinstance(conntype, dtypes.pointer)
                is_stream = isinstance(sdfg.arrays[memlet.data], data.Stream)

                if is_scalar and not memlet.dynamic and not is_stream:
                    out_local_name = "    __" + uconn
                    in_local_name = uconn
                    if not locals_defined:
                        out_local_name = self.memlet_ctor(
                            sdfg, memlet, node.out_connectors[uconn], True)
                        in_memlets = [d for _, _, _, _, d in dfg.in_edges(node)]
                        assert len(in_memlets) == 1
                        in_local_name = self.memlet_ctor(
                            sdfg, in_memlets[0], node.out_connectors[uconn],
                            False)

                    state_dfg = sdfg.nodes()[state_id]

                    if memlet.wcr is not None:
                        if edge.data.dist_subset:
                            self.copy_memory(
                                sdfg, dfg, state_id, node, dst_node,
                                edge, function_stream, result)
                        else:
                            nc = not cpp.is_write_conflicted(
                                dfg, edge,
                                sdfg_schedule=self._cpu_codegen._toplevel_schedule)
                            result.write(
                                self._cpu_codegen.write_and_resolve_expr(
                                    sdfg,
                                    memlet,
                                    nc,
                                    out_local_name,
                                    in_local_name,
                                    dtype=node.out_connectors[uconn]) + ';', sdfg,
                                state_id, node)
                        continue
                    else:
                        try:
                            defined_type, _ = self._dispatcher.defined_vars.get(
                                memlet.data)
                        except KeyError:  # The variable is not defined
                            # This case happens with nested SDFG outputs,
                            # which we skip since the memlets are references
                            if isinstance(node, nodes.NestedSDFG):
                                continue
                            raise

                        if defined_type == DefinedType.Scalar:
                            expr = memlet.data
                        elif defined_type == DefinedType.ArrayInterface:
                            # Special case: No need to write anything between
                            # array interfaces going out
                            try:
                                deftype, _ = self._dispatcher.defined_vars.get(
                                    in_local_name)
                            except KeyError:
                                deftype = None
                            if deftype == DefinedType.ArrayInterface:
                                return

                            expr = '*(%s + %s).ptr_out()' % (
                                memlet.data,
                                cpp.cpp_array_expr(
                                    sdfg, memlet, with_brackets=False))
                        else:
                            # TODO: Always pointer at this point?
                            # TODO: Do nothing?
                            # expr = cpp.cpp_array_expr(sdfg, memlet)
                            # # If there is a type mismatch, cast pointer
                            # expr = cpp.make_ptr_vector_cast(
                            #     sdfg, expr, memlet, conntype, is_scalar,
                            #     defined_type)
                            self.copy_memory(sdfg, dfg, state_id, node, dst_node,
                                             edge, function_stream, result)
                            continue

                        result.write(
                            "%s = %s;\n" % (expr, in_local_name),
                            sdfg,
                            state_id,
                            node,
                        )
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

    def generate_tasklet_preamble(self, *args, **kwargs):
        # Fall back on CPU implementation
        self._cpu_codegen.generate_tasklet_preamble(*args, **kwargs)

    def generate_tasklet_postamble(self, *args, **kwargs):
        # Fall back on CPU implementation
        self._cpu_codegen.generate_tasklet_postamble(*args, **kwargs)
    
    def unparse_tasklet(self, *args, **kwargs):
        # Fall back on CPU implementation
        self._cpu_codegen.unparse_tasklet(*args, **kwargs)
    
    def define_out_memlet(self, *args, **kwargs):
        # Fall back on CPU implementation
        self._cpu_codegen.define_out_memlet(*args, **kwargs)
