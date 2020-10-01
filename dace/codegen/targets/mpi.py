# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import warnings

import dace
from dace import registry, symbolic, dtypes, data, subsets
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets import cpp
from dace.codegen.targets.target import (TargetCodeGenerator, make_absolute,
                                         DefinedType)
from dace.sdfg import nodes
from dace.config import Config

from dace.codegen import cppunparse


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
        print(nodedesc)

        if not nodedesc.transient:  # Nested SDFG connector?
            return

        if (isinstance(nodedesc, data.Array) and
                nodedesc.storage == dtypes.StorageType.Distributed):
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            arrsize = cpp.sym2cpp(nodedesc.total_size)
            ctype = nodedesc.dtype.ctype
            winname = "__mpiwin_{}".format(name)
            callsite_stream.write(
                "{t} *{n} = new {t} DACE_ALIGN(64)[{s}];\n"  # TODO: Is there any reason to use MPI_Alloc_mem?
                "MPI_Win {w};\n"
                "MPI_Win_create({n}, {s} * sizeof({t}), sizeof({t}), "
                "MPI_INFO_NULL, MPI_COMM_WORLD, &{w});\n".format(
                    n=name, s=arrsize, t=ctype, w=winname
                ),
                sdfg,
                state_id,
                node
            )
            self._dispatcher.defined_vars.add(name, DefinedType.Pointer,
                                              ctypedef)
        else:
            return

        # TODO: Fix other Data and Storage options

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)

        if not nodedesc.transient:
            return
        
        if (isinstance(nodedesc, data.Array) and
                nodedesc.storage == dtypes.StorageType.Distributed):
            winname = "__mpiwin_{}".format(name)
            callsite_stream.write(
                "MPI_Win_free(&w);\n"
                "delete[] {n};\n".format(n=name, w=winname),
                sdfg,
                state_id,
                node
            )
        else:
            return

        # TODO: Fix other Data and Storage options

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        # Take care of map header
        assert len(dfg_scope.source_nodes()) == 1
        map_header = dfg_scope.source_nodes()[0]

        function_stream.write("#include <mpi.h>\n"  # TODO: Where does this actually belong?
                              "extern int __dace_comm_size, __dace_comm_rank;",
                              sdfg, state_id, map_header)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        comm_name = "__dace_comm_{}".format(map_header.map.label)
        callsite_stream.write("MPI_Comm {c};\n".format(c=comm_name),
                              sdfg, state_id, map_header)
        callsite_stream.write('{', sdfg, state_id, map_header)

        # if len(map_header.map.params) > 1:
        #     raise NotImplementedError(
        #         'Multi-dimensional MPI maps are not supported')

        ndims = len(map_header.map.params)
        sdims = [e + 1 for _, e, _ in map_header.map.range]
        callsite_stream.write("int dims[{n}];\n"
                              "int coords[{n}];\n"
                              "int periods[{n}];\n"
                              "int reorder = 0;".format(c=comm_name, n=ndims),
                              sdfg, state_id, map_header)  # TODO: Unique names?
        for i, s in enumerate(sdims):
            callsite_stream.write("dims[{i}] = {s};\n"
                                  "periods[{i}] = 0;".format(i=i, s=s),
                                  sdfg, state_id, map_header)  # TODO: Assume non-periodic for now
        callsite_stream.write("MPI_Cart_create(MPI_COMM_WORLD, {n}, dims, "
                              "periods, reorder, &{c});\n"
                              "MPI_Cart_coords({c}, __dace_comm_rank, "
                              "{n}, coords);".format(c=comm_name, n=ndims),
                              sdfg, state_id, map_header)
        callsite_stream.write("MPI_Barrier({c});".format(c=comm_name),
                              sdfg, state_id, map_header)
        
        state = sdfg.node(state_id)
        symtypes = map_header.new_symbols(sdfg, state,
                                          state.symbols_defined_at(map_header))

        for i, (var, r) in enumerate(zip(map_header.map.params,
                                         map_header.map.range)):
            begin, end, skip = r

            callsite_stream.write('{\n', sdfg, state_id, map_header)
            callsite_stream.write(
                "{t} {v} = coords[{i}];\n".format(t=symtypes[var], v=var, i=i),
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

        callsite_stream.write("MPI_Barrier({c});".format(c=comm_name),
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

        # # Parse distributed location lambda method to a sympy expression.
        # # Replace the arguments of the lambda with the parameters
        # # of the MPI map, in order to find the rank-owner of the source data.
        # dist_location = LambdaProperty.from_string(data.dist_location)
        # body = pystr_to_symbolic(unparse(dist_location.body))
        # args = [pystr_to_symbolic(a.arg) for a in dist_location.args.args]
        # if len(args) != len(index):
        #     raise ValueError(
        #         "The number of arguments of the distributed location lambda "
        #         "method does not match the length of the memlet subset")
        # repl = {arg: idx for arg, idx in zip(args, index)}
        # other_rank = body.subs(repl)
        # callsite_stream.write("int other_rank = %s;\n" % str(other_rank),
        #                       sdfg, state_id, [src_node, dst_node])

        # # Use the index matching of the pair (data, map) to create a system
        # # of equations. Solve the system to find the expressions that describe
        # # the coordinates of the ranks that request the source data.
        # eqs = []
        # for k, v in data.dist_shape_map.items():
        #     eqs.append(pystr_to_symbolic(
        #         '{i} - my_{r}'.format(i=index[k], r=parent.map.params[v])))
        # symbols = [pystr_to_symbolic(p) for p in parent.map.params]
        # solution = sympy.solve(eqs, *symbols)
        # repl = {pystr_to_symbolic('my_{r}'.format(r=r)): r
        #         for r in symbols}
        # fixed = {}
        # ranges = {}
        # for i, (var, r) in enumerate(zip(symbols, parent.map.range)):
        #     if var in solution.keys():
        #         fixed[i] = solution[var].subs(repl)
        #     else:
        #         ranges[i] = r

        # # Parse the distributed location of the MPI map to sympy expression.
        # # We will use this expression to compute the actual ranks of the
        # # processes that request the source data.
        # dist_location = LambdaProperty.from_string(parent.map.dist_location)
        # body = pystr_to_symbolic(unparse(dist_location.body))
        # args = [pystr_to_symbolic(a.arg) for a in dist_location.args.args]

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            # int MPI_Get(
            #     void *origin_addr,
            #     int origin_count,
            #     MPI_Datatype origin_datatype, 
            #     int target_rank,
            #     MPI_Aint target_disp,
            #     int target_count,
            #     MPI_Datatype target_datatype,
            #     MPI_Win win
            #     );
            callsite_stream.write(
                "MPI_Get Data:{d} Subset:{s} Target:{t}".format(
                    d=memlet.data, s=memlet.subset, t=index
                ), sdfg, state_id, [src_node, dst_node])
                # TODO: We also need some kind of flush here
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            if memlet.wcr:
                callsite_stream.write(
                    "MPI_Accumulate Data:{d} Subset:{s} Target:{t}".format(
                    d=memlet.data, s=memlet.subset, t=index
                ), sdfg, state_id, [src_node, dst_node])
            else:
                callsite_stream.write(
                    "MPI_Put Data:{d} Subset:{s} Target:{t}".format(
                    d=memlet.data, s=memlet.subset, t=index
                ), sdfg, state_id, [src_node, dst_node])
        else:  # Copy array-to-array
            # src_nodedesc = src_node.desc(sdfg)
            # dst_nodedesc = dst_node.desc(sdfg)

            # if write:
            #     vconn = dst_node.data
            # ctype = "dace::vec<%s, %d>" % (dst_nodedesc.dtype.ctype,
            #                                memlet.veclen)
            raise NotImplementedError
