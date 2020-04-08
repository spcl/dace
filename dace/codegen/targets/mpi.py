import dace
from dace import registry, symbolic, dtypes, subsets
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator, make_absolute
from dace.graph import nodes
from dace.config import Config
from dace.properties import LambdaProperty

from dace.codegen import cppunparse

from dace.frontend.python.astutils import unparse
from dace.symbolic import pystr_to_symbolic
import sympy
from itertools import product


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
    if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
        return 1;

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

    def get_generated_codeobjects(self):
        return [self._codeobj]

    @staticmethod
    def cmake_options():
        compiler = make_absolute(Config.get("compiler", "mpi", "executable"))
        return [
            "-DMPI_CXX_COMPILER=\"{}\"".format(compiler),
            "-DDACE_ENABLE_MPI=ON",
        ]

    @property
    def has_initializer(self):
        return True

    @property
    def has_finalizer(self):
        return True

    def generate_scope(self, sdfg, dfg_scope, state_id, function_stream,
                       callsite_stream):
        # Take care of map header
        assert len(dfg_scope.source_nodes()) == 1
        map_header = dfg_scope.source_nodes()[0]

        function_stream.write('extern int __dace_comm_size, __dace_comm_rank;',
                              sdfg, state_id, map_header)

        # Add extra opening brace (dynamic map ranges, closed in MapExit
        # generator)
        callsite_stream.write('{', sdfg, state_id, map_header)

        # if len(map_header.map.params) > 1:
        #     raise NotImplementedError(
        #         'Multi-dimensional MPI maps are not supported')

        mul_factors = [1]
        param_count = len(map_header.map.params)
        if param_count > 1:
            for r in reversed(map_header.map.range[1:]):
                begin, end, skip = r
                mul_factors.append((end + 1 - begin) // skip)
        print('Mul factors for MPI map are {}'.format(mul_factors))
        from functools import reduce
        mul = reduce(lambda x, y: x * y, mul_factors)
        print('Mul for MPI map is {}'.format(mul))

        callsite_stream.write(
            'int __dace_comm_rank_r = %s;\n' % ('__dace_comm_rank'),
            sdfg, state_id, map_header)

        for var, r, f in zip(map_header.map.params, map_header.map.range, reversed(mul_factors)):
            begin, end, skip = r

            # callsite_stream.write('{\n', sdfg, state_id, map_header)
            # callsite_stream.write(
            #     'auto %s = %s + __dace_comm_rank * (%s);\n' %
            #     (var, cppunparse.pyexpr2cpp(symbolic.symstr(begin)),
            #      cppunparse.pyexpr2cpp(symbolic.symstr(skip))), sdfg, state_id,
            #     map_header)
            callsite_stream.write(
                'int %s = __dace_comm_rank_r / (%s);\n' %
                (var, cppunparse.pyexpr2cpp(symbolic.symstr(mul))),
                sdfg, state_id, map_header)
            callsite_stream.write(
                '__dace_comm_rank_r -= %s * %s;\n' %
                (var, cppunparse.pyexpr2cpp(symbolic.symstr(mul))),
                sdfg, state_id, map_header)
            mul //= f

        to_allocate = dace.sdfg.local_transients(sdfg, dfg_scope, map_header)
        allocated = set()
        for child in dfg_scope.scope_dict(node_to_children=True)[map_header]:
            if not isinstance(child, nodes.AccessNode):
                continue
            if child.data not in to_allocate or child.data in allocated:
                continue
            allocated.add(child.data)
            self._dispatcher.dispatch_allocate(sdfg, dfg_scope, state_id,
                                               child, function_stream,
                                               callsite_stream)
            self._dispatcher.dispatch_initialize(sdfg, dfg_scope, state_id,
                                                 child, function_stream,
                                                 callsite_stream)

        self._dispatcher.dispatch_subgraph(sdfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        pass

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass

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

        while (not isinstance(parent, nodes.MapEntry)
                or parent.map.schedule != dtypes.ScheduleType.MPI):
            parent = dfg.scope_dict()[parent]
            if parent is None:
                raise Exception("Distributed copy outside MPI map")

        mul_factors = [1]
        param_count = len(edge.data.dist_subset)
        if param_count > 1:
            for r in reversed(parent.map.range[1:]):
                begin, end, skip = r
                mul_factors.append((end + 1 - begin) // skip)
        print('Mul factors for MPI map are {}'.format(mul_factors))
        from functools import reduce
        mul = reduce(lambda x, y: x * y, mul_factors)
        print('Mul for MPI map is {}'.format(mul))

        data = sdfg.arrays[memlet.data]

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

        dist_location = LambdaProperty.from_string(data.dist_location)
        body = pystr_to_symbolic(unparse(dist_location.body))
        args = [pystr_to_symbolic(a.arg) for a in dist_location.args.args]
        if len(args) != len(index):
            raise ValueError(
                "The number of arguments of the distributed location lambda "
                "method does not match the length of the memlet subset")
        repl = {arg: idx for arg, idx in zip(args, index)}
        other_rank = body.subs(repl)
        callsite_stream.write("int other_rank = %s;\n" % str(other_rank),
                              sdfg, state_id, [src_node, dst_node])

        eqs = []
        for k, v in data.dist_shape_map.items():
            eqs.append(pystr_to_symbolic(
                '{i} - my_{r}'.format(i=index[k], r=parent.map.params[v])))
        symbols = [pystr_to_symbolic(p) for p in parent.map.params]
        solution = sympy.solve(eqs, *symbols)
        repl = {pystr_to_symbolic('my_{r}'.format(r=r)): r
                for r in symbols}
        fixed = {}
        ranges = {}
        for i, (var, r) in enumerate(zip(symbols, parent.map.range)):
            if var in solution.keys():
                fixed[i] = solution[var].subs(repl)
            else:
                ranges[i] = r

        dist_location = LambdaProperty.from_string(parent.map.dist_location)
        body = pystr_to_symbolic(unparse(dist_location.body))
        args = [pystr_to_symbolic(a.arg) for a in dist_location.args.args]
        # repl = {arg: r for arg, r in zip(args, symbols)}

        # callsite_stream.write("int src_rank = 0;\n", sdfg, state_id, [src_node, dst_node])
        # for r, f in zip(edge.data.dist_subset, reversed(mul_factors[-param_count:])):
        #     begin, end, skip = r
        #     if begin != end:
        #         raise NotImplementedError("Only distributed indices are supported")
        #     callsite_stream.write(
        #         "src_rank += %s * %s;\n" %
        #         (begin, cppunparse.pyexpr2cpp(symbolic.symstr(mul))),
        #         sdfg, state_id,[src_node, dst_node])
        #     mul //= f

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            callsite_stream.write(
                "double {n};\nMPI_Request req;\n"
                "MPI_Irecv(&{n}, 1, MPI_DOUBLE, other_rank, tag, &req)\n".format(n=vconn),
                sdfg,
                state_id,
                [src_node, dst_node],
            )
            for k, v in fixed.items():
                callsite_stream.write(
                    "int {a} = {v};\n".format(a=args[k], v=v),
                    sdfg, state_id, [src_node, dst_node])
            for k, v in ranges.items():
                callsite_stream.write(
                    "for (int {a} = {b}; {a} < {e}; {a} += {s}) {{\n".format(
                        a=args[k], b=v[0], e=v[1]+1, s=v[2]),
                    sdfg, state_id, [src_node, dst_node])
            callsite_stream.write(
                "other_rank = {e};\n".format(e=body),
                sdfg, state_id, [src_node, dst_node])
            callsite_stream.write(
                "MPI_Isend(&{n}, 1, MPI_DOUBLE, other_rank, tag, &req)\n".format(n=vconn)
            )
            for i in range(len(ranges)):
                callsite_stream.write('}\n', sdfg, state_id, [src_node, dst_node])
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            callsite_stream.write(
                 "    double {};".format(uconn),
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

        #     #############################################
        #     # Corner cases

        #     # Writing one index
        #     if (isinstance(memlet.subset, subsets.Indices)
        #             and memlet.wcr is None
        #             and self._dispatcher.defined_vars.get(
        #                 vconn) == DefinedType.Scalar):
        #         stream.write(
        #             "%s = %s;" %
        #             (vconn, self.memlet_ctor(sdfg, memlet, False)),
        #             sdfg,
        #             state_id,
        #             [src_node, dst_node],
        #         )
        #         return
        #     # Writing from/to a stream
        #     if isinstance(sdfg.arrays[memlet.data], data.Stream) or (
        #             isinstance(src_node, nodes.AccessNode)
        #             and isinstance(src_nodedesc, data.Stream)):
        #         # Identify whether a stream is writing to an array
        #         if isinstance(dst_nodedesc,
        #                       (data.Scalar, data.Array)) and isinstance(
        #                           src_nodedesc, data.Stream):
        #             # Stream -> Array - pop bulk
        #             if is_array_stream_view(sdfg, dfg, src_node):
        #                 return  # Do nothing (handled by ArrayStreamView)

        #             array_subset = (memlet.subset
        #                             if memlet.data == dst_node.data else
        #                             memlet.other_subset)
        #             if array_subset is None:  # Need to use entire array
        #                 array_subset = subsets.Range.from_array(dst_nodedesc)

        #             # stream_subset = (memlet.subset
        #             #                  if memlet.data == src_node.data else
        #             #                  memlet.other_subset)
        #             stream_subset = memlet.subset
        #             if memlet.data != src_node.data and memlet.other_subset:
        #                 stream_subset = memlet.other_subset

        #             stream_expr = cpp_offset_expr(src_nodedesc, stream_subset)
        #             array_expr = cpp_offset_expr(dst_nodedesc, array_subset)
        #             assert functools.reduce(lambda a, b: a * b,
        #                                     src_nodedesc.shape, 1) == 1
        #             stream.write(
        #                 "{s}.pop(&{arr}[{aexpr}], {maxsize});".format(
        #                     s=src_node.data,
        #                     arr=dst_node.data,
        #                     aexpr=array_expr,
        #                     maxsize=sym2cpp(array_subset.num_elements())),
        #                 sdfg,
        #                 state_id,
        #                 [src_node, dst_node],
        #             )
        #             return
        #         # Array -> Stream - push bulk
        #         if isinstance(src_nodedesc,
        #                       (data.Scalar, data.Array)) and isinstance(
        #                           dst_nodedesc, data.Stream):
        #             if hasattr(src_nodedesc, "src"):  # ArrayStreamView
        #                 stream.write(
        #                     "{s}.push({arr});".format(s=dst_node.data,
        #                                               arr=src_nodedesc.src),
        #                     sdfg,
        #                     state_id,
        #                     [src_node, dst_node],
        #                 )
        #             else:
        #                 copysize = " * ".join(
        #                     [sym2cpp(s) for s in memlet.subset.size()])
        #                 stream.write(
        #                     "{s}.push({arr}, {size});".format(
        #                         s=dst_node.data,
        #                         arr=src_node.data,
        #                         size=copysize),
        #                     sdfg,
        #                     state_id,
        #                     [src_node, dst_node],
        #                 )
        #             return
        #         else:
        #             # Unknown case
        #             raise NotImplementedError

        #     #############################################

        #     state_dfg = sdfg.nodes()[state_id]

        #     copy_shape, src_strides, dst_strides, src_expr, dst_expr = \
        #         memlet_copy_to_absolute_strides(
        #             self._dispatcher, sdfg, memlet, src_node, dst_node,
        #             self._packed_types)

        #     # Which numbers to include in the variable argument part
        #     dynshape, dynsrc, dyndst = 1, 1, 1

        #     # Dynamic copy dimensions
        #     if any(symbolic.issymbolic(s, sdfg.constants) for s in copy_shape):
        #         copy_tmpl = "Dynamic<{type}, {veclen}, {aligned}, {dims}>".format(
        #             type=ctype,
        #             veclen=1,  # Taken care of in "type"
        #             aligned="false",
        #             dims=len(copy_shape),
        #         )
        #     else:  # Static copy dimensions
        #         copy_tmpl = "<{type}, {veclen}, {aligned}, {dims}>".format(
        #             type=ctype,
        #             veclen=1,  # Taken care of in "type"
        #             aligned="false",
        #             dims=", ".join(sym2cpp(copy_shape)),
        #         )
        #         dynshape = 0

        #     # Constant src/dst dimensions
        #     if not any(
        #             symbolic.issymbolic(s, sdfg.constants)
        #             for s in dst_strides):
        #         # Constant destination
        #         shape_tmpl = "template ConstDst<%s>" % ", ".join(
        #             sym2cpp(dst_strides))
        #         dyndst = 0
        #     elif not any(
        #             symbolic.issymbolic(s, sdfg.constants)
        #             for s in src_strides):
        #         # Constant source
        #         shape_tmpl = "template ConstSrc<%s>" % ", ".join(
        #             sym2cpp(src_strides))
        #         dynsrc = 0
        #     else:
        #         # Both dynamic
        #         shape_tmpl = "Dynamic"

        #     # Parameter pack handling
        #     stride_tmpl_args = [0] * (dynshape + dynsrc +
        #                               dyndst) * len(copy_shape)
        #     j = 0
        #     for shape, src, dst in zip(copy_shape, src_strides, dst_strides):
        #         if dynshape > 0:
        #             stride_tmpl_args[j] = shape
        #             j += 1
        #         if dynsrc > 0:
        #             stride_tmpl_args[j] = src
        #             j += 1
        #         if dyndst > 0:
        #             stride_tmpl_args[j] = dst
        #             j += 1

        #     copy_args = ([src_expr, dst_expr] +
        #                  ([] if memlet.wcr is None else
        #                   [unparse_cr(sdfg, memlet.wcr)]) +
        #                  sym2cpp(stride_tmpl_args))

        #     # Instrumentation: Pre-copy
        #     for instr in self._dispatcher.instrumentation.values():
        #         if instr is not None:
        #             instr.on_copy_begin(sdfg, state_dfg, src_node, dst_node,
        #                                 edge, stream, None, copy_shape,
        #                                 src_strides, dst_strides)

        #     nc = True
        #     if memlet.wcr is not None:
        #         nc = not is_write_conflicted(
        #             dfg, edge, sdfg_schedule=self._toplevel_schedule)
        #     if nc:
        #         stream.write(
        #             """
        #             dace::CopyND{copy_tmpl}::{shape_tmpl}::{copy_func}(
        #                 {copy_args});""".format(
        #                 copy_tmpl=copy_tmpl,
        #                 shape_tmpl=shape_tmpl,
        #                 copy_func="Copy"
        #                 if memlet.wcr is None else "Accumulate",
        #                 copy_args=", ".join(copy_args),
        #             ),
        #             sdfg,
        #             state_id,
        #             [src_node, dst_node],
        #         )
        #     else:  # Conflicted WCR
        #         if dynshape == 1:
        #             raise NotImplementedError(
        #                 "Accumulation of dynamically-shaped "
        #                 "arrays not yet implemented")
        #         elif copy_shape == [
        #                 1
        #         ]:  # Special case: accumulating one element
        #             dst_expr = self.memlet_view_ctor(sdfg, memlet, True)
        #             stream.write(
        #                 write_and_resolve_expr(sdfg, memlet, nc, dst_expr,
        #                                        '*(' + src_expr + ')'), sdfg,
        #                 state_id, [src_node, dst_node])
        #         else:
        #             raise NotImplementedError("Accumulation of arrays "
        #                                       "with WCR not yet implemented")

        # #############################################################
        # # Instrumentation: Post-copy
        # for instr in self._dispatcher.instrumentation.values():
        #     if instr is not None:
        #         instr.on_copy_end(sdfg, state_dfg, src_node, dst_node, edge,
        #                           stream, None)
        # #############################################################