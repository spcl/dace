import dace
from dace import registry, symbolic, dtypes
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator, make_absolute
from dace.graph import nodes
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

    def get_generated_codeobjects(self):
        return [self._codeobj]

    @staticmethod
    def cmake_options():
        options = ['-DDACE_ENABLE_MPI=ON']

        if Config.get("compiler", "mpi", "executable"):
            compiler = make_absolute(
                Config.get("compiler", "mpi", "executable"))
            options.append("-DMPI_CXX_COMPILER=\"{}\"".format(compiler))

        return options

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

        if len(map_header.map.params) > 1:
            raise NotImplementedError(
                'Multi-dimensional MPI maps are not supported')

        for var, r in zip(map_header.map.params, map_header.map.range):
            begin, end, skip = r

            callsite_stream.write('{\n', sdfg, state_id, map_header)
            callsite_stream.write(
                'auto %s = %s + __dace_comm_rank * (%s);\n' %
                (var, cppunparse.pyexpr2cpp(symbolic.symstr(begin)),
                 cppunparse.pyexpr2cpp(symbolic.symstr(skip))), sdfg, state_id,
                map_header)

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
