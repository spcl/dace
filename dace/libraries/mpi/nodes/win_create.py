# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandWinCreateMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]


    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        win_buffer, win_buf_count_str = node.validate(parent_sdfg, parent_state)
        win_buffer_dtype = dace.libraries.mpi.utils.MPI_DDT(win_buffer.dtype.base_type)
        window_name = node.name

        node.fields = [
            f"MPI_Win {window_name}_window;"
        ]
    
        comm = "MPI_COMM_WORLD"
        if node.comm:
            comm = f"__state->{node.comm}_comm"

        code = f"""
            MPI_Win_create(_win_buffer,
                           {win_buf_count_str},
                           sizeof({win_buffer_dtype}),
                           MPI_INFO_NULL,
                           {comm},
                           &__state->{window_name}_window);
            """
        
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          state_fields=node.fields,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.node
class Win_create(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandWinCreateMPI,
    }
    default_implementation = "MPI"

    comm = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, comm=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_win_buffer"}, outputs={"_out"}, **kwargs)
        self.comm = comm

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        win_buffer = None
        for e in state.in_edges(self):
            if e.dst_conn == "_win_buffer":
                win_buffer = sdfg.arrays[e.data.data]

        win_buf_count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_win_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                win_buf_count_str = "*".join(dims)

        return win_buffer, win_buf_count_str
