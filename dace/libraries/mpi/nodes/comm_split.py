# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandCommSplitMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        color, key = node.validate(parent_sdfg, parent_state)

        if node.grid is None:
            comm = "MPI_COMM_WORLD"
        else:
            comm = f"__state->{node.grid}_comm"

        comm_name = node.name

        node.fields = [
            f'MPI_Comm {comm_name}_comm;',
            f'int {comm_name}_rank;',
            f'int {comm_name}_size;',
        ]

        code = f"""
            MPI_Comm_split({comm}, _color, _key, &__state->{comm_name}_comm);
            MPI_Comm_rank(__state->{comm_name}_comm, &__state->{comm_name}_rank);
            MPI_Comm_size(__state->{comm_name}_comm, &__state->{comm_name}_size);
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
class Comm_split(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandCommSplitMPI,
    }
    default_implementation = "MPI"

    grid = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, grid=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_color", "_key"}, outputs={"_out"}, **kwargs)
        self.grid = grid

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        color, key = None, None

        for e in state.in_edges(self):
            if e.dst_conn == "_color":
                color = sdfg.arrays[e.data.data]
            if e.dst_conn == "_key":
                key = sdfg.arrays[e.data.data]
        
        return color, key
