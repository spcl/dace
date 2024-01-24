# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandFreeMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        code = f"""
            MPI_Comm_free(&__state->{node.grid}_comm);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Comm_free(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandFreeMPI,
    }
    default_implementation = "MPI"

    grid = dace.properties.Property(dtype=str, allow_none=False, default=None)

    def __init__(self, name, grid, *args, **kwargs):
        super().__init__(name, *args, inputs={"_in"}, outputs={}, **kwargs)
        self.grid = grid

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        return None
