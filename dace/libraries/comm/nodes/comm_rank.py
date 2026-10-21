# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, comm_tasklet, integer_output, mpi_communicator
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommRankMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = mpi_communicator(node, parent_state)
        return comm_tasklet(node, parent_state, f"int __rank;\nMPI_Comm_rank({comm}, &__rank);\n_rank = __rank;\n")


@library.node
class CommRank(CommNode):
    """The rank of the calling process in the communicator (``_rank``), as ``MPI_Comm_rank``."""

    implementations = {"MPI": ExpandCommRankMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_rank"}, **kwargs)

    def validate(self, sdfg, state):
        integer_output(self, sdfg, state, "_rank")
