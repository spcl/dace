# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, integer_output, mpi_tasklet, resolve_comm
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommRankMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        return mpi_tasklet(node, parent_state, f"int __rank; MPI_Comm_rank({comm}, &__rank); _rank = __rank;")


@library.node
class CommRank(MPINode):
    """``MPI_Comm_rank``: the rank of the calling process, on ``_rank``."""

    implementations = {"MPI": ExpandCommRankMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_rank"}, **kwargs)

    def validate(self, sdfg, state):
        integer_output(self, state, sdfg, "_rank")
