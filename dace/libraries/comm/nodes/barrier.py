# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, mpi_communicator, mpi_tasklet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandBarrierMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        return mpi_tasklet(node, parent_state, f"MPI_Barrier({mpi_communicator(node, parent_state)});")


@library.node
class Barrier(CommNode):
    """Blocks until every process of the communicator reaches it, as ``MPI_Barrier``."""

    implementations = {"MPI": ExpandBarrierMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=set(), **kwargs)
