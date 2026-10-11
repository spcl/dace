# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, mpi_tasklet, resolve_comm
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandBarrierMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        return mpi_tasklet(node, parent_state, f"MPI_Barrier({comm});")


@library.node
class Barrier(MPINode):
    """``MPI_Barrier``: blocks until every process of the communicator reaches it."""

    implementations = {"MPI": ExpandBarrierMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=set(), **kwargs)
