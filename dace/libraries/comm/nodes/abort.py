# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, comm_tasklet, integer_input, mpi_communicator
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandAbortMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        return comm_tasklet(node, parent_state, f"MPI_Abort({mpi_communicator(node, parent_state)}, int(_errorcode));")


@library.node
class Abort(CommNode):
    """Terminates every process of the communicator with the error code ``_errorcode``, as ``MPI_Abort``."""

    implementations = {"MPI": ExpandAbortMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_errorcode"}, outputs=set(), **kwargs)

    def validate(self, sdfg, state):
        integer_input(self, sdfg, state, "_errorcode")
