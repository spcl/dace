# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, integer_input, mpi_tasklet, resolve_comm
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandAbortMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        return mpi_tasklet(node, parent_state, f"MPI_Abort({comm}, _errorcode);")


@library.node
class Abort(MPINode):
    """``MPI_Abort``: terminates every process of the communicator with ``_errorcode``."""

    implementations = {"MPI": ExpandAbortMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=OrderedSet(["_errorcode"]), outputs=set(), **kwargs)

    def validate(self, sdfg, state):
        integer_input(self, state, sdfg, "_errorcode")
