# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, integer_input, mpi_tasklet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommF2cMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        return mpi_tasklet(
            node, parent_state, "_comm = MPI_Comm_f2c((MPI_Fint)_fcomm);", _comm=dtypes.opaque("MPI_Comm")
        )


@library.node
class CommF2c(MPINode):
    """``MPI_Comm_f2c``: the ``MPI_Comm`` (``_comm``) of a Fortran integer communicator handle (``_fcomm``)."""

    implementations = {"MPI": ExpandCommF2cMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_fcomm"}, outputs={"_comm"}, **kwargs)

    def validate(self, sdfg, state):
        integer_input(self, state, sdfg, "_fcomm")
