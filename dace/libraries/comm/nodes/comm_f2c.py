# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, comm_tasklet, validate_integer_connector
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommF2cMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        return comm_tasklet(node, parent_state, "_handle = (long long)(MPI_Comm_f2c((MPI_Fint)(_fcomm)));\n")


@library.node
class CommF2c(CommNode):
    """
    The handle (``_handle``) of the communicator that a Fortran integer handle (``_fcomm``, e.g., ``MPI_COMM_WORLD`` of
    a Fortran caller) stands for, as ``MPI_Comm_f2c``. It is the ``_comm`` of the nodes that communicate over it.
    """

    implementations = {"MPI": ExpandCommF2cMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=["_fcomm"], outputs=["_handle"], **kwargs)

    def validate(self, sdfg, state):
        validate_integer_connector(self, sdfg, state, "_fcomm")
        validate_integer_connector(self, sdfg, state, "_handle")
