# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, comm_tasklet, mpi_communicator, validate_integer_connector
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommSizeMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = mpi_communicator(node, parent_state)
        return comm_tasklet(node, parent_state, f"int __size;\nMPI_Comm_size({comm}, &__size);\n_size = __size;\n")


@library.node
class CommSize(CommNode):
    """The number of processes of the communicator (``_size``), as ``MPI_Comm_size``."""

    implementations = {"MPI": ExpandCommSizeMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_size"}, **kwargs)

    def validate(self, sdfg, state):
        validate_integer_connector(self, sdfg, state, "_size")
