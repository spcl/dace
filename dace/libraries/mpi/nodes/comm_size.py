# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, integer_output, mpi_tasklet, resolve_comm
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommSizeMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        return mpi_tasklet(node, parent_state, f"int __size; MPI_Comm_size({comm}, &__size); _size = __size;")


@library.node
class CommSize(MPINode):
    """``MPI_Comm_size``: the number of processes, on ``_size``."""

    implementations = {"MPI": ExpandCommSizeMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=OrderedSet(["_size"]), **kwargs)

    def validate(self, sdfg, state):
        integer_output(self, state, sdfg, "_size")
