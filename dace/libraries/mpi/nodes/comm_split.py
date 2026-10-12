# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library
from dace.libraries.mpi import environments
from dace.libraries.mpi.nodes.node import MPINode, integer_input, mpi_tasklet, resolve_comm
from dace.ordered import OrderedSet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommSplitMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        return mpi_tasklet(
            node,
            parent_state,
            f"_newcomm = MPI_COMM_NULL; MPI_Comm_split({comm}, _color, _key, &_newcomm);",
            _newcomm=dtypes.opaque("MPI_Comm"),
        )


@library.node
class CommSplit(MPINode):
    """``MPI_Comm_split``: the communicator of the processes sharing ``_color``, ranked by ``_key``, on
    ``_newcomm``."""

    implementations = {"MPI": ExpandCommSplitMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=OrderedSet(["_color", "_key"]), outputs=OrderedSet(["_newcomm"]), **kwargs)

    def validate(self, sdfg, state):
        integer_input(self, state, sdfg, "_color")
        integer_input(self, state, sdfg, "_key")
