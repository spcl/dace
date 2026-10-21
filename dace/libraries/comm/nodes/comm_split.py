# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    communicator_output,
    integer_input,
    mpi_communicator,
    mpi_tasklet,
)
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommSplitMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = mpi_communicator(node, parent_state)
        code = f"""
            MPI_Comm __newcomm;
            MPI_Comm_split({comm}, int(_color), int(_key), &__newcomm);
            _newcomm = (long long)(__newcomm);
            """
        return mpi_tasklet(node, parent_state, code)


@library.node
class CommSplit(CommNode):
    """
    Splits the communicator into one per value of ``_color``, ranked by ``_key``, as ``MPI_Comm_split``. The handle of
    the calling process's new communicator (``_newcomm``) is the ``_comm`` of the nodes that communicate over it; free
    it with :class:`CommFree`.
    """

    implementations = {"MPI": ExpandCommSplitMPI}
    default_implementation = "MPI"
    outputs_mpi_communicator = True

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_color", "_key"}, outputs={"_newcomm"}, **kwargs)

    def validate(self, sdfg, state):
        integer_input(self, sdfg, state, "_color")
        integer_input(self, sdfg, state, "_key")
        communicator_output(self, sdfg, state, "_newcomm")
