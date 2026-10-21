# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import CommNode, comm_tasklet, external_communicator, mpi_communicator
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCommFreeMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        code = f"MPI_Comm __comm = {mpi_communicator(node, parent_state)};\nMPI_Comm_free(&__comm);\n"
        return comm_tasklet(node, parent_state, code)


@library.node
class CommFree(CommNode):
    """Frees the communicator given to ``_comm`` (e.g., one :class:`CommSplit` created), as ``MPI_Comm_free``."""

    implementations = {"MPI": ExpandCommFreeMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_comm"}, outputs=set(), **kwargs)

    def validate(self, sdfg, state):
        if not external_communicator(self, state):
            raise ValueError(f'CommFree "{self.label}" needs the communicator to free on its _comm connector')
