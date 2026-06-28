# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, input_descriptor_name


@dace.library.expansion
class ExpandBarrierMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        # Communicator resolution mirrors the data collectives (Bcast/Scatter):
        # a wired ``_grid`` cartesian sub-comm, else a Fortran handle converted
        # with ``MPI_Comm_f2c``, else the default world.
        init = ""
        comm = "MPI_COMM_WORLD"
        grid = input_descriptor_name(node, parent_state, '_grid')
        if grid:
            comm = "_grid"
        elif node.fcomm:
            init = f"MPI_Comm __comm = MPI_Comm_f2c({node.fcomm});"
            comm = "__comm"
        code = f"""
            {init}
            MPI_Barrier({comm});"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Barrier(MPINode):
    """A collective ``MPI_Barrier`` -- pure synchronisation, no data buffers.

    Carries side effects (so it is never pruned despite having no dataflow) and
    resolves its communicator from an optional ``_grid`` connector, a Fortran
    ``fcomm`` handle, or the default world."""

    implementations = {
        "MPI": ExpandBarrierMPI,
    }
    default_implementation = "MPI"

    fcomm = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, fcomm=None, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=set(), **kwargs)
        self.fcomm = fcomm

    def validate(self, sdfg, state):
        return
