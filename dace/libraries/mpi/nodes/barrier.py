# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm


@dace.library.expansion
class ExpandBarrierMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        # Communicator resolution mirrors the data collectives (Bcast/Scatter):
        # a wired ``_grid`` cartesian sub-comm, else a Fortran handle converted
        # with ``MPI_Comm_f2c``, else the default world.
        init = ""
        comm = resolve_comm(node, parent_state)
        if comm == "MPI_COMM_WORLD" and node.fcomm:
            # Legacy Fortran-comm-handle node property (superseded by a ``_comm``
            # connector fed from a ``Comm_f2c`` node, but kept for direct callers
            # that set ``fcomm`` instead of wiring a connector).
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
