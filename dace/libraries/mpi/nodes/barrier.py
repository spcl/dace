# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors


@dace.library.expansion
class ExpandBarrierMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        init = ""
        comm = resolve_comm(node, parent_state)
        if comm == "MPI_COMM_WORLD" and node.fcomm:
            # fcomm is the legacy per-node handle, superseded by a _comm connector.
            init = f"MPI_Comm __comm = MPI_Comm_f2c({node.fcomm});"
            comm = "__comm"
        code = f"""
            {init}
            MPI_Barrier({comm});"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Barrier(MPINode):
    """Collective ``MPI_Barrier``."""

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
