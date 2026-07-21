# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, expanded_input_connectors


@dace.library.expansion
class ExpandCommF2cMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        code = "_comm = MPI_Comm_f2c((MPI_Fint)_fcomm);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        conn = {c: (dtypes.opaque("MPI_Comm") if c == '_comm' else t) for c, t in tasklet.out_connectors.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class CommF2c(MPINode):
    """``MPI_Comm_f2c``: convert a Fortran integer communicator handle
    (``_fcomm``) to an ``MPI_Comm`` (``_comm``)."""

    # Global properties
    implementations = {
        "MPI": ExpandCommF2cMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_fcomm"}, outputs={"_comm"}, **kwargs)

    def validate(self, sdfg, state):
        fcomm = None
        for e in state.in_edges(self):
            if e.dst_conn == "_fcomm":
                fcomm = sdfg.arrays[e.data.data]
        if fcomm is None:
            raise ValueError("CommF2c requires a Fortran integer communicator handle on the _fcomm connector")
        return fcomm
