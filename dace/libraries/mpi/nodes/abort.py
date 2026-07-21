# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors, validate_integer_descriptor


@dace.library.expansion
class ExpandAbortMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        code = f"""
            MPI_Abort({comm}, _errorcode);"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Abort(MPINode):
    """Collective ``MPI_Abort(comm, errorcode)``."""

    # Global properties
    implementations = {
        "MPI": ExpandAbortMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_errorcode"}, outputs=set(), **kwargs)

    def validate(self, sdfg, state):
        errorcode = None
        for e in state.in_edges(self):
            if e.dst_conn == "_errorcode":
                errorcode = sdfg.arrays[e.data.data]
        validate_integer_descriptor(errorcode, "Abort _errorcode")
        return errorcode
