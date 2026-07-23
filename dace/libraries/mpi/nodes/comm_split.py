# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors, validate_integer_descriptor


@dace.library.expansion
class ExpandCommSplitMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        code = f"""
            _newcomm = MPI_COMM_NULL;
            MPI_Comm_split({comm}, _color, _key, &_newcomm);"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP,
                                          side_effects=True)
        conn = {c: (dtypes.opaque("MPI_Comm") if c == '_newcomm' else t) for c, t in tasklet.out_connectors.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class CommSplit(MPINode):
    """``MPI_Comm_split(comm, color, key)``, producing ``_newcomm``."""

    # Global properties
    implementations = {
        "MPI": ExpandCommSplitMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_color", "_key"}, outputs={"_newcomm"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A two-tuple (color, key) of the data descriptors in the parent SDFG.
        """
        color, key = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_color":
                color = sdfg.arrays[e.data.data]
            if e.dst_conn == "_key":
                key = sdfg.arrays[e.data.data]
        validate_integer_descriptor(color, "CommSplit _color")
        validate_integer_descriptor(key, "CommSplit _key")
        return color, key
