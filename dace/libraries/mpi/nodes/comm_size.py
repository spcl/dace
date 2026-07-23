# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors


@dace.library.expansion
class ExpandCommSizeMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        code = f"""
            int __size;
            MPI_Comm_size({comm}, &__size);
            _size = __size;"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class CommSize(MPINode):
    """``MPI_Comm_size``: the number of processes, on the ``_size`` output."""

    # Global properties
    implementations = {
        "MPI": ExpandCommSizeMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_size"}, **kwargs)

    def validate(self, sdfg, state):
        size = None
        for e in state.out_edges(self):
            if e.src_conn == "_size":
                size = sdfg.arrays[e.data.data]
        if size is None:
            raise ValueError("CommSize requires an outgoing _size connector")
        if size.dtype.base_type not in dtypes.INTEGER_TYPES:
            raise ValueError("CommSize output _size must be an integer!")
        return size
