# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors


@dace.library.expansion
class ExpandCommRankMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate(parent_sdfg, parent_state)
        comm = resolve_comm(node, parent_state)
        code = f"""
            int __rank;
            MPI_Comm_rank({comm}, &__rank);
            _rank = __rank;"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class CommRank(MPINode):
    """``MPI_Comm_rank``: the calling process's rank, on the ``_rank`` output."""

    # Global properties
    implementations = {
        "MPI": ExpandCommRankMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_rank"}, **kwargs)

    def validate(self, sdfg, state):
        rank = None
        for e in state.out_edges(self):
            if e.src_conn == "_rank":
                rank = sdfg.arrays[e.data.data]
        if rank is None:
            raise ValueError("CommRank requires an outgoing _rank connector")
        if rank.dtype.base_type not in dtypes.INTEGER_TYPES:
            raise ValueError("CommRank output _rank must be an integer!")
        return rank
