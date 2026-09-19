# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Query the calling process's rank within a communicator as an explicit
dataflow node (``MPI_Comm_rank``).

The communicator is resolved (see
:func:`dace.libraries.mpi.nodes.node.resolve_comm`) from an optional ``_comm``
(raw ``opaque(MPI_Comm)``) or ``_grid`` (process grid) input connector, else the
default world.  The rank is produced on the ``_rank`` integer-scalar output.
"""
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
        # A local ``int`` receives the rank so the ``_rank`` output can be any
        # integer width (MPI_Comm_rank writes a C ``int``).
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
    """Return the calling process's rank in the resolved communicator on the
    ``_rank`` integer-scalar output via ``MPI_Comm_rank``."""

    # Global properties
    implementations = {
        "MPI": ExpandCommRankMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_rank"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: the ``_rank`` output data descriptor in the parent SDFG.
        """
        rank = None
        for e in state.out_edges(self):
            if e.src_conn == "_rank":
                rank = sdfg.arrays[e.data.data]
        if rank is None:
            raise ValueError("CommRank requires an outgoing _rank connector")
        if rank.dtype.base_type not in dtypes.INTEGER_TYPES:
            raise ValueError("CommRank output _rank must be an integer!")
        return rank
