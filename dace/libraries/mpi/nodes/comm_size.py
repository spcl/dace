# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Query the number of processes in a communicator as an explicit dataflow node
(``MPI_Comm_size``).

The communicator is resolved (see
:func:`dace.libraries.mpi.nodes.node.resolve_comm`) from an optional ``_comm``
(raw ``opaque(MPI_Comm)``) or ``_grid`` (process grid) input connector, else the
default world.  The size is produced on the ``_size`` integer-scalar output.
"""
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
        # A local ``int`` receives the size so the ``_size`` output can be any
        # integer width (MPI_Comm_size writes a C ``int``).
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
    """Return the number of processes in the resolved communicator on the
    ``_size`` integer-scalar output via ``MPI_Comm_size``."""

    # Global properties
    implementations = {
        "MPI": ExpandCommSizeMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs={"_size"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: the ``_size`` output data descriptor in the parent SDFG.
        """
        size = None
        for e in state.out_edges(self):
            if e.src_conn == "_size":
                size = sdfg.arrays[e.data.data]
        if size is None:
            raise ValueError("CommSize requires an outgoing _size connector")
        if size.dtype.base_type not in dtypes.INTEGER_TYPES:
            raise ValueError("CommSize output _size must be an integer!")
        return size
