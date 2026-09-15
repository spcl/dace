# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Create an ``MPI_Comm`` inside an SDFG from a Fortran integer communicator
handle, as an explicit dataflow node.

A Fortran kernel hands MPI calls an ``INTEGER`` communicator (the ``MPI_Fint``
handle from ``MPI_Comm_c2f`` / the Fortran ``comm`` argument).  This node converts
it to a C ``MPI_Comm`` with ``MPI_Comm_f2c`` and produces it on a ``_comm``
output -- an ``opaque(MPI_Comm)`` value that flows through the graph to the
``_comm`` input connector of ``Isend`` / ``Irecv`` / ``Barrier`` / the
collectives (see :func:`dace.libraries.mpi.nodes.node.resolve_comm`).

This is the dataflow-native alternative to the legacy per-node ``fcomm``
property: the communicator becomes a first-class value (created here, consumed
via ``_comm``), so multiple distinct communicators are naturally supported and
the host can pass / receive one across the SDFG boundary as an
``opaque(MPI_Comm)`` argument.
"""
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
    """Convert a Fortran integer communicator handle (``_fcomm``) to an
    ``MPI_Comm`` (``_comm``, ``opaque(MPI_Comm)``) via ``MPI_Comm_f2c``."""

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
