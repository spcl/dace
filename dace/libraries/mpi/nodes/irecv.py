# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandIrecvPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of MPI Irecv.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)


@dace.library.expansion
class ExpandIrecvMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), src, tag = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise(NotImplementedError)

        code = f"MPI_Irecv(_buffer, {count_str}, {mpi_dtype_str}, _src, _tag, MPI_COMM_WORLD, &_request);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Irecv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandIrecvMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_src", "_tag"},
                         outputs={"_buffer", "_request"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, src, tag) of the three data descriptors in the
                 parent SDFG.
        """
        
        buffer, src, tag = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_src":
                src = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]
        
        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):  
            if src_conn == '_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (buffer, count_str), src, tag

