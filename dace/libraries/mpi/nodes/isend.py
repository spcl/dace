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
class ExpandIsendPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of Isend.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)


@dace.library.expansion
class ExpandIsendMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        buffer, count_str, dest, tag, req = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise(NotImplementedError)

        code = f"MPI_Isend(_buffer, {count_str}, {mpi_dtype_str}, _dest, _tag, MPI_COMM_WORLD, _request);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class Isend(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandIsendMPI,
    }
    default_implementation = "MPI"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_buffer", "_dest", "_tag"},
                         outputs={"_request"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: buffer, count, mpi_dtype, req of the input data
        """
        
        buffer, dest, tag, req = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):        
            if e.src_conn == "_request":
                req = sdfg.arrays[e.data.data]
        
        if dest.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Source must be an integer!")
        if tag.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Tag must be an integer!")
        
        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)
            
        return buffer, count_str, dest, tag, req


