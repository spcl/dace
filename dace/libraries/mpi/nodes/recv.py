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
class ExpandRecvPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of MPI Recv.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandRecvMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), src, tag = node.validate(
            parent_sdfg, parent_state)
        dtype = buffer.dtype.base_type
        mpi_dtype_str = "MPI_BYTE"
        if dtype == dace.dtypes.float32:
            mpi_dtype_str = "MPI_FLOAT"
        elif dtype == dace.dtypes.float64:
            mpi_dtype_str = "MPI_DOUBLE" 
        elif dtype == dace.dtypes.complex64:
            mpi_dtype_str = "MPI_COMPLEX"
        elif dtype == dace.dtypes.complex128:
            mpi_dtype_str = "MPI_COMPLEX_DOUBLE"
        elif dtype == dace.dtypes.int32:
            mpi_dtype_str = "MPI_INT"

        else:
            print("The datatype "+str(dtype)+" is not supported!")
            raise(NotImplementedError) 
        if buffer.dtype.veclen > 1:
            raise(NotImplementedError)

        code = f"MPI_Recv(_buffer, {count_str}, {mpi_dtype_str}, _src, _tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Recv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandRecvMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_src", "_tag"},
                         outputs={"_buffer"},
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

