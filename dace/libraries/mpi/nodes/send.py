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
class ExpandSendPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GETRF.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandSendMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), dest, tag = node.validate(parent_sdfg, parent_state)
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

        code = f"MPI_Send(_buffer, {count_str}, {mpi_dtype_str}, _dest, _tag, MPI_COMM_WORLD);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Send(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandSendMPI,
    }
    default_implementation = "MPI"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_buffer", "_dest", "_tag"},
                         outputs={},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: Buffer, count, mpi_dtype of the input data
        """

        # Squeeze input memlets
        # squeezed1 = copy.deepcopy(in_memlets[0].subset)
        # sqdims1 = squeezed1.squeeze()
        
        buffer, dest, tag = None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]
        
        if dest.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Source must be an integer!")
        if tag.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Tag must be an integer!")
        
        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)
            
        #TODO make sure buffer is contiguous!

        return (buffer, count_str), dest, tag

