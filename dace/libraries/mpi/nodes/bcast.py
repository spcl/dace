# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandBcastMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), root = node.validate(parent_sdfg, parent_state)
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
            print("The datatype " + str(dtype) + " is not supported!")
            raise (NotImplementedError)
        if buffer.dtype.veclen > 1:
            raise (NotImplementedError)
        if root.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Bcast root must be an integer!")

        ref = ""
        if isinstance(buffer, dace.data.Scalar):
            ref = "&"

        comm = "MPI_COMM_WORLD"
        if node.grid:
            comm = f"__state->{node.grid}_comm"

        code = f"""
            MPI_Bcast({ref}_inbuffer, {count_str}, {mpi_dtype_str}, _root, {comm});
            _outbuffer = _inbuffer;"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Bcast(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandBcastMPI,
    }
    default_implementation = "MPI"

    grid = dace.properties.Property(dtype=str, allow_none=True, default=None)
    
    def __init__(self, name, grid=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_root"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self.grid = grid

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer, src, tag = None, None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_root":
                root = sdfg.arrays[e.data.data]

        if inbuffer != outbuffer:
            raise (
                ValueError("Bcast input and output buffer must be the same!"))
        if root.dtype.base_type != dace.dtypes.int32:
            raise (ValueError("Bcast root must be an integer!"))

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [symstr(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), root
