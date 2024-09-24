# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandBcastMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), root = node.validate(parent_sdfg, parent_state)
        dtype = buffer.dtype.base_type
        mpi_dtype_str = "MPI_BYTE"
        if dtype == dtypes.float32:
            mpi_dtype_str = "MPI_FLOAT"
        elif dtype == dtypes.float64:
            mpi_dtype_str = "MPI_DOUBLE"
        elif dtype == dtypes.complex64:
            mpi_dtype_str = "MPI_COMPLEX"
        elif dtype == dtypes.complex128:
            mpi_dtype_str = "MPI_COMPLEX_DOUBLE"
        elif dtype == dtypes.int32:
            mpi_dtype_str = "MPI_INT"
        elif dtype == dtypes.int64:
            mpi_dtype_str = "MPI_LONG_LONG"
        else:
            raise NotImplementedError("The datatype " + str(dtype) + " is not supported!")
        if buffer.dtype.veclen > 1:
            raise NotImplementedError
        if root.dtype.base_type != dtypes.int32 and root.dtype.base_type != dtypes.int64:
            raise ValueError("Bcast root must be an integer!")

        ref = ""
        if isinstance(buffer, dace.data.Scalar):
            ref = "&"

        init = ""
        comm = "MPI_COMM_WORLD"
        if node.grid:
            comm = f"__state->{node.grid}_comm"
        elif node.fcomm:
            init = f"MPI_Comm __comm = MPI_Comm_f2c({node.fcomm});"
            comm = "__comm"

        code = f"""
            {init}
            MPI_Bcast({ref}_inbuffer, {count_str}, {mpi_dtype_str}, _root, {comm});
            _outbuffer = _inbuffer;"""
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Bcast(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandBcastMPI,
    }
    default_implementation = "MPI"

    grid = dace.properties.Property(dtype=str, allow_none=True, default=None)
    fcomm = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, grid=None, fcomm=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer", "_root"}, outputs={"_outbuffer"}, **kwargs)
        self.grid = grid
        self.fcomm = fcomm

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
            raise ValueError("Bcast input and output buffer must be the same!")
        if root.dtype.base_type != dtypes.int32 and root.dtype.base_type != dtypes.int64:
            raise ValueError("Bcast root must be an integer!")

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [symstr(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), root
