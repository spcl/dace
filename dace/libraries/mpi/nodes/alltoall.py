# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandAlltoallMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), (outbuffer, out_count_str) = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = "MPI_COMM_WORLD"
        if node.grid:
            comm = f"__state->{node.grid}_comm"

        code = f"""
            int size;
            MPI_Comm_size({comm}, &size);
            int sendrecv_amt = {in_count_str} / size;
            MPI_Alltoall(_inbuffer, sendrecv_amt, {in_mpi_dtype_str}, \
                        _outbuffer, sendrecv_amt, {out_mpi_dtype_str}, \
                        {comm});
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Alltoall(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandAlltoallMPI,
    }
    default_implementation = "MPI"

    grid = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, grid=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)
        self.grid = grid

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]

        in_count_str = "XXX"
        out_count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                out_count_str = "*".join(dims)
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_inbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                in_count_str = "*".join(dims)
        
        return (inbuffer, in_count_str), (outbuffer, out_count_str)
