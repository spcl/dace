# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandAllgatherMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), (outbuffer, out_count_str) = node.validate(
            parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inbuffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""
            int _commsize;
            MPI_Comm_size(MPI_COMM_WORLD, &_commsize);
            MPI_Allgather(_inbuffer, {in_count_str}, {in_mpi_dtype_str},
                          _outbuffer, {out_count_str}/_commsize, {out_mpi_dtype_str},
                          MPI_COMM_WORLD);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Allgather(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandAllgatherMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple inbuffer, outbuffer of the data descriptors in the
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
