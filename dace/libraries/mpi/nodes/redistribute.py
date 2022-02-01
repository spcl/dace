# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
import functools
from .. import environments


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@dace.library.expansion
class ExpandRedistribute(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        inp_buffer, out_buffer = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inp_buffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            out_buffer.dtype.base_type)
        redistr = parent_sdfg.rdistrarrays[node._redistr]
        array_a = parent_sdfg.subarrays[redistr.array_a]
        array_b = parent_sdfg.subarrays[redistr.array_b]

        code = f"""
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Request* req = new MPI_Request[__state->{node._redistr}_sends];
            MPI_Status* status = new MPI_Status[__state->{node._redistr}_sends];
            MPI_Status recv_status;
            if (__state->{array_a.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_sends; ++__idx) {{
                    //printf("I am rank %d and I send to %d\\n", myrank, __state->{node._redistr}_dst_ranks[__idx]);
                    MPI_Isend(_inp_buffer, 1, __state->{node._redistr}_send_types[__idx], __state->{node._redistr}_dst_ranks[__idx], 0, MPI_COMM_WORLD, &req[__idx]);
                }}
            }}
            if (__state->{array_b.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_recvs; ++__idx) {{
                    //printf("I am rank %d and I receive from %d\\n", myrank, __state->{node._redistr}_src_ranks[__idx]);
                    MPI_Recv(_out_buffer, 1, __state->{node._redistr}_recv_types[__idx], __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &recv_status);
                    //MPI_Recv(_out_buffer, {symstr(_prod(out_buffer.shape))}, {out_mpi_dtype_str}, __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &recv_status);
                }}
            }}
            if (__state->{array_a.pgrid}_valid) {{
                MPI_Waitall(__state->{node._redistr}_sends, req, status);
                delete[] req;
                delete[] status;
            }}
            printf("I am rank %d and I finished the redistribution {redistr.array_a} -> {redistr.array_b}\\n", myrank);
            
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Redistribute(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandRedistribute,
    }
    default_implementation = "MPI"

    redistr = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, redistr=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inp_buffer"},
                         outputs={"_out_buffer"},
                         **kwargs)
        self.redistr = redistr

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (inbuffer, outbuffer, root) of the three data
                 descriptors in the parent SDFG.
        """

        inp_buffer, out_buffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_buffer":
                out_buffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_buffer":
                inp_buffer = sdfg.arrays[e.data.data]
        
        return inp_buffer, out_buffer