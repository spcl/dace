# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    NcclExpansion,
    NonblockingCommNode,
    comm_tasklet,
    count_expression,
    gpu_buffers_ready,
    mpi_collective,
    mpi_communicator,
    mpi_request_connector,
    nccl_communicator,
    nccl_tasklet,
    pack_noncontiguous_buffers,
    scaled,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandAllgatherMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate_nonblocking(parent_sdfg, parent_state)
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (inbuffer, in_count_str), (outbuffer, out_count_str) = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = mpi_communicator(node, parent_state)
        code = (
            gpu_buffers_ready(node, parent_sdfg, parent_state)
            + f"""
            int _commsize;
            MPI_Comm_size({comm}, &_commsize);
            """
        )
        code += mpi_collective(
            node,
            "Allgather",
            f"_inbuffer, {in_count_str}, {in_mpi_dtype_str}, _outbuffer, {out_count_str}/_commsize, "
            f"{out_mpi_dtype_str}, {comm}",
        )
        return mpi_request_connector(comm_tasklet(node, parent_state, code))


@dace.library.expansion
class ExpandAllgatherNCCL(NcclExpansion):
    """Gathers with ``ncclAllGather``: every rank contributes its whole input buffer."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (inbuffer, in_count_str), (outbuffer, _) = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_transfer_type(inbuffer.dtype.base_type)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclAllGather(_inbuffer, _outbuffer, {scaled(in_count_str, factor)}, {nccl_type}, "
            "__comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class Allgather(NonblockingCommNode):
    # Global properties
    implementations = {
        "MPI": ExpandAllgatherMPI,
        "NCCL": ExpandAllgatherNCCL,
    }
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)

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
            if src_conn == "_outbuffer":
                out_count_str = count_expression(data.subset)
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == "_inbuffer":
                in_count_str = count_expression(data.subset)

        return (inbuffer, in_count_str), (outbuffer, out_count_str)
