# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    NcclExpansion,
    NonblockingCommNode,
    expanded_input_connectors,
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
class ExpandAlltoallMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        node.validate_request(parent_state)
        (inbuffer, in_count_str), (outbuffer, out_count_str) = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = mpi_communicator(node, parent_state)

        code = f"""
            int size;
            MPI_Comm_size({comm}, &size);
            int sendrecv_amt = {in_count_str} / size;
            """
        code += mpi_collective(
            node,
            "Alltoall",
            f"_inbuffer, sendrecv_amt, {in_mpi_dtype_str}, _outbuffer, sendrecv_amt, {out_mpi_dtype_str}, {comm}",
        )
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        return mpi_request_connector(tasklet)


@dace.library.expansion
class ExpandAlltoallNCCL(NcclExpansion):
    """Exchanges an equal chunk of the input buffer with every rank through ``ncclAlltoAll``, which
    ``dace::comm::AlltoAll`` replaces with grouped point-to-point calls where NCCL lacks it."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state, stream_ordered=True)
        (inbuffer, in_count_str), (outbuffer, _) = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_transfer_type(inbuffer.dtype.base_type)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = f"""
            int __size;
            DACE_NCCL_CHECK(ncclCommCount(__comm, &__size));
            const long long __chunk = ({in_count_str}) / __size;
            dace::comm::AlltoAll(_inbuffer, _outbuffer, {scaled("__chunk", factor)}, {nccl_type}, __comm,
                                 __dace_current_stream);
            """
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class Alltoall(NonblockingCommNode):
    # Global properties
    implementations = {
        "MPI": ExpandAlltoallMPI,
        "NCCL": ExpandAlltoallNCCL,
    }
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)

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
            if src_conn == "_outbuffer":
                dims = [str(e) for e in data.subset.size_exact()]
                out_count_str = "*".join(dims)
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == "_inbuffer":
                dims = [str(e) for e in data.subset.size_exact()]
                in_count_str = "*".join(dims)

        return (inbuffer, in_count_str), (outbuffer, out_count_str)
