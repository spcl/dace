# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    NcclExpansion,
    NonblockingCommNode,
    count_expression,
    gpu_buffers_ready,
    mpi_average,
    mpi_collective,
    mpi_communicator,
    mpi_request_connector,
    mpi_tasklet,
    nccl_communicator,
    nccl_tasklet,
    pack_noncontiguous_buffers,
    scaled,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandReduceScatterMPI(ExpandTransformation):
    """Reduces and scatters with ``MPI_Reduce_scatter_block``."""

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate_nonblocking(parent_sdfg, parent_state)
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        inbuffer, (outbuffer, out_count_str) = node.validate(parent_sdfg, parent_state)
        if inbuffer.dtype.veclen > 1:
            raise NotImplementedError("MPI expansions do not support vector types")
        mpi_dtype_str = utils.MPI_DDT(inbuffer.dtype.base_type)
        comm = mpi_communicator(node, parent_state)
        if node.nonblocking and utils.is_average(node.op):
            raise NotImplementedError(
                "The average of a nonblocking MPI reduction: reduce the sum and divide after the Wait"
            )
        code = gpu_buffers_ready(node, parent_sdfg, parent_state) + mpi_collective(
            node,
            "Reduce_scatter_block",
            f"_inbuffer, _outbuffer, {out_count_str}, {mpi_dtype_str}, {utils.mpi_op(node.op)}, {comm}",
        )
        code += mpi_average(node.op, "_outbuffer", out_count_str, comm)
        return mpi_request_connector(mpi_tasklet(node, parent_state, code))


@dace.library.expansion
class ExpandReduceScatterNCCL(NcclExpansion):
    """Reduces and scatters with ``ncclReduceScatter``."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        inbuffer, (outbuffer, out_count_str) = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_reduction_type(inbuffer.dtype.base_type, node.op)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclReduceScatter(_inbuffer, _outbuffer, {scaled(out_count_str, factor)}, {nccl_type}, "
            f"{utils.nccl_op(node.op)}, __comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class ReduceScatter(NonblockingCommNode):
    """
    Reduces the input buffers of all processes elementwise and scatters the result: the input (``_inbuffer``) holds
    one chunk per process, and process ``i`` receives the reduction of the ``i``-th chunks in its output
    (``_outbuffer``), which is as large as one chunk. The buffers must be contiguous.
    """

    implementations = {
        "MPI": ExpandReduceScatterMPI,
        "NCCL": ExpandReduceScatterNCCL,
    }
    default_implementation = None

    op = dace.properties.Property(
        dtype=str,
        default="MPI_SUM",
        category="Semantics",
        desc="Reduction operator: an MPI operator name (MPI_SUM), a dace.dtypes.ReductionType name (Sum), or a "
        "canonical name (sum, avg, ...; see dace.libraries.comm.utils.reduction_op)",
    )

    def __init__(self, name, op="MPI_SUM", *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)
        self.op = op

    def validate(self, sdfg, state):
        """
        :return: The input buffer's data descriptor, and the output buffer's data descriptor with the number of
                 elements every process receives.
        """
        inbuffer, outbuffer, out_count_str = None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
                out_count_str = count_expression(e.data.subset)
        if inbuffer is None or outbuffer is None:
            raise ValueError(f'ReduceScatter "{self.label}" needs an input and an output buffer')
        return inbuffer, (outbuffer, out_count_str)
