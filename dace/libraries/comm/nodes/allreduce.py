# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    NcclExpansion,
    NonblockingCommNode,
    count_expression,
    expanded_input_connectors,
    mpi_average,
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
class ExpandAllreduceMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate_nonblocking(parent_sdfg, parent_state)
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        if node.nonblocking and utils.is_average(node.op):
            raise NotImplementedError(
                "The average of a nonblocking MPI reduction: reduce the sum and divide after the Wait"
            )
        (inbuffer, count_str), outbuffer, in_place = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = mpi_communicator(node, parent_state)

        buffer = "_inbuffer"
        if in_place:
            buffer = "MPI_IN_PLACE"

        code = mpi_collective(
            node,
            "Allreduce",
            f"{buffer}, _outbuffer, {count_str}, {mpi_dtype_str}, {utils.mpi_op(node.op)}, {comm}",
        )
        code += mpi_average(node.op, "_outbuffer", count_str, comm)
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
class ExpandAllreduceNCCL(NcclExpansion):
    """Reduces with ``ncclAllReduce``, in place if the input and output are the same container."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (inbuffer, count_str), outbuffer, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_reduction_type(inbuffer.dtype.base_type, node.op)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclAllReduce(_inbuffer, _outbuffer, {scaled(count_str, factor)}, {nccl_type}, "
            f"{utils.nccl_op(node.op)}, __comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class Allreduce(NonblockingCommNode):
    # Global properties
    implementations = {
        "MPI": ExpandAllreduceMPI,
        "NCCL": ExpandAllreduceNCCL,
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
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer = None, None
        inpname, outname = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outname = e.data.data
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inpname = e.data.data
                inbuffer = sdfg.arrays[e.data.data]

        in_place = False
        if inpname == outname:
            in_place = True

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == "_outbuffer":
                count_str = count_expression(data.subset)

        return (inbuffer, count_str), outbuffer, in_place
