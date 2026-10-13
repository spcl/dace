# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    CommNode,
    expanded_input_connectors,
    mpi_communicator,
    nccl_communicator,
    nccl_tasklet,
    scaled,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandAllreduceMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, count_str), outbuffer, in_place = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = mpi_communicator(node, parent_state)

        buffer = "_inbuffer"
        if in_place:
            buffer = "MPI_IN_PLACE"

        code = f"""
            MPI_Allreduce({buffer}, _outbuffer, {count_str}, {mpi_dtype_str},
                          {utils.mpi_op(node.op)}, {comm});
            """
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandAllreduceNCCL(ExpandTransformation):
    """Reduces with ``ncclAllReduce``, in place if the input and output are the same container."""

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, count_str), outbuffer, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_reduction_type(inbuffer.dtype.base_type, node.op)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclAllReduce(_inbuffer, _outbuffer, {scaled(count_str, factor)}, {nccl_type}, "
            f"{utils.nccl_op(node.op)}, __comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body)


@dace.library.node
class Allreduce(CommNode):
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
        desc="Reduction operator: an MPI operator name (MPI_SUM) or a dace.dtypes.ReductionType name (Sum)",
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
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), outbuffer, in_place
