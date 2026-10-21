# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    CommNode,
    NcclExpansion,
    buffer_pointer,
    by_reference,
    count_expression,
    gpu_buffers_ready,
    mpi_average,
    mpi_communicator,
    mpi_tasklet,
    nccl_communicator,
    nccl_tasklet,
    pack_noncontiguous_buffers,
    scaled,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandReduceMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (inbuffer, count_str), outbuffer, root, in_place = node.validate(parent_sdfg, parent_state)
        inptr, outptr = (
            buffer_pointer(node, parent_sdfg, parent_state, "_inbuffer"),
            buffer_pointer(node, parent_sdfg, parent_state, "_outbuffer"),
        )
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = mpi_communicator(node, parent_state)
        op = utils.mpi_op(node.op)

        code = gpu_buffers_ready(node, parent_sdfg, parent_state)
        if in_place and not by_reference(node, parent_sdfg, parent_state, "_outbuffer"):
            code += "_outbuffer = _inbuffer;\n"  # The tasklet holds copies: the output starts from the input
        if in_place:
            code += f"""
                int __comm_rank;
                MPI_Comm_rank({comm}, &__comm_rank);
                if (__comm_rank == _root) {{
            """
            code += f"""
                    MPI_Reduce(MPI_IN_PLACE, {outptr}, {count_str}, {mpi_dtype_str}, {op}, _root, {comm});
                }} else {{
            """
        code += f"MPI_Reduce({inptr}, {outptr}, {count_str}, {mpi_dtype_str}, {op}, _root, {comm});"
        if in_place:
            code += "}"
        code += mpi_average(node.op, outptr, count_str, comm, root="_root")
        return mpi_tasklet(node, parent_state, code)


@dace.library.expansion
class ExpandReduceNCCL(NcclExpansion):
    """Reduces with ``ncclReduce``, in place if the input and output are the same container."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (inbuffer, count_str), outbuffer, _, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_reduction_type(inbuffer.dtype.base_type, node.op)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclReduce(_inbuffer, _outbuffer, {scaled(count_str, factor)}, {nccl_type}, "
            f"{utils.nccl_op(node.op)}, int(_root), __comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body)


@dace.library.node
class Reduce(CommNode):
    # Global properties
    implementations = {
        "MPI": ExpandReduceMPI,
        "NCCL": ExpandReduceNCCL,
    }
    default_implementation = None

    op = dace.properties.Property(
        dtype=str,
        default="MPI_SUM",
        category="Semantics",
        desc="Reduction operator: an MPI operator name (MPI_SUM) or a dace.dtypes.ReductionType name (Sum)",
    )

    def __init__(self, name, op="MPI_SUM", *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer", "_root"}, outputs={"_outbuffer"}, **kwargs)
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
            if e.dst_conn == "_root":
                root = sdfg.arrays[e.data.data]

        in_place = False
        if inpname == outname:
            in_place = True

        if root.dtype.base_type != dace.dtypes.int32:
            raise (ValueError("Reduce root must be an integer!"))

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == "_outbuffer":
                count_str = count_expression(data.subset)

        return (inbuffer, count_str), outbuffer, root, in_place
