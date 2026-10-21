# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.libraries.comm import utils
from dace.libraries.comm.nodes.node import (
    MpiExpansion,
    NcclExpansion,
    NonblockingCommNode,
    count_expression,
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


@dace.library.expansion
class ExpandBcastMPI(MpiExpansion):
    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate_nonblocking(parent_sdfg, parent_state)
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (buffer, count_str), root = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = utils.MPI_DDT(buffer.dtype.base_type)
        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        init = ""
        comm = mpi_communicator(node, parent_state)
        if comm == "MPI_COMM_WORLD" and node.fcomm:
            init = f"MPI_Comm __comm = MPI_Comm_f2c({node.fcomm});"
            comm = "__comm"

        code = init + "\n"
        code += mpi_collective(node, "Bcast", f"_inbuffer, {count_str}, {mpi_dtype_str}, _root, {comm}")
        code += "_outbuffer = _inbuffer;"
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dtypes.Language.CPP,
            side_effects=True,
        )
        return mpi_request_connector(tasklet)


@dace.library.expansion
class ExpandBcastNCCL(NcclExpansion):
    """Broadcasts in place with ``ncclBroadcast``."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        (buffer, count_str), _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, buffer)
        if node.fcomm:
            raise NotImplementedError("The NCCL expansion of Bcast does not support Fortran communicators")
        nccl_type, factor = utils.nccl_transfer_type(buffer.dtype.base_type)
        ref = "&" if isinstance(buffer, dace.data.Scalar) else ""
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = (
            f"DACE_NCCL_CHECK(ncclBroadcast({ref}_inbuffer, {ref}_outbuffer, {scaled(count_str, factor)}, "
            f"{nccl_type}, int(_root), __comm, __dace_current_stream));"
        )
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class Bcast(NonblockingCommNode):
    # Global properties
    implementations = {
        "MPI": ExpandBcastMPI,
        "NCCL": ExpandBcastNCCL,
    }
    default_implementation = None

    fcomm = dace.properties.Property(dtype=str, allow_none=True, default=None, category="Semantics")

    def __init__(self, name, fcomm=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer", "_root"}, outputs={"_outbuffer"}, **kwargs)
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
            if src_conn == "_outbuffer":
                count_str = count_expression(data.subset)

        return (inbuffer, count_str), root
