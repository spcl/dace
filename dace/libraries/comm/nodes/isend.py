# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    NcclExpansion,
    comm_tasklet,
    count_expression,
    gpu_buffers_ready,
    mpi_communicator,
    mpi_request_connector,
    mpi_vector_datatype,
    nccl_communicator,
    nccl_point_to_point,
    nccl_request,
    nccl_tasklet,
    validate_integer_descriptor,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandIsendMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), dest, tag, req = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        comm = mpi_communicator(node, parent_state)
        code = "" if node.nosync else gpu_buffers_ready(node, parent_sdfg, parent_state)

        vector_code, vector, vector_count, free = mpi_vector_datatype(ddt)
        code += vector_code
        mpi_dtype_str = vector or mpi_dtype_str
        count_str = vector_count or count_str
        code += f"MPI_Isend(_buffer, {count_str}, {mpi_dtype_str}, int(_dest), int(_tag), {comm}, _request);"
        code += free

        tasklet = comm_tasklet(node, parent_state, code)
        conn = tasklet.in_connectors
        conn = {c: (dtypes.int32 if c == "_dest" else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        return mpi_request_connector(tasklet)


@dace.library.expansion
class ExpandIsendNCCL(NcclExpansion):
    """Enqueues an ``ncclSend`` in the open NCCL group, which the next ``Wait`` (or blocking operation)
    launches. NCCL operations have no requests: the ``_request`` output is optional (see
    :func:`~dace.libraries.comm.nodes.node.nccl_request`)."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, _, ddt), _, _, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, buffer)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = "__state->nccl->Track(__dace_current_stream);\n"
        body += nccl_point_to_point("ncclSend", "_buffer", "int(_dest)", count_str, buffer.dtype.base_type, ddt)
        tasklet = nccl_tasklet(node, parent_state, comm, body, blocking=False)
        request = nccl_request(node, parent_sdfg, parent_state, tasklet)
        tasklet.code = dace.properties.CodeBlock(tasklet.code.as_string + request, dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Isend(CommNode):
    """
    Starts sending a buffer (``_buffer``) to a process (``_dest``) with a tag (``_tag``), and writes a request
    (``_request``) that a ``Wait`` completes.

    Other output connectors only order the send in the dataflow, like the ordering connectors of ``Wait``: connect
    the sent buffer through one (e.g., ``_out_buffer``) and into the ``Wait`` to order the send before its wait
    without a request. NCCL (and RCCL), which have no requests, prefer this for buffers in GPU memory: the order is
    then one of GPU streams, while writing a request (or a token) to host memory makes the host wait for the stream.
    """

    # Global properties
    implementations = {
        "MPI": ExpandIsendMPI,
        "NCCL": ExpandIsendNCCL,
    }
    default_implementation = None

    nosync = dace.properties.Property(
        dtype=bool, default=False, category="Scheduling", desc="Do not sync if memory is on GPU"
    )

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_buffer", "_dest", "_tag"}, outputs={"_request"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: buffer, count, mpi_dtype, req of the input data
        """

        buffer, dest, tag, req = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_request":
                req = sdfg.arrays[e.data.data]

        validate_integer_descriptor(dest, "Destination")
        validate_integer_descriptor(tag, "Tag")

        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == "_buffer":
                count_str = count_expression(data.subset)
                # compute buffer offset
                minelem = data.subset.min_element()
                dims_data = sdfg.arrays[data.data].strides
                buffer_offsets = []
                for idx, m in enumerate(minelem):
                    buffer_offsets += [(str(m) + "*" + str(dims_data[idx]))]
                buffer_offset = "+".join(buffer_offsets)

                # create a ddt which describes the buffer layout IFF the sent data is not contiguous
                ddt = None
                if dace.libraries.comm.utils.is_access_contiguous(data, sdfg.arrays[data.data]):
                    pass
                else:
                    ddt = dace.libraries.comm.utils.create_vector_ddt(data, sdfg.arrays[data.data])
        return (buffer, count_str, buffer_offset, ddt), dest, tag, req
