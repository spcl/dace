# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    NcclExpansion,
    expanded_input_connectors,
    mpi_communicator,
    nccl_communicator,
    nccl_point_to_point,
    nccl_request,
    nccl_tasklet,
    validate_integer_descriptor,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandIrecvMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), src, tag = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        comm = mpi_communicator(node, parent_state)

        code = ""
        if ddt is not None:
            code = f"""static MPI_Datatype newtype;
                        static int init=1;
                        if (init) {{
                           MPI_Type_vector({ddt["count"]}, {ddt["blocklen"]}, {ddt["stride"]}, {ddt["oldtype"]}, &newtype);
                           MPI_Type_commit(&newtype);
                           init = 0;
                        }}
                            """
            mpi_dtype_str = "newtype"
            count_str = "1"
        buffer_offset = 0  # this is here because the frontend already changes the pointer
        code += f"MPI_Irecv(_buffer, {count_str}, {mpi_dtype_str}, int(_src), int(_tag), {comm}, _request);"
        if ddt is not None:
            code += """// MPI_Type_free(&newtype);
            """
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.expansion
class ExpandIrecvNCCL(NcclExpansion):
    """Enqueues an ``ncclRecv`` in the open NCCL group, which the next ``Wait`` (or blocking operation)
    launches. NCCL operations have no requests: the ``_request`` output is optional (see
    :func:`~dace.libraries.comm.nodes.node.nccl_request`)."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, _, ddt), _, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, buffer)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = "__state->nccl->Track(__dace_current_stream);\n"
        body += nccl_point_to_point("ncclRecv", "_buffer", "int(_src)", count_str, buffer.dtype.base_type, ddt)
        tasklet = nccl_tasklet(node, parent_state, comm, body, blocking=False)
        request = nccl_request(node, parent_sdfg, parent_state, tasklet)
        tasklet.code = dace.properties.CodeBlock(tasklet.code.as_string + request, dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Irecv(CommNode):
    # Global properties
    implementations = {
        "MPI": ExpandIrecvMPI,
        "NCCL": ExpandIrecvNCCL,
    }
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_src", "_tag"}, outputs={"_buffer", "_request"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, src, tag) of the three data descriptors in the
                 parent SDFG.
        """

        buffer, src, tag = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_src":
                src = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]

        validate_integer_descriptor(src, "Source")
        validate_integer_descriptor(tag, "Tag")

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == "_buffer":
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)
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
        return (buffer, count_str, buffer_offset, ddt), src, tag
