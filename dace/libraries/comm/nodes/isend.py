# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    expanded_input_connectors,
    mpi_communicator,
    nccl_communicator,
    nccl_point_to_point,
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

        code = ""

        if not node.nosync and buffer.storage == dtypes.StorageType.GPU_Global:
            code += """
            cudaStreamSynchronize(__dace_current_stream);
            """

        if ddt is not None:
            code += f"""static MPI_Datatype newtype;
                        static int init=1;
                        if (init) {{
                           MPI_Type_vector({ddt["count"]}, {ddt["blocklen"]}, {ddt["stride"]}, {ddt["oldtype"]}, &newtype);
                           MPI_Type_commit(&newtype);
                           init=0;
                        }}
                            """
            mpi_dtype_str = "newtype"
            count_str = "1"
        buffer_offset = 0
        code += f"MPI_Isend(&(_buffer[{buffer_offset}]), {count_str}, {mpi_dtype_str}, int(_dest), int(_tag), {comm}, _request);"
        if ddt is not None:
            code += """// MPI_Type_free(&newtype);
            """

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        conn = tasklet.in_connectors
        conn = {c: (dtypes.int32 if c == "_dest" else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.expansion
class ExpandIsendNCCL(ExpandTransformation):
    """Enqueues an ``ncclSend`` in the open NCCL group, which the next ``Wait`` (or blocking operation)
    launches. The request is set to ``MPI_REQUEST_NULL``, as NCCL operations have none."""

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, _, ddt), _, _, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, buffer)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = "__state->nccl->Track(__dace_current_stream);\n"
        body += nccl_point_to_point("ncclSend", "_buffer", "int(_dest)", count_str, buffer.dtype.base_type, ddt)
        body += "*_request = MPI_REQUEST_NULL;\n"
        tasklet = nccl_tasklet(node, parent_state, comm, body, blocking=False)
        tasklet.out_connectors = {
            c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == "_request" else t)
            for c, t in tasklet.out_connectors.items()
        }
        return tasklet


@dace.library.node
class Isend(CommNode):
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
        return (buffer, count_str, buffer_offset, ddt), dest, tag, req
