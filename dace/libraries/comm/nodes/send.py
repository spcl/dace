# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    NcclExpansion,
    buffer_pointer,
    count_expression,
    expanded_input_connectors,
    mpi_communicator,
    mpi_vector_datatype,
    nccl_communicator,
    nccl_point_to_point,
    nccl_tasklet,
    validate_integer_descriptor,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandSendMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), dest, tag = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.comm.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise (NotImplementedError)
        code, vector, vector_count, free = mpi_vector_datatype(ddt)
        mpi_dtype_str = vector or mpi_dtype_str
        count_str = vector_count or count_str
        code += f"""
                MPI_Send({buffer_pointer(buffer, "_buffer")}, {count_str}, {mpi_dtype_str}, int(_dest), int(_tag), {mpi_communicator(node, parent_state)});
                """
        code += free

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            expanded_input_connectors(node, parent_state),
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )
        return tasklet


@dace.library.expansion
class ExpandSendNCCL(NcclExpansion):
    """Sends with ``ncclSend``. NCCL has no tags: sends to a peer match its receives in order."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, _, ddt), _, _ = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, buffer)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = nccl_point_to_point("ncclSend", "_buffer", "int(_dest)", count_str, buffer.dtype.base_type, ddt)
        return nccl_tasklet(node, parent_state, comm, body)


@dace.library.node
class Send(CommNode):
    # Global properties
    implementations = {
        "MPI": ExpandSendMPI,
        "NCCL": ExpandSendNCCL,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None, category="Semantics")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_buffer", "_dest", "_tag"}, outputs={}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: Buffer, count, mpi_dtype of the input data
        """

        # Squeeze input memlets
        # squeezed1 = copy.deepcopy(in_memlets[0].subset)
        # sqdims1 = squeezed1.squeeze()

        buffer, dest, tag = None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_tag":
                tag = sdfg.arrays[e.data.data]

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

        return (buffer, count_str, buffer_offset, ddt), dest, tag
