# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import dtypes
from dace.libraries.mpi.nodes.node import (MPINode, expanded_input_connectors, input_descriptor_name,
                                           validate_integer_descriptor)


@dace.library.expansion
class ExpandIrecvMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), src, tag = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        comm = "MPI_COMM_WORLD"
        grid = input_descriptor_name(node, parent_state, '_grid')
        if grid:
            comm = "_grid"
        if "_comm" in node.in_connectors:
            comm = "_comm"

        code = ""
        if ddt is not None:
            code = f"""static MPI_Datatype newtype;
                        static int init=1;
                        if (init) {{
                           MPI_Type_vector({ddt['count']}, {ddt['blocklen']}, {ddt['stride']}, {ddt['oldtype']}, &newtype);
                           MPI_Type_commit(&newtype);
                           init = 0;
                        }}
                            """
            mpi_dtype_str = "newtype"
            count_str = "1"
        buffer_offset = 0  #this is here because the frontend already changes the pointer
        code += f"MPI_Irecv(_buffer, {count_str}, {mpi_dtype_str}, int(_src), int(_tag), {comm}, _request);"
        if ddt is not None:
            code += f"""// MPI_Type_free(&newtype);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          expanded_input_connectors(node, parent_state),
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class Irecv(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandIrecvMPI,
    }
    default_implementation = "MPI"

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

        validate_integer_descriptor(src, 'Source')
        validate_integer_descriptor(tag, 'Tag')

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_buffer':
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
                if dace.libraries.mpi.utils.is_access_contiguous(data, sdfg.arrays[data.data]):
                    pass
                else:
                    ddt = dace.libraries.mpi.utils.create_vector_ddt(data, sdfg.arrays[data.data])
        return (buffer, count_str, buffer_offset, ddt), src, tag
