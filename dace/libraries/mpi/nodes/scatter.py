# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library, properties
from dace.data import _prod
from dace.libraries.mpi import utils
from dace.sdfg import nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
import functools
from .. import environments


@library.expansion
class ExpandScatterMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), (outbuffer, out_count_str), root = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)
        if root.dtype.base_type != dtypes.int32:
            raise ValueError("Scatter root must be an integer!")

        code = f"""
            int _commsize;
            MPI_Comm_size(MPI_COMM_WORLD, &_commsize);
            MPI_Scatter(_inbuffer, ({in_count_str})/_commsize, {in_mpi_dtype_str},
                        _outbuffer, {out_count_str}, {out_mpi_dtype_str},
                        _root, MPI_COMM_WORLD);
            """
        tasklet = nodes.Tasklet(node.name, node.in_connectors, node.out_connectors, code, language=dtypes.Language.CPP)
        return tasklet


@library.node
class Scatter(nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandScatterMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer", "_root"}, outputs={"_outbuffer"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer, root = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_root":
                root = sdfg.arrays[e.data.data]

        if root.dtype.base_type != dtypes.int32:
            raise (ValueError("Scatter root must be an integer!"))

        in_count_str = "XXX"
        out_count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [symstr(e) for e in data.subset.size_exact()]
                out_count_str = "*".join(dims)
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_inbuffer':
                dims = [symstr(e) for e in data.subset.size_exact()]
                in_count_str = "*".join(dims)

        return (inbuffer, in_count_str), (outbuffer, out_count_str), root


@library.expansion
class ExpandBlockScatterMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_buffer, out_buffer = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = utils.MPI_DDT(out_buffer.dtype.base_type)

        if inp_buffer.dtype.veclen > 1:
            raise NotImplementedError

        code = f"""
            if (__state->{node.scatter_grid}_valid) {{
                MPI_Scatterv(_inp_buffer, __state->{node.subarray_type}_counts, __state->{node.subarray_type}_displs, __state->{node.subarray_type}, _out_buffer, {symstr(_prod(out_buffer.shape))}, {mpi_dtype_str}, 0, __state->{node.scatter_grid}_comm);
            }}
        """
        if node.bcast_grid:
            code += f"MPI_Bcast(_out_buffer, {symstr(_prod(out_buffer.shape))}, {mpi_dtype_str}, 0, __state->{node._bcast_grid}_comm);"

        tasklet = nodes.Tasklet(node.name, node.in_connectors, node.out_connectors, code, language=dtypes.Language.CPP)
        return tasklet


@library.node
class BlockScatter(nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandBlockScatterMPI,
    }
    default_implementation = "MPI"

    subarray_type = properties.Property(dtype=str, default='tmp')
    scatter_grid = properties.Property(dtype=str, default='tmp')
    bcast_grid = properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, subarray_type='tmp', scatter_grid='tmp', bcast_grid=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp_buffer"}, outputs={"_out_buffer"}, **kwargs)
        self.subarray_type = subarray_type
        self.scatter_grid = scatter_grid
        self.bcast_grid = bcast_grid

    def validate(self, sdfg, state):
        """
        :return: A tuple (inbuffer, outbuffer) of the twodata descriptors in the parent SDFG.
        """

        inp_buffer, out_buffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_buffer":
                out_buffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_buffer":
                inp_buffer = sdfg.arrays[e.data.data]

        return inp_buffer, out_buffer
