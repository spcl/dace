# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library
from dace.utils import prod as _prod
from dace.libraries.mpi import utils
from dace.sdfg import nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors, input_descriptor_name


@library.expansion
class ExpandGatherMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), (outbuffer, out_count_str), root = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)
        if root.dtype.base_type != dtypes.int32:
            raise ValueError("Gather root must be an integer!")

        comm = resolve_comm(node, parent_state)

        code = f"""
            int _commsize;
            MPI_Comm_size({comm}, &_commsize);
            MPI_Gather(_inbuffer, {in_count_str}, {in_mpi_dtype_str},
                       _outbuffer, ({out_count_str})/_commsize, {out_mpi_dtype_str},
                       _root, {comm});
            """
        tasklet = nodes.Tasklet(node.name,
                                expanded_input_connectors(node, parent_state),
                                node.out_connectors,
                                code,
                                language=dtypes.Language.CPP)
        return tasklet


@library.node
class Gather(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandGatherMPI,
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
            raise (ValueError("Gather root must be an integer!"))

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
class ExpandBlockGatherMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_buffer, out_buffer = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = utils.MPI_DDT(inp_buffer.dtype.base_type)

        if out_buffer.dtype.veclen > 1:
            raise NotImplementedError

        subarray_type = input_descriptor_name(node, parent_state, '_subarray')
        gather_grid = input_descriptor_name(node, parent_state, '_gather_grid')
        reduce_grid = input_descriptor_name(node, parent_state, '_reduce_grid')
        if subarray_type is None:
            raise ValueError('BlockGather requires an incoming _subarray connector')
        if gather_grid is None:
            raise ValueError('BlockGather requires an incoming _gather_grid connector')

        if reduce_grid:
            code = f"""
                if (__state->{gather_grid}_valid) {{
                    MPI_Reduce(MPI_IN_PLACE, _inp_buffer, {symstr(_prod(inp_buffer.shape))}, {mpi_dtype_str}, MPI_SUM, __state->{reduce_grid}_rank, _reduce_grid);
                    MPI_Gatherv(_inp_buffer, {symstr(_prod(inp_buffer.shape))}, {mpi_dtype_str}, _out_buffer, __state->{subarray_type}_counts, __state->{subarray_type}_displs, __state->{subarray_type}, 0, _gather_grid);
                }} else if (__state->{reduce_grid}_valid) {{
                    MPI_Reduce(_inp_buffer, _inp_buffer, {symstr(_prod(inp_buffer.shape))}, {mpi_dtype_str}, MPI_SUM, 0, _reduce_grid);
                }}
            """
        else:
            code = f"""
                if (__state->{gather_grid}_valid) {{
                    MPI_Gatherv(_inp_buffer, {symstr(_prod(inp_buffer.shape))}, {mpi_dtype_str}, _out_buffer, __state->{subarray_type}_counts, __state->{subarray_type}_displs, __state->{subarray_type}, 0, _gather_grid);
                }}
            """

        tasklet = nodes.Tasklet(node.name,
                                expanded_input_connectors(node, parent_state),
                                node.out_connectors,
                                code,
                                language=dtypes.Language.CPP)
        return tasklet


@library.node
class BlockGather(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandBlockGatherMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp_buffer"}, outputs={"_out_buffer"}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (inbuffer, outbuffer, root) of the three data
                 descriptors in the parent SDFG.
        """

        inp_buffer, out_buffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_buffer":
                out_buffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inp_buffer":
                inp_buffer = sdfg.arrays[e.data.data]

        return inp_buffer, out_buffer
