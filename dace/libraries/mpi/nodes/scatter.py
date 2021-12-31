# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
import functools
from .. import environments


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@dace.library.expansion
class ExpandScatterMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), (outbuffer,
                                   out_count_str), root = node.validate(
                                       parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inbuffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)
        if root.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Scatter root must be an integer!")

        code = f"""
            int _commsize;
            MPI_Comm_size(MPI_COMM_WORLD, &_commsize);
            MPI_Scatter(_inbuffer, ({in_count_str})/_commsize, {in_mpi_dtype_str},
                        _outbuffer, {out_count_str}, {out_mpi_dtype_str},
                        _root, MPI_COMM_WORLD);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandBlockScatterMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inp_buffer, block_sizes, process_grid, color_dims,
         out_buffer) = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inp_buffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            out_buffer.dtype.base_type)

        if inp_buffer.dtype.veclen > 1:
            raise NotImplementedError
        # if root.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("Scatter root must be an integer!")

        num_comp_dims = process_grid.shape[0]
        num_data_dims = block_sizes.shape[0]
        num_color_dims = color_dims.shape[0]

        code = f"""
            MPI_Comm __cart_comm;
            int periods[{num_comp_dims}] = {{0}};
            MPI_Cart_create(MPI_COMM_WORLD, {num_comp_dims}, _process_grid, periods, 0, &__cart_comm);

            if (__cart_comm != MPI_COMM_NULL) {{
                int __cart_rank, __cart_size;
                MPI_Comm_rank(__cart_comm, &__cart_rank);
                MPI_Comm_size(__cart_comm, &__cart_size);

                int process_id[{num_comp_dims}];
                MPI_Cart_coords(__cart_comm, __cart_rank, {num_comp_dims}, process_id);

                MPI_Comm __scatter_comm, __bcast_comm;
                int remain[{num_comp_dims}];
                for (auto i = 0; i < {num_color_dims}; ++i) {{
                    remain[i] = (_color_dims[i] + 1) % 2;
                }}
                MPI_Cart_sub(__cart_comm, _color_dims, &__scatter_comm);
                MPI_Cart_sub(__cart_comm, remain, &__bcast_comm);

                int __scatter_rank, num_blocks;
                MPI_Comm_rank(__scatter_comm, &__scatter_rank);
                MPI_Comm_size(__scatter_comm, &num_blocks);

                MPI_Group __cart_group, __scatter_group;
                MPI_Comm_group(__cart_comm, &__cart_group);
                MPI_Comm_group(__scatter_comm, &__scatter_group);
                int ranks1[1] = {{0}};
                int ranks2[1];
                MPI_Group_translate_ranks(__cart_group, 1, ranks1, __scatter_group, ranks2);

                if (ranks2[0] != MPI_PROC_NULL && ranks2[0] != MPI_UNDEFINED) {{
                    int basic_stride = _block_sizes[{num_data_dims} - 1];
                    int process_strides[{num_data_dims}];
                    int block_strides[{num_data_dims}];
                    int data_strides[{num_data_dims}];
                    process_strides[{num_data_dims} - 1] = 1;
                    block_strides[{num_data_dims} - 1] = _block_sizes[{num_data_dims} - 1];
                    data_strides[{num_data_dims} - 1] = 1;
                    for (auto i = {num_data_dims} - 2; i >= 0; --i) {{
                        block_strides[i] = block_strides[i+1] * _block_sizes[i];
                        process_strides[i] = process_strides[i+1] * _process_grid[_cd_equiv[i+1]];
                        data_strides[i] = block_strides[i] * process_strides[i] / basic_stride;
                    }}

                    MPI_Datatype type, rsized_type;
                    int sizes[{num_data_dims}] = {{{','.join(symstr(s) for s in inp_buffer.shape)}}};
                    int origin[{num_data_dims}] = {{{','.join(['0'] * num_data_dims)}}};
                    MPI_Type_create_subarray({num_data_dims}, sizes, _block_sizes, origin, MPI_ORDER_C, {in_mpi_dtype_str}, &type);
                    MPI_Type_create_resized(type, 0, basic_stride*sizeof({inp_buffer.dtype.ctype}), &rsized_type);
                    MPI_Type_commit(&rsized_type);

                    int* counts = new int[num_blocks];
                    int* displs = new int[num_blocks];
                    int block_id[{num_data_dims}] = {{0}};
                    int displ = 0;
                    for (auto i = 0; i < num_blocks; ++i) {{
                        counts[i] = 1;
                        displs[i] = displ;
                        int idx = {num_data_dims} - 1;
                        while (block_id[idx] + 1 >= _process_grid[_cd_equiv[idx]]) {{
                            block_id[idx] = 0;
                            displ -= data_strides[idx] * (_process_grid[_cd_equiv[idx]] - 1);
                            idx--;
                        }}
                        block_id[idx] += 1;
                        displ += data_strides[idx];
                    }}

                    if (__scatter_rank == 0) {{
                        printf("Number of blocks is %d\\n", num_blocks);
                        printf("Process grid");
                        for (auto i = 0; i < {num_data_dims}; ++i) printf(" %d", _process_grid[_cd_equiv[i]]);
                        printf("\\n");
                        printf("Block strides");
                        for (auto i = 0; i < {num_data_dims}; ++i) printf(" %d", block_strides[i]);
                        printf("\\n");
                        printf("Process strides");
                        for (auto i = 0; i < {num_data_dims}; ++i) printf(" %d", process_strides[i]);
                        printf("\\n");
                        printf("Data strides");
                        for (auto i = 0; i < {num_data_dims}; ++i) printf(" %d", data_strides[i]);
                        printf("\\n");
                        printf("Counts are");
                        for (auto i = 0; i < num_blocks; ++i) printf(" %d", counts[i]);
                        printf("\\n");
                        printf("Displacements are");
                        for (auto i = 0; i < num_blocks; ++i) printf(" %d", displs[i]);
                        printf("\\n");
                    }}
                    MPI_Scatterv(_inp_buffer, counts, displs, rsized_type, _out_buffer, {symstr(_prod(out_buffer.shape))}, {out_mpi_dtype_str}, 0, __scatter_comm);
                    
                    delete[] counts;
                    delete[] displs;
                    MPI_Type_free(&rsized_type);
                }}

                MPI_Bcast(_out_buffer, {symstr(_prod(out_buffer.shape))}, {out_mpi_dtype_str}, 0, __bcast_comm);
            
                MPI_Comm_free(&__scatter_comm);
                MPI_Comm_free(&__bcast_comm);
            }}

            MPI_Comm_free(&__cart_comm);
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Scatter(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandScatterMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_root"},
                         outputs={"_outbuffer"},
                         **kwargs)

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

        if root.dtype.base_type != dace.dtypes.int32:
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


@dace.library.expansion
class ExpandBlockScatterMPI2(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        inp_buffer, out_buffer = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inp_buffer.dtype.base_type)
        out_mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            out_buffer.dtype.base_type)

        if inp_buffer.dtype.veclen > 1:
            raise NotImplementedError
        # if root.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("Scatter root must be an integer!")

        code = f"""
            if (__state->{node._scatter_grid}_valid) {{
                MPI_Scatterv(_inp_buffer, __state->{node._dtype}_counts, __state->{node._dtype}_displs, __state->{node._dtype}, _out_buffer, {symstr(_prod(out_buffer.shape))}, {out_mpi_dtype_str}, 0, __state->{node._scatter_grid}_comm);
            }}
            MPI_Bcast(_out_buffer, {symstr(_prod(out_buffer.shape))}, {out_mpi_dtype_str}, 0, __state->{node._bcast_grid}_comm);
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet



@dace.library.node
class BlockScatter(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandBlockScatterMPI2,
    }
    default_implementation = "MPI"

    def __init__(self, name, dtype, scatter_grid, bcast_grid, *args, **kwargs):
        super().__init__(name,
                         *args,
                        #  inputs={"_inp_buffer", "_block_sizes", "_cd_equiv",
                        #          "_process_grid", "_color_dims"},
                        inputs={"_inp_buffer"},
                         outputs={"_out_buffer"},
                         **kwargs)
        self._dtype = dtype
        self._scatter_grid = scatter_grid
        self._bcast_grid = bcast_grid

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

        # inp_buffer, block_sizes, process_grid, color_dims, out_buffer = (
        #     None, None, None, None, None)
        # for e in state.out_edges(self):
        #     if e.src_conn == "_out_buffer":
        #         out_buffer = sdfg.arrays[e.data.data]
        #         # out_size = e.data.subset.size_exact()
        # for e in state.in_edges(self):
        #     if e.dst_conn == "_inp_buffer":
        #         inp_buffer = sdfg.arrays[e.data.data]
        #         # inp_size = e.data.subset.size_exact()
        #     if e.dst_conn == "_block_sizes":
        #         block_sizes = sdfg.arrays[e.data.data]
        #     if e.dst_conn == "_process_grid":
        #         process_grid = sdfg.arrays[e.data.data]
        #     if e.dst_conn == "_color_dims":
        #         color_dims = sdfg.arrays[e.data.data]
        #     # if e.dst_conn == "_root":
        #     #     root = sdfg.arrays[e.data.data]

        # # if root.dtype.base_type != dace.dtypes.int32:
        # #     raise (ValueError("Scatter root must be an integer!"))

        # # in_count_str = "XXX"
        # # out_count_str = "XXX"
        # # for _, src_conn, _, _, data in state.out_edges(self):
        # #     if src_conn == '_outbuffer':
        # #         dims = [symstr(e) for e in data.subset.size_exact()]
        # #         out_count_str = "*".join(dims)
        # # for _, _, _, dst_conn, data in state.in_edges(self):
        # #     if dst_conn == '_inbuffer':
        # #         dims = [symstr(e) for e in data.subset.size_exact()]
        # #         in_count_str = "*".join(dims)

        # return inp_buffer, block_sizes, process_grid, color_dims, out_buffer
