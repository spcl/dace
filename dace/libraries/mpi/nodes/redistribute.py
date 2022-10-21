# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library, properties, subsets, symbolic
from dace.data import _prod
from dace.libraries.mpi import utils
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.codegen.targets import cpp
from dace import subsets
from dace.libraries.mpi.nodes.node import MPINode


@library.expansion
class ExpandRedistribute(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_buffer, out_buffer = node.validate(parent_sdfg, parent_state)
        redistr = parent_sdfg.rdistrarrays[node.redistr]
        array_a = parent_sdfg.subarrays[redistr.array_a]
        array_b = parent_sdfg.subarrays[redistr.array_b]

        inp_symbols = [symbolic.symbol(f"__inp_s{i}") for i in range(len(inp_buffer.shape))]
        out_symbols = [symbolic.symbol(f"__out_s{i}") for i in range(len(out_buffer.shape))]
        inp_subset = subsets.Indices(inp_symbols)
        out_subset = subsets.Indices(out_symbols)
        inp_offset = cpp.cpp_offset_expr(inp_buffer, inp_subset)
        out_offset = cpp.cpp_offset_expr(out_buffer, out_subset)
        print(inp_offset)
        print(out_offset)

        ###### Send
        send_repl = ""
        for i, s in enumerate(inp_symbols):
            send_repl += f"int {s} = __state->{node.redistr}_fix_send_src[__idx * {len(inp_buffer.shape)} + {i}];\n"
        send_strides = f"int __send_strides[{len(inp_buffer.shape)}];\n"
        send_strides += f"__send_strides[{len(inp_buffer.shape)} - 1] = 1;\n"
        for i in reversed(range(len(inp_buffer.shape) - 1)):
            send_strides += f"__send_strides[{i}] = __send_strides[{i+1}] * __state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i+1}];\n"
        if len(inp_symbols) > 1:
            send_args = ", ".join([
                f"__state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i}], {istride}, __send_strides[{i}]"
                for i, istride in enumerate(inp_buffer.strides) if i > 0
            ])
            send_code = f"""
                int __m0_size = __state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)}];
                #pragma omp parallel for
                for (auto __m0_idx = 0; __m0_idx < __m0_size; ++__m0_idx) {{
                    dace::CopyNDDynamic<{inp_buffer.dtype.ctype}, 1, false, {len(inp_buffer.shape) - 1}>::Dynamic::Copy(
                        _inp_buffer + {inp_offset} + __m0_idx * {inp_buffer.strides[0]},
                        __state->{node.redistr}_send_buffers[__idx] + __m0_idx * __send_strides[0],
                        {send_args}
                    );
                }}
            """
        else:
            send_args = ", ".join([
                f"__state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i}], {istride}, __send_strides[{i}]"
                for i, istride in enumerate(inp_buffer.strides)
            ])
            send_code = f"""
                dace::CopyNDDynamic<{inp_buffer.dtype.ctype}, 1, false, {len(inp_buffer.shape)}>::Dynamic::Copy(
                    _inp_buffer + {inp_offset}, __state->{node.redistr}_send_buffers[__idx], {send_args}
                );
            """

        if inp_buffer.storage == dtypes.StorageType.GPU_Global:
            send_code = f"""
                int __src_offset_0 = {inp_offset};
                int __dst_offset_0 = 0;
            """
            if len(inp_buffer.shape) > 2:
                for i in range(len(inp_buffer.shape) - 2):
                    send_code += f"""
                        int __m{i}_size = __state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i}];
                        for (auto __m{i}_idx = 0; __m{i}_idx < __m{i}_size; ++__m{i}_idx) {{
                            int __src_offset_{i+1} = __src_offset_{i} + __m{i}_idx * {inp_buffer.strides[i]}; 
                            int __dst_offset_{i+1} = __dst_offset_{i} + __m{i}_idx * __send_strides[{i}]; 
                    """
            i = len(inp_buffer.shape) - 2
            send_code += f"""
                int __m{i}_size = __state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i}];
                int __m{i+1}_size = __state->{node.redistr}_fix_send_size[__idx * {len(inp_buffer.shape)} + {i+1}];
                cudaMemcpy2DAsync(
                                  (void*)(__state->{node.redistr}_send_buffers[__idx] + __dst_offset_{i}),
                                  //(void*)(__state->__rdistrarray_0_send_buffers[__idx] + __dst_offset_{i}),
                                  __send_strides[{i}] * sizeof({inp_buffer.dtype.ctype}),
                                  (void*)(_inp_buffer + __src_offset_{i}),
                                  {inp_buffer.strides[i]} * sizeof({inp_buffer.dtype.ctype}),
                                  __m{i+1}_size * sizeof({inp_buffer.dtype.ctype}),
                                  __m{i}_size,
                                  cudaMemcpyDeviceToHost,
                                  //cudaMemcpyDeviceToDevice,
                                  __dace_current_stream);
            """
            if len(inp_buffer.shape) > 2:
                for i in range(len(inp_buffer.shape) - 2):
                    send_code += "}\n"
            send_code += f"cudaStreamSynchronize(__dace_current_stream);"
            # send_code += "printf(\"Send done!\\n\"); fflush(stdout);"

        ####### Recv
        recv_repl = ""
        for i, s in enumerate(out_symbols):
            recv_repl += f"int {s} = __state->{node.redistr}_fix_recv_dst[__idx * {len(out_buffer.shape)} + {i}];\n"
        recv_strides = f"int __recv_strides[{len(out_buffer.shape)}];\n"
        recv_len = f"int __recv_len = __state->{node.redistr}_fix_recv_size[(__idx + 1) * {len(out_buffer.shape)} - 1];\n"
        recv_strides += f"__recv_strides[{len(out_buffer.shape)} - 1] = 1;\n"
        for i in reversed(range(len(out_buffer.shape) - 1)):
            recv_strides += f"__recv_strides[{i}] = __recv_strides[{i+1}] * __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i+1}];\n"
            recv_len += f"__recv_len *= __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i}];\n"
        if len(out_symbols) > 1:
            recv_args = ", ".join([
                f"__state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i}], __recv_strides[{i}], {ostride}"
                for i, ostride in enumerate(out_buffer.strides) if i > 0
            ])
            recv_code = f"""
                int __m0_size = __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)}];
                #pragma omp parallel for
                for (auto __m0_idx = 0; __m0_idx < __m0_size; ++__m0_idx) {{
                    dace::CopyNDDynamic<{out_buffer.dtype.ctype}, 1, false, {len(out_buffer.shape) - 1}>::Dynamic::Copy(
                        __state->{node.redistr}_recv_buffers[__idx] + __m0_idx * __recv_strides[0],
                        _out_buffer + {out_offset} + __m0_idx * {out_buffer.strides[0]},
                        {recv_args}
                    );
                }}
            """
        else:
            recv_args = ", ".join([
                f"__state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i}], __recv_strides[{i}], {ostride}"
                for i, ostride in enumerate(out_buffer.strides)
            ])
            recv_code = f"""
                dace::CopyNDDynamic<{out_buffer.dtype.ctype}, 1, false, {len(out_buffer.shape)}>::Dynamic::Copy(
                    __state->{node.redistr}_recv_buffers[__idx], _out_buffer + {out_offset}, {recv_args}
                );
            """

        if out_buffer.storage == dtypes.StorageType.GPU_Global:
            recv_code = f"""
                int __dst_offset_0 = {out_offset};
                int __src_offset_0 = 0;
            """
            if len(out_buffer.shape) > 2:
                for i in range(len(out_buffer.shape) - 2):
                    recv_code += f"""
                        int __m{i}_size = __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i}];
                        for (auto __m{i}_idx = 0; __m{i}_idx < __m{i}_size; ++__m{i}_idx) {{
                            int __dst_offset_{i+1} = __dst_offset_{i} + __m{i}_idx * {out_buffer.strides[i]}; 
                            int __src_offset_{i+1} = __src_offset_{i} + __m{i}_idx * __recv_strides[{i}]; 
                    """
            i = len(out_buffer.shape) - 2
            recv_code += f"""
                int __m{i}_size = __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i}];
                int __m{i+1}_size = __state->{node.redistr}_fix_recv_size[__idx * {len(out_buffer.shape)} + {i+1}];
                cudaMemcpy2DAsync((void*)(_out_buffer + __dst_offset_{i}),
                                  {out_buffer.strides[i]} * sizeof({out_buffer.dtype.ctype}),
                                  (void*)(__state->{node.redistr}_recv_buffers[__idx] + __src_offset_{i}),
                                  //(void*)(__state->__rdistrarray_0_recv_buffers[__idx] + __src_offset_{i}),
                                  __recv_strides[{i}] * sizeof({out_buffer.dtype.ctype}),
                                  __m{i+1}_size * sizeof({out_buffer.dtype.ctype}),
                                  __m{i}_size,
                                  cudaMemcpyHostToDevice,
                                  //cudaMemcpyDeviceToDevice,
                                  __dace_current_stream);
            """
            if len(out_buffer.shape) > 2:
                for i in range(len(out_buffer.shape) - 2):
                    recv_code += "}\n"
            # recv_code += f"cudaStreamSynchronize(__dace_current_stream);"
            # recv_code += "printf(\"Recv done!\\n\"); fflush(stdout);"

        ####### Copy
        inp_repl = ""
        for i, s in enumerate(inp_symbols):
            inp_repl += f"int {s} = __state->{node.redistr}_self_src[__idx * {len(inp_buffer.shape)} + {i}];\n"
        out_repl = ""
        for i, s in enumerate(out_symbols):
            out_repl += f"int {s} = __state->{node.redistr}_self_dst[__idx * {len(out_buffer.shape)} + {i}];\n"
        if len(inp_symbols) > 1:
            copy_args = ", ".join([
                f"__state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}], {istride}, {ostride}"
                for i, (istride, ostride) in enumerate(zip(inp_buffer.strides, out_buffer.strides)) if i > 0
            ])
            copy_code = f"""
                int __m0_size = __state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)}];
                #pragma omp parallel for
                for (auto __m0_idx = 0; __m0_idx < __m0_size; ++__m0_idx) {{
                    dace::CopyNDDynamic<{inp_buffer.dtype.ctype}, 1, false, {len(inp_buffer.shape) - 1}>::Dynamic::Copy(
                        _inp_buffer + {inp_offset} + __m0_idx * {inp_buffer.strides[0]},
                        _out_buffer + {out_offset} + __m0_idx * {out_buffer.strides[0]},
                        {copy_args}
                    );
                }}
            """
        else:
            copy_args = ", ".join([
                f"__state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}], {istride}, {ostride}"
                for i, (istride, ostride) in enumerate(zip(inp_buffer.strides, out_buffer.strides))
            ])
            copy_code = f"""
                dace::CopyNDDynamic<{inp_buffer.dtype.ctype}, 1, false, {len(inp_buffer.shape)}>::Dynamic::Copy(
                    _inp_buffer + {inp_offset}, _out_buffer + {out_offset}, {copy_args}
                );
            """
        copy_size = "* ".join([f"__state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}]"
                                for i in range(len(inp_buffer.shape))])

        if inp_buffer.storage == dtypes.StorageType.GPU_Global:
            copy_code = f"""
                int __src_offset_0 = {inp_offset};
                int __dst_offset_0 = {out_offset};
            """
            if len(inp_buffer.shape) > 2:
                for i in range(len(inp_buffer.shape) - 2):
                    copy_code += f"""
                        int __m{i}_size = __state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}];
                        for (auto __m{i}_idx = 0; __m{i}_idx < __m{i}_size; ++__m{i}_idx) {{
                            int __src_offset_{i+1} = __src_offset_{i} + __m{i}_idx * {inp_buffer.strides[i]}; 
                            int __dst_offset_{i+1} = __dst_offset_{i} + __m{i}_idx * {out_buffer.strides[i]}; 
                    """
            i = len(inp_buffer.shape) - 2
            copy_code += f"""
                int __m{i}_size = __state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}];
                int __m{i+1}_size = __state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i+1}];
                cudaMemcpy2DAsync((void*)(_out_buffer + __dst_offset_{i}),
                                  {out_buffer.strides[i]} * sizeof({out_buffer.dtype.ctype}),
                                  (void*)(_inp_buffer + __src_offset_{i}),
                                  {inp_buffer.strides[i]} * sizeof({inp_buffer.dtype.ctype}),
                                  __m{i+1}_size * sizeof({inp_buffer.dtype.ctype}),
                                  __m{i}_size,
                                  cudaMemcpyDeviceToDevice,
                                  __dace_current_stream);
            """
            if len(inp_buffer.shape) > 2:
                for i in range(len(inp_buffer.shape) - 2):
                    copy_code += "}\n"
            # copy_code += f"cudaStreamSynchronize(__dace_current_stream);"

        code = f"""
            // int myrank;
            // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            // MPI_Status recv_status;
            if (__state->{array_a.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_sends; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I send to %d\\n", myrank, __state->{node._redistr}_dst_ranks[__idx]);
                    // fflush(stdout);
                    __state->{node._redistr}_total_send_size += __state->{node._redistr}_send_sizes[__idx];
                    // MPI_Isend(_inp_buffer, 1, __state->{node._redistr}_send_types[__idx], __state->{node._redistr}_dst_ranks[__idx], 0, MPI_COMM_WORLD, &__state->{node._redistr}_send_req[__idx]);
                    {send_repl}
                    {send_strides}
                    {send_code}
                    MPI_Isend(__state->{node.redistr}_send_buffers[__idx], __state->{node._redistr}_send_sizes[__idx], {utils.MPI_DDT(array_a.dtype.base_type)}, __state->{node._redistr}_dst_ranks[__idx], 0, MPI_COMM_WORLD, &__state->{node._redistr}_send_req[__idx]);
                }}
            }}
            if (__state->{array_b.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_recvs; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I receive from %d\\n", myrank, __state->{node._redistr}_src_ranks[__idx]);
                    // fflush(stdout);
                    // MPI_Recv(_out_buffer, 1, __state->{node._redistr}_recv_types[__idx], __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &recv_status);
                    // MPI_Irecv(_out_buffer, 1, __state->{node._redistr}_recv_types[__idx], __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &__state->{node._redistr}_recv_req[__idx]);
                    {recv_len}
                    MPI_Irecv(__state->{node.redistr}_recv_buffers[__idx], __recv_len, {utils.MPI_DDT(array_b.dtype.base_type)}, __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &__state->{node._redistr}_recv_req[__idx]);
                }}
                for (auto __idx = 0; __idx < __state->{node._redistr}_self_copies; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I self-copy\\n", myrank);
                    // fflush(stdout);
                    {inp_repl}
                    {out_repl}
                    __state->{node._redistr}_total_copy_size += {copy_size};
                    {copy_code}
                }}
                MPI_Waitall(__state->{node._redistr}_recvs, __state->{node._redistr}_recv_req, __state->{node._redistr}_recv_status);
                for (auto __idx = 0; __idx < __state->{node._redistr}_recvs; ++__idx) {{
                    {recv_repl}
                    {recv_strides}
                    {recv_code}
                }}

            }}
            if (__state->{array_a.pgrid}_valid) {{
                MPI_Waitall(__state->{node._redistr}_sends, __state->{node._redistr}_send_req, __state->{node._redistr}_send_status);
            }}
            {"cudaStreamSynchronize(__dace_current_stream);" if inp_buffer.storage == dtypes.StorageType.GPU_Global else ""}
            // printf("I am rank %d and I finished the redistribution {redistr.array_a} -> {redistr.array_b}\\n", myrank);
            // fflush(stdout);
            
        """

        tasklet = nodes.Tasklet(node.name, node.in_connectors, node.out_connectors, code, language=dtypes.Language.CPP)
        return tasklet


@library.node
class Redistribute(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandRedistribute,
    }
    default_implementation = "MPI"

    redistr = properties.Property(dtype=str, default='tmp')

    def __init__(self, name, redistr='tmp', *args, **kwargs):
        super().__init__(name, *args, inputs={"_inp_buffer"}, outputs={"_out_buffer"}, **kwargs)
        self.redistr = redistr

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
