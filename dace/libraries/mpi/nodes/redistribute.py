# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library, properties, subsets, symbolic
from dace.data import _prod
from dace.libraries.mpi import utils
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.codegen.targets import cpp
from dace import subsets


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
        inp_repl = ""
        for i, s in enumerate(inp_symbols):
            inp_repl += f"int {s} = __state->{node.redistr}_self_src[__idx * {len(inp_buffer.shape)} + {i}];\n"
        out_repl = ""
        for i, s in enumerate(out_symbols):
            out_repl += f"int {s} = __state->{node.redistr}_self_dst[__idx * {len(out_buffer.shape)} + {i}];\n"
        copy_args = ", ".join([
            f"__state->{node.redistr}_self_size[__idx * {len(inp_buffer.shape)} + {i}], {istride}, {ostride}"
            for i, (istride, ostride) in enumerate(zip(inp_buffer.strides, out_buffer.strides))
        ])

        code = f"""
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Request* req = new MPI_Request[__state->{node._redistr}_sends];
            MPI_Status* status = new MPI_Status[__state->{node._redistr}_sends];
            MPI_Status recv_status;
            if (__state->{array_a.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_sends; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I send to %d\\n", myrank, __state->{node._redistr}_dst_ranks[__idx]);
                    // fflush(stdout);
                    MPI_Isend(_inp_buffer, 1, __state->{node._redistr}_send_types[__idx], __state->{node._redistr}_dst_ranks[__idx], 0, MPI_COMM_WORLD, &req[__idx]);
                }}
            }}
            if (__state->{array_b.pgrid}_valid) {{
                for (auto __idx = 0; __idx < __state->{node._redistr}_self_copies; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I self-copy\\n", myrank);
                    // fflush(stdout);
                    {inp_repl}
                    {out_repl}
                    dace::CopyNDDynamic<{inp_buffer.dtype.ctype}, 1, false, {len(inp_buffer.shape)}>::Dynamic::Copy(
                        _inp_buffer + {inp_offset}, _out_buffer + {out_offset}, {copy_args}
                    );
                }}
                for (auto __idx = 0; __idx < __state->{node._redistr}_recvs; ++__idx) {{
                    // printf("({redistr.array_a} -> {redistr.array_b}) I am rank %d and I receive from %d\\n", myrank, __state->{node._redistr}_src_ranks[__idx]);
                    // fflush(stdout);
                    MPI_Recv(_out_buffer, 1, __state->{node._redistr}_recv_types[__idx], __state->{node._redistr}_src_ranks[__idx], 0, MPI_COMM_WORLD, &recv_status);
                }}
            }}
            if (__state->{array_a.pgrid}_valid) {{
                MPI_Waitall(__state->{node._redistr}_sends, req, status);
                delete[] req;
                delete[] status;
            }}
            // printf("I am rank %d and I finished the redistribution {redistr.array_a} -> {redistr.array_b}\\n", myrank);
            // fflush(stdout);
            
        """

        tasklet = nodes.Tasklet(node.name, node.in_connectors, node.out_connectors, code, language=dtypes.Language.CPP)
        return tasklet


@library.node
class Redistribute(nodes.LibraryNode):

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
