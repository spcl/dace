# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandAllreduceMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, count_str), outbuffer, in_place = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = "MPI_COMM_WORLD"
        if node.grid:
            comm = f"__state->{node.grid}_comm"
            code = f"if (__state->{node.grid}_size > 1) {{"
        else:
            code = ""

        buffer = '_inbuffer'
        if in_place:
            buffer = 'MPI_IN_PLACE'

        code += f"""
            MPI_Allreduce({buffer}, _outbuffer, {count_str}, {mpi_dtype_str}, {node.op}, {comm});
            """

        if node.grid:
            code += "}"
        
        # comm = "MPI_COMM_WORLD"
        # if node.grid:
        #     comm = f"__state->{node.grid}_comm"
        #     code = f"if (__state->{node.grid}_size > 1) {{"
        # else:
        #     code = ""
        
        # if in_place:
        #     if comm == "MPI_COMM_WORLD":
        #         code += """
        #             int __world_rank;
        #             MPI_Comm_rank(&__world_rank, MPI_COMM_WORLD);
        #             if (__world_rank == 0) {{
        #         """
        #     else:
        #         code += f"""
        #             if (__state->{node.grid}_rank == 0) {{
        #         """
        #     code += f"""
        #             MPI_Reduce(MPI_IN_PLACE, _outbuffer, {count_str}, {mpi_dtype_str}, {node.op}, 0, {comm});
        #             MPI_Bcast(_outbuffer, {count_str}, {mpi_dtype_str}, 0, {comm});
        #         }} else {{            
        #     """
        # code += f"""
        #     MPI_Reduce(_inbuffer, _outbuffer, {count_str}, {mpi_dtype_str}, {node.op}, 0, {comm});
        #     MPI_Bcast(_outbuffer, {count_str}, {mpi_dtype_str}, 0, {comm});
        # """
        
        # if inbuffer == outbuffer:
        #     code += "}" 
        # if node.grid:
        #     code += "}"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Allreduce(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandAllreduceMPI,
    }
    default_implementation = "MPI"

    op = dace.properties.Property(dtype=str, default='MPI_SUM')
    grid = dace.properties.Property(dtype=str, allow_none=True, default=None)

    def __init__(self, name, op='MPI_SUM', grid=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)
        self.op = op
        self.grid = grid

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer = None, None
        inpname, outname = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outname = e.data.data
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inpname = e.data.data
                inbuffer = sdfg.arrays[e.data.data]
        
        in_place = False
        if inpname == outname:
            in_place = True

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), outbuffer, in_place
