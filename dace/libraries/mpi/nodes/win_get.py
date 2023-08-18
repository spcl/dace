# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode


@dace.library.expansion
class ExpandWinGetMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        outbuffer, out_count_str = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(outbuffer.dtype.base_type)

        window_name = node.window_name

        code = f"""
            MPI_Get(_outbuffer, {out_count_str}, {mpi_dtype_str}, \
                    _target_rank, 0, {out_count_str}, {mpi_dtype_str}, \
                    __state->{window_name}_window);
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Win_get(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandWinGetMPI,
    }
    default_implementation = "MPI"

    window_name = dace.properties.Property(dtype=str, default=None)

    def __init__(self, name, window_name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_target_rank"}, outputs={"_out", "_outbuffer"}, **kwargs)
        self.window_name = window_name

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        outbuffer = None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        out_count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                out_count_str = "*".join(dims)
        
        return outbuffer, out_count_str
