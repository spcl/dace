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
        (inbuffer,
         count_str), outbuffer, in_place = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        comm = "MPI_COMM_WORLD"
        if node._grid:
            comm = f"__state->{node._grid}_comm"

        buffer = '_inbuffer'
        if in_place:
            inbuffer = 'MPI_IN_PLACE'

        code = f"""
            MPI_Allreduce({buffer}, _outbuffer, {count_str}, {mpi_dtype_str},
                          {node._op}, {comm});
            """
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

    def __init__(self, name, op, grid, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self._op = op
        self._grid = grid

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
