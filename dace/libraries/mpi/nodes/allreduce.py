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
         count_str), outbuffer = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dace.libraries.mpi.utils.MPI_DDT(
            inbuffer.dtype.base_type)
        if inbuffer.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""
            MPI_Allreduce(_inbuffer, _outbuffer, {count_str}, {mpi_dtype_str},
                          {node._op}, MPI_COMM_WORLD);
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

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self._op = kwargs.get('op', "MPI_SUM")

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), outbuffer
