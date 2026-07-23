# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data, dtypes, library
from dace.libraries.mpi import utils
from dace.sdfg import nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.mpi.nodes.node import MPINode, resolve_comm, expanded_input_connectors


@library.expansion
class ExpandGathervMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer, in_count_str), outbuffer, recvcounts, displs, root = node.validate(parent_sdfg, parent_state)
        in_mpi_dtype_str = utils.MPI_DDT(inbuffer.dtype.base_type)
        out_mpi_dtype_str = utils.MPI_DDT(outbuffer.dtype.base_type)

        if inbuffer.dtype.veclen > 1:
            raise NotImplementedError
        if root.dtype.base_type != dtypes.int32:
            raise ValueError("Gatherv root must be an integer!")

        comm = resolve_comm(node, parent_state)

        # A scalar send buffer is a value in the tasklet, not a pointer; take its address.
        in_ref = "&" if isinstance(inbuffer, data.Scalar) else ""

        code = f"""
            MPI_Gatherv({in_ref}_inbuffer, {in_count_str}, {in_mpi_dtype_str},
                        _outbuffer, _recvcounts, _displs, {out_mpi_dtype_str},
                        _root, {comm});
            """
        tasklet = nodes.Tasklet(node.name,
                                expanded_input_connectors(node, parent_state),
                                node.out_connectors,
                                code,
                                language=dtypes.Language.CPP,
                                side_effects=True)
        return tasklet


@library.node
class Gatherv(MPINode):

    # Global properties
    implementations = {
        "MPI": ExpandGathervMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer", "_recvcounts", "_displs", "_root"},
                         outputs={"_outbuffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A five-tuple ((inbuffer, in_count_str), outbuffer, recvcounts, displs, root)
                 of the data descriptors in the parent SDFG.
        """
        inbuffer, outbuffer, recvcounts, displs, root = None, None, None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_recvcounts":
                recvcounts = sdfg.arrays[e.data.data]
            if e.dst_conn == "_displs":
                displs = sdfg.arrays[e.data.data]
            if e.dst_conn == "_root":
                root = sdfg.arrays[e.data.data]

        if root.dtype.base_type != dtypes.int32:
            raise ValueError("Gatherv root must be an integer!")
        # MPI_Gatherv takes ``const int[]`` count/displacement arrays.
        if recvcounts.dtype.base_type != dtypes.int32:
            raise ValueError("Gatherv _recvcounts must be an int32 array!")
        if displs.dtype.base_type != dtypes.int32:
            raise ValueError("Gatherv _displs must be an int32 array!")

        in_count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_inbuffer':
                dims = [symstr(e) for e in data.subset.size_exact()]
                in_count_str = "*".join(dims)

        return (inbuffer, in_count_str), outbuffer, recvcounts, displs, root
