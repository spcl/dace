# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data, dtypes, library
from dace.libraries.mpi import environments, utils
from dace.libraries.mpi.nodes.node import MPINode, count_expression, mpi_tasklet, resolve_comm
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandGathervMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        inbuffer = sdutil.in_desc_with_name(node, parent_state, parent_sdfg, "_inbuffer")
        outbuffer = sdutil.out_desc_with_name(node, parent_state, parent_sdfg, "_outbuffer")
        if inbuffer.dtype.veclen > 1:
            raise NotImplementedError
        count = count_expression(
            sdutil.in_edge_with_name(node, parent_state, "_inbuffer").data.subset.num_elements_exact()
        )
        # A scalar send buffer is a value in the tasklet, not a pointer
        in_ref = "&" if isinstance(inbuffer, data.Scalar) else ""
        code = (
            f"MPI_Gatherv({in_ref}_inbuffer, {count}, {utils.MPI_DDT(inbuffer.dtype.base_type)}, _outbuffer, "
            f"_recvcounts, _displs, {utils.MPI_DDT(outbuffer.dtype.base_type)}, _root, "
            f"{resolve_comm(node, parent_state)});"
        )
        return mpi_tasklet(node, parent_state, code)


@library.node
class Gatherv(MPINode):
    """``MPI_Gatherv``: the root receives ``_recvcounts[r]`` elements of rank ``r`` at ``_displs[r]`` of
    ``_outbuffer``."""

    implementations = {"MPI": ExpandGathervMPI}
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(
            name,
            *args,
            inputs=["_inbuffer", "_recvcounts", "_displs", "_root"],
            outputs=["_outbuffer"],
            **kwargs,
        )

    def validate(self, sdfg, state):
        # MPI_Gatherv takes an int root and int arrays of counts and displacements
        for conn in ("_root", "_recvcounts", "_displs"):
            if sdutil.in_desc_with_name(self, state, sdfg, conn).dtype.base_type != dtypes.int32:
                raise ValueError(f"Gatherv {conn} must be int32")
