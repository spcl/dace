# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace import library, properties, symbolic
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    CommNode,
    buffer_pointer,
    count_expression,
    gpu_buffers_ready,
    int_array,
    mpi_communicator,
    mpi_tasklet,
    pack_noncontiguous_buffers,
)
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandGathervMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        # The root places the blocks at the displacements of the output: only the input moves to a packed copy
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state, outputs=())
        inbuffer, outbuffer = node.validate(parent_sdfg, parent_state)
        count = count_expression(sdutil.in_edge_with_name(node, parent_state, "_inbuffer").data.subset)
        code = gpu_buffers_ready(node, parent_sdfg, parent_state)
        code += int_array("__recv_counts", node.recv_counts)
        code += int_array("__recv_displs", node.displacements())
        code += (
            f"MPI_Gatherv({buffer_pointer(node, parent_sdfg, parent_state, '_inbuffer')}, {count}, {utils.MPI_DDT(inbuffer.dtype.base_type)}, _outbuffer, "
            f"__recv_counts, __recv_displs, {utils.MPI_DDT(outbuffer.dtype.base_type)}, int(_root), "
            f"{mpi_communicator(node, parent_state)});\n"
        )
        return mpi_tasklet(node, parent_state, code)


@library.node
class Gatherv(CommNode):
    """
    Gathers a different number of elements from every process at the root, as ``MPI_Gatherv``: the root receives
    ``recv_counts[r]`` elements of process ``r`` at offset ``recv_displs[r]`` of ``_outbuffer`` (by default, after
    the blocks of the processes before it). The counts and displacements are symbolic expressions, one per process
    of the communicator; a value computed at run time reaches them through an inter-state assignment.
    """

    implementations = {"MPI": ExpandGathervMPI}
    default_implementation = "MPI"

    recv_counts = properties.ShapeProperty(
        default=None, allow_none=True, category="Semantics", desc="Number of elements received from each process"
    )
    recv_displs = properties.ShapeProperty(
        default=None,
        allow_none=True,
        category="Semantics",
        desc="Offset in the output of each process's elements; None places them back to back",
    )

    def __init__(self, name, recv_counts=None, recv_displs=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer", "_root"}, outputs={"_outbuffer"}, **kwargs)
        self.recv_counts = None if recv_counts is None else [symbolic.pystr_to_symbolic(c) for c in recv_counts]
        self.recv_displs = None if recv_displs is None else [symbolic.pystr_to_symbolic(d) for d in recv_displs]

    def displacements(self) -> list[symbolic.SymbolicType]:
        """The offset in the output of each process's elements."""
        if self.recv_displs is not None:
            return list(self.recv_displs)
        offsets, offset = [], 0
        for count in self.recv_counts:
            offsets.append(offset)
            offset = offset + count
        return offsets

    @property
    def free_symbols(self) -> set[str]:
        result = super().free_symbols
        for value in (*(self.recv_counts or ()), *(self.recv_displs or ())):
            result.update(str(s) for s in symbolic.pystr_to_symbolic(value).free_symbols)
        return result

    def validate(self, sdfg, state):
        """
        :return: The data descriptors of the input and output buffers.
        """
        if not self.recv_counts:
            raise ValueError(f'Gatherv "{self.label}" needs one receive count per process')
        if self.recv_displs is not None and len(self.recv_displs) != len(self.recv_counts):
            raise ValueError(f'Gatherv "{self.label}" needs as many displacements as receive counts')
        inbuffer = sdutil.in_desc_with_name(self, state, sdfg, "_inbuffer")
        outbuffer = sdutil.out_desc_with_name(self, state, sdfg, "_outbuffer")
        if inbuffer.dtype.veclen > 1:
            raise NotImplementedError("MPI expansions do not support vector types")
        return inbuffer, outbuffer
