# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from collections.abc import Sequence

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import symbolic
from dace.libraries.comm import environments, utils
from dace.libraries.comm.nodes.node import (
    NcclExpansion,
    NonblockingCommNode,
    comm_tasklet,
    gpu_buffers_ready,
    int_array,
    mpi_collective,
    mpi_communicator,
    mpi_request_connector,
    nccl_communicator,
    nccl_tasklet,
    pack_noncontiguous_buffers,
    validate_nccl_buffers,
)
from dace.transformation.transformation import ExpandTransformation


def _counts_and_displacements(name: str, counts: Sequence, factor: int, ctype: str, persistent: bool = False) -> str:
    """C++ arrays ``<name>_counts`` and ``<name>_displs`` (exclusive prefix sums) of ``counts`` times ``factor``,
    ``static`` if ``persistent`` (for arrays that must outlive the code, e.g., of a nonblocking operation)."""
    storage = "static " if persistent else ""
    return f"""
        {int_array(f"{name}_counts", [symbolic.pystr_to_symbolic(c) * factor for c in counts], ctype, persistent)}
        {storage}{ctype} {name}_displs[{len(counts)}];
        {name}_displs[0] = 0;
        for (int __i = 1; __i < {len(counts)}; ++__i) {name}_displs[__i] = {name}_displs[__i - 1] + {name}_counts[__i - 1];
        """


@dace.library.expansion
class ExpandAlltoallvMPI(ExpandTransformation):
    """Exchanges with ``MPI_Alltoallv``."""

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        node.validate_nonblocking(parent_sdfg, parent_state)
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        inbuffer, outbuffer = node.validate(parent_sdfg, parent_state)
        if inbuffer.dtype.veclen > 1:
            raise NotImplementedError("MPI expansions do not support vector types")
        mpi_dtype_str = utils.MPI_DDT(inbuffer.dtype.base_type)
        comm = mpi_communicator(node, parent_state)
        # A nonblocking exchange reads the counts and displacements until it completes: they outlive the tasklet
        code = gpu_buffers_ready(node, parent_sdfg, parent_state)
        code += _counts_and_displacements("__send", node.send_counts, 1, "int", node.nonblocking)
        code += _counts_and_displacements("__recv", node.recv_counts, 1, "int", node.nonblocking)
        code += mpi_collective(
            node,
            "Alltoallv",
            f"_inbuffer, __send_counts, __send_displs, {mpi_dtype_str}, "
            f"_outbuffer, __recv_counts, __recv_displs, "
            f"{mpi_dtype_str}, {comm}",
        )
        return mpi_request_connector(comm_tasklet(node, parent_state, code))


@dace.library.expansion
class ExpandAlltoallvNCCL(NcclExpansion):
    """Exchanges with grouped ``ncclSend``/``ncclRecv`` calls (``dace::comm::AlltoAllv``)."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        pack_noncontiguous_buffers(node, parent_sdfg, parent_state)
        inbuffer, outbuffer = node.validate(parent_sdfg, parent_state)
        validate_nccl_buffers(node, inbuffer, outbuffer)
        nccl_type, factor = utils.nccl_transfer_type(inbuffer.dtype.base_type)
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = _counts_and_displacements("__send", node.send_counts, factor, "size_t")
        body += _counts_and_displacements("__recv", node.recv_counts, factor, "size_t")
        body += f"""
            dace::comm::AlltoAllv(_inbuffer, __send_counts, __send_displs, _outbuffer, __recv_counts, __recv_displs,
                                  {nccl_type}, __comm, __dace_current_stream);
            """
        return nccl_tasklet(node, parent_state, comm, body, sdfg=parent_sdfg)


@dace.library.node
class Alltoallv(NonblockingCommNode):
    """
    Exchanges a different number of elements with every process, as ``MPI_Alltoallv``: the input (``_inbuffer``)
    holds ``send_counts[i]`` elements for process ``i`` after those for the processes before it, and the output
    (``_outbuffer``) receives ``recv_counts[i]`` elements from process ``i`` in the same order. The counts are
    symbolic expressions, one per process of the communicator. The buffers must be contiguous.
    """

    implementations = {
        "MPI": ExpandAlltoallvMPI,
        "NCCL": ExpandAlltoallvNCCL,
    }
    default_implementation = None

    send_counts = dace.properties.ShapeProperty(
        default=None, allow_none=True, category="Semantics", desc="Number of elements sent to each process"
    )
    recv_counts = dace.properties.ShapeProperty(
        default=None, allow_none=True, category="Semantics", desc="Number of elements received from each process"
    )

    def __init__(self, name, send_counts=None, recv_counts=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_inbuffer"}, outputs={"_outbuffer"}, **kwargs)
        self.send_counts = None if send_counts is None else [symbolic.pystr_to_symbolic(c) for c in send_counts]
        self.recv_counts = None if recv_counts is None else [symbolic.pystr_to_symbolic(c) for c in recv_counts]

    @property
    def free_symbols(self) -> set[str]:
        result = super().free_symbols
        for count in (*(self.send_counts or ()), *(self.recv_counts or ())):
            result.update(str(s) for s in symbolic.pystr_to_symbolic(count).free_symbols)
        return result

    def validate(self, sdfg, state):
        """
        :return: The data descriptors of the input and output buffers.
        """
        if not self.send_counts or not self.recv_counts or len(self.send_counts) != len(self.recv_counts):
            raise ValueError(f'Alltoallv "{self.label}" needs as many send counts as receive counts, one per process')
        inbuffer, outbuffer = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        if inbuffer is None or outbuffer is None:
            raise ValueError(f'Alltoallv "{self.label}" needs an input and an output buffer')
        return inbuffer, outbuffer
