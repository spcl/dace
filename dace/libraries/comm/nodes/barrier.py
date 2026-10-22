# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, library
from dace.libraries.comm import environments
from dace.libraries.comm.nodes.node import (
    CommNode,
    NcclExpansion,
    comm_tasklet,
    mpi_communicator,
    nccl_communicator,
    nccl_tasklet,
)
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandBarrierMPI(ExpandTransformation):
    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        return comm_tasklet(node, parent_state, f"MPI_Barrier({mpi_communicator(node, parent_state)});")


@library.expansion
class ExpandBarrierNCCL(NcclExpansion):
    """NCCL has no barrier: an in-place ``ncclAllReduce`` of one byte in a new device buffer synchronizes the
    streams of the processes instead. The host does not wait for it."""

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        buffer, _ = parent_sdfg.add_array(
            "barrier_buffer",
            [1],
            dtypes.uint8,
            storage=dtypes.StorageType.GPU_Global,
            lifetime=dtypes.AllocationLifetime.Persistent,
            transient=True,
            find_new_name=True,
        )
        node.add_in_connector("_inbuffer")
        node.add_out_connector("_outbuffer")
        parent_state.add_edge(parent_state.add_read(buffer), None, node, "_inbuffer", Memlet(f"{buffer}[0]"))
        parent_state.add_edge(node, "_outbuffer", parent_state.add_write(buffer), None, Memlet(f"{buffer}[0]"))
        comm = nccl_communicator(node, parent_sdfg, parent_state)
        body = "DACE_NCCL_CHECK(ncclAllReduce(_inbuffer, _outbuffer, 1, ncclUint8, ncclSum, __comm, __dace_current_stream));"
        return nccl_tasklet(node, parent_state, comm, body)


@library.node
class Barrier(CommNode):
    """Blocks until every process of the communicator reaches it, as ``MPI_Barrier`` (with NCCL, see
    :class:`ExpandBarrierNCCL`)."""

    implementations = {"MPI": ExpandBarrierMPI, "NCCL": ExpandBarrierNCCL}
    default_implementation = None
    side_effects = True

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=set(), **kwargs)
