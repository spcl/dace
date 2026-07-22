# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end: a strided (column) halo exchange made MPI-legal by MpiPackUnpack.

Column ``COL`` of A is a non-contiguous slice (stride N in C layout) -- the shape a permuted/blocked halo
takes. ``MpiPackUnpack`` packs it into a contiguous transient before the ``Sendrecv`` and unpacks the
received contiguous buffer back into column ``COL`` of B. A 2-rank ring then checks B's column equals the
left neighbour's rank. Marked ``mpi`` -- run under ``mpirun -n 2``.
"""
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
from dace.transformation.layout.mpi_pack_unpack import MpiPackUnpack
import numpy as np
import pytest

COL = 2

###############################################################################


def make_sdfg(dtype):
    n = dace.symbol("n")
    sdfg = dace.SDFG("mpi_pack_exchange")
    state = sdfg.add_state("dataflow")
    sdfg.add_array("A", [n, n], dtype, transient=False)  # send column COL of A
    sdfg.add_array("B", [n, n], dtype, transient=False)  # recv into column COL of B
    for nm in ("src", "dest", "sendtag", "recvtag"):
        sdfg.add_array(nm, [1], dace.dtypes.int32, transient=False)

    A = state.add_access("A")
    B = state.add_access("B")
    sr = mpi.nodes.sendrecv.Sendrecv("sendrecv")
    col = f"0:n, {COL}:{COL + 1}"
    state.add_memlet_path(A, sr, dst_conn="_inbuffer", memlet=Memlet.simple("A", col))
    state.add_memlet_path(state.add_access("dest"), sr, dst_conn="_dest", memlet=Memlet.simple("dest", "0:1"))
    state.add_memlet_path(state.add_access("src"), sr, dst_conn="_src", memlet=Memlet.simple("src", "0:1"))
    state.add_memlet_path(state.add_access("sendtag"), sr, dst_conn="_sendtag", memlet=Memlet.simple("sendtag", "0:1"))
    state.add_memlet_path(state.add_access("recvtag"), sr, dst_conn="_recvtag", memlet=Memlet.simple("recvtag", "0:1"))
    state.add_memlet_path(sr, B, src_conn="_outbuffer", memlet=Memlet.simple("B", col))

    # Strided columns -> contiguous packed buffers, so Sendrecv sees no stride.
    assert MpiPackUnpack().apply_pass(sdfg, {}) == 2  # one pack (send) + one unpack (recv)
    return sdfg


###############################################################################


def _test_mpi(sdfg, dtype):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    drank = (rank + 1) % size
    srank = (rank - 1 + size) % size
    if size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    mpi_sdfg = None
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    N = 16
    A = np.full((N, N), rank, dtype=dtype)  # every column of A holds this rank
    B = np.zeros((N, N), dtype=dtype)
    mpi_sdfg(A=A,
             B=B,
             src=np.array([srank], dtype=np.int32),
             dest=np.array([drank], dtype=np.int32),
             sendtag=np.array([7], dtype=np.int32),
             recvtag=np.array([7], dtype=np.int32),
             n=N)
    # B's column COL was received from the left neighbour -> holds srank.
    if not np.allclose(B[:, COL], srank):
        raise ValueError(f"rank {rank}: B[:,{COL}] = {B[:, COL]} (expected {srank})")


@pytest.mark.mpi
def test_mpi_pack_unpack_exchange():
    _test_mpi(make_sdfg(np.float64), np.float64)


if __name__ == "__main__":
    test_mpi_pack_unpack_exchange()
