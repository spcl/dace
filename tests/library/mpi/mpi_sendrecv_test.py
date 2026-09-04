# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime test for the combined MPI_Sendrecv node: a deadlock-free ring exchange.

Rank r sends its buffer (full of ``r``) to rank ``r+1`` and receives from rank ``r-1`` in one call, so
the received buffer must be full of ``(r-1) mod size``. Marked ``mpi`` -- run under ``mpirun -n 2 ...
--with-mpi``.
"""
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
import pytest

###############################################################################


def make_sdfg(dtype):
    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_sendrecv")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n], dtype, transient=False)  # send buffer
    sdfg.add_array("y", [n], dtype, transient=False)  # recv buffer
    sdfg.add_array("src", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("dest", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("sendtag", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("recvtag", [1], dace.dtypes.int32, transient=False)
    x = state.add_access("x")
    y = state.add_access("y")
    src = state.add_access("src")
    dest = state.add_access("dest")
    sendtag = state.add_access("sendtag")
    recvtag = state.add_access("recvtag")

    sr = mpi.nodes.sendrecv.Sendrecv("sendrecv")

    state.add_memlet_path(x, sr, dst_conn="_inbuffer", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(dest, sr, dst_conn="_dest", memlet=Memlet.simple(dest, "0:1", num_accesses=1))
    state.add_memlet_path(src, sr, dst_conn="_src", memlet=Memlet.simple(src, "0:1", num_accesses=1))
    state.add_memlet_path(sendtag, sr, dst_conn="_sendtag", memlet=Memlet.simple(sendtag, "0:1", num_accesses=1))
    state.add_memlet_path(recvtag, sr, dst_conn="_recvtag", memlet=Memlet.simple(recvtag, "0:1", num_accesses=1))
    state.add_memlet_path(sr, y, src_conn="_outbuffer", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    drank = (rank + 1) % commsize
    srank = (rank - 1 + commsize) % commsize
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    mpi_sdfg = None
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    size = 128
    A = np.full(size, rank, dtype=dtype)
    B = np.zeros(size, dtype=dtype)
    src = np.array([srank], dtype=np.int32)
    dest = np.array([drank], dtype=np.int32)
    sendtag = np.array([23], dtype=np.int32)
    recvtag = np.array([23], dtype=np.int32)
    mpi_sdfg(x=A, y=B, src=src, dest=dest, sendtag=sendtag, recvtag=recvtag, n=size)
    # B should hold the left neighbour's rank (srank).
    if not np.allclose(B, np.full(size, srank, dtype=dtype)):
        raise (ValueError("The received values are not what I expected."))


@pytest.mark.mpi
def test_mpi_sendrecv():
    _test_mpi("MPI Sendrecv", make_sdfg(np.float64), np.float64)


if __name__ == "__main__":
    test_mpi_sendrecv()
