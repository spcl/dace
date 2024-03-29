# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import utils
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
import pytest

###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_send_recv")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n], dtype, transient=False)
    sdfg.add_array("y", [n], dtype, transient=False)
    sdfg.add_array("src", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("dest", [1], dace.dtypes.int32, transient=False)
    sdfg.add_array("tag", [1], dace.dtypes.int32, transient=False)
    x = state.add_access("x")
    y = state.add_access("y")
    src = state.add_access("src")
    dest = state.add_access("dest")
    tag = state.add_access("tag")

    send_node = mpi.nodes.send.Send("send")
    recv_node = mpi.nodes.recv.Recv("recv")

    state.add_memlet_path(x, send_node, dst_conn="_buffer", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(dest, send_node, dst_conn="_dest", memlet=Memlet.simple(dest, "0:1", num_accesses=1))
    state.add_memlet_path(tag, send_node, dst_conn="_tag", memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    state.add_memlet_path(recv_node, y, src_conn="_buffer", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(src, recv_node, dst_conn="_src", memlet=Memlet.simple(src, "0:1", num_accesses=1))
    state.add_memlet_path(tag, recv_node, dst_conn="_tag", memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    drank = (rank + 1) % commsize
    srank = (rank - 1 + commsize) % commsize
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()

    size = 128
    A = np.full(size, rank, dtype=dtype)
    B = np.zeros(size, dtype=dtype)
    src = np.array([srank], dtype=np.int32)
    dest = np.array([drank], dtype=np.int32)
    tag = np.array([23], dtype=np.int32)
    mpi_sdfg(x=A, y=B, src=src, dest=dest, tag=tag, n=size)
    # now B should be an array of size, containing srank
    if not np.allclose(B, np.full(size, srank, dtype=dtype)):
        raise (ValueError("The received values are not what I expected."))


@pytest.mark.mpi
def test_mpi():
    _test_mpi("MPI Send/Recv", make_sdfg(np.float64), np.float64)


###############################################################################


@dace.program
def dace_send_recv(rank: dace.int32, size: dace.int32):
    src = np.full([1], (rank - 1) % size, dtype=np.int32)
    dst = np.full([1], (rank + 1) % size, dtype=np.int32)
    sbuf = np.full([1], rank, dtype=np.int32)
    rbuf = np.zeros([1], dtype=np.int32)
    if rank % 2 == 0:
        dace.comm.Recv(rbuf, src, tag=42)
        dace.comm.Send(sbuf, dst, tag=42)
    else:
        dace.comm.Send(sbuf, dst, tag=42)
        dace.comm.Recv(rbuf, src, tag=42)
    return rbuf


@pytest.mark.mpi
def test_dace_send_recv():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    sdfg = None
    if rank == 0:
        sdfg = dace_send_recv.to_sdfg(simplify=True)
        # Disable OpenMP section to allow blocking
        sdfg.openmp_sections = False
    mpi_sdfg = utils.distributed_compile(sdfg, comm)

    val = mpi_sdfg(rank=rank, size=commsize)
    assert (val[0] == (rank - 1) % commsize)


###############################################################################

if __name__ == "__main__":
    test_mpi()
    test_dace_send_recv()
###############################################################################
