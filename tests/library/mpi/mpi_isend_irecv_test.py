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
    sdfg.add_array("send_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
    sdfg.add_array("recv_req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)

    sdfg.add_array("stat_source", [1], dace.dtypes.int32, transient=True)
    sdfg.add_array("stat_count", [1], dace.dtypes.int32, transient=True)
    sdfg.add_array("stat_tag", [1], dace.dtypes.int32, transient=True)
    sdfg.add_array("stat_cancelled", [1], dace.dtypes.int32, transient=True)

    x = state.add_access("x")
    y = state.add_access("y")
    src = state.add_access("src")
    dest = state.add_access("dest")
    tag = state.add_access("tag")
    send_req = state.add_access("send_req")
    recv_req = state.add_access("recv_req")

    stat_source = state.add_access("stat_source")
    stat_tag = state.add_access("stat_tag")

    send_node = mpi.nodes.isend.Isend("isend")
    recv_node = mpi.nodes.irecv.Irecv("irecv")
    wait_node = mpi.nodes.wait.Wait("wait")

    state.add_memlet_path(x, send_node, dst_conn="_buffer", memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(send_node,
                          send_req,
                          src_conn="_request",
                          memlet=Memlet.simple(send_req, "0:1", num_accesses=1))
    state.add_memlet_path(dest, send_node, dst_conn="_dest", memlet=Memlet.simple(dest, "0:1", num_accesses=1))
    state.add_memlet_path(tag, send_node, dst_conn="_tag", memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    state.add_memlet_path(recv_node, y, src_conn="_buffer", memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(recv_node,
                          recv_req,
                          src_conn="_request",
                          memlet=Memlet.simple(recv_req, "0:1", num_accesses=1))
    state.add_memlet_path(recv_req,
                          wait_node,
                          dst_conn="_request",
                          memlet=Memlet.simple(recv_req, "0:1", num_accesses=1))

    state.add_memlet_path(wait_node,
                          stat_tag,
                          src_conn="_stat_tag",
                          memlet=Memlet.simple(stat_tag, "0:1", num_accesses=1))
    state.add_memlet_path(wait_node,
                          stat_source,
                          src_conn="_stat_source",
                          memlet=Memlet.simple(stat_source, "0:1", num_accesses=1))

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
    _test_mpi("MPI Isend/Irecv", make_sdfg(np.float64), np.float64)

###############################################################################

@pytest.mark.mpi
def test_isend_irecv():
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    @dace.program
    def mpi4py_isend_irecv(rank: dace.int32, size: dace.int32):
        src = (rank - 1) % size
        dst = (rank + 1) % size
        req = np.empty((2, ), dtype=MPI.Request)
        sbuf = np.full((1,), rank, dtype=np.int32)
        req[0] = commworld.Isend(sbuf, dst, tag=0)
        rbuf = np.empty((1, ), dtype=np.int32)
        req[1] = commworld.Irecv(rbuf, src, tag=0)
        MPI.Request.Waitall(req)
        return rbuf

    sdfg = None
    if rank == 0:
        sdfg = mpi4py_isend_irecv.to_sdfg(simplify=True)
    func = utils.distributed_compile(sdfg, commworld)

    val = func(rank=rank, size=size)
    ref = mpi4py_isend_irecv.f(rank, size)

    assert (val[0] == ref[0])


###############################################################################

if __name__ == "__main__":
    test_mpi()
    test_isend_irecv()
###############################################################################
