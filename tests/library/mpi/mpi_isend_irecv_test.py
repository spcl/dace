# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
import numpy as np
from mpi4py import MPI as MPI4PY
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
    sdfg.add_array("send_req", [1],
                   dace.dtypes.opaque("MPI_Request"),
                   transient=True)
    sdfg.add_array("recv_req", [1],
                   dace.dtypes.opaque("MPI_Request"),
                   transient=True)

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

    state.add_memlet_path(x,
                          send_node,
                          dst_conn="_buffer",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(send_node,
                          send_req,
                          src_conn="_request",
                          memlet=Memlet.simple(send_req, "0:1", num_accesses=1))
    state.add_memlet_path(dest,
                          send_node,
                          dst_conn="_dest",
                          memlet=Memlet.simple(dest, "0:1", num_accesses=1))
    state.add_memlet_path(tag,
                          send_node,
                          dst_conn="_tag",
                          memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    state.add_memlet_path(recv_node,
                          y,
                          src_conn="_buffer",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
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
                          memlet=Memlet.simple(stat_source,
                                               "0:1",
                                               num_accesses=1))

    state.add_memlet_path(src,
                          recv_node,
                          dst_conn="_src",
                          memlet=Memlet.simple(src, "0:1", num_accesses=1))
    state.add_memlet_path(tag,
                          recv_node,
                          dst_conn="_tag",
                          memlet=Memlet.simple(tag, "0:1", num_accesses=1))
    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    drank = (rank + 1) % commsize
    srank = (rank - 1 + commsize) % commsize
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError(
            "This test is supposed to be run with at least two processes!")
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


# TODO: The test deadlocks in the CI (Ubuntu 18.04, MPICH 3.3a2)
# but works fine in up-to-date systems, including when using pytest.
@pytest.mark.skip
def test_mpi():
    _test_mpi("MPI Send/Recv", make_sdfg(np.float64), np.float64)


###############################################################################

myrank = dace.symbol('myrank', dtype=dace.int32)
mysize = dace.symbol('mysize', dtype=dace.int32)


@dace.program
def dace_send_recv():
    tmp1 = np.full([1], myrank, dtype=np.int32)
    tmp2 = np.zeros([1], dtype=np.int32)
    if myrank == 0:
        dace.comm.Send(tmp1, 1, tag=42)
        dace.comm.Recv(tmp2, mysize - 1, tag=42)
    else:
        dace.comm.Recv(tmp2, (myrank - 1) % mysize, tag=42)
        dace.comm.Send(tmp1, (myrank + 1) % mysize, tag=42)
    return tmp2


# TODO: The test is redundant. It must be updated to use Isend/Irecv.
@pytest.mark.skip
def test_dace_send_recv():
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError(
            "This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = dace_send_recv.compile()
        comm.Barrier()

    prv_rank = mpi_sdfg(myrank=rank, mysize=commsize)

    assert (prv_rank[0] == (rank - 1) % commsize)


###############################################################################

if __name__ == "__main__":
    test_mpi()
    test_dace_send_recv()
###############################################################################
