# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.mpi import nodes as mpi
from dace.memlet import Memlet


def _compile_per_rank(sdfg: dace.SDFG):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    compiled = None
    for r in range(comm.Get_size()):
        if r == comm.Get_rank():
            compiled = sdfg.compile()
        comm.Barrier()
    return compiled


def _write(state: dace.SDFGState, node, conn: str, name: str):
    state.add_edge(node, conn, state.add_write(name), None, Memlet(f"{name}[0]"))


def _add_comm(sdfg: dace.SDFG, state: dace.SDFGState, producer, conn: str, consumer):
    """Pass the ``MPI_Comm`` on ``conn`` of ``producer`` to the ``_comm`` connector of ``consumer``."""
    name = f"_comm_of_{producer.label}"
    sdfg.add_scalar(name, dace.dtypes.opaque("MPI_Comm"), transient=True)
    handle = state.add_access(name)
    state.add_edge(producer, conn, handle, None, Memlet(name))
    consumer.add_in_connector("_comm", dace.dtypes.opaque("MPI_Comm"))
    state.add_edge(handle, None, consumer, "_comm", Memlet(name))


@pytest.mark.mpi
def test_comm_rank_and_size():
    from mpi4py import MPI

    sdfg = dace.SDFG("mpi_comm_rank_and_size")
    state = sdfg.add_state()
    sdfg.add_array("rank", [1], dace.int32)
    sdfg.add_array("size", [1], dace.int32)
    rank = mpi.CommRank("rank")
    size = mpi.CommSize("size")
    _write(state, rank, "_rank", "rank")
    _write(state, size, "_size", "size")
    state.add_node(mpi.Barrier("barrier"))

    rank_out, size_out = np.full(1, -1, np.int32), np.full(1, -1, np.int32)
    _compile_per_rank(sdfg)(rank=rank_out, size=size_out)
    assert rank_out[0] == MPI.COMM_WORLD.Get_rank()
    assert size_out[0] == MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi
def test_comm_split_of_a_fortran_handle():
    """Ranks split by parity from the Fortran handle of the world communicator; each reads its rank in its half."""
    from mpi4py import MPI

    sdfg = dace.SDFG("mpi_comm_split_of_a_fortran_handle")
    state = sdfg.add_state()
    for name in ("fcomm", "color", "key", "subrank"):
        sdfg.add_array(name, [1], dace.int32)
    f2c = mpi.CommF2c("f2c")
    split = mpi.CommSplit("split")
    subrank = mpi.CommRank("subrank")
    state.add_edge(state.add_read("fcomm"), None, f2c, "_fcomm", Memlet("fcomm[0]"))
    state.add_edge(state.add_read("color"), None, split, "_color", Memlet("color[0]"))
    state.add_edge(state.add_read("key"), None, split, "_key", Memlet("key[0]"))
    _add_comm(sdfg, state, f2c, "_comm", split)
    _add_comm(sdfg, state, split, "_newcomm", subrank)
    _write(state, subrank, "_rank", "subrank")

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    out = np.full(1, -1, np.int32)
    _compile_per_rank(sdfg)(
        fcomm=np.array([world.py2f()], np.int32),
        color=np.array([rank % 2], np.int32),
        key=np.array([rank], np.int32),
        subrank=out,
    )
    assert out[0] == rank // 2


@pytest.mark.mpi
def test_gatherv():
    """Rank ``r`` sends ``r + 1`` copies of ``r``; the root receives them back to back."""
    from mpi4py import MPI

    n = dace.symbol("n")
    sdfg = dace.SDFG("mpi_gatherv")
    state = sdfg.add_state()
    sdfg.add_array("send", [n], dace.float64)
    sdfg.add_array("recv", ["total"], dace.float64)
    sdfg.add_array("counts", ["procs"], dace.int32)
    sdfg.add_array("displs", ["procs"], dace.int32)
    sdfg.add_array("root", [1], dace.int32)
    gatherv = mpi.Gatherv("gatherv")
    state.add_edge(state.add_read("send"), None, gatherv, "_inbuffer", Memlet("send[0:n]"))
    state.add_edge(state.add_read("counts"), None, gatherv, "_recvcounts", Memlet("counts[0:procs]"))
    state.add_edge(state.add_read("displs"), None, gatherv, "_displs", Memlet("displs[0:procs]"))
    state.add_edge(state.add_read("root"), None, gatherv, "_root", Memlet("root[0]"))
    state.add_edge(gatherv, "_outbuffer", state.add_write("recv"), None, Memlet("recv[0:total]"))

    procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    counts = np.arange(1, procs + 1, dtype=np.int32)
    displs = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    total = int(counts.sum())
    recv = np.full(total, -1.0)
    _compile_per_rank(sdfg)(
        send=np.full(rank + 1, float(rank)),
        recv=recv,
        counts=counts,
        displs=displs,
        root=np.zeros(1, np.int32),
        n=rank + 1,
        total=total,
        procs=procs,
    )
    if rank == 0:
        assert np.array_equal(recv, np.repeat(np.arange(procs, dtype=np.float64), counts))


def test_abort_calls_mpi_abort_on_the_resolved_communicator():
    sdfg = dace.SDFG("mpi_abort")
    state = sdfg.add_state()
    sdfg.add_array("errorcode", [1], dace.int32)
    abort = mpi.Abort("abort")
    state.add_edge(state.add_read("errorcode"), None, abort, "_errorcode", Memlet("errorcode[0]"))
    tasklet = abort.implementations["MPI"].expansion(abort, state, sdfg)
    assert tasklet.code.as_string == "MPI_Abort(MPI_COMM_WORLD, _errorcode);"


if __name__ == "__main__":
    test_comm_rank_and_size()
    test_comm_split_of_a_fortran_handle()
    test_gatherv()
    test_abort_calls_mpi_abort_on_the_resolved_communicator()
