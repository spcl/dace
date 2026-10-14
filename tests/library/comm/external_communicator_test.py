# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Communication nodes over a communicator given as an argument (the ``_comm`` connector)."""

import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.libraries.comm import communicators
from dace.memlet import Memlet


def _add_comm(sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.LibraryNode) -> None:
    if "comm" not in sdfg.arrays:
        sdfg.add_scalar("comm", dace.int64)
    node.add_in_connector("_comm")
    state.add_edge(state.add_read("comm"), None, node, "_comm", Memlet("comm[0]"))


def _collective_sdfg(name: str, node: dace.nodes.LibraryNode, in_size: str, out_size: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array("inbuf", [in_size], dace.float64)
    sdfg.add_array("outbuf", [out_size], dace.float64)
    state.add_edge(state.add_read("inbuf"), None, node, "_inbuffer", Memlet(f"inbuf[0:{in_size}]"))
    state.add_edge(node, "_outbuffer", state.add_write("outbuf"), None, Memlet(f"outbuf[0:{out_size}]"))
    _add_comm(sdfg, state, node)
    return sdfg


def _compile_in_turn(sdfg: dace.SDFG, world):
    """Compiles on one rank after the other: the ranks share the build folder."""
    compiled = None
    for r in range(world.Get_size()):
        if r == world.Get_rank():
            compiled = sdfg.compile()
        world.Barrier()
    return compiled


def _subgroup(world):
    """The communicator of every rank but the last (MPI_COMM_NULL on the last), to tell it apart from the world."""
    group = world.Get_group().Incl(list(range(world.Get_size() - 1)))
    return world.Create_group(group)


@pytest.mark.mpi
def test_allreduce_external_mpi_communicator():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 3:
        raise ValueError("This test is supposed to be run with at least three processes")
    sub = _subgroup(world)

    node = comm.Allreduce("allreduce")
    node.implementation = "MPI"
    sdfg = _collective_sdfg("external_comm_allreduce", node, "n", "n")
    compiled = _compile_in_turn(sdfg, world)

    if sub == MPI.COMM_NULL:
        return
    inbuf = np.full(8, world.Get_rank() + 1, dtype=np.float64)
    outbuf = np.zeros(8)
    compiled(inbuf=inbuf, outbuf=outbuf, comm=communicators.mpi_handle(sub), n=8)
    expected = sum(r + 1 for r in range(sub.Get_size()))
    assert np.allclose(outbuf, expected)


@pytest.mark.mpi
def test_allgather_external_mpi_communicator():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 3:
        raise ValueError("This test is supposed to be run with at least three processes")
    sub = _subgroup(world)
    size = world.Get_size() - 1

    node = comm.Allgather("allgather")
    node.implementation = "MPI"
    sdfg = _collective_sdfg("external_comm_allgather", node, "n", f"{size} * n")
    compiled = _compile_in_turn(sdfg, world)

    if sub == MPI.COMM_NULL:
        return
    inbuf = np.full(4, world.Get_rank(), dtype=np.float64)
    outbuf = np.zeros(4 * size)
    compiled(inbuf=inbuf, outbuf=outbuf, comm=communicators.mpi_handle(sub), n=4)
    assert np.allclose(outbuf, np.repeat(np.arange(size, dtype=np.float64), 4))


@pytest.mark.mpi
def test_send_recv_external_mpi_communicator():
    """Ranks are numbered within the given communicator: rank 0 of a communicator that starts at world rank 1."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 3:
        raise ValueError("This test is supposed to be run with at least three processes")
    group = world.Get_group().Incl([1, 2])
    sub = world.Create_group(group)

    sdfg = dace.SDFG("external_comm_send_recv")
    sdfg.add_array("buf", [8], dace.float64)
    sdfg.add_scalar("peer", dace.int32)
    sdfg.add_scalar("tag", dace.int32)
    sdfg.add_scalar("is_sender", dace.int32)
    send_state = sdfg.add_state("send")
    recv_state = sdfg.add_state("recv")
    start = sdfg.add_state("start", is_start_block=True)
    end = sdfg.add_state("end")
    sdfg.add_edge(start, send_state, dace.InterstateEdge("is_sender != 0"))
    sdfg.add_edge(start, recv_state, dace.InterstateEdge("is_sender == 0"))
    sdfg.add_edge(send_state, end, dace.InterstateEdge())
    sdfg.add_edge(recv_state, end, dace.InterstateEdge())

    send = comm.Send("send")
    send.implementation = "MPI"
    send_state.add_edge(send_state.add_read("buf"), None, send, "_buffer", Memlet("buf[0:8]"))
    send_state.add_edge(send_state.add_read("peer"), None, send, "_dest", Memlet("peer[0]"))
    send_state.add_edge(send_state.add_read("tag"), None, send, "_tag", Memlet("tag[0]"))
    _add_comm(sdfg, send_state, send)

    recv = comm.Recv("recv")
    recv.implementation = "MPI"
    recv_state.add_edge(recv, "_buffer", recv_state.add_write("buf"), None, Memlet("buf[0:8]"))
    recv_state.add_edge(recv_state.add_read("peer"), None, recv, "_src", Memlet("peer[0]"))
    recv_state.add_edge(recv_state.add_read("tag"), None, recv, "_tag", Memlet("tag[0]"))
    _add_comm(sdfg, recv_state, recv)

    compiled = _compile_in_turn(sdfg, world)
    if sub == MPI.COMM_NULL:
        return
    sub_rank = sub.Get_rank()
    buf = np.full(8, 7.0) if sub_rank == 0 else np.zeros(8)
    compiled(
        buf=buf,
        peer=np.int32(1 - sub_rank),
        tag=np.int32(3),
        is_sender=np.int32(sub_rank == 0),
        comm=communicators.mpi_handle(sub),
    )
    assert np.allclose(buf, 7.0)


def test_nccl_external_communicator_needs_no_mpi():
    """An NCCL node with a ``_comm`` connector uses the given communicator and bootstraps nothing over MPI."""
    node = comm.Allreduce("allreduce")
    node.implementation = "NCCL"
    sdfg = _collective_sdfg("external_comm_nccl", node, "n", "n")
    for name in ("inbuf", "outbuf"):
        sdfg.arrays[name].storage = dace.StorageType.GPU_Global
    sdfg.expand_library_nodes()

    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1
    assert "((ncclComm_t)(_comm))" in tasklets[0].code.as_string
    environments = set().union(*(t.environments for t in tasklets))
    assert comm.environments.NCCL.full_class_path() in environments
    assert comm.environments.MPI.full_class_path() not in environments
    assert "MPI_COMM_WORLD" not in "".join(code.as_string for code in sdfg.init_code.values())


def test_nccl_world_communicator_bootstraps_over_mpi():
    """Without a ``_comm`` connector, the NCCL expansion creates the world communicator over MPI."""
    sdfg = dace.SDFG("world_comm_nccl")
    state = sdfg.add_state()
    sdfg.add_array("inbuf", ["n"], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("outbuf", ["n"], dace.float64, storage=dace.StorageType.GPU_Global)
    node = comm.Allreduce("allreduce")
    node.implementation = "NCCL"
    state.add_edge(state.add_read("inbuf"), None, node, "_inbuffer", Memlet("inbuf[0:n]"))
    state.add_edge(node, "_outbuffer", state.add_write("outbuf"), None, Memlet("outbuf[0:n]"))
    sdfg.expand_library_nodes()

    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert comm.environments.MPI.full_class_path() in tasklets[0].environments
    assert "__state->nccl->Create(MPI_COMM_WORLD);" in sdfg.init_code["frame"].as_string


if __name__ == "__main__":
    test_allreduce_external_mpi_communicator()
    test_allgather_external_mpi_communicator()
    test_send_recv_external_mpi_communicator()
    test_nccl_external_communicator_needs_no_mpi()
    test_nccl_world_communicator_bootstraps_over_mpi()
