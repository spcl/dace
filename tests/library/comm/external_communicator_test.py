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
    """The communicator of every rank but the last (MPI_COMM_NULL on the last), to tell it apart from the world.
    On two processes, it has one member."""
    group = world.Get_group().Incl(list(range(world.Get_size() - 1)))
    return world.Create_group(group)


@pytest.mark.mpi
def test_allreduce_external_mpi_communicator():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
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
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
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
    """Ranks are numbered within the given communicator: the world in reverse order, so that its rank 0 is the
    last world rank."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
    group = world.Get_group().Incl(list(reversed(range(world.Get_size()))))
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
    sub_rank = sub.Get_rank()
    if sub_rank > 1:
        return
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


def test_nccl_unique_id_keeps_every_byte():
    """The unique ID is binary: it must not stop at its first zero byte."""
    if not comm.NCCL.is_installed():
        pytest.skip("NCCL is not installed")
    assert len(communicators.nccl_unique_id()) == communicators.NCCL_UNIQUE_ID_BYTES


@pytest.mark.gpu
def test_nccl_external_communicator_single_process():
    """Collectives and a self-exchange over an NCCL communicator created in Python, with no MPI in the program."""
    cp = pytest.importorskip("cupy")
    if not comm.NCCL.is_installed():
        pytest.skip("NCCL is not installed")
    gpu = dace.StorageType.GPU_Global
    sdfg = dace.SDFG("external_comm_nccl_single")
    sdfg.add_scalar("comm", dace.int64)
    for name in ("a", "b", "c", "d"):
        sdfg.add_array(name, ["n"], dace.float32, storage=gpu)
    for name in ("peer", "tag", "send_token", "recv_token"):
        sdfg.add_scalar(name, dace.int32, transient=True)

    init = sdfg.add_state("init")
    tasklet = init.add_tasklet("init", {}, {"p", "t"}, "p = 0\nt = 0")
    init.add_edge(tasklet, "p", init.add_write("peer"), None, Memlet("peer[0]"))
    init.add_edge(tasklet, "t", init.add_write("tag"), None, Memlet("tag[0]"))

    state = sdfg.add_state_after(init, "collectives")
    allreduce = comm.Allreduce("allreduce", op="avg")
    state.add_edge(state.add_read("a"), None, allreduce, "_inbuffer", Memlet("a[0:n]"))
    b = state.add_access("b")
    state.add_edge(allreduce, "_outbuffer", b, None, Memlet("b[0:n]"))
    reduce_scatter = comm.ReduceScatter("reduce_scatter")
    state.add_edge(b, None, reduce_scatter, "_inbuffer", Memlet("b[0:n]"))
    state.add_edge(reduce_scatter, "_outbuffer", state.add_write("c"), None, Memlet("c[0:n]"))

    p2p = sdfg.add_state_after(state, "exchange")
    isend, irecv, wait = comm.Isend("isend"), comm.Irecv("irecv"), comm.Wait("wait")
    peer, tag = p2p.add_read("peer"), p2p.add_read("tag")
    p2p.add_edge(p2p.add_read("c"), None, isend, "_buffer", Memlet("c[0:n]"))
    p2p.add_edge(peer, None, isend, "_dest", Memlet("peer[0]"))
    p2p.add_edge(tag, None, isend, "_tag", Memlet("tag[0]"))
    p2p.add_edge(peer, None, irecv, "_src", Memlet("peer[0]"))
    p2p.add_edge(tag, None, irecv, "_tag", Memlet("tag[0]"))
    send_token, recv_token = p2p.add_access("send_token"), p2p.add_access("recv_token")
    p2p.add_edge(isend, "_request", send_token, None, Memlet("send_token[0]"))
    p2p.add_edge(irecv, "_request", recv_token, None, Memlet("recv_token[0]"))
    received = p2p.add_access("d")
    p2p.add_edge(irecv, "_buffer", received, None, Memlet("d[0:n]"))
    wait.remove_out_connector("_stat_tag")
    wait.remove_out_connector("_stat_source")
    wait.add_in_connector("_recv_token")
    wait.add_in_connector("_in_d")
    wait.add_out_connector("_out_d")
    p2p.add_edge(send_token, None, wait, "_request", Memlet("send_token[0]"))
    p2p.add_edge(recv_token, None, wait, "_recv_token", Memlet("recv_token[0]"))
    p2p.add_edge(received, None, wait, "_in_d", Memlet("d[0:n]"))
    p2p.add_edge(wait, "_out_d", p2p.add_write("d"), None, Memlet("d[0:n]"))
    for st, node in ((state, allreduce), (state, reduce_scatter), (p2p, isend), (p2p, irecv)):
        _add_comm(sdfg, st, node)
    for node in (allreduce, reduce_scatter, isend, irecv, wait):
        node.implementation = "NCCL"
    compiled = sdfg.compile()

    handle = communicators.nccl_init_rank(communicators.nccl_unique_id(), 0, 1)
    try:
        n = 1000
        a = cp.arange(n, dtype=cp.float32)
        b, c, d = (cp.zeros(n, dtype=cp.float32) for _ in range(3))
        compiled(a=a, b=b, c=c, d=d, comm=handle, n=n)
        cp.cuda.runtime.deviceSynchronize()
        for result in (b, c, d):
            assert cp.allclose(result, a)
    finally:
        communicators.nccl_destroy(handle)


if __name__ == "__main__":
    test_allreduce_external_mpi_communicator()
    test_allgather_external_mpi_communicator()
    test_send_recv_external_mpi_communicator()
    test_nccl_external_communicator_needs_no_mpi()
    test_nccl_world_communicator_bootstraps_over_mpi()
    test_nccl_unique_id_keeps_every_byte()
    test_nccl_external_communicator_single_process()
