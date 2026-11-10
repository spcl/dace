# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ReduceScatter, Alltoallv, averaging reductions and the ordering connectors of Wait."""

import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.memlet import Memlet


def _compile_in_turn(sdfg: dace.SDFG):
    """Compiles on one rank after the other: the ranks share the build folder."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    compiled = None
    for r in range(world.Get_size()):
        if r == world.Get_rank():
            compiled = sdfg.compile()
        world.Barrier()
    return compiled


def _collective_sdfg(name: str, node: dace.nodes.LibraryNode, in_size, out_size, storage=None) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    kwargs = {} if storage is None else {"storage": storage}
    sdfg.add_array("inbuf", [in_size], dace.float64, **kwargs)
    sdfg.add_array("outbuf", [out_size], dace.float64, **kwargs)
    state.add_edge(state.add_read("inbuf"), None, node, "_inbuffer", Memlet(f"inbuf[0:{in_size}]"))
    state.add_edge(node, "_outbuffer", state.add_write("outbuf"), None, Memlet(f"outbuf[0:{out_size}]"))
    return sdfg


@pytest.mark.mpi
@pytest.mark.parametrize("op", ["MPI_SUM", "avg", "max"])
def test_reduce_scatter(op):
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()
    node = comm.ReduceScatter("reduce_scatter", op=op)
    node.implementation = "MPI"
    sdfg = _collective_sdfg(f"reduce_scatter_{op}", node, f"{size} * n", "n")
    compiled = _compile_in_turn(sdfg)

    n = 5
    contributions = [np.arange(size * n, dtype=np.float64) * (r + 1) for r in range(size)]
    outbuf = np.zeros(n)
    compiled(inbuf=contributions[rank].copy(), outbuf=outbuf, n=n)
    reduced = {"MPI_SUM": np.sum, "avg": np.mean, "max": np.max}[op](contributions, axis=0)
    assert np.allclose(outbuf, reduced[rank * n : (rank + 1) * n])


@pytest.mark.mpi
def test_allreduce_average():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()
    node = comm.Allreduce("allreduce_avg", op="avg")
    node.implementation = "MPI"
    sdfg = _collective_sdfg("allreduce_avg", node, "n", "n")
    compiled = _compile_in_turn(sdfg)

    outbuf = np.zeros(6)
    compiled(inbuf=np.full(6, float(rank)), outbuf=outbuf, n=6)
    assert np.allclose(outbuf, (size - 1) / 2)


@pytest.mark.mpi
def test_alltoallv_uneven():
    """Rank ``r`` sends ``r + d + 1`` elements to rank ``d``, with symbolic counts."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()
    me = dace.symbol("me")
    send_counts = [me + d + 1 for d in range(size)]
    recv_counts = [s + me + 1 for s in range(size)]
    node = comm.Alltoallv("alltoallv", send_counts=send_counts, recv_counts=recv_counts)
    node.implementation = "MPI"
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)
    sdfg = _collective_sdfg("alltoallv_uneven", node, total_send, total_recv)
    compiled = _compile_in_turn(sdfg)

    inbuf = np.concatenate([np.full(rank + d + 1, 100.0 * rank + d) for d in range(size)])
    outbuf = np.zeros(sum(s + rank + 1 for s in range(size)))
    compiled(inbuf=inbuf, outbuf=outbuf, me=rank)
    expected = np.concatenate([np.full(s + rank + 1, 100.0 * s + rank) for s in range(size)])
    assert np.allclose(outbuf, expected)


@pytest.mark.mpi
def test_wait_orders_receive_buffer():
    """A ring exchange in one state: the reader of the receive buffer follows the Wait only through its ordering
    connectors."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()

    sdfg = dace.SDFG("wait_ordering")
    state = sdfg.add_state()
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("y", [8], dace.float64)
    sdfg.add_array("received", [8], dace.float64, transient=True)
    for name in ("dest", "src", "tag"):
        sdfg.add_scalar(name, dace.int32)
    sdfg.add_array("reqs", [2], dace.dtypes.opaque("MPI_Request"), transient=True)

    isend = comm.Isend("isend")
    irecv = comm.Irecv("irecv")
    waitall = comm.Waitall("waitall")
    for node in (isend, irecv, waitall):
        node.implementation = "MPI"
    tag = state.add_read("tag")
    reqs = state.add_access("reqs")
    state.add_edge(state.add_read("x"), None, isend, "_buffer", Memlet("x[0:8]"))
    state.add_edge(state.add_read("dest"), None, isend, "_dest", Memlet("dest[0]"))
    state.add_edge(tag, None, isend, "_tag", Memlet("tag[0]"))
    state.add_edge(isend, "_request", reqs, None, Memlet("reqs[0]"))
    received = state.add_access("received")
    state.add_edge(irecv, "_buffer", received, None, Memlet("received[0:8]"))
    state.add_edge(state.add_read("src"), None, irecv, "_src", Memlet("src[0]"))
    state.add_edge(tag, None, irecv, "_tag", Memlet("tag[0]"))
    state.add_edge(irecv, "_request", reqs, None, Memlet("reqs[1]"))
    waitall.add_in_connector("_in_received")
    waitall.add_out_connector("_out_received")
    state.add_edge(reqs, None, waitall, "_request", Memlet("reqs[0:2]"))
    state.add_edge(received, None, waitall, "_in_received", Memlet("received[0:8]"))
    after = state.add_access("received")
    state.add_edge(waitall, "_out_received", after, None, Memlet("received[0:8]"))
    state.add_mapped_tasklet(
        "double",
        {"i": "0:8"},
        {"r": Memlet("received[i]")},
        "o = 2 * r",
        {"o": Memlet("y[i]")},
        input_nodes={"received": after},
        external_edges=True,
    )
    compiled = _compile_in_turn(sdfg)

    x = np.full(8, float(rank))
    y = np.zeros(8)
    compiled(
        x=x,
        y=y,
        dest=np.int32((rank + 1) % size),
        src=np.int32((rank - 1) % size),
        tag=np.int32(0),
    )
    assert np.allclose(y, 2.0 * ((rank - 1) % size))


@pytest.mark.mpi
def test_allgather_of_strided_view():
    """A transposed view as the buffer: the expansion packs it into a contiguous transient."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    size, rank = world.Get_size(), world.Get_rank()
    sdfg = dace.SDFG("allgather_strided_view")
    state = sdfg.add_state()
    sdfg.add_array("x", [4, 3], dace.float64)
    sdfg.add_view("xt", [3, 4], dace.float64, strides=[1, 3])
    sdfg.add_array("out", [size * 3, 4], dace.float64)
    xt = state.add_access("xt")
    state.add_edge(state.add_read("x"), None, xt, "views", Memlet("x[0:4, 0:3]"))
    node = comm.Allgather("allgather")
    node.implementation = "MPI"
    state.add_edge(xt, None, node, "_inbuffer", Memlet("xt[0:3, 0:4]"))
    state.add_edge(node, "_outbuffer", state.add_write("out"), None, Memlet(f"out[0:{size * 3}, 0:4]"))
    compiled = _compile_in_turn(sdfg)

    x = np.arange(12, dtype=np.float64).reshape(4, 3) + 100 * rank
    out = np.zeros((size * 3, 4))
    compiled(x=x, out=out)
    expected = np.concatenate([(np.arange(12, dtype=np.float64).reshape(4, 3) + 100 * r).T for r in range(size)])
    assert np.allclose(out, expected)


def _expanded_code(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_nccl_reduce_scatter_and_alltoallv_expansions():
    rs = comm.ReduceScatter("reduce_scatter", op="avg")
    rs.implementation = "NCCL"
    code = _expanded_code(_collective_sdfg("nccl_rs", rs, "4 * n", "n", storage=dace.StorageType.GPU_Global))
    assert "ncclReduceScatter(_inbuffer, _outbuffer, n, ncclFloat64, ncclAvg" in code

    a2a = comm.Alltoallv("alltoallv", send_counts=["a", "b"], recv_counts=["b", "a"])
    a2a.implementation = "NCCL"
    sdfg = _collective_sdfg("nccl_a2av", a2a, "a + b", "a + b", storage=dace.StorageType.GPU_Global)
    assert {"a", "b"} <= a2a.free_symbols
    code = _expanded_code(sdfg)
    assert "dace::comm::AlltoAllv(" in code


def test_nccl_point_to_point_without_requests():
    """NCCL operations have no requests: Isend/Irecv/Wait without ``_request`` connectors need no MPI type."""
    sdfg = dace.SDFG("nccl_p2p_no_requests")
    state = sdfg.add_state()
    gpu = dace.StorageType.GPU_Global
    sdfg.add_array("x", [8], dace.float64, storage=gpu)
    sdfg.add_array("y", [8], dace.float64, storage=gpu)
    for name in ("peer", "tag"):
        sdfg.add_scalar(name, dace.int32)
    isend = comm.Isend("isend")
    isend.remove_out_connector("_request")
    irecv = comm.Irecv("irecv")
    irecv.remove_out_connector("_request")
    wait = comm.Wait("wait")
    wait.remove_in_connector("_request")
    wait.remove_out_connector("_stat_tag")
    wait.remove_out_connector("_stat_source")
    for node in (isend, irecv, wait):
        node.implementation = "NCCL"
    state.add_edge(state.add_read("x"), None, isend, "_buffer", Memlet("x[0:8]"))
    state.add_edge(state.add_read("peer"), None, isend, "_dest", Memlet("peer[0]"))
    state.add_edge(state.add_read("tag"), None, isend, "_tag", Memlet("tag[0]"))
    y = state.add_access("y")
    state.add_edge(irecv, "_buffer", y, None, Memlet("y[0:8]"))
    state.add_edge(state.add_read("peer"), None, irecv, "_src", Memlet("peer[0]"))
    state.add_edge(state.add_read("tag"), None, irecv, "_tag", Memlet("tag[0]"))
    wait.add_in_connector("_in_y")
    wait.add_out_connector("_out_y")
    state.add_edge(y, None, wait, "_in_y", Memlet("y[0:8]"))
    state.add_edge(wait, "_out_y", state.add_write("y"), None, Memlet("y[0:8]"))
    sdfg.validate()
    code = _expanded_code(sdfg)
    assert "MPI_REQUEST_NULL" not in code
    assert "_stat_tag" not in code
    assert "__state->nccl->Wait(__dace_current_stream);" in code  # Connected to GPU memory: waits on the device


def test_nccl_point_to_point_with_token_requests():
    """An integer request container orders an NCCL Isend before its Wait without MPI types."""
    sdfg = dace.SDFG("nccl_p2p_token_requests")
    state = sdfg.add_state()
    sdfg.add_array("x", [8], dace.float64, storage=dace.StorageType.GPU_Global)
    for name in ("peer", "tag"):
        sdfg.add_scalar(name, dace.int32)
    sdfg.add_scalar("token", dace.int32, transient=True)
    isend = comm.Isend("isend")
    wait = comm.Wait("wait")
    wait.remove_out_connector("_stat_tag")
    wait.remove_out_connector("_stat_source")
    for node in (isend, wait):
        node.implementation = "NCCL"
    state.add_edge(state.add_read("x"), None, isend, "_buffer", Memlet("x[0:8]"))
    state.add_edge(state.add_read("peer"), None, isend, "_dest", Memlet("peer[0]"))
    state.add_edge(state.add_read("tag"), None, isend, "_tag", Memlet("tag[0]"))
    token = state.add_access("token")
    state.add_edge(isend, "_request", token, None, Memlet("token[0]"))
    state.add_edge(token, None, wait, "_request", Memlet("token[0]"))
    sdfg.validate()
    code = _expanded_code(sdfg)
    assert "MPI_REQUEST_NULL" not in code
    assert "_request = 0;" in code


if __name__ == "__main__":
    for op in ["MPI_SUM", "avg", "max"]:
        test_reduce_scatter(op)
    test_allreduce_average()
    test_alltoallv_uneven()
    test_wait_orders_receive_buffer()
    test_allgather_of_strided_view()
    test_nccl_reduce_scatter_and_alltoallv_expansions()
    test_nccl_point_to_point_without_requests()
    test_nccl_point_to_point_with_token_requests()
