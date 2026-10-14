# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Nonblocking collectives: a collective with ``nonblocking=True`` starts its operation (``MPI_I*``) and writes a
request, which a ``Wait`` in a later state completes.

Run under MPI::

    mpirun -n 2 python -m pytest --with-mpi -m mpi tests/library/comm/nonblocking_collectives_test.py
"""

import warnings

import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.libraries.comm import communicators
from dace.memlet import Memlet


def _world():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
    return world


def _compile_in_turn(sdfg: dace.SDFG):
    """Compiles on one rank after the other: the ranks share the build folder."""
    world = _world()
    compiled = None
    for r in range(world.Get_size()):
        if r == world.Get_rank():
            compiled = sdfg.compile()
        world.Barrier()
    return compiled


def _nonblocking_sdfg(
    name: str, node: dace.nodes.LibraryNode, in_size, out_size, storage=None, request_type=None
) -> dace.SDFG:
    """``node`` starts in the first state, writing its request; a ``Wait`` completes it in the second state, ordered
    before the readers of the output through its ordering connectors."""
    sdfg = dace.SDFG(name)
    start = sdfg.add_state("start")
    kwargs = {} if storage is None else {"storage": storage}
    sdfg.add_array("inbuf", [in_size], dace.float64, **kwargs)
    sdfg.add_array("outbuf", [out_size], dace.float64, **kwargs)
    sdfg.add_array("request", [1], request_type or dace.opaque("MPI_Request"), transient=True)
    node.add_out_connector("_request")
    node.nonblocking = True
    start.add_edge(start.add_read("inbuf"), None, node, "_inbuffer", Memlet(f"inbuf[0:{in_size}]"))
    start.add_edge(node, "_outbuffer", start.add_write("outbuf"), None, Memlet(f"outbuf[0:{out_size}]"))
    start.add_edge(node, "_request", start.add_write("request"), None, Memlet("request[0]"))

    complete = sdfg.add_state_after(start, "complete")
    wait = comm.Wait("wait")
    wait.implementation = node.implementation
    wait.remove_out_connector("_stat_tag")
    wait.remove_out_connector("_stat_source")
    wait.add_in_connector("_in_buffer")
    wait.add_out_connector("_out_buffer")
    complete.add_edge(complete.add_read("request"), None, wait, "_request", Memlet("request[0]"))
    complete.add_edge(complete.add_read("outbuf"), None, wait, "_in_buffer", Memlet(f"outbuf[0:{out_size}]"))
    complete.add_edge(wait, "_out_buffer", complete.add_write("outbuf"), None, Memlet(f"outbuf[0:{out_size}]"))
    return sdfg


def _mpi(node: dace.nodes.LibraryNode) -> dace.nodes.LibraryNode:
    node.implementation = "MPI"
    return node


@pytest.mark.mpi
def test_nonblocking_allreduce_allgather_reduce_scatter():
    world = _world()
    size, rank = world.Get_size(), world.Get_rank()
    n = 5
    data = np.arange(size * n, dtype=np.float64) * (rank + 1)

    allreduce = _compile_in_turn(_nonblocking_sdfg("nb_allreduce", _mpi(comm.Allreduce("allreduce")), "n", "n"))
    out = np.zeros(n)
    allreduce(inbuf=data[:n].copy(), outbuf=out, n=n)
    assert np.allclose(out, np.arange(n) * size * (size + 1) / 2)

    allgather = _compile_in_turn(_nonblocking_sdfg("nb_allgather", _mpi(comm.Allgather("allgather")), "n", "s * n"))
    out = np.zeros(size * n)
    allgather(inbuf=data[:n].copy(), outbuf=out, n=n, s=size)
    assert np.allclose(out, np.concatenate([np.arange(n) * (r + 1) for r in range(size)]))

    node = _mpi(comm.ReduceScatter("reduce_scatter", op="max"))
    reduce_scatter = _compile_in_turn(_nonblocking_sdfg("nb_reduce_scatter", node, "s * n", "n"))
    out = np.zeros(n)
    reduce_scatter(inbuf=data.copy(), outbuf=out, n=n, s=size)
    assert np.allclose(out, (np.arange(size * n) * size)[rank * n : (rank + 1) * n])


@pytest.mark.mpi
def test_nonblocking_alltoall_alltoallv_bcast():
    world = _world()
    size, rank = world.Get_size(), world.Get_rank()
    n = 3

    alltoall = _compile_in_turn(_nonblocking_sdfg("nb_alltoall", _mpi(comm.Alltoall("alltoall")), "s * n", "s * n"))
    out = np.zeros(size * n)
    alltoall(inbuf=np.arange(size * n, dtype=np.float64) + 100 * rank, outbuf=out, n=n, s=size)
    assert np.allclose(out, np.concatenate([np.arange(rank * n, (rank + 1) * n) + 100 * s for s in range(size)]))

    # Rank ``r`` sends ``r + d + 1`` elements to rank ``d``, with symbolic counts (which the exchange reads until it
    # completes)
    me = dace.symbol("me")
    send_counts = [me + d + 1 for d in range(size)]
    recv_counts = [s + me + 1 for s in range(size)]
    node = _mpi(comm.Alltoallv("alltoallv", send_counts=send_counts, recv_counts=recv_counts))
    alltoallv = _compile_in_turn(_nonblocking_sdfg("nb_alltoallv", node, sum(send_counts), sum(recv_counts)))
    inbuf = np.concatenate([np.full(rank + d + 1, 100.0 * rank + d) for d in range(size)])
    out = np.zeros(sum(s + rank + 1 for s in range(size)))
    alltoallv(inbuf=inbuf, outbuf=out, me=rank)
    assert np.allclose(out, np.concatenate([np.full(s + rank + 1, 100.0 * s + rank) for s in range(size)]))

    sdfg = _nonblocking_sdfg("nb_bcast", _mpi(comm.Bcast("bcast")), "n", "n")
    sdfg.add_scalar("root", dace.int32)
    start = sdfg.start_state
    bcast = next(node for node in start.nodes() if isinstance(node, comm.Bcast))
    start.add_edge(start.add_read("root"), None, bcast, "_root", Memlet("root[0]"))
    # In place: the output is the input
    for edge in list(start.in_edges(bcast)):
        if edge.dst_conn == "_inbuffer":
            start.remove_node(edge.src)
            start.add_edge(start.add_read("outbuf"), None, bcast, "_inbuffer", Memlet("outbuf[0:n]"))
    del sdfg.arrays["inbuf"]
    bcast_program = _compile_in_turn(sdfg)
    out = np.arange(n, dtype=np.float64) + 10 * rank
    bcast_program(outbuf=out, root=1, n=n)
    assert np.allclose(out, np.arange(n) + 10)


def _expanded_code(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_nonblocking_mpi_expansion():
    code = _expanded_code(_nonblocking_sdfg("nb_allreduce_code", _mpi(comm.Allreduce("allreduce")), "n", "n"))
    assert "MPI_Iallreduce(" in code and "_request);" in code
    assert "MPI_Wait(_request" in code


def test_nonblocking_restrictions():
    """MPI cannot average without blocking; packed copies of strided buffers would not outlive the operation; the
    request must be connected."""
    sdfg = _nonblocking_sdfg("nb_average", _mpi(comm.Allreduce("allreduce", op="avg")), "n", "n")
    with pytest.raises(NotImplementedError, match="average"):
        sdfg.expand_library_nodes()

    sdfg = _nonblocking_sdfg("nb_strided", _mpi(comm.Allreduce("allreduce")), "n", "n")
    sdfg.add_array("matrix", ["n", "n"], dace.float64)
    start = sdfg.start_state
    node = next(node for node in start.nodes() if isinstance(node, comm.Allreduce))
    for edge in list(start.in_edges(node)):
        start.remove_node(edge.src)
    start.add_edge(start.add_read("matrix"), None, node, "_inbuffer", Memlet("matrix[0:n, 0]"))
    with pytest.raises(ValueError, match="contiguous"):
        sdfg.expand_library_nodes()

    sdfg = _nonblocking_sdfg("nb_no_request", _mpi(comm.Allreduce("allreduce")), "n", "n")
    start = sdfg.start_state
    for node in list(start.nodes()):
        if isinstance(node, dace.nodes.AccessNode) and node.data == "request":
            start.remove_node(node)
    with pytest.raises(ValueError, match="no _request output"):
        sdfg.expand_library_nodes()


def test_nonblocking_nccl_packs_strided_buffers():
    """NCCL orders the copies into and out of packed buffers on its stream, so a nonblocking node may have strided
    buffers (MPI may not, see test_nonblocking_restrictions)."""
    node = comm.Allreduce("allreduce")
    node.implementation = "NCCL"
    gpu = dace.StorageType.GPU_Global
    sdfg = _nonblocking_sdfg("nb_nccl_strided", node, "n", "n", storage=gpu, request_type=dace.int32)
    sdfg.add_array("matrix", ["n", "n"], dace.float64, storage=gpu)
    start = sdfg.start_state
    for edge in list(start.in_edges(node)):
        start.remove_node(edge.src)
    start.add_edge(start.add_read("matrix"), None, node, "_inbuffer", Memlet("matrix[0:n, 0]"))
    code = _expanded_code(sdfg)
    assert "ncclAllReduce(" in code
    assert any("packed" in name for name in sdfg.arrays)


def test_nonblocking_nccl_expansion():
    """NCCL (or RCCL) collectives run asynchronously on the stream of their node, nonblocking or not: the collective
    is launched right away, and its Wait orders the stream after it on the device, which is what lets code generation
    keep it on a stream (a host-side wait would force the default stream)."""
    node = comm.Allreduce("allreduce")
    node.implementation = "NCCL"
    sdfg = _nonblocking_sdfg(
        "nb_nccl_allreduce", node, "n", "n", storage=dace.StorageType.GPU_Global, request_type=dace.int32
    )
    code = _expanded_code(sdfg)
    assert code.index("ncclAllReduce(") < code.index("__state->nccl->Launch();")
    assert "Track(" not in code
    assert "_request = 0;" in code
    assert "__state->nccl->Wait(__dace_current_stream);" in code


@pytest.mark.gpu
def test_nccl_collective_orders_on_streams():
    """An NCCL allreduce needs no request or wait: its reader follows it on the device, independent work runs on
    another stream, and the host never waits for a stream after the collective."""
    cp = pytest.importorskip("cupy")
    if not comm.NCCL.is_installed():
        pytest.skip("NCCL is not installed")
    gpu = dace.StorageType.GPU_Global
    sdfg = dace.SDFG("nccl_stream_order")
    sdfg.add_scalar("comm", dace.int64)
    for name in ("inbuf", "reduced", "outbuf", "other", "other_out"):
        sdfg.add_array(name, ["n"], dace.float64, storage=gpu, transient=name == "reduced")
    state = sdfg.add_state()
    node = comm.Allreduce("allreduce")
    node.implementation = "NCCL"
    node.add_in_connector("_comm")
    state.add_edge(state.add_read("inbuf"), None, node, "_inbuffer", Memlet("inbuf[0:n]"))
    state.add_edge(state.add_read("comm"), None, node, "_comm", Memlet("comm[0]"))
    reduced = state.add_access("reduced")
    state.add_edge(node, "_outbuffer", reduced, None, Memlet("reduced[0:n]"))
    state.add_mapped_tasklet(
        "reader",
        {"i": "0:n"},
        {"r": Memlet("reduced[i]")},
        "o = r + 1",
        {"o": Memlet("outbuf[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        input_nodes={"reduced": reduced},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "independent",
        {"i": "0:n"},
        {"x": Memlet("other[i]")},
        "y = 2 * x",
        {"y": Memlet("other_out[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not stream-aware.*")
        compiled = sdfg.compile()
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code())
    assert "StreamSynchronize(__dace_current_stream)" not in code

    handle = communicators.nccl_init_rank(communicators.nccl_unique_id(), 0, 1)
    try:
        n = 1000
        inbuf, other = cp.arange(n, dtype=cp.float64), cp.ones(n, dtype=cp.float64)
        outbuf, other_out = cp.zeros(n, dtype=cp.float64), cp.zeros(n, dtype=cp.float64)
        compiled(inbuf=inbuf, outbuf=outbuf, other=other, other_out=other_out, comm=handle, n=n)
        assert cp.allclose(outbuf, inbuf + 1)
        assert cp.allclose(other_out, 2)
    finally:
        communicators.nccl_destroy(handle)


if __name__ == "__main__":
    test_nonblocking_allreduce_allgather_reduce_scatter()
    test_nonblocking_alltoall_alltoallv_bcast()
    test_nonblocking_mpi_expansion()
    test_nonblocking_restrictions()
    test_nonblocking_nccl_packs_strided_buffers()
    test_nonblocking_nccl_expansion()
    test_nccl_collective_orders_on_streams()
