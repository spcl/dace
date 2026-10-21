# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Communicator queries, splits, barriers, aborts and Gatherv."""

import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.libraries.comm import communicators
from dace.memlet import Memlet


def _compile_in_turn(sdfg: dace.SDFG, world):
    """Compiles on one rank after the other: the ranks share the build folder."""
    compiled = None
    for r in range(world.Get_size()):
        if r == world.Get_rank():
            compiled = sdfg.compile()
        world.Barrier()
    return compiled


def _world():
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
    return world


def _write(state: dace.SDFGState, node: dace.nodes.LibraryNode, connector: str, name: str) -> None:
    state.add_edge(node, connector, state.add_write(name), None, Memlet(f"{name}[0]"))


def _pass_handle(sdfg: dace.SDFG, state: dace.SDFGState, producer, connector: str, consumer) -> None:
    """Connects the communicator handle on ``connector`` of ``producer`` to the ``_comm`` connector of ``consumer``."""
    name = f"{producer.label}_handle"
    sdfg.add_scalar(name, dace.int64, transient=True)
    handle = state.add_access(name)
    state.add_edge(producer, connector, handle, None, Memlet(name))
    consumer.add_in_connector("_comm")
    state.add_edge(handle, None, consumer, "_comm", Memlet(name))


@pytest.mark.mpi
def test_comm_rank_and_size():
    world = _world()
    sdfg = dace.SDFG("mpi_comm_rank_and_size")
    state = sdfg.add_state()
    sdfg.add_array("rank", [1], dace.int32)
    sdfg.add_array("size", [1], dace.int32)
    _write(state, comm.CommRank("rank"), "_rank", "rank")
    _write(state, comm.CommSize("size"), "_size", "size")
    state.add_node(comm.Barrier("barrier"))

    rank, size = np.full(1, -1, np.int32), np.full(1, -1, np.int32)
    _compile_in_turn(sdfg, world)(rank=rank, size=size)
    assert rank[0] == world.Get_rank()
    assert size[0] == world.Get_size()


@pytest.mark.mpi
def test_comm_rank_of_an_external_communicator():
    """The rank in the reversed world, passed as a handle on ``_comm``."""
    world = _world()
    reversed_world = world.Split(0, world.Get_size() - 1 - world.Get_rank())
    sdfg = dace.SDFG("mpi_comm_rank_of_an_external_communicator")
    state = sdfg.add_state()
    sdfg.add_array("rank", [1], dace.int32)
    sdfg.add_scalar("handle", dace.int64)
    node = comm.CommRank("rank")
    node.add_in_connector("_comm")
    state.add_edge(state.add_read("handle"), None, node, "_comm", Memlet("handle[0]"))
    _write(state, node, "_rank", "rank")

    rank = np.full(1, -1, np.int32)
    _compile_in_turn(sdfg, world)(rank=rank, handle=communicators.mpi_handle(reversed_world))
    assert rank[0] == world.Get_size() - 1 - world.Get_rank()
    reversed_world.Free()


@pytest.mark.mpi
def test_comm_split_of_a_fortran_handle():
    """The ranks split by parity from the Fortran handle of the world; each reads its rank in its half, then frees it."""
    world = _world()
    sdfg = dace.SDFG("mpi_comm_split_of_a_fortran_handle")
    state = sdfg.add_state()
    for name in ("fcomm", "color", "key", "subrank"):
        sdfg.add_array(name, [1], dace.int32)
    f2c = comm.CommF2c("f2c")
    split = comm.CommSplit("split")
    subrank = comm.CommRank("subrank")
    state.add_edge(state.add_read("fcomm"), None, f2c, "_fcomm", Memlet("fcomm[0]"))
    state.add_edge(state.add_read("color"), None, split, "_color", Memlet("color[0]"))
    state.add_edge(state.add_read("key"), None, split, "_key", Memlet("key[0]"))
    _pass_handle(sdfg, state, f2c, "_handle", split)
    _pass_handle(sdfg, state, split, "_newcomm", subrank)
    _write(state, subrank, "_rank", "subrank")
    free = sdfg.add_state_after(state, "free")
    free_node = comm.CommFree("free")
    free.add_edge(free.add_read("split_handle"), None, free_node, "_comm", Memlet("split_handle"))

    rank = world.Get_rank()
    subrank_out = np.full(1, -1, np.int32)
    _compile_in_turn(sdfg, world)(
        fcomm=np.array([world.py2f()], np.int32),
        color=np.array([rank % 2], np.int32),
        key=np.array([rank], np.int32),
        subrank=subrank_out,
    )
    assert subrank_out[0] == rank // 2


def _gatherv_sdfg(name: str, procs: int, displs=None) -> dace.SDFG:
    """Gathers ``send[0:n]`` of every process into ``recv`` at the root; the counts come from the ``counts`` array
    through an inter-state assignment, one symbol per process."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("send", ["n"], dace.float64)
    sdfg.add_array("recv", ["total"], dace.float64)
    sdfg.add_array("counts", [procs], dace.int32)
    sdfg.add_array("root", [1], dace.int32)
    first = sdfg.add_state("read_counts")
    state = sdfg.add_state_after(first, "gather", assignments={f"c{r}": f"counts[{r}]" for r in range(procs)})
    gatherv = comm.Gatherv("gatherv", recv_counts=[f"c{r}" for r in range(procs)], recv_displs=displs)
    state.add_edge(state.add_read("send"), None, gatherv, "_inbuffer", Memlet("send[0:n]"))
    state.add_edge(state.add_read("root"), None, gatherv, "_root", Memlet("root[0]"))
    state.add_edge(gatherv, "_outbuffer", state.add_write("recv"), None, Memlet("recv[0:total]"))
    return sdfg


@pytest.mark.mpi
def test_gatherv_with_counts_read_at_run_time():
    """Process ``r`` sends ``r + 1`` copies of ``r``; the root receives them back to back."""
    world = _world()
    procs, rank = world.Get_size(), world.Get_rank()
    counts = np.arange(1, procs + 1, dtype=np.int32)
    recv = np.full(int(counts.sum()), -1.0)
    _compile_in_turn(_gatherv_sdfg("mpi_gatherv", procs), world)(
        send=np.full(rank + 1, float(rank)),
        recv=recv,
        counts=counts,
        root=np.zeros(1, np.int32),
        n=rank + 1,
        total=int(counts.sum()),
    )
    if rank == 0:
        assert np.array_equal(recv, np.repeat(np.arange(procs, dtype=np.float64), counts))


@pytest.mark.mpi
def test_gatherv_at_explicit_displacements():
    """Every process sends one element; the root places process ``r``'s at ``2 * r``, leaving gaps."""
    world = _world()
    procs, rank = world.Get_size(), world.Get_rank()
    recv = np.full(2 * procs, -1.0)
    sdfg = _gatherv_sdfg("mpi_gatherv_displs", procs, displs=[2 * r for r in range(procs)])
    _compile_in_turn(sdfg, world)(
        send=np.full(1, float(rank)),
        recv=recv,
        counts=np.ones(procs, np.int32),
        root=np.zeros(1, np.int32),
        n=1,
        total=2 * procs,
    )
    if rank == 0:
        assert np.array_equal(recv[0::2], np.arange(procs, dtype=np.float64))
        assert np.all(recv[1::2] == -1.0)


def _expanded_code(node, sdfg: dace.SDFG, state: dace.SDFGState) -> str:
    return node.implementations["MPI"].expansion(node, state, sdfg).code.as_string


def test_counts_are_cpp_expressions():
    """An element count keeps its parentheses, and a floor division never reaches C++ as ``//``."""
    sdfg = dace.SDFG("count_expressions")
    state = sdfg.add_state()
    sdfg.add_array("a", ["N + 1", "M"], dace.float64)
    sdfg.add_array("b", ["N"], dace.float64)
    allreduce = comm.Allreduce("allreduce")
    state.add_edge(state.add_read("a"), None, allreduce, "_inbuffer", Memlet("a[0:N+1, 0:M]"))
    state.add_edge(allreduce, "_outbuffer", state.add_write("a"), None, Memlet("a[0:N+1, 0:M]"))
    assert "(M * (N + 1))" in _expanded_code(allreduce, sdfg, state)

    state = sdfg.add_state()
    sdfg.add_array("root", [1], dace.int32)
    bcast = comm.Bcast("bcast")
    state.add_edge(state.add_read("b"), None, bcast, "_inbuffer", Memlet("b[0:N//2]"))
    state.add_edge(state.add_read("root"), None, bcast, "_root", Memlet("root[0]"))
    state.add_edge(bcast, "_outbuffer", state.add_write("b"), None, Memlet("b[0:N//2]"))
    assert "//" not in _expanded_code(bcast, sdfg, state)


def _fortran_allreduce_sdfg(name: str, element_order: str, nonblocking: bool = False) -> dace.SDFG:
    """An in-place Allreduce of a column-major (Fortran) ``a(3, 4)``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [3, 4], dace.float64, strides=[1, 3])
    state = sdfg.add_state()
    allreduce = comm.Allreduce("allreduce")
    allreduce.element_order = element_order
    allreduce.nonblocking = nonblocking
    state.add_edge(state.add_read("a"), None, allreduce, "_inbuffer", Memlet("a[0:3, 0:4]"))
    state.add_edge(allreduce, "_outbuffer", state.add_write("a"), None, Memlet("a[0:3, 0:4]"))
    if nonblocking:
        sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"))
        state.add_edge(allreduce, "_request", state.add_write("req"), None, Memlet("req[0]"))
    return sdfg


@pytest.mark.parametrize("element_order, packed", [("row_major", True), ("memory", False)])
def test_a_column_major_buffer_moves_in_place_in_memory_order(element_order: str, packed: bool):
    sdfg = _fortran_allreduce_sdfg(f"fortran_allreduce_{element_order}", element_order)
    sdfg.expand_library_nodes(recursive=False)
    assert any("_packed" in name for name in sdfg.arrays) == packed


def test_a_nonblocking_collective_takes_a_column_major_buffer_in_memory_order():
    sdfg = _fortran_allreduce_sdfg("fortran_iallreduce", "memory", nonblocking=True)
    state = sdfg.start_state
    node = next(n for n in state.nodes() if isinstance(n, comm.Allreduce))
    node.validate_nonblocking(sdfg, state)
    with pytest.raises(ValueError, match="contiguous"):
        node.element_order = "row_major"
        node.validate_nonblocking(sdfg, state)


@pytest.mark.mpi
def test_a_column_major_allreduce_in_memory_order():
    world = _world()
    a = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4) * (world.Get_rank() + 1))
    sdfg = _fortran_allreduce_sdfg("mpi_fortran_allreduce", "memory")
    _compile_in_turn(sdfg, world)(a=a)
    scale = sum(r + 1 for r in range(world.Get_size()))
    assert np.array_equal(a, np.arange(12, dtype=np.float64).reshape(3, 4) * scale)


def test_abort_over_the_world():
    sdfg = dace.SDFG("mpi_abort")
    state = sdfg.add_state()
    sdfg.add_array("errorcode", [1], dace.int32)
    abort = comm.Abort("abort")
    state.add_edge(state.add_read("errorcode"), None, abort, "_errorcode", Memlet("errorcode[0]"))
    code = abort.implementations["MPI"].expansion(abort, state, sdfg).code.as_string
    assert "MPI_Abort(MPI_COMM_WORLD, int(_errorcode));" in code


if __name__ == "__main__":
    test_comm_rank_and_size()
    test_comm_rank_of_an_external_communicator()
    test_comm_split_of_a_fortran_handle()
    test_gatherv_with_counts_read_at_run_time()
    test_gatherv_at_explicit_displacements()
    test_counts_are_cpp_expressions()
    test_a_column_major_buffer_moves_in_place_in_memory_order("row_major", True)
    test_a_column_major_buffer_moves_in_place_in_memory_order("memory", False)
    test_a_nonblocking_collective_takes_a_column_major_buffer_in_memory_order()
    test_a_column_major_allreduce_in_memory_order()
    test_abort_over_the_world()
