# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Buffers of communication nodes: element order from the layout of the array, scalars, outputs written on the root
only, derived datatypes and request statuses.

The tests marked ``mpi`` need two processes::

    mpirun -n 2 python -m pytest --with-mpi -m mpi tests/library/comm/mpi_buffers_test.py
"""

import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.libraries import comm
from dace.libraries.comm import utils
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


def _code(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def _packed(sdfg: dace.SDFG) -> list[dace.data.Array]:
    return [desc for name, desc in sdfg.arrays.items() if "_packed" in name]


def _fortran_allreduce_sdfg(name: str, subset: str = "0:3, 0:4", nonblocking: bool = False) -> dace.SDFG:
    """An in-place Allreduce of ``subset`` of a column-major (Fortran) ``a(3, 4)``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [3, 4], dace.float64, strides=[1, 3])
    state = sdfg.add_state()
    allreduce = comm.Allreduce("allreduce")
    allreduce.implementation = "MPI"
    allreduce.nonblocking = nonblocking
    state.add_edge(state.add_read("a"), None, allreduce, "_inbuffer", Memlet(f"a[{subset}]"))
    state.add_edge(allreduce, "_outbuffer", state.add_write("a"), None, Memlet(f"a[{subset}]"))
    if nonblocking:
        sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"))
        state.add_edge(allreduce, "_request", state.add_write("req"), None, Memlet("req[0]"))
    return sdfg


def _allgather_of_view_sdfg(name: str) -> dace.SDFG:
    """An Allgather of ``xt``, a transposed view of the row-major ``x(4, 3)``."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array("x", [4, 3], dace.float64)
    sdfg.add_view("xt", [3, 4], dace.float64, strides=[1, 3])
    sdfg.add_array("out", [6, 4], dace.float64)
    xt = state.add_access("xt")
    state.add_edge(state.add_read("x"), None, xt, "views", Memlet("x[0:4, 0:3]"))
    node = comm.Allgather("allgather")
    node.implementation = "MPI"
    state.add_edge(xt, None, node, "_inbuffer", Memlet("xt[0:3, 0:4]"))
    state.add_edge(node, "_outbuffer", state.add_write("out"), None, Memlet("out[0:6, 0:4]"))
    return sdfg


def test_a_column_major_buffer_moves_in_place():
    sdfg = _fortran_allreduce_sdfg("fortran_allreduce_in_place")
    sdfg.expand_library_nodes(recursive=False)
    assert not _packed(sdfg)


def test_a_column_major_section_packs_in_column_major_order():
    sdfg = _fortran_allreduce_sdfg("fortran_allreduce_section", subset="0:2, 1:4")
    sdfg.expand_library_nodes(recursive=False)
    (packed,) = _packed(sdfg)
    assert packed.is_packed_fortran_strides() and not packed.is_packed_c_strides()


def test_a_transposed_view_of_a_row_major_array_packs_in_row_major_order():
    sdfg = _allgather_of_view_sdfg("allgather_view_packs")
    sdfg.expand_library_nodes(recursive=False)
    (packed,) = _packed(sdfg)
    assert packed.is_packed_c_strides()


def test_a_nonblocking_collective_takes_a_column_major_buffer():
    sdfg = _fortran_allreduce_sdfg("fortran_iallreduce", nonblocking=True)
    state = sdfg.start_state
    node = next(n for n in state.nodes() if isinstance(n, comm.Allreduce))
    node.validate_nonblocking(sdfg, state)


def _iallreduce_scalar_sdfg(name: str, by_reference: bool, nonblocking: bool = True) -> dace.SDFG:
    """An Allreduce of the scalar ``s`` in place (nonblocking: with its Wait in the next state), whose buffer
    connectors are pointers if ``by_reference``, else left to inference (copies)."""
    sdfg = dace.SDFG(name)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("value", [1], dace.float64)
    sdfg.add_array("result", [1], dace.float64)
    start = sdfg.add_state("start")
    s = start.add_access("s")
    start.add_nedge(start.add_read("value"), s, Memlet("value[0] -> s[0]"))
    node = comm.Allreduce("allreduce")
    node.implementation = "MPI"
    if by_reference:
        node.in_connectors["_inbuffer"] = dace.pointer(dace.float64)
        node.out_connectors["_outbuffer"] = dace.pointer(dace.float64)
    start.add_edge(s, None, node, "_inbuffer", Memlet("s"))
    start.add_edge(node, "_outbuffer", start.add_write("s"), None, Memlet("s"))
    last = start
    if nonblocking:
        node.nonblocking = True
        node.add_out_connector("_request")
        sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
        start.add_edge(node, "_request", start.add_write("req"), None, Memlet("req[0]"))
        last = sdfg.add_state_after(start, "wait")
        wait = comm.Wait("wait")
        wait.implementation = "MPI"
        wait.remove_out_connector("_stat_tag")
        wait.remove_out_connector("_stat_source")
        last.add_edge(last.add_read("req"), None, wait, "_request", Memlet("req[0]"))
    done = sdfg.add_state_after(last, "done")
    done.add_nedge(done.add_read("s"), done.add_write("result"), Memlet("s[0] -> result[0]"))
    return sdfg


@pytest.mark.parametrize("by_reference", [False, True])
def test_a_buffer_is_passed_by_the_type_of_its_connector(by_reference: bool):
    """A pointer connector is passed as is; a copy in the tasklet (a value connector) by its address."""
    code = _code(_iallreduce_scalar_sdfg(f"allreduce_scalar_ref_{by_reference}", by_reference, nonblocking=False))
    outbuffer = "_outbuffer" if by_reference else "&_outbuffer"
    assert f"MPI_Allreduce(MPI_IN_PLACE, {outbuffer}, 1, MPI_DOUBLE" in code
    assert ("_outbuffer = _inbuffer;" in code) != by_reference  # Copies: the output starts from the input


def test_a_nonblocking_collective_refuses_a_copied_buffer():
    sdfg = _iallreduce_scalar_sdfg("iallreduce_scalar_copy", by_reference=False)
    with pytest.raises(ValueError, match="copies"):
        sdfg.expand_library_nodes()


@pytest.mark.mpi
def test_a_nonblocking_allreduce_of_a_scalar_by_reference():
    world = _world()
    result = np.zeros(1)
    sdfg = _iallreduce_scalar_sdfg("mpi_iallreduce_scalar", by_reference=True)
    _compile_in_turn(sdfg, world)(value=np.array([world.Get_rank() + 1.0]), result=result)
    assert result[0] == sum(r + 1 for r in range(world.Get_size()))


@pytest.mark.mpi
def test_an_allreduce_of_one_array_element():
    """A single element is a copy in the tasklet, as for a scalar."""
    world = _world()
    sdfg = dace.SDFG("mpi_allreduce_element")
    sdfg.add_array("x", [3], dace.float64)
    state = sdfg.add_state()
    node = comm.Allreduce("allreduce")
    node.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, node, "_inbuffer", Memlet("x[1]"))
    state.add_edge(node, "_outbuffer", state.add_write("x"), None, Memlet("x[1]"))
    x = np.array([0.0, world.Get_rank() + 1.0, 0.0])
    _compile_in_turn(sdfg, world)(x=x)
    assert np.array_equal(x, [0.0, sum(r + 1 for r in range(world.Get_size())), 0.0])


def _gather_sdfg(name: str) -> dace.SDFG:
    """A Gather of ``x(2)`` into the column ``out[:, 1]`` of ``out(4, 2)``, which is strided."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [2], dace.float64)
    sdfg.add_array("out", [4, 2], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    state = sdfg.add_state()
    gather = comm.Gather("gather")
    gather.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, gather, "_inbuffer", Memlet("x[0:2]"))
    state.add_edge(state.add_read("root"), None, gather, "_root", Memlet("root[0]"))
    state.add_edge(gather, "_outbuffer", state.add_write("out"), None, Memlet("out[0:4, 1]"))
    return sdfg


@pytest.mark.mpi
def test_a_gather_into_a_strided_buffer():
    world = _world()
    if world.Get_size() != 2:
        pytest.skip("Gathers two elements from each of two processes")
    rank = world.Get_rank()
    x = np.full(2, rank + 1, np.float64)
    out = np.full((4, 2), -1.0)
    _compile_in_turn(_gather_sdfg("mpi_gather_strided"), world)(x=x, out=out, root=np.zeros(1, np.int32))
    if rank == 0:  # The output is undefined off the root
        assert np.array_equal(out, [[-1, 1], [-1, 1], [-1, 2], [-1, 2]])


@pytest.mark.mpi
def test_a_reduce_into_a_strided_buffer():
    world = _world()
    rank = world.Get_rank()
    sdfg = dace.SDFG("mpi_reduce_strided")
    sdfg.add_array("x", [4], dace.float64)
    sdfg.add_array("out", [4, 2], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    state = sdfg.add_state()
    reduce = comm.Reduce("reduce")
    reduce.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, reduce, "_inbuffer", Memlet("x[0:4]"))
    state.add_edge(state.add_read("root"), None, reduce, "_root", Memlet("root[0]"))
    state.add_edge(reduce, "_outbuffer", state.add_write("out"), None, Memlet("out[0:4, 1]"))
    out = np.full((4, 2), -1.0)
    _compile_in_turn(sdfg, world)(x=np.full(4, rank + 1.0), out=out, root=np.zeros(1, np.int32))
    if rank == 0:  # The output is undefined off the root
        expected = np.full((4, 2), -1.0)
        expected[:, 1] = sum(r + 1 for r in range(world.Get_size()))
        assert np.array_equal(out, expected)


def test_every_comm_node_has_side_effects():
    library_nodes = [
        cls
        for cls in vars(comm.nodes).values()
        if isinstance(cls, type) and issubclass(cls, dace.nodes.LibraryNode) and cls is not comm.nodes.node.CommNode
    ]
    assert library_nodes
    for cls in library_nodes:
        assert issubclass(cls, comm.nodes.node.CommNode), cls.__name__


_BUFFER_NODES = [comm.Reduce, comm.Gather, comm.Scatter, comm.Allreduce, comm.Bcast, comm.Allgather, comm.Alltoall]


@pytest.mark.parametrize("node_type", _BUFFER_NODES)
@pytest.mark.parametrize("implementation", ["MPI", "NCCL"])
def test_comm_tasklets_have_side_effects(node_type, implementation: str):
    """A process whose output no one reads still takes part: dead-dataflow elimination must keep the call."""
    name = f"side_effects_{node_type.__name__.lower()}_{implementation.lower()}"
    storage = dace.StorageType.GPU_Global if implementation == "NCCL" else dace.StorageType.Default
    node = node_type(node_type.__name__.lower())
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [4], dace.float64, storage=storage)
    sdfg.add_array("y", [8], dace.float64, storage=storage)
    sdfg.add_array("root", [1], dace.int32)
    state = sdfg.add_state()
    state.add_edge(state.add_read("x"), None, node, "_inbuffer", Memlet("x[0:4]"))
    if "_root" in node.in_connectors:
        state.add_edge(state.add_read("root"), None, node, "_root", Memlet("root[0]"))
    out = "x" if node_type is comm.Bcast else "y"  # Bcast is in place
    state.add_edge(node, "_outbuffer", state.add_write(out), None, Memlet(f"{out}[0:4]"))
    expansion = node.implementations[implementation].expansion(node, state, sdfg)
    assert expansion.side_effects


def _scalar_reduction_sdfg(name: str, node: dace.nodes.LibraryNode, in_place: bool) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_scalar("s", dace.float64)
    sdfg.add_array("result", [1], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    state = sdfg.add_state()
    out = "s" if in_place else "t"
    state.add_edge(state.add_read("s"), None, node, "_inbuffer", Memlet("s"))
    if isinstance(node, comm.Reduce):
        sdfg.add_array("root", [1], dace.int32)
        state.add_edge(state.add_read("root"), None, node, "_root", Memlet("root[0]"))
    written = state.add_access(out)
    state.add_edge(node, "_outbuffer", written, None, Memlet(out))
    state.add_nedge(written, state.add_write("result"), Memlet(f"{out}[0] -> result[0]"))
    return sdfg


@pytest.mark.mpi
@pytest.mark.parametrize("op, in_place", [("MPI_SUM", False), ("MPI_SUM", True), ("avg", False)])
def test_an_allreduce_of_a_scalar(op: str, in_place: bool):
    world = _world()
    node = comm.Allreduce("allreduce", op=op)
    node.implementation = "MPI"
    sdfg = _scalar_reduction_sdfg(f"mpi_allreduce_scalar_{op.lower()}_{in_place}", node, in_place)
    result = np.zeros(1)
    _compile_in_turn(sdfg, world)(s=float(world.Get_rank() + 1), result=result)
    total = sum(r + 1 for r in range(world.Get_size()))
    assert result[0] == (total / world.Get_size() if op == "avg" else total)


@pytest.mark.mpi
def test_a_reduce_of_a_scalar():
    world = _world()
    node = comm.Reduce("reduce")
    node.implementation = "MPI"
    sdfg = _scalar_reduction_sdfg("mpi_reduce_scalar", node, in_place=False)
    result = np.zeros(1)
    _compile_in_turn(sdfg, world)(s=float(world.Get_rank() + 1), result=result, root=np.zeros(1, np.int32))
    if world.Get_rank() == 0:
        assert result[0] == sum(r + 1 for r in range(world.Get_size()))


def test_an_mpi_communicator_is_refused_by_an_nccl_expansion():
    sdfg = dace.SDFG("nccl_takes_mpi_handle")
    sdfg.add_scalar("color", dace.int32)
    sdfg.add_scalar("key", dace.int32)
    sdfg.add_scalar("handle", dace.int64, transient=True)
    sdfg.add_array("x", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    split = comm.CommSplit("split")
    state.add_edge(state.add_read("color"), None, split, "_color", Memlet("color"))
    state.add_edge(state.add_read("key"), None, split, "_key", Memlet("key"))
    handle = state.add_access("handle")
    state.add_edge(split, "_newcomm", handle, None, Memlet("handle"))
    allreduce = comm.Allreduce("allreduce")
    allreduce.add_in_connector("_comm")
    state.add_edge(handle, None, allreduce, "_comm", Memlet("handle"))
    state.add_edge(state.add_read("x"), None, allreduce, "_inbuffer", Memlet("x[0:4]"))
    state.add_edge(allreduce, "_outbuffer", state.add_write("x"), None, Memlet("x[0:4]"))
    with pytest.raises(ValueError, match="MPI communicator"):
        allreduce.implementations["NCCL"].expansion(allreduce, state, sdfg)


def _isend_sdfg(name: str, storage: dace.StorageType) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [4], dace.float64, storage=storage)
    sdfg.add_array("dest", [1], dace.int32)
    sdfg.add_array("tag", [1], dace.int32)
    sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"))
    state = sdfg.add_state()
    isend = comm.Isend("isend")
    isend.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, isend, "_buffer", Memlet("x[0:4]"))
    state.add_edge(state.add_read("dest"), None, isend, "_dest", Memlet("dest[0]"))
    state.add_edge(state.add_read("tag"), None, isend, "_tag", Memlet("tag[0]"))
    state.add_edge(isend, "_request", state.add_write("req"), None, Memlet("req[0]"))
    return sdfg


def test_an_isend_of_gpu_memory_synchronizes_the_stream_of_the_backend():
    code = _code(_isend_sdfg("isend_gpu_sync", dace.StorageType.GPU_Global))
    backend = common.get_gpu_backend()
    assert f"{backend}StreamSynchronize(__dace_current_stream)" in code
    assert {"cuda": "hip", "hip": "cuda"}[backend] + "StreamSynchronize" not in code


@pytest.mark.parametrize("node_type", [comm.Allreduce, comm.Reduce, comm.Bcast, comm.Gather, comm.Allgather])
def test_an_mpi_call_on_gpu_memory_waits_for_the_stream(node_type):
    """GPU-aware MPI reads and writes device memory from the host: the kernels before it must have finished."""
    sdfg = dace.SDFG(f"mpi_gpu_sync_{node_type.__name__.lower()}")
    sdfg.add_array("x", [4], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("y", [8], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("root", [1], dace.int32)
    state = sdfg.add_state()
    node = node_type(node_type.__name__.lower())
    node.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, node, "_inbuffer", Memlet("x[0:4]"))
    if "_root" in node.in_connectors:
        state.add_edge(state.add_read("root"), None, node, "_root", Memlet("root[0]"))
    out = "x" if node_type is comm.Bcast else "y"
    state.add_edge(node, "_outbuffer", state.add_write(out), None, Memlet(f"{out}[0:4]"))
    code = _code(sdfg)
    sync = code.index(f"{common.get_gpu_backend()}StreamSynchronize(__dace_current_stream)")
    assert sync < code.index("MPI_")


def test_an_isend_without_sync_does_not_wait_for_the_stream():
    sdfg = _isend_sdfg("isend_gpu_nosync", dace.StorageType.GPU_Global)
    next(n for n in sdfg.start_state.nodes() if isinstance(n, comm.Isend)).nosync = True
    assert "StreamSynchronize" not in _code(sdfg)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_an_allreduce_of_complex_numbers(dtype):
    world = _world()
    name = np.dtype(dtype).name
    sdfg = dace.SDFG(f"mpi_allreduce_{name}")
    sdfg.add_array("x", [4], dace.dtypes.dtype_to_typeclass(dtype))
    state = sdfg.add_state()
    node = comm.Allreduce("allreduce")
    node.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, node, "_inbuffer", Memlet("x[0:4]"))
    state.add_edge(node, "_outbuffer", state.add_write("x"), None, Memlet("x[0:4]"))
    x = np.full(4, complex(world.Get_rank() + 1, 1), dtype=dtype)
    _compile_in_turn(sdfg, world)(x=x)
    size = world.Get_size()
    assert np.allclose(x, complex(sum(r + 1 for r in range(size)), size))


def _send_recv_sdfg(name: str, strides: list, sent: str, received: str) -> dace.SDFG:
    """Rank 0 sends ``sent`` of ``a(4, 4)`` (with ``strides``) to rank 1, which receives it into ``received``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [4, 4], dace.float64, strides=strides)
    sdfg.add_array("b", [4, 4], dace.float64, strides=strides)
    for scalar in ("peer", "tag"):
        sdfg.add_array(scalar, [1], dace.int32)
    sdfg.add_array("rank", [1], dace.int32)
    init = sdfg.add_state("init")
    send_state = sdfg.add_state("send")
    recv_state = sdfg.add_state("recv")
    done = sdfg.add_state("done")
    sdfg.add_edge(init, send_state, dace.InterstateEdge("rank[0] == 0"))
    sdfg.add_edge(init, recv_state, dace.InterstateEdge("rank[0] != 0"))
    sdfg.add_edge(send_state, done, dace.InterstateEdge())
    sdfg.add_edge(recv_state, done, dace.InterstateEdge())
    send = comm.Send("send")
    send.implementation = "MPI"
    send_state.add_edge(send_state.add_read("a"), None, send, "_buffer", Memlet(f"a[{sent}]"))
    send_state.add_edge(send_state.add_read("peer"), None, send, "_dest", Memlet("peer[0]"))
    send_state.add_edge(send_state.add_read("tag"), None, send, "_tag", Memlet("tag[0]"))
    recv = comm.Recv("recv")
    recv.implementation = "MPI"
    recv_state.add_edge(recv, "_buffer", recv_state.add_write("b"), None, Memlet(f"b[{received}]"))
    recv_state.add_edge(recv_state.add_read("peer"), None, recv, "_src", Memlet("peer[0]"))
    recv_state.add_edge(recv_state.add_read("tag"), None, recv, "_tag", Memlet("tag[0]"))
    return sdfg


def test_a_strided_send_creates_and_frees_its_datatype_on_every_call():
    code = _code(_send_recv_sdfg("send_vector_per_call", [4, 1], "0:2, 1:4", "0:2, 1:4"))
    assert "static" not in code
    assert code.count("MPI_Type_commit(&__vector);") == code.count("MPI_Type_free(&__vector);") == 2


@pytest.mark.parametrize(
    "strides, blocklen, count, stride", [([4, 1], "3", "2", "4"), ([1, 4], "2", "3", "4")], ids=["row", "column"]
)
def test_a_vector_datatype_runs_along_the_unit_stride(strides, blocklen, count, stride):
    desc = dace.data.Array(dace.float64, [4, 4], strides=strides)
    ddt = utils.create_vector_ddt(Memlet("a[0:2, 1:4]"), desc)
    assert (ddt["blocklen"], ddt["count"], ddt["stride"]) == (blocklen, count, stride)


@pytest.mark.mpi
@pytest.mark.parametrize("order", ["C", "F"])
def test_a_strided_section_is_sent_and_received(order: str):
    world = _world()
    rank = world.Get_rank()
    strides = [4, 1] if order == "C" else [1, 4]
    sdfg = _send_recv_sdfg(f"mpi_send_recv_section_{order}", strides, "0:2, 1:4", "2:4, 0:3")
    a = np.array(np.arange(16, dtype=np.float64).reshape(4, 4), order=order)
    b = np.zeros((4, 4), order=order)
    compiled = _compile_in_turn(sdfg, world)
    if rank < 2:
        peer = np.array([1 - rank], np.int32)
        compiled(a=a, b=b, peer=peer, tag=np.array([7], np.int32), rank=np.array([rank], np.int32))
    if rank == 1:
        assert np.array_equal(b[2:4, 0:3], a[0:2, 1:4])


def test_waitall_statuses_are_a_stack_array_of_the_request_count():
    sdfg = dace.SDFG("waitall_statuses")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("req", ["n"], dace.dtypes.opaque("MPI_Request"))
    state = sdfg.add_state()
    waitall = comm.Waitall("waitall")
    waitall.implementation = "MPI"
    state.add_edge(state.add_read("req"), None, waitall, "_request", Memlet("req[0:n]"))
    sdfg.expand_library_nodes(recursive=False)
    code = sdfg.generate_code()[0].clean_code
    assert "MPI_Status _s[n];" in code  # A variable-length array
    assert "MPI_Waitall(n, _request, _s);" in code


def _allreduce_within_sdfg(name: str, out: str) -> dace.SDFG:
    """An Allreduce of ``x[0:4]`` into the subset ``out`` of the same array ``x(8)``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [8], dace.float64)
    state = sdfg.add_state()
    allreduce = comm.Allreduce("allreduce")
    allreduce.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, allreduce, "_inbuffer", Memlet("x[0:4]"))
    state.add_edge(allreduce, "_outbuffer", state.add_write("x"), None, Memlet(f"x[{out}]"))
    return sdfg


def test_an_allreduce_between_disjoint_subsets_is_not_in_place():
    code = _code(_allreduce_within_sdfg("allreduce_disjoint", "4:8"))
    assert "MPI_IN_PLACE" not in code


def test_an_allreduce_between_overlapping_subsets_is_refused():
    with pytest.raises(ValueError, match="overlapping"):
        _code(_allreduce_within_sdfg("allreduce_overlapping", "2:6"))


@pytest.mark.mpi
def test_an_allreduce_into_another_subset_of_its_input_array():
    world = _world()
    rank = world.Get_rank()
    x = np.concatenate([np.full(4, rank + 1.0), np.zeros(4)])
    _compile_in_turn(_allreduce_within_sdfg("mpi_allreduce_disjoint", "4:8"), world)(x=x)
    total = sum(r + 1.0 for r in range(world.Get_size()))
    assert np.array_equal(x, np.concatenate([np.full(4, rank + 1.0), np.full(4, total)]))


@pytest.mark.mpi
def test_a_reduce_scatter_into_one_element():
    world = _world()
    size, rank = world.Get_size(), world.Get_rank()
    sdfg = dace.SDFG("mpi_reduce_scatter_one_element")
    sdfg.add_array("x", [size], dace.float64)
    sdfg.add_array("y", [2], dace.float64)
    state = sdfg.add_state()
    reduce_scatter = comm.ReduceScatter("reduce_scatter")
    reduce_scatter.implementation = "MPI"
    state.add_edge(state.add_read("x"), None, reduce_scatter, "_inbuffer", Memlet(f"x[0:{size}]"))
    state.add_edge(reduce_scatter, "_outbuffer", state.add_write("y"), None, Memlet("y[1]"))
    x = np.arange(size, dtype=np.float64) + rank
    y = np.zeros(2)
    _compile_in_turn(sdfg, world)(x=x, y=y)
    assert np.array_equal(y, [0, sum(rank + r for r in range(size))])


def test_the_overlap_schedule_resolves_implementations_as_expansion_does():
    from types import SimpleNamespace

    from dace.libraries.comm import schedule

    with dace.config.set_temporary("library", "comm", "default_implementation", value="NCCL"):
        assert not schedule._on_gpu_streams(SimpleNamespace(node=comm.CommRank("rank")))  # Only MPI implements it
        assert schedule._on_gpu_streams(SimpleNamespace(node=comm.Allreduce("allreduce")))


if __name__ == "__main__":
    test_a_column_major_buffer_moves_in_place()
    test_a_column_major_section_packs_in_column_major_order()
    test_a_transposed_view_of_a_row_major_array_packs_in_row_major_order()
    test_a_nonblocking_collective_takes_a_column_major_buffer()
    test_a_buffer_is_passed_by_the_type_of_its_connector(False)
    test_a_buffer_is_passed_by_the_type_of_its_connector(True)
    test_a_nonblocking_collective_refuses_a_copied_buffer()
    test_a_nonblocking_allreduce_of_a_scalar_by_reference()
    test_an_allreduce_of_one_array_element()
    test_a_gather_into_a_strided_buffer()
    test_a_reduce_into_a_strided_buffer()
    test_every_comm_node_has_side_effects()
    for implementation in ("MPI", "NCCL"):
        for node_type in _BUFFER_NODES:
            test_comm_tasklets_have_side_effects(node_type, implementation)
    test_an_allreduce_of_a_scalar("MPI_SUM", False)
    test_an_allreduce_of_a_scalar("MPI_SUM", True)
    test_an_allreduce_of_a_scalar("avg", False)
    test_a_reduce_of_a_scalar()
    test_an_mpi_communicator_is_refused_by_an_nccl_expansion()
    test_an_isend_of_gpu_memory_synchronizes_the_stream_of_the_backend()
    for node_type in (comm.Allreduce, comm.Reduce, comm.Bcast, comm.Gather, comm.Allgather):
        test_an_mpi_call_on_gpu_memory_waits_for_the_stream(node_type)
    test_an_isend_without_sync_does_not_wait_for_the_stream()
    test_an_allreduce_of_complex_numbers(np.complex64)
    test_an_allreduce_of_complex_numbers(np.complex128)
    test_a_strided_send_creates_and_frees_its_datatype_on_every_call()
    test_a_vector_datatype_runs_along_the_unit_stride([4, 1], "3", "2", "4")
    test_a_vector_datatype_runs_along_the_unit_stride([1, 4], "2", "3", "4")
    test_a_strided_section_is_sent_and_received("C")
    test_a_strided_section_is_sent_and_received("F")
    test_waitall_statuses_are_a_stack_array_of_the_request_count()
    test_an_allreduce_between_disjoint_subsets_is_not_in_place()
    test_an_allreduce_between_overlapping_subsets_is_refused()
    test_an_allreduce_into_another_subset_of_its_input_array()
    test_a_reduce_scatter_into_one_element()
    test_the_overlap_schedule_resolves_implementations_as_expansion_does()
