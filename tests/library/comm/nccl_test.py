# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NCCL expansions of the communication library nodes.

The expansion tests need neither a GPU nor NCCL. The runtime tests run under ``mpirun`` with one GPU
per rank, which each rank must see as its device 0, e.g., with Open MPI:
``mpirun -n 2 sh -c 'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK pytest --with-mpi -m "gpu and mpi" ...'``.
"""

import numpy as np
import pytest

import dace
from dace.libraries import comm
from dace.memlet import Memlet
from dace.sdfg import nodes

GPU = dace.StorageType.GPU_Global

###############################################################################
# SDFG builders


def _add_array(sdfg: dace.SDFG, name: str, shape, dtype, storage=GPU) -> None:
    sdfg.add_array(name, shape, dtype, storage=storage)


def _add_int(sdfg: dace.SDFG, name: str) -> None:
    """A rank or tag. A scalar, like the frontend creates: a host array next to GPU buffers would make the
    schedule of the node ambiguous."""
    sdfg.add_scalar(name, dace.int32)


def _wire(state: dace.SDFGState, node: nodes.LibraryNode, inputs: dict, outputs: dict) -> None:
    for conn, (name, subset) in inputs.items():
        state.add_edge(state.add_read(name), None, node, conn, Memlet(f"{name}[{subset}]"))
    for conn, (name, subset) in outputs.items():
        state.add_edge(node, conn, state.add_write(name), None, Memlet(f"{name}[{subset}]"))


def make_allreduce(dtype=dace.float64, op="MPI_SUM", storage=GPU, in_place=False) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_allreduce")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["n"], dtype, storage)
    out = "inbuf" if in_place else "outbuf"
    if not in_place:
        _add_array(sdfg, "outbuf", ["n"], dtype, storage)
    node = comm.Allreduce("allreduce", op)
    _wire(state, node, {"_inbuffer": ("inbuf", "0:n")}, {"_outbuffer": (out, "0:n")})
    return sdfg


def make_reduce(dtype=dace.float64, op="MPI_SUM") -> dace.SDFG:
    sdfg = dace.SDFG("nccl_reduce")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["n"], dtype)
    _add_array(sdfg, "outbuf", ["n"], dtype)
    _add_int(sdfg, "root")
    node = comm.Reduce("reduce", op)
    _wire(state, node, {"_inbuffer": ("inbuf", "0:n"), "_root": ("root", "0")}, {"_outbuffer": ("outbuf", "0:n")})
    return sdfg


def make_bcast(dtype=dace.float64) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_bcast")
    state = sdfg.add_state()
    _add_array(sdfg, "buf", ["n"], dtype)
    _add_int(sdfg, "root")
    node = comm.Bcast("bcast")
    _wire(state, node, {"_inbuffer": ("buf", "0:n"), "_root": ("root", "0")}, {"_outbuffer": ("buf", "0:n")})
    return sdfg


def make_allgather(dtype=dace.float64) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_allgather")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["n"], dtype)
    _add_array(sdfg, "outbuf", ["m"], dtype)
    node = comm.Allgather("allgather")
    _wire(state, node, {"_inbuffer": ("inbuf", "0:n")}, {"_outbuffer": ("outbuf", "0:m")})
    return sdfg


def make_alltoall(dtype=dace.float64) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_alltoall")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["n"], dtype)
    _add_array(sdfg, "outbuf", ["n"], dtype)
    node = comm.Alltoall("alltoall")
    _wire(state, node, {"_inbuffer": ("inbuf", "0:n")}, {"_outbuffer": ("outbuf", "0:n")})
    return sdfg


def make_scatter(dtype=dace.float64) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_scatter")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["n"], dtype)
    _add_array(sdfg, "outbuf", ["m"], dtype)
    _add_int(sdfg, "root")
    node = comm.Scatter("scatter")
    _wire(state, node, {"_inbuffer": ("inbuf", "0:n"), "_root": ("root", "0")}, {"_outbuffer": ("outbuf", "0:m")})
    return sdfg


def make_gather(dtype=dace.float64) -> dace.SDFG:
    sdfg = dace.SDFG("nccl_gather")
    state = sdfg.add_state()
    _add_array(sdfg, "inbuf", ["m"], dtype)
    _add_array(sdfg, "outbuf", ["n"], dtype)
    _add_int(sdfg, "root")
    node = comm.Gather("gather")
    _wire(state, node, {"_inbuffer": ("inbuf", "0:m"), "_root": ("root", "0")}, {"_outbuffer": ("outbuf", "0:n")})
    return sdfg


def make_send_recv(dtype=dace.float64, columns: str = "0:n") -> dace.SDFG:
    """Rank ``src`` sends ``x[:, columns]`` to rank ``dst``, which receives it into ``y[:, columns]``."""
    sdfg = dace.SDFG("nccl_send_recv")
    sdfg.add_symbol("rank", dace.int32)
    start = sdfg.add_state(is_start_block=True)
    send_state = sdfg.add_state()
    recv_state = sdfg.add_state()
    sdfg.add_array("x", [4, "n"], dtype, storage=GPU)
    sdfg.add_array("y", [4, "n"], dtype, storage=GPU)
    for name in ("src", "dst", "tag"):
        _add_int(sdfg, name)
    _wire(
        send_state,
        comm.Send("send"),
        {"_buffer": ("x", f"0:4, {columns}"), "_dest": ("dst", "0"), "_tag": ("tag", "0")},
        {},
    )
    _wire(
        recv_state,
        comm.Recv("recv"),
        {"_src": ("src", "0"), "_tag": ("tag", "0")},
        {"_buffer": ("y", f"0:4, {columns}")},
    )
    sdfg.add_edge(start, send_state, dace.InterstateEdge("rank == src"))
    sdfg.add_edge(start, recv_state, dace.InterstateEdge("rank == dst"))
    end = sdfg.add_state()
    sdfg.add_edge(start, end, dace.InterstateEdge("rank != src and rank != dst"))
    sdfg.add_edge(send_state, end, dace.InterstateEdge())
    sdfg.add_edge(recv_state, end, dace.InterstateEdge())
    return sdfg


def make_ring_exchange(dtype=dace.float64) -> dace.SDFG:
    """Every rank sends ``x`` to ``dst`` and receives ``y`` from ``src`` with Isend/Irecv/Waitall."""
    sdfg = dace.SDFG("nccl_ring_exchange")
    state = sdfg.add_state()
    _add_array(sdfg, "x", ["n"], dtype)
    _add_array(sdfg, "y", ["n"], dtype)
    for name in ("src", "dst", "tag"):
        _add_int(sdfg, name)
    sdfg.add_transient("req", [2], dace.dtypes.opaque("MPI_Request"))
    isend, irecv = comm.Isend("isend"), comm.Irecv("irecv")
    _wire(state, isend, {"_buffer": ("x", "0:n"), "_dest": ("dst", "0"), "_tag": ("tag", "0")}, {})
    _wire(state, irecv, {"_src": ("src", "0"), "_tag": ("tag", "0")}, {"_buffer": ("y", "0:n")})
    req = state.add_access("req")
    state.add_edge(isend, "_request", req, None, Memlet("req[0]"))
    state.add_edge(irecv, "_request", req, None, Memlet("req[1]"))
    waitall = comm.Waitall("waitall")
    state.add_edge(req, None, waitall, "_request", Memlet("req[0:2]"))
    return sdfg


def make_barrier() -> dace.SDFG:
    sdfg = dace.SDFG("nccl_barrier")
    sdfg.add_state().add_node(comm.Barrier("barrier"))
    return sdfg


def use_nccl(sdfg: dace.SDFG) -> dace.SDFG:
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, comm.nodes.node.CommNode):
            node.implementation = "NCCL"
    return sdfg


def expanded_code(sdfg: dace.SDFG) -> str:
    use_nccl(sdfg).expand_library_nodes()
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))


###############################################################################
# Expansion tests


def test_portable_nodes_follow_the_configured_implementation():
    for node_type in (
        comm.Barrier,
        comm.Send,
        comm.Recv,
        comm.Isend,
        comm.Irecv,
        comm.Wait,
        comm.Waitall,
        comm.Bcast,
        comm.Reduce,
        comm.Allreduce,
        comm.Allgather,
        comm.Alltoall,
        comm.Scatter,
        comm.Gather,
    ):
        assert set(node_type.implementations) == {"MPI", "NCCL"}, node_type
        assert node_type.default_implementation is None, node_type
    # Nodes built on MPI datatypes or MPI state keep MPI regardless of the configuration
    for node_type in (comm.Dummy, comm.Redistribute, comm.BlockScatter, comm.BlockGather):
        assert node_type.default_implementation == "MPI", node_type


def test_allreduce_expansion():
    code = expanded_code(make_allreduce())
    assert "ncclAllReduce(_inbuffer, _outbuffer, n, ncclFloat64, ncclSum" in code
    assert "__state->nccl->Get(MPI_COMM_WORLD)" in code
    assert "__state->nccl->Begin();" in code and "__state->nccl->Launch();" in code
    assert "__dace_current_stream" in code


def test_barrier_expansion():
    sdfg = make_barrier()
    code = expanded_code(sdfg)
    assert "ncclAllReduce(_inbuffer, _outbuffer, 1, ncclUint8, ncclSum" in code
    buffer = sdfg.arrays["barrier_buffer"]
    assert buffer.storage == GPU and buffer.transient and buffer.total_size == 1


def test_reduction_operators():
    assert "ncclMax" in expanded_code(make_allreduce(op="MPI_MAX"))
    assert "ncclMin" in expanded_code(make_allreduce(op="Min"))
    assert "ncclProd" in expanded_code(make_allreduce(op="MPI_PROD"))
    with pytest.raises(NotImplementedError, match="reduction operator"):
        expanded_code(make_allreduce(dtype=dace.int32, op="MPI_LAND"))


def test_complex_sum_reduces_components():
    code = expanded_code(make_allreduce(dtype=dace.complex128))
    assert "(n) * 2, ncclFloat64, ncclSum" in code
    with pytest.raises(NotImplementedError, match="cannot reduce"):
        expanded_code(make_allreduce(dtype=dace.complex64, op="MPI_MAX"))


def test_untyped_data_moves_as_bytes():
    code = expanded_code(make_bcast(dtype=dace.int16))
    assert "ncclBroadcast(_inbuffer, _outbuffer, (n) * 2, ncclUint8, int(_root)" in code


def test_host_buffers_are_rejected():
    with pytest.raises(ValueError, match="GPU global memory"):
        expanded_code(make_allreduce(storage=dace.StorageType.CPU_Heap))


def test_collective_expansions():
    assert "ncclReduce(_inbuffer, _outbuffer, n, ncclFloat64, ncclSum, int(_root)" in expanded_code(make_reduce())
    assert "ncclAllGather(_inbuffer, _outbuffer, n, ncclFloat32" in expanded_code(make_allgather(dace.float32))
    alltoall = expanded_code(make_alltoall())
    assert "dace::comm::AlltoAll(_inbuffer, _outbuffer, __chunk, ncclFloat64, __comm" in alltoall
    assert "const long long __chunk = (n) / __size;" in alltoall
    scatter, gather = expanded_code(make_scatter()), expanded_code(make_gather())
    assert "dace::comm::Scatter(_inbuffer, _outbuffer, m, ncclFloat64, int(_root), __comm" in scatter
    assert "dace::comm::Gather(_inbuffer, _outbuffer, m, ncclFloat64, int(_root), __comm" in gather
    assert "__state->nccl->Launch();" in alltoall


def test_point_to_point_expansions():
    code = expanded_code(make_send_recv())
    assert "ncclSend(_buffer, (4 * n), ncclFloat64, int(_dest)" in code
    assert "ncclRecv(_buffer, (4 * n), ncclFloat64, int(_src)" in code
    assert code.count("__state->nccl->Launch();") == 2


def test_strided_point_to_point_moves_rows():
    code = expanded_code(make_send_recv(columns="1:3"))
    assert "for (long long __row = 0; __row < 4; ++__row)" in code
    assert "ncclSend(_buffer + __row * (n), 2, ncclFloat64" in code


def test_nonblocking_operations_wait_for_completion():
    code = expanded_code(make_ring_exchange())
    # Isend and Irecv join the open group and leave it to Waitall
    assert code.count("__state->nccl->Track(__dace_current_stream);") == 2
    assert "__state->nccl->Launch();" not in code
    assert code.count("*_request = MPI_REQUEST_NULL;") == 2
    assert "__state->nccl->Wait();" in code


def test_process_grid_communicator_is_created_at_initialization():
    sdfg = make_allreduce()
    grid = sdfg.add_pgrid([1])
    state = sdfg.start_state
    for node in list(state.nodes()):
        if isinstance(node, comm.Allreduce):
            node.add_in_connector("_grid")
            state.add_edge(state.add_read(grid), None, node, "_grid", Memlet(data=grid))
    # A second node on the same grid must not create its communicator twice
    second = comm.Allreduce("allreduce2")
    second.add_in_connector("_grid")
    _wire(state, second, {"_inbuffer": ("outbuf", "0:n")}, {"_outbuffer": ("inbuf", "0:n")})
    state.add_edge(state.add_read(grid), None, second, "_grid", Memlet(data=grid))

    code = expanded_code(sdfg)
    assert code.count("__state->nccl->Get(_grid)") == 2
    init = sdfg.init_code["frame"].as_string
    assert init.count(f"__state->nccl->Create(__state->{grid});") == 1
    # The NCCL communicator is created from the MPI one, so it comes after the grid's own initialization
    assert init.index(f"__state->nccl->Create(__state->{grid});") > init.index(f"&__state->{grid})")


def test_fortran_communicator_is_rejected():
    sdfg = make_bcast()
    next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, comm.Bcast)).fcomm = "fcomm"
    with pytest.raises(NotImplementedError, match="Fortran"):
        expanded_code(sdfg)


def test_reduction_operator_names():
    assert comm.reduction_op("MPI_SUM") == comm.reduction_op("Sum") == comm.reduction_op("sum") == "sum"
    assert comm.mpi_op("Product") == "MPI_PROD"
    assert comm.mpi_op("MPI_MAXLOC") == "MPI_MAXLOC"
    assert comm.nccl_op("Max") == "ncclMax"
    with pytest.raises(ValueError):
        comm.reduction_op("MPI_SUMM")


def test_deprecated_mpi_library_alias():
    with pytest.warns(DeprecationWarning):
        import dace.libraries.mpi  # noqa: F401 (the import is the test; it warns on first import only)
    from dace.libraries.mpi.nodes.send import Send

    assert Send is comm.Send


###############################################################################
# Runtime tests


def _setup_ranks():
    """The MPI world, after skipping unless NCCL is installed and every rank has its own GPU."""
    from mpi4py import MPI

    cp = pytest.importorskip("cupy")
    if not comm.NCCL.is_installed():
        pytest.skip("NCCL is not installed")
    world = MPI.COMM_WORLD
    # DaCe runs on device 0 of each process, so the ranks must see different GPUs as their device 0
    if len(set(world.allgather(cp.cuda.Device(0).pci_bus_id))) < world.Get_size():
        pytest.skip("NCCL needs one GPU per rank: give each rank its own through CUDA_VISIBLE_DEVICES")
    return world, cp


def _compile(sdfg: dace.SDFG, world):
    from dace.sdfg.utils import distributed_compile

    return distributed_compile(use_nccl(sdfg), world)


@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("in_place", [False, True])
def test_allreduce(in_place):
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()
    func = _compile(make_allreduce(in_place=in_place), world)
    inbuf = cp.full(16, rank + 1, dtype=np.float64)
    if in_place:
        func(inbuf=inbuf, n=16)
        result = inbuf
    else:
        result = cp.zeros(16, dtype=np.float64)
        func(inbuf=inbuf, outbuf=result, n=16)
    assert np.allclose(result.get(), size * (size + 1) / 2)


@pytest.mark.gpu
@pytest.mark.mpi
def test_bcast_and_reduce():
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()
    root = size - 1

    buf = cp.full(8, rank, dtype=np.float32)
    _compile(make_bcast(dace.float32), world)(buf=buf, root=root, n=8)
    assert np.allclose(buf.get(), size - 1)

    inbuf = cp.full(8, rank + 1, dtype=np.int64)
    outbuf = cp.zeros(8, dtype=np.int64)
    _compile(make_reduce(dace.int64, "MPI_MAX"), world)(inbuf=inbuf, outbuf=outbuf, root=root, n=8)
    if rank == size - 1:
        assert np.array_equal(outbuf.get(), np.full(8, size))


@pytest.mark.gpu
@pytest.mark.mpi
def test_allgather_and_alltoall():
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()

    inbuf = cp.full(4, rank, dtype=np.float64)
    outbuf = cp.zeros(4 * size, dtype=np.float64)
    _compile(make_allgather(), world)(inbuf=inbuf, outbuf=outbuf, n=4, m=4 * size)
    assert np.array_equal(outbuf.get(), np.repeat(np.arange(size, dtype=np.float64), 4))

    # Chunk p of rank r holds 100 * r + p, so after the exchange chunk p of rank r holds 100 * p + r
    inbuf = cp.asarray(np.repeat(100 * rank + np.arange(size, dtype=np.float64), 3))
    outbuf = cp.zeros(3 * size, dtype=np.float64)
    _compile(make_alltoall(), world)(inbuf=inbuf, outbuf=outbuf, n=3 * size)
    assert np.array_equal(outbuf.get(), np.repeat(100 * np.arange(size, dtype=np.float64) + rank, 3))


@pytest.mark.gpu
@pytest.mark.mpi
def test_scatter_and_gather():
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()
    root = 0
    full = np.arange(5 * size, dtype=np.float64)

    inbuf = cp.asarray(full) if rank == 0 else cp.zeros(5 * size, dtype=np.float64)
    part = cp.zeros(5, dtype=np.float64)
    _compile(make_scatter(), world)(inbuf=inbuf, outbuf=part, root=root, n=5 * size, m=5)
    assert np.array_equal(part.get(), full[5 * rank : 5 * (rank + 1)])

    gathered = cp.zeros(5 * size, dtype=np.float64)
    _compile(make_gather(), world)(inbuf=part, outbuf=gathered, root=root, n=5 * size, m=5)
    if rank == 0:
        assert np.array_equal(gathered.get(), full)


@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("columns", ["0:n", "1:3"])
def test_send_recv(columns):
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()
    if size < 2:
        pytest.skip("A blocking send needs a receiving rank other than the sender")
    func = _compile(make_send_recv(columns=columns), world)
    x = cp.asarray(np.arange(4 * 5, dtype=np.float64).reshape(4, 5) + 100 * rank)
    y = cp.zeros((4, 5), dtype=np.float64)
    src, dst, tag = 0, 1, 7
    func(x=x, y=y, src=src, dst=dst, tag=tag, rank=rank, n=5)
    if rank == 1:
        expected = np.zeros((4, 5))
        cols = slice(0, 5) if columns == "0:n" else slice(1, 3)
        expected[:, cols] = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)[:, cols]
        assert np.array_equal(y.get(), expected)


@pytest.mark.gpu
@pytest.mark.mpi
def test_ring_exchange():
    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()
    func = _compile(make_ring_exchange(), world)
    x = cp.full(32, rank, dtype=np.float64)
    y = cp.zeros(32, dtype=np.float64)
    src, dst, tag = (rank - 1) % size, (rank + 1) % size, 0
    func(x=x, y=y, src=src, dst=dst, tag=tag, n=32)
    assert np.allclose(y.get(), (rank - 1) % size)


@pytest.mark.gpu
@pytest.mark.mpi
def test_barrier():
    world, _ = _setup_ranks()
    _compile(make_barrier(), world)()


@pytest.mark.gpu
@pytest.mark.mpi
def test_frontend_allreduce_with_configured_nccl():
    from mpi4py import MPI

    world, cp = _setup_ranks()
    rank, size = world.Get_rank(), world.Get_size()

    @dace.program
    def allreduce_on_gpu(buf: dace.float64[10] @ GPU):
        dace.comm.Allreduce(MPI.IN_PLACE, buf, MPI.SUM)

    buf = cp.full(10, rank + 1, dtype=np.float64)
    with dace.config.set_temporary("library", "comm", "default_implementation", value="NCCL"):
        sdfg = allreduce_on_gpu.to_sdfg(buf)
        func = _compile(sdfg, world)
    func(buf=buf)
    assert np.allclose(buf.get(), size * (size + 1) / 2)


###############################################################################

if __name__ == "__main__":
    test_portable_nodes_follow_the_configured_implementation()
    test_allreduce_expansion()
    test_barrier_expansion()
    test_reduction_operators()
    test_complex_sum_reduces_components()
    test_untyped_data_moves_as_bytes()
    test_host_buffers_are_rejected()
    test_collective_expansions()
    test_point_to_point_expansions()
    test_strided_point_to_point_moves_rows()
    test_nonblocking_operations_wait_for_completion()
    test_process_grid_communicator_is_created_at_initialization()
    test_fortran_communicator_is_rejected()
    test_reduction_operator_names()
    test_deprecated_mpi_library_alias()
    test_allreduce(False)
    test_allreduce(True)
    test_bcast_and_reduce()
    test_allgather_and_alltoall()
    test_scatter_and_gather()
    test_send_recv("0:n")
    test_send_recv("1:3")
    test_ring_exchange()
    test_barrier()
    test_frontend_allreduce_with_configured_nccl()
