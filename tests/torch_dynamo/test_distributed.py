# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``torch.distributed`` collectives and point-to-point operations lower to :mod:`dace.libraries.comm` nodes.

The tests run on CPU tensors, whose communication goes over MPI, so they run under MPI with a ``gloo`` process group
on the same ranks::

    mpirun -n 2 --oversubscribe python -m pytest --with-mpi -m mpi tests/torch_dynamo/test_distributed.py

Expected values are computed locally from the (deterministic) inputs of every rank. The tests are marked ``mpi``
only, not ``torch``: the ``torch`` CI job runs without ``mpirun``.
"""

import os
import tempfile

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as fc  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


def _init_process_group() -> str:
    """Initializes a gloo process group over the MPI ranks (the compiled programs communicate over MPI)."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_size() < 2:
        raise ValueError("This test is supposed to be run with at least two processes")
    path = world.bcast(tempfile.mktemp(prefix="dace_pg_") if world.Get_rank() == 0 else None)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method="file://" + path, rank=world.Get_rank(), world_size=world.Get_size())
    return path


def _destroy_process_group(path: str) -> None:
    rank = dist.get_rank()
    dist.destroy_process_group()
    if rank == 0 and os.path.exists(path):
        os.remove(path)


@pytest.fixture(scope="module", autouse=True)
def process_group():
    path = _init_process_group()
    yield
    _destroy_process_group(path)


def _data(rank: int, *shape: int) -> torch.Tensor:
    """The input of rank ``rank``: distinct per rank and element."""
    return torch.arange(int(torch.Size(shape).numel()), dtype=torch.float32).reshape(shape) * 0.25 + 10 * rank


def _compile(backend, fn):
    return torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)


@pytest.mark.mpi
@pytest.mark.parametrize("op", ["sum", "avg", "max"])
def test_all_reduce_in_place(op):
    """``dist.all_reduce`` on an intermediate, then more computation; compiles once across sizes."""
    reduce_op = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG, "max": dist.ReduceOp.MAX}[op]
    size, rank = dist.get_world_size(), dist.get_rank()

    def f(x):
        y = x * 2
        dist.all_reduce(y, op=reduce_op)
        return y + 1

    backend = DaceBackend()
    compiled = _compile(backend, f)
    for n in (4, 7):
        everyone = torch.stack([_data(r, n, 3) * 2 for r in range(size)])
        reduced = {"sum": everyone.sum(0), "avg": everyone.mean(0), "max": everyone.amax(0)}[op]
        torch.testing.assert_close(compiled(_data(rank, n, 3)), reduced + 1)
    assert backend.compile_count == 1


@pytest.mark.mpi
def test_gather_scatter_broadcast():
    """Functional all-gather, reduce-scatter and broadcast; the all-gather input is a strided view."""
    size, rank = dist.get_world_size(), dist.get_rank()
    group = dist.group.WORLD

    def f(x, y):
        gathered = fc.all_gather_tensor(x.t(), 0, group)
        scattered = fc.reduce_scatter_tensor(y, "sum", 0, group)
        broadcast = fc.broadcast(y * 3, 1, group)
        return gathered, scattered, broadcast

    backend = DaceBackend()
    compiled = _compile(backend, f)
    for n in (3, 4):  # Sizes that equal no other size, so that Dynamo does not unify them
        x, y = _data(rank, 5, n), _data(rank, size * n, 7)
        gathered, scattered, broadcast = compiled(x, y)
        torch.testing.assert_close(gathered, torch.cat([_data(r, 5, n).t() for r in range(size)]))
        total = sum(_data(r, size * n, 7) for r in range(size))
        torch.testing.assert_close(scattered, total[rank * n : (rank + 1) * n])
        torch.testing.assert_close(broadcast, _data(1, size * n, 7) * 3)
    assert backend.compile_count == 1


@pytest.mark.mpi
def test_all_to_all_single():
    """Even splits, and uneven splits where rank ``r`` sends ``r + d + 1`` rows to rank ``d``."""
    size, rank = dist.get_world_size(), dist.get_rank()
    group = dist.group.WORLD
    in_splits = [rank + d + 1 for d in range(size)]
    out_splits = [s + rank + 1 for s in range(size)]

    def f(x, z):
        even = fc.all_to_all_single(x, None, None, group)
        uneven = fc.all_to_all_single(z, out_splits, in_splits, group)
        return even, uneven

    backend = DaceBackend()
    compiled = _compile(backend, f)
    x, z = _data(rank, size * 2, 3), _data(rank, sum(in_splits), 2)
    even, uneven = compiled(x, z)
    torch.testing.assert_close(even, torch.cat([_data(s, size * 2, 3)[rank * 2 : rank * 2 + 2] for s in range(size)]))
    expected = []
    for s in range(size):
        start = sum(s + d + 1 for d in range(rank))
        expected.append(_data(s, sum(s + d + 1 for d in range(size)), 2)[start : start + s + rank + 1])
    torch.testing.assert_close(uneven, torch.cat(expected))


@pytest.mark.mpi
def test_subgroup():
    """A group without rank 0: group ranks differ from global ranks, and rank 0 does not communicate."""
    size, rank = dist.get_world_size(), dist.get_rank()
    members = list(range(1, size))
    group = dist.new_group(members)
    if rank == 0:
        return

    def f(x):
        reduced = fc.all_reduce(x, "sum", group)
        broadcast = fc.broadcast(x, 0, group)  # Group rank 0 is global rank 1
        return reduced, broadcast

    backend = DaceBackend()
    reduced, broadcast = _compile(backend, f)(_data(rank, 5))
    torch.testing.assert_close(reduced, sum(_data(r, 5) for r in members))
    torch.testing.assert_close(broadcast, _data(1, 5))


@pytest.mark.mpi
def test_ring_exchange():
    """``batch_isend_irecv`` in a ring; the received tensor is used after its wait."""
    size, rank = dist.get_world_size(), dist.get_rank()
    right, left = (rank + 1) % size, (rank - 1) % size

    def f(x):
        received = torch.empty_like(x)
        requests = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, x * 2, right), dist.P2POp(dist.irecv, received, left)]
        )
        for request in requests:
            request.wait()
        return received + 1

    backend = DaceBackend()
    with torch._dynamo.config.patch(enable_p2p_compilation=True):
        compiled = _compile(backend, f)
        for n in (3, 5):
            torch.testing.assert_close(compiled(_data(rank, n)), _data(left, n) * 2 + 1)
    assert backend.compile_count == 1


@pytest.mark.mpi
def test_training_through_collectives():
    """Gradients flow through the autograd-aware gather and reduce-scatter (each the other's backward)."""
    size, rank = dist.get_world_size(), dist.get_rank()
    group = dist.group.WORLD

    def weight(r: int) -> torch.Tensor:
        return _data(r, 3, 2) * 0.1 - 1

    def f(x, w):
        gathered = fc.all_gather_tensor_autograd(x, 0, group)  # (size * n, 3)
        scattered = fc.reduce_scatter_tensor_autograd(gathered @ w, "sum", 0, group)  # (n, 2)
        return (scattered * scattered).sum()

    backend = DaceBackend()
    compiled = _compile(backend, f)
    n = 4
    x = _data(rank, n, 3).requires_grad_()
    w = weight(rank).requires_grad_()
    compiled(x, w).backward()

    # Reference: the same computation on the inputs of every rank, in one process
    xs = [_data(r, n, 3).requires_grad_() for r in range(size)]
    ws = [weight(r).requires_grad_() for r in range(size)]
    full = torch.cat(xs)
    total = sum(full @ wr for wr in ws)
    sum(((total[r * n : (r + 1) * n]) ** 2).sum() for r in range(size)).backward()
    torch.testing.assert_close(x.grad, xs[rank].grad)
    torch.testing.assert_close(w.grad, ws[rank].grad)


@pytest.mark.mpi
def test_dtensor_redistribute():
    """A column-parallel product of DTensors, gathered by ``redistribute``. The graph takes and returns the
    ``DeviceMesh`` (an opaque object), so that the result is a DTensor again."""
    size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (size,))
    weight = _data(0, 6, 2 * size) * 0.01
    inputs = _data(1, 8, 6) * 0.01
    w = distribute_tensor(weight, mesh, [Shard(1)])
    x = distribute_tensor(inputs, mesh, [Replicate()])

    def f(x, w):
        return torch.relu(x @ w).redistribute(mesh, [Replicate()]).sum(dim=1)

    backend = DaceBackend()
    out = _compile(backend, f)(x, w)
    assert isinstance(out, DTensor)
    torch.testing.assert_close(out.to_local(), torch.relu(inputs @ weight).sum(dim=1))
    assert backend.compile_count == 1


@pytest.mark.mpi
def test_dtensor_uneven_shards():
    """Columns that do not divide evenly among the ranks: ``redistribute`` pads the shards to equal size (by an amount
    that is symbolic in the rank's shard size) before its all-gather."""
    size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (size,))
    columns = 2 * size + 1
    weight = _data(0, 6, columns) * 0.01
    inputs = _data(1, 8, 6) * 0.01
    w = distribute_tensor(weight, mesh, [Shard(1)])
    x = distribute_tensor(inputs, mesh, [Replicate()])

    def f(x, w):
        return torch.relu(x @ w).redistribute(mesh, [Replicate()]) * 2

    out = _compile(DaceBackend(), f)(x, w)
    torch.testing.assert_close(out.to_local(), torch.relu(inputs @ weight) * 2)


@pytest.mark.mpi
def test_distributed_compilation():
    """Rank 0 compiles, and the other ranks load its build."""
    size, rank = dist.get_world_size(), dist.get_rank()

    def f(x):
        return fc.all_reduce(x * 2, "sum", dist.group.WORLD) + 1

    backend = DaceBackend(distributed_compilation=True)
    for n in (4, 7):
        torch.testing.assert_close(_compile(backend, f)(_data(rank, n)), sum(_data(r, n) * 2 for r in range(size)) + 1)
    folders = [None] * size
    dist.all_gather_object(folders, backend.last_sdfg.build_folder)
    assert all(folder == folders[0] for folder in folders)


@pytest.mark.mpi
def test_distributed_compilation_failure(monkeypatch):
    """A compilation that fails on rank 0 raises on every rank, instead of leaving the others waiting in the
    program's collective."""
    rank = dist.get_rank()

    def f(x):
        return fc.all_reduce(x * 2, "sum", dist.group.WORLD) + 1

    if rank == 0:

        def fail(sdfg, *args, **kwargs):
            raise RuntimeError("injected compilation failure")

        monkeypatch.setattr(dace.SDFG, "compile", fail)
    with pytest.raises(Exception) as info:
        _compile(DaceBackend(distributed_compilation=True), f)(_data(rank, 4))
    if rank == 0:
        assert "injected compilation failure" in str(info.value)
    else:
        assert "Compilation failed on rank 0: RuntimeError: injected compilation failure" in str(info.value)


@pytest.mark.mpi
def test_distributed_compilation_of_rank_dependent_program():
    """A program that depends on the rank (Dynamo folds ``get_rank()``) cannot share rank 0's build."""
    rank = dist.get_rank()

    def f(x):
        return fc.all_reduce(x * (dist.get_rank() + 1), "sum", dist.group.WORLD)

    with pytest.raises(Exception) as info:
        _compile(DaceBackend(distributed_compilation=True), f)(_data(rank, 4))
    assert "differs from the SDFG that rank 0 compiled" in str(info.value)


if __name__ == "__main__":
    # Run under MPI, e.g., ``mpirun -n 2 python tests/torch_dynamo/test_distributed.py``
    path = _init_process_group()
    for op in ["sum", "avg", "max"]:
        test_all_reduce_in_place(op)
    test_gather_scatter_broadcast()
    test_all_to_all_single()
    test_subgroup()
    test_ring_exchange()
    test_training_through_collectives()
    test_dtensor_redistribute()
    test_dtensor_uneven_shards()
    test_distributed_compilation()
    test_distributed_compilation_of_rank_dependent_program()
    _destroy_process_group(path)
