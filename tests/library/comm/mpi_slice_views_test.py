# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization leaves no View in front of a communication: a slice (``a[0:1]``, ``r[t:t+1]``, a strided column)
is read in place, with the offset and the count from the memlet. Run with ``mpirun -n 2 pytest --with-mpi``."""

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.libraries.comm.nodes.node import CommNode
from dace.sdfg import nodes, utils
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")
T = dace.symbol("T")
LENGTH, STEPS, ROWS = 8, 3, 4


def views_at_communication(sdfg: dace.SDFG) -> list[str]:
    """The Views wired to a communication node."""
    found = []
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, CommNode):
            continue
        for edge in state.all_edges(node):
            end = edge.src if edge.dst is node else edge.dst
            if isinstance(end, nodes.AccessNode) and isinstance(end.desc(state.sdfg), dt.View):
                found.append(end.data)
    return found


def canonical(program, comm) -> object:
    """``program`` canonicalized on rank 0, with no View next to a communication, compiled on every rank."""
    sdfg, found = None, []
    if comm.Get_rank() == 0:
        try:
            sdfg = program.to_sdfg(simplify=True)
            canonicalize(sdfg, validate=True)
            found = views_at_communication(sdfg)
        except Exception as error:  # every rank must reach the bcast below, or the others hang in it
            found = [f"{type(error).__name__}: {error}"]
    # every rank asserts, so a failure on rank 0 does not leave the others waiting in the compile
    found = comm.bcast(found, root=0)
    assert not found, found
    return utils.distributed_compile(sdfg, comm)


@pytest.mark.mpi
def test_a_halo_of_one_element_slices_needs_no_view():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    @dace.program
    def halo(a: dace.float64[N], ghost: dace.float64[2], left: dace.int32, right: dace.int32):
        req = dace.define_local([4], dtype=MPI.Request)
        MPI.COMM_WORLD.Isend(a[0:1], left, 0, req[0])
        MPI.COMM_WORLD.Isend(a[N - 1 : N], right, 1, req[1])
        MPI.COMM_WORLD.Irecv(ghost[0:1], left, 1, req[2])
        MPI.COMM_WORLD.Irecv(ghost[1:2], right, 0, req[3])
        MPI.Request.Waitall(req)

    func = canonical(halo, comm)
    left, right = (rank - 1) % size, (rank + 1) % size
    a = 10.0 * rank + np.arange(LENGTH, dtype=np.float64)
    ghost = np.zeros(2)
    func(a=a, ghost=ghost, left=np.int32(left), right=np.int32(right), N=LENGTH)
    assert ghost[0] == 10.0 * left + LENGTH - 1 and ghost[1] == 10.0 * right


@pytest.mark.mpi
def test_an_allreduce_of_one_step_of_a_time_series_needs_no_view():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    @dace.program
    def residual(r: dace.float64[T]):
        for t in range(T):
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, r[t : t + 1], op=MPI.SUM)

    func = canonical(residual, comm)
    r = (rank + 1) * np.arange(STEPS, dtype=np.float64)
    func(r=r, T=STEPS)
    assert np.array_equal(r, size * (size + 1) / 2 * np.arange(STEPS, dtype=np.float64))


@pytest.mark.mpi
def test_a_strided_column_goes_round_a_ring_without_a_view():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    @dace.program
    def column(a: dace.float64[ROWS, 3], b: dace.float64[ROWS, 3], left: dace.int32, right: dace.int32):
        req = dace.define_local([2], dtype=MPI.Request)
        MPI.COMM_WORLD.Isend(a[:, 0], right, 0, req[0])
        MPI.COMM_WORLD.Irecv(b[:, 2], left, 0, req[1])
        MPI.Request.Waitall(req)

    func = canonical(column, comm)
    left, right = (rank - 1) % size, (rank + 1) % size
    a = 10.0 * rank + np.arange(ROWS * 3, dtype=np.float64).reshape(ROWS, 3)
    b = np.zeros((ROWS, 3))
    func(a=a, b=b, left=np.int32(left), right=np.int32(right))
    expected = np.zeros((ROWS, 3))
    expected[:, 2] = 10.0 * left + np.arange(ROWS * 3, dtype=np.float64).reshape(ROWS, 3)[:, 0]
    assert np.array_equal(b, expected)


def request_copies(sdfg: dace.SDFG) -> list[str]:
    """The containers a communication writes its request into that are not the program's request array ``req``."""
    return [
        edge.data.data
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, CommNode)
        for edge in state.out_edges_by_connector(node, "_request")
        if edge.data.data != "req"
    ]


@pytest.mark.mpi
def test_a_start_writes_its_request_element_in_place():
    """The frontend writes ``Isend(..., req[0])``'s request into a scalar and copies it back to ``req[0]`` in the next
    state; canonicalization lets the call write ``req[0]`` itself, so no copy of a request lies between the starts and
    the wait (where the last one would fuse with the interior update and pin the wait in front of it)."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    @dace.program
    def halo(ap: dace.float64[N + 2], b: dace.float64[N + 2], left: dace.int32, right: dace.int32):
        req = dace.define_local([4], dtype=MPI.Request)
        MPI.COMM_WORLD.Isend(ap[1:2], left, 0, req[0])
        MPI.COMM_WORLD.Isend(ap[N : N + 1], right, 1, req[1])
        MPI.COMM_WORLD.Irecv(ap[0:1], left, 1, req[2])
        MPI.COMM_WORLD.Irecv(ap[N + 1 : N + 2], right, 0, req[3])
        for i in dace.map[2:N]:
            b[i] = ap[i - 1] + ap[i] + ap[i + 1]
        MPI.Request.Waitall(req)
        b[1] = ap[0] + ap[1] + ap[2]
        b[N] = ap[N - 1] + ap[N] + ap[N + 1]

    copies = None
    if rank == 0:
        sdfg = halo.to_sdfg(simplify=True)
        canonicalize(sdfg, validate=True)
        copies = request_copies(sdfg)
    copies = comm.bcast(copies, root=0)
    assert copies == [], copies

    func = canonical(halo, comm)
    left, right = (rank - 1) % size, (rank + 1) % size
    ap = 10.0 * rank + np.arange(LENGTH + 2, dtype=np.float64)
    b = np.zeros(LENGTH + 2)
    func(ap=ap, b=b, left=np.int32(left), right=np.int32(right), N=LENGTH)
    expected = ap.copy()
    expected[0], expected[LENGTH + 1] = 10.0 * left + LENGTH, 10.0 * right + 1
    stencil = expected[:-2] + expected[1:-1] + expected[2:]
    assert np.array_equal(ap, expected) and np.array_equal(b[1 : LENGTH + 1], stencil)
