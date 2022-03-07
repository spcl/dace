# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import dace.dtypes as dtypes
import dace.frontend.common.distr as comm
import numpy as np
import pytest


@pytest.mark.mpi
def test_process_grid():

    P = dace.symbol('P', dace.int32)

    sdfg = dace.SDFG("process_grid_test")
    sdfg.add_symbol('P', dace.int32)
    _, darr = sdfg.add_array("dims", (2, ), dtype=dace.int32)
    _, parr = sdfg.add_array("periods", (2, ), dtype=dace.int32)
    _, carr = sdfg.add_array("coords", (2, ), dtype=dace.int32)
    _, varr = sdfg.add_array("valid", (1, ), dtype=dace.bool_)

    state = sdfg.add_state("start")
    pgrid_name = comm._cart_create(None, sdfg, state, [1, P])

    state2 = sdfg.add_state("main")
    sdfg.add_edge(state, state2, dace.InterstateEdge())
    tasklet = state2.add_tasklet(
        "MPI_Cart_get", {}, {'d', 'p', 'c', 'v'},
        f"MPI_Cart_get(__state->{pgrid_name}_comm, P, d, p, c);\nv = __state->{pgrid_name}_valid;", dtypes.Language.CPP)
    dims = state2.add_write("dims")
    periods = state2.add_write("periods")
    coords = state2.add_write("coords")
    valid = state2.add_write("valid")
    state2.add_edge(tasklet, 'd', dims, None, dace.Memlet.from_array("dims", darr))
    state2.add_edge(tasklet, 'p', periods, None, dace.Memlet.from_array("periods", parr))
    state2.add_edge(tasklet, 'c', coords, None, dace.Memlet.from_array("coords", carr))
    state2.add_edge(tasklet, 'v', valid, None, dace.Memlet("valid[0]"))

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    dims = np.zeros((2, ), dtype=np.int32)
    periods = np.zeros((2, ), dtype=np.int32)
    coords = np.zeros((2, ), dtype=np.int32)
    valid = np.zeros((1, ), dtype=np.bool_)
    func(dims=dims, periods=periods, coords=coords, valid=valid, P=size)

    assert (np.array_equal(dims, [1, size]))
    assert (np.array_equal(periods, [0, 0]))
    assert (np.array_equal(coords, [0, rank]))
    assert (valid[0])


@pytest.mark.mpi
def test_sub_grid():

    P = dace.symbol('P', dace.int32)

    sdfg = dace.SDFG("sub_grid_test")
    sdfg.add_symbol('P', dace.int32)
    _, darr = sdfg.add_array("dims", (1, ), dtype=dace.int32)
    _, parr = sdfg.add_array("periods", (1, ), dtype=dace.int32)
    _, carr = sdfg.add_array("coords", (1, ), dtype=dace.int32)
    _, varr = sdfg.add_array("valid", (1, ), dtype=dace.bool_)

    state = sdfg.add_state("start")
    parent_pgrid_name = comm._cart_create(None, sdfg, state, [1, P])
    pgrid_name = comm._cart_sub(None, sdfg, state, parent_pgrid_name, [False, True])

    state2 = sdfg.add_state("main")
    sdfg.add_edge(state, state2, dace.InterstateEdge())
    tasklet = state2.add_tasklet(
        "MPI_Cart_get", {}, {'d', 'p', 'c', 'v'},
        f"MPI_Cart_get(__state->{pgrid_name}_comm, P, &d, &p, &c);\nv = __state->{pgrid_name}_valid;",
        dtypes.Language.CPP)
    dims = state2.add_write("dims")
    periods = state2.add_write("periods")
    coords = state2.add_write("coords")
    valid = state2.add_write("valid")
    state2.add_edge(tasklet, 'd', dims, None, dace.Memlet.from_array("dims", darr))
    state2.add_edge(tasklet, 'p', periods, None, dace.Memlet.from_array("periods", parr))
    state2.add_edge(tasklet, 'c', coords, None, dace.Memlet.from_array("coords", carr))
    state2.add_edge(tasklet, 'v', valid, None, dace.Memlet("valid[0]"))

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    dims = np.zeros((1, ), dtype=np.int32)
    periods = np.zeros((1, ), dtype=np.int32)
    coords = np.zeros((1, ), dtype=np.int32)
    valid = np.zeros((1, ), dtype=np.bool_)
    func(dims=dims, periods=periods, coords=coords, valid=valid, P=size)

    assert (np.array_equal(dims, [size]))
    assert (np.array_equal(periods, [0]))
    assert (np.array_equal(coords, [rank]))
    assert (valid[0])


def test_process_grid_bcast():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def pgrid_bcast(A: dace.int32[10]):
        pgrid = dace.comm.Cart_create([1, P])
        dace.comm.Bcast(A, grid=pgrid)

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = pgrid_bcast.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=pgrid_bcast.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    if rank == 0:
        A = np.arange(10, dtype=np.int32)
    else:
        A = np.zeros((10, ), dtype=np.int32)
    func(A=A, P=size)

    assert (np.array_equal(A, np.arange(10, dtype=np.int32)))


def test_sub_grid_bcast():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def pgrid_bcast(A: dace.int32[10], rank: dace.int32):
        pgrid = dace.comm.Cart_create([2, P // 2])
        sgrid = dace.comm.Cart_sub(pgrid, [False, True])
        dace.comm.Bcast(A, grid=pgrid)
        B = np.empty_like(A)
        B[:] = rank % 10
        dace.comm.Bcast(B, grid=sgrid)
        A[:] = B

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    last_rank = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = pgrid_bcast.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=pgrid_bcast.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    if rank == 0:
        A = np.arange(10, dtype=np.int32)
    else:
        A = np.ones((10, ), dtype=np.int32)
    func(A=A, rank=rank, P=size)

    if rank < size // 2:
        assert (np.array_equal(A, np.zeros((10, ), dtype=np.int32)))
    elif rank < last_rank:
        assert (np.array_equal(A, np.full_like(A, fill_value=(size // 2) % 10)))
    else:
        assert (np.array_equal(A, np.full_like(A, fill_value=rank % 10)))


if __name__ == "__main__":
    test_process_grid()
    test_sub_grid()
    test_process_grid_bcast()
    test_sub_grid_bcast()
