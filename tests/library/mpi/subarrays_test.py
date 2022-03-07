import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import numpy as np
import pytest


@pytest.mark.mpi
def test_subarray_scatter():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def block_scatter(A: dace.int32[8*P, 8*P]):
        scatter_grid = dace.comm.Cart_create([2, P//2])
        lA = np.empty_like(A, shape=(4*P, 16))
        subarray = dace.comm.BlockScatter(A, lA, scatter_grid)
        return lA
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = block_scatter.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=block_scatter.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(64*even_size*even_size, dtype=np.int32).reshape(8*even_size, 8*even_size).copy()
    lA_ref = A.reshape(2, 4*even_size, even_size//2, 16).transpose(0, 2, 1, 3)
    if rank == 0:
        lA = func(A=A, P=even_size)
    else:
        lA = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        assert (np.array_equal(lA, lA_ref[rank // (even_size//2), rank % (even_size//2)]))


@pytest.mark.mpi
def test_subarray_scatter_bcast():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def block_scatter_bcast(A: dace.int32[8*P]):
        pgrid = dace.comm.Cart_create([2, P//2])
        scatter_grid = dace.comm.Cart_sub(pgrid, [False, True], exact_grid=0)
        bcast_grid = dace.comm.Cart_sub(pgrid, [True, False])
        lA = np.empty_like(A, shape=(16,))
        subarray = dace.comm.BlockScatter(A, lA, scatter_grid, bcast_grid)
        return lA
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = block_scatter_bcast.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=block_scatter_bcast.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(8*even_size, dtype=np.int32)

    if rank == 0:
        lA = func(A=A, P=even_size)
    else:
        lA = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        lbound = (rank % (even_size//2)) * 16
        ubound = (rank % (even_size//2) + 1) * 16
        assert (np.array_equal(lA, A[lbound:ubound]))


@pytest.mark.mpi
def test_subarray_gather():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def block_gather(lA: dace.int32[4*P, 16]):
        gather_grid = dace.comm.Cart_create([2, P//2])
        A = np.empty_like(lA, shape=(8*P, 8*P))
        subarray = dace.comm.BlockGather(lA, A, gather_grid)
        return A
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = block_gather.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=block_gather.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A_ref = np.arange(64*even_size*even_size, dtype=np.int32).reshape(8*even_size, 8*even_size)
    lA = A_ref.reshape(2, 4*even_size, even_size//2, 16).transpose(0, 2, 1, 3)
    if rank < even_size:
        A = func(lA=lA[rank // (even_size//2), rank % (even_size//2)].copy(), P=even_size)
    else:
        A = func(lA=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank == 0:
        assert (np.array_equal(A, A_ref))


@pytest.mark.mpi
def test_subarray_gather_reduce():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def block_gather_reduce(lA: dace.int32[16]):
        pgrid = dace.comm.Cart_create([2, P//2])
        gather_grid = dace.comm.Cart_sub(pgrid, [False, True], exact_grid=0)
        reduce_grid = dace.comm.Cart_sub(pgrid, [True, False])
        A = np.empty_like(lA, shape=(8*P))
        subarray = dace.comm.BlockGather(lA, A, gather_grid, reduce_grid)
        return A
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = block_gather_reduce.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=block_gather_reduce.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A_ref = np.arange(8*even_size, dtype=np.int32)
    if rank < even_size:
        lbound = (rank % (even_size//2)) * 16
        ubound = (rank % (even_size//2) + 1) * 16
        A = func(lA=A_ref[lbound:ubound].copy(), P=even_size)
    else:
        A = func(lA=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank == 0:
        assert (np.array_equal(A, 2 * A_ref))


if __name__ == "__main__":
    test_subarray_scatter()
    test_subarray_scatter_bcast()
    test_subarray_gather()
    test_subarray_gather_reduce()
