import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import dace.dtypes as dtypes
import numpy as np
import pytest


@pytest.mark.mpi
def test_redistribute_matrix_2d_2d():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d(A: dace.int32[50, 100 // (P//2)]):

        a_grid = dace.comm.Cart_create([2, P//2])
        b_grid = dace.comm.Cart_create([P//2, 2])

        B = np.empty_like(A, shape=(100 // (P//2), 50))

        a_arr = dace.comm.Subarray((100, 100), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((100, 100), B, process_grid=b_grid)

        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)

        return B
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    last_rank = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = matrix_2d_2d.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=matrix_2d_2d.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(10000, dtype=np.int32).reshape(100, 100)
    lA = A.reshape(2, 50, size//2, 100 // (size//2)).transpose(0, 2, 1, 3)
    lB = A.reshape(size//2, 100 // (size//2), 2, 50).transpose(0, 2, 1, 3)
    if rank < last_rank:
        B = func(A=lA[rank // 2, rank % 2].copy(), P=size)
    else:
        B = func(A=np.zeros((1, ), dtype=np.int32), P=size)

    if rank < last_rank:
        assert (np.array_equal(B, lB[rank // (size//2), rank % (size//2)]))


@pytest.mark.mpi
def test_redistribute_matrix_2d_2d_2():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d_2(A: dace.int32[50, 100 // (P//2)]):

        a_grid = dace.comm.Cart_create([2, P//2])
        b_grid = dace.comm.Cart_create([2 * (P//2), 1])

        B = np.empty_like(A, shape=(100 // (2 * (P//2)), 100))

        a_arr = dace.comm.Subarray((100, 100), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((100, 100), B, process_grid=b_grid)

        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)

        return B
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    last_rank = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = matrix_2d_2d_2.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=matrix_2d_2d_2.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(10000, dtype=np.int32).reshape(100, 100)
    lA = A.reshape(2, 50, size//2, 100 // (size//2)).transpose(0, 2, 1, 3)
    lB = A.reshape(last_rank, 100//last_rank, 1, 100).transpose(0, 2, 1, 3)
    if rank < last_rank:
        B = func(A=lA[rank // 2, rank % 2].copy(), P=size)
    else:
        B = func(A=np.zeros((1, ), dtype=np.int32), P=size)

    if rank < last_rank:
        assert (np.array_equal(B, lB[rank, 0]))


if __name__ == "__main__":
    test_redistribute_matrix_2d_2d()
    test_redistribute_matrix_2d_2d_2()
