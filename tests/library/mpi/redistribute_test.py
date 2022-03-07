import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import dace.dtypes as dtypes
import numpy as np
import pytest


@pytest.mark.mpi
def test_redistribute_matrix_2d_2d():
    """
     _______________________         _______________________
    |     |     |     |     |       |           |           |
    |     |     |     |     |       |___________|___________|
    |     |     |     |     |       |           |           |
    |_____|_____|_____|_____|   ->  |___________|___________|
    |     |     |     |     |       |           |           |
    |     |     |     |     |       |___________|___________|
    |     |     |     |     |       |           |           |
    |_____|_____|_____|_____|       |___________|___________|
    """

    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d(A: dace.int32[4 * P, 16]):

        a_grid = dace.comm.Cart_create([2, P // 2])
        b_grid = dace.comm.Cart_create([P // 2, 2])

        B = np.empty_like(A, shape=(16, 4 * P))

        a_arr = dace.comm.Subarray((8 * P, 8 * P), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((8 * P, 8 * P), B, process_grid=b_grid)

        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)

        return B

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

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

    A = np.arange(64 * even_size * even_size, dtype=np.int32).reshape(8 * even_size, 8 * even_size)
    lA = A.reshape(2, 4 * even_size, even_size // 2, 16).transpose(0, 2, 1, 3)
    lB = A.reshape(even_size // 2, 16, 2, 4 * even_size).transpose(0, 2, 1, 3)
    if rank < even_size:
        B = func(A=lA[rank // (even_size // 2), rank % (even_size // 2)].copy(), P=even_size)
    else:
        B = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        assert (np.array_equal(B, lB[rank // 2, rank % 2]))


@pytest.mark.mpi
def test_redistribute_matrix_2d_2d_2():
    """
     _______________________         _______________________
    |     |     |     |     |       |_______________________|
    |     |     |     |     |       |_______________________|
    |     |     |     |     |       |_______________________|
    |_____|_____|_____|_____|   ->  |_______________________|
    |     |     |     |     |       |_______________________|
    |     |     |     |     |       |_______________________|
    |     |     |     |     |       |_______________________|
    |_____|_____|_____|_____|       |_______________________|
    """

    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d_2(A: dace.int32[4 * P, 16]):

        a_grid = dace.comm.Cart_create([2, P // 2])
        b_grid = dace.comm.Cart_create([P, 1])

        B = np.empty_like(A, shape=(8, 8 * P))

        a_arr = dace.comm.Subarray((8 * P, 8 * P), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((8 * P, 8 * P), B, process_grid=b_grid)

        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)

        return B

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

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

    A = np.arange(64 * even_size * even_size, dtype=np.int32).reshape(8 * even_size, 8 * even_size)
    lA = A.reshape(2, 4 * even_size, even_size // 2, 16).transpose(0, 2, 1, 3)
    lB = A.reshape(even_size, 8, 1, 8 * even_size).transpose(0, 2, 1, 3)
    if rank < even_size:
        B = func(A=lA[rank // (even_size // 2), rank % (even_size // 2)].copy(), P=even_size)
    else:
        B = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        assert (np.array_equal(B, lB[rank, 0]))


@pytest.mark.mpi
def test_redistribute_matrix_2d_2d_3():
    """
    The numbers are example tile IDs, NOT MPI ranks.
     _______________________         ___________
    |0    |1    |2    |3    |       |0    |4    |
    |     |     |     |     |       |     |     |
    |     |     |     |     |       |     |     |
    |_____|_____|_____|_____|   ->  |_____|_____|
    |4    |5    |6    |7    |       |1    |5    |
    |     |     |     |     |       |     |     |
    |     |     |     |     |       |     |     |
    |_____|_____|_____|_____|       |_____|_____|
                                    |2    |6    |
                                    |     |     |
                                    |     |     |
                                    |_____|_____|
                                    |3    |7    |
                                    |     |     |
                                    |     |     |
                                    |_____|_____|
    """

    P = dace.symbol('P', dace.int32)

    @dace.program
    def matrix_2d_2d_3(A: dace.int32[4 * P, 16]):

        a_grid = dace.comm.Cart_create([2, P // 2])
        b_grid = dace.comm.Cart_create([P // 2, 2])

        B = np.empty_like(A)

        a_arr = dace.comm.Subarray((8 * P, 8 * P), A, process_grid=a_grid)
        b_arr = dace.comm.Subarray((8 * P, 8 * P), B, process_grid=b_grid, correspondence=(1, 0))

        rdistr = dace.comm.Redistribute(A, a_arr, B, b_arr)

        return B

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = matrix_2d_2d_3.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=matrix_2d_2d_3.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(64 * even_size * even_size, dtype=np.int32).reshape(8 * even_size, 8 * even_size)
    lA = A.reshape(2, 4 * even_size, even_size // 2, 16).transpose(0, 2, 1, 3)
    lB = A.reshape(2, 4 * even_size, even_size // 2, 16).transpose(2, 0, 1, 3)
    if rank < even_size:
        B = func(A=lA[rank // (even_size // 2), rank % (even_size // 2)].copy(), P=even_size)
    else:
        B = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        assert (np.array_equal(B, lB[rank // 2, rank % 2]))


@pytest.mark.mpi
def test_redistribute_vector_2d_2d():
    """
    The numbers are example tile IDs, NOT MPI ranks. "(r)" means that the tile is a replica.
     ____________________        _______________________        ___________
    |____________________|  ->  |0____|1____|2____|3____|  ->  |0  __|zero_|
                                |0(r)_|1(r)_|2(r)_|3(r)_|      |1____|zero_|
                                                               |2____|zero_|
                                                               |3____|zero_|
    """

    P = dace.symbol('P', dace.int32)

    @dace.program
    def vector_2d_2d(A: dace.int32[8 * P]):

        a_grid = dace.comm.Cart_create([2, P // 2])
        a_scatter_grid = dace.comm.Cart_sub(a_grid, [False, True], exact_grid=0)
        a_bcast_grid = dace.comm.Cart_sub(a_grid, [True, False])
        b_grid = dace.comm.Cart_create([P // 2, 2])
        b_scatter_grid = dace.comm.Cart_sub(b_grid, [True, False], exact_grid=0)
        b_bcast_grid = dace.comm.Cart_sub(b_grid, [False, True])

        lA = np.empty_like(A, shape=(16, ))
        a_subarr = dace.comm.BlockScatter(A, lA, a_scatter_grid, a_bcast_grid)
        lB = np.zeros_like(A, shape=(16, ))
        b_subarr = dace.comm.Subarray((8 * P, ), lB, process_grid=b_scatter_grid)
        redistr = dace.comm.Redistribute(lA, a_subarr, lB, b_subarr)

        return lB

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    even_size = (size // 2) * 2

    if size < 2:
        raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = vector_2d_2d.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=vector_2d_2d.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(8 * even_size, dtype=np.int32)
    lB_ref = A.reshape(even_size // 2, 16)
    if rank < even_size:
        lB = func(A=A, P=even_size)
    else:
        lB = func(A=np.zeros((1, ), dtype=np.int32), P=even_size)

    if rank < even_size:
        if rank % 2 == 0:
            assert (np.array_equal(lB, lB_ref[rank // 2]))
        else:
            assert (np.array_equal(lB, np.zeros_like(lB)))


if __name__ == "__main__":
    test_redistribute_matrix_2d_2d()
    test_redistribute_matrix_2d_2d_2()
    test_redistribute_matrix_2d_2d_3()
    test_redistribute_vector_2d_2d()
