import dace
from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
import dace.dtypes as dtypes
import dace.frontend.common.distr as comm
import numpy as np
import pytest


@pytest.mark.mpi
def test_subarray_scatter():

    P = dace.symbol('P', dace.int32)

    @dace.program
    def block_scatter(A: dace.int32[100, 100]):
        scatter_grid = dace.comm.Cart_create([2, P//2])
        lA = np.empty_like(A, shape=(50, 100 // (P//2)))
        subarray = dace.comm.BlockScatter(A, lA, scatter_grid)
        return lA
    
    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()
    last_rank = (size // 2) * 2

    # if size < 2:
    #     raise ValueError("Please run this test with at least two processes.")

    if rank == 0:
        sdfg = block_scatter.to_sdfg()
        func = sdfg.compile()
    commworld.Barrier()
    if rank > 0:
        sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=block_scatter.name))
        func = CompiledSDFG(sdfg, ReloadableDLL(".dacecache/{n}/build/lib{n}.so".format(n=sdfg.name), sdfg.name))
    commworld.Barrier()

    A = np.arange(10000, dtype=np.int32).reshape(100, 100).copy()
    lA_ref = A.reshape(2, 50, size//2, 100 // (size//2)).transpose(0, 2, 1, 3)
    if rank == 0:
        lA = func(A=A, P=size)
    else:
        lA = func(A=np.zeros((1, ), dtype=np.int32), P=size)

    if rank < last_rank:
        assert (np.array_equal(lA, lA_ref[rank // 2, rank % 2]))


if __name__ == "__main__":
    test_subarray_scatter()
