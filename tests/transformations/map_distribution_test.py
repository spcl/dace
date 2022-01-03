# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A test for the ElementWiseArrayOperation transformation. """

import dace
import numpy as np
from dace.transformation.dataflow import ElementWiseArrayOperation
import pytest

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def eao_mpi(A: dace.float64[N], B: dace.float64[N]):
    return A * B


@pytest.mark.mpi
def test_eao_mpi():
    from mpi4py import MPI as MPI4PY
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = eao_mpi.to_sdfg(coarsen=True)
            mpi_sdfg.apply_transformations(ElementWiseArrayOperation)
            mpi_exec = mpi_sdfg.compile()
        comm.Barrier()

    length = 128 * commsize
    A = np.random.randn(length)
    B = np.random.randn(length)
    C = mpi_exec(A=A, B=B, N=length, commsize=commsize)
    if rank == 0:
        assert (np.allclose(C, A * B))
    else:
        assert (True)


if __name__ == '__main__':
    test_eao_mpi()
