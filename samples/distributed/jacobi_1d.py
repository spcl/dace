# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed Jacobi-1D sample."""
import dace as dc
import numpy as np
import os
import sys
import timeit
from dace.sdfg.utils import load_precompiled_sdfg

from mpi4py import MPI

N = dc.symbol('N', dtype=dc.int64)
lN = dc.symbol('lN', dtype=dc.int64)
rank = dc.symbol('rank', dtype=dc.int32)
size = dc.symbol('size', dtype=dc.int32)


def relerr(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


@dc.program
def jacobi_1d_shared(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@dc.program
def jacobi_1d_dist(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):

    lA = np.zeros((lN + 2, ), dtype=A.dtype)
    lB = np.zeros((lN + 2, ), dtype=B.dtype)
    tAB = np.empty((lN, ), dtype=A.dtype)

    dc.comm.Scatter(A, tAB)
    lA[1:-1] = tAB
    dc.comm.Scatter(B, tAB)
    lB[1:-1] = tAB

    for t in range(1, TSTEPS):
        if rank > 0:
            dc.comm.Recv(lA[0], rank - 1, t)
            dc.comm.Send(lA[1], rank - 1, t)
        if rank < size - 1:
            dc.comm.Send(lA[-2], rank + 1, t)
            dc.comm.Recv(lA[-1], rank + 1, t)
        if rank == 0:
            lB[2:-1] = 0.33333 * (lA[1:-2] + lA[2:-1] + lA[3:])
        elif rank == size - 1:
            lB[1:-2] = 0.33333 * (lA[:-3] + lA[1:-2] + lA[2:-1])
        else:
            lB[1:-1] = 0.33333 * (lA[:-2] + lA[1:-1] + lA[2:])
        if rank > 0:
            dc.comm.Recv(lB[0], rank - 1, t)
            dc.comm.Send(lB[1], rank - 1, t)
        if rank < size - 1:
            dc.comm.Send(lB[-2], rank + 1, t)
            dc.comm.Recv(lB[-1], rank + 1, t)
        if rank == 0:
            lA[2:-1] = 0.33333 * (lB[1:-2] + lB[2:-1] + lB[3:])
        elif rank == size - 1:
            lA[1:-2] = 0.33333 * (lB[:-3] + lB[1:-2] + lB[2:-1])
        else:
            lA[1:-1] = 0.33333 * (lB[:-2] + lB[1:-1] + lB[2:])

    tAB[:] = lA[1:-1]
    dc.comm.Gather(tAB, A)
    tAB[:] = lB[1:-1]
    dc.comm.Gather(tAB, B)


def init_data(N, datatype):

    A = np.fromfunction(lambda i: (i + 2) / N, shape=(N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, shape=(N, ), dtype=datatype)

    return A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


if __name__ == "__main__":

    # Initialization
    TSTEPS, N = 50, 1000
    A, B = init_data(N, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    lN = N // size

    if size < 2 or lN * size != 1000:
        if rank == 0:
            print("Please run this sample with at least 2 MPI ranks. Furthermore, "
                  "the number of MPI ranks must divide evenly N (by default 1000).")
        sys.exit(0)

    mpi_sdfg = jacobi_1d_dist.to_sdfg()
    if rank == 0:
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dc.Config.get('default_build_folder')
        mpi_func = load_precompiled_sdfg(os.path.join(build_folder, jacobi_1d_dist.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=A, B=B, TSTEPS=TSTEPS, N=N, lN=lN, rank=rank, size=size)

    comm.Barrier()

    stmt = ("mpi_func(A=A, B=B, TSTEPS=TSTEPS, N=N, "
            "lN=lN, rank=rank, size=size)")
    setup = "A, B = init_data(N, np.float64); comm.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)

    comm.Barrier()

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time))

        refA, refB = init_data(N, np.float64)
        shared_sdfg = jacobi_1d_shared.compile()
        shared_sdfg(A=refA, B=refB, TSTEPS=TSTEPS, N=N)

        print("=======Validation=======")
        assert (np.allclose(A, refA))
        assert (np.allclose(B, refB))
        print("OK")
