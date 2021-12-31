# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed Jacobi-2D sample."""
import dace as dc
import numpy as np
import os
import timeit
from dace.sdfg.utils import load_precompiled_sdfg

from dace.transformation.dataflow import MapFusion

lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
lNy = dc.symbol('lNy', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
pi = dc.symbol('pi', dtype=dc.int32, integer=True, nonnegative=True)
pj = dc.symbol('pj', dtype=dc.int32, integer=True, nonnegative=True)
rank = dc.symbol('rank', dtype=dc.int32, integer=True, nonnegative=True)
size = dc.symbol('size', dtype=dc.int32, integer=True, positive=True)
noff = dc.symbol('noff', dtype=dc.int32, integer=True, nonnegative=True)
soff = dc.symbol('soff', dtype=dc.int32, integer=True, nonnegative=True)
woff = dc.symbol('woff', dtype=dc.int32, integer=True, nonnegative=True)
eoff = dc.symbol('eoff', dtype=dc.int32, integer=True, nonnegative=True)
Nx = Px * lNx
Ny = Py * lNy

nn = dc.symbol('nn', dtype=dc.int32, integer=True)
ns = dc.symbol('ns', dtype=dc.int32, integer=True)
nw = dc.symbol('nw', dtype=dc.int32, integer=True)
ne = dc.symbol('ne', dtype=dc.int32, integer=True)

MPI_Request = dc.opaque("MPI_Request")


@dc.program
def jacobi_2d_shared(TSTEPS: dc.int64, A: dc.float64[Nx, Ny],
                     B: dc.float64[Nx, Ny]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dc.program
def jacobi_2d_dist(TSTEPS: dc.int64, A: dc.float64[Nx, Ny], B: dc.float64[Nx,
                                                                          Ny]):

    lA = np.zeros((lNx + 2, lNy + 2), dtype=A.dtype)
    lB = np.zeros((lNx + 2, lNy + 2), dtype=B.dtype)
    tAB = np.empty((lNx, lNy), dtype=A.dtype)

    req = np.empty((8, ), dtype=MPI_Request)

    Av = np.reshape(A, (Px, lNx, Py, lNy))
    A2 = np.transpose(Av, axes=(0, 2, 1, 3))
    Bv = np.reshape(B, (Px, lNx, Py, lNy))
    B2 = np.transpose(Bv, axes=(0, 2, 1, 3))

    dc.comm.Scatter(A2, tAB)
    lA[1:-1, 1:-1] = tAB
    dc.comm.Scatter(B2, tAB)
    lB[1:-1, 1:-1] = tAB

    for t in range(1, TSTEPS):

        dc.comm.Isend(lA[1, 1:-1], nn, 0, req[0])
        dc.comm.Isend(lA[-2, 1:-1], ns, 1, req[1])
        dc.comm.Isend(lA[1:-1, 1], nw, 2, req[2])
        dc.comm.Isend(lA[1:-1, -2], ne, 3, req[3])
        dc.comm.Irecv(lA[0, 1:-1], nn, 1, req[4])
        dc.comm.Irecv(lA[-1, 1:-1], ns, 0, req[5])
        dc.comm.Irecv(lA[1:-1, 0], nw, 3, req[6])
        dc.comm.Irecv(lA[1:-1, -1], ne, 2, req[7])

        dc.comm.Waitall(req)

        lB[1 + noff:-1 - soff, 1 + woff:-1 -
           eoff] = 0.2 * (lA[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                          lA[1 + noff:-1 - soff, woff:-2 - eoff] +
                          lA[1 + noff:-1 - soff, 2 + woff:-eoff] +
                          lA[2 + noff:-soff, 1 + woff:-1 - eoff] +
                          lA[noff:-2 - soff, 1 + woff:-1 - eoff])

        dc.comm.Isend(lB[1, 1:-1], nn, 0, req[0])
        dc.comm.Isend(lB[-2, 1:-1], ns, 1, req[1])
        dc.comm.Isend(lB[1:-1, 1], nw, 2, req[2])
        dc.comm.Isend(lB[1:-1, -2], ne, 3, req[3])
        dc.comm.Irecv(lB[0, 1:-1], nn, 1, req[4])
        dc.comm.Irecv(lB[-1, 1:-1], ns, 0, req[5])
        dc.comm.Irecv(lB[1:-1, 0], nw, 3, req[6])
        dc.comm.Irecv(lB[1:-1, -1], ne, 2, req[7])

        dc.comm.Waitall(req)

        lA[1 + noff:-1 - soff, 1 + woff:-1 -
           eoff] = 0.2 * (lB[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                          lB[1 + noff:-1 - soff, woff:-2 - eoff] +
                          lB[1 + noff:-1 - soff, 2 + woff:-eoff] +
                          lB[2 + noff:-soff, 1 + woff:-1 - eoff] +
                          lB[noff:-2 - soff, 1 + woff:-1 - eoff])

    tAB[:] = lA[1:-1, 1:-1]
    dc.comm.Gather(tAB, A2)
    tAB[:] = lB[1:-1, 1:-1]
    dc.comm.Gather(tAB, B2)

    A[:] = np.transpose(A2, (0, 2, 1, 3))
    B[:] = np.transpose(B2, (0, 2, 1, 3))


def init_data(N, datatype):

    A = np.fromfunction(lambda i, j: i * (j + 2) / N,
                        shape=(N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N,
                        shape=(N, N),
                        dtype=datatype)

    return A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


grid = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2), 16: (4, 4)}

if __name__ == "__main__":

    TSTEPS, N = 100, 280

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    pi = rank // Py
    pj = rank % Py
    lNx = N // Px
    lNy = N // Py
    noff = soff = woff = eoff = 0
    nn = (pi - 1) * Py + pj
    ns = (pi + 1) * Py + pj
    nw = pi * Py + (pj - 1)
    ne = pi * Py + (pj + 1)
    if pi == 0:
        noff = 1
        nn = MPI.PROC_NULL
    if pi == Px - 1:
        soff = 1
        ns = MPI.PROC_NULL
    if pj == 0:
        woff = 1
        nw = MPI.PROC_NULL
    if pj == Py - 1:
        eoff = 1
        ne = MPI.PROC_NULL

    def setup_func(rank):
        if rank == 0:
            return init_data(N, np.float64)
        else:
            return (np.empty(
                (N, N), dtype=np.float64), np.empty((N, N), dtype=np.float64))

    A, B = setup_func(rank)

    mpi_sdfg = None
    mpi_sdfg = jacobi_2d_dist.to_sdfg(strict=False)
    if rank == 0:
        mpi_sdfg.coarsen_dataflow()
        mpi_sdfg.apply_transformations_repeated([MapFusion])
        mpi_sdfg.coarsen_dataflow()
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dc.Config.get('default_build_folder')
        mpi_func = load_precompiled_sdfg(
            os.path.join(build_folder, jacobi_2d_dist.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=A,
             B=B,
             TSTEPS=TSTEPS,
             lNx=lNx,
             lNy=lNy,
             rank=rank,
             size=size,
             Px=Px,
             Py=Py,
             pi=pi,
             pj=pj,
             noff=noff,
             soff=soff,
             woff=woff,
             eoff=eoff,
             nn=nn,
             ns=ns,
             nw=nw,
             ne=ne)

    comm.Barrier()

    stmt = (
        "mpi_func(A=A, B=B, TSTEPS=TSTEPS, lNx=lNx, lNy=lNy, rank=rank, size=size, "
        "Px=Px, Py=Py, pi=pi, pj=pj, "
        "noff=noff, soff=soff, woff=woff, eoff=eoff, "
        "nn=nn, ns=ns, nw=nw, ne=ne)")
    setup = "A, B = setup_func(rank); comm.Barrier()"
    repeat = 10

    raw_time_list = timeit.repeat(stmt,
                                  setup=setup,
                                  repeat=repeat,
                                  number=1,
                                  globals=ldict)
    raw_time = np.median(raw_time_list)

    comm.Barrier()

    if rank == 0:
        ms_time = time_to_ms(raw_time)
        print("Median is {}ms".format(ms_time))

        refA, refB = init_data(N, np.float64)
        shared_sdfg = jacobi_2d_shared.compile()
        shared_sdfg(A=refA,
                    B=refB,
                    TSTEPS=TSTEPS,
                    lNx=lNx,
                    lNy=lNy,
                    Px=Px,
                    Py=Py)

        print("=======Validation=======")
        assert (np.allclose(A, refA))
        assert (np.allclose(B, refB))
        print("OK")
