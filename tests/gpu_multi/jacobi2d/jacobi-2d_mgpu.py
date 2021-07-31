# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly multi GPU distributed Jacobi-2D sample."""
import dace
import numpy as np
import os
import timeit

from dace.transformation.dataflow import MapFusion

d_int = dace.int64
d_float = dace.float64
np_float = np.float64

lNx = dace.symbol('lNx', dtype=d_int, integer=True, positive=True)
lNy = dace.symbol('lNy', dtype=d_int, integer=True, positive=True)
Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
pi = dace.symbol('pi', dtype=dace.int32, integer=True, nonnegative=True)
pj = dace.symbol('pj', dtype=dace.int32, integer=True, nonnegative=True)
rank = dace.symbol('rank', dtype=dace.int32, integer=True, nonnegative=True)
size = dace.symbol('size', dtype=dace.int32, integer=True, positive=True)
noff = dace.symbol('noff', dtype=dace.int32, integer=True, nonnegative=True)
soff = dace.symbol('soff', dtype=dace.int32, integer=True, nonnegative=True)
woff = dace.symbol('woff', dtype=dace.int32, integer=True, nonnegative=True)
eoff = dace.symbol('eoff', dtype=dace.int32, integer=True, nonnegative=True)
Nx = Px * lNx
Ny = Py * lNy

nn = dace.symbol('nn', dtype=dace.int32, integer=True)
ns = dace.symbol('ns', dtype=dace.int32, integer=True)
nw = dace.symbol('nw', dtype=dace.int32, integer=True)
ne = dace.symbol('ne', dtype=dace.int32, integer=True)

MPI_Request = dace.opaque("MPI_Request")


@dace.program
def jacobi_2d_dist(TSTEPS: d_int, A: d_float[Nx, Ny], B: d_float[Nx, Ny]):

    lA = np.zeros((lNx + 2, lNy + 2), dtype=A.dtype)
    lB = np.zeros((lNx + 2, lNy + 2), dtype=B.dtype)
    tAB = np.empty((lNx, lNy), dtype=A.dtype)

    req = np.empty((8, ), dtype=MPI_Request)

    Av = np.reshape(A, (Px, lNx, Py, lNy))
    A2 = np.transpose(Av, axes=(0, 2, 1, 3))
    Bv = np.reshape(B, (Px, lNx, Py, lNy))
    B2 = np.transpose(Bv, axes=(0, 2, 1, 3))

    dace.comm.Scatter(A2, tAB)
    lA[1:-1, 1:-1] = tAB
    dace.comm.Scatter(B2, tAB)
    lB[1:-1, 1:-1] = tAB

    for t in range(1, TSTEPS):

        dace.comm.Isend(lA[1, 1:-1], nn, 0, req[0])
        dace.comm.Isend(lA[-2, 1:-1], ns, 1, req[1])
        dace.comm.Isend(lA[1:-1, 1], nw, 2, req[2])
        dace.comm.Isend(lA[1:-1, -2], ne, 3, req[3])
        dace.comm.Irecv(lA[0, 1:-1], nn, 1, req[4])
        dace.comm.Irecv(lA[-1, 1:-1], ns, 0, req[5])
        dace.comm.Irecv(lA[1:-1, 0], nw, 3, req[6])
        dace.comm.Irecv(lA[1:-1, -1], ne, 2, req[7])

        dace.comm.Waitall(req)

        lB[1 + noff:-1 - soff, 1 + woff:-1 -
           eoff] = 0.2 * (lA[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                          lA[1 + noff:-1 - soff, woff:-2 - eoff] +
                          lA[1 + noff:-1 - soff, 2 + woff:-eoff] +
                          lA[2 + noff:-soff, 1 + woff:-1 - eoff] +
                          lA[noff:-2 - soff, 1 + woff:-1 - eoff])

        dace.comm.Isend(lB[1, 1:-1], nn, 0, req[0])
        dace.comm.Isend(lB[-2, 1:-1], ns, 1, req[1])
        dace.comm.Isend(lB[1:-1, 1], nw, 2, req[2])
        dace.comm.Isend(lB[1:-1, -2], ne, 3, req[3])
        dace.comm.Irecv(lB[0, 1:-1], nn, 1, req[4])
        dace.comm.Irecv(lB[-1, 1:-1], ns, 0, req[5])
        dace.comm.Irecv(lB[1:-1, 0], nw, 3, req[6])
        dace.comm.Irecv(lB[1:-1, -1], ne, 2, req[7])

        dace.comm.Waitall(req)

        lA[1 + noff:-1 - soff, 1 + woff:-1 -
           eoff] = 0.2 * (lB[1 + noff:-1 - soff, 1 + woff:-1 - eoff] +
                          lB[1 + noff:-1 - soff, woff:-2 - eoff] +
                          lB[1 + noff:-1 - soff, 2 + woff:-eoff] +
                          lB[2 + noff:-soff, 1 + woff:-1 - eoff] +
                          lB[noff:-2 - soff, 1 + woff:-1 - eoff])

    tAB[:] = lA[1:-1, 1:-1]
    dace.comm.Gather(tAB, A2)
    tAB[:] = lB[1:-1, 1:-1]
    dace.comm.Gather(tAB, B2)

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
            return init_data(N, np_float)
        else:
            return (np.empty((N, N),
                             dtype=np_float), np.empty((N, N), dtype=np_float))

    A, B = setup_func(rank)

    mpi_sdfg = None
    mpi_sdfg = jacobi_2d_dist.to_sdfg(strict=False)
    if rank == 0:
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_transformations_repeated([MapFusion])
        mpi_sdfg.apply_strict_transformations()
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dace.Config.get('default_build_folder')
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

        refA, refB = init_data(N, np_float)
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
