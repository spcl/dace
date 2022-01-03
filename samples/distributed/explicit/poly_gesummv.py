# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed Gesummv sample."""
import dace as dc
import numpy as np
import os
from dace.sdfg.utils import load_precompiled_sdfg
from mpi4py import MPI

lM = dc.symbol('lM', dtype=dc.int64, integer=True, positive=True)
lN = dc.symbol('lN', dtype=dc.int64, integer=True, positive=True)
lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
lMy = dc.symbol('lMy', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
M = lM * Px
N = lN * Py  # == lNx * Px


def relerr(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


@dc.program
def gesummv_shared(alpha: dc.float64, beta: dc.float64, A: dc.float64[M, N],
                   B: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M]):

    y[:] = alpha * A @ x + beta * B @ x


@dc.program
def gesummv_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[M, N],
                  B: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M]):

    lA = np.empty((lM, lN), dtype=A.dtype)
    lB = np.empty((lM, lN), dtype=B.dtype)
    lx = np.empty((lNx, ), dtype=x.dtype)

    Av = np.reshape(A, (Px, lM, Py, lN))
    A2 = np.transpose(Av, axes=(0, 2, 1, 3))
    Bv = np.reshape(B, (Px, lM, Py, lN))
    B2 = np.transpose(Bv, axes=(0, 2, 1, 3))
    dc.comm.BCScatter(x, lx, (lNx, 1))

    tmp1 = distr.MatMult(lA, lx, (M, N), b_block_sizes=(lNx, 1))
    tmp2 = distr.MatMult(lB, lx, (M, N), b_block_sizes=(lNx, 1))

    tmp1[:] = alpha * tmp1 + beta * tmp2

    dc.comm.BCGather(tmp1, y, (lM, 1))


@dc.program
def gesummv_distr2(alpha: dc.float64, beta: dc.float64, A: dc.float64[lM, lN],
                   B: dc.float64[lM,
                                 lN], x: dc.float64[lN], y: dc.float64[lMy]):

    tmp1 = distr.MatMult(A, x, (Px * lM, Py * lN), c_block_sizes=(lMy, 1))
    tmp2 = distr.MatMult(B, x, (M, N), c_block_sizes=(lMy, 1))
    y[:] = alpha * tmp1 + beta * tmp2


def init_data(M, N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    rng = np.random.default_rng(42)
    A = rng.random((M, N), dtype=datatype)
    B = rng.random((M, N), dtype=datatype)
    x = rng.random((N, ), dtype=datatype)
    y = rng.random((M, ), dtype=datatype)

    return alpha, beta, A, B, x, y


def time_to_ms(raw):
    return int(round(raw * 1000))


grid = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2), 16: (4, 4)}

if __name__ == "__main__":

    # Initialization
    M, N = 6400, 5600

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    lM = M // Px
    lN = N // Py
    lMy = M // Py

    def setup_func(rank):
        if rank == 0:
            return init_data(M, N, np.float64)
        else:
            return (1.5, 1.2, None, None, np.empty((N, ),
                                                   dtype=np.float64), None)

    alpha, beta, A, B, x, y = setup_func(rank)

    lA = np.empty((lM, lN), dtype=np.float64)
    lB = np.empty((lM, lN), dtype=np.float64)
    lx = np.empty((lN, ), dtype=np.float64)
    ly = np.zeros((lMy, ), dtype=np.float64)

    A2, B2 = None, None
    if rank == 0:
        Av = np.reshape(A, (Px, lM, Py, lN))
        A2 = np.transpose(Av, axes=(0, 2, 1, 3)).copy()
        Bv = np.reshape(B, (Px, lM, Py, lN))
        B2 = np.transpose(Bv, axes=(0, 2, 1, 3)).copy()
    comm.Scatter(A2, lA)
    comm.Scatter(B2, lB)

    comm.Bcast(x, root=0)
    pi = rank // Py
    pj = rank % Py
    lx[:] = x[pj * lN:(pj + 1) * lN]

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = gesummv_distr2.to_sdfg(coarsen=False)
        mpi_sdfg.coarsen_dataflow()
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dc.Config.get('default_build_folder')
        mpi_func = load_precompiled_sdfg(
            os.path.join(build_folder, gesummv_distr2.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA,
             B=lB,
             x=lx,
             alpha=alpha,
             beta=beta,
             y=ly,
             lM=lM,
             lN=lN,
             lMy=lMy,
             Px=Px,
             Py=Py)

    # print(rank, ly)

    if rank == 0:
        y[0:lMy] = ly
        for i in range(Py):
            if i == pj:
                continue
            comm.Recv(ly, source=i, tag=i)
            y[i * lMy:(i + 1) * lMy] = ly
    elif pi == 0:
        comm.Send(ly, dest=0, tag=pj)

    comm.Barrier()

    stmt = ("mpi_func(A=lA, B=lB, x=lx, alpha=alpha, beta=beta, y=ly, "
            "lM=lM, lN=lN, lMy=lMy, Px=Px, Py=Py)")
    setup = "comm.Barrier()"
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

        alpha, beta, refA, refB, refx, refy = init_data(M, N, np.float64)
        shared_sdfg = gesummv_shared.compile()
        refout = shared_sdfg(A=refA,
                             B=refB,
                             x=refx,
                             alpha=alpha,
                             beta=beta,
                             y=refy,
                             lM=lM,
                             lN=lN,
                             lNx=lNx,
                             Px=Px,
                             Py=Py)

        print("=======Validation=======")
        assert (np.allclose(refy, y))
        print("OK")
