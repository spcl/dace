# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly distributed Gemm sample with block-cyclic distribution."""
import dace as dc
import numpy as np
import os
import timeit
from dace.sdfg.utils import load_precompiled_sdfg
from mpi4py import MPI

lNI = dc.symbol('lNI', dtype=dc.int64, integer=True, positive=True)
lNJ = dc.symbol('lNJ', dtype=dc.int64, integer=True, positive=True)
lNKa = dc.symbol('lNKa', dtype=dc.int64, integer=True, positive=True)
lNKb = dc.symbol('lNKb', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
Bx = dc.symbol('Bx', dtype=dc.int32, integer=True, positive=True)
By = dc.symbol('By', dtype=dc.int32, integer=True, positive=True)

NI = lNI * Px
NJ = lNJ * Py
NK = lNKa * Py  # == lNKb * Px


def relerr(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


@dc.program
def gemm_shared(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
                A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):

    C[:] = alpha * A @ B + beta * C


@dc.program
def gemm_distr(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
               A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):

    lA = np.empty((lNI, lNKa), dtype=A.dtype)
    lB = np.empty((lNKb, lNJ), dtype=B.dtype)
    lC = np.empty((lNI, lNJ), dtype=A.dtype)

    Av = np.reshape(A, (Px, lNI, Py, lNKa))
    A2 = np.transpose(Av, axes=(0, 2, 1, 3))
    Bv = np.reshape(B, (Px, lNKb, Py, lNJ))
    B2 = np.transpose(Bv, axes=(0, 2, 1, 3))
    Cv = np.reshape(C, (Px, lNI, Py, lNJ))
    C2 = np.transpose(Cv, axes=(0, 2, 1, 3))
    dc.comm.Scatter(A2, lA)
    dc.comm.Scatter(B2, lB)
    dc.comm.Scatter(C2, lC)

    tmp = distr.MatMult(lA, lB, (NI, NJ, NK))

    lC[:] = alpha * tmp + beta * lC

    dc.comm.Gather(lC, C2)
    C[:] = np.transpose(C2, (0, 2, 1, 3))


@dc.program
def gemm_distr2(alpha: dc.float64, beta: dc.float64, C: dc.float64[lNI, lNJ],
                A: dc.float64[lNI, lNKa], B: dc.float64[lNKb, lNJ]):

    tmp = distr.MatMult(A, B, (lNI * Px, lNJ * Py, NK))
    C[:] = alpha * tmp + beta * C


@dc.program
def gemm_distr3(alpha: dc.float64, beta: dc.float64, C: dc.float64[lNI, lNJ],
                A: dc.float64[lNI, lNKa], B: dc.float64[lNKb, lNJ]):

    tmp = distr.MatMult(A, B, (lNI * Px, lNJ * Py, NK), (Bx, By), (Bx, By))
    C[:] = alpha * tmp + beta * C


def init_data(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    rng = np.random.default_rng(42)
    C = np.zeros((NI, NJ), dtype=datatype)
    A = np.arange(0, NI * NK, dtype=datatype).reshape(NI, NK)
    B = np.arange(NI * NK, NI * NK + NK * NJ, dtype=datatype).reshape(NK, NJ)

    return alpha, beta, C, A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


grid = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2), 16: (4, 4)}

if __name__ == "__main__":

    # Initialization
    NI, NJ, NK = 4096, 4096, 4096

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    Bx, By = 128, 128
    lNI = NI // Px
    lNJ = NJ // Py
    lNKa = NK // Py
    lNKb = NK // Px

    BI = NI // (Px * Bx)
    BJ = NJ // (Py * By)
    BKa = NK // (Py * By)
    BKb = NK // (Px * Bx)

    def setup_func(rank):
        if rank == 0:
            return init_data(NI, NJ, NK, np.float64)
        else:
            return (1.5, 1.2, None, None, None)

    alpha, beta, C, A, B = setup_func(rank)

    lA = np.empty((lNI, lNKa), dtype=np.float64)
    lB = np.empty((lNKb, lNJ), dtype=np.float64)
    lC = np.empty((lNI, lNJ), dtype=np.float64)

    A2, B2, C2 = None, None, None
    if rank == 0:

        A2 = np.empty((Px, Py, lNI, lNKa), dtype=np.float64)
        for pi in range(Px):
            for pj in range(Py):
                for li in range(BI):
                    for lj in range(BKa):
                        si = (pi + li * Px) * Bx
                        sj = (pj + lj * Py) * By
                        A2[pi, pj, li * Bx:li * Bx + Bx,
                           lj * By:lj * By + By] = A[si:si + Bx, sj:sj + By]

        B2 = np.empty((Px, Py, lNKb, lNJ), dtype=np.float64)
        for pi in range(Px):
            for pj in range(Py):
                for li in range(BKb):
                    for lj in range(BJ):
                        si = (pi + li * Px) * Bx
                        sj = (pj + lj * Py) * By
                        B2[pi, pj, li * Bx:li * Bx + Bx,
                           lj * By:lj * By + By] = B[si:si + Bx, sj:sj + By]

        C2 = np.empty((Px, Py, lNI, lNJ), dtype=np.float64)
        for pi in range(Px):
            for pj in range(Py):
                for li in range(BI):
                    for lj in range(BJ):
                        si = (pi + li * Px) * Bx
                        sj = (pj + lj * Py) * By
                        C2[pi, pj, li * Bx:li * Bx + Bx,
                           lj * By:lj * By + By] = C[si:si + Bx, sj:sj + By]

    comm.Scatter(A2, lA)
    comm.Scatter(B2, lB)
    comm.Scatter(C2, lC)

    tC = np.copy(lC)

    mpi_sdfg = None
    if rank == 0:
        mpi_sdfg = gemm_distr3.to_sdfg(coarsen=False)
        mpi_sdfg.coarsen_dataflow()
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dc.Config.get('default_build_folder')
        mpi_func = load_precompiled_sdfg(
            os.path.join(build_folder, gemm_distr3.name))

    ldict = locals()

    comm.Barrier()

    mpi_func(A=lA,
             B=lB,
             C=tC,
             alpha=alpha,
             beta=beta,
             NI=NI,
             NJ=NJ,
             NK=NK,
             lNI=lNI,
             lNJ=lNJ,
             lNKa=lNKa,
             lNKb=lNKb,
             Px=Px,
             Py=Py,
             Bx=Bx,
             By=By)

    comm.Gather(tC, C2)
    if rank == 0:
        for pi in range(Px):
            for pj in range(Py):
                for li in range(BI):
                    for lj in range(BJ):
                        si = (pi + li * Px) * Bx
                        sj = (pj + lj * Py) * By
                        C[si:si + Bx,
                          sj:sj + By] = C2[pi, pj, li * Bx:li * Bx + Bx,
                                           lj * By:lj * By + By]

    comm.Barrier()

    stmt = ("mpi_func(A=lA, B=lB, C=tC, alpha=alpha, beta=beta, "
            "NI=NI, NJ=NJ, NK=NK, lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb, "
            "Px=Px, Py=Py, Bx=Bx, By=By)")
    setup = "tC = np.copy(lC); comm.Barrier()"
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

        alpha, beta, refC, refA, refB = init_data(NI, NJ, NK, np.float64)
        shared_sdfg = gemm_shared.compile()
        shared_sdfg(A=refA,
                    B=refB,
                    C=refC,
                    alpha=alpha,
                    beta=beta,
                    NI=NI,
                    NJ=NJ,
                    NK=NK,
                    lNI=lNI,
                    lNJ=lNJ,
                    lNKa=lNKa,
                    lNKb=lNKb,
                    Px=Px,
                    Py=Py)

        print("=======Validation=======")
        assert (np.allclose(C, refC))
        print("OK")
