import numpy as np
import dace as dc
from mpi4py import MPI

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL


# NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))

lNI = dc.symbol('lNI', dtype=dc.int64, integer=True, positive=True)
lNJ = dc.symbol('lNJ', dtype=dc.int64, integer=True, positive=True)
lNKa = dc.symbol('lNKa', dtype=dc.int64, integer=True, positive=True)
lNKb = dc.symbol('lNKb', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)

NI = lNI * Px
NJ = lNJ * Py
NK = lNKa * Py  # == lNKb * Px


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


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

    # bsizesA = np.empty((2,), dtype=np.int32)
    # bsizesA[0] = lNI
    # bsizesA[1] = lNKa

    # bsizesB = np.empty((2,), dtype=np.int32)
    # bsizesB[0] = lNKb
    # bsizesB[1] = lNJ

    # bsizesC = np.empty((2,), dtype=np.int32)
    # bsizesC[0] = lNI
    # bsizesC[1] = lNJ

    # gdescA, ldescA = dc.comm.BCScatter(A, lA, bsizesA)
    # gdescB, ldescB = dc.comm.BCScatter(B, lB, bsizesB)
    # gdescC, ldescC = dc.comm.BCScatter(C, lC, bsizesC)
    dc.comm.BCScatter(A, lA, (lNI, lNKa))
    dc.comm.BCScatter(B, lB, (lNKb, lNJ))
    dc.comm.BCScatter(C, lC, (lNI, lNJ))

    # tmp, gdesctmp, ldesctmp = distr.MatMult(lA, ldescA, lB, ldescB)
    tmp  = distr.MatMult(A, B, lA, lB, (lNI, lNKa), (lNKb, lNJ))

    lC[:] = alpha * tmp + beta * lC

    # dc.comm.BCGather(lC, C, ldescC, gdescC)
    dc.comm.BCGather(lC, C, (lNI, lNJ))


def init_data(NI, NJ, NK, datatype):

    from dace.libraries.standard.memory import aligned_ndarray

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = aligned_ndarray(np.empty((NI, NJ), dtype=datatype), alignment=4096)
    for i in range(NI):
        for j in range(NJ):
            C[i, j] = ((i * j + 1) % NI) / NI
    A = aligned_ndarray(np.empty((NI, NK), dtype=datatype), alignment=4096)
    for i in range(NI):
        for k in range(NK):
            A[i, k] = (i * (k + 1) % NK) / NK
    B = aligned_ndarray(np.empty((NK, NJ), dtype=datatype), alignment=4096)
    for k in range(NK):
        for j in range(NJ):
            C[i, j] = (k * (j + 2) % NJ) / NJ

    return alpha, beta, C, A, B


if __name__ == "__main__":

    # Initialization
    NI, NJ, NK = 2000, 2300, 2600  # 4000, 4600, 5200
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    lNI = NI // Px
    lNJ = NJ // Py
    lNKa = NK // Py
    lNKb = NK // Px

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    if rank == 0:
        mpi_sdfg = gemm_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=gemm_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gemm_distr.name),
            gemm_distr.name))
    comm.Barrier()
  
    mpi_func(A=A, B=B, C=C, alpha=alpha, beta=beta,
             NI=NI, NJ=NJ, NK=NK,
             lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb,
             Px=Px, Py=Py)

    comm.Barrier()

    if rank == 0:
        alpha, beta, refC, refA, refB = init_data(NI, NJ, NK, np.float64)
        shared_sdfg = gemm_shared.compile()
        shared_sdfg(A=refA, B=refB, C=refC, alpha=alpha, beta=beta,
                    NI=NI, NJ=NJ, NK=NK,
                    lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb,
                    Px=Px, Py=Py)

        print(relerr(refC, C))