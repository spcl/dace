import numpy as np
import dace as dc
from mpi4py import MPI

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
from dace.transformation.dataflow import ElementWiseArrayOperation2D, RedundantComm2D


NI, NJ, NK = (
    dc.symbol(s, dtype=dc.int64, integer=True, positive=True)
    for s in ('NI', 'NJ', 'NK'))

# lNI = dc.symbol('lNI', dtype=dc.int64, integer=True, positive=True)
# lNJ = dc.symbol('lNJ', dtype=dc.int64, integer=True, positive=True)
# lNKa = dc.symbol('lNKa', dtype=dc.int64, integer=True, positive=True)
# lNKb = dc.symbol('lNKb', dtype=dc.int64, integer=True, positive=True)
# Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
# Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)

# NI = lNI * Px
# NJ = lNJ * Py
# NK = lNKa * Py  # == lNKb * Px


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def gemm(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
         A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):

    C[:] = alpha * A @ B + beta * C 


# @dc.program
# def gemm_distr(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
#                A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):

#     lA = np.zeros((lNI, lNKa), dtype=A.dtype)
#     lB = np.zeros((lNKb, lNJ), dtype=B.dtype)
#     lC = np.zeros((lNI, lNJ), dtype=A.dtype)

#     bsizesA = np.empty((2,), dtype=np.int32)
#     bsizesA[0] = lNI
#     bsizesA[1] = lNKa

#     bsizesB = np.empty((2,), dtype=np.int32)
#     bsizesB[0] = lNKb
#     bsizesB[1] = lNJ

#     bsizesC = np.empty((2,), dtype=np.int32)
#     bsizesC[0] = lNI
#     bsizesC[1] = lNJ

#     gdescA, ldescA = dc.comm.BCScatter(A, lA, bsizesA)
#     gdescB, ldescB = dc.comm.BCScatter(B, lB, bsizesB)
#     gdescC, ldescC = dc.comm.BCScatter(C, lC, bsizesC)

#     tmp, gdesctmp, ldesctmp = distr.MatMult(lA, ldescA, lB, ldescB)

#     lC[:] = alpha * tmp + beta * lC

#     dc.comm.BCGather(lC, C, ldescC, gdescC)


def init_data(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.empty((NI, NJ), dtype=datatype)
    for i in range(NI):
        for j in range(NJ):
            C[i, j] = ((i * j + 1) % NI) / NI
    A = np.empty((NI, NK), dtype=datatype)
    for i in range(NI):
        for k in range(NK):
            A[i, k] = (i * (k + 1) % NK) / NK
    B = np.empty((NK, NJ), dtype=datatype)
    for k in range(NK):
        for j in range(NJ):
            B[i, j] = (k * (j + 2) % NJ) / NJ
    # C = np.zeros((NI, NJ), dtype=datatype)
    # A = np.arange(0, NI*NK, dtype=datatype).reshape(NI, NK)
    # B = np.ones((NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


if __name__ == "__main__":

    # Initialization
    NI, NJ, NK = 2000, 2300, 2600
    # NI, NJ, NK = 8, 8, 8
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # if rank == 0:
    #     print("----B-----")
    #     print(B)
    #     print()

    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")

    if rank == 0:
        mpi_sdfg = gemm.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_transformations_repeated(ElementWiseArrayOperation2D)
        mpi_sdfg.expand_library_nodes()
        mpi_sdfg.apply_strict_transformations()
        # mpi_sdfg.apply_transformations_repeated(RedundantComm2D)
        # mpi_sdfg.apply_strict_transformations()
        mpi_func = mpi_sdfg.compile()

        # print("----A-----")
        # print(A)
        # print()
        # print("----B-----")
        # print(B)
        # print()
        # print("----C-----")
        # print(C)
        # print()

    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(n=gemm.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gemm.name), gemm.name))
    comm.Barrier()
  
    Px = Py = int(np.sqrt(size))
    mpi_func(A=A, B=B, C=C, alpha=alpha, beta=beta,
             NI=NI, NJ=NJ, NK=NK, commsize=size, Px=Px, Py=Py)

    comm.Barrier()

    if rank == 0:
        alpha, beta, refC, refA, refB = init_data(NI, NJ, NK, np.float64)

        # print("----refB-----")
        # print(refB)
        # print()

        refC[:] = alpha * refA @ refB + beta * refC
        print(relerr(refC, C))

        # print("----refC-----")
        # print(refC)
        # print()
        # print("----C-----")
        # print(C)
        # print()
    # if rank == 0:
    #     alpha, beta, refC, refA, refB = init_data(NI, NJ, NK, np.float64)
    #     shared_sdfg = gemm_shared.compile()
    #     shared_sdfg(A=refA, B=refB, C=refC, alpha=alpha, beta=beta,
    #                 NI=NI, NJ=NJ, NK=NK,
    #                 lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb,
    #                 Px=Px, Py=Py)

    #     print(relerr(refC, C))