import numpy as np
import dace as dc
from mpi4py import MPI


lM = dc.symbol('lM', dtype=dc.int64, integer=True, positive=True)
lN = dc.symbol('lN', dtype=dc.int64, integer=True, positive=True)
lKa = dc.symbol('lKa', dtype=dc.int64, integer=True, positive=True)
lKb = dc.symbol('lKb', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
M = Px * lM
K = Px * lKb  # == Py * lKA
N = Py * lN


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def matmult_shared(A: dc.float64[M, K], B: dc.float64[K, N], C: dc.float64[M, N]):
    
    C[:] = A @ B


@dc.program
def matmult_dist(A: dc.float64[M, K], B: dc.float64[K, N], C: dc.float64[M, N]):

    lA = np.zeros((lM, lKa), dtype=A.dtype)
    lB = np.zeros((lKb, lN), dtype=B.dtype)

    bsizesA = np.empty((2,), dtype=np.int32)
    bsizesA[0] = M // Px
    bsizesA[1] = K // Py

    bsizesB = np.empty((2,), dtype=np.int32)
    bsizesB[0] = K // Px
    bsizesB[1] = N // Py

    gdescA, ldescA = dc.comm.BCScatter(A, lA, bsizesA)
    gdescB, ldescB = dc.comm.BCScatter(B, lB, bsizesB)

    lC, gdescC, ldescC = distr.MatMult(lA, ldescA, lB, ldescB)

    dc.comm.BCGather(lC, C, ldescC, gdescC)


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
    M, N, K = 2000, 1000, 3000
    # _, _, C, A, B = init_data(M, N, K, np.float64)
    # M, N, K = 8, 4, 16
    A = np.arange(0, M*K).reshape(M, K).astype(np.float64)
    B = np.arange(0, K*N).reshape(K, N).astype(np.float64)
    C = np.zeros((M, N), dtype=np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    lM = M // Px
    lKa = K // Py
    lKb = K // Px
    lN = N // Py

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = matmult_dist.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, B=B, C=C, M=M, N=N, K=K, lM=lM, lKa=lKa, lKb=lKb, lN=lN, Px=Px, Py=Py)

    # for r in range(0, size):
    #     if r == rank:
    #         print("I am %d" % rank)
    #         print(A)
    #     comm.Barrier()

    if rank == 0:
        # _, _, refC, refA, refB = init_data(M, N, K, np.float64)
        refA = np.arange(0, M*K).reshape(M, K).astype(np.float64)
        refB = np.arange(0, K*N).reshape(K, N).astype(np.float64)
        refC = np.zeros((M, N), dtype=np.float64)
        shared_sdfg = matmult_shared.compile()
        shared_sdfg(A=refA, B=refB, C=refC, M=M, N=N, K=K, lM=lM, lKa=lKa, lKb=lKb, lN=lN, Px=Px, Py=Py)

        print(relerr(refC, C))

        # print(refC)
        # print(C)
        # print()
        # print(refA)
        # print(A)
        # print()
        # print(refB)
        # print(B)
