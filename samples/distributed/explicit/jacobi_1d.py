import numpy as np
import dace as dc
from mpi4py import MPI


N = dc.symbol('N', dtype=dc.int64)
lN = dc.symbol('lN', dtype=dc.int64)
rank = dc.symbol('rank', dtype=dc.int32)
size = dc.symbol('size', dtype=dc.int32)


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def jacobi_1d_shared(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):
    
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@dc.program
def jacobi_1d_dist(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):

    lA = np.zeros((lN + 2,), dtype=A.dtype)
    lB = np.zeros((lN + 2,), dtype=B.dtype)
    tAB = np.empty((lN, ), dtype=A.dtype)
    rl = np.empty((1,), dtype=A.dtype)
    rr = np.empty((1,), dtype=A.dtype)

    dace.comm.Scatter(A, tAB)
    lA[1:-1] = tAB
    dace.comm.Scatter(B, tAB)
    lB[1:-1] = tAB
    
    for t in range(1, TSTEPS):
        if rank > 0:
            dace.comm.Recv(rl, rank - 1, t)
            lA[0] = rl
            dace.comm.Send(lA[1], rank - 1, t)
        if rank < size - 1:
            dace.comm.Send(lA[-2], rank + 1, t)
            dace.comm.Recv(rr, rank + 1, t)
            lA[-1] = rr
        if rank == 0:
            lB[2:-1] = 0.33333 * (lA[1:-2] + lA[2:-1] + lA[3:])
        elif rank == size - 1:
            lB[1:-2] = 0.33333 * (lA[:-3] + lA[1:-2] + lA[2:-1])
        else:
            lB[1:-1] = 0.33333 * (lA[:-2] + lA[1:-1] + lA[2:])
        if rank > 0:
            dace.comm.Recv(rl, rank - 1, t)
            lB[0] = rl
            dace.comm.Send(lB[1], rank - 1, t)
        if rank < size - 1:
            dace.comm.Send(lB[-2], rank + 1, t)
            dace.comm.Recv(rr, rank + 1, t)
            lB[-1] = rr
        if rank == 0:
            lA[2:-1] = 0.33333 * (lB[1:-2] + lB[2:-1] + lB[3:])
        elif rank == size - 1:
            lA[1:-2] = 0.33333 * (lB[:-3] + lB[1:-2] + lB[2:-1])
        else:
            lA[1:-1] = 0.33333 * (lB[:-2] + lB[1:-1] + lB[2:])
    
    tAB[:] = lA[1:-1]
    dace.comm.Gather(tAB, A)
    tAB[:] = lB[1:-1]
    dace.comm.Gather(tAB, B)


def init_data(N, datatype):

    A = np.empty((N, ), dtype=datatype)
    B = np.empty((N, ), dtype=datatype)
    for i in range(N):
        A[i] = (i + 2) / N
        B[i] = (i + 3) / N

    return A, B


if __name__ == "__main__":

    # Initialization
    TSTEPS, N = 1000, 4000
    A, B = init_data(N, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    lN = N // size

    mpi_sdfg = None
    if size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = jacobi_1d_dist.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, B=B, TSTEPS=TSTEPS, N=N, lN=lN, rank=rank, size=size)

    if rank == 0:
        refA, refB = init_data(N, np.float64)
        # print(refA)
        # print(refB)
        # print()
        jacobi_1d_shared(TSTEPS, refA, refB)

        print(relerr(refA, A))
        print(relerr(refB, B))
        # print()
        # print(refA)
        # print(A)
        # print()
        # print(refB)
        # print(B)
