import numpy as np
import dace as dc
from mpi4py import MPI


N = dc.symbol('N', dtype=dc.int64)
lN = dc.symbol('lN', dtype=dc.int64)
Px = dc.symbol('Px', dtype=dc.int32)
Py = dc.symbol('Py', dtype=dc.int32)
pi = dc.symbol('pi', dtype=dc.int32)
pj = dc.symbol('pj', dtype=dc.int32)
rank = dc.symbol('rank', dtype=dc.int32)
size = dc.symbol('size', dtype=dc.int32)
noff = dc.symbol('noff', dtype=dc.int32)
soff = dc.symbol('soff', dtype=dc.int32)
woff = dc.symbol('woff', dtype=dc.int32)
eoff = dc.symbol('eoff', dtype=dc.int32)


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def jacobi_2d_shared(TSTEPS: dc.int64, A: dc.float64[N, N], B: dc.float64[N, N]):
    
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] +
                                 A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] +
                                 B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


@dc.program
def jacobi_2d_dist(TSTEPS: dc.int64, A: dc.float64[N, N], B: dc.float64[N, N]):

    lA = np.zeros((lN + 2, lN + 2), dtype=A.dtype)
    lB = np.zeros((lN + 2, lN + 2), dtype=B.dtype)
    tAB = np.empty((lN, lN), dtype=A.dtype)
    rn = np.empty((lN,), dtype=A.dtype)
    rs = np.empty((lN,), dtype=A.dtype)
    rw = np.empty((lN,), dtype=A.dtype)
    re = np.empty((lN,), dtype=A.dtype)

    dace.comm.Scatter(A, tAB)
    lA[1:-1, 1:-1] = tAB
    dace.comm.Scatter(B, tAB)
    lB[1:-1, 1:-1] = tAB
    
    for t in range(1, TSTEPS):
        # NORTH
        dace.comm.Send(lA[1,:], (pi-1)*Px + pj, t)
        dace.comm.Recv(rn, (pi-1)*Px + pj, t)
        lA[0, 1:-1] = rn
        # SOUTH
        dace.comm.Recv(rs, (pi+1)*Px + pj, t)
        dace.comm.Send(lA[-2,:], (pi+1)*Px + pj, t)
        lA[-1, 1:-1] = rs
        # WEST
        dace.comm.Send(lA[:, 1], pi*Px + (pj-1), t)
        dace.comm.Recv(rw, pi*Px + (pj-1), t)
        lA[1:-1, 0] = rw
        # EAST
        dace.comm.Recv(re, pi*Px + (pj+1), t)
        dace.comm.Send(lA[:, -2], pi*Px + (pj+1), t)
        lA[1:-1, -1] = re

        lB[1+noff:-1-soff, 1+woff:-1-eoff] = 0.2 * (
            lA[1+noff:-1-soff, 1+woff:-1-eoff] +
            lA[1+noff:-1-soff, woff:-2-eoff] +
            lA[1+noff:-1-soff, 2+woff:-eoff] +
            lA[2+noff:-soff, 1+woff:-1-eoff] +
            lA[noff:-2-soff, 1+woff:-1-eoff])

        # NORTH
        dace.comm.Send(lB[1,:], (pi-1)*Px + pj, t)
        dace.comm.Recv(rn, (pi-1)*Px + pj, t)
        lB[0, 1:-1] = rn
        # SOUTH
        dace.comm.Recv(rs, (pi+1)*Px + pj, t)
        dace.comm.Send(lB[-2,:], (pi+1)*Px + pj, t)
        lB[-1, 1:-1] = rs
        # WEST
        dace.comm.Send(lB[:, 1], pi*Px + (pj-1), t)
        dace.comm.Recv(rw, pi*Px + (pj-1), t)
        lB[1:-1, 0] = rw
        # EAST
        dace.comm.Recv(re, pi*Px + (pj+1), t)
        dace.comm.Send(lB[:, -2], pi*Px + (pj+1), t)
        lB[1:-1, -1] = re

        lA[1+noff:-1-soff, 1+woff:-1-eoff] = 0.2 * (
            lB[1+noff:-1-soff, 1+woff:-1-eoff] +
            lB[1+noff:-1-soff, woff:-2-eoff] +
            lB[1+noff:-1-soff, 2+woff:-eoff] +
            lB[2+noff:-soff, 1+woff:-1-eoff] +
            lB[noff:-2-soff, 1+woff:-1-eoff])
    
    tAB[:] = lA[1:-1, 1:-1]
    dace.comm.Gather(tAB, A)
    tAB[:] = lB[1:-1, 1:-1]
    dace.comm.Gather(tAB, B)


def init_data(N, datatype):

    A = np.empty((N, N), dtype=datatype)
    B = np.empty((N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            A[i, j] = i * (j + 2) / N
            B[i, j] = i * (j + 3) / N

    return A, B


if __name__ == "__main__":

    # Initialization
    TSTEPS, N = 20, 100  # 500, 1300  # 1000, 2800
    A, B = init_data(N, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    pi = rank // Px
    pj = rank % Py
    lN = N // Px
    noff = soff = woff = eoff = 0
    if pi == 0:
        noff = 1
    if pi == Px - 1:
        soff = 1
    if pj == 0:
        woff = 1
    if pj == Py - 1:
        soff = 1

    mpi_sdfg = None
    if size < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = jacobi_2d_dist.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, B=B, TSTEPS=TSTEPS, N=N, lN=lN, rank=rank, size=size,
             Px=Px, Py=Py, pi=pi, pj=pj, noff=noff, soff=soff, woff=woff, eoff=eoff)

    if rank == 0:
        refA, refB = init_data(N, np.float64)
        # print(refA)
        # print(refB)
        # print()
        jacobi_2d_shared(TSTEPS, refA, refB)

        print(relerr(refA, A))
        print(relerr(refB, B))
        # print()
        # print(refA)
        # print(A)
        # print()
        # print(refB)
        # print(B)
