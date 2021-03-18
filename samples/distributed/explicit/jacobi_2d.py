import numpy as np
import dace as dc
from mpi4py import MPI


# N = dc.symbol('N', dtype=dc.int64, integer=True, positive=True)
lN = dc.symbol('lN', dtype=dc.int64, integer=True, positive=True)
# Px = Py = N / lN
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
# Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
pi = dc.symbol('pi', dtype=dc.int32, integer=True, nonnegative=True)
pj = dc.symbol('pj', dtype=dc.int32, integer=True, nonnegative=True)
rank = dc.symbol('rank', dtype=dc.int32, integer=True, nonnegative=True)
size = dc.symbol('size', dtype=dc.int32, integer=True, positive=True)
noff = dc.symbol('noff', dtype=dc.int32, integer=True, nonnegative=True)
soff = dc.symbol('soff', dtype=dc.int32, integer=True, nonnegative=True)
woff = dc.symbol('woff', dtype=dc.int32, integer=True, nonnegative=True)
eoff = dc.symbol('eoff', dtype=dc.int32, integer=True, nonnegative=True)
N = Px * lN

nn = dc.symbol('nn', dtype=dc.int32, integer=True)
ns = dc.symbol('ns', dtype=dc.int32, integer=True)
nw = dc.symbol('nw', dtype=dc.int32, integer=True)
ne = dc.symbol('ne', dtype=dc.int32, integer=True)

MPI_Request = dc.opaque("MPI_Request")


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

    req = np.empty((8,), dtype=MPI_Request)

    Av = np.reshape(A, (Px, lN, Px, lN))
    A2 = np.transpose(Av, axes=(0, 2, 1, 3))
    Bv = np.reshape(B, (Px, lN, Px, lN))
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

        lB[1+noff:-1-soff, 1+woff:-1-eoff] = 0.2 * (
            lA[1+noff:-1-soff, 1+woff:-1-eoff] +
            lA[1+noff:-1-soff, woff:-2-eoff] +
            lA[1+noff:-1-soff, 2+woff:-eoff] +
            lA[2+noff:-soff, 1+woff:-1-eoff] +
            lA[noff:-2-soff, 1+woff:-1-eoff])

        dc.comm.Isend(lB[1, 1:-1], nn, 0, req[0])
        dc.comm.Isend(lB[-2, 1:-1], ns, 1, req[1])
        dc.comm.Isend(lB[1:-1, 1], nw, 2, req[2])
        dc.comm.Isend(lB[1:-1, -2], ne, 3, req[3])
        dc.comm.Irecv(lB[0, 1:-1], nn, 1, req[4])
        dc.comm.Irecv(lB[-1, 1:-1], ns, 0, req[5])
        dc.comm.Irecv(lB[1:-1, 0], nw, 3, req[6])
        dc.comm.Irecv(lB[1:-1, -1], ne, 2, req[7])

        dc.comm.Waitall(req)

        lA[1+noff:-1-soff, 1+woff:-1-eoff] = 0.2 * (
            lB[1+noff:-1-soff, 1+woff:-1-eoff] +
            lB[1+noff:-1-soff, woff:-2-eoff] +
            lB[1+noff:-1-soff, 2+woff:-eoff] +
            lB[2+noff:-soff, 1+woff:-1-eoff] +
            lB[noff:-2-soff, 1+woff:-1-eoff])
    
    tAB[:] = lA[1:-1, 1:-1]
    dc.comm.Gather(tAB, A2)
    tAB[:] = lB[1:-1, 1:-1]
    dc.comm.Gather(tAB, B2)

    A[:] = np.transpose(A2, (0, 2, 1, 3))
    B[:] = np.transpose(B2, (0, 2, 1, 3))


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
    TSTEPS, N = 2, 8  # 500, 1300  # 1000, 2800
    # A, B = init_data(N, np.float64)
    A = np.arange(0, N*N).reshape(N, N).astype(np.float64)
    B = np.zeros((N, N), dtype=np.float64)
    # B = np.arange(0, N*N).reshape(N, N)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    pi = rank // Px
    pj = rank % Px
    lN = N // Px
    noff = soff = woff = eoff = 0
    nn = (pi-1)*Px + pj
    ns = (pi+1)*Px + pj
    nw = pi*Px + (pj-1)
    ne = pi*Px + (pj+1)
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

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            print("I am %d" % rank)
            print(nn, ns, nw, ne)
            print(noff, soff, woff, eoff)
            print("=====================")
            mpi_sdfg = jacobi_2d_dist.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, B=B, TSTEPS=TSTEPS, N=N, lN=lN, rank=rank, size=size,
             Px=Px, Py=Py, pi=pi, pj=pj,
             noff=noff, soff=soff, woff=woff, eoff=eoff,
             nn=nn, ns=ns, nw=nw, ne=ne)

    if rank == 0:
        # refA, refB = init_data(N, np.float64)
        refA = np.arange(0, N*N).reshape(N, N).astype(np.float64)
        refB = np.zeros((N, N), dtype=np.float64)
        print(refA)
        print(refB)
        print()
        # jacobi_2d_shared(TSTEPS, refA, refB)
        shared_sdfg = jacobi_2d_shared.compile()
        shared_sdfg(A=refA, B=refB, TSTEPS=TSTEPS, N=N, lN=lN, Px=Px)

        print(relerr(refA, A))
        print(relerr(refB, B))
        print()
        print(refA)
        print(A)
        print()
        print(refB)
        print(B)
