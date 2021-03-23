import numpy as np
import dace as dc
from mpi4py import MPI


lMA = dc.symbol('lMA', dtype=dc.int64, integer=True, positive=True)
lMz = dc.symbol('lMz', dtype=dc.int64, integer=True, positive=True)
lNA = dc.symbol('lNA', dtype=dc.int64, integer=True, positive=True)
lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
M = Px * lMA
N = Py * lNA


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def matvec_shared(A: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M], z: dc.float64[N]):
    
    y[:] = A @ x
    z[:] = y @ A


@dc.program
def matvec_dist(A: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M], z: dc.float64[N]):

    lA = np.zeros((lMA, lNA), dtype=A.dtype)
    lx = np.zeros((lNA,), dtype=x.dtype)

    bsizesA = np.empty((2,), dtype=np.int32)
    bsizesA[0] = M // Px
    bsizesA[1] = N // Py

    bsizesx = np.empty((1,), dtype=np.int32)
    bsizesx[0] = N // (Px * Py)
    gdescA, ldescA = dc.comm.BCScatter(A, lA, bsizesA)
    gdescx, ldescx = dc.comm.BCScatter(x, lx, bsizesx)

    ly, gdescy, ldescy = distr.MatMult(lA, ldescA, lx, ldescx)
    dc.comm.BCGather(ly, y, ldescy, gdescy)

    lz, gdescz, ldescz = distr.MatMult(ly, ldescy, lA, ldescA)
    dc.comm.BCGather(lz, z, ldescz, gdescz)


if __name__ == "__main__":

    # Initialization
    M, N = 2000, 1000
    # M, N = 16, 8
    A = np.arange(0, M*N).reshape(M, N).astype(np.float64)
    x = np.arange(0, N).astype(np.float64)
    y = np.zeros((M, ), dtype=np.float64)
    z = np.zeros((N, ), dtype=np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    lMA = M // Px
    lNA = N // Py
    lMz = M // (Px * Py)
    lNx = N // (Px * Py)

    if rank == 0:
        print(lMA, lNA, lMz, lNx, flush=True)
    comm.Barrier()

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = matvec_dist.to_sdfg(strict=False)
            mpi_sdfg.apply_strict_transformations()
            mpi_sdfg = mpi_sdfg.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, x=x, y=y, z=z, M=M, N=N, lMA=lMA, lNA=lNA, lMz=lMz, lNx=lNx, Px=Px, Py=Py)

    # for r in range(0, size):
    #     if r == rank:
    #         print("I am %d" % rank)
    #         print(A)
    #     comm.Barrier()

    if rank == 0:
        refA = np.arange(0, M*N).reshape(M, N).astype(np.float64)
        refx = np.arange(0, N).astype(np.float64)
        refy = np.zeros((M, ), dtype=np.float64)
        refz = np.zeros((N, ), dtype=np.float64)
        shared_sdfg = matvec_shared.compile()
        shared_sdfg(A=refA, x=refx, y=refy, z=refz, M=M, N=N, lMA=lMA, lNA=lNA, lMz=lMz, lNx=lNx, Px=Px, Py=Py)

        print(relerr(refy, y))
        print(relerr(refz, z))
