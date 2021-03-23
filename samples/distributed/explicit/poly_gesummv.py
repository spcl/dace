import numpy as np
import dace as dc
from mpi4py import MPI


# NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))

lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
lNy = dc.symbol('lNy', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
N = lNx * Px  # == lNy * Py


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def gesummv_shared(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
                   B: dc.float64[N, N], x: dc.float64[N], y: dc.float64[N]):

    y[:] = alpha * A @ x + beta * B @ x


@dc.program
def gesummv_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
                  B: dc.float64[N, N], x: dc.float64[N], y: dc.float64[N]):

    lA = np.zeros((lNx, lNy), dtype=A.dtype)
    lB = np.zeros((lNx, lNy), dtype=B.dtype)
    lx = np.zeros((lNx,), dtype=x.dtype)

    bsizes = np.empty((2,), dtype=np.int32)
    bsizes[0] = lNx
    bsizes[1] = N // Py

    gdescA, ldescA = dc.comm.BCScatter(A, lA, bsizes)
    gdescB, ldescB = dc.comm.BCScatter(B, lB, bsizes)
    gdescx, ldescx = dc.comm.BCScatter(x, lx, bsizes[0])

    tmp1, gdesctmp1, ldesctmp1 = distr.MatMult(lA, ldescA, lx, ldescx)
    tmp2, gdesctmp2, ldesctmp2 = distr.MatMult(lB, ldescB, lx, ldescx)

    tmp1[:] = alpha * tmp1 + beta * tmp2

    dc.comm.BCGather(tmp1, y, ldesctmp1, gdesctmp1)


def init_data(N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.empty((N, N), dtype=datatype)
    B = np.empty((N, N), dtype=datatype)
    tmp = np.empty((N, ), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    for i in range(N):
        x[i] = (i % N) % N
        for j in range(N):
            A[i, j] = ((i * j + 1) % N) / N
            B[i, j] = ((i * j + 2) % N) / N

    return alpha, beta, A, B, tmp, x, y


if __name__ == "__main__":

    # Initialization
    N = 2800
    alpha, beta, A, B, tmp, x, y = init_data(N, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px = Py = int(np.sqrt(size))
    lNx = N // Px
    lNy = N // Py

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, size):
        if r == rank:
            mpi_sdfg = gesummv_distr.to_sdfg(strict=False)
            mpi_sdfg.apply_strict_transformations()
            mpi_sdfg = mpi_sdfg.compile()
        comm.Barrier()
  
    mpi_sdfg(A=A, B=B, x=x, alpha=alpha, beta=beta, y=y, 
             N=N, lNx=lNx, lNy=lNy, Px=Px, Py=Py)

    if rank == 0:
        alpha, beta, refA, refB, tmp, refx, refy = init_data(N, np.float64)
        shared_sdfg = gesummv_shared.compile()
        refout = shared_sdfg(A=refA, B=refB, x=refx, alpha=alpha, beta=beta,
                             y=refy, N=N, lNx=lNx, lNy=lNy, Px=Px, Py=Py)

        print(relerr(refy, y))
