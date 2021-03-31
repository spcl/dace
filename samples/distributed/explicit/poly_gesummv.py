import numpy as np
import dace as dc
import timeit
from mpi4py import MPI

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL


# NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))

lM = dc.symbol('lM', dtype=dc.int64, integer=True, positive=True)
lN = dc.symbol('lN', dtype=dc.int64, integer=True, positive=True)
lNx = dc.symbol('lNx', dtype=dc.int64, integer=True, positive=True)
Px = dc.symbol('Px', dtype=dc.int32, integer=True, positive=True)
Py = dc.symbol('Py', dtype=dc.int32, integer=True, positive=True)
M = lM * Px
N = lN * Py  # == lNx * Px


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def gesummv_shared(alpha: dc.float64, beta: dc.float64, A: dc.float64[M, N],
                   B: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M]):

    y[:] = alpha * A @ x + beta * B @ x


@dc.program
def gesummv_distr(alpha: dc.float64, beta: dc.float64, A: dc.float64[M, N],
                  B: dc.float64[M, N], x: dc.float64[N], y: dc.float64[M]):

    lA = np.empty((lM, lN), dtype=A.dtype)
    lB = np.empty((lM, lN), dtype=B.dtype)
    lx = np.empty((lNx,), dtype=x.dtype)

    # bsizes = np.empty((2,), dtype=np.int32)
    # bsizes[0] = lNx
    # bsizes[1] = N // Py

    dc.comm.BCScatter(A, lA, (lM, lN))
    dc.comm.BCScatter(B, lB, (lM, lN))
    dc.comm.BCScatter(x, lx, (lNx, 1))

    tmp1 = distr.MatMult(A, x, lA, lx, (lM, lN), (lNx, 1))
    tmp2 = distr.MatMult(B, x, lB, lx, (lM, lN), (lNx, 1))

    tmp1[:] = alpha * tmp1 + beta * tmp2

    dc.comm.BCGather(tmp1, y, (lM, 1))


def init_data(M, N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    # A = np.empty((N, N), dtype=datatype)
    # B = np.empty((N, N), dtype=datatype)
    # tmp = np.empty((N, ), dtype=datatype)
    # x = np.empty((N, ), dtype=datatype)
    # y = np.empty((N, ), dtype=datatype)
    # for i in range(N):
    #     x[i] = (i % N) % N
    #     for j in range(N):
    #         A[i, j] = ((i * j + 1) % N) / N
    #         B[i, j] = ((i * j + 2) % N) / N
    rng = np.random.default_rng(42)
    A = rng.random((M, N), dtype=datatype)
    B = rng.random((M, N), dtype=datatype)
    x = rng.random((N,), dtype=datatype)
    y = rng.random((M,), dtype=datatype)

    return alpha, beta, A, B, x, y


def time_to_ms(raw):
    return int(round(raw * 1000))


grid = {
    1: (1, 1),
    2: (2, 1),
    4: (2, 2),
    8: (4, 2),
    16: (4, 4)
}

if __name__ == "__main__":

    # Initialization
    M, N = 3200, 2800

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    lM = M // Px
    lN = N // Py
    lNx = N // Px

    def setup_func(rank):
        if rank == 0:
            return init_data(M, N, np.float64)
        else:
            return (
                1.5, 1.2,
                np.empty((M, N), dtype=np.float64),
                np.empty((M, N), dtype=np.float64),
                np.empty((N,), dtype=np.float64),
                np.empty((M,), dtype=np.float64))
    
    alpha, beta, A, B, x, y = setup_func(rank)

    mpi_sdfg = None
    # if size < 2:
    #     raise ValueError("This test is supposed to be run with at least two processes!")
    if rank == 0:
        mpi_sdfg = gesummv_distr.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_func= mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
            n=gesummv_distr.name))
        mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
            ".dacecache/{n}/build/lib{n}.so".format(n=gesummv_distr.name),
            gesummv_distr.name))

    ldict = locals()

    comm.Barrier()
  
    mpi_func(A=A, B=B, x=x, alpha=alpha, beta=beta, y=y, 
             lM=lM, lN=lN, lNx=lNx, Px=Px, Py=Py)

    comm.Barrier()

    stmt = ("mpi_func(A=A, B=B, x=x, alpha=alpha, beta=beta, y=y, "
            "lM=lM, lN=lN, lNx=lNx, Px=Px, Py=Py)")
    setup = "alpha, beta, A, B, x, y = setup_func(rank); comm.Barrier()"
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
        refout = shared_sdfg(A=refA, B=refB, x=refx, alpha=alpha, beta=beta,
                             y=refy, lM=lM, lN=lN, lNx=lNx, Px=Px, Py=Py)

        print("=======Validation=======")
        print(relerr(refy, y))
