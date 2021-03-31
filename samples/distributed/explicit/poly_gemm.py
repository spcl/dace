import numpy as np
import dace as dc
import timeit
from mpi4py import MPI

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL

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

    dc.comm.BCScatter(A, lA, (lNI, lNKa))
    dc.comm.BCScatter(B, lB, (lNKb, lNJ))
    dc.comm.BCScatter(C, lC, (lNI, lNJ))

    tmp  = distr.MatMult(A, B, lA, lB, (lNI, lNKa), (lNKb, lNJ))

    lC[:] = alpha * tmp + beta * lC

    dc.comm.BCGather(lC, C, (lNI, lNJ))


def init_data(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    rng = np.random.default_rng(42)
    C = rng.random((NI, NJ), dtype=datatype)
    A = rng.random((NI, NK), dtype=datatype)
    B = rng.random((NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


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
    # NI, NJ, NK = 2000, 2300, 2600  # 4000, 4600, 5200
    NI, NJ, NK = 4000, 4600, 5200

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    Px, Py = grid[size]
    lNI = NI // Px
    lNJ = NJ // Py
    lNKa = NK // Py
    lNKb = NK // Px

    def setup_func(rank):
        if rank == 0:
            return init_data(NI, NJ, NK, np.float64)
        else:
            return (
                1.5, 1.2,
                np.empty((NI, NJ), dtype=np.float64),
                np.empty((NI, NK), dtype=np.float64),
                np.empty((NK, NJ), dtype=np.float64))
    
    alpha, beta, C, A, B = setup_func(rank)

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

    ldict = locals()
    
    comm.Barrier()

    mpi_func(A=A, B=B, C=C, alpha=alpha, beta=beta,
             NI=NI, NJ=NJ, NK=NK,
             lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb,
             Px=Px, Py=Py)

    comm.Barrier()

    stmt = ("mpi_func(A=A, B=B, C=C, alpha=alpha, beta=beta, "
            "NI=NI, NJ=NJ, NK=NK, lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb, "
            "Px=Px, Py=Py)")
    setup = "alpha, beta, C, A, B = setup_func(rank); comm.Barrier()"
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
        shared_sdfg(A=refA, B=refB, C=refC, alpha=alpha, beta=beta,
                    NI=NI, NJ=NJ, NK=NK,
                    lNI=lNI, lNJ=lNJ, lNKa=lNKa, lNKb=lNKb,
                    Px=Px, Py=Py)

        print("=======Validation=======")
        print(relerr(refC, C))