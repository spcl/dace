# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implicitly distributed (transformed shared-memory SDFG) Gemm sample."""
import numpy as np
import dace as dc
from mpi4py import MPI

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
from dace.transformation.dataflow import ElementWiseArrayOperation2D, RedundantComm2D


NI, NJ, NK = (
    dc.symbol(s, dtype=dc.int64, integer=True, positive=True)
    for s in ('NI', 'NJ', 'NK'))


def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)


@dc.program
def gemm(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ],
         A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):

    C[:] = alpha * A @ B + beta * C 


def init_data(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI,
                        shape=(NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK,
                        shape=(NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ,
                        shape=(NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


if __name__ == "__main__":

    # Initialization
    NI, NJ, NK = 2000, 2300, 2600
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        mpi_sdfg = gemm.to_sdfg(strict=False)
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_transformations_repeated(ElementWiseArrayOperation2D)
        mpi_sdfg.expand_library_nodes()
        mpi_sdfg.apply_strict_transformations()
        mpi_sdfg.apply_transformations_repeated(RedundantComm2D)
        mpi_sdfg.apply_strict_transformations()
        mpi_func = mpi_sdfg.compile()

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
        refC[:] = alpha * refA @ refB + beta * refC
        assert(np.allclose(refC, C))
