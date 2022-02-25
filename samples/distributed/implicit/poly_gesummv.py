# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implicitly distributed (transformed shared-memory SDFG) Gesummv sample."""
import dace as dc
import numpy as np
import os
from dace.sdfg.utils import load_precompiled_sdfg
from mpi4py import MPI

from dace.transformation.dataflow import (ElementWiseArrayOperation, ElementWiseArrayOperation2D, RedundantComm2D)

N = dc.symbol('N', dtype=dc.int64, integer=True, positive=True)


def relerr(ref, val):
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


@dc.program
def gesummv(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], B: dc.float64[N, N], x: dc.float64[N],
            y: dc.float64[N]):

    y[:] = alpha * A @ x + beta * B @ x


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

    if rank == 0:
        with dc.config.set_temporary('library', 'blas', 'default_implementation', value='PBLAS'):
            mpi_sdfg = gesummv.to_sdfg(simplify=False)
            mpi_sdfg.simplify()
            mpi_sdfg.apply_transformations_repeated(ElementWiseArrayOperation2D)
            mpi_sdfg.apply_transformations_repeated(ElementWiseArrayOperation)
            mpi_sdfg.expand_library_nodes()
            mpi_sdfg.simplify()
            mpi_sdfg.apply_transformations_repeated(RedundantComm2D)
            mpi_sdfg.simplify()
            mpi_func = mpi_sdfg.compile()
    comm.Barrier()
    if rank > 0:
        build_folder = dc.Config.get('default_build_folder')
        mpi_func = load_precompiled_sdfg(os.path.join(build_folder, gesummv.name))
    comm.Barrier()

    Px, Py = 1, size
    mpi_func(A=A, B=B, x=x, alpha=alpha, beta=beta, y=y, N=N, commsize=size, Px=Px, Py=Py)

    comm.Barrier()

    if rank == 0:
        alpha, beta, refA, refB, tmp, refx, refy = init_data(N, np.float64)
        refy[:] = alpha * refA @ refx + beta * refB @ refx
        assert (np.allclose(refy, y))
