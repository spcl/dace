# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
import dace
import numpy as np
import scipy as sp
from mpi4py import MPI
from dace.transformation.dataflow import MPITransformMap

N = dace.symbol('N')


@dace.program
def axpy(A: dace.float64, X: dace.float64[N], Y: dace.float64[N]):

    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    N = args["N"]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()

    if rank == 0:
        print('Scalar-vector multiplication %d (MPI, ranks = %d)' % (N, ranks))
    else:
        dace.Config.set('debugprint', value=False)

    # Initialize arrays: Randomize A and X, zero Y
    a = dace.float64(np.random.rand())
    x = np.random.rand(N).astype(np.float64)
    y = np.random.rand(N).astype(np.float64)
    regression = (a * x + y)

    sdfg = axpy.to_sdfg()

    # Transform program to run with MPI
    sdfg.apply_transformations(MPITransformMap)

    # Compile MPI program once
    if ranks == 1:
        csdfg = sdfg.compile()
        print('Compiled, exiting')
        exit(0)
    else:
        # Use cached compiled file
        dace.Config.set('compiler', 'use_cache', value=True)
        csdfg = sdfg.compile()

    csdfg(A=a, X=x, Y=y, N=N)

    # Get range handled by this rank
    partition = N // ranks
    reg = regression[partition * rank:partition * (rank + 1)]
    res = y[partition * rank:partition * (rank + 1)]

    diff = np.linalg.norm(reg - res)
    print("== Rank %d == Difference:" % rank, diff)
    if rank == 0:
        print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
