#!/usr/bin/env python

import argparse
import dace
import math
import numpy as np
import scipy as sp
from mpi4py import MPI

from dace.sdfg import SDFG
from dace.codegen.compiler import CompiledSDFG, ReloadableDLL

from functools import reduce
from itertools import product
from typing import List, Tuple

N = dace.symbol('N')
PX = dace.symbol('PX')
PY = dace.symbol('PY')
PM = dace.symbol('PM')
BX = dace.symbol('BX')
BY = dace.symbol('BY')
BM = dace.symbol('BM')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    Y[:] += A * X[:]


if __name__ == "__main__":

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Arguments/symbols
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1024)
    parser.add_argument("PX", type=int, nargs="?", default=size)
    parser.add_argument("PY", type=int, nargs="?", default=size)
    parser.add_argument("PM", type=int, nargs="?", default=size)
    parser.add_argument("BX", type=int, nargs="?", default=64)
    parser.add_argument("BY", type=int, nargs="?", default=64)
    parser.add_argument("BM", type=int, nargs="?", default=64)
    args = vars(parser.parse_args())

    # Set symbols
    N.set(args["N"])
    PX.set(args["PX"])
    PY.set(args["PY"])
    PM.set(args["PM"])
    BX.set(args["BX"])
    BY.set(args["BY"])
    BM.set(args["BM"]) 

    # Initialize arrays: Randomize A, X, and Y
    A_arr = np.random.rand(1).astype(np.float64)
    A = A_arr[0]
    X = np.random.rand(N.get()).astype(np.float64)
    Y = np.random.rand(N.get()).astype(np.float64)

    from distr_helper import distr_exec
    from dace.transformation.dataflow import MapFusion
    args = {'A': A, 'X': X, 'Y': Y, 'N': N}
    data_distr = {'X': ([PX], [BX]), 'Y': ([PY], [BY])}
    itsp_distr = ([PM], [BM])
    distr_exec(axpy, args,
               output=['Y'], ref_func=lambda A, X, Y: [A * X + Y],
               data_distr=data_distr, itsp_distr=itsp_distr,
               other_trans={MapFusion: dict()})
