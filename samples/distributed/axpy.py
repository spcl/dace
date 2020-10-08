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

    # Validate decomposition parameters
    def validate_param(rank: int, size: int,
                       param: dace.symbol, label: str) -> int:
        if param.get() > size:
            if rank == 0:
                print("Not enough ranks for the requested {l} decomposition "
                      "(size = {s}, PX = {px}). Setting {n} equal to comm "
                      "size".format(l=label, s=size, px=param.get(),
                                    n=param.name))
            return size
        return param.get()
    
    PX.set(validate_param(rank, size, PX, "X vector"))
    PY.set(validate_param(rank, size, PY, "Y vector"))
    PM.set(validate_param(rank, size, PM, "iteration space"))

    if rank == 0:
        print("==== Program start ====")
        print("BLAS axpy kernel (Y += A * X) with N = {}".format(N.get()))
        print("PX = {}, PY = {}, PM = {}".format(PX.get(), PY.get(), PM.get()))
        print("BX = {}, BY = {}, BM = {}".format(BX.get(), BY.get(), BM.get()))

    # Create SDFG
    if rank == 0:
        from dace.transformation.dataflow import (BlockCyclicData,
                                                  BlockCyclicMap, MapFusion)
        sdfg = axpy.to_sdfg()
        sdfg.apply_strict_transformations()
        sdfg.add_process_grid("X", (PX,))
        sdfg.add_process_grid("Y", (PY,))
        sdfg.add_process_grid("M", (PM,))
        sdfg.apply_transformations([MapFusion, BlockCyclicData,
                                    BlockCyclicData, BlockCyclicMap],
                                    options=[
                                        {},
                                        {'dataname': 'X',
                                        'gridname': 'X',
                                        'block': (BX,)},
                                        {'dataname': 'Y',
                                        'gridname': 'Y',
                                        'block': (BY,)},
                                        {'gridname': 'M',
                                        'block': (BM,)}])
        sdfg.save("rma_axpy.sdfg")
        func = sdfg.compile()
    
    comm.Barrier()

    if rank > 0:
        sdfg = SDFG.from_file("rma_axpy.sdfg")
        func = CompiledSDFG(sdfg, ReloadableDLL(
            '.dacecache/axpy/build/libaxpy.so', sdfg.name))
    

    # Initialize arrays: Randomize A, X, and Y
    A_arr = np.random.rand(1).astype(np.float64)
    X = np.random.rand(N.get()).astype(np.float64)
    Y = np.random.rand(N.get()).astype(np.float64)
    # A = np.float64(2.0)
    # X = np.arange(N.get(), dtype=np.float64)
    # Y = np.ones(N.get(), dtype=np.float64)
    
    # Use rank 0's arrays
    comm.Bcast(A_arr, root=0)
    A = np.float64(A_arr[0])
    comm.Bcast(X, root=0)
    comm.Bcast(Y, root=0)

    # Initialize regression arrays
    A_regression = np.float64(0.0)
    X_regression = np.ndarray([N.get()], dtype=np.float64)
    Y_regression = np.ndarray([N.get()], dtype=np.float64)
    A_regression = A
    X_regression[:] = X[:]
    Y_regression[:] = Y[:]

    # Extract local data
    def get_coords(rank: int,
                   process_grid: List[dace.symbol]) -> Tuple[bool, List[int]]:
        # Check if rank belongs to comm
        pg_int = [p.get() for p in process_grid]
        cart_size = reduce(lambda a, b: a * b, pg_int, 1)
        if rank >= cart_size:
            return False, []
        # Compute strides
        n = len(process_grid)
        size = 1
        strides = [None] * n
        for i in range(n - 1, -1, -1):
            strides[i] = size
            size *= process_grid[i].get()
        # Compute coords
        rem = rank
        coords = [None] * n
        for i in range(n):
            coords[i] = int(rem / strides[i])
            rem %= strides[i]
        return True, coords

    def extract_local(global_data, rank: int,
                      process_grid: List[dace.symbol],
                      block_sizes: List[dace.symbol]):
        
        fits, coords = get_coords(rank, process_grid)
        if not fits:
            return np.empty([0], dtype=global_data.dtype)

        local_shape = [math.ceil(n / (b.get() * p.get()))
                       for n, p, b in zip(global_data.shape, process_grid,
                                          block_sizes)]
        local_shape.extend([b.get() for b in block_sizes])
        local_data = np.zeros(local_shape, dtype=global_data.dtype)

        n = len(global_data.shape)
        for l in product(*[range(ls) for ls in local_shape[:n]]):
            gstart = [(li * p.get() + c) * b.get()
                      for li, p, c, b in zip(l, process_grid, coords,
                                             block_sizes)]
            gfinish = [min(n, s + b.get())
                       for n, s, b in zip(global_data.shape, gstart,
                                          block_sizes)]
            gindex = tuple(slice(s, f, 1) for s, f in zip(gstart, gfinish))
            # Validate range
            rng = [f - s for s, f in zip(gstart, gfinish)]
            if np.any(np.less(rng, 0)):
                continue
            block_slice = tuple(slice(0, r, 1) for r in rng)
            lindex = l + block_slice
            try:
                local_data[lindex] = global_data[gindex]
            except Exception as e:
                print(rank, coords, process_grid, block_sizes)
                print(l, lindex, gstart, gfinish, gindex)
                raise e
        
        return local_data
    
    local_X = extract_local(X, rank, [PX], [BX])
    local_Y = extract_local(Y, rank, [PY], [BY])


    func(A=A, X=local_X, Y=local_Y, N=N,
         PX=PX, BX=BX, PY=PY, BY=BY, PM=PM, BM=BM)

    c_axpy = sp.linalg.blas.get_blas_funcs('axpy',
                                           arrays=(X_regression, Y_regression))
    c_axpy(X_regression, Y_regression, N.get(), A_regression)

    local_Y_ref = extract_local(Y_regression, rank, [PY], [BY])
    if local_Y.size > 0:
        norm_diff = np.linalg.norm(local_Y - local_Y_ref)
        norm_ref = np.linalg.norm(local_Y_ref)
        relerror = norm_diff / norm_ref
        print("Rank {r} relative_error: {d}".format(r=rank, d=relerror))

