# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the PBLAS GEMV library node. """

import dace
import numpy as np

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils
from mpi4py import MPI


# Symbols

# Process grid
P, Px, Py = (dace.symbol(s, positive=True) for s in ('P', 'Px', 'Py'))
# Global sizes
GM, GN = (dace.symbol(s, positive=True) for s in ('GM', 'GN'))
# Local sizes
LMx, LMy, LNx, LNy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy'))

# # Stencils
# noff = dc.symbol('noff', dtype=dc.int32, integer=True, nonnegative=True)
# soff = dc.symbol('soff', dtype=dc.int32, integer=True, nonnegative=True)
# woff = dc.symbol('woff', dtype=dc.int32, integer=True, nonnegative=True)
# eoff = dc.symbol('eoff', dtype=dc.int32, integer=True, nonnegative=True)
# nn = dc.symbol('nn', dtype=dc.int32, integer=True)
# ns = dc.symbol('ns', dtype=dc.int32, integer=True)
# nw = dc.symbol('nw', dtype=dc.int32, integer=True)
# ne = dc.symbol('ne', dtype=dc.int32, integer=True)
# MPI_Request = dc.opaque("MPI_Request")


# Helper methods

def relerr(ref, val):
    return np.linalg.norm(ref-val) / np.linalg.norm(ref)

def time_to_ms(raw):
    return int(round(raw * 1000))

def l2g(idx, pidx, bsize):
    return idx + pidx * bsize

grid = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    8: (2, 4),
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16)
}


rng = np.random.default_rng(42)


def test_pgemv():

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    @dace.program
    def pdgemv(A: dace.float64[LMx, LNy], x: dace.float64[LNy]):
        return dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(LMx, 1))
    
    sdfg = None
    if rank == 0:
        sdfg = pdgemv.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    func = utils.distributed_compile(sdfg, commworld)
    
    M, N = size * 4, size * 2
    # A = rng.random((M, N), dtype=np.float64)
    A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    # A = np.ones((M, N), dtype=np.float64)
    # for k in range(M):
        # A[k] = k
    # x = rng.random((N,), dtype=np.float64)
    x = np.ones((N,), dtype=np.float64)
    y = A @ x

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    # print(lA)
    lx = x[j*tj:(j+1)*tj].copy()
    # print(lx)

    ref = y[i*ti:(i+1)*ti]
    val = func(A=lA, x=lx, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    print(rank, val, ref, flush=True)
    # print(x.reshape(1, N) @ A.reshape(N, M))
    # assert(np.allclose(val, ref))

    commworld.Barrier()

    @dace.program
    def pdgemv_T(A: dace.float64[LMx, LNy], x: dace.float64[LMx]):
        return dace.distr.MatMult(x, A, (Px*LMx, Py*LNy), c_block_sizes=(LNy, 1))
    
    sdfg1 = None
    if rank == 0:
        sdfg1 = pdgemv_T.to_sdfg(simplify=True)
        sdfg1 = auto_optimize(sdfg1, dace.DeviceType.CPU)
    func1 = utils.distributed_compile(sdfg1, commworld)
    
    M, N = size * 4, size * 2
    # A = rng.random((M, N), dtype=np.float64)
    A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    # A = np.ones((M, N), dtype=np.float64)
    # for k in range(M):
        # A[k] = k
    # x = rng.random((N,), dtype=np.float64)
    x = np.ones((M,), dtype=np.float64)
    y = A.T @ x

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lx = x[i*ti:(i+1)*ti].copy()

    ref = y[j*tj:(j+1)*tj]
    val = func1(A=lA, x=lx, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    print(rank, val, ref, flush=True)
    # assert(np.allclose(val, ref))

    commworld.Barrier()

    @dace.program
    def atax(A: dace.float64[LMx, LNy], x: dace.float64[LNy], y: dace.float64[LNy]):
        tmp = dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(LMx, 1))
        y[:] = dace.distr.MatMult(tmp, A, (M, N), c_block_sizes=(LNy, 1))
    
    sdfg2 = None
    if rank == 0:
        sdfg2 = atax.to_sdfg(simplify=True)
        sdfg2 = auto_optimize(sdfg2, dace.DeviceType.CPU)
    func2 = utils.distributed_compile(sdfg2, commworld)
    
    M, N = size * 4, size * 2
    # A = rng.random((M, N), dtype=np.float64)
    A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    # A = np.ones((M, N), dtype=np.float64)
    # for k in range(M):
        # A[k] = k
    # x = rng.random((N,), dtype=np.float64)
    x = np.ones((N,), dtype=np.float64)
    y = A.T @ (A @ x)

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    # print(lA)
    lx = x[j*tj:(j+1)*tj].copy()
    # print(lx)

    ref = y[j*tj:(j+1)*tj]
    val = np.ndarray((tj,), dtype=np.float64)
    func2(A=lA, x=lx, y=val, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    print(rank, val, ref, flush=True)
    # print(x.reshape(1, N) @ A.reshape(N, M))
    # assert(np.allclose(val, ref))


if __name__ == '__main__':

    test_pgemv()



def adjust_size(size, scal_func, comm_size, divisor):
    candidate = size * scal_func(comm_size)
    if candidate // divisor != candidate:
        candidate = np.ceil(candidate / divisor) * divisor
    return int(candidate)


# # ===== Programs ==============================================================

# # ===== atax =====

# atax_sizes = [[20000, 25000]]  #[[1800, 2200], [3600, 4400], [7200, 8800], [14400, 17600]]

# @dc.program
# def atax_shmem(A: dc.float64[M, N], x: dc.float64[N], y:dc.float64[N]):
#     y[:] = (A @ x) @ A

# @dc.program
# def atax_distr(A: dc.float64[lM, lN], x: dc.float64[lN], y:dc.float64[lN]):
#     tmp = distr.MatMult(A, x, (Px*lM, Py*lN), c_block_sizes=(lMy, 1))
#     y[:] = distr.MatMult(tmp, A, (M, N))

# def atax_shmem_init(M, N, datatype):
#     fn = datatype(N)
#     A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M),
#                         shape=(M, N), dtype=datatype)
#     x = np.fromfunction(lambda i: 1 + (i / fn), shape=(N,), dtype=datatype)
#     y = np.empty((N,), dtype=datatype)
#     return A, x, y

# def atax_distr_init(M, N, lM, lN, datatype, pi, pj):
#     fn = datatype(N)
#     A = np.fromfunction(lambda i, j: ((l2g(i, pi, lM) + l2g(j, pj, lN)) % N) / (5 * M),
#                         shape=(lM, lN), dtype=datatype)
#     x = np.fromfunction(lambda i: 1 + (l2g(i, pj, lN) / fn),
#                         shape=(lN,), dtype=datatype)
#     y = np.empty((lN,), dtype=datatype)
#     return A, x, y

# def atax(sizes, validate=True):

#     # MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     Px, Py = grid[size]
#     # Fix for grid issue with gemv
#     if Px != Py:
#         Px, Py = 1, size
#     # Px, Py = 1, size
#     pi = rank // Py
#     pj = rank % Py

#     if rank == 0:
#         print("===== atax =====")
#         print("sizes: {}".format(sizes), flush=True)

#     M, N = sizes
#     M = adjust_size(M, lambda x: np.sqrt(x), size, max(Px, Py))
#     N = adjust_size(N, lambda x: np.sqrt(x), size, max(Px, Py))
#     if rank == 0:
#         print("adjusted sizes: {}".format((M, N)), flush=True)

#     # Symbolic sizes
#     lM = M // Px
#     lN = N // Py
#     lNx = N // Py
#     lMy = M // Py

#     lA, lx, ly = atax_distr_init(M, N, lM, lN, np.float64, pi, pj)
#     if rank == 0:
#         print("data initialized", flush=True)

#     mpi_sdfg = None
#     if rank == 0:
#         mpi_sdfg = atax_distr.to_sdfg(strict=False)
#         print(mpi_sdfg.free_symbols)
#         mpi_sdfg.apply_strict_transformations()
#         mpi_sdfg.apply_transformations_repeated([MapFusion])
#         mpi_sdfg.apply_strict_transformations()
#         mpi_func= mpi_sdfg.compile()
#     comm.Barrier()
#     if rank > 0:
#         mpi_sdfg = dc.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
#             n=atax_distr.name))
#         mpi_func = CompiledSDFG(mpi_sdfg, ReloadableDLL(
#             ".dacecache/{n}/build/lib{n}.so".format(n=atax_distr.name),
#             atax_distr.name))

#     ldict = locals()

#     comm.Barrier()

#     mpi_func(A=lA, x=lx, y=ly,
#              lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
#     # print(rank, 'Hello World!', flush=True)
    
#     comm.Barrier()

#     if validate:

#         if rank == 0:
#             y = np.empty((N,), dtype=np.float64)
#             y[0:lN] = ly
#             for i in range(Py):
#                 if i == pj:
#                     continue
#                 else:
#                     comm.Recv(ly, source=i, tag=i)
#                     y[i*lN:(i+1)*lN] = ly
#         elif pi == 0:
#             comm.Send(ly, dest=0, tag=pj)
        
#         comm.Barrier()

#     stmt = ("mpi_func(A=lA, x=lx, y=ly, "
#             "lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)")
#     setup = "comm.Barrier()"
#     repeat = 10

#     comm.Barrier()

#     raw_time_list = timeit.repeat(stmt,
#                                   setup=setup,
#                                   repeat=repeat,
#                                   number=1,
#                                   globals=ldict)
#     raw_time = np.median(raw_time_list)

#     if rank == 0:
#         ms_time = time_to_ms(raw_time)
#         print("Median is {}ms".format(ms_time), flush=True)
#         write_time("atax", (M, N), raw_time_list, append=False)

#     if validate:

#         if rank == 0:
#             refA, refx, refy = atax_shmem_init(M, N, np.float64)
#             shared_sdfg = atax_shmem.compile()
#             shared_sdfg(A=refA, x=refx, y=refy,
#                         lM=lM, lN=lN, lNx=lNx, lMy=lMy, Px=Px, Py=Py)
#             error = relerr(refy, y)
#             print("validation: {} ({})".format(error < 1e-12, error), flush=True)