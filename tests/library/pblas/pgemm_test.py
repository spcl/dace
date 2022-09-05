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
GM, GN, GK, GR, GS, GT = (dace.symbol(s, positive=True) for s in ('GM', 'GN', 'GK', 'GR', 'GS', 'GT'))
# Local sizes
LMx, LMy, LNx, LNy, LKx, LKy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy', 'LKx', 'LKy'))
LRx, LRy, LSx, LSy, LTx, LTy = (dace.symbol(s, positive=True) for s in ('LRx', 'LRy', 'LSx', 'LSy', 'LTx', 'LTy'))


grid = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    8: (4, 2),
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

    # NPx, NPy = min(grid[size]), max(grid[size])
    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    Mmult = 512
    Nmult = 1024
    Kmult = 768
    Rmult = 128
    Smult = 2048
    M, N, K, R, S = size * Mmult, size * Nmult, size * Kmult, size * Rmult, size * Smult

    @dace.program
    def pdgemm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy]):
        return dace.distr.MatMult(A, B, (LMx*Px, LNy*Py, GK))
    
    sdfg = None
    if rank == 0:
        sdfg = pdgemm.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    func = utils.distributed_compile(sdfg, commworld)
    
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
    C = A @ B

    ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
    lA = A[i*ti:(i+1)*ti, j*tkj:(j+1)*tkj].copy()
    lB = B[i*tki:(i+1)*tki, j*tj:(j+1)*tj].copy()

    val = func(A=lA, B=lB, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, GK=K, Px=NPx, Py=NPy)
    ref = C[i*ti:(i+1)*ti, j*tj:(j+1)*tj]
    assert(np.allclose(val, ref))

    commworld.Barrier()

    @dace.program
    def k2mm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy], C: dace.float64[LNx, LRy]):
        tmp = dace.distr.MatMult(A, B, (LMx*Px, LNy*Py, GK))
        return dace.distr.MatMult(tmp, C, (GM, GR, GN))
    
    sdfg2 = None
    if rank == 0:
        sdfg2 = k2mm.to_sdfg(simplify=True)
        sdfg2 = auto_optimize(sdfg2, dace.DeviceType.CPU)
    func2 = utils.distributed_compile(sdfg2, commworld)
    
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
    C = rng.random((N, R), dtype=np.float64)
    D = A @ B @ C

    ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
    tji, tr = N // NPx, R // NPy
    lA = A[i*ti:(i+1)*ti, j*tkj:(j+1)*tkj].copy()
    lB = B[i*tki:(i+1)*tki, j*tj:(j+1)*tj].copy()
    lC = C[i*tji:(i+1)*tji, j*tr:(j+1)*tr].copy()

    val = func2(A=lA, B=lB, C=lC, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, LNx=tji, LRy=tr,
                GM=M, GN=N, GK=K, GR=R, Px=NPx, Py=NPy)
    ref = D[i*ti:(i+1)*ti, j*tr:(j+1)*tr]
    assert(np.allclose(val, ref))
    exit(1)

    commworld.Barrier()

    # @dace.program
    # def atax(A: dace.float64[LMx, LNy], x: dace.float64[LNy], y: dace.float64[LNy]):
    #     tmp = dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(LMx, 1))
    #     y[:] = dace.distr.MatMult(tmp, A, (M, N), c_block_sizes=(LNy, 1))
    @dace.program
    def atax(A: dace.float64[LMx, LNy], x: dace.float64[GN], y: dace.float64[GN]):
        tmp = dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(GM, 1))
        y[:] = dace.distr.MatMult(tmp, A, (GM, GN), c_block_sizes=(GN, 1))
    
    sdfg2 = None
    if rank == 0:
        sdfg2 = atax.to_sdfg(simplify=True)
        sdfg2 = auto_optimize(sdfg2, dace.DeviceType.CPU)
    func2 = utils.distributed_compile(sdfg2, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    # A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    x = rng.random((N,), dtype=np.float64)
    # x = np.ones((N,), dtype=np.float64)
    y = A.T @ (A @ x)

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lx = x[j*tj:(j+1)*tj].copy()

    # val = np.ndarray((tj,), dtype=np.float64)
    # func2(A=lA, x=lx, y=val, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    # tmp = np.ndarray((M,), dtype=np.float64)
    val = np.ndarray((N,), dtype=np.float64)
    func2(A=lA, x=x, y=val, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
    # print(rank, val, flush=True)
    # if rank < NPx:
    #     ref = y[rank*tj:(rank+1)*tj]
    #     # print(rank, val, ref, flush=True)
    #     assert(np.allclose(val, ref))
    if rank == 0:
        assert(np.allclose(val, y))
    

    @dace.program
    def bicg(A: dace.float64[LMx, LNy], p: dace.float64[GN], r: dace.float64[GM],
             o1: dace.float64[GN], o2: dace.float64[GM]):
        o1[:] = dace.distr.MatMult(r, A, (Px*LMx, Py*LNy), c_block_sizes=(GN, 1))
        o2[:] = dace.distr.MatMult(A, p, (GM, GN), c_block_sizes=(GM, 1))
    
    sdfg3 = None
    if rank == 0:
        sdfg3 = bicg.to_sdfg(simplify=True)
        sdfg3 = auto_optimize(sdfg3, dace.DeviceType.CPU)
    func3 = utils.distributed_compile(sdfg3, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    p = rng.random((N,), dtype=np.float64)
    r = rng.random((M,), dtype=np.float64)
    o1 = A.T @ r
    o2 = A @ p

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()

    val1 = np.ndarray((N,), dtype=np.float64)
    val2 = np.ndarray((M,), dtype=np.float64)
    func3(A=lA, p=p, r=r, o1=val1, o2=val2, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
    if rank == 0:
        assert(np.allclose(val1, o1))
        assert(np.allclose(val2, o2))
    
    @dace.program
    def gemver(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], u1: dace.float64[LMx],
               v1: dace.float64[LNy], u2: dace.float64[LMx], v2: dace.float64[LNy], w: dace.float64[GM],
               x: dace.float64[GN], y: dace.float64[GM], z: dace.float64[GN]):
        A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
        tmp1 = dace.distr.MatMult(y, A, (Px*LMx, Py*LNy), c_block_sizes=(GN, 1))
        x += beta * tmp1 + z
        xf = dace.float32(x)
        tmp2 = dace.distr.MatMult(A, x, (M, N), c_block_sizes=(GM, 1))
        w += alpha * tmp2
    
    sdfg4 = None
    if rank == 0:
        sdfg4 = gemver.to_sdfg(simplify=True)
        sdfg4 = auto_optimize(sdfg4, dace.DeviceType.CPU)
    func4 = utils.distributed_compile(sdfg4, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    u1 = rng.random((M,), dtype=np.float64)
    v1 = rng.random((N,), dtype=np.float64)
    u2 = rng.random((M,), dtype=np.float64)
    v2 = rng.random((N,), dtype=np.float64)
    w = rng.random((M,), dtype=np.float64)
    x = rng.random((N,), dtype=np.float64)
    y = rng.random((M,), dtype=np.float64)
    z = rng.random((N,), dtype=np.float64)
    alpha = 1.5
    beta = 1.3
    A2 = A + np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    tmp1 = A2.T @ y
    x2 = x + beta * tmp1 + z
    tmp2 = A2 @ x2
    w2 = w + alpha * tmp2

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lu1 = u1[i*ti:(i+1)*ti].copy()
    lv1 = v1[j*tj:(j+1)*tj].copy()
    lu2 = u2[i*ti:(i+1)*ti].copy()
    lv2 = v2[j*tj:(j+1)*tj].copy()

    func4(alpha=alpha, beta=beta, A=lA, u1=lu1, v1=lv1, u2=lu2, v2=lv2, w=w, x=x, y=y, z=z,
          LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
    refA = A2[i*ti:(i+1)*ti, j*tj:(j+1)*tj]
    assert(np.allclose(lA, refA))
    if rank == 0:
        assert(np.allclose(x, x2))
        assert(np.allclose(w, w2))
    
    @dace.program
    def gesummv(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], B: dace.float64[LMx, LNy],
                x: dace.float64[GN], y: dace.float64[GM]):
        tmp1 = dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(GM, 1))
        tmp2 = dace.distr.MatMult(B, x, (M, N), c_block_sizes=(GM, 1))
        y[:] = alpha * tmp1 + beta * tmp2
    
    sdfg5 = None
    if rank == 0:
        sdfg5 = gesummv.to_sdfg(simplify=True)
        sdfg5 = auto_optimize(sdfg5, dace.DeviceType.CPU)
    func5 = utils.distributed_compile(sdfg5, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    B = rng.random((M, N), dtype=np.float64)
    x = rng.random((N,), dtype=np.float64)
    alpha = 1.5
    beta = 1.3
    y = alpha * A @ x + beta * B @ x

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lB = B[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()

    val = np.ndarray((M,), dtype=np.float64)
    func5(alpha=alpha, beta=beta, A=lA, B=lB, x=x, y=val, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
    if rank == 0:
        assert(np.allclose(val, y))


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

# @dace.program
# def atax_shmem(A: dace.float64[M, N], x: dace.float64[N], y:dace.float64[N]):
#     y[:] = (A @ x) @ A

# @dace.program
# def atax_distr(A: dace.float64[lM, lN], x: dace.float64[lN], y:dace.float64[lN]):
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
#         mpi_sdfg = dace.SDFG.from_file(".dacecache/{n}/program.sdfg".format(
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