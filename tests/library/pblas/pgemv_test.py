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
# noff = dace.symbol('noff', dtype=dace.int32, integer=True, nonnegative=True)
# soff = dace.symbol('soff', dtype=dace.int32, integer=True, nonnegative=True)
# woff = dace.symbol('woff', dtype=dace.int32, integer=True, nonnegative=True)
# eoff = dace.symbol('eoff', dtype=dace.int32, integer=True, nonnegative=True)
# nn = dace.symbol('nn', dtype=dace.int32, integer=True)
# ns = dace.symbol('ns', dtype=dace.int32, integer=True)
# nw = dace.symbol('nw', dtype=dace.int32, integer=True)
# ne = dace.symbol('ne', dtype=dace.int32, integer=True)
# MPI_Request = dace.opaque("MPI_Request")


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
    8: (4, 2),
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16)
}


rng = np.random.default_rng(42)


def test_pgemm():

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    # NPx, NPy = min(grid[size]), max(grid[size])
    NPx, NPy = grid[size]
    cart_comm = commworld.Create_cart((NPx, NPy))
    i, j = cart_comm.Get_coords(rank)

    Mmult = 512
    Nmult = 1024
    M, N = size * Mmult, size * Nmult

    # @dace.program
    # def pdgemv(A: dace.float64[LMx, LNy], x: dace.float64[LNy]):
    #     return dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(LMx, 1))
    @dace.program
    def pdgemv(A: dace.float64[LMx, LNy], x: dace.float64[GN]):
        return dace.distr.MatMult(A, x, (Px*LMx, Py*LNy), c_block_sizes=(GM, 1))
    
    sdfg = None
    if rank == 0:
        sdfg = pdgemv.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    func = utils.distributed_compile(sdfg, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    # A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    x = rng.random((N,), dtype=np.float64)
    # x = np.ones((N,), dtype=np.float64)
    y = A @ x

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lx = x[j*tj:(j+1)*tj].copy()

    # val = func(A=lA, x=lx, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    val = func(A=lA, x=x, GM=M, GN=N, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    # print(rank, val, flush=True)
    # if rank < NPx:
    #     ref = y[rank*ti:(rank+1)*ti]
    #     print(rank, val, ref, flush=True)
    #     assert(np.allclose(val, ref))
    if rank == 0:
        assert(np.allclose(val, y))

    commworld.Barrier()

    # @dace.program
    # def pdgemv_T(A: dace.float64[LMx, LNy], x: dace.float64[LMx]):
    #     return dace.distr.MatMult(x, A, (Px*LMx, Py*LNy), c_block_sizes=(LNy, 1))
    @dace.program
    def pdgemv_T(A: dace.float64[LMx, LNy], x: dace.float64[GM]):
        return dace.distr.MatMult(x, A, (Px*LMx, Py*LNy), c_block_sizes=(GN, 1))
    
    sdfg1 = None
    if rank == 0:
        sdfg1 = pdgemv_T.to_sdfg(simplify=True)
        sdfg1 = auto_optimize(sdfg1, dace.DeviceType.CPU)
    func1 = utils.distributed_compile(sdfg1, commworld)
    
    A = rng.random((M, N), dtype=np.float64)
    # A = np.arange(M*N, dtype=np.float64).reshape(M, N)
    # A = np.ones((M, N), dtype=np.float64)
    x = rng.random((M,), dtype=np.float64)
    # x = np.ones((M,), dtype=np.float64)
    # x = np.arange(M, dtype=np.float64).copy()
    y = A.T @ x

    ti, tj = M // NPx, N // NPy
    lA = A[i*ti:(i+1)*ti, j*tj:(j+1)*tj].copy()
    lx = x[i*ti:(i+1)*ti].copy()

    # val = func1(A=lA, x=lx, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
    val = func1(A=lA, x=x, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
    # if rank < NPx:
    #     ref = y[rank*tj:(rank+1)*tj]
    #     print(rank, val, ref, flush=True)
    #     # assert(np.allclose(val, ref))
    if rank == 0:
        assert(np.allclose(val, y))

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

    test_pgemm()
