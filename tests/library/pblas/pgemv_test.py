# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the PBLAS GEMV library node. """

import dace
import numpy as np
import pytest

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils

# Symbols

# Process grid
P, Px, Py = (dace.symbol(s, positive=True) for s in ('P', 'Px', 'Py'))
# Global sizes
GM, GN = (dace.symbol(s, positive=True) for s in ('GM', 'GN'))
# Local sizes
LMx, LMy, LNx, LNy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy'))

rng = np.random.default_rng(42)


@pytest.mark.scalapack
def test_pgemv():

    from mpi4py import MPI

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    # DaCe programs
    @dace.program
    def pdgemv(A: dace.float64[LMx, LNy], x: dace.float64[GN]):
        return dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))

    @dace.program
    def pdgemv_T(A: dace.float64[LMx, LNy], x: dace.float64[GM]):
        return dace.distr.MatMult(x, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))

    @dace.program
    def atax(A: dace.float64[LMx, LNy], x: dace.float64[GN], y: dace.float64[GN]):
        tmp = dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
        y[:] = dace.distr.MatMult(tmp, A, (GM, GN), c_block_sizes=(GN, 1))

    @dace.program
    def bicg(A: dace.float64[LMx, LNy], p: dace.float64[GN], r: dace.float64[GM], o1: dace.float64[GN],
             o2: dace.float64[GM]):
        o1[:] = dace.distr.MatMult(r, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
        o2[:] = dace.distr.MatMult(A, p, (GM, GN), c_block_sizes=(GM, 1))

    @dace.program
    def gemver(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], u1: dace.float64[LMx],
               v1: dace.float64[LNy], u2: dace.float64[LMx], v2: dace.float64[LNy], w: dace.float64[GM],
               x: dace.float64[GN], y: dace.float64[GM], z: dace.float64[GN]):
        A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
        tmp1 = dace.distr.MatMult(y, A, (Px * LMx, Py * LNy), c_block_sizes=(GN, 1))
        x += beta * tmp1 + z
        tmp2 = dace.distr.MatMult(A, x, (GM, GN), c_block_sizes=(GM, 1))
        w += alpha * tmp2

    @dace.program
    def gesummv(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LNy], B: dace.float64[LMx, LNy],
                x: dace.float64[GN], y: dace.float64[GM]):
        tmp1 = dace.distr.MatMult(A, x, (Px * LMx, Py * LNy), c_block_sizes=(GM, 1))
        tmp2 = dace.distr.MatMult(B, x, (GM, GN), c_block_sizes=(GM, 1))
        y[:] = alpha * tmp1 + beta * tmp2

    def optimize(program):
        if rank == 0:
            sdfg = program.to_sdfg(simplify=True)
            return auto_optimize(sdfg, dace.DeviceType.CPU)
        else:
            return None

    def compile(sdfg, name):
        # A shape configuration of its own gets a name of its own: auto_optimize gives the
        # intermediate vectors Persistent lifetime, so their size is fixed by __dace_init_ and a
        # second one is a second program, never a second init on this one. The grid is NOT among
        # those symbols any more -- see test_pgemm_multiple_grids_one_compiled_object.
        if sdfg is not None:
            sdfg.name = name
        return utils.distributed_compile(sdfg, commworld)

    sdfgs = [optimize(prog) for prog in (pdgemv, pdgemv_T, atax, bicg, gemver, gesummv)]
    base_names = [sdfg.name for sdfg in sdfgs] if rank == 0 else [None] * len(sdfgs)

    # Every (NPx, NPy) with NPx * NPy == size, tall to wide. Both extents divide size, so a problem
    # size that is a multiple of size splits evenly.
    for NPx, NPy in [(size // npy, npy) for npy in range(1, size + 1) if size % npy == 0]:

        cart_comm = commworld.Create_cart((NPx, NPy))
        i, j = cart_comm.Get_coords(rank)

        Mmult = 37
        Nmult = 48
        M, N = size * Mmult, size * Nmult

        for shapes in range(2):  # The sizes are permuted at the end of each iteration.

            if rank == 0:
                print(f"Testing PBLAS GEMV on a [{NPx}, {NPy}] grid with sizes ({M}, {N}).", flush=True)

            func, func1, func2, func3, func4, func5 = [
                compile(sd, f'{base}_{NPx}x{NPy}_{shapes}') for sd, base in zip(sdfgs, base_names)
            ]

            A = rng.random((M, N), dtype=np.float64)
            x = rng.random((N, ), dtype=np.float64)
            y = A @ x

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            val = func(A=lA, x=x, GM=M, GN=N, LMx=ti, LNy=tj, Px=NPx, Py=NPy)
            if rank == 0:
                assert (np.allclose(val, y))

            commworld.Barrier()

            A = rng.random((M, N), dtype=np.float64)
            x = rng.random((M, ), dtype=np.float64)
            y = A.T @ x

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            val = func1(A=lA, x=x, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
            if rank == 0:
                assert (np.allclose(val, y))

            commworld.Barrier()

            A = rng.random((M, N), dtype=np.float64)
            x = rng.random((N, ), dtype=np.float64)
            y = A.T @ (A @ x)

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            val = np.ndarray((N, ), dtype=np.float64)
            func2(A=lA, x=x, y=val, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
            if rank == 0:
                assert (np.allclose(val, y))

            commworld.Barrier()

            A = rng.random((M, N), dtype=np.float64)
            p = rng.random((N, ), dtype=np.float64)
            r = rng.random((M, ), dtype=np.float64)
            o1 = A.T @ r
            o2 = A @ p

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            val1 = np.ndarray((N, ), dtype=np.float64)
            val2 = np.ndarray((M, ), dtype=np.float64)
            func3(A=lA, p=p, r=r, o1=val1, o2=val2, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
            if rank == 0:
                assert (np.allclose(val1, o1))
                assert (np.allclose(val2, o2))

            commworld.Barrier()

            A = rng.random((M, N), dtype=np.float64)
            u1 = rng.random((M, ), dtype=np.float64)
            v1 = rng.random((N, ), dtype=np.float64)
            u2 = rng.random((M, ), dtype=np.float64)
            v2 = rng.random((N, ), dtype=np.float64)
            w = rng.random((M, ), dtype=np.float64)
            x = rng.random((N, ), dtype=np.float64)
            y = rng.random((M, ), dtype=np.float64)
            z = rng.random((N, ), dtype=np.float64)
            alpha = 1.5
            beta = 1.3
            A2 = A + np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
            tmp1 = A2.T @ y
            x2 = x + beta * tmp1 + z
            tmp2 = A2 @ x2
            w2 = w + alpha * tmp2

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()
            lu1 = u1[i * ti:(i + 1) * ti].copy()
            lv1 = v1[j * tj:(j + 1) * tj].copy()
            lu2 = u2[i * ti:(i + 1) * ti].copy()
            lv2 = v2[j * tj:(j + 1) * tj].copy()

            func4(alpha=alpha,
                  beta=beta,
                  A=lA,
                  u1=lu1,
                  v1=lv1,
                  u2=lu2,
                  v2=lv2,
                  w=w,
                  x=x,
                  y=y,
                  z=z,
                  LMx=ti,
                  LNy=tj,
                  GM=M,
                  GN=N,
                  Px=NPx,
                  Py=NPy)
            refA = A2[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
            assert (np.allclose(lA, refA))
            if rank == 0:
                assert (np.allclose(x, x2))
                assert (np.allclose(w, w2))

            commworld.Barrier()

            A = rng.random((M, N), dtype=np.float64)
            B = rng.random((M, N), dtype=np.float64)
            x = rng.random((N, ), dtype=np.float64)
            alpha = 1.5
            beta = 1.3
            y = alpha * A @ x + beta * B @ x

            ti, tj = M // NPx, N // NPy
            lA = A[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()
            lB = B[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            val = np.ndarray((M, ), dtype=np.float64)
            func5(alpha=alpha, beta=beta, A=lA, B=lB, x=x, y=val, LMx=ti, LNy=tj, GM=M, GN=N, Px=NPx, Py=NPy)
            if rank == 0:
                assert (np.allclose(val, y))

            M, N = N, M


if __name__ == '__main__':

    test_pgemv()
