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
GM, GN, GK, GR, GS, GT = (dace.symbol(s, positive=True) for s in ('GM', 'GN', 'GK', 'GR', 'GS', 'GT'))
# Local sizes
LMx, LMy, LNx, LNy, LKx, LKy = (dace.symbol(s, positive=True) for s in ('LMx', 'LMy', 'LNx', 'LNy', 'LKx', 'LKy'))
LRx, LRy, LSx, LSy, LTx, LTy = (dace.symbol(s, positive=True) for s in ('LRx', 'LRy', 'LSx', 'LSy', 'LTx', 'LTy'))

grids = {
    1: [(1, 1)],
    2: [(2, 1), (1, 2)],
    3: [(3, 1), (1, 3)],
    4: [(4, 1), (2, 2), (1, 4)],
    5: [(5, 1), (1, 5)],
    6: [(6, 1), (3, 2), (2, 3), (1, 6)],
    7: [(7, 1), (1, 7)],
    8: [(8, 1), (4, 2), (2, 4), (1, 8)]
}

rng = np.random.default_rng(42)


# NOTE: The test passes with MKLMPICH, ReferenceMPICH, and ReferenceOpenMPI. It segfaults with MKLOpenMPI.
# @pytest.mark.scalapack
@pytest.mark.skip
def test_pgemm():

    from mpi4py import MPI

    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grids:
        raise NotImplementedError("Please run this test with 1-8 MPI processes.")

    # DaCe programs
    @dace.program
    def pdgemm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy]):
        return dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))

    @dace.program
    def gemm(alpha: dace.float64, beta: dace.float64, C: dace.float64[LMx, LNy], A: dace.float64[LMx, LKy],
             B: dace.float64[LKx, LNy]):
        C[:] = alpha * dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK)) + beta * C

    @dace.program
    def k2mm(alpha: dace.float64, beta: dace.float64, A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy],
             C: dace.float64[LNx, LRy], D: dace.float64[LMx, LRy]):
        tmp = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
        D[:] = alpha * dace.distr.MatMult(tmp, C, (GM, GR, GN)) + beta * D

    @dace.program
    def k3mm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy], C: dace.float64[LNx, LRy], D: dace.float64[LRx, LSy],
             E: dace.float64[LMx, LSy]):
        tmp1 = dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))
        tmp2 = dace.distr.MatMult(tmp1, C, (GM, GR, GN))
        E[:] = dace.distr.MatMult(tmp2, D, (GM, GS, GR))

    def optimize(program):
        if rank == 0:
            sdfg = program.to_sdfg(simplify=True)
            return auto_optimize(sdfg, dace.DeviceType.CPU)
        else:
            return None

    def compile(sdfg):
        return utils.distributed_compile(sdfg, commworld)

    sdfgs = []
    for prog in (pdgemm, gemm, k2mm, k3mm):
        sdfgs.append(optimize(prog))

    # Test for different grids possible with the given number of MPI processes.
    grid_dims = grids[size]
    for NPx, NPy in grid_dims:

        cart_comm = commworld.Create_cart((NPx, NPy))
        i, j = cart_comm.Get_coords(rank)

        Mmult = 39
        Nmult = 57
        Kmult = 43
        Rmult = 32
        Smult = 67
        M, N, K, R, S = size * Mmult, size * Nmult, size * Kmult, size * Rmult, size * Smult

        for _ in range(5):  # The sizes are permuted at the end of each iteration.

            if rank == 0:
                print(f"Testing PBLAS GEMM on a [{NPx}, {NPy}] grid with sizes ({M}, {N}, {K}, {R}, {S}).", flush=True)

            funcs = []
            for sd in sdfgs:
                funcs.append(compile(sd))
            func, func1, func2, func3 = funcs

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = A @ B

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()

            val = func(A=lA, B=lB, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, GK=K, Px=NPx, Py=NPy)
            ref = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
            assert (np.allclose(val, ref))

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((M, N), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            C2 = alpha * A @ B + beta * C

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj].copy()

            func1(alpha=alpha, beta=beta, C=lC, A=lA, B=lB, LMx=ti, LNy=tj, LKx=tki, LKy=tkj, GK=K, Px=NPx, Py=NPy)
            ref = C2[i * ti:(i + 1) * ti, j * tj:(j + 1) * tj]
            assert (np.allclose(lC, ref))

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((N, R), dtype=np.float64)
            D = rng.random((M, R), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            D2 = alpha * A @ B @ C + beta * D

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            tji, tr = N // NPx, R // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * tji:(i + 1) * tji, j * tr:(j + 1) * tr].copy()
            lD = D[i * ti:(i + 1) * ti, j * tr:(j + 1) * tr].copy()

            func2(alpha=alpha,
                  beta=beta,
                  A=lA,
                  B=lB,
                  C=lC,
                  D=lD,
                  LMx=ti,
                  LNy=tj,
                  LKx=tki,
                  LKy=tkj,
                  LNx=tji,
                  LRy=tr,
                  GM=M,
                  GN=N,
                  GK=K,
                  GR=R,
                  Px=NPx,
                  Py=NPy)
            ref = D2[i * ti:(i + 1) * ti, j * tr:(j + 1) * tr]
            assert (np.allclose(lD, ref))

            commworld.Barrier()

            A = rng.random((M, K), dtype=np.float64)
            B = rng.random((K, N), dtype=np.float64)
            C = rng.random((N, R), dtype=np.float64)
            D = rng.random((R, S), dtype=np.float64)
            alpha = 1.5
            beta = 1.2
            E = A @ B @ C @ D

            ti, tj, tki, tkj = M // NPx, N // NPy, K // NPx, K // NPy
            tji, tri, trj, ts = N // NPx, R // NPx, R // NPy, S // NPy
            lA = A[i * ti:(i + 1) * ti, j * tkj:(j + 1) * tkj].copy()
            lB = B[i * tki:(i + 1) * tki, j * tj:(j + 1) * tj].copy()
            lC = C[i * tji:(i + 1) * tji, j * trj:(j + 1) * trj].copy()
            lD = D[i * tri:(i + 1) * tri, j * ts:(j + 1) * ts].copy()

            val = np.ndarray((ti, ts), dtype=np.float64)
            func3(A=lA,
                  B=lB,
                  C=lC,
                  D=lD,
                  E=val,
                  LMx=ti,
                  LNy=tj,
                  LKx=tki,
                  LKy=tkj,
                  LNx=tji,
                  LRx=tri,
                  LRy=trj,
                  LSy=ts,
                  GM=M,
                  GN=N,
                  GK=K,
                  GR=R,
                  GS=S,
                  Px=NPx,
                  Py=NPy)
            ref = E[i * ti:(i + 1) * ti, j * ts:(j + 1) * ts]
            assert (np.allclose(val, ref))

            M, N, K, R, S = N, K, R, S, M


if __name__ == '__main__':

    test_pgemm()
