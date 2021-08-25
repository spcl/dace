# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

from dace.transformation.dataflow import MapFusion
from dace.transformation.dataflow import StencilDetection
from dace.transformation.dataflow import SimpleTaskletFusion


@dace.program
def stencil1d(A: dace.float32[12], B: dace.float32[12]):
    B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])


def test_stencil1d():

    A = np.arange(12, dtype=np.float32)
    ref = np.zeros((12, ), dtype=np.float32)
    stencil1d.f(A, ref)

    sdfg = stencil1d.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations(StencilDetection)
    assert (num == 1)
    B1 = np.zeros((12, ), dtype=np.float32)
    sdfg(A=A, B=B1)

    sdfg.apply_transformations(StencilDetection)
    B2 = np.zeros((12, ), dtype=np.float32)
    sdfg(A=A, B=B2)

    assert (np.allclose(B2, B1))
    assert (np.allclose(B2, ref))


@dace.program
def jacobi1d(TMAX: dace.int32, A: dace.float32[12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.3333 * (B[:-2] + B[1:-1] + B[2:])


def test_jacobi1d():
    rng = np.random.default_rng(42)
    A = rng.random(12, dtype=np.float32)
    B = rng.random(12, dtype=np.float32)
    refA = A.copy()
    refB = B.copy()

    jacobi1d.f(100, refA, refB)

    sdfg = jacobi1d.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations_repeated(StencilDetection)
    assert (num == 2)
    sdfg(TMAX=100, A=A, B=B)

    assert (np.allclose(A, refA))
    assert (np.allclose(B, refB))


@dace.program
def jacobi2d(TMAX: dace.int32, A: dace.float32[12, 12], B: dace.float32[12,
                                                                        12]):
    for _ in range(TMAX):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


def test_jacobi2d():
    rng = np.random.default_rng(42)
    A = rng.random((12, 12), dtype=np.float32)
    B = rng.random((12, 12), dtype=np.float32)
    refA = A.copy()
    refB = B.copy()

    jacobi2d.f(100, refA, refB)

    sdfg = jacobi2d.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations_repeated(StencilDetection)
    assert (num == 2)
    sdfg(TMAX=100, A=A, B=B)

    assert (np.allclose(A, refA))
    assert (np.allclose(B, refB))


if __name__ == '__main__':
    test_stencil1d()
    test_jacobi1d()
    test_jacobi2d()
