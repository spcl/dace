# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import pytest

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


@dace.program
def jacobi1d_with_scalar(TMAX: dace.int32, one_third: dace.float32,
                         A: dace.float32[12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = one_third * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = one_third * (B[:-2] + B[1:-1] + B[2:])


def test_jacobi1d_with_scalar():
    rng = np.random.default_rng(42)
    A = rng.random(12, dtype=np.float32)
    B = rng.random(12, dtype=np.float32)
    one_third = np.float32(1/3)
    refA = A.copy()
    refB = B.copy()

    jacobi1d_with_scalar.f(100, one_third, refA, refB)

    sdfg = jacobi1d_with_scalar.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations_repeated(StencilDetection)
    assert (num == 2)
    sdfg(TMAX=100, one_third=one_third, A=A, B=B)

    assert (np.allclose(A, refA))
    assert (np.allclose(B, refB))



nx, ny = (dace.symbol(s) for s in ('nx', 'ny'))


@dace.program
def build_up_b(b: dace.float64[ny, nx], rho: dace.float64, dt: dace.float64,
               u: dace.float64[ny, nx], v: dace.float64[ny, nx],
               dx: dace.float64, dy: dace.float64):

    b[1:-1,
      1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                      ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                      ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))


def test_CFD_build_up_b():

    nx = 41
    ny = 41
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    rho = 1
    dt = .001

    rng = np.random.default_rng(42)
    u = rng.random((ny, nx))
    v = rng.random((ny, nx))

    b = np.zeros((ny, nx))
    ref = np.zeros((ny, nx))

    build_up_b.f(ref, rho, dt, u, v, dx, dy)

    sdfg = build_up_b.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations_repeated(StencilDetection)
    assert (num == 1)
    sdfg(b=b, rho=rho, dt=dt, u=u, v=v, dx=dx, dy=dy, nx=nx, ny=ny)

    assert (np.allclose(b, ref))


@dace.program
def pressure_poisson(nit: dace.int32,
                     p: dace.float64[ny, nx],
                     dx: dace.float64, dy: dace.float64,
                     b: dace.float64[ny, nx]):
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                         (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2


@pytest.mark.skip
def test_CFD_pressure_poisson():

    nx = 41
    ny = 41
    nit = 50
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    rng = np.random.default_rng(42)
    p = rng.random((ny, nx))
    b = rng.random((ny, nx))
    ref = p.copy()

    pressure_poisson.f(nit, ref, dx, dy, b)

    sdfg = pressure_poisson.to_sdfg(strict=True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    num = sdfg.apply_transformations_repeated(StencilDetection)
    assert (num == 1)
    sdfg(nit=nit, p=p, dx=dx, dy=dy, b=b, nx=nx, ny=ny)

    assert (np.allclose(p, ref))


if __name__ == '__main__':
    # test_stencil1d()
    # test_jacobi1d()
    # test_jacobi2d()
    test_jacobi1d_with_scalar()
    # test_CFD_build_up_b()
    # test_CFD_pressure_poisson()
