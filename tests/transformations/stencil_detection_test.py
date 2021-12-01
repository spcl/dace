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
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.LibraryNode):
            node.implementation = 'intel_fpga'
    sdfg.apply_fpga_transformations()
    # from dace.transformation.auto import auto_optimize as opt
    # opt.set_fast_implementations(sdfg, dace.dtypes.DeviceType.FPGA)
    sdfg.expand_library_nodes()
    sdfg.save('test.sdfg')
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

I, J, K = 100, 100, 100


@dace.program
def hdiff(in_field: dace.float64[I + 4, J + 4, K],
          out_field: dace.float64[I, J, K], coeff: dace.float64[I, J, K]):

    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(
        (res1 *
         (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res1,
    )

    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(
        (res2 *
         (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res2,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])


@pytest.mark.skip
def test_hdiff():

    rng = np.random.default_rng(42)
    in_field = rng.random((I+4, J+4, K))
    coeff = rng.random((I, J, K))
    out_field = np.zeros((I, J, K))
    ref = np.zeros((I, J, K))

    hdiff.f(in_field, coeff, ref)

    sdfg = hdiff.to_sdfg(strict=True)
    # from dace.transformation.auto import auto_optimize as autoopt
    # sdfg = autoopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    # autoopt.greedy_fuse(sdfg, True)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    sdfg.save('hdiff_before.sdfg')
    num = sdfg.apply_transformations_repeated(StencilDetection)
    sdfg.save('hdiff_after.sdfg')
    print(num)
    sdfg(in_field=in_field, coeff=coeff, out_field=out_field)


BET_M = 0.5
BET_P = 0.5


@dace.program
def vadv(utens_stage: dace.float64[I, J, K], u_stage: dace.float64[I, J, K],
         wcon: dace.float64[I + 1, J, K], u_pos: dace.float64[I, J, K],
         utens: dace.float64[I, J, K], dtr_stage: dace.float64):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs[:] = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term[:] = -as_ * (
            u_stage[:, :, k - 1] -
            u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_[:] = gav * BET_M
        acol[:] = gav * BET_P
        bcol[:] = dtr_stage - acol

        # update the d column
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    

@pytest.mark.skip
def test_vadv():

    # rng = np.random.default_rng(42)
    # in_field = rng.random((I+4, J+4, K))
    # coeff = rng.random((I, J, K))
    # out_field = np.zeros((I, J, K))
    # ref = np.zeros((I, J, K))

    # hdiff.f(in_field, coeff, ref)

    sdfg = vadv.to_sdfg(strict=True)
    from dace.transformation.auto import auto_optimize as autoopt
    sdfg = autoopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    # autoopt.greedy_fuse(sdfg, True)
    # sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    sdfg.save('vadv_before.sdfg')
    num = sdfg.apply_transformations_repeated(StencilDetection)
    sdfg.save('vadv_after.sdfg')
    print(num)
    # sdfg(in_field=in_field, coeff=coeff, out_field=out_field)


C_in, C_out, H, K, N, W = (dace.symbol(s, dace.int64)
                           for s in ('C_in', 'C_out', 'H', 'K', 'N', 'W'))


# Deep learning convolutional operator (stride = 1)
@dace.program
def conv2d(input: dace.float32[N, H, W, C_in], weights: dace.float32[K, K, C_in,
                                                                 C_out]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def test_conv2d():
    # ref = np.zeros((I, J, K))

    # hdiff.f(in_field, coeff, ref)

    sdfg = conv2d.to_sdfg(strict=True)
    from dace.transformation.auto import auto_optimize as autoopt
    sdfg = autoopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    # autoopt.greedy_fuse(sdfg, True)
    # sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    sdfg.save('conv2d_before.sdfg')
    num = sdfg.apply_transformations_repeated(StencilDetection)
    sdfg.save('conv2d_after.sdfg')
    print(num)
    # sdfg(in_field=in_field, coeff=coeff, out_field=out_field)


if __name__ == '__main__':
    # test_stencil1d()
    # test_jacobi1d()
    test_jacobi2d()
    # test_jacobi1d_with_scalar()
    # test_CFD_build_up_b()
    # test_CFD_pressure_poisson()
    # test_hdiff()
    # test_vadv()
    # test_conv2d()
