# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize


nx, ny, nit = (dace.symbol(s, dace.int64) for s in ('nx', 'ny', 'nit'))


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val-ref)
    return np.linalg.norm(val-ref) / np.linalg.norm(ref)


@dace.program
def build_up_b(rho: dace.float64, dt: dace.float64, dx: dace.float64, dy: dace.float64,
               u: dace.float64[ny, nx], v: dace.float64[ny, nx]):
    b = np.zeros_like(u)
    b[1:-1,
      1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                      ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                      ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 - 2 *
                          ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                           (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 - 2 *
                         ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                          (v[1:-1, 1] - v[1:-1, -1]) /
                          (2 * dx)) - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))

    return b


@dace.program
def pressure_poisson_periodic(p: dace.float64[ny, nx], dx: dace.float64,
                              dy: dace.float64, b: dace.float64[ny, nx]):
    pn = np.empty_like(p)

    for q in range(nit):
        pn[:] = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                         (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                       (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1,
          0] = (((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 +
                 (pn[2:, 0] + pn[0:-2, 0]) * dx**2) / (2 * (dx**2 + dy**2)) -
                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])

        # Wall boundary conditions, pressure
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0


@dace.program
def dace_channel_flow(nit: dace.int64, u: dace.float64[ny, nx], v: dace.float64[ny, nx],
                      dt: dace.float64, dx: dace.float64, dy: dace.float64,
                      p: dace.float64[ny, nx], rho: dace.float64, nu: dace.float64,
                      F: dace.float64):
    udiff = 1.0
    stepcount = 0

    while udiff > .001:
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        pressure_poisson_periodic(p, dx, dy, b, nit=nit)

        u[1:-1,
          1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
                   (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                   vn[1:-1, 1:-1] * dt / dy *
                   (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) *
                   (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu *
                   (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                   F * dt)

        v[1:-1,
          1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
                   (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                   vn[1:-1, 1:-1] * dt / dy *
                   (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) *
                   (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
                   (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2
        u[1:-1, -1] = (
            un[1:-1, -1] - un[1:-1, -1] * dt / dx *
            (un[1:-1, -1] - un[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
            (un[1:-1, -1] - un[0:-2, -1]) - dt / (2 * rho * dx) *
            (p[1:-1, 0] - p[1:-1, -2]) + nu *
            (dt / dx**2 *
             (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2]) + dt / dy**2 *
             (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1,
          0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (un[1:-1, 0] - un[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
                (un[1:-1, 0] - un[0:-2, 0]) - dt / (2 * rho * dx) *
                (p[1:-1, 1] - p[1:-1, -1]) + nu *
                (dt / dx**2 *
                 (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) + dt / dy**2 *
                 (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (
            vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
            (vn[1:-1, -1] - vn[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
            (vn[1:-1, -1] - vn[0:-2, -1]) - dt / (2 * rho * dy) *
            (p[2:, -1] - p[0:-2, -1]) + nu *
            (dt / dx**2 *
             (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) + dt / dy**2 *
             (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1,
          0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (vn[1:-1, 0] - vn[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
                (vn[1:-1, 0] - vn[0:-2, 0]) - dt / (2 * rho * dy) *
                (p[2:, 0] - p[0:-2, 0]) + nu *
                (dt / dx**2 *
                 (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) + dt / dy**2 *
                 (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1

    return stepcount


def numpy_channel_flow(nit, u, v, dt, dx, dy, p, rho, nu, F):

    def build_up_b(rho, dt, dx, dy, u, v):
        b = np.zeros_like(u)
        b[1:-1,
        1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                    (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                        ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                        (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                        ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

        # Periodic BC Pressure @ x = 2
        b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx) +
                                        (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                            ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 - 2 *
                            ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                            (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                            ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

        # Periodic BC Pressure @ x = 0
        b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                    (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                            ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 - 2 *
                            ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                            (v[1:-1, 1] - v[1:-1, -1]) /
                            (2 * dx)) - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))

        return b

    def pressure_poisson_periodic(nit, p, dx, dy, b):
        pn = np.empty_like(p)

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                            (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

            # Periodic BC Pressure @ x = 2
            p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 +
                            (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                        (2 * (dx**2 + dy**2)) * b[1:-1, -1])

            # Periodic BC Pressure @ x = 0
            p[1:-1,
            0] = (((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 +
                    (pn[2:, 0] + pn[0:-2, 0]) * dx**2) / (2 * (dx**2 + dy**2)) -
                    dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])

            # Wall boundary conditions, pressure
            p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
            p[0, :] = p[1, :]  # dp/dy = 0 at y = 0

    udiff = 1
    stepcount = 0

    while udiff > .001:
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        pressure_poisson_periodic(nit, p, dx, dy, b)

        u[1:-1,
          1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
                   (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                   vn[1:-1, 1:-1] * dt / dy *
                   (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) *
                   (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu *
                   (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                   F * dt)

        v[1:-1,
          1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
                   (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                   vn[1:-1, 1:-1] * dt / dy *
                   (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) *
                   (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
                   (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2
        u[1:-1, -1] = (
            un[1:-1, -1] - un[1:-1, -1] * dt / dx *
            (un[1:-1, -1] - un[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
            (un[1:-1, -1] - un[0:-2, -1]) - dt / (2 * rho * dx) *
            (p[1:-1, 0] - p[1:-1, -2]) + nu *
            (dt / dx**2 *
             (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2]) + dt / dy**2 *
             (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1,
          0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (un[1:-1, 0] - un[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
                (un[1:-1, 0] - un[0:-2, 0]) - dt / (2 * rho * dx) *
                (p[1:-1, 1] - p[1:-1, -1]) + nu *
                (dt / dx**2 *
                 (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) + dt / dy**2 *
                 (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (
            vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
            (vn[1:-1, -1] - vn[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
            (vn[1:-1, -1] - vn[0:-2, -1]) - dt / (2 * rho * dy) *
            (p[2:, -1] - p[0:-2, -1]) + nu *
            (dt / dx**2 *
             (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) + dt / dy**2 *
             (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1,
          0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (vn[1:-1, 0] - vn[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
                (vn[1:-1, 0] - vn[0:-2, 0]) - dt / (2 * rho * dy) *
                (p[2:, 0] - p[0:-2, 0]) + nu *
                (dt / dx**2 *
                 (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) + dt / dy**2 *
                 (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1

    return stepcount


def initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / ((nx - 1) * (ny - 1))
    return u, v, p, dx, dy, dt


def run_channel_flow(device_type: dace.dtypes.DeviceType):
    '''
    Runs channel-flow for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    ny, nx, nit, rho, nu, F = (61, 61, 5, 1.0, 0.1, 1.0)
    u, v, p, dx, dy, dt = initialize(ny, nx)
    dace_u, dace_v, dace_p = u.copy(), v.copy(), p.copy()

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = dace_channel_flow.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(nit=nit, u=dace_u, v=dace_v, dt=dt, dx=dx, dy=dy, p=dace_p, rho=rho, nu=nu, F=F, ny=ny, nx=nx)

    # Compute ground truth and Validate result
    numpy_channel_flow(nit, u, v, dt, dx, dy, p, rho, nu, F)
    assert (np.allclose(dace_u, u) or relerror(dace_u, u) < 1e-10)
    assert (np.allclose(dace_v, v) or relerror(dace_v, v) < 1e-10)
    assert (np.allclose(dace_p, p) or relerror(dace_p, p) < 1e-10)
    return sdfg


def test_cpu():
    run_channel_flow(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_channel_flow(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_channel_flow(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_channel_flow(dace.dtypes.DeviceType.GPU)
